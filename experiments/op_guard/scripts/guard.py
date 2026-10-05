#!/usr/bin/env python
"""Common guard set: evaluate one openpilot candidate (serving ONNX) on every guard line and print one pass / fail table.

    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate it_dw3-s0                 # subset mode (default)
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate shipped --mode full
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate X.onnx --lines navtest,wod --full navtest
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate it_dw3-s0 --report-only   # only re-render the table

Paired lines compare with the `shipped` candidate of the same mode; if its lines are missing, shipped is run first.
GPU pool (jevdrive.cl.pool): the open-loop chain is one pool job (`guard.py --chain-only --gpu {gpu}`, --ol-vram GB, --ol-cpu cores,
log $DATA_DIR/runs/op_guard/<candidate>/<mode>/logs/ol-pool/); the closed-loop lines submit their units as pool jobs themselves
(cllib.run_lines); guard.py blocks on both (pool.wait). --gpu N skips the pool for the open-loop chain (an already-held card).
--dry-run prints the open-loop pool job and the closed-loop unit specs, runs nothing. Lines are `line_<id>.py` scripts that follow
guardlib's contract.
Output: $DATA_DIR/runs/op_guard/<candidate>/<mode>/{lines/*.json, guard.json, guard.md, logs/}, copied to
experiments/op_guard/results/<candidate>/<mode>/ (commit that copy).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guardlib as G  # noqa: E402

sys.path.insert(0, str(G.REPO))

OPEN_LOOP = ["navtest", "navhard", "wod", "drift", "negatives"]
CLOSED_LOOP = ["hugsim", "b2d_turns", "b2d_ds"]
WANT: list = []


def run(cmd, log: Path, env=None, cwd=G.REPO) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}  ({time.strftime('%F %T')})\n")
        f.flush()
        return subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=cwd)


def line_cmd(line: str, cand: str, mode: str, gpu, cpus: str, force: bool, dry_run: bool = False):
    script = G.GUARD / "scripts" / f"line_{line}.py"
    if not script.is_file():
        return None
    c = [sys.executable, script, "--candidate", cand, "--mode", mode, "--gpu", gpu]
    if dry_run:
        c += ["--dry-run"]
    if cpus:
        c += ["--cpus", cpus]
    if line in CLOSED_LOOP:   # the first closed-loop line submits every wanted closed-loop line's units; later calls only collect
        c += ["--cl-lines", ",".join(x for x in CLOSED_LOOP if x in WANT)]
    return c + (["--force"] if force else [])


def run_chain(lines, cand, mode_of, out: Path, gpu, cpus: str, force: bool, say, dry_run: bool = False):
    for line in lines:
        mode = mode_of(line)
        if G.done(out, line) and not force:
            say(f"{line}: cached ({out}/lines/{line}.json)")
            continue
        if dry_run and line not in CLOSED_LOOP:
            say(f"{line}: would run on card {gpu}")
            continue
        c = line_cmd(line, cand, mode, gpu, cpus, force, dry_run)
        if c is None:
            G.write_line(out, line, cand, mode, [G.row(line, "line script missing", None, ok=None, note=f"scripts/line_{line}.py not wired")], time.time(), status="stub")
            say(f"{line}: no script, stub")
            continue
        t0 = time.time()
        say(f"{line}: start ({mode}, card {gpu})")
        if dry_run:
            print(subprocess.run(list(map(str, c)), capture_output=True, text=True, cwd=G.REPO).stdout)
            continue
        rc = run(c, out / "logs" / f"{line}.log")
        if rc != 0 or not G.done(out, line):
            G.write_line(out, line, cand, mode, [G.row(line, "line failed", None, ok=None, note=f"rc {rc}; see logs/{line}.log")], t0, status="error")
        say(f"{line}: rc {rc} in {(time.time() - t0) / 60:.1f} min")


def fmt(v, d=3):
    if v is None:
        return "-"
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{d}f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def render(cand: str, mode: str, out: Path) -> dict:
    """Merge the line files into guard.json / guard.md (one row per readout, pass / FAIL / n/a per row)."""
    rows, lines = [], {}
    for line in G.LINES:
        d = G.load_line(out, line)
        if d is None:
            continue
        lines[line] = dict(status=d["status"], mode=d["mode"], runtime_s=d["runtime_s"])
        for r in d["rows"]:
            rows.append(dict(line=line, mode=d["mode"], status=d["status"], **r))
    ev = [r for r in rows if r["pass"] is not None]
    fail = [r for r in rows if r["pass"] is False]
    na = [r for r in rows if r["pass"] is None]
    verdict = "FAIL" if fail else ("PASS" if ev else "NO EVALUABLE LINE")
    res = dict(candidate=cand, mode=mode, verdict=verdict, n_pass=len(ev) - len(fail), n_fail=len(fail), n_na=len(na), lines=lines, rows=rows,
               t=time.strftime("%F %T"))
    (out / "guard.json").write_text(json.dumps(res, indent=1, default=float))
    md = [f"# Guard set: {cand} ({mode} mode)", "", f"Verdict: **{verdict}** ({len(ev) - len(fail)} pass, {len(fail)} fail, {len(na)} not evaluable). {res['t']}", "",
          "| line | readout | candidate | shipped | delta | rule | result | note |", "|---|---|---:|---:|---:|---|:--|---|"]
    for r in rows:
        res_s = "n/a" if r["pass"] is None else ("pass" if r["pass"] else "**FAIL**")
        ci = f" [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]" if r.get("ci") else ""
        md.append(f"| {r['line']} ({r['mode']}) | {r['metric']} | {fmt(r['value'])} | {fmt(r['ref'])} | {fmt(r['delta'])}{ci} | {r['rule']} | {res_s} | {r['note']} |")
    md += ["", "Runtime per line (min): " + ", ".join(f"{k} {v['runtime_s'] / 60:.1f}" for k, v in lines.items()) + f"; sum {sum(v['runtime_s'] for v in lines.values()) / 60:.1f}", ""]
    (out / "guard.md").write_text("\n".join(md))
    dst = G.GUARD / "results" / cand / mode
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("guard.md", "guard.json"):
        shutil.copy(out / f, dst / f)
    shutil.copytree(out / "lines", dst / "lines", dirs_exist_ok=True)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="registry name (candidates.json) or a path to a serving .onnx")
    ap.add_argument("--mode", choices=("subset", "full"), default="subset")
    ap.add_argument("--full", default="", help="comma list of lines to run in full mode while --mode subset (e.g. navtest)")
    ap.add_argument("--lines", default="", help="comma list, default all: " + ",".join(G.LINES))
    ap.add_argument("--gpu", type=int, default=None, help="run the open-loop chain here on this already-held card (--cpus: its cores), not in the pool")
    ap.add_argument("--cpus", default="")
    ap.add_argument("--ol-vram", type=float, default=24.0, help="VRAM GB the open-loop pool job declares")
    ap.add_argument("--ol-cpu", type=int, default=16, help="cores the open-loop pool job pins")
    ap.add_argument("--chain-only", action="store_true", help="internal: run only the given lines on --gpu (the open-loop pool job), no render")
    ap.add_argument("--dry-run", action="store_true", help="print the pool specs, run nothing")
    ap.add_argument("--force", action="store_true", help="recompute every line (shipped's lines are reused unless --force-ref)")
    ap.add_argument("--force-ref", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    cand = G.resolve(a.candidate)
    name = cand["name"]
    want = [x for x in (a.lines.split(",") if a.lines else list(G.LINES)) if x]
    bad = [x for x in want if x not in G.LINES]
    if bad:
        sys.exit(f"unknown lines {bad}; known {list(G.LINES)}")
    WANT[:] = want
    full = set(a.full.split(",")) if a.full else set()
    mode_of = lambda line: "full" if (a.mode == "full" or line in full) else "subset"  # noqa: E731
    out = G.run_dir(name, a.mode)
    if a.report_only:
        res = render(name, a.mode, out)
        print((out / "guard.md").read_text())
        return 0 if res["verdict"] == "PASS" else 1
    # a line that runs in full mode lives in the same lines/ dir (its json carries mode); shipped is needed for the paired lines
    if a.chain_only:
        WANT[:] = []
        run_chain(want, name, mode_of, out, a.gpu, a.cpus, a.force, lambda m: print(f"{time.strftime('%H:%M:%S')} [{name}/{a.mode}] {m}", flush=True))
        return 0 if all(G.done(out, x) for x in want) else 1
    if name != G.SHIPPED:
        shipped_out = G.run_dir(G.SHIPPED, a.mode)
        missing = [ln for ln in want if not G.done(shipped_out, ln)]
        if missing:
            print(f"shipped reference lacks {missing}: running shipped first")
            cmd = [sys.executable, __file__, "--candidate", G.SHIPPED, "--mode", a.mode, "--lines", ",".join(missing),
                   "--ol-vram", str(a.ol_vram), "--ol-cpu", str(a.ol_cpu)] + (["--dry-run"] if a.dry_run else [])
            if full:
                cmd += ["--full", a.full]
            if a.gpu is not None:
                cmd += ["--gpu", str(a.gpu), "--cpus", a.cpus]
            if subprocess.call(cmd) != 0:
                print("shipped reference run reported a non-pass; continuing")
    t_start = time.time()
    log = out / "logs" / "guard.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def say(msg):
        with lock:
            s = f"{time.strftime('%H:%M:%S')} [{name}/{a.mode}] {msg}"
            print(s, flush=True)
            with log.open("a") as f:
                f.write(s + "\n")
            (out / "STATUS").write_text(s + "\n")

    ol = [x for x in OPEN_LOOP if x in want]
    cl = [x for x in CLOSED_LOOP if x in want]
    ol_job = None
    if ol and a.gpu is None:                  # the open-loop chain as one pool job on whichever card has room
        from jevdrive.cl import pool as P
        todo = [x for x in ol if a.force or not G.done(out, x)]
        if todo:
            cmd = [sys.executable, __file__, "--candidate", a.candidate, "--mode", a.mode, "--lines", ",".join(todo), "--chain-only",
                   "--gpu", "{gpu}", "--cpus", "{cpus}"] + (["--full", a.full] if full else []) + (["--force"] if a.force else [])
            kw = dict(owner="op_guard " + name, vram_gb=a.ol_vram, cpu=a.ol_cpu, log_dir=str(out / "logs" / "ol-pool"), cwd=str(G.REPO))
            if a.dry_run:
                say(f"would submit open-loop chain {json.dumps(kw)}: {' '.join(map(str, cmd))}")
            else:
                ol_job = P.submit(list(map(str, cmd)), name=f"guard-ol-{name}", **kw)
                say(f"open-loop chain {todo}: pool job {ol_job} (log {out}/logs/ol-pool/log.txt)")
        ol = []
    gpu = a.gpu if a.gpu is not None else 0      # closed-loop lines ignore --gpu (their units go to the pool)
    ths = [threading.Thread(target=run_chain, args=(ls, name, mode_of, out, gpu, a.cpus, a.force, say, a.dry_run)) for ls in (ol, cl) if ls]
    [t.start() for t in ths]
    [t.join() for t in ths]
    if ol_job:
        st = P.wait([ol_job], 60)[ol_job]
        say(f"open-loop chain pool job {ol_job}: {st}")
    if a.dry_run:
        return 0
    res = render(name, a.mode, out)
    (out / ("DONE" if res["verdict"] != "NO EVALUABLE LINE" else "ERROR")).write_text(json.dumps(dict(verdict=res["verdict"], t=res["t"])) + "\n")
    say(f"finished in {(time.time() - t_start) / 60:.1f} min: {res['verdict']}")
    print((out / "guard.md").read_text())
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
