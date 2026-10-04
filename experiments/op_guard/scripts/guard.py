#!/usr/bin/env python
"""Common guard set: evaluate one openpilot candidate (serving ONNX) on every guard line and print one pass / fail table.

    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate it_dw3-s0                 # subset mode (default)
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate shipped --mode full
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate X.onnx --lines navtest,wod --full navtest
    .venv/bin/python experiments/op_guard/scripts/guard.py --candidate it_dw3-s0 --report-only   # only re-render the table

Paired lines compare with the `shipped` candidate of the same mode; if its lines are missing, shipped is run first.
Leases: one row for the open-loop chain (1 card) and one for the closed-loop chain (the other free cards), through
`jevdrive.cl.lease`; both are released at the end. Lines are `line_<id>.py` scripts that follow guardlib's contract.
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

OPEN_LOOP = ["navtest", "navhard", "wod", "drift", "negatives"]
CLOSED_LOOP = ["hugsim", "b2d_turns", "b2d_ds"]
CL_LANE_ENV = "OP_GUARD_LANE"
WANT: list = []


def run(cmd, log: Path, env=None, cwd=G.REPO) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}  ({time.strftime('%F %T')})\n")
        f.flush()
        return subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}, cwd=cwd)


def sh_lease(lane: str, n: int, note: str):
    """Lease up to n free cards (fewer if the box is shared); returns (lane, cards, cpus string) or None."""
    from jevdrive.cl import lease as L
    for k in range(n, 0, -1):
        r = subprocess.run([sys.executable, "-m", "jevdrive.cl", "lease", lane, "--gpus", str(k), "--status", note], capture_output=True, text=True, cwd=G.REPO)
        if r.returncode == 0 and "granted" in r.stdout:
            ls = L.get(lane)
            cards = sorted(ls.cards)
            cpus = ",".join(ls.cards[c]["cpus"] for c in cards if ls.cards[c]["cpus"])
            return lane, cards, cpus
    return None


def release(lane: str, note: str):
    subprocess.run([sys.executable, "-m", "jevdrive.cl", "release", lane, note], capture_output=True, text=True, cwd=G.REPO)


def line_cmd(line: str, cand: str, mode: str, gpu: int, cpus: str, force: bool):
    script = G.GUARD / "scripts" / f"line_{line}.py"
    if not script.is_file():
        return None
    c = [sys.executable, script, "--candidate", cand, "--mode", mode, "--gpu", gpu]
    if cpus:
        c += ["--cpus", cpus]
    if line in CLOSED_LOOP:   # the first closed-loop line packs every wanted closed-loop line into one lane; later calls only collect
        c += ["--cl-lines", ",".join(x for x in CLOSED_LOOP if x in WANT)]
    return c + (["--force"] if force else [])


def run_chain(lines, cand, mode_of, out: Path, gpu: int, cpus: str, lane: str | None, force: bool, say):
    for line in lines:
        mode = mode_of(line)
        if G.done(out, line) and not force:
            say(f"{line}: cached ({out}/lines/{line}.json)")
            continue
        c = line_cmd(line, cand, mode, gpu, cpus, force)
        if c is None:
            G.write_line(out, line, cand, mode, [G.row(line, "line script missing", None, ok=None, note=f"scripts/line_{line}.py not wired")], time.time(), status="stub")
            say(f"{line}: no script, stub")
            continue
        t0 = time.time()
        say(f"{line}: start ({mode}, card {gpu})")
        rc = run(c, out / "logs" / f"{line}.log", env={CL_LANE_ENV: lane} if lane else {})
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
    ap.add_argument("--cards", type=int, default=3, help="most cards to lease (fewer when the box is shared)")
    ap.add_argument("--gpu", type=int, default=None, help="skip leasing: run everything on this already-held card (--cpus: its cores)")
    ap.add_argument("--cpus", default="")
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
    if name != G.SHIPPED:
        shipped_out = G.run_dir(G.SHIPPED, a.mode)
        missing = [ln for ln in want if not G.done(shipped_out, ln)]
        if missing:
            print(f"shipped reference lacks {missing}: running shipped first")
            cmd = [sys.executable, __file__, "--candidate", G.SHIPPED, "--mode", a.mode, "--lines", ",".join(missing), "--cards", str(a.cards)]
            if full:
                cmd += ["--full", a.full]
            if a.gpu is not None:
                cmd += ["--gpu", str(a.gpu), "--cpus", a.cpus]
            if subprocess.call(cmd) != 0:
                print("shipped reference run reported a non-pass; continuing")
    t_start = time.time()
    log = out / "logs" / "guard.log"
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
    leases = []
    try:
        if a.gpu is not None:
            chains = [(ol + cl, a.gpu, a.cpus, None)]
        else:
            tag = f"op-guard-{name}-{a.mode}"[:40]
            need_ol, need_cl = bool(ol), bool(cl)
            lo = sh_lease(tag + "-ol", 1, f"guard open-loop {time.strftime('%F %H:%M')}") if need_ol else None
            if need_ol and lo is None:
                sys.exit("no free card for the open-loop chain (python -m jevdrive.cl probe)")
            leases.append(lo)
            lc = sh_lease(tag + "-cl", max(a.cards - 1, 1), f"guard closed-loop {time.strftime('%F %H:%M')}") if need_cl else None
            if need_cl and lc is None:
                say("no free card for the closed-loop chain: running it after the open-loop chain on the same card")
                chains = [(ol + cl, lo[1][0], lo[2], lo[0])]
            else:
                leases.append(lc)
                chains = ([(ol, lo[1][0], lo[2], lo[0])] if lo else []) + ([(cl, lc[1][0], lc[2], lc[0])] if lc else [])
            for x in leases:
                if x:
                    say(f"lease {x[0]}: cards {x[1]} cpus {x[2]}")
        ths = [threading.Thread(target=run_chain, args=(ls, name, mode_of, out, g, c, lane, a.force, say)) for ls, g, c, lane in chains]
        [t.start() for t in ths]
        [t.join() for t in ths]
    finally:
        for x in leases:
            if x:
                release(x[0], f"done {time.strftime('%F %T')}")
    res = render(name, a.mode, out)
    (out / ("DONE" if res["verdict"] != "NO EVALUABLE LINE" else "ERROR")).write_text(json.dumps(dict(verdict=res["verdict"], t=res["t"])) + "\n")
    say(f"finished in {(time.time() - t_start) / 60:.1f} min: {res['verdict']}")
    print((out / "guard.md").read_text())
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
