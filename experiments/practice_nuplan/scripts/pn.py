#!/usr/bin/env python3
"""PRACTICE-NUPLAN (plans/2026-10-19-pn-prereg.md): new start states on the 28 train logs = arena train-split windows away from the 1 031
public start points, rolled by JTN-L18-s0 with frame dumps, as S rows merged into on1_JT; a 1 000-step JTN pilot against its control.
Box, repo root.

  select   AlpaSim venv ($P; sqlite of the raw nuPlan test logs). The arena set's train-split windows (runs/alpasim_proxy/arena/set/set.csv.gz,
           3 s grid, 5.5 s windows, 10 decisions) whose start frame w0 is >= MIN_OFF_S from the start frame (token - 1.5 s) of every public
           train scene of the same log segment; asserts no window comes from a held log (c0b_report.log_of); registers
           alpasim_nuplan/arena-train (all 5 495 train windows) and alpasim_nuplan/practice-starts (the kept ones), check_disjoint with
           onpol-held (ids) and log-disjoint; writes $W/select/{windows.csv, tok_times.json, meta.json, lists/c<j>.txt, lists/smoke.txt}:
           read chunks of <= CHUNK windows inside one shard cache (log / time order), and a 10-window smoke list (shard 0, strata mixed).
  stage2   the staged-launch checklist on finished roll chunks (prereg 3.2) -> exit 0 / 2, json on stdout
  gate     the pilot gate on held 460 @18 (prereg 4): zeros(pilot) <= zeros(control) + 2 -> results/pilot.{json,md}; exit 0 either way
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
DATA = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
W = DATA / "runs/practice_nuplan"
ARENA = DATA / "runs/alpasim_proxy/arena/set"
RES = REPO / "experiments/practice_nuplan/results"
HZ, BEFORE, MIN_OFF_S, CHUNK = 20, 30, 1.5, 110
ZERO_TYPES = ("collision_at_fault", "left_corridor_laterally", "offroad")   # the scorer's zero types in results-summary failure_reason


def log_of(s):                                       # c0b_report.log_of
    return "_".join(s.split("_")[:2])


def cmd_select(a):
    sys.path.insert(0, str(REPO / "experiments/alpasim_proxy/scripts"))
    import wait_set as WS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("practice_nuplan", "select", config=vars(a)) as run:
        held, train = splits.load("alpasim_nuplan/onpol-held"), splits.load("alpasim_nuplan/onpol-train")
        run.use_split(held), run.use_split(train)
        R = [r for r in csv.DictReader(gzip.open(ARENA / "set.csv.gz", "rt")) if r["split"] == "train"]
        hl = {log_of(s) for s in held.members}
        assert not any(log_of(r["id"]) in hl for r in R), "an arena train window in a held log"
        pub = {}
        for s in train.members:
            pub.setdefault(s.rsplit("-", 1)[0], []).append(s.rsplit("-", 1)[1])
        out, tokt = [], {}
        for seg in sorted({r["log"] for r in R}):
            L = WS.Log(seg)
            starts = np.array([L.idx[t] - BEFORE for t in pub.get(seg, []) if t in L.idx], float)
            for r in (x for x in R if x["log"] == seg):
                w0 = int(r["start_idx"])
                assert L.tok[w0 + BEFORE].hex() == r["token"], r["id"]
                d = float(np.abs(starts - w0).min() / HZ) if len(starts) else float("inf")
                tokt[r["id"]] = int(L.ts[w0 + BEFORE])
                out.append(dict(id=r["id"], log=seg, shard=int(r["shard"]), start_idx=w0, stratum=r["stratum"], d_pub_s=d, keep=d >= MIN_OFF_S))
        K = sorted([o for o in out if o["keep"]], key=lambda o: (o["shard"], o["log"], o["start_idx"]))
        at = splits.define("alpasim_nuplan", "arena-train", [o["id"] for o in out], unit="scene",
                           origin="experiments/practice_nuplan/scripts/pn.py select: every train-split window of the arena set "
                                  "(runs/alpasim_proxy/arena/set/set.csv.gz, 3 s grid, 5.5 s windows; experiments/alpasim_proxy arena.py pick)",
                           used_by=["experiments/practice_nuplan"], status="frozen",
                           notes="windows of the 28 onpol-train logs only (asserted log-disjoint from onpol-held); ids <segment>-<20 Hz token of w0 + 30>")
        ps = splits.define("alpasim_nuplan", "practice-starts", [o["id"] for o in K], unit="scene",
                           origin=f"experiments/practice_nuplan/scripts/pn.py select: arena-train windows whose start frame is >= {MIN_OFF_S} s "
                                  "from every public onpol-train scene start (token - 1.5 s) in the same log segment",
                           used_by=["experiments/practice_nuplan"], status="frozen",
                           notes="new start states for on-policy rows (prereg plans/2026-10-19-pn-prereg.md); never a held log")
        splits.check_disjoint(ps, held), splits.check_disjoint(at, held)
        assert not ({log_of(x) for x in ps.members} & hl)
        run.use_split(at), run.use_split(ps)
        S = W / "select"
        (S / "lists").mkdir(parents=True, exist_ok=True)
        chunks, j = [], 0
        for k in sorted({o["shard"] for o in K}):
            g = [o for o in K if o["shard"] == k]
            m = -(-len(g) // CHUNK)
            for q in range(m):
                part = g[q * len(g) // m:(q + 1) * len(g) // m]
                (S / "lists" / f"c{j}.txt").write_text("".join(o["id"] + "\n" for o in part))
                for o in part:
                    o["chunk"] = j
                chunks.append(dict(chunk=j, shard=k, windows=len(part)))
                j += 1
        rng = np.random.default_rng(0)
        g0 = [o for o in K if o["shard"] == 0]
        sm = []
        for st in ("static", "launch", "braking", "turn", "cruise", "other"):
            c = [o for o in g0 if o["stratum"] == st]
            sm += [c[i]["id"] for i in rng.choice(len(c), min(2 if st != "static" else 1, len(c)), replace=False)] if c else []
        sm = sm[:10]
        (S / "lists/smoke.txt").write_text("".join(x + "\n" for x in sm))
        with open(S / "windows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ["id", "log", "shard", "chunk", "start_idx", "stratum", "d_pub_s", "keep"])
            w.writeheader()
            w.writerows([o | {"chunk": o.get("chunk", -1)} for o in out])
        (S / "tok_times.json").write_text(json.dumps(tokt))
        from collections import Counter
        meta = dict(train_windows=len(out), kept=len(K), dropped_near_public=len(out) - len(K), min_off_s=MIN_OFF_S, logs=len({log_of(o["id"]) for o in K}),
                    segments=len({o["log"] for o in K}), strata=dict(Counter(o["stratum"] for o in K)), chunks=chunks, smoke=sm,
                    splits=dict(arena_train=at.id, practice_starts=ps.id))
        (S / "meta.json").write_text(json.dumps(meta, indent=1))
        run.summary.update({k: v for k, v in meta.items() if k != "chunks"})
        print(json.dumps({k: v for k, v in meta.items() if k not in ("chunks", "smoke")}, indent=1))


def roll_stats(runs):
    """Finished roll chunk dirs -> per-chunk and total checklist numbers."""
    rows = []
    for d in runs:
        f = Path(d) / "aggregate/results-summary.json"
        if not f.exists():
            rows.append(dict(run=str(d), done=False))
            continue
        ro = json.loads(f.read_text())["rollouts"]
        dumps = glob.glob(f"{d}/driver-logs/dump/*.npz")
        mb = sum(os.path.getsize(x) for x in dumps) / 2 ** 20
        bad = [r for r in ro if r.get("failure_reason") and r["failure_reason"] not in ZERO_TYPES]   # amendment 2: a scorer zero is not a failed rollout
        sc = [0.0 if r.get("failure_reason") else (r.get("score") or 0.0) for r in ro if r not in bad]
        rows.append(dict(run=str(d), done=True, rollouts=len(ro), failed=len(bad), dumps=len(dumps), dump_mb=mb,
                         mean=float(np.mean(sc)) if sc else float("nan"), zeros=sum(not s > 0 for s in sc)))
    return rows


def cmd_stage2(a):
    rows = roll_stats(a.runs)
    ok = [r for r in rows if r["done"]]
    n = sum(r["rollouts"] for r in ok)
    fail = sum(r["failed"] for r in ok)
    dumps = sum(r["dumps"] for r in ok)
    mb = sum(r["dump_mb"] for r in ok)
    per_win = dumps / max(n - fail, 1)
    mean = float(np.mean([r["mean"] for r in ok])) if ok else float("nan")
    need_gb = mb / max(n, 1) * a.total_windows / 1024
    st = os.statvfs(DATA)
    free_gb = st.f_bavail * st.f_frsize / 2 ** 30
    chk = dict(all_done=len(ok) == len(rows), fail_rate=fail / max(n, 1), dumps_per_window=per_win, mean_score=mean,
               dump_gb_projected_all=need_gb, disk_free_gb=free_gb)
    chk["pass"] = bool(chk["all_done"] and chk["fail_rate"] <= 0.05 and 8.0 <= per_win <= 11.0 and 0.4 <= mean <= 1.0
                       and need_gb <= free_gb - a.reserve_gb)
    print(json.dumps(dict(check=chk, chunks=rows), indent=1))
    sys.exit(0 if chk["pass"] else 2)


def cmd_gate(a):
    sys.path[:0] = [str(REPO / "experiments/defect_fix/scripts")]
    import df
    out = {}
    for t in (a.pilot, a.control, *a.ref):
        r = df.cl_read(t, [a.control])
        out[t] = dict(held={k: v for k, v in (r["held"] or {}).items() if k != "pair"}, pair=(r["held"] or {}).get("pair", {}).get(a.control))
    P, C = out[a.pilot]["held"], out[a.control]["held"]
    assert P.get("scenes") == C.get("scenes") == 460, "held 460 reads incomplete"
    v = dict(rule=f"zeros(pilot) <= zeros(control) + {a.margin}", zeros_pilot=P["zeros"], zeros_control=C["zeros"],
             pass_=bool(P["zeros"] <= C["zeros"] + a.margin), paired_vs_control=out[a.pilot]["pair"])
    rec = dict(pilot=a.pilot, control=a.control, refs=a.ref, verdict=v, reads=out)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "pilot.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    L = [f"# PN pilot gate: {'PASS' if v['pass_'] else 'FAIL'}", "",
         f"Held 460 @18 (rm_chain.sh read, 6 chunks). Gate (prereg 4): {v['rule']}. Paired = tag minus {a.control}, session-clustered 95 % CI "
         "(defect_fix df.cl_read).", "",
         "| tag | zeros | mean | at-fault | offroad | lateral zeros | front / side / rear | progress_rel | creep / no-launch | paired vs control [CI] |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for t, x in out.items():
        h, p = x["held"], x["pair"]
        pr = f"{p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]" if p and t != a.control else ""
        L.append(f"| {t} | {h['zeros']} | {h['mean']:.4f} | {h['at_fault']} | {h['offroad']} | {h['lat_zeros']} | {h['coll']['front']} / "
                 f"{h['coll']['lateral']} / {h['coll']['rear']} | {h['progress_rel']:.3f} | {h['creep']} / {h['no_launch']} | {pr} |")
    (RES / "pilot.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    p = sub.add_parser("stage2")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--total-windows", type=int, required=True)
    p.add_argument("--reserve-gb", type=float, default=300.0)
    p = sub.add_parser("gate")
    p.add_argument("--pilot", required=True), p.add_argument("--control", required=True), p.add_argument("--ref", nargs="*", default=[])
    p.add_argument("--margin", type=int, default=2)
    a = ap.parse_args()
    dict(select=cmd_select, stage2=cmd_stage2, gate=cmd_gate)[a.cmd](a)


if __name__ == "__main__":
    main()
