#!/usr/bin/env python3
"""P4 box determinism read (results/p4_self/box_det.md) over rf_p4det.sh runs: every identity session against (a) the Tokyo reference of
its scene and (b) the scene's box anchor (the first session of the first run given): served plan at every common decision k (bit-identical
count, first diverging k, largest |diff| over all plan points and over the first 15 (the 4 s horizon, docs/long-runs.md), per k band),
controller track (rows, largest and final position difference), score, zero kind and time (rf_p4.zero_time = rf_branch's rule).
rf_p4.py s1's comparison, plus the first diverging k and the boundedness bands. AlpaSim venv (rf_tax.series reads metrics.parquet).
  rf_p4det.py --runs <run dir>[,..] --ref <id8>=<run dir>[,..] --out <json>    a run dir = rf_p4det.sh's <root>/<arm> or a pai_eval chunk
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_p4 as F  # noqa: E402

BANDS = ((0, 50), (50, 100), (100, 150), (150, 200))


def sessions(run):
    rows = {r["rollout_id"]: (sim, r) for sim, r in F.rollouts(run)}
    p = os.path.join(run, "driver", "start.jsonl")
    st = {}
    if os.path.exists(p):
        for line in open(p):
            x = json.loads(line)
            st[x["session"]] = x["t"]
    out = []
    for s, rr in F.drive_records(run).items():
        rr = [x for x in rr if "poses" in x]
        sim, row = rows.get(s, (None, None))
        if rr and row:
            kz, tz, _ = F.zero_time(sim, rr[0]["scene"], row)
            out.append(dict(run=os.path.basename(run.rstrip("/")), scene=rr[0]["scene"], session=s, rr=rr, sim=sim, row=row, t=st.get(s, 0),
                            score=float(row.get("score") or 0), zero=kz or "-", t_zero=None if tz is None or not np.isfinite(tz) else float(tz)))
    return sorted(out, key=lambda x: x["t"])


def track(x):
    f = os.path.join(x["sim"], "controller", f"alpasim_controller_{x['session']}.csv")
    return np.array([[float(r["x"]), float(r["y"])] for r in csv.DictReader(open(f))]) if os.path.exists(f) else np.zeros((0, 2))


def srv(x):
    return np.asarray(x.get("poses_branch", x["poses"]), float)


def cmp(a, b):
    B = {x["k"]: x for x in b["rr"]}
    pr = [(x["k"], srv(x), srv(B[x["k"]])) for x in a["rr"] if x["k"] in B]
    ks = np.array([k for k, _, _ in pr])
    d = np.array([np.abs(p - q).max() for _, p, q in pr])
    d15 = np.array([np.abs(p[:15] - q[:15]).max() for _, p, q in pr])
    ta, tb = track(a), track(b)
    m = min(len(ta), len(tb))
    e = np.hypot(*(ta[:m] - tb[:m]).T) if m else np.array([np.nan])
    nz = np.flatnonzero(d > 0)
    return dict(n=len(ks), n_a=len(a["rr"]), n_b=len(b["rr"]), equal=int((d == 0).sum()), first_k=int(ks[nz[0]]) if len(nz) else None,
                max=float(d.max()), max15=float(d15.max()), d_first=float(d[nz[0]]) if len(nz) else 0.0,
                bands={f"{lo}-{hi}": float(d[(ks >= lo) & (ks < hi)].max()) if ((ks >= lo) & (ks < hi)).any() else None for lo, hi in BANDS},
                bands15={f"{lo}-{hi}": float(d15[(ks >= lo) & (ks < hi)].max()) if ((ks >= lo) & (ks < hi)).any() else None for lo, hi in BANDS},
                track_rows=[len(ta), len(tb)], track_max=float(np.nanmax(e)), track_end=float(e[-1]),
                score=[a["score"], b["score"]], zero=[a["zero"], b["zero"]], t_zero=[a["t_zero"], b["t_zero"]],
                identical=bool(len(ks) and not len(nz) and len(ta) == len(tb) and m and float(np.nanmax(e)) == 0.0 and a["score"] == b["score"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = [s for run in a.runs.split(",") for s in sessions(run)]
    R = {}
    for kv in a.ref.split(","):
        k, run = kv.split("=", 1)
        for s in sessions(run):
            if s["scene"][7:15] == k:
                R.setdefault(k, []).append(s)
    anchor, out = {}, []
    for s in S:
        anchor.setdefault(s["scene"], s)
    for k, rr in R.items():                                       # Tokyo against Tokyo (sessions after the first)
        for s in rr[1:]:
            out.append(dict(scene=k, run="tokyo:" + s["run"], session=s["session"][:8], vs="tokyo", **cmp(s, rr[0])))
    for s in S:
        k = s["scene"][7:15]
        for vs, b in (("tokyo", (R.get(k) or [None])[0]), ("box", anchor[s["scene"]])):
            if b is not None and b is not s:
                out.append(dict(scene=k, run=s["run"], session=s["session"][:8], vs=vs, **cmp(s, b)))
    ref = {k: dict(run=v[0]["run"], session=v[0]["session"], score=v[0]["score"], zero=v[0]["zero"], t_zero=v[0]["t_zero"], n_dec=len(v[0]["rr"]))
           for k, v in R.items()}
    anc = {s["scene"][7:15]: dict(run=s["run"], session=s["session"], score=s["score"], zero=s["zero"], t_zero=s["t_zero"], n_dec=len(s["rr"]))
           for s in anchor.values()}
    json.dump(dict(ref=ref, anchor=anc, n_sessions=len(S), rows=out), open(a.out, "w"), indent=1)
    print("| scene | run | session | vs | decisions | equal | first k | d at first k | max (all pts) | max (15 pts) | track max / end m | score | zero |")
    print("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
    for r in out:
        print(f"| {r['scene']} | {r['run']} | {r['session']} | {r['vs']} | {r['n']} | {r['equal']} | {r['first_k']} | {r['d_first']:.4f} | {r['max']:.4f} | "
              f"{r['max15']:.4f} | {r['track_max']:.4f} / {r['track_end']:.4f} | {r['score'][0]:.4f} / {r['score'][1]:.4f} | {r['zero'][0]} / {r['zero'][1]} |")


if __name__ == "__main__":
    main()
