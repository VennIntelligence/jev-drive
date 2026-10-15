#!/usr/bin/env python3
"""MODE AVERAGING (plans/2026-10-15-mode-averaging-prereg.md; decision 306): is the plan's turn deficit larger where the junction offers
more exits, smaller where the command already names the turn, and does the plan near the entry look like an average of modes (partial
turns) or a choice between modes (straight or full)? CPU only; stored reads are read, never written.

  thk   (box, repo .venv)  rf_modeavg.py thk --runs TAG=<run>,<run> ... --held <RECEDING held pkl> --out <pkl>
        the flow head's logged samples (drive.jsonl `th_k`, noise rows 0 .. K-1) of every decision of the strict-turn rollouts.
  read  (Mac, .venv)  rf_modeavg.py read --dec <RECEDING decisions.parquet> --pai <pai pkl> --held <held pkl> --rmem NAME=<pkl> ...
        --lanes pai=<pkl> held=<pkl> --out DIR --fig PNG       Reading 1 -> DIR/{mode_avg.md, mode_avg.json, turns.csv, decisions.csv.gz}
  flow  (Mac, .venv)  Reading 2 on the FLOW1 closed-loop reads (see main()).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]


# ---------------------------------------------------------------- thk (box)
def thk(a):
    held = pickle.load(open(a.held, "rb"))
    out = {}
    for spec in a.runs:
        tag, runs = spec.split("=", 1)
        rids = {r["rid"] for r in held if r["set"] == tag}
        for run in runs.split(","):
            for line in open(os.path.join(run, "driver-logs", "drive.jsonl")):
                if '"th_k"' not in line:
                    continue
                r = json.loads(line)
                if r.get("session") in rids:
                    out.setdefault(r["session"], []).append((int(r["t0"]), np.asarray(r["th_k"], np.float32)))
        print(tag, len(rids), "strict rollouts,", sum(k in rids for k in out), "with samples", flush=True)
    res = {}
    for rid, v in out.items():
        v.sort(key=lambda x: x[0])
        t0 = np.array([x[0] for x in v], np.int64)
        keep = np.r_[True, np.diff(t0) > 0]
        res[rid] = dict(t0=t0[keep], th_k=np.stack([x[1] for x in v])[keep])
    pickle.dump(res, open(a.out, "wb"), protocol=4)
    print(len(res), "rollouts ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("thk")
    t.add_argument("--runs", nargs="+", required=True), t.add_argument("--held", required=True), t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"thk": thk}[a.cmd](a)
