#!/usr/bin/env python
"""Scored scenario sample of the HUGSIM zero-shot exam (pre-registered in todos/2026-09-25-hugsim-exam/README.md):
k scenarios per (dataset, difficulty) stratum, drawn with a fixed seed from research/results/hugsim-exam/scenarios.csv,
leaving out scenarios that need the HD map (not set up) and every scene used by the adapter checklist.

    python scripts/hugsim/zs_sample.py --k 4 > todos/2026-09-25-hugsim-exam/scored.txt
"""
import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows = list(csv.DictReader(open(REPO / "research" / "results" / "hugsim-exam" / "scenarios.csv")))
check = {(p.split("/")[0], p.split("/")[1].rsplit("-", 2)[0])
         for p in (REPO / "todos" / "2026-09-25-hugsim-exam" / "checklist.txt").read_text().split()}
pool = [r for r in rows if r["hd_map"] == "False" and (r["dataset"], r["scene"]) not in check]
rng = np.random.default_rng(a.seed)
out = []
for ds in ("nuscenes", "waymo", "kitti360", "pandaset"):
    for diff in ("easy", "medium", "hard", "extreme"):
        s = sorted(r["scenario"] for r in pool if r["dataset"] == ds and r["difficulty"] == diff)
        out += [f"{ds}/{x}.yaml" for x in sorted(rng.choice(s, min(a.k, len(s)), replace=False))]
print("\n".join(out))
