#!/usr/bin/env python3
"""M1 read-out (decision 205): closed-loop scene scores of fixed drivers against an unfixed one on the same scenes. Box, python + numpy.

  m1_report.py --base NAME=<run dir or glob>[,<...>] --fix NAME=<...> [--fix ...] --scenes <txt> [--groups <json: group -> [scenes]>] [--json out]

Per driver: scenes, mean scene score, zeros by flag, scenes at 1; per fixed driver the paired difference to the base with two 95% bootstrap
intervals (10 000 resamples, seed 0): resampling scenes, and resampling whole logs (`date_vehicle`); scenes fixed (0 -> > 0) and broken
(> 0 -> 0). With --groups the same per scene group.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

FAIL = ("collision_at_fault", "offroad", "left_corridor_laterally")
B = 10000


def load(spec):
    name, dirs = spec.split("=", 1)
    R = {}
    for D in (Path(d) for pat in dirs.split(",") for d in sorted(glob.glob(pat)) if (Path(d) / "aggregate/results-summary.json").exists()):
        for r in json.loads((D / "aggregate/results-summary.json").read_text())["rollouts"]:
            R.setdefault(r["clipgt_id"], r)
    return name, R


def log_of(s):
    return "_".join(s.split("_")[:2])


def ci(d, logs):
    rng = np.random.default_rng(0)
    n = len(d)
    a = d[rng.integers(0, n, (B, n))].mean(1)
    u, inv = np.unique(logs, return_inverse=True)
    sums, cnt = np.bincount(inv, d), np.bincount(inv)
    pick = rng.integers(0, len(u), (B, len(u)))
    b = sums[pick].sum(1) / cnt[pick].sum(1)
    return np.percentile(a, [2.5, 97.5]), np.percentile(b, [2.5, 97.5])


def flags(r):
    return [f for f in FAIL if r["score_metrics"].get(f)]


def table(base, fixes, scenes, title):
    bn, BR = base
    scenes = [s for s in scenes if s in BR and all(s in R for _, R in fixes)]
    out = {"n": len(scenes)}
    if not scenes:
        return [], out
    logs = np.array([log_of(s) for s in scenes])
    T = [f"### {title} ({len(scenes)} scenes)", "",
         "| driver | mean scene score | zeros | collision / offroad / corridor | at 1 | diff to base | 95% CI (scenes) | 95% CI (logs) | fixed 0 -> > 0 | broken > 0 -> 0 |",
         "|:--|--:|--:|:--|--:|--:|:--|:--|--:|--:|"]
    b = np.array([BR[s]["score"] for s in scenes])
    for name, R in [base] + fixes:
        x = np.array([R[s]["score"] for s in scenes])
        fl = [flags(R[s]) for s in scenes if R[s]["score"] == 0]
        row = dict(mean=float(x.mean()), zeros=int((x == 0).sum()), flags=[sum(f in q for q in fl) for f in FAIL], ones=int((x == 1).sum()))
        cell = f"| {name} | {x.mean():.4f} | {row['zeros']} | {' / '.join(map(str, row['flags']))} | {row['ones']} |"
        if R is BR:
            cell += " | | | | |"
        else:
            d = x - b
            c1, c2 = ci(d, logs)
            row.update(diff=float(d.mean()), ci_scene=c1.tolist(), ci_log=c2.tolist(), fixed=int(((b == 0) & (x > 0)).sum()), broken=int(((b > 0) & (x == 0)).sum()))
            cell += f" {d.mean():+.4f} | [{c1[0]:+.4f}, {c1[1]:+.4f}] | [{c2[0]:+.4f}, {c2[1]:+.4f}] | {row['fixed']} | {row['broken']} |"
        T.append(cell)
        out[name] = row
    return T + [""], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True), ap.add_argument("--fix", action="append", default=[]), ap.add_argument("--scenes", required=True)
    ap.add_argument("--groups"), ap.add_argument("--json")
    a = ap.parse_args()
    base, fixes = load(a.base), [load(f) for f in a.fix]
    scenes = Path(a.scenes).read_text().split()
    T, J = table(base, fixes, scenes, "all")
    J = {"all": J}
    if a.groups:
        for g, ss in json.loads(Path(a.groups).read_text()).items():
            t, j = table(base, fixes, [s for s in scenes if s in set(ss)], g)
            T += t
            J[g] = j
    print("\n".join(T))
    if a.json:
        Path(a.json).write_text(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
