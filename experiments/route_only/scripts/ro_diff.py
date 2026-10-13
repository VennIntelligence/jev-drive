#!/usr/bin/env python3
"""ROUTE-ONLY: scenes an arm broke / fixed against the reference (score > 0 vs 0) with the zero type. ro_diff.py "<ref glob>" "<arm glob>" """
import json, sys, glob, collections
def load(pat):
    m = {}
    for d in glob.glob(pat):
        for r in json.load(open(f"{d}/aggregate/results-summary.json"))["rollouts"]:
            m[r["clipgt_id"]] = r
    return m
def zt(r):
    if "sanity" in str(r.get("failure_reason") or ""): return "sanity"
    m = r.get("metrics") or {}
    for k, n in (("collision_at_fault", "collision"), ("offroad", "offroad"), ("left_corridor_laterally", "corridor")):
        if (m.get(k) or 0) > 0: return n
    return "other"
A, B = load(sys.argv[1]), load(sys.argv[2])
c = sorted(set(A) & set(B))
same = sum(1 for s in c if (A[s].get("score") or 0) == (B[s].get("score") or 0) and abs((A[s]["metrics"].get("dist_to_gt_trajectory") or 0) - (B[s]["metrics"].get("dist_to_gt_trajectory") or 0)) < 1e-6)
print("common", len(c), "identical score+GT", same)
br = [(s, zt(B[s])) for s in c if (A[s].get("score") or 0) > 0 and (B[s].get("score") or 0) == 0]
fx = [(s, zt(A[s])) for s in c if (A[s].get("score") or 0) == 0 and (B[s].get("score") or 0) > 0]
print("broken (ref>0, arm 0)", len(br), collections.Counter(t for _, t in br)); [print("  B", s, t, round(A[s]["score"], 3)) for s, t in br]
print("fixed (ref 0, arm>0)", len(fx), collections.Counter(t for _, t in fx)); [print("  F", s, t) for s, t in fx]
