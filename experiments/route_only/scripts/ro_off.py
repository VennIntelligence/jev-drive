#!/usr/bin/env python3
"""ROUTE-ONLY: lateral offset of the ego from its own carried route on served decisions (route_path info: off, s_e) and the
max offset in scenes the arm broke. Box, AlpaSim venv.   ro_off.py "<arm run glob>" "<reference run glob>" """
import json, sys, glob, numpy as np, collections
def recs(pat):
    S = {}
    for d in glob.glob(pat):
        for l in open(f"{d}/driver-logs/drive.jsonl"):
            if '"route"' not in l: continue
            o = json.loads(l)
            if o.get("kind") == "drive": S.setdefault(o["scene"], {}).setdefault(o["session"], []).append(o)
    return {s: max(v.values(), key=lambda x: max(r["t"] for r in x)) for s, v in S.items()}
def lat(r):
    q = r['route']; g = max(0.0, -(q.get('s_e') or 0.0))
    return float(np.sqrt(max(q['off'] ** 2 - g ** 2, 0.0)))
def res(pat):
    m = {}
    for d in glob.glob(pat):
        for r in json.load(open(f"{d}/aggregate/results-summary.json"))["rollouts"]: m[r["clipgt_id"]] = r
    return m
R, A, B = recs(sys.argv[1]), res(sys.argv[1]), res(sys.argv[2])
off = np.array([lat(r) for v in R.values() for r in v if r["route"].get("cov", 0) > 0 and r["route"].get("off") is not None])
print("served decisions", len(off), "ego-route lateral offset median %.2f p90 %.2f p99 %.2f; share > 3 m %.3f, > 6 m %.3f" % (np.median(off), *np.percentile(off, [90, 99]), (off > 3).mean(), (off > 6).mean()))
mx = {s: max([lat(r) for r in v if r["route"].get("cov", 0) > 0 and r["route"].get("off") is not None] or [0]) for s, v in R.items()}
br = [s for s in B if s in A and (B[s].get("score") or 0) > 0 and (A[s].get("score") or 0) == 0]
print("broken scenes: max ego-route lateral offset on served decisions", {s[-8:]: round(mx.get(s, 0), 1) for s in br})
far = [s for s in mx if mx[s] > 6]
print("scenes with a served decision > 6 m off:", len(far), "of", len(mx), "; zero there arm / ref:", sum((A[s].get("score") or 0) == 0 for s in far if s in A), "/", sum((B[s].get("score") or 0) == 0 for s in far if s in B))
