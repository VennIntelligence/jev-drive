#!/usr/bin/env python3
"""SPIN-FIX: a JEV_TRACK read vs the plain read of the same checkpoint on nuPlan held (route_mem/read/<tag>): the gate's firing share, and the
paired score difference split by whether the gate opened in the rollout. sf_share.py <gate_tag>:<plain_tag> ..."""
import sys, glob, json, os
import numpy as np
def load(tag):
    S, G = {}, {}
    for d in sorted(glob.glob(f"/root/autodl-tmp/ujs/runs/route_mem/read/{tag}/c*/run")):
        sess = {}
        for ro in json.load(open(d + "/aggregate/results-summary.json"))["rollouts"]:
            S[ro["clipgt_id"]] = float(ro.get("score") or 0); sess[ro["rollout_id"]] = ro["clipgt_id"]
        for l in open(d + "/driver-logs/drive.jsonl"):
            r = json.loads(l)
            if r.get("kind") == "drive" and "track" in r and r["session"] in sess:
                G.setdefault(sess[r["session"]], []).append(r["track"].get("open", 0))
    return S, G
for gate, plain in [a.split(":") for a in sys.argv[1:]]:
    Sg, G = load(gate); Sp, _ = load(plain)
    calls = np.concatenate([np.array(v) for v in G.values()]) if G else np.zeros(1)
    both = [s for s in Sg if s in Sp]
    op = [s for s in both if any(G.get(s, []))]; cl = [s for s in both if s not in op]
    d = lambda L: (np.sum([Sg[s] - Sp[s] for s in L]), sum(Sg[s] == 0 and Sp[s] > 0 for s in L), sum(Sg[s] > 0 and Sp[s] == 0 for s in L))
    print(f"{gate} vs {plain}: {len(both)} scenes, gate open on {calls.mean()*100:.2f} % of {len(calls)} calls, in {len(op)} rollouts")
    for n, L in (("opened", op), ("never", cl)):
        s, nz, rz = d(L)
        print(f"   gate {n:>6}: {len(L):3d} rollouts, sum score diff {s:+.2f} (mean over all {s/len(both):+.4f}), new zeros {nz}, recovered {rz}")
