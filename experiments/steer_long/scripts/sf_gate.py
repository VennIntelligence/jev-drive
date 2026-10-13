#!/usr/bin/env python3
"""SPIN-FIX offline gate screen (numpy only, either box): track_ref's gate check (curv reference, per path segment curvature against the
full mode's cap) on every logged plan of a read, with the logged ego speed (track v0, else ego[4] x 10). Per scene group: share of steps and
of rollouts where the gate would open at each threshold, and the rollouts' largest ratio. Open loop: the logged states are the read's own.
  sf_gate.py --drive <drive.jsonl> ... [--group name=<scenes.txt> ...] [--thr 1 1.25 1.5] [--top N] [--cfg k=v,...]"""
import argparse, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../alpasim/lib"))
import track_ref as TR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", nargs="+", required=True)
    ap.add_argument("--group", nargs="*", default=[])
    ap.add_argument("--thr", nargs="*", type=float, default=[1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--cfg", default="")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (kv.split("=") for kv in filter(None, a.cfg.split(",")))}
    G = {}
    for gs in a.group:
        n, f = gs.split("=")
        for s in open(f).read().split():
            G.setdefault(s if s.startswith("clipgt-") or "." in s else f"clipgt-{s}", []).append(n)
    R = {}                                                                # session -> (scene, [(t, ratio)])
    for f in a.drive:
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") != "drive" or "poses" not in r or ("infer" in r and not r["infer"]):
                continue
            v0 = r["track"]["v0"] if "track" in r else (r["fix"]["v0"] if "fix" in r else r["ego"][4] * 10)
            _, info = TR.build(np.asarray(r["poses"]), v0, 0.0, mode="curv", **kw)
            R.setdefault(r["session"], (r["scene"], []))[1].append((r["now"], info["gate"], v0))
    rows = {}
    for sess, (sc, xs) in R.items():
        g = np.array([x[1] for x in xs])
        for n in ["all", *G.get(sc, [])]:
            rows.setdefault(n, []).append((sc, g, xs))
    print(f"{'group':>10} {'roll':>5} {'steps':>7} " + " ".join(f"{'st>' + str(t):>9} {'ro>' + str(t):>9}" for t in a.thr))
    for n, L in rows.items():
        allg = np.concatenate([g for _, g, _ in L])
        print(f"{n:>10} {len(L):>5} {len(allg):>7} " + " ".join(f"{(allg > t).mean():>9.4f} {np.mean([(g > t).any() for _, g, _ in L]):>9.3f}" for t in a.thr))
    if a.top:
        for n, L in rows.items():
            if n == "all":
                continue
            print(f"-- {n}: rollouts by max ratio")
            for sc, g, xs in sorted(L, key=lambda z: -z[1].max())[:a.top]:
                i = int(g.argmax())
                print(f"   {sc[:40]:>40} max {g.max():6.2f} at step {i} v0 {xs[i][2]:5.2f}  steps>1 {(g > 1).sum():3d}  >1.5 {(g > 1.5).sum():3d}")


if __name__ == "__main__":
    main()
