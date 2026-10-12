#!/usr/bin/env python3
"""Route-night Tokyo: per-rollout facts of PAI 281 reads (pai_eval.sh run roots), plain python3 + numpy. Per label: mean score, zeros, handoff
spins (|vx wz| > 6 m/s^2 within 1.5 s after t = 1.7 s, steer-long's rule), and a turn proxy (no map read): turn scenes = scenes where a
reference label scored > 0 without a handoff spin and its own path turned 60 to 200 deg net (yaw integrated from the controller CSV, until its first zero event or the end); a label
completes a turn scene when its own path also turns >= 60 deg with the same sign and the scene scores > 0. Left / right by sign.

  rf_pai281.py --rows rsg=/data/runs/route_night/r281/rsg ys0=/data/runs/onpol_long/pai281/ys0 ... --ref ybase,ys0,ys1 [--out md]
"""
import argparse, csv, glob, json, os

import numpy as np


def read(root):
    out = {}
    for sim in glob.glob(os.path.join(root, "runs", "*", "sim")):
        f = os.path.join(sim, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rollouts"]:
            c = os.path.join(sim, "controller", f"alpasim_controller_{r['rollout_id']}.csv")
            if not os.path.exists(c):
                continue
            rows = list(csv.DictReader(open(c)))
            t = np.array([float(x["timestamp_us"]) for x in rows]) / 1e6
            vx, wz = (np.array([float(x[k]) for x in rows]) for k in ("vx", "wz"))
            rel = t - t.min() + 0.1
            w = (rel > 1.7) & (rel <= 3.2)
            yaw = np.degrees(np.r_[0, np.cumsum(wz[1:] * np.diff(t))])
            m = r.get("metrics", {})
            kind = "" if r["score"] else ("collision" if m.get("collision_at_fault") else "offroad" if m.get("offroad") else "corridor" if m.get("left_corridor_laterally") else "other")
            out[r["clipgt_id"]] = dict(score=float(r["score"] or 0), spin=bool(w.any() and np.abs(vx * wz)[w].max() > 6), turn=float(yaw[-1]), kind=kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--ref", default="ybase,ys0,ys1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = {k: read(v) for k, v in (x.split("=", 1) for x in a.rows)}
    common = sorted(set.intersection(*[set(d) for d in D.values()]))
    refs = [r for r in a.ref.split(",") if r in D]
    ok = lambda r, s: D[r][s]["score"] > 0 and not D[r][s]["spin"] and 60 <= abs(D[r][s]["turn"]) <= 200  # noqa: E731  a scored, non-spinning turn
    turns = {s: np.sign(next(D[r][s]["turn"] for r in refs if ok(r, s))) for s in common if any(ok(r, s) for r in refs)}
    L = [f"Scenes scored by every label: {len(common)}; turn scenes (a scoring reference label turned 60-200 deg): {len(turns)} "
         f"(left {sum(v > 0 for v in turns.values())}, right {sum(v < 0 for v in turns.values())})", "",
         "| label | mean | zeros | collision | offroad | corridor | handoff spins | turn done left | turn done right |", "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k, d in D.items():
        z = [d[s]["kind"] for s in common if d[s]["kind"]]
        done = lambda sg: [s for s, v in turns.items() if v == sg and d[s]["score"] > 0 and d[s]["turn"] * sg >= 60]  # noqa: E731
        nl, nr = sum(v > 0 for v in turns.values()), sum(v < 0 for v in turns.values())
        L.append(f"| {k} | {np.mean([d[s]['score'] for s in common]):.4f} | {len(z)} | {z.count('collision')} | {z.count('offroad')} | {z.count('corridor')} | "
                 f"{sum(d[s]['spin'] for s in common)} | {len(done(1))}/{nl} | {len(done(-1))}/{nr} |")
    print("\n".join(L))
    if a.out:
        open(a.out, "w").write("\n".join(L) + "\n")
        json.dump({k: {s: d[s] for s in common} for k, d in D.items()} | {"_turns": {s: float(v) for s, v in turns.items()}}, open(a.out[:-3] + ".json", "w"))


if __name__ == "__main__":
    main()
