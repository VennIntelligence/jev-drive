"""COL1, PAI track: baseline against the diagnostic arm v1 (prereg amendments 1 and 2), from col1_pai_extract.py logs only.
  col1_pai_arm.py --base <x dir> [...] --arm <x dir> [...] --out <md>          (numpy)
Tables: scores and zeros by flag, the paired per-scene difference with a scene bootstrap, the controller's commands in the 1.2 s after
the hand-over for every scene, and ego speed against the log's at 4 / 8 / 12 s.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import col1_lib as L  # noqa: E402

FLAGS = ("collision_at_fault", "offroad", "left_corridor_laterally")


def rows(dirs):
    out = {}
    for d in dirs:
        for sc, x in L.load(f"{d}/logs.pkl").items():
            ego, gt = x["actors"]["EGO"], x["logged"][0]["traj"]
            T0, c = ego[0, 0], x["ctrl"]
            ci = {n: i for i, n in enumerate(x["ctrl_cols"])}
            t = (c[:, 0] - T0) * 1e-6
            w = (t > 1.75) & (t < 3.0)
            ts = T0 + np.array([4e6, 8e6, 12e6])
            out[sc[7:15]] = dict(score=x["summary"]["score"], flag=x["summary"].get("failure_reason") or "", v_ho=float(L.speed(ego, T0 + 1.7e6)[0]),
                                 steer=float(np.abs(c[w, ci["front_steering_angle"]]).max()), amin=float(c[w, ci["acceleration"]].min()),
                                 amax=float(c[w, ci["acceleration"]].max()), v=L.speed(ego, ts), vlog=L.speed(gt, ts),
                                 driven=x["summary"]["metrics"]["dist_traveled_m"], logd=x["summary"]["metrics"]["gt_dist_traveled_m"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True), ap.add_argument("--arm", nargs="+", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B, A = rows(a.base), rows(a.arm)
    S = sorted(set(B) & set(A), key=lambda s: B[s]["v_ho"])
    md = [f"# COL1 PAI: baseline against the speed-continuity arm v1 ({len(S)} scenes, P2H10-F-s0)\n",
          "| | mean scene score | score 1 | zeros | at-fault collision | offroad | left corridor |", "|---|--:|--:|--:|--:|--:|--:|"]
    for n, R in (("baseline", B), ("v1 (trajectory starts at the ego's speed)", A)):
        md.append(f"| {n} | {np.mean([R[s]['score'] for s in S]):.4f} | {sum(R[s]['score'] == 1 for s in S)} | {sum(R[s]['score'] == 0 for s in S)} | "
                  + " | ".join(str(sum(R[s]["flag"] == f for s in S)) for f in FLAGS) + " |")
    d = np.array([A[s]["score"] - B[s]["score"] for s in S])
    r = np.random.default_rng(0)
    lo, hi = np.percentile(d[r.integers(0, len(d), (10000, len(d)))].mean(1), [2.5, 97.5])
    md.append(f"\nPaired difference v1 - baseline: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (scene bootstrap, 10 000 draws, seed 0); "
              f"scenes up / down / same: {int((d > 0.01).sum())} / {int((d < -0.01).sum())} / {int((np.abs(d) <= 0.01).sum())}. "
              "One run per arm; the simulator's run-to-run spread is not in the interval.\n")
    md.append("Zero flag, baseline -> v1 (scenes): " + ", ".join(
        f"{k[0] or 'none'} -> {k[1] or 'none'}: {v}" for k, v in sorted(
            {(B[s]["flag"], A[s]["flag"]): sum((B[q]["flag"], A[q]["flag"]) == (B[s]["flag"], A[s]["flag"]) for q in S) for s in S}.items())) + "\n")
    md += ["| scene | speed at the hand-over m/s | baseline: score, flag | v1: score, flag | max abs steer rad, baseline / v1 | min accel m/s^2, baseline / v1 | "
           "max accel, baseline / v1 | ego speed at 4 / 8 / 12 s, baseline | v1 | log | driven m, baseline / v1 / log |",
           "|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    f3 = lambda v: " / ".join(f"{x:.1f}" for x in v)  # noqa: E731
    for s in S:
        b, v = B[s], A[s]
        md.append(f"| {s} | {b['v_ho']:.1f} | {b['score']:.2f} {b['flag']} | {v['score']:.2f} {v['flag']} | {b['steer']:.3f} / {v['steer']:.3f} | "
                  f"{b['amin']:.1f} / {v['amin']:.1f} | {b['amax']:.1f} / {v['amax']:.1f} | {f3(b['v'])} | {f3(v['v'])} | {f3(b['vlog'])} | "
                  f"{b['driven']:.0f} / {v['driven']:.0f} / {b['logd']:.0f} |")
    for n, R in (("baseline", B), ("v1", A)):
        q = np.array([R[s]["v"][2] / max(R[s]["vlog"][2], 1.0) for s in S if R[s]["flag"] not in ("offroad",)])
        md.append(f"\n{n}: ego speed at 12 s over the log's (scenes not offroad, n = {len(q)}): median {np.median(q):.2f}, above 1.2 in {int((q > 1.2).sum())}, "
                  f"below 0.8 in {int((q < 0.8).sum())}; driven distance over the log's, all scenes: median "
                  f"{np.median([R[s]['driven'] / R[s]['logd'] for s in S]):.2f}.")
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
