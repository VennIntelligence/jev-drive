#!/usr/bin/env python3
"""SPIN-FIX simulator pose faults (numpy only, either box). Every rollout of the given roots: the first controller row with a pose jump
(|dxy| > 1.6 x vx dt + 0.5 m) or a tilted pose (|qx| or |qy| > 0.3), with the spin detector's verdict; with --scene, the driver's anchor yaw
(full quaternion) against the controller's planar yaw 2 atan2(qz, qw) per inference in [--from, --to].
  sf_tilt.py <root> ... [--list scenes.txt] [--scene 8hex --from T --to T]"""
import argparse, csv, glob, json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from sf_detect import detect
from sf_trace import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+"); ap.add_argument("--list"); ap.add_argument("--scene")
    ap.add_argument("--from", dest="lo", type=float, default=0.0); ap.add_argument("--to", dest="hi", type=float, default=99.0)
    a = ap.parse_args()
    only = set(open(a.list).read().split()) if a.list else None
    for root in a.roots:
        if a.scene:
            ro, C, R = load(root, a.scene)
            ts = C["timestamp_us"]; t = (ts - ts[0]) / 1e6 + 0.1
            yaw = np.unwrap(2 * np.arctan2(C["qz"], C["qw"]))
            print(f"{root}: {ro['clipgt_id']}")
            for r in R:
                tr = (r["now"] - ts[0]) / 1e6 + 0.1
                if a.lo <= tr <= a.hi:
                    i = int(np.abs(t - tr).argmin()); an = r["anchor"]
                    print(f"  {tr:5.1f} anchor yaw {an[2]:7.3f} controller {yaw[i]:7.3f} diff {(an[2] - yaw[i] + np.pi) % (2 * np.pi) - np.pi:6.3f} "
                          f"tilt {np.hypot(C['qx'][i], C['qy'][i]):4.2f} u {C['u_steering_angle'][i]:6.3f}")
            continue
        n, hits = 0, []
        for sim in sorted(glob.glob(os.path.join(root, "runs", "*", "sim"))) + ([root] if os.path.exists(os.path.join(root, "aggregate")) else []):
            for ro in json.load(open(os.path.join(sim, "aggregate", "results-summary.json")))["rollouts"]:
                c = os.path.join(sim, "controller", f"alpasim_controller_{ro['rollout_id']}.csv")
                if (only and ro["clipgt_id"] not in only) or not os.path.exists(c):
                    continue
                rows = list(csv.DictReader(open(c)))
                C = {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}
                t = (C["timestamp_us"] - C["timestamp_us"][0]) / 1e6 + 0.1
                n += 1
                dxy = np.r_[0, np.hypot(np.diff(C["x"]), np.diff(C["y"]))]
                tilt = np.maximum(np.abs(C["qx"]), np.abs(C["qy"]))
                bad = (dxy > 1.6 * np.abs(C["vx"]) * 0.1 + 0.5) | (tilt > 0.3)
                if bad.any():
                    i = int(np.argmax(bad)); d = detect(t, C["vx"], C["vy"], C["wz"], C["front_steering_angle"])
                    hits.append(f"{ro['clipgt_id'][-36:][:8]} t {t[i]:.1f} at ({C['x'][i]:.1f}, {C['y'][i]:.1f}) jump {dxy[i]:.2f} m tilt {tilt[i]:.2f} "
                                f"spin {d['kind'] or '-'} {d['t_first'] or ''} score {float(ro.get('score') or 0):.3f}")
        print(f"{root}: {n} rollouts, {len(hits)} with a pose jump / tilt" + "".join(f"\n    {h}" for h in hits))


if __name__ == "__main__":
    main()
