#!/usr/bin/env python
"""Offline candidate table for the selective low-speed rule (gain s(a) = clip((a - d0) / (d1 - d0), 0, 1) on the plan's 1 s direction a),
from phi_steps.csv (baseline logs, lowspeed_sel_dist.py).   python <this> results/lowspeed_ctrl_selective/phi_steps.csv"""
import csv, sys
import numpy as np
R = {}
for r in csv.DictReader(open(sys.argv[1])):
    R.setdefault(r["class"], []).append(float(r["abs_phi_deg"]))
R = {k: np.array(v) for k, v in R.items()}
s = lambda a, d0, d1: np.clip((a - d0) / (d1 - d0), 0, 1)  # noqa: E731
print("| d0 | d1 | spin-event steps 1-4: residual <= 1 deg | median residual ratio | non-spin event steps 1-4 changed > 1 deg | non-spin all v<3.5 changed > 1 deg | > 2 deg | mean change deg |")
print("|---|---|---|---|---|---|---|---|")
for d0, d1 in [(1, 3), (1.5, 4), (2, 4), (2, 6), (3, 6), (3, 8), (2, 8), (1, 6)]:
    a = R["ev14_spin"]; p = a * s(a, d0, d1)
    a2 = R["ev14_non"]; p2 = a2 * s(a2, d0, d1)
    a3 = R["nonspin_all"]; p3 = a3 * s(a3, d0, d1)
    print(f"| {d0} | {d1} | {(p <= 1).mean():.2f} | {np.median(p / a):.2f} | {((a2 - p2) > 1).mean():.3f} | {((a3 - p3) > 1).mean():.3f} | {((a3 - p3) > 2).mean():.3f} | {(a3 - p3).mean():.2f} |")
a = R["ev14_spin"]
print(f"\nspin-event steps 1-4: n={len(a)}, baseline |phi| <= 1 deg {(a <= 1).mean():.2f}, <= 3 deg {(a <= 3).mean():.2f}; non-spin event steps n={len(R['ev14_non'])}, <= 3 deg {(R['ev14_non'] <= 3).mean():.2f}")
