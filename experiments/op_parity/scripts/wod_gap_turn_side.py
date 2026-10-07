"""WOD val turn frames: which side of the top-rated path each arm ends on at 5 s (post hoc read on wod_gap/frames.csv)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "research"))
import plot_style as ps  # noqa: E402

ARMS = {"WP2_s0": ("WP2 (seed 0)", ps.PALETTE["blue"]), "shipped": ("shipped", ps.PALETTE["orange"]), "log": ("log", ps.BASELINE)}
NEAR = 3.0  # m: |longitudinal miss| below this, so the lateral sign is not an along-path shortfall

d = pd.read_csv(ROOT / "results/wod_gap/frames.csv")
t = d[d.intent >= 2]
sign = np.where(t.intent == 2, 1, -1)  # e_lat is + left of the rater; inward = toward the turn side
rows = []
ps.apply()
fig, axes = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.6), sharex=True, sharey=True)
for ax, (a, (name, c)) in zip(axes, ARMS.items()):
    lat, lon = t[f"e_lat5_{a}"].to_numpy() * sign, t[f"e_lon5_{a}"].to_numpy()
    near = np.abs(lon) < NEAR
    rows.append(dict(arm=name, n=len(t), median_inward_m=np.median(lat), inside_gt1m=int((lat > 1).sum()), wide_gt1m=int((lat < -1).sum()),
                     median_lon_m=np.median(lon), n_near=int(near.sum()), near_median_inward_m=np.median(lat[near]),
                     near_inside_gt1m=int((lat[near] > 1).sum()), near_wide_gt1m=int((lat[near] < -1).sum())))
    ax.axvline(0, color="#999999", lw=.5, zorder=0); ax.axhline(0, color="#999999", lw=.5, zorder=0)
    ax.axhspan(-NEAR, NEAR, color="#DDDDDD", alpha=.35, lw=0, zorder=0)
    ax.scatter(lat[~near], lon[~near], s=9, facecolors="none", edgecolors=c, linewidths=.6)
    ax.scatter(lat[near], lon[near], s=9, color=c, linewidths=0)
    ax.axvline(np.median(lat), color=c, lw=.9, ls="--")
    ax.set_title(f"{name}: median {np.median(lat):+.2f} m, inside {int((lat > 1).sum())} / wide {int((lat < -1).sum())}", fontsize=7.5)
axes[0].set_ylabel("longitudinal miss at 5 s, + = further (m)")
axes[0].set_xlim(-8, 10); axes[0].set_ylim(-22, 22)
fig.supxlabel("lateral miss at 5 s, + = inside of the turn (m)", fontsize=8.5)
fig.tight_layout()
ps.save(fig, ROOT / "figs/wod_gap/10_turn_side")
out = pd.DataFrame(rows).round(2)
out.to_csv(ROOT / "results/wod_gap/turn_side.csv", index=False)
print(out.to_string(index=False))
