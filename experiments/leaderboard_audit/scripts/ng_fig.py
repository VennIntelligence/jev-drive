"""Night-gap audit step 4: figure from results/night_gap/summary.json (Mac, .venv).  .venv/bin/python experiments/leaderboard_audit/scripts/ng_fig.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results/night_gap"
S = json.load(open(R / "summary.json"))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
C = {"op": "#1f6feb", "cv": "#8b949e"}
a = ax[0]
for i, (tag, nm) in enumerate((("op", "openpilot"), ("cv", "constant-arc baseline"))):
    d, n = S["wod_night_vs_day"][f"{tag}_ade3"]["mean_b"], S["wod_night_vs_day"][f"{tag}_ade3"]["mean_a"]
    a.bar([i * 3, i * 3 + 1], [d, n], color=[C[tag], C[tag]], alpha=1.0, edgecolor="k")
    for x, v, l in ((i * 3, d, "day"), (i * 3 + 1, n, "night")):
        a.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=9); a.text(x, 0.04, l, ha="center", color="w", fontsize=9, rotation=90, va="bottom")
a.set_xticks([0.5, 3.5]); a.set_xticklabels(["openpilot", "const-arc"]); a.set_ylabel("ADE@3s (m)"); a.set_title("WOD-E2E val, same 1 437 frames")
b = ax[1]
rows = [("WOD ADE@3s  openpilot", S["wod_night_vs_day"]["op_ade3"]["matched"], "op"), ("WOD ADE@3s  const-arc", S["wod_night_vs_day"]["cv_ade3"]["matched"], "cv"),
        ("nuScenes L2avg  openpilot", S["nusc_night_vs_day"]["op_l2avg"]["matched"], "op"), ("nuScenes L2avg  const-arc", S["nusc_night_vs_day"]["cv_l2avg"]["matched"], "cv")]
for i, (l, (p, lo, hi), t) in enumerate(rows[::-1]):
    b.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=C[t], capsize=3)
b.axvline(0, color="k", lw=.8); b.set_yticks(range(4)); b.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
b.set_xlabel("night - day error (m), speed x turn matched, 95% CI"); b.set_title("error gap (positive = night worse)")
c = ax[2]
rows = [("WOD RFS  openpilot", S["wod_rfs_night_vs_day"]["op_rfs"]["matched"], "op"), ("WOD RFS  const-arc", S["wod_rfs_night_vs_day"]["cv_rfs"]["matched"], "cv")]
for i, (l, (p, lo, hi), t) in enumerate(rows[::-1]):
    c.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", color=C[t], capsize=3)
c.axvline(0, color="k", lw=.8); c.set_yticks(range(2)); c.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8); c.set_ylim(-.6, 1.6)
c.set_xlabel("night - day RFS, matched, 95% CI"); c.set_title("RFS gap (negative = night worse)")
fig.suptitle("Shipped openpilot (Cinque), open loop: night vs day (sequence-cluster bootstrap)", fontsize=10)
fig.tight_layout(); fig.savefig(R / "night_gap.png", dpi=140)
