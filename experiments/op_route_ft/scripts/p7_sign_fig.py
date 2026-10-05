"""Figure for results/p7_sign_check.md from p7_sign_check.json (no new run): sign agreement plan vs yaw rate per
arm / executor, and median plan vs measured curvature by speed bin, p7 vs curv."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as ps

D = json.load(open(Path(__file__).resolve().parents[1] / "results" / "p7_sign_check.json"))
K = "y1_vs_yawrate (v>0.5, |y1|>0.05)"
ps.apply()
fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})

arms = [("shipped", "shipped"), ("rc-ctl-s0", "rc-ctl"), ("rc-bear-s0", "rc-bear"), ("rc-poly-s0", "rc-poly"), ("POOLED", "pooled")]
x = np.arange(len(arms))
for j, (ex, c, lab) in enumerate([("p7", ps.PALETTE["vermillion"], "p7 (track plan)"), ("curv", ps.PALETTE["blue"], "curv (action)")]):
    v = [D.get(f"{k}|{ex}", {}).get(K, {}).get("agree", np.nan) for k, _ in arms]
    a.bar(x + (j - .5) * .38, v, .36, color=c, label=lab)
    for xi, vi in zip(x, v):
        if not np.isnan(vi):
            a.text(xi + (j - .5) * .38, vi + .008, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5)
a.axhline(.5, color="#999999", lw=.6, ls="--")
a.text(len(arms) - .55, .51, "chance", fontsize=6.5, color="#777777", ha="right")
a.set_xticks(x); a.set_xticklabels([l for _, l in arms])
a.set_ylim(.4, 1.14); a.set_ylabel("sign(plan y@1s) = sign(yaw rate)")
a.legend(loc="upper center", ncol=2); a.grid(axis="x", visible=False)
ps.panel(a, "(a)")

bins = [("v[0.3,1.5)", "0.3-1.5"), ("v[1.5,3)", "1.5-3"), ("v[3,99)", ">3")]
xb = np.arange(len(bins))
for ex, c, ls in [("p7", ps.PALETTE["vermillion"], "-"), ("curv", ps.PALETTE["blue"], "-")]:
    pk = [D[f"POOLED|{ex}"][k]["plan_k3s_med"] for k, _ in bins]
    mk = [D[f"POOLED|{ex}"][k]["meas_k_med"] for k, _ in bins]
    b.plot(xb, pk, "o-", color=c, label=f"{ex} plan")
    b.plot(xb, mk, "s--", color=c, ms=3, label=f"{ex} measured")
b.axhline(.1, color="#999999", lw=.6, ls=":")
b.text(-.1, .105, "R = 10 m", fontsize=6.5, color="#777777", ha="left", va="bottom")
b.set_yscale("log"); b.set_ylim(2e-4, .6); b.set_xticks(xb); b.set_xticklabels([l for _, l in bins])
b.set_xlabel("speed bin (m/s)"); b.set_ylabel("median curvature (1/m)")
b.legend(ncol=2, loc="lower left")
ps.panel(b, "(b)")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figs" / "p7_sign_check"
ps.save(fig, out)
out.with_suffix(".pdf").unlink()
