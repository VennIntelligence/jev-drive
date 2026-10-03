"""Figure for results/spin_attribution.md.  python spin_attr_fig.py <out_dir> <png>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

o = Path(sys.argv[1])
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
leaky = {"frac_v3", "frac_v1", "frac_stop", "launch_rate"}
for k, ag, col in ((0, "cinque", "#1f77b4"), (1, "lebowski", "#d62728")):
    A = pd.read_csv(o / f"q1_auc_{ag}.csv")
    A = A.reindex(A.auc.sub(0.5).abs().sort_values(ascending=False).index).head(12)[::-1]
    y = np.arange(len(A)) + (-0.18 if k == 0 else 0.18)
    ax[0].errorbar(A.auc, y, xerr=[A.auc - A.lo, A.hi - A.auc], fmt="o", color=col, label=ag, capsize=2, ms=4)
    if k == 0:
        ax[0].set_yticks(np.arange(len(A)))
        ax[0].set_yticklabels([f + (" *" if f in leaky else "") for f in A.feature])
ax[0].axvline(0.5, color="k", lw=0.8)
ax[0].set_xlabel("AUC for spin (10 / 11 vs 54 / 53 scenes), 95% bootstrap CI")
ax[0].set_title("(a) per-feature separation, top 12 by |AUC - 0.5| (Cinque order; * window-length dependent)")
ax[0].legend()
M = pd.read_csv(o / "q3_pairs.csv")
names = ["cinque", "lebowski", "pilot", "it_dw3", "ln1", "ln3"]
J = np.eye(len(names))
for _, r in M.iterrows():
    if r.a in names and r.b in names:
        i, j = names.index(r.a), names.index(r.b)
        J[i, j] = J[j, i] = r.jaccard
im = ax[1].imshow(J, vmin=0, vmax=0.6, cmap="Blues")
for i in range(len(names)):
    for j in range(len(names)):
        ax[1].text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=45)
ax[1].set_yticks(range(len(names)))
ax[1].set_yticklabels(names)
ax[1].set_title("(b) Jaccard of spin scene sets (64 scenarios)")
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=130)
