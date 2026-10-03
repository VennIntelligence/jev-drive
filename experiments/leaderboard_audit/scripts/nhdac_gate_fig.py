"""Figure for the gated tracker compensation: navtrain PDMS delta against the fraction of tokens compensated (python with pandas + matplotlib)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parents[1]
d = pd.read_csv(R / "results/navhard_dac/gated_navtrain_grid.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
col = {0.25: "#1b9e77", 0.5: "#7570b3", 0.75: "#d95f02", 1.0: "#e7298a"}
for a, g in zip(ax, ("gE", "gK")):
    for (m, al), x in d[d.gate == g].groupby(["mode", "alpha"]):
        x = x.sort_values("frac")
        a.plot(x.frac, x.delta, "-o" if m == "path" else "--s", color=col[al], ms=4, label=f"{m} alpha {al:g}")
    a.axhline(0, color="k", lw=0.8)
    a.set_xscale("log"); a.set_xticks([5, 10, 20, 35, 50, 100]); a.set_xticklabels(["5", "10", "20", "35", "50", "100"])
    a.set_xlabel("share of tokens compensated (%), largest gate score first"); a.set_title({"gE": "gate gE: predicted tracking error", "gK": "gate gK: plan end heading"}[g], fontsize=10)
ax[0].set_ylabel("navtrain PDMS delta vs uncompensated")
ax[1].legend(fontsize=7, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig(R / "figs/navhard_gated_comp_navtrain.png", dpi=130)
