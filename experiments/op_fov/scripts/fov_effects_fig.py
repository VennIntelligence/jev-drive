"""Per-arm effect plot for the FOV pilot: dA_act, d lead, straight plan-speed ratio, with the cluster-bootstrap CIs
already stored in results/pilot_table.csv and results/pilot_frz_table.csv (no model run)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as ps

R = Path(__file__).resolve().parents[1] / "results"
df = pd.concat([pd.read_csv(R / "pilot_table.csv"), pd.read_csv(R / "pilot_frz_table.csv")])
arms = ["W90", "W116", "W90h", "R40", "Wfrz"]
col = {"W90": ps.PALETTE["blue"], "W116": ps.PALETTE["sky_blue"], "W90h": ps.PALETTE["green"],
       "R40": ps.PALETTE["vermillion"], "Wfrz": ps.BASELINE}
panels = [("turn", "A_act", "diff", r"$\Delta A_{act}$ vs N (turns)", 0.0),
          ("turn", "lead", "diff", r"$\Delta$ lead vs N (s, turns)", 0.0),
          ("straight", "v_ratio", "ratio", "plan speed / N (straights)", 1.0)]
ps.apply()
fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.3))
for ax, (st, m, op, lab, ref) in zip(axs, panels):
    for i, a in enumerate(arms):
        r = df[(df.arm == a) & (df.set == st) & (df.metric == m) & (df.op == op)].iloc[0]
        ax.errorbar(r["mean"], i, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", ms=3.5,
                    color=col[a], capsize=2, lw=1.1)
    ax.axvline(ref, color="#999999", lw=.6, zorder=0)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms if ax is axs[0] else [])
    ax.invert_yaxis()
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
axs[2].axvspan(0.98, 1.02, color="#DDDDDD", alpha=.5, lw=0)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "figs" / "fov_effects"
ps.save(fig, out)
(out.with_suffix(".pdf")).unlink()
