"""Figure: zero-shot image-command effects per overlay family (navtrain), from results/nav_effects.csv (img_report.py).

  python experiments/op_img_cmd/scripts/img_figs.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
import plot_style as PS  # noqa: E402

TOP = REPO / "experiments" / "op_img_cmd"
FAMS = [("sign", "sign", "sky_blue"), ("arrow_road", "road arrow", "blue"), ("band", "band", "blue"), ("lines", "lane lines", "blue"),
        ("fill_grey", "grey fill", "green"), ("fill_grass", "grass fill", "green"), ("cones", "cones", "vermillion"),
        ("wall", "wall", "vermillion"), ("barrier", "barrier", "vermillion"), ("combo", "combo*", "purple")]
PANELS = [("delta", "move toward the commanded\nbranch, paired (m)", "(a) paired shift"),
          ("uptake", "fraction of a full\nbranch switch", "(b) uptake"),
          ("fcorrect", "plans nearer the commanded\nbranch (none = chance)", "(c) correct"),
          ("dx", "plan x change vs\nno overlay (m)", "(d) plan length")]


def main():
    PS.apply()
    E = pd.read_csv(TOP / "results" / "nav_effects.csv")
    E = E[(E.vbin == "all")]
    fig, axs = plt.subplots(1, 4, figsize=(PS.DOUBLE_COLUMN_IN, 2.3), constrained_layout=True)
    x = np.arange(len(FAMS))
    for ax, (k, ylab, title) in zip(axs, PANELS):
        for j, tau in enumerate((4.0, 6.0)):
            r = E[E.tau == tau].set_index("fam").reindex([f for f, *_ in FAMS])
            lo, hi = (r[f"{k}_lo"], r[f"{k}_hi"]) if f"{k}_lo" in r else (r[k], r[k])
            for i, (f, lab, col) in enumerate(FAMS):
                ax.errorbar(x[i] + (j - 0.5) * 0.32, r[k].iloc[i], yerr=[[r[k].iloc[i] - lo.iloc[i]], [hi.iloc[i] - r[k].iloc[i]]],
                            fmt="o" if tau == 4 else "s", ms=2.6, color=PS.PALETTE[col], mfc=PS.PALETTE[col] if tau == 4 else "white",
                            capsize=1.2, elinewidth=.7)
        if k == "fcorrect":
            ax.axhline(0.5, color=PS.BASELINE, lw=.6, ls="--")
        else:
            PS.zero_line(ax)
        PS.bars(ax)
        ax.set_xticks(x, [lab for _, lab, _ in FAMS], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylab)
        PS.panel(ax, title)
    h = [plt.Line2D([], [], marker="o", ls="", color="#444444", ms=3, label="4 s"),
         plt.Line2D([], [], marker="s", ls="", color="#444444", mfc="white", ms=3, label="6 s")]
    fig.legend(handles=h, loc="outside upper center", ncol=2, fontsize=7.5)
    print(PS.save(fig, TOP / "figs" / "zeroshot_effects"))


if __name__ == "__main__":
    main()
