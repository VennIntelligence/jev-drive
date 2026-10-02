"""Figure: the three primary effects by speed bin and domain, from results/effects.csv (cc_report.py). Mac or box, CPU.

  python experiments/op_common_cause/scripts/cc_figs.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
import plot_style as PS  # noqa: E402

TOP = REPO / "experiments" / "op_common_cause"
DOM = {"nav": ("navtrain (real, GIMM history)", PS.PALETTE["blue"]), "wod": ("WOD (real stream)", PS.PALETTE["sky_blue"]),
       "wodt": ("WOD turn frames", PS.PALETTE["green"]), "carla": ("CARLA (P4 stream)", PS.PALETTE["vermillion"])}
PANELS = [("E1_G_deg", "plan heading at 3 s per\n10 deg/s fake history yaw (deg)", "(a) follows fake history rotation"),
          ("E2_dlat_repeat", "change of |lateral error| at 3 s (m)", "(b) static history (t0 frame repeated)"),
          ("E5_toward_cmd_pulse_m", "plan shift toward the command\nat 3 s (m)", "(c) turn-desire pulse, turn frames")]
BINS = ["stop", "low", "mid", "high"]


def main():
    PS.apply()
    E = pd.read_csv(TOP / "results" / "effects.csv")
    fig, axs = plt.subplots(1, 3, figsize=(PS.DOUBLE_COLUMN_IN, 2.4), constrained_layout=True)
    for ax, (eff, ylab, title) in zip(axs, PANELS):
        doms = [d for d in DOM if ((E.domain == d) & (E.effect == eff)).any()]
        w = 0.8 / len(doms)
        for k, d in enumerate(doms):
            r = E[(E.domain == d) & (E.effect == eff)].set_index("group").reindex(BINS)
            ok = r["n"].fillna(0) >= 20
            x = np.arange(len(BINS)) + (k - (len(doms) - 1) / 2) * w
            ax.errorbar(x[ok], r["mean"][ok], yerr=[(r["mean"] - r["lo"])[ok], (r["hi"] - r["mean"])[ok]], fmt="o", ms=3,
                        color=DOM[d][1], label=DOM[d][0], capsize=1.5, elinewidth=.8)
        PS.zero_line(ax)
        PS.bars(ax)
        ax.set_xticks(range(len(BINS)), ["<0.5", "0.5-3", "3-8", ">8"])
        ax.set_xlabel("speed at t0 (m/s)")
        ax.set_ylabel(ylab)
        PS.panel(ax, title)
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7.5)
    (TOP / "figs").mkdir(exist_ok=True)
    print(PS.save(fig, TOP / "figs" / "primary_effects"))


if __name__ == "__main__":
    main()
