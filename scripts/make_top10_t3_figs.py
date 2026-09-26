"""Figure for the top-10 exams, executor T3 (todos/2026-09-26-top10-intersection.md, results): directional flip rate of
BridgeDrive (both channels), BLUE and its base SimLingo on the P5 v1 BehaviorAgent pairs, per family, per frame (a) and per pair (b), next
to TFv6's two channels from the original recording on the same judge.

    python scripts/make_top10_t3_figs.py      # reads research/results/top10-exams/p5_t3_*.csv
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "research" / "results" / "top10-exams"
spec = importlib.util.spec_from_file_location("plot_style", REPO / "research" / "plot_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
ps.apply()

EX = {"BridgeDrive target speed": ("BridgeDrive target speed", ps.PALETTE["vermillion"]),
      "BridgeDrive waypoint speed 2 s": ("BridgeDrive waypoint", ps.PALETTE["orange"]),
      "BLUE waypoint speed 2 s": ("BLUE waypoint", ps.PALETTE["blue"]),
      "SimLingo waypoint speed 2 s": ("SimLingo waypoint", ps.PALETTE["sky_blue"]),
      "TFv6 target speed": ("TFv6 target speed", ps.PALETTE["green"]),
      "TFv6 waypoint speed 2 s": ("TFv6 waypoint", ps.BASELINE)}
FAM = ["pooled", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "PedestrianCrossing", "HighwayCutIn",
       "ParkingCutIn", "StaticCutIn"]
SHORT = {"pooled": "Pooled", "DynamicObjectCrossing": "DynObjCross", "ParkingCrossingPedestrian": "ParkingPed",
         "PedestrianCrossing": "PedCross", "HighwayCutIn": "HwyCutIn", "ParkingCutIn": "ParkCutIn",
         "StaticCutIn": "StaticCutIn", "pedestrian": "Pedestrian", "cut-in": "Cut-in"}


def panel(ax, df, scopes, val, lo, hi, n):
    exs = list(EX)
    w = 0.8 / len(exs)
    for j, e in enumerate(exs):
        d = df[df.examinee == e].set_index("scope").reindex(scopes)
        x = np.arange(len(scopes)) + (j - (len(exs) - 1) / 2) * w
        lab, c = EX[e]
        ax.bar(x, d[val], w, color=c, label=lab, linewidth=0)
        ax.errorbar(x, d[val], yerr=[d[val] - d[lo], d[hi] - d[val]], fmt="none", ecolor="#333333", elinewidth=0.5,
                    capsize=1.2)
    k = df[df.examinee == exs[0]].set_index("scope").reindex(scopes)[n].fillna(0).astype(int)
    ax.set_xticks(np.arange(len(scopes)))
    ax.set_xticklabels([f"{SHORT[s]}\n($n$={m})" for s, m in zip(scopes, k)], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional flip rate")
    ps.bars(ax)


def main():
    fl = pd.read_csv(RES / "p5_t3_flip_rates.csv")
    pp = pd.read_csv(RES / "p5_t3_per_pair.csv")
    pp = pp[pp.window == "(b) per pair"]
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.6), gridspec_kw={"width_ratios": [7, 3]})
    panel(axs[0], fl, FAM, "flip_rate", "flip_lo", "flip_hi", "n_reactive")
    ps.panel(axs[0], "(a) per frame")
    panel(axs[1], pp, ["pooled", "pedestrian", "cut-in"], "flip", "lo", "hi", "n")
    ps.panel(axs[1], "(b) per pair")
    axs[1].set_ylabel("")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=(0, 0.14, 1, 1), pad=0.3)
    print(ps.save(fig, REPO / "research" / "figs" / "top10-t3-flip-rates"))


if __name__ == "__main__":
    main()
