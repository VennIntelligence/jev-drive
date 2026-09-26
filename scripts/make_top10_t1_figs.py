"""Figure for the top-10 exams, executor T1 (todos/2026-09-26-top10-intersection.md, results): directional flip rate of
SparseDriveV2 and ZTRS on the P5 v1 BehaviorAgent pairs and the I3 HUGSIM vehicle pairs, per family, next to the
published reference examinees on the same judge (TFv6 waypoint channel, openpilot Cinque `ridge_late` prior).

    python scripts/make_top10_t1_figs.py      # reads research/results/{top10-exams,reactivity,elicitation}
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "research" / "results"
spec = importlib.util.spec_from_file_location("plot_style", REPO / "research" / "plot_style.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
ps.apply()

EX = {"SparseDriveV2": ("SparseDriveV2", ps.PALETTE["vermillion"]), "ZTRS": ("ZTRS", ps.PALETTE["blue"]),
      "TFv6 waypoint speed 2 s": ("TFv6 waypoint", ps.PALETTE["green"]),
      "openpilot": ("openpilot ridge_late (P5-fit)", ps.BASELINE)}
P5_FAM = ["pooled", "DynamicObjectCrossing", "ParkingCrossingPedestrian", "PedestrianCrossing", "HighwayCutIn",
          "ParkingCutIn", "StaticCutIn"]
SHORT = {"pooled": "Pooled", "DynamicObjectCrossing": "DynObjCross", "ParkingCrossingPedestrian": "ParkingPed",
         "PedestrianCrossing": "PedCross", "HighwayCutIn": "HwyCutIn", "ParkingCutIn": "ParkCutIn",
         "StaticCutIn": "StaticCutIn", "static": "Static", "cutin": "Cut-in", "oncoming": "Oncoming"}


def tables():
    p5 = pd.read_csv(RES / "top10-exams" / "t1_p5_flip_rates.csv")
    op5 = pd.read_csv(RES / "reactivity" / "v1-ba" / "mc" / "flip_rates.csv")
    p5 = pd.concat([p5, op5[op5.examinee == "prior [cinque]"].assign(examinee="openpilot")])
    i3 = pd.read_csv(RES / "top10-exams" / "t1_i3_flip_rates.csv")
    oi3 = pd.read_csv(RES / "elicitation" / "i3" / "flip_rates.csv")
    i3 = pd.concat([i3, oi3[oi3.examinee == "ridge_late op-cinque temporal"].assign(examinee="openpilot")])
    return p5, i3


def panel(ax, df, scopes, exs):
    w = 0.8 / len(exs)
    for j, e in enumerate(exs):
        d = df[df.examinee == e].set_index("scope").reindex(scopes)
        x = np.arange(len(scopes)) + (j - (len(exs) - 1) / 2) * w
        lab, c = EX[e]
        ax.bar(x, d.flip_rate, w, color=c, label=lab, linewidth=0)
        ax.errorbar(x, d.flip_rate, yerr=[d.flip_rate - d.flip_lo, d.flip_hi - d.flip_rate], fmt="none",
                    ecolor="#333333", elinewidth=0.5, capsize=1.2)
    n = df[df.examinee == exs[0]].set_index("scope").reindex(scopes).n_reactive.astype(int)
    ax.set_xticks(np.arange(len(scopes)))
    ax.set_xticklabels([f"{SHORT[s]}\n($n$={k})" for s, k in zip(scopes, n)], fontsize=6.5)
    ax.axhline(0.05, color=ps.BASELINE, linewidth=0.6, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional flip rate")
    ps.bars(ax)


def main():
    p5, i3 = tables()
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.35), gridspec_kw={"width_ratios": [7, 4]})
    panel(axs[0], p5, P5_FAM, ["SparseDriveV2", "ZTRS", "TFv6 waypoint speed 2 s", "openpilot"])
    ps.panel(axs[0], "(a) P5 v1, CARLA, BehaviorAgent labels")
    panel(axs[1], i3, ["pooled", "static", "cutin", "oncoming"], ["SparseDriveV2", "ZTRS", "openpilot"])
    ps.panel(axs[1], "(b) I3, HUGSIM 3DGS real appearance")
    axs[1].set_ylabel("")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3)
    out = REPO / "research" / "figs" / "top10-t1-flip-rates"
    print(ps.save(fig, out))


if __name__ == "__main__":
    main()
