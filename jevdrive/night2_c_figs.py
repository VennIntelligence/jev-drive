"""Figures of night queue 2, executor C (N5, N6), from the committed result tables:

  night2-n5-depth-recall     pedestrian BEV recall per distance bin, flat ground vs metric depth, P5 and nuScenes
  night2-n6-backbone-flips   P5 v1 BA pedestrian and cut-in flip per backbone: ridge_late vs pair-Delta, 3 seeds

    python -m jevdrive.night2_c_figs n5|n6
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES, FIGS = REPO / "research/results/night2", REPO / "research/figs"
BINS = ("0-10", "10-20", "20-40")


def _ps():
    from .p4_carla import _style
    return _style()


def n5():
    import matplotlib.pyplot as plt
    ps = _ps()
    t = pd.read_csv(RES / "N5/recall.csv")
    t = t[(t.gate == "gate") & (t.detector == "yolo26x-640")].set_index("placement")
    arms = [("flat", "Flat ground", ps.BASELINE), ("unidepth", "UniDepth v2 (K given)", ps.PALETTE["blue"]),
            ("da3", "DA3 Metric-L", ps.PALETTE["orange"])]
    arms = [a for a in arms if a[0] in t.index]
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.2), sharey=True)
    w = 0.8 / (len(arms) + 1)
    for ax, ds, lab in zip(axs, ("p5", "nusc"), ("(a) P5 (CARLA)", "(b) nuScenes val")):
        x = np.arange(len(BINS))
        for j, (k, name, c) in enumerate(arms):
            ax.bar(x + (j - len(arms) / 2) * w, [t.loc[k, f"{ds}_ped_{b}"] for b in BINS], w, color=c, label=name, linewidth=0)
        img = [t.loc["flat", f"{ds}_ped_img_{b}"] for b in BINS]
        ax.bar(x + (len(arms) / 2) * w, img, w, color="none", edgecolor="#333333", linewidth=0.6, hatch="////",
               label="Image plane (no BEV)")
        ax.axhline(0.40, color="#333333", linewidth=0.6, linestyle="--")
        ax.axhline(0.30, color="#333333", linewidth=0.6, linestyle=":")
        ax.set_xticks(x - w / 2)
        ax.set_xticklabels([f"{b} m" for b in BINS])
        ax.set_ylim(0, 1)
        ps.bars(ax)
        ps.panel(ax, lab)
    axs[0].set_ylabel("Pedestrian recall")
    axs[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(pad=0.3)
    ps.save(fig, FIGS / "night2-n5-depth-recall")
    plt.close(fig)


def n6():
    import matplotlib.pyplot as plt
    ps = _ps()
    c = pd.read_csv(RES / "N6/criteria_seeds.csv")
    c = c[c.prior == "cinque"]
    rows = [("qwen L18_last", "Qwen3-VL 4B"), ("vjepa2 mean", "V-JEPA 2 L"), ("dinov2 patch_mean", "DINOv2 B"),
            ("siglip2 patch_mean", "SigLIP2 So400m"), ("opsmall temporal", "openpilot small")]
    arms = [("ridge_late {}", "ridge_late", ps.BASELINE), ("pair-Δ {} [cinque]", "pair-Δ (single stream)", ps.PALETTE["vermillion"]),
            ("pair-Δ dual {} [cinque]", "pair-Δ (+ openpilot)", ps.PALETTE["orange"])]
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.4), sharey=True)
    w = 0.8 / len(arms)
    for ax, (col, lab) in zip(axs, (("ped_flip", "(a) Pedestrian"), ("cutin_flip", "(b) Cut-in"))):
        x = np.arange(len(rows))
        for j, (pat, name, clr) in enumerate(arms):
            v = [c[c.arm == pat.format(k)][col].to_numpy() for k, _ in rows]
            m = np.array([a.mean() if len(a) else np.nan for a in v])
            lo = np.array([a.min() if len(a) else np.nan for a in v])
            hi = np.array([a.max() if len(a) else np.nan for a in v])
            xx = x + (j - (len(arms) - 1) / 2) * w
            ax.bar(xx, m, w, color=clr, label=name, linewidth=0)
            ax.errorbar(xx, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.5)
            if col == "ped_flip" and j == 1:            # the N6 bar: seed-mean null false flip + 10 pp
                thr = [c[c.arm == pat.format(k)].null_ff_oos.mean() + 0.10 for k, _ in rows]
                ax.scatter(xx, thr, marker="_", s=60, color="#000000", zorder=3, label="null + 10 pp")
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in rows], rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ps.bars(ax)
        ps.panel(ax, lab)
    axs[0].set_ylabel("Directional flip rate")
    axs[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    ps.save(fig, FIGS / "night2-n6-backbone-flips")
    plt.close(fig)


if __name__ == "__main__":
    {"n5": n5, "n6": n6}[sys.argv[1]]()
