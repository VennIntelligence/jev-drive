"""Figure for todos/2026-09-24-driving-backbones: cross-fit deltas vs `ridge ego`, general vs driving backbones.

Reads research/results/driving-backbones/crossfit_vs_ego_p3drive.csv (pulled from the box run) and writes
research/figs/driving-backbones-crossfit.png (+ .pdf beside it, not committed).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as S  # noqa: E402

ROWS = [  # (arm in the csv, label, colour)
    ("A ridge_late pooled (qwen4b L18)", "Qwen3-VL-4B L18 (A)", S.BASELINE),
    ("d2 qwenvid L18_last", "Qwen3-VL-4B video L18", S.BASELINE),
    ("d vjepa2 mean", "V-JEPA 2 ViT-L", S.BASELINE),
    ("op-small temporal", "openpilot small", S.PALETTE["sky_blue"]),
    ("op-cinque temporal", "openpilot Cinque", S.PALETTE["blue"]),
    ("op-lebowski temporal", "openpilot Lebowski", S.PALETTE["green"]),
    ("alp L18_mean", "Alpamayo 1.5 L18", S.PALETTE["vermillion"]),
    ("fusion op-cinque temporal + A", "Cinque + A", S.PALETTE["purple"]),
]
NATIVE = [("native op-cinque (no fit)", "Cinque native plan", S.PALETTE["blue"]),
          ("native op-lebowski (no fit)", "Lebowski native plan", S.PALETTE["green"])]
PANELS = [("(i) pre-onset, deciles 1-9", r"pre-onset $\Delta$ADE (m)", ROWS),
          ("(ii) all frames, deciles 1-9", r"all frames $\Delta$ADE (m)", ROWS),
          ("(iii) RFS, rater frames", r"$\Delta$RFS", ROWS + NATIVE)]


def main():
    t = pd.read_csv(ROOT / "research/results/driving-backbones/crossfit_vs_ego_p3drive.csv")
    S.apply()
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, 2.5), sharey=False)
    for k, (ax, (readout, xlabel, rows)) in enumerate(zip(axes, PANELS)):
        g = t[t.readout == readout].set_index("arm")
        rows = [r for r in rows if r[0] in g.index]
        y = np.arange(len(rows))[::-1]
        for yi, (arm, label, c) in zip(y, rows):
            r = g.loc[arm]
            ax.errorbar(r.delta, yi, xerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o", color=c, ms=3.2,
                        capsize=1.5, elinewidth=.8)
        ax.axvline(0, color="#999999", lw=.5, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([r[1] for r in rows] if k in (0, 2) else [])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        S.panel(ax, "(" + "abc"[k] + ")")
    fig.tight_layout(w_pad=.6)
    out = S.save(fig, ROOT / "research/figs/driving-backbones-crossfit")
    print(out)


if __name__ == "__main__":
    main()
