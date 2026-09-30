"""Figure of todos/2026-10-01-log-expert-audit.md from research/results/log-expert-audit/{counts,errors}.csv."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "research/results/log-expert-audit", REPO / "research/figs"
SL = ["start", "stop", "turn_onset", "in_turn", "nudge", "lane_change"]
COMP = {"start": "lon", "stop": "lon", "turn_onset": "lat", "in_turn": "lat", "nudge": "lat", "lane_change": "lat"}
TICK = {s: f"{s.replace('_', chr(10))}\n({COMP[s]})" for s in SL}
# (label, counts dataset / split, errors dataset / plan_source / calib, colour)
DS = [("WOD-E2E train", ("WOD-E2E", "train"), ("WOD-E2E train", "r2 teacher, front-only 5 Hz", "raw"), S.PALETTE["blue"]),
      ("NAVSIM navtrain (GIMM plan)", ("NAVSIM navtrain", "train"), ("NAVSIM navtrain", "GIMM-interpolated (skill-pack N1/N2 rows)", "raw"), S.PALETTE["orange"]),
      ("nuScenes train", ("nuScenes", "train"), ("nuScenes train", "r2 teacher, 20 Hz clock", "raw"), S.PALETTE["green"])]


def main():
    c = pd.read_csv(RES / "counts.csv")
    c = c[c.variant == "base"]
    e = pd.read_csv(RES / "errors.csv")
    S.apply()
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, 2.5), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    x, w = np.arange(len(SL)), 0.26
    for k, (lab, (dn, sp), (en, src, cal), col) in enumerate(DS):
        off = (k - 1) * w
        cc = c[(c.dataset == dn) & (c.split == sp)].set_index("slice")
        axs[0].bar(x + off, [cc.loc[s, "events"] for s in SL], w, color=col, label=lab)
        ee = e[(e.dataset == en) & (e.plan_source == src) & (e.calib == cal)].set_index("slice")
        r = np.array([[ee.loc[s, f"{COMP[s]}3_ratio"], ee.loc[s, f"{COMP[s]}3_rlo"], ee.loc[s, f"{COMP[s]}3_rhi"]] for s in SL])
        axs[1].errorbar(x + off, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt="o", ms=3, elinewidth=.7, color=col)
        cp = np.array([[ee.loc[s, "capture"], ee.loc[s, "capture_lo"], ee.loc[s, "capture_hi"]] for s in SL])
        axs[2].errorbar(x + off, cp[:, 0], yerr=[cp[:, 0] - cp[:, 1], cp[:, 2] - cp[:, 0]], fmt="o", ms=3, elinewidth=.7, color=col)
    axs[0].set_yscale("log")
    for v in (100, 1000):
        axs[0].axhline(v, color="#777777", lw=.5, ls="--", zorder=0)
    axs[0].set_ylabel("independent events (train split)")
    axs[1].set_yscale("log")
    axs[1].axhline(1.0, color="#777777", lw=.5, ls="--", zorder=0)
    axs[1].axhline(1.5, color="#777777", lw=.5, ls=":", zorder=0)
    axs[1].set_ylabel("native ADE@3s ratio to control")
    axs[2].set_ylim(0, 1.05)
    axs[2].set_ylabel("native reproduces >= half\nof the human manoeuvre")
    for a, lab in zip(axs, ("(a)", "(b)", "(c)")):
        a.set_xticks(x, [TICK[s] for s in SL], fontsize=6.5)
        a.grid(axis="x", visible=False)
        S.panel(a, lab)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.3), ncol=3, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.3, top=0.93, wspace=0.38)
    print(S.save(fig, FIGS / "log-expert-audit"))


if __name__ == "__main__":
    main()
