"""Figures for the WOD-E2E zero-shot exam (todos/2026-09-24-zeroshot-exam/wod-e2e.md), from the small result files
pulled to research/results/wod-zeroshot/. Runs on the Mac:  .venv/bin/python scripts/wod_zeroshot_figs.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = ROOT / "research/results/wod-zeroshot", ROOT / "research/figs"
COL = {"alpamayo_nav": S.PALETTE["blue"], "alpamayo_nonav": S.PALETTE["sky_blue"], "op_lebowski": S.PALETTE["vermillion"],
       "op_cinque": S.PALETTE["orange"], "op_small": S.PALETTE["purple"]}
LABEL = {"rater_best": "top-rated rater traj.", "logged_future": "logged future", "ours cls ego": "ours: cls ego (train-fit)",
         "ours cls_late vision+ego": "ours: cls_late vision+ego", "ours ridge ego": "ours: ridge ego",
         "cv": "constant velocity", "zero": "stand still", "rater_worst": "worst-rated rater traj.",
         "alpamayo_nav | E[1 sample]": "Alpamayo 1.5, nav", "alpamayo_nonav | E[1 sample]": "Alpamayo 1.5, no nav",
         "alpamayo_nav | medoid-of-6": "Alpamayo 1.5, nav, medoid-of-6",
         "op_lebowski": "openpilot Lebowski", "op_cinque": "openpilot Cinque v3", "op_small": "openpilot small"}
LEADER = {"RAP": 8.043, "Poutine": 7.986, "AutoVLA": 7.556, "OpenEMMA": 5.158}  # public test split


def color(row):
    for k, c in COL.items():
        if row.startswith(k):
            return c
    return S.BASELINE


def fig_rfs(res):
    rows = [r for r in LABEL if r in set(res.row)]
    d = res.set_index("row").loc[rows].sort_values("rfs")
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN, 2.9))
    y = np.arange(len(d))
    for i, (r, q) in enumerate(d.iterrows()):
        ax.errorbar(q.rfs, i, xerr=[[q.rfs - q.rfs_lo], [q.rfs_hi - q.rfs]], fmt="o", ms=3, color=color(r), lw=.8,
                    capsize=1.5)
    place = {"OpenEMMA": (0, "left"), "AutoVLA": (0, "right"), "Poutine": (1, "right"), "RAP": (1, "left")}
    for name, v in LEADER.items():  # public test split, not paired
        ax.axvline(v, color="#999999", lw=.5, ls=":", zorder=0)
        row, ha = place[name]
        ax.text(v + (.04 if ha == "left" else -.04), -1.0 - .6 * row, name, fontsize=5.5, ha=ha, va="top", color="#666666")
    ax.set_yticks(y, [LABEL[r] for r in d.index])
    ax.set_ylim(-2.3, len(d) - .5)
    ax.set_xlabel("RFS (cluster mean), val, n = 479")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=.47, right=.98, bottom=.15, top=.98)
    return S.save(fig, FIGS / "wod-zeroshot-rfs")


def fig_clusters(clus):
    style = {"logged_future": dict(marker="D", color="#222222", mfc="none"), "cv": dict(marker="o", color=S.BASELINE),
             "ours cls ego": dict(marker="s", color=S.BASELINE, mfc="none")}
    keep = [r for r in ("logged_future", "cv", "ours cls ego", "alpamayo_nav | E[1 sample]", "op_cinque", "op_lebowski")
            if r in clus.index]
    n = clus.loc["n"]
    cols = n.sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COLUMN_IN, 2.2))
    x = np.arange(len(cols))
    for j, r in enumerate(keep):
        st = style.get(r, dict(marker="o", color=color(r)))
        ax.plot(x + (j - (len(keep) - 1) / 2) * .1, clus.loc[r, cols].astype(float), ls="none", ms=3.2, mew=.8,
                label=LABEL[r], **st)
    names = {"Interections": "Intersections", "Foreign Object Debris": "Foreign obj.\ndebris",
             "Multi-Lane Maneuvers": "Multi-lane\nmaneuvers", "Single-Lane Maneuvers": "Single-lane\nmaneuvers",
             "Special Vehicles": "Special\nvehicles", "Cut_ins": "Cut-ins"}
    ax.set_xticks(x, [f"{names.get(c, c)}\n(n={int(n[c])})" for c in cols], fontsize=6.5)
    ax.set_ylabel("RFS (frame mean in cluster)")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=6.5, handletextpad=.2, columnspacing=1)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=.07, right=.99, bottom=.27, top=.88)
    return S.save(fig, FIGS / "wod-zeroshot-clusters")


if __name__ == "__main__":
    S.apply()
    res = pd.read_csv(RES / "results.csv")
    print(fig_rfs(res))
    print(fig_clusters(pd.read_csv(RES / "clusters.csv", index_col=0)))
