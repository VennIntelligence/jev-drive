#!/usr/bin/env python
"""Figures of research/openpilot-openloop-standing.md from research/results/openpilot-openloop/*.csv (Mac, project venv).

  openloop-wod-rfs        WOD-E2E val rater frames: RFS (cluster mean, 95% CI) per row, test-split leaderboard as lines
  openloop-wod-timeline   openpilot's native plan under the exam's input vs NAVSIM's 1.5 s / 2 Hz timeline
  openloop-navsim         navtest PDMS / EPDMS per row with 95% CI, literature as lines
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "research/results/openpilot-openloop", REPO / "research/figs"
C = {"op": S.PALETTE["blue"], "head": S.PALETTE["sky_blue"], "alp": S.PALETTE["vermillion"], "ours": S.PALETTE["green"],
     "base": S.BASELINE}


def family(row: str) -> str:
    if row.startswith("op-") and ("ridge_late" in row or "cls_late" in row):
        return "head"
    if row.startswith("op-"):
        return "op"
    if row.lower().startswith("alpamayo"):
        return "alp"
    if row.startswith("ours"):
        return "ours"
    return "base"


LABEL = {"head": "frozen openpilot temporal + our head", "op": "openpilot native plan (zero-shot)",
         "alp": "Alpamayo 1.5 (zero-shot)", "ours": "our head, no openpilot", "base": "reference"}


def forest(ax, df, col, lo, hi, board=None, xlabel=""):
    df = df.sort_values(col)
    y = np.arange(len(df))
    for fam in LABEL:
        m = df["row"].map(family).to_numpy() == fam
        if m.any():
            ax.errorbar(df[col][m], y[m], xerr=[df[col][m] - df[lo][m], df[hi][m] - df[col][m]], fmt="o", ms=3,
                        color=C[fam], elinewidth=.8, capsize=0, label=LABEL[fam])
    ax.set_yticks(y, df["row"])
    ax.grid(axis="y", visible=False)
    for name, v in (board or {}).items():
        ax.axvline(v, color="#999999", lw=.5, ls="--", zorder=0)
        ax.text(v, -.6, name, rotation=90, ha="right", va="bottom", fontsize=6.5, color="#666666")
    ax.set_xlabel(xlabel)


def wod_rfs():
    df = pd.read_csv(RES / "wod_main.csv")
    df = df[~df["row"].str.contains("@")]                                  # timeline variants: their own figure
    board = pd.read_csv(RES / "wod_board.csv")
    board = {r.row.replace(" (test split)", ""): r.rfs_cluster for r in board.itertuples() if r.row.split()[0] in ("RAP", "Poutine", "UniPlan", "AutoVLA")}
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COLUMN_IN, 3.4))
    forest(ax, df, "rfs_cluster", "ci_lo", "ci_hi", board, "RFS (cluster mean, val rater frames, n = 479)")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    print(S.save(fig, FIGS / "openloop-wod-rfs"))


def wod_timeline():
    t = pd.read_csv(RES / "wod_timeline.csv")
    t = t[t["variant"].isin(["ctx1.5", "nav2hz", "nav2hz-dilate"])]
    models = ["small", "cinque", "lebowski"]
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN, 2.2))
    w = .26
    cols = {"exam input (10 Hz, 10 s)": S.PALETTE["blue"], "10 Hz, 1.5 s": S.PALETTE["sky_blue"],
            "NAVSIM input (2 Hz, 1.5 s)": S.PALETTE["vermillion"]}
    for j, (lab, c) in enumerate(cols.items()):
        vals, err = [], []
        for m in models:
            r = t[(t.model == m) & (t.variant == ("ctx1.5" if j == 1 else "nav2hz"))].iloc[0]
            if j == 0:
                vals.append(r.base_rfs_cluster); err.append((0, 0))
            else:
                vals.append(r.rfs_cluster); err.append((r.d_vs_base - r.lo, r.hi - r.d_vs_base))
        ax.bar(np.arange(3) + (j - 1) * w, vals, w, color=c, label=lab, yerr=np.array(err).T if j else None,
               error_kw={"elinewidth": .7, "capsize": 1.5})
    ax.axhline(7.103, color=S.BASELINE, lw=.7, ls="--")
    ax.text(2.45, 7.15, "constant velocity", ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax.set_xticks(range(3), ["small", "Cinque", "Lebowski"])
    ax.set_ylim(4, 8.6)
    ax.set_ylabel("RFS (cluster mean)")
    S.bars(ax)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, 1.28), ncol=2)
    fig.tight_layout()
    print(S.save(fig, FIGS / "openloop-wod-timeline"))


def navsim():
    p = RES / "navsim_navtest.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    board = pd.read_csv(RES / "navsim_board.csv").set_index("row")
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 3.4), sharey=True)
    order = df[df.metric == "EPDMS"].sort_values("score")["row"].tolist()
    for ax, metric, bcol in zip(axs, ("PDMS", "EPDMS"), ("PDMS", "EPDMS")):
        d = df[df.metric == metric].set_index("row").reindex(order).reset_index()
        d = d[d["row"] != "human (log)"]
        y = np.arange(len(d))
        for fam in LABEL:
            m = d["row"].map(family).to_numpy() == fam
            if m.any():
                ax.errorbar(d["score"][m], y[m], xerr=[d["score"][m] - d["ci_lo"][m], d["ci_hi"][m] - d["score"][m]],
                            fmt="o", ms=3, color=C[fam], elinewidth=.8, label=LABEL[fam])
        ax.set_yticks(y, d["row"])
        ax.grid(axis="y", visible=False)
        for name in ("Ego Status MLP (blind, navtrain)", "TransFuser", "DiffusionDrive"):
            v = board.loc[name, bcol]
            if pd.notna(v):
                ax.axvline(v, color="#999999", lw=.5, ls="--", zorder=0)
                ax.text(v, len(d) - .3, name.replace(" (blind, navtrain)", ""), rotation=90, ha="right", va="top",
                        fontsize=6.5, color="#666666")
        ax.set_xlabel(f"navtest {metric} (n = 12 146)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(.55, 0))
    fig.tight_layout(rect=(0, .1, 1, 1))
    print(S.save(fig, FIGS / "openloop-navsim"))


if __name__ == "__main__":
    S.apply()
    wod_rfs()
    wod_timeline()
    navsim()
