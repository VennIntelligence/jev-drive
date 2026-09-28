#!/usr/bin/env python
"""Figures of research/openpilot-openloop-integration.md from research/results/op-interp/*.csv (Mac, project venv).

  op-interp-wod     WOD-E2E rater frames: openpilot RFS per 2 Hz feed (hold / blend / warp / RIFE / GIMM / real 10 Hz),
                    plan as is vs re-timed to the measured speed; and synthesized-frame PSNR vs the RFS gap recovered
  op-interp-navsim  navtest subset: PDMS of the Cinque ladder (feed x adapter) with references
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "research/results/op-interp", REPO / "research/figs"
FEEDS = ["hold", "blend", "warp", "rife", "gimm", "real"]
FEED_LABEL = {"hold": "hold (exam)", "blend": "cross-fade", "warp": "ego-motion warp", "rife": "RIFE 4.26",
              "gimm": "GIMM-VFI", "real": "real 10 Hz"}
FEED_COLOR = {"hold": S.BASELINE, "blend": S.PALETTE["yellow"], "warp": S.PALETTE["green"], "rife": S.PALETTE["sky_blue"],
              "gimm": S.PALETTE["blue"], "real": S.PALETTE["black"]}


def wod():
    df = pd.read_csv(RES / "wod_results.csv")
    df = df[~df["frames"].str.startswith("exam")]
    img = pd.read_csv(RES / "wod_img.csv")
    S.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.2), gridspec_kw={"width_ratios": [2.3, 1]})
    models = [m for m in ("small", "cinque", "lebowski") if (df.model == m).any()]
    w = 0.13
    for j, f in enumerate(FEEDS):
        for i, m in enumerate(models):
            for ad, mk, dx in (("base", "o", -w / 4), ("retime", "D", w / 4)):
                r = df[(df.model == m) & (df.frames == f) & (df.adapter == ad)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + (j - 2.5) * w + dx
                ax.errorbar(x, r.rfs, yerr=[[r.rfs - r.rfs_lo], [r.rfs_hi - r.rfs]], fmt=mk, ms=3, elinewidth=.6,
                            color=FEED_COLOR[f], mfc=FEED_COLOR[f] if ad == "base" else "white",
                            label=f"{FEED_LABEL[f]}" if (i == 0 and ad == "base") else None)
    ax.axhline(7.10, color=S.BASELINE, lw=.6, ls="--")
    ax.text(len(models) - .5, 7.12, "constant velocity", ha="right", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(range(len(models)), [m.capitalize() for m in models])
    ax.set_ylabel("RFS (cluster mean)")
    S.bars(ax)
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", color="#444444", ms=3), plt.Line2D([], [], marker="D", ls="", color="#444444", mfc="white", ms=3)]
    lab += ["plan as is", "re-timed to measured speed"]
    ax.legend(h, lab, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    S.panel(ax, "(a)")
    c = df[(df.model == "cinque") & (df.adapter == "base")].set_index("frames")
    ps = img[img.view == "road"].groupby("method")["psnr_mean"].mean()
    for f in FEEDS[:-1]:
        if f in c.index and f in ps.index:
            bx.scatter(ps[f], 100 * c.loc[f, "recovered"], color=FEED_COLOR[f], s=14, zorder=3)
            bx.annotate(FEED_LABEL[f], (ps[f], 100 * c.loc[f, "recovered"]), textcoords="offset points", xytext=(3, -8), fontsize=6.5)
    bx.set_xlabel("PSNR of synthesized frames (dB)")
    bx.set_ylabel("RFS gap recovered (%)")
    bx.set_ylim(-5, 105)
    S.panel(bx, "(b)")
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    print(S.save(fig, FIGS / "op-interp-wod"))


def navsim():
    df = pd.read_csv(RES / "nav_results.csv")
    S.apply()
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN, 2.6))
    rows = df[df.name.str.contains("cinque") & ~df.name.str.startswith("exam")].sort_values("pdms")
    refs = df[df.name.str.startswith("exam")]
    y = np.arange(len(rows))
    col = [FEED_COLOR.get(n.split("-")[0].split("_")[0], S.PALETTE["purple"]) for n in rows.name]
    ax.errorbar(rows.pdms, y, xerr=[rows.pdms - rows.lo, rows.hi - rows.pdms], fmt="none", ecolor="#999999", elinewidth=.6)
    ax.scatter(rows.pdms, y, c=col, s=12, zorder=3)
    ax.set_yticks(y, [n.replace("-cinque__", " + ") for n in rows.name], fontsize=6.5)
    for r in refs.itertuples():
        ax.axvline(r.pdms, color="#BBBBBB", lw=.5, ls="--", zorder=0)
        ax.text(r.pdms, len(rows) - .4, r.name.replace("exam ", ""), rotation=90, fontsize=5.5, va="top", ha="right", color="#666666")
    ax.set_xlabel("PDMS (navtest subset)")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.38)
    print(S.save(fig, FIGS / "op-interp-navsim"))


if __name__ == "__main__":
    wod()
    if (RES / "nav_results.csv").exists():
        navsim()
