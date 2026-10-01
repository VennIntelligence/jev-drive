#!/usr/bin/env python
"""Figures of research/openpilot-openloop-integration.md from experiments/op_openloop/results/op-interp/*.csv (Mac, project venv).

  op-interp-wod     WOD-E2E rater frames: openpilot RFS per 2 Hz feed (hold / blend / warp / RIFE / GIMM / real 10 Hz),
                    plan as is vs re-timed to the measured speed; and synthesized-frame PSNR vs the RFS gap recovered
  op-interp-navsim  navtest subset: PDMS of the Cinque ladder (feed x adapter) with references
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("research",)]
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "experiments/op_openloop/results/op-interp", REPO / "research/figs"
FEEDS = ["hold", "blend", "warp", "rife", "gimm", "real", "pre"]
FEED_LABEL = {"hold": "hold (exam)", "blend": "cross-fade", "warp": "ego-motion warp", "rife": "RIFE 4.26",
              "gimm": "GIMM-VFI", "real": "real 10 Hz", "pre": "warp + pre-roll"}
FEED_COLOR = {"hold": S.BASELINE, "blend": S.PALETTE["yellow"], "warp": S.PALETTE["green"], "rife": S.PALETTE["sky_blue"],
              "gimm": S.PALETTE["blue"], "real": S.PALETTE["black"], "pre": S.PALETTE["purple"]}
PRE = {"cinque": "warp_pre1.5", "lebowski": "warp_pre3.3"}      # pre-roll to fill each model's context (1.6 / 4.8 s)


def wod():
    df = pd.read_csv(RES / "wod_results.csv")
    df = df[~df["frames"].str.startswith("exam")]
    img = pd.read_csv(RES / "wod_img.csv")
    S.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.55), gridspec_kw={"width_ratios": [2.3, 1]})
    models = [m for m in ("small", "cinque", "lebowski") if (df.model == m).any()]
    w = 0.12
    for j, f in enumerate(FEEDS):
        for i, m in enumerate(models):
            for ad, mk, dx in (("base", "o", -w / 4), ("retime", "D", w / 4)):
                r = df[(df.model == m) & (df.frames == (PRE.get(m, "-") if f == "pre" else f)) & (df.adapter == ad)]
                if not len(r):
                    continue
                r = r.iloc[0]
                x = i + (j - 3) * w + dx
                ax.errorbar(x, r.rfs, yerr=[[r.rfs - r.rfs_lo], [r.rfs_hi - r.rfs]], fmt=mk, ms=3, elinewidth=.6,
                            color=FEED_COLOR[f], mfc=FEED_COLOR[f] if ad == "base" else "white",
                            label=f"{FEED_LABEL[f]}" if (m == "cinque" and ad == "base") else None)
    ax.axhline(7.10, color=S.BASELINE, lw=.6, ls="--")
    ax.text(1.42, 7.04, "constant velocity", ha="center", va="top", fontsize=6.5, color="#555555")
    ax.set_xticks(range(len(models)), [m.capitalize() for m in models])
    ax.set_ylabel("RFS (cluster mean)")
    S.bars(ax)
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", color="#444444", ms=3), plt.Line2D([], [], marker="D", ls="", color="#444444", mfc="white", ms=3)]
    lab += ["plan as is", "re-timed to measured speed"]
    fig.legend(h, lab, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=7)
    S.panel(ax, "(a)")
    c = df[(df.model == "cinque") & (df.adapter == "base")].set_index("frames")
    ps = img[img.view == "road"].groupby("method")["psnr_mean"].mean()
    off = {"hold": (4, 2), "blend": (4, -3), "warp": (-8, 5), "rife": (-4, -10), "gimm": (-30, 6)}
    for f in ("hold", "blend", "warp", "rife", "gimm"):
        if f in c.index and f in ps.index:
            bx.scatter(ps[f], 100 * c.loc[f, "recovered"], color=FEED_COLOR[f], s=14, zorder=3)
            bx.annotate(FEED_LABEL[f], (ps[f], 100 * c.loc[f, "recovered"]), textcoords="offset points", xytext=off[f], fontsize=6.5)
    bx.set_xlim(20, 27)
    bx.set_xlabel("PSNR of synthesized frames (dB)")
    bx.set_ylabel("RFS gap recovered (%)")
    bx.set_ylim(-5, 105)
    S.panel(bx, "(b)")
    fig.subplots_adjust(bottom=0.36, wspace=0.3)
    print(S.save(fig, FIGS / "op-interp-wod"))


NAV_ROWS = [  # (pose file, label, colour key), bottom to top
    ("exam cv", "constant velocity", "ref"), ("hold-small__base", "small: hold (exam feed)", "hold"),
    ("hold-cinque__base", "Cinque: hold (exam feed)", "hold"), ("hold-lebowski__base", "Lebowski: hold (exam feed)", "hold"),
    ("blend-cinque__base", "Cinque: cross-fade", "blend"), ("exam heads_cls_ego_K1024", "blind head (ego only)", "ref"),
    ("hold-cinque__retime", "Cinque: hold + retime", "hold"), ("exam heads_cls_late_cinque_temporal", "frozen temporal + cls_late head", "ref"),
    ("warp-lebowski__base", "Lebowski: warp", "warp"), ("warp_pre3.3-lebowski__base", "Lebowski: warp + 3.3 s pre-roll", "pre"),
    ("rife-small__base", "small: RIFE", "rife"), ("rife-cinque__retime", "Cinque: RIFE + retime", "rife"),
    ("rife-cinque__base", "Cinque: RIFE", "rife"), ("warp-cinque__base", "Cinque: ego-motion warp", "warp"),
    ("gimm_g0.2-cinque__retime", "Cinque: GIMM-VFI + retime", "gimm"), ("gimm_g0.2-cinque__base", "Cinque: GIMM-VFI", "gimm"),
    ("exam human", "human (log)", "ref")]


def navsim():
    df = pd.read_csv(RES / "nav_results.csv").set_index("name")
    S.apply()
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN, 3.0))
    rows = [r for r in NAV_ROWS if r[0] in df.index]
    for y, (k, lab, c) in enumerate(rows):
        r = df.loc[k]
        col = S.BASELINE if c == "ref" else FEED_COLOR[c]
        ax.errorbar(r.pdms, y, xerr=[[r.pdms - r.lo], [r.hi - r.pdms]], fmt="s" if c == "ref" else "o", ms=3, elinewidth=.6,
                    color=col, mfc="white" if c == "ref" else col)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=6.5)
    for v, name in ((84.0, "TransFuser"), (88.1, "DiffusionDrive")):
        ax.axvline(v, color="#BBBBBB", lw=.5, ls="--", zorder=0)
        ax.text(v, -0.9, name, rotation=90, fontsize=5.5, va="bottom", ha="right", color="#777777")
    ax.set_xlabel("PDMS (navtest, 2,000-token subset)")
    ax.set_ylim(-1, len(rows))
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.47, right=0.98, bottom=0.14, top=0.98)
    print(S.save(fig, FIGS / "op-interp-navsim"))


if __name__ == "__main__":
    wod()
    if (RES / "nav_results.csv").exists():
        navsim()
