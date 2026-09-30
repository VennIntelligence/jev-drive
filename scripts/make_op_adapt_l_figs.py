"""Figures of todos/2026-10-01-op-adapt-L-prereg.md from research/results/op-adapt-L/metrics_all.csv (one row per model x metric).

  op-adapt-L-capture.png    capture-rate change on WOD val (start / stop / turn onset), per model, 95% segment-cluster CI
  op-adapt-L-triggers.png   false-trigger change on the contrast sets, drift on non-imitated frames, slow / fast change
  op-adapt-L-rfs.png        RFS change on the 479 rater frames (raw and x1.06) and the nuScenes val capture change
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "research/results/op-adapt-L", REPO / "research/figs"
P = S.PALETTE
FAMILY = [("main", P["blue"]), ("noint", P["vermillion"]), ("only_", P["green"]), ("dw", P["orange"]), ("nocontrast", P["purple"]),
          ("long", P["sky_blue"]), ("tr_", P["black"]), ("sel_", "#888888")]


def colour(model: str) -> str:
    for k, c in FAMILY:
        if model.startswith(k):
            return c
    return "#888888"


def order(models):
    fam = [k for k, _ in FAMILY]
    return sorted(models, key=lambda m: (next((i for i, k in enumerate(fam) if m.startswith(k)), 99), m))


def get(df, model, st, sl, metric, xs=1.0):
    r = df[(df.model == model) & (df.set == st) & (df.slice == sl) & (df.metric == metric) & (df.xscale == xs)]
    return r.iloc[0] if len(r) else None


def dots(ax, df, models, st, sl, metric, scale=1.0, xs=1.0, horizontal=True):
    for i, m in enumerate(models):
        r = get(df, m, st, sl, metric, xs)
        if r is None:
            continue
        ax.errorbar(r.delta * scale, i, xerr=[[(r.delta - r.lo) * scale], [(r.hi - r.delta) * scale]], fmt="o", ms=3, elinewidth=.7,
                    color=colour(m))
    ax.set_yticks(range(len(models)), models, fontsize=6)
    ax.invert_yaxis()
    S.zero_line(ax)


def main():
    df = pd.read_csv(RES / "metrics_all.csv")
    models = order(sorted(df.model.unique()))
    S.apply()
    h = max(2.6, 0.13 * len(models) + 0.9)
    # capture
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, h), sharey=True)
    for ax, (sl, m, t) in zip(axs, (("start", "cap_start", "start from stop"), ("stop", "cap_stop", "stop onset"),
                                    ("turn_onset", "cap_turn_onset", "turn onset"))):
        dots(ax, df, models, "wodval", sl, m, 100)
        ax.set_xlabel("capture-rate change vs original (pp)")
        ax.set_title(t, fontsize=8)
    axs[0].set_ylabel("")
    S.save(fig, FIGS / "op-adapt-L-capture")
    plt.close(fig)
    # triggers and drift
    fig, axs = plt.subplots(1, 4, figsize=(S.DOUBLE_COLUMN_IN, h), sharey=True)
    for ax, (sl, m, t) in zip(axs[:3], (("stay", "false_start", "false start on stay"), ("control", "false_stop", "false stop on control"),
                                        ("control", "false_turn", "false turn on control"))):
        dots(ax, df, models, "wodval", sl, m, 100)
        ax.axvline(2.0, color=S.BASELINE, lw=.6, ls=":")
        ax.set_xlabel("change vs original (pp)")
        ax.set_title(t, fontsize=8)
    for i, mo in enumerate(models):
        r = get(df, mo, "wodval", "other", "drift_median")
        if r is not None:
            axs[3].plot(r.adapt, i, "o", ms=3, color=colour(mo))
    axs[3].axvline(0.10, color=S.BASELINE, lw=.6, ls=":")
    axs[3].set_xlabel("plan drift, median (m)")
    axs[3].set_title("drift on other frames", fontsize=8)
    S.save(fig, FIGS / "op-adapt-L-triggers")
    plt.close(fig)
    # rfs and nuScenes
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, h), sharey=True)
    for ax, (xs, t) in zip(axs[:2], ((1.0, "RFS change, raw plan"), (1.06, "RFS change, longitudinal x1.06"))):
        dots(ax, df, models, "rater", "all", "rfs", 1.0, xs)
        ax.axvline(-0.10, color=S.BASELINE, lw=.6, ls=":")
        ax.set_xlabel("RFS change vs original")
        ax.set_title(t, fontsize=8)
    for k, (sl, m, mk) in enumerate((("start", "cap_start", "o"), ("stop", "cap_stop", "s"), ("turn_onset", "cap_turn_onset", "^"))):
        for i, mo in enumerate(models):
            r = get(df, mo, "nusval", sl, m)
            if r is not None:
                axs[2].errorbar(100 * r.delta, i + (k - 1) * 0.2, xerr=[[100 * (r.delta - r.lo)], [100 * (r.hi - r.delta)]], fmt=mk, ms=2.5,
                                elinewidth=.6, color=colour(mo))
    S.zero_line(axs[2])
    axs[2].set_xlabel("nuScenes val capture change (pp)\n(o start, s stop, ^ turn onset)")
    axs[2].set_title("out-of-dataset transfer", fontsize=8)
    S.save(fig, FIGS / "op-adapt-L-rfs")
    plt.close(fig)


if __name__ == "__main__":
    main()
