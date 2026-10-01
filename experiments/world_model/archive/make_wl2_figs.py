#!/usr/bin/env python
"""Figures of research/wl2-results.md from experiments/world_model/results/wl2/results/*.csv (Mac, project venv).

  wl2-c1c        C1c AUC on the occupied subset per arm, and the brake-vs-hold direction
  wl2-c2-c3      C2 AUC (cg) and C3a H per arm
  wl2-slowshift  executed offset of the shift-type candidates, and the predicted 2 s e_y against the executed one
  wl2-c4         continuous C4 examinee: flip rate against the out-of-sample null
  wl2-curves     inner-val loss of every arm
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("research",)]
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

RES, FIGS = REPO / "experiments/world_model/results/wl2/results", REPO / "research/figs"
P = S.PALETTE
COLOR = {"vrep": S.BASELINE, "A": P["sky_blue"], "B": P["blue"], "W": P["vermillion"], "Bs": P["green"], "T": P["purple"],
         "Bh": P["orange"], "Bhc": P["yellow"], "Ax": P["sky_blue"], "Bx": P["blue"]}
NAME = {"vrep": "v1-rep", "A": "A", "B": "B", "W": "worig", "Bs": "B-same", "T": "T", "Bh": "B-holdout", "Bhc": "B-holdout-combo", "Ax": "A (xfit)", "Bx": "B (xfit)"}


def ci(x):
    return ast.literal_eval(x) if isinstance(x, str) else [np.nan, np.nan]


def errbar(ax, xs, val, cis, colors):
    lo = np.array([v - c[0] for v, c in zip(val, cis)])
    hi = np.array([c[1] - v for v, c in zip(val, cis)])
    ax.bar(xs, val, color=colors, width=0.62, zorder=2)
    ax.errorbar(xs, val, yerr=[lo, hi], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=2, zorder=3)


def c1c():
    d = pd.read_csv(RES / "c1_main.csv")
    S.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.3))
    x = np.arange(len(d))
    cols = [COLOR[k] for k in d.arm]
    errbar(a, x, d.auc_S, d.auc_S_ci.map(ci), cols)
    a.axhline(0.75, color="#222222", lw=0.6, ls="--")
    a.axhline(0.5, color="#999999", lw=0.5)
    a.set_ylim(0.3, 0.95)
    a.set_ylabel("AUC, still occupied vs left")
    b.set_ylabel("brake > hold at 2 s")
    errbar(b, x, d.brake_gt_hold, d.brake_gt_hold_ci.map(ci), cols)
    b.axhline(0.85, color="#222222", lw=0.6, ls="--")
    b.set_ylim(0, 1.05)
    for ax in (a, b):
        ax.set_xticks(x, [NAME[k] for k in d.arm], rotation=30, ha="right")
        S.bars(ax)
    S.panel(a, "(a)")
    S.panel(b, "(b)")
    fig.tight_layout()
    S.save(fig, FIGS / "wl2-c1c")


def c2c3():
    d = pd.read_csv(RES / "c2_c3.csv")
    d = d[(d.set == "main") & (d.label == "cg")]
    S.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.3))
    x = np.arange(len(d))
    cols = [COLOR[k] for k in d.arm]
    errbar(a, x, d.auc_learn, d.auc_learn_ci.map(ci), cols)
    a.axhline(0.85, color="#222222", lw=0.6, ls="--")
    a.set_ylim(0.6, 1.0)
    a.set_ylabel("C-learn AUC (cg)")
    errbar(b, x, d.H, d.H_ci.map(ci), cols)
    b.axhline(0.5, color="#222222", lw=0.6, ls="--")
    b.axhline(0.25, color="#222222", lw=0.6, ls=":")
    b.set_ylim(-0.1, 1.0)
    b.set_ylabel(r"$H = (U_{op}-U_{sel})/(U_{op}-U_{or})$")
    for ax in (a, b):
        ax.set_xticks(x, [NAME[k] for k in d.arm], rotation=30, ha="right")
        S.bars(ax)
    S.panel(a, "(a)")
    S.panel(b, "(b)")
    fig.tight_layout()
    S.save(fig, FIGS / "wl2-c2-c3")


def slowshift():
    s = pd.read_csv(RES / "slow_shift.csv")
    o = s[s.section == "offset"]
    e = pd.read_csv(RES / "ego_error_main.csv")
    S.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})
    acts = ["nudge_L", "shift_L_slow", "shift_R_slow", "shift_L", "shift_R"]
    cmd = {"nudge_L": 1.5, "shift_L_slow": 3.0, "shift_R_slow": 3.0, "shift_L": 3.0, "shift_R": 3.0}
    x = np.arange(len(acts))
    oo = o.set_index("action").loc[acts]
    a.bar(x, oo.median_m, color=P["blue"], width=0.6, zorder=2, label="executed median")
    a.errorbar(x, oo.median_m, yerr=[oo.median_m - oo.q25_m, oo.q75_m - oo.median_m], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=2, zorder=3)
    a.scatter(x, [cmd[k] for k in acts], marker="_", s=180, color=P["vermillion"], zorder=4, label="commanded")
    a.set_xticks(x, [k.replace("_", r"\_") for k in acts], rotation=30, ha="right")
    a.set_ylabel("lateral offset from op path at 3 s (m)")
    a.legend(loc="upper left")
    S.bars(a)
    S.panel(a, "(a)")
    arms = [k for k in ("B", "Bs", "T") if k in set(e.arm)]
    w = 0.24
    for i, k in enumerate(arms):
        ee = e[e.arm == k].set_index("action").loc[acts]
        b.bar(x + (i - (len(arms) - 1) / 2) * w, ee.med_abs_err_m, width=w, color=COLOR[k], label=NAME[k], zorder=2)
    pe = e[e.arm == arms[0]].set_index("action").loc[acts]
    b.scatter(x, pe.persistence_med_abs_err_m, marker="_", s=150, color="#222222", zorder=4, label="hold current $e_y$")
    b.set_xticks(x, [k.replace("_", r"\_") for k in acts], rotation=30, ha="right")
    b.set_ylabel(r"median $|\hat e_y - e_y|$ at 2 s (m)")
    b.legend(loc="upper left", ncol=2)
    S.bars(b)
    S.panel(b, "(b)")
    fig.tight_layout()
    S.save(fig, FIGS / "wl2-slowshift")


def c4():
    d = pd.read_csv(RES / "c4_flips.csv")
    d = d[d.examinee.str.contains("risk mean")]
    S.apply()
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.2), sharey=True)
    for ax, cls, t in zip(axs, ("ped", "cutin"), ("pedestrian", "cut-in")):
        x = d[d.cls == cls].reset_index(drop=True)
        xs = np.arange(len(x))
        col = [COLOR.get(e.split()[0], S.BASELINE) if "Q risk" not in e else S.BASELINE for e in x.examinee]
        ax.bar(xs, x.flip_rate, color=col, width=0.6, zorder=2)
        ax.errorbar(xs, x.flip_rate, yerr=[x.flip_rate - x.flip_lo, x.flip_hi - x.flip_rate], fmt="none", ecolor="#222222", elinewidth=0.7, capsize=2, zorder=3)
        ax.scatter(xs, x.false_flip_null_oos + 0.10, marker="_", s=160, color=P["vermillion"], zorder=4, label="null + 10 pp")
        ax.set_xticks(xs, [e.replace(" risk mean", "").replace("Q", "Q (model-free)") for e in x.examinee], rotation=30, ha="right")
        S.bars(ax)
        ax.set_xlabel(t)
    axs[0].set_ylabel("flip rate, obs pairs")
    axs[0].legend(loc="upper right")
    fig.tight_layout()
    S.save(fig, FIGS / "wl2-c4")


def curves():
    d = pd.read_csv(RES / "curves.csv")
    S.apply()
    fig, ax = plt.subplots(figsize=(S.SINGLE_COLUMN_IN, 2.3))
    for arm, g in d.groupby("arm"):
        m = g.groupby("step").val.mean()
        ax.plot(m.index, m.values, color=COLOR[arm], label=NAME[arm])
    ax.set_xlabel("step")
    ax.set_ylabel("inner-val loss (seed mean)")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    S.save(fig, FIGS / "wl2-curves")


if __name__ == "__main__":
    for f in (c1c, c2c3, slowshift, c4, curves):
        f()
