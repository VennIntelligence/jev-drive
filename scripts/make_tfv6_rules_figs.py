#!/usr/bin/env python
"""Figures for todos/2026-09-25-tfv6-rules-interface/README.md from research/results/tfv6-rules-interface/*.csv.

  b2d-family-sr        SR per hazard family, top public entries and our TFv6 A1, route-bootstrap 95% CI
  b2d-family-gaps      paired DS / SR gaps between adjacent entries against the run-to-run noise band
  tfv6-rules-pairs     closed-loop pairs: reaction rate, null false rate and hazard collisions per arm,
                       and the reaction statistic S on x- against the weather null
PNG to research/figs (committed), PDF next to it (not committed).
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.plots import COL, OKABE_ITO, PAGE, STYLE, save  # noqa: E402
from jevdrive.tfv6_rules import FAMILY, SUDDEN  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "research" / "results" / "tfv6-rules-interface"
FIGS = Path(__file__).resolve().parents[1] / "research" / "figs"
LABEL = {"TFv6 A1": "TFv6 (ours, A1)", "tfv6.user": "TFv6 (user rerun)", "blue.r1": "BLUE", "sparsedrivev2":
         "SparseDriveV2", "simlingo.user": "SimLingo (user rerun)", "r2se": "R2SE", "tfpp": "TF++",
         "minddrive3b": "MindDrive-3B", "orion_lite.a100": "Orion-Lite"}
COLOR = {"TFv6 A1": "#D55E00", "tfv6.user": "#E69F00", "blue.r1": "#0072B2", "sparsedrivev2": "#009E73",
         "simlingo.user": "#56B4E9", "r2se": "#CC79A7", "tfpp": "#7F7F7F", "minddrive3b": "#000000",
         "orion_lite.a100": "#F0E442"}
FAM_LABEL = {"all": "all (B2D-209)", "sudden": "sudden hazards", "vru_emerging": "VRU emerging",
             "vru_crossing": "VRU crossing", "cut_in": "cut-in", "lead_hard_brake": "lead hard brake",
             "junction_violator": "junction violator", "unprotected_turn": "unprotected turn",
             "merge_lane_change": "merge / lane change", "obstacle_bypass": "obstacle bypass",
             "emergency_vehicle": "emergency vehicle", "routine_control": "routine control"}
ARM_COLOR = {"A1": "#D55E00", "A0": "#E69F00", "B1": "#0072B2", "B0": "#56B4E9"}


def family_sr():
    fam = pd.read_csv(RES / "family.csv")
    runs = [r for r in LABEL if r in set(fam.run)]
    groups = ["all", "sudden", *SUDDEN, *[f for f in FAMILY if f not in SUDDEN]]
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(3, 4, figsize=(PAGE, 4.6), sharex=True)
        for ax, g in zip(axes.flat, groups):
            for i, r in enumerate(runs):
                row = fam[(fam.run == r) & (fam.family == g)]
                if not len(row):
                    continue
                row = row.iloc[0]
                y = len(runs) - 1 - i
                ax.errorbar(row.SR, y, xerr=[[row.SR - row.SR_lo], [row.SR_hi - row.SR]], fmt="o",
                            color=COLOR[r], ms=3, elinewidth=0.8, capsize=0, label=LABEL[r])
            ax.set_title(f"{FAM_LABEL[g]} (n = {int(fam[(fam.family == g)].n.max())})", fontsize=7)
            ax.set_yticks([])
            ax.set_xlim(-2, 102)
            ax.grid(True, axis="x")
        for ax in axes[-1]:
            ax.set_xlabel("success rate (%)")
        h, l = axes.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, handlelength=1.0)
        save(fig, FIGS, "b2d-family-sr")


def family_gaps():
    cmp_ = pd.read_csv(RES / "compare.csv")
    order = [r for r in LABEL if r in set(cmp_.a) | set(cmp_.b)]
    adj = [(a, b) for a, b in zip(order, order[1:])]
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.3), sharey=True)
        for ax, (m, lab) in zip(axes, (("DS", "paired DS gap (points)"), ("SR", "paired SR gap (points)"))):
            for j, g in enumerate(("all", "sudden")):
                for i, (a, b) in enumerate(adj):
                    row = cmp_[(cmp_.a == a) & (cmp_.b == b) & (cmp_.group == g)]
                    if not len(row):
                        continue
                    row = row.iloc[0]
                    y = len(adj) - 1 - i + (0.18 if g == "all" else -0.18)
                    ax.errorbar(row["d" + m], y, xerr=[[row["d" + m] - row[f"d{m}_lo"]], [row[f"d{m}_hi"] - row["d" + m]]],
                                fmt="o" if g == "all" else "s", color="#000000" if g == "all" else "#D55E00", ms=3,
                                elinewidth=0.8, label=("all routes" if g == "all" else "sudden-hazard routes") if i == 0 else None)
            if m == "DS":
                bar = float(cmp_.noise_bar.iloc[0])
                ax.axvspan(-bar, bar, color="#7F7F7F", alpha=0.2, lw=0, label="run-to-run 95% band")
            ax.axvline(0, color="#7F7F7F", lw=0.6)
            ax.set_xlabel(lab)
            ax.grid(True, axis="x")
        axes[0].set_yticks(range(len(adj)))
        axes[0].set_yticklabels([f"{LABEL[a]} $-$ {LABEL[b]}" for a, b in adj][::-1])
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
        save(fig, FIGS, "b2d-family-gaps")


def rules_pairs():
    if not (RES / "pair_summary.csv").exists():
        return
    s = pd.read_csv(RES / "pair_summary.csv").set_index("arm").reindex(["A1", "A0", "B1", "B0"]).dropna(how="all")
    pr = pd.read_csv(RES / "pairs.csv")
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.2), gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axes[0]
        x = np.arange(len(s))
        for k, (col, lab, off) in enumerate((("RR", "reaction rate (x$^-$)", -0.25), ("FR_oos", "null false rate", 0.0),
                                              ("HC", "hazard collision (x$^+$)", 0.25))):
            y = s[col].to_numpy() * 100
            err = None
            if col + "_lo" in s:
                err = [y - s[col + "_lo"].to_numpy() * 100, s[col + "_hi"].to_numpy() * 100 - y]
            ax.bar(x + off, y, 0.25, color=OKABE_ITO[[5, 7, 6][k]], label=lab, yerr=err, error_kw={"lw": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(s.index)
        ax.set_ylabel("share of pairs (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=6)
        ax = axes[1]
        for i, arm in enumerate(s.index):
            for vs, mk, off in (("minus", "o", -0.15), ("null", "x", 0.15)):
                v = pr[(pr.arm == arm) & (pr.vs == vs)].S.dropna()
                ax.scatter(np.full(len(v), i + off) + np.random.default_rng(0).uniform(-0.06, 0.06, len(v)), v,
                           s=5, marker=mk, color=ARM_COLOR[arm], lw=0.6,
                           label=("x$^-$ (hazard hidden)" if vs == "minus" else "weather null") if i == 0 else None)
            ax.hlines(-s.loc[arm, "tau"], i - 0.3, i + 0.3, color="#000000", lw=0.8,
                      label=r"$-\tau_{arm}$" if i == 0 else None)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s.index)
        ax.set_ylabel(r"$S=\min_t\,[v^+(t)-v(t)]$ (m/s)")
        ax.axhline(0, color="#7F7F7F", lw=0.5)
        ax.legend(loc="lower right", fontsize=6)
        save(fig, FIGS, "tfv6-rules-pairs")


if __name__ == "__main__":
    family_sr()
    family_gaps()
    rules_pairs()
