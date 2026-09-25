#!/usr/bin/env python
"""Figures of the HUGSIM controller acceptance (todos/2026-09-25-closed-loop-infra-acceptance/hugsim-controllers.md),
from the per-step tables scripts/hugsim/preset_eval.py writes (pulled to research/results/infra-acceptance/hugsim/).

    .venv/bin/python scripts/hugsim/preset_figs.py
  hugsim-ctrl-crosstrack   signed cross-track to the logged path vs time, the 9 static acceptance scenes
  hugsim-ctrl-ecdf         ECDF of |lateral error| at 0.5 s vs the fed plan, acceptance and held-out static scenes
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from jevdrive import plots as P  # noqa: E402

RES = REPO / "research" / "results" / "infra-acceptance" / "hugsim"
FIGS = REPO / "research" / "figs"
TODO = REPO / "todos" / "2026-09-25-closed-loop-infra-acceptance"
CTRL = {"ideal": ("ideal (reference)", P.COLOR["baseline"]), "official": ("official", P.OKABE_ITO[6]),
        "fixed": ("fixed (PR #57)", P.OKABE_ITO[5]), "fixed2": ("fixed2 (0.25 s iLQR)", P.OKABE_ITO[3])}


DS = {"nuscenes": "nuScenes", "waymo": "Waymo", "kitti360": "KITTI-360", "pandaset": "PandaSet"}


def static(name, label=False):
    rows = (TODO / name).read_text().split()
    if label:
        return {Path(l).stem: f"{DS[Path(l).parent.name]} {Path(l).stem.replace('scene-', '').replace('-00', '')}" for l in rows}
    return [Path(l).stem for l in rows]


def crosstrack(steps):
    scen = static("hugsim-static.txt", label=True)
    fig, axs = plt.subplots(3, 3, figsize=(P.PAGE, 4.2), sharex=False)
    for ax, sc in zip(axs.flat, scen):
        for c, (lab, col) in CTRL.items():
            d = steps[(steps.scenario == sc) & (steps.controller == c)]
            if len(d):
                ax.plot(d.t, d.log_xt, color=col, label=lab, lw=0.9 if c != "ideal" else 1.4)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(scen[sc], pad=2)
        ax.grid(True)
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    for ax in axs[:, 0]:
        ax.set_ylabel("cross-track [m]")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    P.legend_below(fig, axs[-1, 1])
    P.save(fig, FIGS, "hugsim-ctrl-crosstrack")


def ecdf(sets):
    fig, axs = plt.subplots(1, len(sets), figsize=(P.PAGE, 2.0), sharey=True)
    for ax, (title, steps, lst) in zip(np.atleast_1d(axs), sets):
        scen = static(lst)
        for c, (lab, col) in CTRL.items():
            x = np.sort(np.abs(steps[(steps.controller == c) & steps.scenario.isin(scen)].lat50.dropna()))
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=col, label=lab)
        ax.axvline(0.30, color="k", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 5)
        ax.set_xlabel("|lateral error| at 0.5 s [m]")
        ax.set_title(title, pad=2)
        ax.grid(True)
    np.atleast_1d(axs)[0].set_ylabel("fraction of steps")
    fig.tight_layout()
    P.legend_below(fig, np.atleast_1d(axs)[-1])
    P.save(fig, FIGS, "hugsim-ctrl-ecdf")


def main():
    mpl.rcParams.update(P.STYLE)
    acc = pd.read_csv(RES / "accept_steps.csv")
    crosstrack(acc)
    sets = [("acceptance set (9 static scenes)", acc, "hugsim-static.txt")]
    if (RES / "val_steps.csv").exists():
        sets.append(("held-out set (8 static scenes)", pd.read_csv(RES / "val_steps.csv"), "hugsim-val-static.txt"))
    ecdf(sets)


if __name__ == "__main__":
    main()
