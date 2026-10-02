"""Figure for experiments/hugsim/results/controller_spin.md: what the plan, the steering and the heading do step by step in the
PR #57 Cinque runs on scene-0013-medium-00 (page case hugsim-03), and, if given, the closed-loop reruns of the same scenario
under fixed / fixed2 / ideal controllers.
    python spin_case_fig.py <zs_steps.jsonl of the exam run> <out.png> [name=<zs_steps.jsonl> ...]
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(p):
    L = [json.loads(x) for x in open(p)][1:]
    t = np.array([r["t"] for r in L])
    th = np.degrees([r["theta"] for r in L])
    phi = np.degrees([np.arctan2(r["plan"][1][0], r["plan"][1][1]) for r in L])
    return t, th, phi, np.array([r["steer"] for r in L]), np.array([r["v"] for r in L])


def main(src, out, *extra):
    runs = {"exam run (PR #57)": src, **dict(e.split("=", 1) for e in extra)}
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for name, p in runs.items():
        t, th, phi, st, v = load(p)
        ax[0, 0].plot(t, th, label=name)
        ax[0, 1].plot(t, phi)
        ax[1, 0].plot(t, st)
        ax[1, 1].plot(t, v)
    for a, ttl in zip(ax.ravel(), ("heading (deg, - = left)", "plan direction at 1 s relative to the ego heading (deg)",
                                   "steering angle (rad)", "speed (m/s)")):
        a.set_title(ttl, fontsize=10)
        a.grid(alpha=.3)
        a.set_xlim(0, 8)
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main(*sys.argv[1:])
