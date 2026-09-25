#!/usr/bin/env python3
"""Figures of the Alpamayo Bench2Drive stall diagnosis (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md).

Inputs: research/results/zeroshot-b2d/alp-stall/ (routes.csv, episodes.csv, standstill*.json, clamp_replay.json, all written
by scripts/zeroshot_b2d_alp_stall.py and the box one-offs recorded in the doc) and the 1833 route logs (ticks/plans.jsonl)
pulled from the box into --route-dir. Mac: .venv/bin/python scripts/zeroshot_b2d_alp_stall_figs.py --route-dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as ps  # noqa: E402

RES = ROOT / "research/results/zeroshot-b2d/alp-stall"
FIGS = ROOT / "research/figs"
C_MOVE, C_PIN, C_FREE = ps.PALETTE["sky_blue"], ps.PALETTE["vermillion"], ps.PALETTE["orange"]
C_FIXED, C_ZOO, C_CLAMP = ps.BASELINE, ps.PALETTE["vermillion"], ps.PALETTE["blue"]


def fig_overview():
    r = pd.read_csv(RES / "routes.csv")
    r = r[r.run == "full"].copy()
    r["route"] = r.route.astype(str)
    e = pd.read_csv(RES / "episodes.csv")
    e["route"] = e.route.astype(str)
    pinned = e[e.ncol_in > 0].groupby("route").dur.sum()
    r["pinned_s"] = r.route.map(pinned).fillna(0.0)
    r["other_s"] = (r.stall_s - r.pinned_s).clip(lower=0)
    r["move_s"] = r.sim_s - r.stall_s
    r = r.sort_values(["pinned_s", "stall_s"]).reset_index(drop=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.35), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(r))
    a.bar(x, r.move_s, width=1.0, color=C_MOVE, label="moving ($v\\geq0.5$ m/s)", linewidth=0)
    a.bar(x, r.other_s, width=1.0, bottom=r.move_s, color=C_FREE, label="stalled, no collision", linewidth=0)
    a.bar(x, r.pinned_s, width=1.0, bottom=r.move_s + r.other_s, color=C_PIN,
          label="stalled after a collision ($\\geq$10 s)", linewidth=0)
    a.axhline(200, color="#555555", linewidth=.6, linestyle="--")
    a.text(1, 203, "4000-tick cap (200 s)", fontsize=7, va="bottom", color="#555555")
    a.set_xlim(-0.5, len(r) - 0.5)
    a.set_ylim(0, 225)
    a.set_xlabel("route (n = 117, sorted)")
    a.set_ylabel("simulated time (s)")
    a.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.0, 0.93))
    ps.bars(a)
    ps.panel(a, "(a)")

    fixed = json.loads((RES / "standstill-free-smoke-alpamayo.json").read_text())
    zoo = json.loads((RES / "standstill-free-full220-alpamayo-zoopid.json").read_text())
    clamp = json.loads((RES / "clamp_replay.json").read_text())["full220-alpamayo-zoopid"]
    kinds = [("stay", "stay"), ("rev", "reverse"), ("go", "go")]
    w = 0.26
    for k, (key, name) in enumerate(kinds):
        vals = [fixed[key]["frac_ticks_throttle"], zoo[key]["frac_ticks_throttle"], 1 - clamp[key]["brake_clamped"]]
        for j, (v, c) in enumerate(zip(vals, (C_FIXED, C_ZOO, C_CLAMP))):
            b.bar(k + (j - 1) * w, v, width=w, color=c, linewidth=0)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=lab) for c, lab in ((C_FIXED, "fixed controller (smoke)"), (C_ZOO, "Zoo PID (full run)"),
                                                        (C_CLAMP, "Zoo PID + forward-only (replay)"))]
    b.set_xticks(range(len(kinds)))
    b.set_xticklabels(["stay\n(x@3s<1.2 m)", "reverse\n(x<-0.3 m)", "go\n(x@3s$\\geq$1.2 m)"])
    b.set_ylabel("share of throttle at standstill")
    b.set_ylim(0, 1.45)
    b.set_yticks([0, .25, .5, .75, 1])
    b.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=1)
    ps.bars(b)
    ps.panel(b, "(b)")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.92, wspace=0.28)
    return ps.save(fig, FIGS / "alp-b2d-stall-overview")


def fig_route(route_dir, t_col=24.2, t_lo=12.0, t_hi=40.0):
    ticks = [json.loads(line) for line in open(route_dir / "ticks.jsonl")]
    plans = [json.loads(line) for line in open(route_dir / "plans.jsonl")]
    t = np.array([q["t"] for q in ticks])
    m = (t >= t_lo) & (t <= t_hi)
    v, thr, br = (np.array([q[k] for q in ticks]) for k in ("v", "throttle", "brake"))
    pt = np.array([p["t"] for p in plans])
    x = np.array([[q[0] for q in p["path"]] for p in plans])
    pm = (pt >= t_lo) & (pt <= t_hi)
    x3, xmin = x[:, 11], x[:, :12].min(1)
    cls = np.where(xmin < -0.3, "rev", np.where(x3 < 1.2, "stay", "go"))

    fig, axs = plt.subplots(3, 1, figsize=(ps.DOUBLE_COLUMN_IN, 3.1), sharex=True)
    axs[0].plot(t[m], v[m], color="#222222")
    axs[0].set_ylabel("speed (m/s)")
    colors = {"stay": C_FREE, "rev": C_PIN, "go": C_MOVE}
    for k, name in (("stay", "stay"), ("rev", "reverse"), ("go", "go")):
        s = pm & (cls == k)
        axs[1].scatter(pt[s], x3[s], s=9, color=colors[k], label=name, zorder=3)
    axs[1].plot(pt[pm], x3[pm], color="#999999", linewidth=.5, zorder=2)
    ps.zero_line(axs[1])
    axs[1].set_ylabel("plan $x$ at 3 s (m)")
    axs[1].set_ylim(-6, 14)
    axs[1].legend(loc="upper left", ncol=3, fontsize=7)
    axs[2].fill_between(t[m], 0, thr[m], step="post", color=ps.PALETTE["green"], alpha=.8, linewidth=0, label="throttle")
    axs[2].fill_between(t[m], 0, -br[m], step="post", color=ps.PALETTE["purple"], alpha=.8, linewidth=0, label="brake (neg.)")
    axs[2].set_ylabel("pedal")
    axs[2].set_ylim(-1.1, 1.1)
    axs[2].legend(loc="lower left", ncol=2, fontsize=7)
    axs[2].set_xlabel("simulation time (s)")
    for a in axs:
        a.axvline(t_col, color="#555555", linestyle="--", linewidth=.7)
    axs[0].text(t_col + 0.3, axs[0].get_ylim()[1] * 0.8, "collision with static.prop.trafficwarning", fontsize=7)
    for a, lab in zip(axs, "abc"):
        ps.panel(a, "(%s)" % lab)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.95, hspace=0.32)
    return ps.save(fig, FIGS / "alp-b2d-stall-route1833")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-dir", type=Path, required=True, help="dir with ticks.jsonl / plans.jsonl of route 1833")
    a = ap.parse_args()
    ps.apply()
    print(fig_overview())
    print(fig_route(a.route_dir))


if __name__ == "__main__":
    main()
