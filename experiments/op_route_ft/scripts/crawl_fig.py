"""Figures for results/crawl.md: three representative turns (time series + active harness rule) and the per-arm stopped-time budget.

    .venv/bin/python experiments/op_route_ft/scripts/crawl_fig.py DUMP.json CRAWL.json --figs experiments/op_route_ft/figs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import crawl as K  # noqa: E402

SRC = {"plan": "#1f77b4", "base": "#ff7f0e", "lead": "#9467bd", "latch": "#d62728", "warm": "#bbbbbb"}
CAUSE = {"warm-up": "#bbbbbb", "model": "#1f77b4", "red light": "#d62728", "blocker": "#9467bd", "harness": "#ff7f0e"}
CASES = [("zones-on", "24944", 0, "A. zones-on, route 24944 turn 0 (free road, red light 55 m ahead): stop-go limit cycle"),
         ("shipped", "10255", 0, "B. shipped (zones off), route 10255 turn 0: start at a red light, stops in the junction"),
         ("shipped", "15102", 0, "C. shipped (zones off), route 15102 turn 0: a parked car 5 m ahead, ego 2 m off the route")]


def case_fig(D, path):
    fig, axes = plt.subplots(3, 3, figsize=(17, 9.5), gridspec_kw=dict(height_ratios=[3, 2, 1.3]), sharex="col")
    for c, (arm, rid, tn, title) in enumerate(CASES):
        r = next(x for x in D if x["arm"] == arm and x["route"] == rid and x["turn"] == tn)
        tk = r["ticks"]
        t = np.array([x["t"] for x in tk])
        v = np.array([x["v"] for x in tk])
        vp = np.array([x["vplan"] if x["vplan"] else [np.nan] * 5 for x in tk])
        aa = np.array([x["act_a"] if x["act_a"] is not None else np.nan for x in tk])
        ap = np.array([x["aplan"][0] if x["aplan"] else np.nan for x in tk])
        a0, a1, a2 = axes[:, c]
        a0.set_title(title, fontsize=8.5, loc="left")
        a0.plot(t, vp[:, 3], color="#9ecae1", lw=1.2, label="plan speed at +3 s")
        a0.plot(t, vp[:, 1], color="#1f77b4", lw=1.4, label="plan speed at +1 s")
        a0.plot(t, v, color="k", lw=1.8, label="executed speed")
        stop = v < K.V_STOP
        a0.fill_between(t, 0, 1, where=stop, transform=a0.get_xaxis_transform(), color="#999999", alpha=0.18, lw=0)
        a0.axhline(1.0, color="#ff7f0e", lw=0.8, ls=":")
        a0.text(t[0] + 0.2, 1.08, "plan_vmin 1 m/s: below it the plan is not in the speed arbitration after 1.5 s", fontsize=6.5, color="#ff7f0e")
        a0.set_ylabel("speed (m/s)")
        a0.set_ylim(-0.2, 7)
        if c == 0:
            a0.legend(fontsize=7, loc="upper right")
        a1.plot(t, ap, color="#1f77b4", lw=1.2, label="plan accel at t = 0")
        a1.plot(t, aa, color="#d62728", lw=1.2, label="action head accel")
        a1.axhline(0, color="k", lw=0.5)
        a1.set_ylabel("accel (m/s$^2$)")
        a1.set_ylim(-2.2, 2.2)
        if c == 0:
            a1.legend(fontsize=7, loc="upper right")
        # rule strips: binding source, ground-truth context
        rows = [("binding source", [("warm" if K.is_warm(x) else "latch" if x["latch"] else x["src"] or "warm") for x in tk]),]
        for i, (nm, labs) in enumerate(rows):
            for k in range(len(t) - 1):
                a2.axhspan(2, 3, xmin=0, xmax=0)  # keep axes ordering stable
                a2.fill_between([t[k], t[k + 1]], 2, 3, color=SRC.get(labs[k], "#ccc"), lw=0)
        ctx = [("red light <30 m", [K.red(x) for x in tk], "#d62728"), ("actor in the way", [K.blocker(x) for x in tk], "#9467bd")]
        for i, (nm, on, col) in enumerate(ctx):
            for k in range(len(t) - 1):
                if on[k]:
                    a2.fill_between([t[k], t[k + 1]], 1 - i, 2 - i, color=col, lw=0)
        a2.set_yticks([2.5, 1.5, 0.5])
        a2.set_yticklabels(["binding source", "red light", "actor ahead"], fontsize=7)
        a2.set_ylim(0, 3)
        a2.set_xlabel("time since turn entry (s)")
        for ax in (a0, a1, a2):
            ax.set_xlim(t[0], t[-1])
            ax.axvline(0, color="#555", lw=0.6, ls="--")
    h = [plt.Rectangle((0, 0), 1, 1, color=v) for v in SRC.values()]
    fig.legend(h, ["binding: " + k for k in SRC], ncol=5, fontsize=8, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def budget_fig(S, path):
    arms = list(S)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
    bot = np.zeros(len(arms))
    for c in CAUSE:
        h = np.array([S[a]["stopped_s_by_cause"].get(c, 0) for a in arms])
        ax[0].bar(arms, h, bottom=bot, color=CAUSE[c], label=c)
        bot += h
    ax[0].set_ylabel("stopped s per turn (15 s window)")
    ax[0].legend(fontsize=8)
    ax[0].tick_params(axis="x", labelrotation=20)
    ax[0].set_title("Stopped time by cause", fontsize=9, loc="left")
    x = np.arange(len(arms))
    ax[1].bar(x - 0.2, [S[a]["v_med_nowarm"] for a in arms], 0.4, color="k", label="median executed speed")
    ax[1].bar(x + 0.2, [S[a]["vplan1_med_nowarm"] for a in arms], 0.4, color="#1f77b4", label="median plan speed at +1 s")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(arms, rotation=20)
    ax[1].set_ylabel("m/s (warm-up ticks dropped)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("The executed speed equals what the model plans", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("crawl")
    ap.add_argument("--figs", required=True)
    a = ap.parse_args()
    D = json.load(open(a.dump))
    S = json.load(open(a.crawl))["summary"]
    Path(a.figs).mkdir(exist_ok=True)
    case_fig(D, Path(a.figs) / "crawl_cases.png")
    budget_fig(S, Path(a.figs) / "crawl_budget.png")
