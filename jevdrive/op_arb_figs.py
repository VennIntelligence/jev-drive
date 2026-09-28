"""Figures for research/openpilot-closedloop-integration.md (op-arb). Reads a copy of runs/op_arb/arms (plans.jsonl and
results.json per finished attempt) and writes research/figs/op_arb_*.png / .pdf through research/plot_style.

    .venv/bin/python -m jevdrive.op_arb_figs p1 <runs/op_arb dir>      # phase-1 diagnosis panels
    .venv/bin/python -m jevdrive.op_arb_figs p2 <runs/op_arb dir>      # phase-2 per-route DS by arm
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import op_arb_report as R  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research"))
import plot_style as S  # noqa: E402

FIGS = REPO / "research" / "figs"


def p1(rootdir: Path) -> None:
    S.apply()
    steps = []
    for arm in ("native", "oshadow"):
        _, st = R.load(rootdir / "arms" / f"p1-{arm}")
        st["arm"] = arm
        steps.append(st)
    st = pd.concat(steps, ignore_index=True)
    fig, ax = plt.subplots(1, 3, figsize=(S.DOUBLE_COLUMN_IN, 2.15), gridspec_kw={"wspace": 0.42})
    # (a) standstill: plan displacement at 5 s by ground-truth context
    s = st[(st.v < 0.2) & ~st.warm].copy()
    s["ctx"] = R.context(s)
    groups = [("free", "no hazard"), ("red", "red light"), ("lead", "stopped lead")]
    data = [s[s.ctx == k].x5.clip(-0.5, 6).to_numpy() for k, _ in groups]
    bp = ax[0].boxplot(data, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True)
    for b, c in zip(bp["boxes"], (S.PALETTE["green"], S.PALETTE["vermillion"], S.PALETTE["orange"])):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        b.set_linewidth(0.6)
    for k in ("medians", "whiskers", "caps"):
        for ln in bp[k]:
            ln.set_color("#333333")
            ln.set_linewidth(0.7)
    ax[0].axhline(2.0, color=S.BASELINE, ls="--", lw=0.7)
    ax[0].text(3.45, 2.05, "release 2 m", ha="right", va="bottom", fontsize=7, color=S.BASELINE)
    ax[0].set_xticks([1, 2, 3], [f"{lab}\n(n={len(d)})" for (_, lab), d in zip(groups, data)], fontsize=7)
    ax[0].set_ylabel("plan displacement at 5 s (m)")
    ax[0].set_ylim(-0.3, 6)
    S.panel(ax[0], "(a) at standstill")
    S.bars(ax[0])
    # (b) rolling on a free road: planned speed change over 5 s against the current speed
    m = st[(st.arm == "oshadow") & (st.v > 1.0) & ~st.warm & ~st.zone].copy()
    m["ctx"] = R.context(m)
    m = m[m.ctx == "free"]
    edges = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    mid, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        g = (m.vp5 - m.vp0)[(m.v >= a) & (m.v < b)]
        if len(g) >= 5:
            mid.append((a + b) / 2)
            med.append(g.median())
            lo.append(g.quantile(0.25))
            hi.append(g.quantile(0.75))
    ax[1].fill_between(mid, lo, hi, color=S.PREDICTION, alpha=0.2, lw=0)
    ax[1].plot(mid, med, color=S.PREDICTION, marker="o", ms=2.5)
    S.zero_line(ax[1])
    ax[1].set_xlabel("current speed (m/s)")
    ax[1].set_ylabel(r"plan $v(5\,$s$) - v(0)$ (m/s)")
    S.panel(ax[1], "(b) rolling, free road")
    # (c) turns: fraction of the route's lateral offset at 15 m that the plan follows
    rows = R.turn_rows("oshadow", st[(st.arm == "oshadow") & (st.v > 1.0) & ~st.warm], arcs=(15.0,))
    t = pd.DataFrame(rows)
    x = np.arange(len(t))
    ax[2].bar(x - 0.18, t["plan follow @15 m"], 0.34, color=S.PREDICTION, label="route desire")
    ax[2].bar(x + 0.18, t["twin follow @15 m"], 0.34, color=S.BASELINE, label="no desire")
    ax[2].axhline(1.0, color="#999999", lw=0.5, ls=":")
    S.zero_line(ax[2])
    ax[2].set_xticks(x, ["before the turn\n(0-20 m)", "inside the turn"], fontsize=7)
    ax[2].set_ylabel("plan / route lateral offset at 15 m")
    ax[2].legend(loc="center left", fontsize=7)
    S.panel(ax[2], "(c) route turns")
    S.bars(ax[2])
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.88)
    print(S.save(fig, FIGS / "op_arb_p1_diag"))


def p2(rootdir: Path) -> None:
    S.apply()
    res = R.evaluate(rootdir, rootdir / "results_fig", "p2")["routes"]
    arms = [a for a in ("base", "acc", "e2e", "switch", "oplat") if a in set(res.arm)]
    routes = sorted(res.route.unique(), key=int)
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COLUMN_IN, 2.0))
    col = {"base": S.BASELINE, "acc": S.PALETTE["sky_blue"], "e2e": S.PALETTE["blue"], "switch": S.PALETTE["orange"],
           "oplat": S.PALETTE["vermillion"]}
    for k, a in enumerate(arms):
        g = res[res.arm == a].set_index("route").reindex(routes)
        ax.scatter(np.arange(len(routes)) + (k - (len(arms) - 1) / 2) * 0.13, g.DS, s=9, color=col[a], label=a, zorder=3)
    ax.set_xticks(np.arange(len(routes)), routes, fontsize=7)
    ax.set_xlabel("Bench2Drive 0.0.4 val route")
    ax.set_ylabel("driving score")
    ax.set_ylim(-3, 103)
    ax.legend(ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.84)
    print(S.save(fig, FIGS / "op_arb_p2_ds"))


if __name__ == "__main__":
    {"p1": p1, "p2": p2}[sys.argv[1]](Path(sys.argv[2]))
