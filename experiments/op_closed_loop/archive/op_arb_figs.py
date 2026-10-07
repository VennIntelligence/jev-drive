"""Figures for research/openpilot-diagnosis/index.html (op-arb). Reads a copy of runs/op_arb/arms (plans.jsonl and
results.json per finished attempt) and writes research/figs/op_arb_*.png / .pdf through research/plot_style.

    .venv/bin/python -m experiments.op_closed_loop.archive.op_arb_figs p1 <runs/op_arb dir>      # phase-1 diagnosis panels
    .venv/bin/python -m experiments.op_closed_loop.archive.op_arb_figs p2 <dir with per_route.csv>  # phase-2 per-route DS by arm
    .venv/bin/python -m experiments.op_closed_loop.archive.op_arb_figs drive <results dir> <tag>   # op-drive: per-route DS and control shares
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("research",)]

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.op_closed_loop.lib import op_arb_report as R  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
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
    res = pd.read_csv(rootdir / "per_route.csv", dtype={"route": str})   # experiments.op_closed_loop.lib.op_arb_report eval output
    arms = [a for a in ("base", "acc", "acc2", "e2e", "switch", "oplat") if a in set(res.arm)]
    routes = sorted(res.route.unique(), key=int)
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COLUMN_IN, 2.0))
    col = {"base": S.BASELINE, "acc": S.PALETTE["sky_blue"], "acc2": S.PALETTE["green"], "e2e": S.PALETTE["blue"], "switch": S.PALETTE["orange"],
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


def drive(rootdir: Path, tag: str) -> None:
    """op-drive (fc65452:todos/2026-09-29-op-drive.md): (a) per-route DS of each arm, mean over TM seeds with the seed range;
    (b) who controlled the drive arm, as a share of distance (lateral) and of moving steps (longitudinal)."""
    S.apply()
    res = pd.read_csv(rootdir / "per_route.csv", dtype={"route": str})     # experiments.op_closed_loop.lib.op_arb_report drive output
    arms = [a for a in ("dbase", "dbaseslow", "dlon", "drive") if a in set(res.arm)]
    lab = {"dbase": "base (8 m/s)", "dbaseslow": "base, speed-matched", "dlon": "openpilot long. only", "drive": "openpilot drive"}
    col = {"dbase": S.BASELINE, "dbaseslow": S.PALETTE["sky_blue"], "dlon": S.PALETTE["orange"], "drive": S.PALETTE["blue"]}
    routes = sorted(res.route.unique(), key=int)
    fig, ax = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.6), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.3})
    for k, a in enumerate(arms):
        g = res[res.arm == a].groupby("route").DS.agg(["mean", "min", "max"]).reindex(routes)
        x = np.arange(len(routes)) + (k - (len(arms) - 1) / 2) * 0.17
        ax[0].errorbar(x, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], fmt="o", ms=3, lw=0.7,
                       color=col[a], label=lab[a], zorder=3)
    ax[0].set_xticks(np.arange(len(routes)), routes, fontsize=6, rotation=90 if len(routes) > 12 else 0)
    ax[0].set_xlabel("Bench2Drive 0.0.4 val route")
    ax[0].set_ylabel("driving score")
    ax[0].set_ylim(-3, 103)
    ax[0].legend(ncol=len(arms), loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=7)
    S.panel(ax[0], "a")
    d = res[res.arm == "drive"]
    lat = [d.lat_op.mean(), d.lat_zone.mean(), d.lat_div.mean()]
    lon = [d.lon_lead.mean(), d.lon_plan.mean(), max(d.lon_op.mean() - d.lon_lead.mean() - d.lon_plan.mean(), 0.0)]
    parts = [("lateral", lat, ["openpilot steers", "route: junction zone", "route: divergence"],
              [S.PALETTE["blue"], S.BASELINE, S.PALETTE["vermillion"]]),
             ("longitudinal", lon, ["lead head", "plan", "stop latch"],
              [S.PALETTE["green"], S.PALETTE["sky_blue"], S.PALETTE["purple"]])]
    for y, (name, vals, names, cols) in enumerate(parts):
        left = 0.0
        for v, n, c in zip(vals, names, cols):
            ax[1].barh(y, v, left=left, color=c, height=0.6, label=n)
            if v > 0.08:
                ax[1].text(left + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
            left += v
    ax[1].set_yticks([0, 1], ["lateral\n(distance)", "longitudinal\n(moving steps)"], fontsize=7)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("share controlled")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.3, -0.42), ncol=2, fontsize=5.5, handlelength=1)
    S.panel(ax[1], "b")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.36, top=0.8)
    print(S.save(fig, FIGS / f"op_drive_{tag}"))


if __name__ == "__main__":
    {"p1": p1, "p2": p2}[sys.argv[1]](Path(sys.argv[2])) if sys.argv[1] != "drive" else drive(Path(sys.argv[2]), sys.argv[3])
