"""Figures for research/articles/closed-loop-and-control/.

Every number comes from files already in the repo:
  fig1  docs/bench2drive-cost.md (profile, camera-rig and real-policy tables, copied below verbatim)
  fig2  research/results/b2d/full220-results.csv
  fig3  todos/2026-09-22-b2d-controller/results/formal-v4-comparison/routes.csv
        todos/2026-09-22-b2d-controller/results/comfort-v3-v4-v1/artifacts/per-case.csv
  fig4  todos/2026-09-23-controller-next/lateral_v2/results/analysis-v1/summary.json
  fig5, fig6 are copies of existing controller figures (see copy_existing()).

Run locally (no CARLA, no box):
  uv run --no-project --with matplotlib --with pandas --with pillow python scripts/make_article_closed_loop_figs.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import plot_style as ps  # noqa: E402

OUT = ROOT / "research/articles/closed-loop-and-control/figs"
CTRL = ROOT / "todos/2026-09-22-b2d-controller/results"
NEXT = ROOT / "todos/2026-09-23-controller-next"

P = ps.PALETTE
# One colour per arm across every figure in the article.
ARM = {
    "carla": ps.BASELINE,          # CARLA lateral PID + PI(.5,.25): the v4 reference
    "pursuit": P["blue"],          # pursuit max + PI(.5,.25): v4 candidate = lateral-v2 production
    "tcp": P["orange"],            # TCP vendor preset
    "slipack": P["vermillion"],    # slip-frame pursuit + Ackermann inverse (lateral v2 candidate)
}
PHASE = {"wait": P["sky_blue"], "world": P["blue"], "tree": P["orange"], "other": "#BBBBBB"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{name}: {(OUT / f'{name}.png').stat().st_size / 1024:.0f} KB")


# ---------------------------------------------------------------- fig 1
def fig1_tick_cost():
    # docs/bench2drive-cost.md, "Cameras cost per sensor, not per pixel" (route 24240, idle box)
    rigs = [  # label, ms/tick, sensor wait
        ("no camera", 29.1, 0.4),
        ("1 cam 1600x900", 92.4, 58.7),
        ("3 cam 1600x900", 158.3, 128.0),
        ("3 cam 800x450", 150.9, 117.5),
        ("3 cam 400x225", 153.4, 120.3),
        ("6 cam 1600x900", 263.3, 227.1),
    ]
    # docs/bench2drive-cost.md, "With a real policy in the loop"
    policy = [  # model, config, ms/tick
        ("DINOv2-B", "1 cam, every tick", 196.6),
        ("DINOv2-B", "448x252, dec. 4, overlap", 44.5),
        ("Qwen3-VL-4B", "3 cam, every tick", 520.9),
        ("Qwen3-VL-4B", "dec. 4, overlap, 0-copy", 85.2),
        ("Qwen3-VL-4B", "+ render 800x450", 71.1),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.15),
                               gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.75})
    y = np.arange(len(rigs))[::-1]
    tot = np.array([r[1] for r in rigs]); wait = np.array([r[2] for r in rigs])
    a.barh(y, wait, color=PHASE["wait"], height=0.62, label="waiting for camera frames")
    a.barh(y, tot - wait, left=wait, color=PHASE["other"], height=0.62, label="tree + world.tick + rest")
    for yi, t in zip(y, tot):
        a.text(t + 4, yi, f"{t:.0f}", va="center", fontsize=7)
    a.set_yticks(y, [r[0] for r in rigs])
    a.set_xlabel("ms per tick (no policy)")
    a.set_xlim(0, 300)
    a.legend(loc="upper right", fontsize=6.8)
    a.grid(axis="y", visible=False)
    ps.panel(a, "(a)")

    y = np.arange(len(policy))[::-1]
    cols = [P["green"] if m.startswith("DINO") else P["purple"] for m, _, _ in policy]
    b.barh(y, [p[2] for p in policy], color=cols, height=0.62)
    for yi, p in zip(y, policy):
        b.text(p[2] + 8, yi, f"{p[2]:.1f}", va="center", fontsize=7)
    b.axvline(50, color="#555555", ls="--", lw=0.7)
    b.text(215, 3, "dashed: real time (50 ms)", fontsize=6.8, color="#555555", va="center")
    b.set_yticks(y, [f"{m}: {c}" for m, c, _ in policy], fontsize=6.8)
    b.set_xlabel("ms per tick (policy in the loop)")
    b.set_xlim(0, 600)
    b.grid(axis="y", visible=False)
    ps.panel(b, "(b)")
    save(fig, "fig1_tick_cost")


# ---------------------------------------------------------------- fig 2
def fig2_town_cost():
    d = pd.read_csv(ROOT / "research/results/b2d/full220-results.csv")
    fin = d[d.status == "finished"].copy()
    fail_routes = (d.groupby("route_id").status.apply(lambda s: (s == "finished").any()))
    never = set(fail_routes[~fail_routes].index)
    g = fin.groupby("town").agg(ms=("total_ms", "mean"), wt=("world_tick_ms", "mean"),
                                tree=("tree_ms", "mean"), n=("route_id", "nunique"))
    g["never"] = [len({r for r in never if (d[d.route_id == r].town.iloc[0] == t)}) for t in g.index]
    g["other"] = (g.ms - g.wt - g.tree).clip(lower=0)
    g = g.sort_values("ms")
    wall = d.groupby("town").wall_s.sum() / d.wall_s.sum()

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.35),
                               gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.32})
    y = np.arange(len(g))
    a.barh(y, g.wt, color=PHASE["world"], height=0.66, label="world.tick")
    a.barh(y, g.tree, left=g.wt, color=PHASE["tree"], height=0.66, label="scenario tree")
    a.barh(y, g.other, left=g.wt + g.tree, color=PHASE["other"], height=0.66, label="rest")
    for yi, (t, r) in zip(y, g.iterrows()):
        s = f"{r.ms:.0f} ms, n={int(r.n + r.never)}, wall {100 * wall[t]:.0f}%"
        if r.never:
            s += f", {int(r.never)} never finished"
        a.text(r.ms + 2, yi, s, va="center", fontsize=6.3)
    a.set_yticks(y, g.index)
    a.set_xlim(0, 290)
    a.set_xlabel("mean ms per tick (finished routes, 8 workers, shared GPU)")
    a.legend(loc="lower right", fontsize=6.8)
    a.grid(axis="y", visible=False)
    ps.panel(a, "(a)")

    big = fin.town.isin(["Town12", "Town13"])
    bins = np.arange(0, 4001, 250)
    b.hist([fin.ticks[big], fin.ticks[~big]], bins=bins, stacked=True,
           color=[P["vermillion"], P["sky_blue"]], label=["Town12/13", "other towns"], edgecolor="white", lw=0.3)
    cap = (fin.ticks >= 4000).sum()
    b.axvline(4000, color="#555555", ls="--", lw=0.7)
    b.text(3900, b.get_ylim()[1] * 0.9, f"4000-tick cap:\n{cap}/{len(fin)} routes", ha="right", va="top", fontsize=6.8)
    b.set_xlabel("ticks per finished route")
    b.set_ylabel("routes")
    b.legend(loc="upper left", fontsize=6.8)
    ps.panel(b, "(b)")
    save(fig, "fig2_town_cost")


# ---------------------------------------------------------------- fig 3
def fig3_score_vs_driving():
    d = pd.read_csv(CTRL / "formal-v4-comparison/routes.csv")
    d = d[d.selected_attempt == True]  # noqa: E712  60 official rows
    k = ["route_id", "seed"]
    c = d[d.preset == "carla"].set_index(k)
    p = d[d.preset == "pursuit"].set_index(k)
    dds = p.score_composed - c.score_composed
    dcte = p.tracking_truth_cross_track_m_rms - c.tracking_truth_cross_track_m_rms

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.45), gridspec_kw={"wspace": 0.38})
    for seed, m in ((0, "o"), (1, "s")):
        s = dds.index.get_level_values("seed") == seed
        a.scatter(dcte[s], dds[s], marker=m, s=18 if m == "o" else 12, color=ARM["pursuit"],
                  facecolor="none" if seed else ARM["pursuit"], lw=0.8, label=f"TM seed {seed}", zorder=3)
    ps.zero_line(a)
    a.axvline(0, color="#999999", lw=0.5, zorder=0)
    notes = [((25381, 0), (10, -24), "25381, both seeds:\nCARLA+PI 1 vehicle collision,\npursuit 0; CTE equal"),
             ((2091, 1), (-10, 14), "2091 seed 1"),
             ((2091, 0), (-60, 18), "2091 seed 0: both collide,\npursuit stops at 79% RC"),
             ((27494, 0), (-30, 16), "27494"), ((3514, 0), (4, 14), "3514")]
    for key, off, txt in notes:
        a.annotate(txt, (dcte.loc[key], dds.loc[key]), xytext=off, textcoords="offset points", fontsize=6.2,
                   ha="left",
                   arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666"))
    nz = int((dds.abs() < 1e-9).sum())
    a.text(-0.53, 44, f"{nz}/{len(dds)} route-seed pairs: $\\Delta$DS = 0", fontsize=6.5)
    a.set_ylim(-5, 46)
    a.set_xlabel(r"$\Delta$ full-route CTE RMS (m), pursuit $-$ CARLA+PI")
    a.set_ylabel(r"$\Delta$ Driving Score, pursuit $-$ CARLA+PI")
    a.set_xlim(-0.55, 0.45)
    a.legend(loc="lower left", fontsize=6.8)
    ps.panel(a, "(a)")

    q = pd.read_csv(CTRL / "comfort-v3-v4-v1/artifacts/per-case.csv")
    q = q[q["mode"] == "raw"].pivot_table(index="case", columns=["version", "metric"], values="rms")
    rel = lambda m: 100 * (q[("v4", m)] / q[("v3", m)] - 1)  # noqa: E731
    mets = [("a_long_mps2", "long. accel"), ("j_long_mps3", "long. jerk"),
            ("a_right_mps2", "lat. accel"), ("j_right_mps3", "lat. jerk")]
    rng = np.random.default_rng(0)
    for i, (m, lab) in enumerate(mets):
        v = rel(m).values
        b.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=9, color=ARM["pursuit"] if "long" in m else P["green"],
                  alpha=0.8, lw=0, zorder=3)
        b.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="#222222", lw=1.0)
    ps.zero_line(b)
    b.set_xticks(range(4), [l for _, l in mets])
    b.set_ylabel("per-case RMS change, v4 vs v3 (%)")
    b.grid(axis="x", visible=False)
    ps.panel(b, "(b)")
    save(fig, "fig3_score_vs_driving")


# ---------------------------------------------------------------- fig 4
def fig4_lateral_v2():
    s = json.load(open(NEXT / "lateral_v2/results/analysis-v1/summary.json"))
    wins = [w["name"] for w in s["windows"]]
    held = {w["name"]: w["heldout"] for w in s["windows"]}
    order = [w for w in wins if not held[w]] + [w for w in wins if held[w]]
    pa, pp = s["per_arm"], s["paired"]
    x = np.arange(len(order))

    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.DOUBLE_COLUMN_IN, 4.1), sharex=True,
                               gridspec_kw={"hspace": 0.18})
    for arm, col, dx, lab in (("prod-kfix", ARM["pursuit"], -0.12, "production (pursuit max+PI), fixed-k pose"),
                              ("slipack-kfix", ARM["slipack"], 0.12, "slip frame + Ackermann, fixed-k pose")):
        for i, w in enumerate(order):
            r = pa[f"{w}|{arm}"]["cte_rms"]
            thin = r["n"] < 10
            lo, hi = r["ci"] if r["ci"] and r["ci"][0] is not None else (r["median"], r["median"])
            a.errorbar(i + dx, r["median"], yerr=[[r["median"] - lo], [hi - r["median"]]], fmt="o", ms=3.6,
                       color=col, mfc="white" if thin else col, capsize=1.6, lw=0.8,
                       label=lab if i == 0 else None, zorder=3)
        tr = arm.replace("kfix", "truth")
        a.scatter(x + dx, [pa[f"{w}|{tr}"]["cte_rms"]["median"] for w in order], marker="x", s=14, lw=0.8,
                  color=col, zorder=3)
    a.axhline(0.25, color="#555555", ls=":", lw=0.7)
    a.text(len(order) - 0.5, 0.255, "P1 gate on 26966: 0.25 m", ha="right", va="bottom", fontsize=6.5, color="#555555")
    a.set_ylabel("window CTE RMS (m)\nmedian of 10 seeds, 95% CI")
    a.set_ylim(0, 0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ARM["pursuit"], ms=4),
         Line2D([], [], marker="o", ls="", color=ARM["slipack"], ms=4),
         Line2D([], [], marker="x", ls="", color="#444444", ms=4)]
    l = ["production (pursuit max + PI), fixed-k pose", "slip frame + Ackermann, fixed-k pose",
         "same controller, truth pose (diagnostic)"]
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    ps.panel(a, "(a)")

    for key, col, dx, lab in (("steer_rate_p95_ratio", P["purple"], -0.12, "emitted steer-rate P95 (guard +20%)"),
                              ("lat_acc_p95_ratio", P["green"], 0.12, "lateral acceleration P95 (guard +10%)")):
        for i, w in enumerate(order):
            r = pp[f"{w}|slipack-kfix-vs-prod-kfix"][key]
            m = 100 * r["mean"]
            thin = r["n"] < 10
            lo, hi = (100 * r["ci"][0], 100 * r["ci"][1]) if (r["ci"] and r["ci"][0] is not None) else (m, m)
            b.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3.6, color=col,
                       mfc="white" if thin else col, capsize=1.6, lw=0.8, label=lab if i == 0 else None, zorder=3)
    b.axhline(20, color=P["purple"], ls="--", lw=0.6)
    b.axhline(10, color=P["green"], ls="--", lw=0.6)
    ps.zero_line(b)
    b.set_ylabel("paired relative change,\ncandidate vs production (%)")
    b.legend(loc="upper left", bbox_to_anchor=(0.36, 1.0), fontsize=6.5)
    b.set_ylim(-8, 72)
    for ax in (a, b):
        ax.axvline(3.5, color="#888888", lw=0.6)
    a.text(1.5, 0.475, "development windows", ha="center", fontsize=7)
    a.text(7.5, 0.47, "held-out windows (geometry-selected)", ha="center", fontsize=7)
    b.set_xticks(x, [w + ("*" if pa[f"{w}|slipack-kfix"]["cte_rms"]["n"] < 10 else "") for w in order],
                 rotation=30, ha="right")
    b.grid(axis="x", visible=False)
    ps.panel(b, "(b)")
    save(fig, "fig4_lateral_v2")


# ---------------------------------------------------------------- copies
def copy_existing():
    src = NEXT / "lateral_physics/figs"
    for ext in ("png", "pdf"):
        shutil.copy2(src / f"lateral-physics.{ext}", OUT / f"fig5_lateral_physics.{ext}")
    from PIL import Image  # contact sheet is a photo strip; palette-quantise to stay under ~500 KB
    im = Image.open(NEXT / "lateral_v2/figs/26966-contact-sheet.png").convert("RGB")
    im.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(OUT / "fig6_26966_contact_sheet.png", optimize=True)
    print(f"fig6: {(OUT / 'fig6_26966_contact_sheet.png').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    ps.apply()
    fig1_tick_cost()
    fig2_town_cost()
    fig3_score_vs_driving()
    fig4_lateral_v2()
    copy_existing()
