#!/usr/bin/env python
"""Figures of the NAVSIM zero-shot exam (todos/2026-09-24-zeroshot-exam/navsim.md), from the report tables of
scripts/navsim_zs_report.py. Runs on the box in envs/navsim2 (OPENBLAS_CORETYPE=Haswell, see docs/navsim.md);
PDF + PNG to $DATA_DIR/runs/navsim_zs/report/figs/, PNGs then copied to research/figs/.

  subscores  navtest EPDMS sub-score means per agent
  command    navtest EPDMS by driving command: Alpamayo nav vs no-nav (paired subset), openpilot none vs cmd
  failures   the failure classes of Alpamayo nav: for each class the first two tokens in token order (no picking),
             front-wide model image at t0 + BEV of the prediction, the log and the history
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "research")]
import plot_style as S  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402

REP = Z.root("report")
FIG = Z.root("report", "figs")
COLORS = {"human": "#4D4D4D", "cv": "#AAAAAA", "alpamayo_nav": S.PALETTE["blue"], "alpamayo_nonav": S.PALETTE["sky_blue"],
          "lebowski_none": S.PALETTE["vermillion"], "cinque_none": S.PALETTE["orange"], "small_none": S.PALETTE["yellow"],
          "lebowski_cmd": S.PALETTE["purple"]}
LABEL = {"human": "Human (log)", "cv": "Constant velocity", "alpamayo_nav": "Alpamayo 1.5 (nav)",
         "alpamayo_nonav": "Alpamayo 1.5 (no nav)", "lebowski_none": "openpilot Lebowski", "cinque_none": "openpilot Cinque v3",
         "small_none": "openpilot small", "lebowski_cmd": "Lebowski + turn desire"}
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
       "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
SUBL = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]


def scores():
    return pd.read_csv(REP / "scores_navtest.csv.gz")


def fig_subscores(sc):
    agents = [a for a in ("human", "cv", "alpamayo_nav", "lebowski_none", "cinque_none", "small_none")
              if a in set(sc.agent)]
    v2 = sc[sc.metric == "EPDMS"]
    fig, ax = plt.subplots(figsize=(S.DOUBLE_COLUMN_IN, 1.9))
    w = 0.8 / len(agents)
    for i, a in enumerate(agents):
        m = 100 * v2[v2.agent == a][SUB + ["score"]].mean()
        ax.bar(np.arange(len(SUB) + 1) + (i - (len(agents) - 1) / 2) * w, m.values, w, color=COLORS[a], label=LABEL[a])
    ax.set_xticks(np.arange(len(SUB) + 1), SUBL + ["EPDMS"])
    ax.set_ylabel("mean over navtest (%)")
    ax.set_ylim(0, 105)
    S.bars(ax)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.14, top=0.78)
    return S.save(fig, FIG / "navsim-zs-subscores")


def fig_command(sc):
    idx = Z.load_index("navtest")
    cmd = {e["token"]: int(np.argmax(e["cmd"][-1])) for e in idx}
    sub = Z.nonav_subset(idx)
    v2 = sc[sc.metric == "EPDMS"].copy()
    v2["cmd"] = v2.token.map(cmd)
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 1.8), sharey=True)
    for ax, (pairs, title, toks) in zip(axes, (
            (("alpamayo_nav", "alpamayo_nonav"), "(a) Alpamayo, paired 3000-token subset", sub),
            (("lebowski_none", "lebowski_cmd"), "(b) openpilot Lebowski, all navtest", None))):
        for i, a in enumerate(pairs):
            d = v2[v2.agent == a]
            if toks is not None:
                d = d[d.token.isin(toks)]
            m = [100 * d[d.cmd == c].score.mean() for c in (0, 1, 2)]
            ax.bar(np.arange(3) + (i - 0.5) * 0.38, m, 0.38, color=COLORS[a], label=LABEL[a])
        ax.set_xticks(range(3), ["left", "straight", "right"])
        S.panel(ax, title)
        S.bars(ax)
        ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7)
        ax.set_ylim(0, 62)
    axes[0].set_ylabel("EPDMS (%)")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.14, top=0.86, wspace=0.08)
    return S.save(fig, FIG / "navsim-zs-command")


def _front_wide(e, maps):
    imgs = {}
    for c, s in maps.sources():
        flag = cv2.IMREAD_COLOR if s == 1 else cv2.IMREAD_REDUCED_COLOR_4
        imgs[(c, s)] = cv2.imread(e["cams"][-1][c]["path"], flag)[..., ::-1].copy()
    return maps.render(imgs)[1].transpose(1, 2, 0)              # camera index 1 = front-wide


def fig_failures(sc):
    v2 = sc[(sc.metric == "EPDMS") & (sc.agent == "alpamayo_nav")].sort_values("token")
    classes = [("no_at_fault_collisions", "at-fault collision"), ("drivable_area_compliance", "leaves drivable area"),
               ("ego_progress", "low progress (EP < 0.2)")]
    picks = []
    for col, name in classes:
        bad = v2[v2[col] < (0.2 if col == "ego_progress" else 1)]
        if col == "ego_progress":
            bad = bad[(bad.no_at_fault_collisions == 1) & (bad.drivable_area_compliance == 1)]
        picks += [(t, name, len(bad) / len(v2)) for t in bad.token.head(2)]
    idx = {e["token"]: e for e in Z.load_index("navtest")}
    pred = np.load(Z.root("preds", "navtest") / "alpamayo_nav_repeat_main.npz")
    pred = dict(zip(pred["tokens"].tolist(), pred["poses"]))
    fut = np.load(Z.root("index") / "navtest_future.npz")
    fut = dict(zip(fut["tokens"].tolist(), fut["poses"]))
    maps = Z.AlpamayoMaps(idx[picks[0][0]]["cams"][-1])
    fig = plt.figure(figsize=(S.DOUBLE_COLUMN_IN, 1.35 * len(picks) / 2 + 0.3))
    for k, (tok, name, rate) in enumerate(picks):
        e = idx[tok]
        r, c = divmod(k, 2)
        ax_i = fig.add_axes([0.01 + 0.5 * c, 1 - (r + 1) / (len(picks) / 2) * 0.95, 0.30, 0.9 / (len(picks) / 2)])
        ax_b = fig.add_axes([0.33 + 0.5 * c, 1 - (r + 1) / (len(picks) / 2) * 0.95 + 0.02, 0.15, 0.85 / (len(picks) / 2)])
        ax_i.imshow(_front_wide(e, maps))
        ax_i.set_axis_off()
        ax_i.text(4, 18, f"{name}  (class rate {100 * rate:.1f}%)\ncmd {['left', 'straight', 'right', '?'][int(np.argmax(e['cmd'][-1]))]}, "
                  f"v = {np.linalg.norm(e['vel'][-1]):.1f} m/s", color="white", fontsize=6.5, va="top")
        h, p, g = e["pose"], pred[tok], fut[tok]
        ax_b.plot(-h[:, 1], h[:, 0], color="#999999", lw=0.9)
        ax_b.plot(-g[:, 1], g[:, 0], "k--", lw=0.9, label="log")
        ax_b.plot(np.r_[0, -p[:, 1]], np.r_[0, p[:, 0]], color=COLORS["alpamayo_nav"], lw=1.1, label="Alpamayo")
        ax_b.set_aspect("equal", adjustable="datalim")
        ax_b.tick_params(labelsize=6)
        ax_b.grid(True)
        if k == 0:
            ax_b.legend(fontsize=6, loc="lower left", handlelength=1.2)
    return S.save(fig, FIG / "navsim-zs-failures")


def fig_inputs(sc=None, k=0):
    """Adapter check: native nuPlan views vs what each model sees, for navtest token k of the index (not picked)."""
    from jevdrive.openpilot.frames import unpack_luma
    e = Z.load_index("navtest")[k]
    maps = Z.AlpamayoMaps(e["cams"][-1])
    imgs = {}
    for c, s_ in maps.sources():
        imgs[(c, s_)] = cv2.imread(e["cams"][-1][c]["path"], cv2.IMREAD_COLOR if s_ == 1 else cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1].copy()
    alp = maps.render(imgs).transpose(0, 2, 3, 1)
    raw = [cv2.imread(e["cams"][-1][c]["path"], cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1] for c in ("CAM_L0", "CAM_F0", "CAM_R0")]
    op = np.load(Z.root("openpilot", "navtest") / "frames.npy", mmap_mode="r")[k, 3]
    fig = plt.figure(figsize=(S.DOUBLE_COLUMN_IN, 3.9))
    gs = fig.add_gridspec(3, 12, hspace=0.25, wspace=0.05, left=0.005, right=0.995, top=0.95, bottom=0.01)
    rows = [[(raw[i], t) for i, t in enumerate(("nuPlan CAM_L0", "nuPlan CAM_F0", "nuPlan CAM_R0"))],
            [(alp[i], t) for i, t in enumerate(("Alpamayo cross-left", "Alpamayo front-wide", "Alpamayo cross-right"))],
            [(alp[3], "Alpamayo front-tele"), (unpack_luma(op[0]), "openpilot road frame"), (unpack_luma(op[1]), "openpilot wide frame")]]
    for r, row in enumerate(rows):
        for c, (im, t) in enumerate(row):
            ax = fig.add_subplot(gs[r, 4 * c:4 * c + 4])
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_title(t, fontsize=7, pad=2)
            ax.set_axis_off()
    return S.save(fig, FIG / "navsim-zs-inputs")


def fig_bev(sc=None, n=8, seed=0):
    """Adapter check: n random navtest tokens (seed 0), log vs predictions in the NAVSIM rear-axle frame, with the
    4 s heading as an arrow. Up = forward (x), left = +y."""
    idx = Z.load_index("navtest")
    pick = np.random.default_rng(seed).choice(len(idx), n, replace=False)
    fut = np.load(Z.root("index") / "navtest_future.npz")
    g = dict(zip(fut["tokens"].tolist(), fut["poses"]))
    src = {"alpamayo_nav": Z.root("preds", "navtest") / "alpamayo_nav_repeat_main.npz",
           "lebowski_none": Z.root("openpilot", "navtest") / "lebowski_none.npz",
           "cv": Z.root("preds", "navtest") / "cvreplay.npz"}
    P = {}
    for a, f in src.items():
        z = np.load(f)
        P[a] = dict(zip(z["tokens"].tolist(), z["poses"]))
    fig, axes = plt.subplots(2, n // 2, figsize=(S.DOUBLE_COLUMN_IN, 3.2))
    for ax, k in zip(axes.ravel(), pick):
        e = idx[k]
        t = e["token"]
        ax.plot(-e["pose"][:, 1], e["pose"][:, 0], color="#BBBBBB", lw=0.9)
        for a, style in (("cv", dict(color=COLORS["cv"], lw=0.8)), ("lebowski_none", dict(color=COLORS["lebowski_none"], lw=1.0)),
                         ("alpamayo_nav", dict(color=COLORS["alpamayo_nav"], lw=1.1)), ("log", dict(color="k", ls="--", lw=0.9))):
            p = g[t] if a == "log" else P[a][t]
            ax.plot(np.r_[0, -p[:, 1]], np.r_[0, p[:, 0]], label=LABEL.get(a, "log"), **style)
            ax.annotate("", xy=(-p[-1, 1] - 2 * np.sin(p[-1, 2]), p[-1, 0] + 2 * np.cos(p[-1, 2])), xytext=(-p[-1, 1], p[-1, 0]),
                        arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=0.6, mutation_scale=5))
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=6)
        ax.set_title(f"cmd {['L', 'S', 'R', '?'][int(np.argmax(e['cmd'][-1]))]}, {np.linalg.norm(e['vel'][-1]):.1f} m/s", fontsize=7, pad=2)
    axes[0, 0].legend(fontsize=6, loc="upper left", handlelength=1.2)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.06, top=0.93, wspace=0.3, hspace=0.3)
    return S.save(fig, FIG / "navsim-zs-bev")


if __name__ == "__main__":
    S.apply()
    sc = scores()
    for f in (fig_inputs, fig_bev, fig_subscores, fig_command, fig_failures):
        try:
            print(f.__name__, f(sc))
        except Exception as ex:  # a figure whose inputs are not there yet is skipped, the others still render
            print(f.__name__, "skipped:", repr(ex))
