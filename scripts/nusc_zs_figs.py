#!/usr/bin/env python
"""Figures of the nuScenes / PhysicalAI-AV zero-shot exam (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).
Project venv on the box; PDF + PNG to $DATA_DIR/runs/nusc_zs/figs/, PNGs then copied to research/figs/.

  adapter  one tail keyframe: Alpamayo's 4 model images, the native nuScenes front cameras, openpilot's road / wide
  bev      tail keyframes (in no evaluation set): both models' predictions with heading ticks over the ego's own
           future (as far as the scene goes) and 1.5 s of history, rear-axle t0 frame
  pai      PhysicalAI-AV: native front-wide vs openpilot's road / wide, and per-clip ADE openpilot vs Alpamayo
  results  nuScenes L2 / collision per row with scene-bootstrap CIs and the published rows
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "research"), str(REPO / "scripts")]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plot_style as S  # noqa: E402
from jevdrive import nuscenes_zs as Z  # noqa: E402
from jevdrive.common import data_dir, dataroot  # noqa: E402

FIG = Z.root("figs")
C = {"alp": S.PALETTE["blue"], "lebowski": S.PALETTE["vermillion"], "cinque": S.PALETTE["orange"],
     "small": S.PALETTE["yellow"], "cv": S.BASELINE, "gt": "#000000"}


def latest(pattern: str) -> Path:
    return sorted(data_dir().glob(pattern))[-1]


def rgb(path, size=None):
    import cv2
    im = cv2.imread(str(path))[..., ::-1]
    return cv2.resize(im, size) if size else im


def op_views(scene: dict, t0: int):
    """openpilot road / wide luma for the CAM_FRONT frame at t0, exactly as the runner packs it."""
    from PIL import Image
    from jevdrive.openpilot.frames import unpack_luma
    from nusc_zs_openpilot import op_index, pack
    ix, _ = op_index(scene)
    im = Image.open(dataroot() / scene["cams"]["CAM_FRONT"]["path"][Z.latest_frame(scene, "CAM_FRONT", t0 + 1)])
    im.draft("YCbCr", im.size)
    ycc = np.asarray(im.convert("YCbCr"))
    return [unpack_luma(pack(ycc, i)) for i in ix]


def fig_adapter(a):
    idx = Z.load_index()
    viz = latest("runs/nusc_zs/alpamayo_viz/*/")
    tok = a.token or sorted(p.stem for p in viz.glob("*.npz"))[0]
    e = next(x for x in idx["samples"] if x["token"] == tok)
    sc = idx["scenes"][e["scene"]]
    fr = np.load(viz / f"{tok}.npz")["frames"][:, 3].transpose(0, 2, 3, 1)
    nat = {c: rgb(dataroot() / sc["cams"][c]["path"][Z.frame_at(sc, c, e["t0"])]) for c in
           ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")}
    road, wide = op_views(sc, e["t0"])
    S.apply()
    fig, ax = plt.subplots(3, 3, figsize=(S.DOUBLE_COLUMN_IN, 3.2), gridspec_kw={"wspace": 0.03, "hspace": 0.18})
    panels = [(fr[0], "(a) Alpamayo cross-left"), (fr[1], "(b) Alpamayo front-wide"), (fr[2], "(c) Alpamayo cross-right"),
              (nat["CAM_FRONT_LEFT"], "(d) CAM_FRONT_LEFT (native)"), (nat["CAM_FRONT"], "(e) CAM_FRONT (native)"),
              (nat["CAM_FRONT_RIGHT"], "(f) CAM_FRONT_RIGHT (native)"), (fr[3], "(g) Alpamayo front-tele"),
              (road, "(h) openpilot road (Y)"), (wide, "(i) openpilot wide (Y)")]
    for axx, (im, t) in zip(ax.ravel(), panels):
        axx.imshow(im, cmap="gray" if im.ndim == 2 else None, aspect="auto")
        axx.set_title(t, fontsize=7, pad=2)
        axx.axis("off")
    S.save(fig, FIG / "nusc-zeroshot-adapter")
    print(tok, e["scene"], sc["location"])


def _op_rear(plan, dev):
    p = np.stack([plan[:, 0], -plan[:, 1]], -1)
    psi = -plan[:, 3].astype(np.float64)
    c, s = np.cos(psi), np.sin(psi)
    return dev + p - np.stack([c * dev[0] - s * dev[1], s * dev[0] + c * dev[1]], -1), psi


def _ticks(ax, xy, yaw, color, every):
    for k in range(every - 1, len(xy), every):
        dx, dy = np.cos(yaw[k]), np.sin(yaw[k])
        ax.annotate("", (-(xy[k, 1] + 1.6 * dy), xy[k, 0] + 1.6 * dx), (-xy[k, 1], xy[k, 0]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6, mutation_scale=5))


def fig_bev(a):
    T_IDXS = 10.0 * (np.arange(33) / 32) ** 2          # openpilot plan times (model.py, no onnx here)
    idx = Z.load_index()
    by = {e["token"]: e for e in idx["samples"]}
    bench = latest("runs/nusc_zs/alpamayo_bench/*/")
    alp = {}
    for line in (bench / "tail_B4.jsonl").read_text().splitlines():
        r = json.loads(line)
        alp[r["token"]] = r
    op = {m: np.load(Z.root("preds") / f"op_{m}_none_tail.npz") for m in ("lebowski", "cinque", "small")}
    op = {m: dict(zip(z["tokens"].tolist(), z["plan"])) for m, z in op.items()}
    toks = [t for t in sorted(alp) if all(t in op[m] for m in op)]
    rng = np.random.default_rng(0)
    toks = [toks[k] for k in sorted(rng.choice(len(toks), min(6, len(toks)), replace=False))]
    S.apply()
    fig, axes = plt.subplots(1, len(toks), figsize=(S.DOUBLE_COLUMN_IN, 2.4))
    for ax, t in zip(np.atleast_1d(axes), toks):
        e = by[t]
        sc = idx["scenes"][e["scene"]]
        tt = np.arange(-15, 65) * 1e5 + e["t0"]
        tt = tt[tt <= sc["pose_t"][-1]]
        xyz, R = Z.ego_at(sc, tt)
        R0t = Z.ego_at(sc, [e["t0"]])[1][0].T
        loc = (xyz - Z.ego_at(sc, [e["t0"]])[0][0]) @ R0t.T
        yaw = Z.yaw_of(R0t @ R)
        past, fut = tt <= e["t0"], tt >= e["t0"]
        ax.plot(-loc[past, 1], loc[past, 0], color="#999999", lw=0.8, label="history 1.5 s")
        ax.plot(-loc[fut, 1], loc[fut, 0], "--", color=C["gt"], lw=0.9, label="ego future (log)")
        _ticks(ax, loc[fut], yaw[fut], C["gt"], 10)
        r = alp[t]
        axyz = np.asarray(r["xyz"])
        ax.plot(-axyz[:, 1], axyz[:, 0], color=C["alp"], lw=1.0, label="Alpamayo 1.5 (6.4 s)")
        _ticks(ax, axyz, np.asarray(r["yaw"]), C["alp"], 10)
        dev = sc["cams"]["CAM_FRONT"]["t_ego"][:2]
        for m in ("lebowski", "cinque", "small"):
            rear, psi = _op_rear(op[m][t], dev)
            k = T_IDXS <= 6.4
            ax.plot(-rear[k, 1], rear[k, 0], color=C[m], lw=0.9, label=f"openpilot {m} (6.4 s)")
            if m == "lebowski":
                sel = [int(np.argmin(np.abs(T_IDXS - s))) for s in (1, 2, 3, 4, 5, 6)]
                _ticks(ax, rear[sel][:, :2] if rear.ndim == 2 else rear[sel], psi[sel], C[m], 1)
        ax.set_aspect("equal", "datalim")
        ax.set_title(f"{e['scene']} ({'SG' if sc['location'].startswith('singapore') else 'BOS'})", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("longitudinal (m)")
    fig.supxlabel("lateral (m, right +)", y=0.17, fontsize=8)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.5, handlelength=1.5)
    fig.subplots_adjust(bottom=0.3, top=0.9, left=0.07, right=0.99, wspace=0.45)
    S.save(fig, FIG / "nusc-zeroshot-bev")
    print(toks)


def fig_pai(a):
    root = data_dir() / "runs/pai_op"
    z = np.load(root / "per_clip.npz")
    prev = sorted(root.glob("frames/*/*_views.jpg"))
    S.apply()
    fig = plt.figure(figsize=(S.DOUBLE_COLUMN_IN, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.22, left=0.01, right=0.99, bottom=0.3, top=0.9)
    ax0 = fig.add_subplot(gs[0])
    if prev:
        ax0.imshow(rgb(prev[0]))
    ax0.set_title("(a) front-wide 120° (left), openpilot road / wide (right, Y)", fontsize=7)
    ax0.axis("off")
    ax1 = fig.add_subplot(gs[1])
    ade = {m: np.linalg.norm(z[m] - z["gt"], axis=-1).mean(1) for m in ("lebowski", "cinque", "small")}
    ade["cv"] = np.linalg.norm(z["cv"] - z["gt"], axis=-1).mean(1)
    order = np.argsort(z["alp_mean"])
    x = np.arange(len(order))
    ax1.plot(x, z["alp_mean"][order], "o-", color=C["alp"], ms=2.5, lw=0.7, label="Alpamayo 1.5 (one sample)")
    for m in ("lebowski", "cinque", "small"):
        ax1.plot(x, ade[m][order], "o", color=C[m], ms=2.2, label=f"openpilot {m}")
    ax1.plot(x, ade["cv"][order], "x", color=C["cv"], ms=2.5, label="constant velocity")
    ax1.set_yscale("log")
    ax1.set_xlabel("clip (sorted by Alpamayo ADE)")
    ax1.set_ylabel("ADE over 6.4 s (m)")
    ax1.set_title("(b) per-clip ADE, n = 31", fontsize=7)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=6.5, handlelength=1.5)
    S.save(fig, FIG / "pai-openpilot")


def fig_results(a):
    import pandas as pd
    df = pd.read_csv(Z.root("score") / "results.csv")
    d = df[df.set == a.set]
    rows = [r for r in ("cv", "cvv", "openpilot small", "openpilot Cinque", "openpilot Lebowski", "Alpamayo 1.5 no-nav",
                        "Alpamayo 1.5 nav") if r in set(d.row)]
    col = {"cv": C["cv"], "cvv": C["cv"], "openpilot small": C["small"], "openpilot Cinque": C["cinque"],
           "openpilot Lebowski": C["lebowski"], "Alpamayo 1.5 nav": C["alp"], "Alpamayo 1.5 no-nav": S.PALETTE["sky_blue"]}
    pub_l2 = {"UniAD": 0.66, "VAD-Base": 0.37, "VAD-Base, no ego": 0.72, "GoStraight": 0.83,
              "Ego-MLP": 0.35}   # BEV-Planner Table 1, unified implementation, n = 5119
    pub_col = {"UniAD": 0.62, "VAD-Base": 0.33, "VAD-Base, no ego": 0.54, "GoStraight": 1.08,
               "Ego-MLP": 0.37}
    S.apply()
    names = rows[::-1] + list(pub_l2)[::-1]          # ours at the bottom, published rows above
    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE_COLUMN_IN, 2.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.17, top=0.97, wspace=0.08)
    for ax, k, pub, xl in ((axes[0], "l2_avg", pub_l2, "L2, mean of 1/2/3 s (m)"),
                           (axes[1], "col_bevp_avg", pub_col, "collision, BEV-Planner def., mean of 1/2/3 s (%)")):
        for i, r in enumerate(rows[::-1]):
            q = d[d.row == r].iloc[0]
            ax.errorbar(q[k], i, xerr=[[q[k] - q[k + "_lo"]], [q[k + "_hi"] - q[k]]], fmt="o", color=col[r],
                        ms=3, capsize=1.5, lw=0.8)
        for j, n in enumerate(list(pub)[::-1]):
            ax.plot(pub[n], len(rows) + j, "D", mfc="white", mec=S.BASELINE, ms=3)
        ax.axhline(len(rows) - 0.5, color="#999999", lw=0.5)
        ax.set_xlabel(xl)
        S.bars(ax)
    axes[0].set_yticks(np.arange(len(names)), rows[::-1] + [f"{n} [pub.]" for n in list(pub_l2)[::-1]])
    axes[0].set_ylim(-0.6, len(names) - 0.4)
    S.save(fig, FIG / "nusc-zeroshot-results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fig", choices=("adapter", "bev", "pai", "results"))
    ap.add_argument("--token", default="")
    ap.add_argument("--set", default="main")
    a = ap.parse_args()
    {"adapter": fig_adapter, "bev": fig_bev, "pai": fig_pai, "results": fig_results}[a.fig](a)
