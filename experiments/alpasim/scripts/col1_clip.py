#!/usr/bin/env python3
"""Lane COL1 clips, nuPlan scenes: one GIF per case of a case list, on the box with AlpaSim's venv (reads only what col1_nuplan.py /
col1_lead.py left under $DATA_DIR/runs/alpasim/col1/: cases.pkl, lead/spec.json, lead/replay/<set>.pkl with model frames, ctrl/ runs).

  col1_clip.py --cases experiments/alpasim/results/collisions/nuplan_clip_cases.csv --out <dir> [--only id,id] [--max-mb 1.4 (MiB, under 1.5 MB)]

Case list columns: id, set (a replay set of col1_lead.py: a driver tag for a collision rollout, `extra` for a control rollout of the
P2H10-F-s0 re-run), scene, obj (actor id whose gap is traced; empty = the nearest object in the straight-ahead corridor), kind, origin, look.
Layout, time-synced, 4 frames per simulated second (decisions are 2 Hz, so the model frames change every other frame):
  left    bird's-eye view, logged heading at t = 0 up: mapped road area (grey, when lead/maps.pkl has the scene), logged path (dashed) and the
          logged ego (outline), other vehicles (grey; the traced object orange), the driven trail and ego box (blue; red from the impact on),
          the plan of the newest decision (green)
  right   what the adapter received at that decision: openpilot's road frame (top) and wide frame (bottom) of the newest policy slot,
          both sampled from CAM_F0 by the driver (replayed offline through the real driver class from the logged messages)
  bottom  ego speed against the log's; the simulator's bumper gap to the traced object (label) with the shipped lead head's distance where
          lead prob >= 0.5; lead prob of the shipped policy (P0) and of the served checkpoint (FT). Red line = first at-fault contact.
Map, objects and the logged path are privileged: shown for the reader, never fed to the driver.
"""
import argparse
import csv
import glob
import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1_lib as L  # noqa: E402
import col1_lead as LE  # noqa: E402

O = LE.O
LD = LE.LD
BLUE, RED, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#8a8a8a"


def rgb(p):
    """(6, 128, 256) packed YUV420 (BT.601 full range) -> (256, 512, 3) uint8 (c1_review.rgb)."""
    Y = np.empty((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = p[0], p[1], p[2], p[3]
    U, V = (np.repeat(np.repeat(p[k].astype(np.float32), 2, 0), 2, 1) - 128 for k in (4, 5))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def record(st, scene, spec, cases):
    """The c1_extract record of a rollout: cases.pkl for a collision case, else read from the control re-run that holds the scene."""
    if (st, scene) in cases:
        return cases[st, scene]
    import c1_extract as X
    runs = json.loads((O / "ctrl/manifest.json").read_text())["P2H10-F-s0"]
    run = next(r for r in runs if glob.glob(f"{r}/rollouts/{scene}/*/rollout.asl"))
    o = X.one_log((run, scene))[1]
    o["summary"] = next(r for r in json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"] if r["clipgt_id"] == scene)
    return o


def render(case, o, recs, rd, out, max_mb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    from PIL import Image
    e, g = L.ego(o), L.gt(o)
    t_ev = L.first_event(o, "collision_at_fault") if o["summary"]["score_metrics"].get("collision_at_fault") else None
    aid = case["obj"] or None
    t_end = min(e[-1, 0], recs[-1]["now"] + 0.5e6)
    ts = np.arange(0, t_end + 1, 0.25e6)
    fine = np.arange(0, t_end + 1, 1e5)
    lab = [LE.label(L, o, t, aid) for t in fine]
    gap = np.array([np.nan if x[0] is None else x[0] for x in lab])
    v_e = np.array([x[2] for x in lab])
    v_g = np.array([L.speed_at(g, t) for t in fine])
    sig = {s: np.array([LE.signals(r, s) for r in recs]) for s in ("P0", "FT")}
    now = np.array([r["now"] for r in recs])
    # view: logged heading at t = 0 up, centred on the impact (or the middle of the driven path)
    h0 = g[0, 3]
    Rv = L.rot(np.pi / 2 - h0)
    ctr = L.interp_pose(e, t_ev if t_ev else e[len(e) // 2, 0])[:2]
    tf = lambda xy: (np.atleast_2d(xy) - ctr) @ Rv.T  # noqa: E731
    W = 20.0
    road = None
    if rd is not None:
        from shapely.geometry import box as rect
        from shapely import affinity
        clipb = rect(ctr[0] - 2 * W, ctr[1] - 2 * W, ctr[0] + 2 * W, ctr[1] + 2 * W)
        road = rd["area"].union(rd["lane"]).intersection(clipb)
    frames = []
    for t in ts:
        k = max(int(np.searchsorted(now, t, side="right")) - 1, 0)
        fig = plt.figure(figsize=(9.6, 6.0), dpi=100)
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.78], left=0.04, right=0.985, top=0.94, bottom=0.075, hspace=0.32, wspace=0.55)
        ax = fig.add_subplot(gs[0:2, 0:3])
        if road is not None and not road.is_empty:
            for geom in getattr(road, "geoms", [road]):
                if geom.geom_type == "Polygon":
                    ax.add_patch(MP(tf(np.array(geom.exterior.coords)), fc="#e9e9e9", ec="#c8c8c8", lw=0.5, zorder=0))
                    for h in geom.interiors:
                        ax.add_patch(MP(tf(np.array(h.coords)), fc="white", ec="#c8c8c8", lw=0.5, zorder=0.1))
        ax.plot(*tf(g[:, 1:3]).T, "--", color=GREY, lw=1.2, zorder=1)
        ax.add_patch(MP(tf(np.array(L.box(*L.interp_pose(g, t)).exterior.coords)), fc="none", ec=GREY, lw=1.0, ls="--", zorder=2))
        for a, tr in o["actors"].items():
            if a == "EGO" or not (tr[0, 0] <= t <= tr[-1, 0]):
                continue
            p = L.interp_pose(tr, t)
            if np.hypot(*(p[:2] - ctr)) > 1.6 * W:
                continue
            ax.add_patch(MP(tf(np.array(L.box(*p, *o["size"][a][:2]).exterior.coords)), fc=ORANGE if a == aid else "#b5b5b5", ec="k", lw=0.5, zorder=3))
        hit = t_ev is not None and t >= t_ev
        pe = L.interp_pose(e, t)
        tr_ = e[e[:, 0] <= t]
        ax.plot(*tf(tr_[:, 1:3]).T, color=BLUE, lw=1.5, zorder=4)
        pl = L.plan_c(o, k) if "traj" in o["drive"][k] else None
        if pl is not None:
            ax.plot(*tf(pl[:, 1:3]).T, color=GREEN, lw=2.0, zorder=5)
        ax.add_patch(MP(tf(np.array(L.box(*pe).exterior.coords)), fc=RED if hit else BLUE, ec="k", lw=0.6, alpha=0.9, zorder=6))
        ax.set_xlim(-W, W), ax.set_ylim(-W, W), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{case['id']}  {case['set']}  t = {t * 1e-6:.2f} s" + ("   CONTACT" if hit else ""), fontsize=9, color=RED if hit else "k", loc="left")
        ax.text(0.02, 0.02, "blue ego / green plan / dashed log / orange traced object", transform=ax.transAxes, fontsize=7, color="#333", bbox=dict(fc="white", ec="none", alpha=0.7, pad=1), zorder=9)
        fr = recs[k].get("frame")
        for j, name in enumerate(("road frame (model input)", "wide frame (model input)")):
            a2 = fig.add_subplot(gs[j, 3:6])
            if fr is not None:
                a2.imshow(rgb(fr[j]))
            a2.set_xticks([]), a2.set_yticks([])
            a2.set_title(f"{name}, decision {k} at {now[k] * 1e-6:.2f} s, command {'LSRU'[recs[k]['cmd']]}", fontsize=8, loc="left", pad=2)
            if hit:
                for sp in a2.spines.values():
                    sp.set_color(RED), sp.set_linewidth(2)
        tx = fine * 1e-6
        panels = [("speed m/s", [(tx, v_e, BLUE, "-", "ego"), (tx, v_g, GREY, "--", "log")]),
                  ("gap m", [(tx, gap, ORANGE, "-", "simulator gap")]), ("lead prob", [])]
        for j, (yl, lines) in enumerate(panels):
            a3 = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
            for x, y, c, ls, lb in lines:
                a3.plot(x, y, ls, color=c, lw=1.4, label=lb)
            if j == 1:
                m = sig["P0"][:, 0] >= 0.5
                a3.plot(now[m] * 1e-6, sig["P0"][m, 1], "o", color="k", ms=3.5, label="lead head d (p >= 0.5)")
                top = np.nanmax(np.r_[gap[np.isfinite(gap)], 10.0]) if np.isfinite(gap).any() else 40.0
                a3.set_ylim(-2, min(max(top * 1.15, 10), 80))
            if j == 2:
                a3.step(now * 1e-6, sig["P0"][:, 0], where="post", color="k", lw=1.4, label="P0 (shipped)")
                a3.step(now * 1e-6, sig["FT"][:, 0], where="post", color=GREEN, lw=1.2, label="FT (served)")
                a3.axhline(0.5, color="#bbb", lw=0.6), a3.set_ylim(-0.03, 1.03)
            a3.axvline(t * 1e-6, color="k", lw=0.8)
            if t_ev is not None:
                a3.axvline(t_ev * 1e-6, color=RED, lw=1.2)
            a3.set_xlim(0, t_end * 1e-6), a3.set_ylabel(yl, fontsize=8, labelpad=1), a3.tick_params(labelsize=7, pad=1)
            a3.set_xlabel("s", fontsize=7, labelpad=0), a3.legend(fontsize=6, loc="lower left" if j == 2 else "upper left", frameon=False, handlelength=1.2)
        b = io.BytesIO()
        fig.savefig(b, format="png")
        plt.close(fig)
        frames.append(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    f = Path(out) / f"{case['id']}.gif"
    dur = [250] * len(frames)
    dur[-1] = 1500
    if t_ev is not None:                                    # hold the first contact frame
        dur[int(np.argmax(ts >= t_ev))] = 1200
    for scale, ncol in ((1.0, 128), (1.0, 96), (0.9, 64), (0.8, 64), (0.7, 48), (0.6, 32)):
        fs = [x.resize((int(x.width * scale), int(x.height * scale)), Image.LANCZOS) if scale < 1 else x for x in frames]
        pick = sorted({0, len(fs) // 3, 2 * len(fs) // 3, len(fs) - 1})        # one palette for the clip, from frames before and after the contact
        mos = Image.new("RGB", (fs[0].width, fs[0].height * len(pick)))
        for i, j in enumerate(pick):
            mos.paste(fs[j], (0, i * fs[0].height))
        pal = mos.quantize(ncol, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
        q = [x.quantize(palette=pal, dither=Image.Dither.NONE) for x in fs]
        q[0].save(f, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
        if f.stat().st_size <= max_mb * 2**20:
            break
    print(f"{f} {f.stat().st_size / 2**20:.2f} MB, {len(frames)} frames, scale {scale}, {ncol} colours, impact {None if t_ev is None else round(t_ev * 1e-6, 2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True), ap.add_argument("--out", required=True), ap.add_argument("--only", default=""), ap.add_argument("--max-mb", type=float, default=1.4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec, cases = json.loads((LD / "spec.json").read_text()), pickle.load(open(O / "cases.pkl", "rb"))
    maps = pickle.load(open(LD / "maps.pkl", "rb")) if (LD / "maps.pkl").exists() else {}
    rep = {}
    for c in csv.DictReader(open(a.cases)):
        if a.only and c["id"] not in a.only.split(","):
            continue
        if c["set"] not in rep:
            rep[c["set"]] = pickle.load(open(LD / "replay" / f"{c['set']}.pkl", "rb"))
        recs = rep[c["set"]][c["scene"]]
        assert "frame" in recs[0], f"{c['id']}: no model frames in the replay of {c['set']} (list the scene in `col1_lead.py msgs --frames`)"
        o = record(c["set"], c["scene"], spec, cases)
        render(c, o, recs, L.road(maps[c["scene"]]) if c["scene"] in maps else None, a.out, a.max_mb)


if __name__ == "__main__":
    main()
