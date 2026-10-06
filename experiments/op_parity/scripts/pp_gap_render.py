"""op_parity gap page, GIFs: BEV (map, logged agents, P2 / P0 / WA-JEPA plans with their ego boxes over time, logged future) next to the road and
wide frames P2 actually received (protocol W adapter input, from frames.npz). Phase 1 (1.6 s): the 8 context slots t = -1.4 .. 0 s with the t0 scene;
phase 2 (4 s): the future plays at real time (5 fps) with the last input frame held. P2 is seed 0's plan (the seed-mean is not a driveable plan).

  python experiments/op_parity/scripts/pp_gap_render.py [--out DIR]     (box .venv: matplotlib + PIL; reads gap/cases.json, gap/frames.npz)
"""
import argparse
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
GAP = D / "runs/op_parity/gap"
TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
COL = {"GT": "#2e9d4f", "WA": "#1f5fbf", "P2": "#d62728", "P0": "#8c8c8c"}
NAME = {"GT": "logged future", "WA": "WA-JEPA", "P2": "P2 (seed 0)", "P0": "P0 shipped"}
AG = {"vehicle": "#3a9a9a", "pedestrian": "#b04fc0", "bicycle": "#7b5ea7", "red_light": "#ff6a00"}


def interp_pose(P, t):
    """P (8, 3) poses at 0.5 .. 4 s in the t0 rear-axle frame (+ origin at 0) -> (x, y, yaw) at t."""
    ts = np.r_[0, 0.5 * np.arange(1, 9)]
    P = np.vstack([[0, 0, 0], np.asarray(P)])
    return np.array([np.interp(t, ts, P[:, 0]), np.interp(t, ts, P[:, 1]), np.interp(t, ts, np.unwrap(P[:, 2]))])


def box(pose, dims):
    x, y, h = pose
    f, r, w = dims["front"], dims["rear"], dims["width"] / 2
    c = np.array([[f, w], [f, -w], [-r, -w], [-r, w]])
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    return c @ R.T + [x, y]


def to_plot(xy):
    xy = np.asarray(xy)
    return np.stack([-xy[..., 1], xy[..., 0]], -1)                # ego x forward -> up, ego y left -> left


def draw(c, t, frames, slot, fmt, hist):
    fig = plt.figure(figsize=(8.8, 4.9), dpi=80)
    ax = fig.add_axes([0.005, 0.02, 0.40, 0.90])
    tr = {k: np.asarray(c["plans"][v]) for k, v in (("P2", "P2s0"), ("P0", "P0"), ("WA", "WA"))} | {"GT": np.asarray(c["gt"])}
    fmax = max(35.0, max(p[:, 0].max() for p in tr.values()) + 10)
    half = 20.0
    ax.set_xlim(-half, half), ax.set_ylim(-12, fmax if fmax < 70 else 70)
    ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
    ax.set_facecolor("#f4f4f2")
    for p in c["polygons"]:
        col = {"area": "#dcdcd6", "route": "#c6d9ef", "lane": None}[p["kind"]]
        pts = to_plot(p["exterior"])
        if col:
            ax.add_patch(Polygon(pts, closed=True, fc=col, ec="none", zorder=1 if p["kind"] == "area" else 2))
        else:
            ax.add_patch(Polygon(pts, closed=True, fc="none", ec="#bbbbb4", lw=0.4, zorder=2))
    k = int(round(max(t, 0) / 0.1))
    for a in c["agents"].values():
        g = a["polys"][min(k, len(a["polys"]) - 1)]
        if g is None:
            continue
        pts = to_plot(g)
        if a["kind"] == "red_light":
            ax.plot(pts[:, 0], pts[:, 1], color=AG["red_light"], lw=2, zorder=4)
        else:
            ax.add_patch(Polygon(pts, closed=True, fc=AG.get(a["kind"], "#8a8f5a"), ec="k", lw=0.3, alpha=0.9, zorder=4))
    tt = max(t, 0)
    for key in ("GT", "P0", "WA", "P2"):
        P = tr[key]
        xy = to_plot(np.vstack([[0, 0], P[:, :2]]))
        ax.plot(xy[:, 0], xy[:, 1], color=COL[key], lw=1.0 if key == "P0" else 1.8, ls="--" if key in ("GT", "P0") else "-", alpha=0.55, zorder=5)
        ax.plot(xy[1:, 0], xy[1:, 1], ".", color=COL[key], ms=3, zorder=5)
        ax.add_patch(Polygon(to_plot(box(interp_pose(P, tt), c["dims"])), closed=True, fc=COL[key], ec=COL[key], alpha=0.35 if key in ("GT", "P0") else 0.55,
                             lw=1.2, zorder=6))
    for i, key in enumerate(("GT", "WA", "P2", "P0")):
        ax.plot([], [], color=COL[key], lw=2, label=NAME[key], ls="--" if key in ("GT", "P0") else "-")
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.85, borderpad=0.3, handlelength=1.6)
    ax.text(0.02, 0.985, f"t = {t:+.1f} s", transform=ax.transAxes, va="top", fontsize=9, weight="bold")
    # camera panels
    for v, (nm, y0, hh) in enumerate((("road view", 0.50, 0.42), ("wide view", 0.07, 0.42))):
        axi = fig.add_axes([0.415, y0, 0.58, hh])
        axi.imshow(frames[slot, v]), axi.set_xticks([]), axi.set_yticks([])
        axi.set_title(f"{nm}: P2 input, t = {c['slot_t'][slot]:+.1f} s" + ("" if hist else " (held)"), fontsize=8, pad=2)
    fig.text(0.415, 0.012, fmt, fontsize=7.5, va="bottom", family="monospace")
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba")
    plt.close(fig)
    w, h = fig.canvas.get_width_height()
    return Image.frombuffer("RGBA", (w, h), buf.getvalue()).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(GAP / "clips"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cs = json.loads((GAP / "cases.json").read_text())["cases"]
    fr = np.load(GAP / "frames.npz")
    slot_t = fr["slot_t"].tolist()
    for c in cs:
        c["slot_t"] = slot_t
        frames = fr[c["token"]]
        j = TERMS.index(c["metric"])
        s = c["sub"]
        fmt = (f"{c['metric']}  P2 s0/s1 {s['P2s0'][j]:.2f}/{s['P2s1'][j]:.2f}   P0 {s['P0'][j]:.2f}   WA-JEPA {s['WA'][j]:.2f}\n"
               f"log {c['log'][-22:]}  cmd {c['cmd']}  v0 {c['speed']:.1f} m/s")
        imgs = []
        for i, t in enumerate(slot_t):
            imgs.append(draw(c, t, frames, i, fmt, True))
        for t in np.arange(0.2, 4.01, 0.2):
            imgs.append(draw(c, float(t), frames, len(slot_t) - 1, fmt, False))
        q = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in imgs]
        dur = [200] * (len(q) - 1) + [1500]
        f = out / f"{c['id']}.gif"
        q[0].save(f, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True)
        print(f, f"{f.stat().st_size / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
