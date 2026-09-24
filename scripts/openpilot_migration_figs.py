"""Figures for todos/2026-09-24-zeroshot-exam/openpilot-migration.md (run on the Mac, research/plot_style.py).

  smoke    RUN/attempts/<route>/1/{plans,ticks}.jsonl of the pre-registered smoke -> start-of-route timeline
  rigs     rows.jsonl of scripts/openpilot_rig_study.py eval -> per-factor error bars (+ results CSV)
  sheet    sheet_*.npz of scripts/openpilot_rig_study.py sheet -> model-frame thumbnails per rig
  frames   comma1M model frame vs CARLA model frames (JPEG dumps) side by side
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import plot_style as ps  # noqa: E402

FIGS = Path(__file__).resolve().parents[1] / "research" / "figs"
CAM_X = 1.779


def fig_smoke(a):
    import matplotlib.pyplot as plt
    ps.apply()
    routes = a.routes
    fig, axes = plt.subplots(3, len(routes), figsize=(ps.DOUBLE_COLUMN_IN, 3.6), sharex="col")
    for j, r in enumerate(routes):
        d = a.run / "attempts" / r / "1"
        P = [json.loads(l) for l in open(d / "plans.jsonl")]
        T = [json.loads(l) for l in open(d / "ticks.jsonl")]
        t0 = T[0]["t"]
        tp = np.array([p["t"] for p in P]) - t0
        tt = np.array([t["t"] for t in T]) - t0
        m, mt = tp < a.tmax, tt < a.tmax
        x5 = np.array([p["path"][19][0] for p in P]) - CAM_X      # camera-frame plan x at 5 s
        y2 = np.array([p["path"][7][1] for p in P])
        ax = axes[0, j]
        ax.plot(tp[m], x5[m], color=ps.PREDICTION, label="plan $x$ at 5 s (m)")
        ax.plot(tt[mt], [t["v"] for t, k in zip(T, mt) if k], color=ps.PALETTE["vermillion"], label="speed (m/s)")
        ax.set_title(f"route {r}", fontsize=8)
        ax.set_ylim(-5, 40)
        ax = axes[1, j]
        ax.plot(tt[mt], [t["throttle"] for t, k in zip(T, mt) if k], color=ps.PALETTE["green"], label="throttle")
        ax.plot(tt[mt], [-t["brake"] for t, k in zip(T, mt) if k], color=ps.BASELINE, label="$-$brake")
        ax = axes[2, j]
        ax.plot(tp[m], y2[m], color=ps.PREDICTION, label="plan $y$ at 2 s (m, left +)")
        ax.plot(tt[mt], [-5 * t["steer"] for t, k in zip(T, mt) if k], color=ps.PALETTE["orange"],
                label="$-5\\times$steer")
        axes[2, j].set_xlabel("time (s)")
    for i in range(3):
        axes[i, -1].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    print(ps.save(fig, FIGS / "openpilot-migration-smoke-start"))


def load_rows(path):
    return [json.loads(l) for l in open(path)]


GROUPS = [
    ("reference", ["native"]),
    ("single camera", ["wide-only", "single-pinhole60", "single-pinhole90", "single-pinhole120", "fisheye180",
                       "fisheye180-as-pinhole", "fisheye180-720p"]),
    ("dataset rigs", ["nuplan-f0", "nuplan-l0f0r0", "waymo-front3", "b2d-nuscenes3", "tfpp-110"]),
    ("mount", ["pitch+2", "pitch-2", "pitch+5", "yaw+2", "height0.92", "height1.43", "height1.60", "height1.81",
               "height2.20"]),
    ("image", ["res0.5", "res0.25", "blur1.5", "blur3", "noise6", "dark0.5", "bright1.6", "lowcontrast", "gray",
               "fullrange", "squeeze", "jpeg75", "jpeg30"]),
    ("timing", ["t-10hz", "t-wide-lag50ms", "t-jitter", "t-cold-1.5s", "t-navsim-2hz-1.5s"]),
    ("combined", ["wod-like", "navsim-like"]),
]


def pooled(rows, keys=("lat@2s", "lon@2s", "lat@4s", "lon@4s", "curv_mae", "curv_r")):
    """Frame-weighted mean over segments per (variant, model); per-segment values kept for a paired CI."""
    out = {}
    for r in rows:
        out.setdefault((r["variant"], r["model"]), []).append(r)
    res = {}
    for k, rs in out.items():
        n = np.array([r["n"] for r in rs], float)
        res[k] = {m: float(np.average([r.get(m, np.nan) for r in rs], weights=n)) for m in keys}
        res[k]["n"] = int(n.sum())
        res[k]["segs"] = {r["seg"]: r for r in rs}
    return res


def fig_rigs(a):
    import csv
    import matplotlib.pyplot as plt
    ps.apply()
    rows = load_rows(a.rows)
    P = pooled(rows)
    models = ["small", "cinque", "lebowski"]
    colors = {"small": ps.PALETTE["sky_blue"], "cinque": ps.PALETTE["blue"], "lebowski": ps.PALETTE["vermillion"]}
    # results table: every variant, every model, pooled, plus paired delta vs native (same segments / frames)
    table = []
    for g, vs in GROUPS:
        for v in vs:
            for m in models:
                if (v, m) not in P:
                    continue
                p = P[(v, m)]
                ref = P[("native", m)]
                segs = [s for s in p["segs"] if s in ref["segs"]]
                if v.startswith("t-cold") or v.startswith("t-navsim"):
                    ref_row = None
                else:
                    ref_row = ref
                d = {}
                for key in ("lat@2s", "lon@2s", "lat@4s"):
                    per = np.array([p["segs"][s][key] - ref["segs"][s][key] for s in segs])
                    d[key] = float(p[key] - ref_row[key]) if ref_row else np.nan
                    d[key + "_segmin"], d[key + "_segmax"] = (float(per.min()), float(per.max())) if ref_row else (np.nan, np.nan)
                table.append(dict(group=g, variant=v, model=m, n=p["n"], **{k: round(p[k], 4) for k in
                                  ("lat@2s", "lon@2s", "lat@4s", "lon@4s", "curv_mae", "curv_r")},
                                  **{f"d_{k}": round(x, 4) for k, x in d.items()}))
    with open(a.csv, "w") as fh:
        w = csv.DictWriter(fh, list(table[0]))
        w.writeheader()
        w.writerows(table)
    variants = [v for _, vs in GROUPS for v in vs if (v, "cinque") in P and not v.startswith("t-cold")
                and not v.startswith("t-navsim")]
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 4.6), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, key, lab in ((axes[0], "lat@2s", "lateral error at 2 s (m)"), (axes[1], "lon@2s", "longitudinal error at 2 s (m)")):
        for k, m in enumerate(models):
            vals = [P[(v, m)][key] if (v, m) in P else np.nan for v in variants]
            ax.scatter(vals, y + (k - 1) * 0.22, s=9, color=colors[m], label=m if key == "lat@2s" else None, zorder=3)
        for m in models:
            ax.axvline(P[("native", m)][key], color=colors[m], lw=0.5, ls="--", zorder=1)
        ax.set_xlabel(lab)
        ax.set_xscale("log")
        ps.bars(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(variants, fontsize=6.5)
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    print(ps.save(fig, FIGS / "openpilot-migration-rigs"))


def fig_sheet(a):
    import matplotlib.pyplot as plt
    ps.apply()
    z = np.load(a.sheet)
    names = [v for v in a.variants if v in z.files]
    n = len(names)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(ps.DOUBLE_COLUMN_IN, 1.05 * rows))
    for ax, v in zip(axes.ravel(), names):
        ax.imshow(z[v], cmap="gray", vmin=0, vmax=255)
        ax.set_title(v, fontsize=7, pad=1.5)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    print(ps.save(fig, FIGS / "openpilot-migration-sheet"))


def fig_frames(a):
    """Top: comma1M road | wide model frame (luma); below: CARLA dumps (road | wide, already luma JPEG)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    ps.apply()
    z = np.load(a.sheet)
    ims = [("comma1M (real)", z["native"])]
    for p in a.carla:
        im = np.asarray(Image.open(p).convert("L"))[:256]
        ims.append((f"CARLA {Path(p).stem}", im))
    fig, axes = plt.subplots(len(ims), 1, figsize=(ps.SINGLE_COLUMN_IN, 0.9 * len(ims)))
    for ax, (t, im) in zip(axes, ims):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.axhline(47.6, color=ps.PALETTE["orange"], lw=0.4)
        ax.set_title(t, fontsize=7, pad=1.5)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    print(ps.save(fig, FIGS / "openpilot-migration-frames"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("smoke")
    p.add_argument("run", type=Path)
    p.add_argument("--routes", nargs="+", default=["24211", "1711", "3564"])
    p.add_argument("--tmax", type=float, default=12.0)
    p = sp.add_parser("rigs")
    p.add_argument("rows", type=Path)
    p.add_argument("--csv", type=Path, required=True)
    p = sp.add_parser("sheet")
    p.add_argument("sheet", type=Path)
    p.add_argument("--variants", nargs="+", required=True)
    p = sp.add_parser("frames")
    p.add_argument("sheet", type=Path)
    p.add_argument("--carla", nargs="+", required=True)
    a = ap.parse_args()
    {"smoke": fig_smoke, "rigs": fig_rigs, "sheet": fig_sheet, "frames": fig_frames}[a.cmd](a)
