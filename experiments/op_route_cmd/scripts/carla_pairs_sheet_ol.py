"""Contact sheet of the open-loop-rig CARLA pairs (carla_pairs_render_ol.py): one row per pose = BEV of every exit polyline, the model's road
frame and wide frame at t0 (as the model gets them) with the exit polylines projected on the ground plane (camera x 1.59 m, z 1.86 m, rpy 0)
and the nominal horizon row drawn, and the wide pano (left | centre | right, 90 deg each) with the same polylines. The projection is a geometry check only; the polylines are never part of the image.

  $DATA_DIR/envs/jevdrive/bin/python carla_pairs_sheet.py --plan <poses.pkl> --frames <dir> --out sheet.png [--n 5] [--ids a,b] [--towns T1,T2]
"""
import argparse, os, pickle, sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
from jevdrive import camgeom as G  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
from jevdrive.openpilot import interface as IF  # noqa: E402
CAM_X, CAM_Y, CAM_H = IF.B2D_SPEC_MOUNT
PANO_YAW, PANO_WH, PANO_F = (60.0, 0.0, -60.0), (960, 540), 480.0


def unpack(fr):
    """(6, 128, 256) packed -> (256, 512, 3) RGB uint8 (BT.601 limited range)."""
    Y = np.zeros((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
    cb = cv2.resize(fr[4].astype(np.float32), (512, 256), interpolation=cv2.INTER_LINEAR) - 128
    cr = cv2.resize(fr[5].astype(np.float32), (512, 256), interpolation=cv2.INTER_LINEAR) - 128
    Y = (Y - 16) * (255 / 219)                    # limited-range BT.601 (OpenpilotModel.pack)
    cb, cr = cb * (255 / 224), cr * (255 / 224)
    rgb = np.stack([Y + 1.402 * cr, Y - 0.344136 * cb - 0.714136 * cr, Y + 1.772 * cb], -1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def project(poly, mask, K):
    p = poly[mask]
    xc, yc = p[:, 0] - CAM_X, p[:, 1] - CAM_Y
    ok = xc > 1.5
    return K[0, 2] - K[0, 0] * yc[ok] / xc[ok], K[1, 2] + K[1, 1] * CAM_H / xc[ok]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--ids", default="")
    ap.add_argument("--towns", default="")
    ap.add_argument("--only-3", action="store_true")
    ap.add_argument("--dpi", type=int, default=70)
    ap.add_argument("--lane-change", action="store_true")
    a = ap.parse_args()
    P = [p for p in pickle.load(open(a.plan, "rb")) if (Path(a.frames) / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    if a.towns:
        P = [p for p in P if p["town"] in a.towns.split(",")]
    if a.only_3:
        P = [p for p in P if p["n_exits"] >= 3]
    if a.ids:
        P = [p for ids in a.ids.split(",") for p in P if p["id"] == ids]
    else:   # round robin over towns, one pose per junction, 3-exit roads first (--lane-change: poses that have a lane-change exit)
        if a.lane_change:
            P = [p for p in P if any(e["lane_change"] for e in p["exits"])]
        P.sort(key=lambda p: (-p["n_exits"], p["id"]))
        by = {}
        for p in P:
            by.setdefault(p["town"], []).append(p)
        sel, seen = [], set()
        while any(by.values()) and len(sel) < a.n:
            for t in sorted(by):
                while by[t]:
                    p = by[t].pop(0)
                    if (t, p["junction"]) not in seen:
                        seen.add((t, p["junction"]))
                        sel.append(p)
                        break
        P = sel[: a.n]
    fig, axs = plt.subplots(len(P), 4, figsize=(21, 3.4 * len(P)), gridspec_kw=dict(width_ratios=[0.7, 1.2, 1.2, 3.0]))
    axs = np.atleast_2d(axs)
    for r, p in enumerate(P):
        fr = np.load(Path(a.frames) / "frames" / p["town"] / (p["id"] + ".npz"))["frames"]
        ax = axs[r, 0]
        for k, e in enumerate(p["exits"]):
            c = COLORS[k % len(COLORS)]
            q = e["poly"][e["pmask"]]
            ax.plot(-q[:, 1], q[:, 0], "o-", color=c, ms=3, lw=1.5,
                    label=f"#{e['index_from_left']} {e['cls']} {e['angle']:+.0f}°" + (" (lane chg)" if e["lane_change"] else ""))
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(-80, 80), ax.set_ylim(-10, 155), ax.set_aspect("equal"), ax.grid(alpha=0.3), ax.tick_params(labelsize=6)
        ax.set_title(f"{p['town']} J{p['junction']} d={p['d']:.0f} m {p['profile']} v0={p['v0']:.1f} {p['weather']}", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        for c, (mi, name, k, f) in enumerate([(0, "road", "road", fr[-1][0]), (1, "wide", "wide", fr[-1][1])], 1):
            ax = axs[r, c]
            img = unpack(f)
            ax.imshow(img)
            K = G.OP_K[k]
            ax.axhline(K[1, 2], color="w", ls="--", lw=0.8, alpha=0.8)
            for kk, e in enumerate(p["exits"]):
                u, v = project(e["poly"], e["pmask"], K)
                ax.plot(u, v, "-", color=COLORS[kk % len(COLORS)], lw=2.6, alpha=0.95)
            ax.set_xlim(0, 512), ax.set_ylim(256, 0), ax.axis("off")
            ax.set_title(f"{name} model frame t0 (dashed: horizon row {K[1, 2]:.1f})", fontsize=7)
        ax = axs[r, 3]
        z = np.load(Path(a.frames) / "pano" / p["town"] / (p["id"] + ".npz"))
        tiles = [cv2.cvtColor(cv2.imdecode(z[k], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for k in ("jpeg_l", "jpeg_c", "jpeg_r")]
        ax.imshow(np.concatenate(tiles, 1))
        for kk, e in enumerate(p["exits"]):
            q = e["poly"][e["pmask"]]
            for t, yw in enumerate(PANO_YAW):
                c, s_ = np.cos(np.radians(yw)), np.sin(np.radians(yw))
                xc, yc = q[:, 0] - CAM_X, q[:, 1] - CAM_Y
                xp, yp = xc * c + yc * s_, -xc * s_ + yc * c
                ok = xp > 1.5
                ax.plot(t * PANO_WH[0] + PANO_WH[0] / 2 - PANO_F * yp[ok] / xp[ok], PANO_WH[1] / 2 + PANO_F * CAM_H / xp[ok], "-",
                        color=COLORS[kk % len(COLORS)], lw=2.0)
        ax.set_xlim(0, 3 * PANO_WH[0]), ax.set_ylim(PANO_WH[1], 0), ax.axis("off")
        ax.set_title("wide pano t0: left (+60 deg) | centre | right (-60 deg), 90 deg each, 960 x 540 JPEG", fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi)
    from PIL import Image
    Image.open(a.out).convert("RGB").quantize(192).save(a.out, optimize=True)
    print(a.out, os.path.getsize(a.out) // 1024, "KB", len(P), "rows")


if __name__ == "__main__":
    main()
