"""Contact sheet of CARLA counterfactual pairs: one row per pose = BEV of every exit polyline, the model's road frame and wide frame at t0
with the exit polylines projected on the ground plane (camera 1.22 m, calibration rpy 0) and the nominal horizon row drawn, and the oldest
history wide frame. The projection is a geometry check only; the polylines are never part of the image.

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
CAM_X, CAM_Y, CAM_H = 1.519, 0.026, 1.22


def unpack(fr):
    """(6, 128, 256) packed -> (256, 512, 3) RGB uint8 (libjpeg full-range YCbCr)."""
    Y = np.zeros((256, 512), np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = fr[0], fr[1], fr[2], fr[3]
    cb = cv2.resize(fr[4].astype(np.float32), (512, 256), interpolation=cv2.INTER_LINEAR) - 128
    cr = cv2.resize(fr[5].astype(np.float32), (512, 256), interpolation=cv2.INTER_LINEAR) - 128
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
    a = ap.parse_args()
    P = [p for p in pickle.load(open(a.plan, "rb")) if (Path(a.frames) / "frames" / p["town"] / (p["id"] + ".npz")).exists()]
    if a.towns:
        P = [p for p in P if p["town"] in a.towns.split(",")]
    if a.only_3:
        P = [p for p in P if p["n_exits"] >= 3]
    if a.ids:
        P = [p for ids in a.ids.split(",") for p in P if p["id"] == ids]
    else:   # spread over towns, 3-exit roads first
        P.sort(key=lambda p: (-p["n_exits"], p["town"], p["id"]))
        seen, sel = set(), []
        for p in P:
            if (p["town"], p["junction"]) not in seen:
                seen.add((p["town"], p["junction"]))
                sel.append(p)
        P = sel[: a.n]
    fig, axs = plt.subplots(len(P), 4, figsize=(19, 3.6 * len(P)), gridspec_kw=dict(width_ratios=[0.7, 1.5, 1.5, 1.5]))
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
                ax.plot(u, v, "-", color=COLORS[kk % len(COLORS)], lw=2.0, alpha=0.9)
            ax.set_xlim(0, 512), ax.set_ylim(256, 0), ax.axis("off")
            ax.set_title(f"{name} model frame t0 (dashed = nominal horizon row {K[1, 2]:.1f}); polylines = check only", fontsize=7)
        ax = axs[r, 3]
        ax.imshow(unpack(fr[0][1]))
        ax.axis("off")
        ax.set_title("wide, oldest history frame (t0 - 1.8 s)", fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi)
    from PIL import Image
    Image.open(a.out).convert("RGB").quantize(128).save(a.out, optimize=True)
    print(a.out, os.path.getsize(a.out) // 1024, "KB", len(P), "rows")


if __name__ == "__main__":
    main()
