"""Review sheet for real vs render (HUGSIM nuScenes): per row one frame, columns = openpilot's road model frame from the real sequence and
from the 3DGS render at the same pose, and the source CAM_FRONT pair. On both model frames: lane lines, road edges and the plan of the
real run (green) and of the render run (magenta), projected with the model intrinsics (lines at their own z, plan on the model road z).
Project venv, CPU: .venv/bin/python experiments/leaderboard_audit/scripts/rvr_figs.py  -> $DATA_DIR/runs/real_vs_render/real_vs_render.png"""
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jevdrive.openpilot.frames import MEDMODEL_K, VIEW_FROM_DEVICE  # noqa: E402
import rvr_analyze as A  # noqa: E402
import rvr_hugsim_op as H  # noqa: E402

RUN = A.RUN
X_IDXS, T_IDXS = A.X_IDXS, A.T_IDXS


def unpack_y(p):
    Y = np.empty((256, 512), np.uint8)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = p[0], p[1], p[2], p[3]
    return Y


def proj(pts):
    v = pts @ VIEW_FROM_DEVICE.T
    ok = v[:, 2] > 1.0
    uv = v[:, :2] / v[:, 2:3]
    uv = uv @ MEDMODEL_K[:2, :2].T + MEDMODEL_K[:2, 2]
    return uv[ok]


def draw(ax, row, hs, color, ls):
    n = 1
    ll = row[hs["lane_lines"]:hs["lane_lines"] + 264].reshape(4, 33, 2)
    lp = A.sig(row[hs["lane_lines_prob"]:hs["lane_lines_prob"] + 8][1::2])
    re = row[hs["road_edges"]:hs["road_edges"] + 132].reshape(2, 33, 2)
    pl = row[hs["plan"]:hs["plan"] + 495].reshape(33, 15)
    m = X_IDXS <= 60
    for j in range(4):
        if lp[j] > 0.3:
            uv = proj(np.stack([X_IDXS[m], ll[j, m, 0], ll[j, m, 1]], -1))
            ax.plot(uv[:, 0], uv[:, 1], color=color, ls=ls, lw=1.6 if j in (1, 2) else 0.8, alpha=0.9)
    for j in range(2):
        uv = proj(np.stack([X_IDXS[m], re[j, m, 0], re[j, m, 1]], -1))
        ax.plot(uv[:, 0], uv[:, 1], color=color, ls=ls, lw=2.4, alpha=0.6)
    z = np.nanmean(ll[1:3, 5:15, 1])
    t = T_IDXS <= 6
    uv = proj(np.stack([pl[t, 0], pl[t, 1], np.full(t.sum(), z)], -1))
    ax.plot(uv[:, 0], uv[:, 1], color=color, ls=ls, lw=3, alpha=0.8, marker=".", ms=3)
    return n


def main():
    S = A.load()
    z = {s: np.load(RUN / "hugsim" / s / "stream.npz") for s in S}
    hs = json.loads(str(next(iter(z.values()))["info"]))["heads_slices"]
    # rows: the frames with the largest, a median and a small road-edge gap (frames >= 3 s, moving > 2 m/s)
    cand = []
    for s, x in S.items():
        for k in range(A.FIRST, len(x["v"]), 6):
            if x["v"][k] > 2:
                cand.append((float(A.absdiff(x["real"]["edge_y10"][k:k + 1], x["render"]["edge_y10"][k:k + 1])[0]), s, k))
    cand.sort()
    by_scene = {}
    for c in cand[::-1]:
        by_scene.setdefault(c[1], c)
    top = sorted(by_scene.values())[::-1]
    picks = [top[0], top[len(top) // 4], top[len(top) // 2], cand[len(cand) // 10]]
    op = H.adapter()
    fig, axes = plt.subplots(len(picks), 3, figsize=(17, 3.3 * len(picks)), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for r, (gap, s, k) in enumerate(picks):
        arr = H.load(s)
        for c, src in enumerate(("real", "render")):
            rgb = {cam: np.asarray(arr[src][k, j]) for j, cam in enumerate(H.CAMS)}
            Y = unpack_y(op.pack(rgb)[0])
            ax = axes[r, c]
            ax.imshow(Y, cmap="gray", vmin=0, vmax=255)
            draw(ax, z[s]["real"][k], hs, "#18c25a", "-")
            draw(ax, z[s]["render"][k], hs, "#ff2bd6", "--")
            ax.set_xlim(0, 512), ax.set_ylim(256, 0), ax.axis("off")
            ax.set_title(f"{s} frame {k} ({z[s]['v'][k]:.1f} m/s): openpilot road frame from the {src.upper()} sequence" if c == 0 else
                         f"same pose, 3DGS RENDER; edge gap at 10 m {gap:.2f} m", fontsize=9)
        both = np.concatenate([np.asarray(arr["real"][k, 0]), np.asarray(arr["render"][k, 0])], 0)
        axes[r, 2].imshow(both), axes[r, 2].axis("off")
        axes[r, 2].set_title("CAM_FRONT 800x450: real (top) / render (bottom)", fontsize=9)
    fig.suptitle("Real vs 3DGS render at the same pose, HUGSIM nuScenes scenes. Green solid = openpilot outputs on the real sequence, "
                 "magenta dashed = on the render (lane lines thin/thick, road edges wide, plan dotted)", fontsize=10)
    fig.tight_layout()
    out = RUN / "real_vs_render.png"
    fig.savefig(out, dpi=110)
    print(out, picks)


if __name__ == "__main__":
    main()
