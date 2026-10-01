"""Compact contact sheet of top-s_ego-decile rater frames for the article figure.

Each cell: a panorama of the three front cameras (front_left | front | front_right) above a bird's-eye view
drawn landscape (forward to the right, left up), log in black, the three rated proposals coloured by rank,
dotted boxes the 3 s / 5 s trust regions. Frames are picked from the 20 audited in verify_top_decile.py.

Runs on the box (needs the raw shards):
  uv run python experiments/prediag/archive/make_top_decile_sheet.py --audit $DATA_DIR/runs/waymo_l0/top_decile_audit/<run> --out <dir>
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/prediag/archive", "research",)]
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research"))
import verify_top_decile as v  # noqa: E402
import plot_style  # noqa: E402
from jevdrive import waymo  # noqa: E402

SHORT = {"Cut_ins": "Cut-in", "Foreign Object Debris": "Debris", "Multi-Lane Maneuvers": "Multi-lane",
         "Single-Lane Maneuvers": "Single-lane", "Intersections": "Intersection", "Interections": "Intersection", "Pedestrians": "Pedestrian",
         "Cyclists": "Cyclist", "Others": "Other"}
RANK_COL = [plot_style.PALETTE["green"], plot_style.PALETTE["sky_blue"], plot_style.PALETTE["orange"]]
# positions (1-based) in the 20-frame audit sheet: one per scene type where possible, plus the one inside
DEFAULT_PICK = "1,6,8,10,11,13,15,16,20"


def panorama(d, row, h=260):
    from PIL import Image
    df = d["df"]
    shard = waymo.shard_dir() / str(df.shard.to_numpy()[row])
    fd = os.open(shard, os.O_RDONLY)
    try:
        ims = [Image.open(io.BytesIO(os.pread(fd, int(df[f"{c}_len"].to_numpy()[row]),
                                              int(df[f"{c}_off"].to_numpy()[row])))).convert("RGB")
               for c in ("front_left", "front", "front_right")]
    finally:
        os.close(fd)
    ims = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
    out = Image.new("RGB", (sum(im.width for im in ims), h))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width
    return np.asarray(out)


def sheet(d, t, idx, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    plot_style.apply()
    ncol, nrow = 3, int(np.ceil(len(idx) / 3))
    fig = plt.figure(figsize=(plot_style.DOUBLE_COLUMN_IN, 1.62 * nrow + 0.25))
    outer = fig.add_gridspec(nrow, ncol, left=0.035, right=0.995, top=0.975, bottom=0.075,
                             hspace=0.28, wspace=0.10)
    k3, k5 = (int(round(s * v.FREQ)) - 1 for s in v.HORIZ_S)
    for j, i in enumerate(idx):
        r = t.iloc[i]
        g = outer[j // ncol, j % ncol].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.04)
        ax = fig.add_subplot(g[0])
        ax.imshow(panorama(d, d["rows"][d["pos"][i]]), aspect="auto")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        tag = " (floored)" if r.log_floored else ""
        ax.set_title(f"{SHORT.get(r.cluster, r.cluster)}, log RFS {r.log_rfs:.1f}{tag}", fontsize=7.5, pad=2)

        bev = fig.add_subplot(g[1])
        traj, sc, log_xy = d["rtraj"][i], np.asarray(d["scores"][i]), d["fut"][d["pos"][i]]
        for rank, p in enumerate(np.argsort(-sc, kind="stable")):
            c = RANK_COL[rank]
            bev.plot(traj[p, :, 0], traj[p, :, 1], "-", color=c, lw=0.9)
            for kk, lat, lng in ((k3, r.lat_thr3, r.lng_thr3), (k5, r.lat_thr5, r.lng_thr5)):
                step = traj[p, kk] - (traj[p, kk - 1] if kk else np.zeros(2))
                ang = np.degrees(np.arctan2(step[1], step[0])) if np.any(step) else 0.0
                box = Rectangle((-lng, -lat), 2 * lng, 2 * lat, fc="none", ec=c, lw=0.45, ls=":")
                box.set_transform(Affine2D().rotate_deg(ang).translate(*traj[p, kk]) + bev.transData)
                bev.add_patch(box)
        bev.plot(log_xy[:, 0], log_xy[:, 1], "-", color="black", lw=1.3)
        bev.plot([0], [0], "o", color="black", ms=2.2)
        xs = np.concatenate([traj[:, :, 0].ravel(), log_xy[:, 0]])
        ys = np.concatenate([traj[:, :, 1].ravel(), log_xy[:, 1]])
        bev.set_xlim(min(-2.0, xs.min() - 2), xs.max() * 1.08 + 4)
        half = max(4.0, np.abs(ys).max() + 2)
        bev.set_ylim(-half, half)
        bev.tick_params(labelsize=6, pad=1, length=1.5)
        bev.locator_params(axis="y", nbins=3)
        bev.locator_params(axis="x", nbins=5)
    handles = [Line2D([], [], color="black", lw=1.3, label="logged future")]
    handles += [Line2D([], [], color=c, lw=0.9, label=l)
                for c, l in zip(RANK_COL, ("best-rated proposal", "2nd", "3rd"))]
    handles += [Line2D([], [], color="#555555", lw=0.6, ls=":", label="trust region at 3 s / 5 s")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "top-decile-rater-frames.pdf")
    png = out / "top-decile-rater-frames.png"
    fig.savefig(png, dpi=220)
    plt.close(fig)
    from PIL import Image                     # photos make the PNG large; a 256-colour palette keeps it < 500 KB
    Image.open(png).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT,
                                              dither=Image.Dither.FLOYDSTEINBERG).save(png, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, help="verify_top_decile run dir (rater_frames.csv, sheet_top.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", default=DEFAULT_PICK, help="1-based positions in sheet_top.csv")
    ap.add_argument("--set", default="qwen_front3")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = Path(a.audit)
    d = v.load(a.set, a.seed)
    t = pd.read_csv(audit / "rater_frames.csv")
    top = pd.read_csv(audit / "sheet_top.csv")
    names = top.frame_name.to_numpy()[[int(p) - 1 for p in a.pick.split(",")]]
    pos = {f: k for k, f in enumerate(t.frame_name)}
    idx = [pos[f] for f in names]
    sheet(d, t, idx, out)
    t.iloc[idx].to_csv(out / "sheet_picked.csv", index=False)


if __name__ == "__main__":
    main()
