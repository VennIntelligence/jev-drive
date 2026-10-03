"""Lane EDGE-FIX review sheet (plans/2026-10-04-roadedge-diagnosis-plan.md, addenda 3-4): openpilot's t0 road and wide model
frames of the current rig (CAM_F0 at its true height) next to the virtual rig, for a few navtest / navhard tokens.
Dashed line: openpilot's horizon row (road 47.6, wide 151.8 of 256); ticks: where flat ground at 10 / 20 / 40 m ahead of the
camera lands for the rig's camera height (row = cy + f h / d). envs/openpilot, CPU.

  edge_vcam_figs.py --rigs 0:0 1.40:0 [1.40:-0.3] --out experiments/skill_pack/figs/roadedge_vcam_frames.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

K = {"road": (910.0, 47.6), "wide": (455.0, 151.8)}


def rgb(p):
    from jevdrive.op_interp import unpack
    Y, U, V = (x.astype(np.float32) for x in unpack(p))
    U, V = (np.repeat(np.repeat(c, 2, 0), 2, 1) - 128 for c in (U, V))
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1) / 255, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigs", nargs="+", default=["0:0", "1.40:0"], help="height:pitch (height 0 = true height)")
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "experiments/skill_pack/figs/roadedge_vcam_frames.png"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import op_lb
    from jevdrive import navsim_zs as Z
    rigs = [tuple(map(float, r.split(":"))) for r in a.rigs]
    ents = []
    for split, k in (("navtest", a.tokens - 1), ("navhard_two_stage", 1)):
        idx = Z.load_index(split, slim=True)
        ents += [idx[i] for i in np.random.default_rng(1).choice(len(idx), k, replace=False)]
    cols = [(v, r) for v in ("road", "wide") for r in rigs]
    fig, ax = plt.subplots(len(ents), len(cols), figsize=(3.2 * len(cols), 1.75 * len(ents) + 0.4), squeeze=False)
    for i, e in enumerate(ents):
        ht = float(e["cams"][-1]["CAM_F0"]["t"][2]) + op_lb.ZG
        frames = {r: op_lb._vcam_job((e, r[0], r[1]))[0][3] for r in rigs}
        for j, (v, r) in enumerate(cols):
            x = ax[i, j]
            x.imshow(rgb(frames[r][0 if v == "road" else 1]))
            f, cy = K[v]
            h = r[0] if r[0] > 0 else ht
            x.axhline(cy, color="#e8b400", ls="--", lw=0.8)
            for d in (10, 20, 40):
                y = cy + f * h / d
                if y < 256:
                    x.plot([0, 18], [y, y], color="#00c8ff", lw=1.2)
                    x.text(20, y, f"{d} m", color="#00c8ff", fontsize=6, va="center")
            x.set_xticks([]), x.set_yticks([])
            if i == 0:
                x.set_title(f"{v}, camera {h:.2f} m" + ("" if r[0] > 0 else " (true)") + (f", pitch {r[1]:+g} deg" if r[1] else ""), fontsize=7)
        ax[i, 0].set_ylabel(e["token"][:8] + ("\nnavhard" if i == len(ents) - 1 else "\nnavtest"), fontsize=7)
    fig.tight_layout(pad=0.3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
