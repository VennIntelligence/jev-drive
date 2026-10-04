"""Contact sheet of the reprojection arm in the model's own view (CPU; env with cv2 + matplotlib).

For four anchors (one per category) and both model views (road, wide): the anchor frame, the real frame G at +1.0 s, the nominal reprojection R of the anchor
to the real pose at +1.0 s, |G - R|, then R under ego yaw +2 / -2 deg and a 0.5 m lateral offset (left; the 3-step swerve of wm_common.lat_profile).
The world model has no image output: its prediction is a latent only, so it has no column here.
  python experiments/world_model/scripts/wm_sheet.py <out.png> [seg_prefix:i ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
import wm_common as C  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

STEP = 5


def rgb(p):
    """packed (6, 128, 256) -> (256, 512, 3) uint8 (JPEG YCbCr, chroma replicated)."""
    Y, U, V = (x.astype(np.float32) for x in I.unpack(p))
    U = U.repeat(2, 0).repeat(2, 1) - 128
    V = V.repeat(2, 0).repeat(2, 1) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = sys.argv[1]
    anchors = json.load(open(C.RUN / "anchors.json"))
    want = sys.argv[2:]
    if want:
        pick = [a for a in anchors if f"{a['seg'][:12]}:{a['i']}" in want]
    else:
        pick = []
        for cat in ("launch", "cruise", "curve", "natdev"):
            pick += [a for a in anchors if a["cat"] == cat][:1]
    cols = ["anchor (t0)", "G: real +1.0 s", "R nominal +1.0 s", "|G - R|", "R yaw +2 deg", "R yaw -2 deg", "R offset +0.5 m"]
    fig, ax = plt.subplots(2 * len(pick), len(cols), figsize=(2.6 * len(cols), 1.45 * 2 * len(pick)))
    for r, an in enumerate(pick):
        z = np.load(C.RUN / "prep" / f"{an['seg']}.npz")
        op, pose, v, cam = z["op"], z["pose"], z["v"], z["cam"]
        i = an["i"]
        P, sv = C.nominal(pose[i: i + C.K + 1])
        arms = C.arms(P, sv)
        warp = lambda Q: I.warp_frame(op[i], cam, Q[STEP], np.zeros(3))  # noqa: E731
        frames = [op[i], op[i + STEP], warp(arms["nom"]), None, warp(arms["yaw+2"]), warp(arms["yaw-2"]), warp(arms["lat+0.5"]) if "lat+0.5" in arms else None]
        for view, name in enumerate(("road", "wide")):
            for c, f in enumerate(frames):
                a_ = ax[2 * r + view, c]
                a_.set_xticks([]), a_.set_yticks([])
                for s in a_.spines.values():
                    s.set_visible(False)
                if c == 3:
                    img = np.abs(rgb(frames[1][view]).astype(int) - rgb(frames[2][view]).astype(int)).clip(0, 255).astype(np.uint8)
                elif f is None:
                    a_.text(0.5, 0.5, "n/a (v too low)", ha="center", va="center", transform=a_.transAxes, fontsize=7)
                    continue
                else:
                    img = rgb(f[view])
                a_.imshow(img)
                if r == 0 and view == 0:
                    a_.set_title(cols[c], fontsize=8)
                if c == 0:
                    a_.set_ylabel(f"{an['cat']} {name}\nv={v[i]:.1f} m/s", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=72)
    print("saved", out)


if __name__ == "__main__":
    main()
