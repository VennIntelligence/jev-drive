"""The two openpilot model inputs (road f = 910 px, wide f = 455 px) next to the raw CAM_F0 frame, for case 01 and other wrong-direction
scenes. openpilot env (numpy, PIL, matplotlib). tokens: a file of tokens (first = case 01). The inputs are the cached t0 key frames of the
native run (the frames cache the model was fed). Rectangles on the raw frame: the field of view each input covers (rays through the dataset
calibration, undistorted pinhole positions of the input's corners).
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import op_lb as B  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def to_rgb(p):
    Y, U, V = (x.astype(np.float32) for x in I.unpack(p))
    U = U.repeat(2, 0).repeat(2, 1) - 128
    V = V.repeat(2, 0).repeat(2, 1) - 128
    return np.clip(np.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -1), 0, 255).astype(np.uint8)


def main():
    toks = Path(sys.argv[1]).read_text().split()
    out = Path(sys.argv[2])
    mt = B.meta("lb_navhard")
    row = {t: i for i, t in enumerate(mt["names"])}
    keys = B.Keys("lb_navhard")
    idx = Z.load_index("navhard_two_stage", slim=True)
    byt = {e["token"]: e for e in idx}
    fig, axs = plt.subplots(len(toks), 3, figsize=(16, 3.0 * len(toks)), gridspec_kw=dict(width_ratios=[1.78, 2, 2]))
    for r, t in enumerate(toks):
        e = byt[t]
        cam = e["cams"][-1]["CAM_F0"]
        raw = np.asarray(Image.open(cam["path"]).convert("RGB"))
        fr = keys[row[t]][3]                       # t0 key frame (2, 6, 128, 256)
        a = axs[r, 0]
        a.imshow(raw)
        K = np.asarray(cam["K"], float)
        for f, col, nm in ((910.0, "#ffcc00", "road 31 deg"), (455.0, "#00e5ff", "wide 59 deg")):
            hw, hh = 256 / f, (256 - 47.6) / f if f == 910.0 else 128 / f      # tan of the half angles (road: horizon at row 47.6)
            hh_top = 47.6 / f if f == 910.0 else (0.5 * (256 + 47.6)) / f
            hh_bot = (256 - 47.6) / f if f == 910.0 else (256 - 0.5 * (256 + 47.6)) / f
            xs = [-hw, hw, hw, -hw, -hw]
            ys = [-hh_top, -hh_top, hh_bot, hh_bot, -hh_top]
            u = K[0, 0] * np.array(xs) + K[0, 2]
            v = K[1, 1] * np.array(ys) + K[1, 2]
            a.plot(u, v, color=col, lw=2, label=nm)
        a.set_title(f"{t[:8]} {e['map'].split('-')[-1]} cmd {['left', 'straight', 'right', '?'][int(np.argmax(e['cmd'][-1]))]}", fontsize=9)
        a.axis("off")
        if r == 0:
            a.legend(loc="lower left", fontsize=7)
        for c, (k, nm) in enumerate(((1, "wide input (512 x 256)"), (0, "road input (512 x 256)")), start=1):
            axs[r, c].imshow(to_rgb(fr[k]))
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(nm)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print("saved", out)


if __name__ == "__main__":
    main()
