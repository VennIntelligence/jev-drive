#!/usr/bin/env python
"""op_wide_ft figures.

  wide_fig.py inputs OUT.png     (box) the model input of the two arms at CARLA dev junction poses: road frame (shared), wide 58.7 deg (W58),
                                 wide 116 deg (W116), luma as the network gets it, t0 frame; green = the wide horizon row 151.8
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def cmd_inputs(out):
    from PIL import Image, ImageDraw
    from jevdrive.common import data_dir
    from jevdrive.openpilot.frames import unpack_luma
    root = data_dir() / "runs/op_wide_ft/carla"
    tab = np.load(root / "p58/samples/route_carla/tab.npz", allow_pickle=True)
    r = np.load(root / "p58/route.npz", allow_pickle=True)
    a, b = (np.load(root / k / "samples/route_carla/imgs.npy", mmap_mode="r") for k in ("p58", "p116"))
    # dev poses 10 m before a 3-exit junction, moving, with a sharp (R_min < 10 m) exit
    sharp = {int(p) for p, rm in zip(r["pose_row"], r["turn_rmin"]) if rm < 10}
    cand = [i for i in np.flatnonzero((tab["split"] == "dev") & (tab["d"] == 10) & (tab["n_exits"] == 3) & (tab["profile"] == "brake")) if i in sharp]
    pick = [cand[k] for k in np.linspace(0, len(cand) - 1, 3).astype(int)]
    tiles = []
    for i in pick:
        row = [unpack_luma(a[i, -1, 0]), unpack_luma(a[i, -1, 1]), unpack_luma(b[i, -1, 1])]
        tiles.append(np.concatenate(row, 1))
    im = Image.fromarray(np.concatenate(tiles, 0)).convert("RGB")
    d = ImageDraw.Draw(im)
    for k in range(len(pick)):
        y0 = 256 * k
        for x0 in (512, 1024):
            d.line([(x0, y0 + 151.8), (x0 + 511, y0 + 151.8)], fill=(40, 200, 60), width=1)
        for x0, lab in ((0, "road (both arms)"), (512, "wide 58.7 deg (W58)"), (1024, "wide 116 deg (W116)")):
            d.text((x0 + 6, y0 + 4), lab + ("  " + str(tab["id"][pick[k]]) if x0 == 0 else ""), fill=(255, 255, 0))
    im.save(out)
    print("wrote", out, [str(tab["id"][i]) for i in pick])


if __name__ == "__main__":
    {"inputs": cmd_inputs}[sys.argv[1]](*sys.argv[2:])
