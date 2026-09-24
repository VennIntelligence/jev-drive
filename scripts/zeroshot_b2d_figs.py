#!/usr/bin/env python
"""Figures of todos/2026-09-24-zeroshot-exam/bench2drive.md, from images pulled off the box into one directory:

  real_alpamayo.jpg     PhysicalAI-AV clip 030c760c, newest frame of the 4 cameras at 576x320 (2x2)
  carla_alpamayo.jpg    policy-server dump of the same 4 cameras rendered in CARLA, with the predicted path
  real_openpilot.png    comma1M segment 0045b4fe frame 600: road and wide model frames (Y)
  carla_openpilot.jpg   policy-server dump of the CARLA road and wide model frames, with the plan
  strip_<model>_*.jpg   dumps along route 2390 for the timeline strip

    .venv/bin/python scripts/zeroshot_b2d_figs.py <image dir>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevdrive.plots import PAGE, STYLE, save  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "research" / "figs"


def crop_text(path, keep):
    """Drop the text bar the server writes under each dump."""
    im = np.asarray(Image.open(path).convert("RGB"))
    return im[:keep]


def inputs(src):
    with plt.rc_context({**STYLE, "savefig.dpi": 150}):  # photos: 150 dpi keeps the PNG under 500 KB
        fig, ax = plt.subplots(2, 2, figsize=(PAGE, 3.3), gridspec_kw={"height_ratios": [640, 256 * 1152 / 1024]})
        panels = [(crop_text(src / "real_alpamayo.jpg", 640), "(a) Alpamayo, PhysicalAI-AV (real)"),
                  (crop_text(src / "carla_alpamayo.jpg", 640), "(b) Alpamayo, CARLA rig + f-theta resampling"),
                  (np.asarray(Image.open(src / "real_openpilot.png").convert("RGB")), "(c) openpilot, comma1M (real)"),
                  (crop_text(src / "carla_openpilot.jpg", 256), "(d) openpilot, CARLA rig + modeld warp")]
        for a, (im, label) in zip(ax.ravel(), panels):
            a.imshow(im)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_xlabel(label)
        fig.tight_layout(h_pad=0.4, w_pad=0.4)
        save(fig, OUT, "zeroshot-b2d-inputs")


def strip(src, model, keep, name):
    files = sorted(src.glob("strip_%s_*.jpg" % model))
    if not files:
        return
    with plt.rc_context({**STYLE, "savefig.dpi": 150}):  # photos: 150 dpi keeps the PNG under 500 KB
        fig, ax = plt.subplots(1, len(files), figsize=(PAGE, 1.3 if model == "openpilot" else 1.25))
        for a, f in zip(np.atleast_1d(ax), files):
            im = crop_text(f, keep)
            if model == "alpamayo":
                im = im[:320, 576:]       # front-wide tile
            else:
                im = im[:, :512]          # road model frame
            a.imshow(im)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_xlabel("t = %.1f s" % (int(f.stem.split("_")[-1]) / 10))
        fig.tight_layout(w_pad=0.2)
        save(fig, OUT, name)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    inputs(src)
    strip(src, "alpamayo", 640, "zeroshot-b2d-2390-alpamayo")
    strip(src, "openpilot", 256, "zeroshot-b2d-2390-openpilot")
