#!/usr/bin/env python
"""Board views, step 2 (Mac, any env with numpy / PIL / matplotlib / cv2): the review sheet from bv_dump.py tiles.
rows = board x scene; columns = source frame | openpilot road input | openpilot wide input. Each model input carries
openpilot's nominal horizon row (cy of its model intrinsics: road 47.6, wide 151.8 of 256) and the pitch-implied horizon
for this frame; the title carries the heuristic sky share of the wide input.

  python experiments/leaderboard_audit/scripts/bv_sheet.py <tiles_dir> experiments/leaderboard_audit/figs/board_views.png
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

T = Path(sys.argv[1])
OUTP = sys.argv[2]
CY = {"road": 47.6, "wide": 0.5 * (256 + 47.6)}
BOARDS = [("wod", "WOD-E2E"), ("navtest", "navtest"), ("navhard", "navhard stage 2"), ("hugsim", "HUGSIM"), ("b2d", "B2D / CARLA")]


def sky_fraction(im, hrow):
    """Heuristic: pixels above the nominal horizon row that look like sky (blue-ish or near-white low chroma, bright) and connect to the top edge."""
    x = im.astype(int)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    m = (((b > r + 8) & (lum > 100)) | ((lum > 170) & (x.max(-1) - x.min(-1) < 35))).astype(np.uint8)
    m[int(hrow):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(m)
    top = set(np.unique(lab[0][lab[0] > 0]))
    sky = np.isin(lab, list(top)) if top else np.zeros_like(m, bool)
    return float(sky.mean()), sky


rows = [(b, n, s) for b, n in BOARDS for s in ("straight", "turn")]
fig, ax = plt.subplots(len(rows), 3, figsize=(15, 2.55 * len(rows)), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.03, "hspace": 0.16})
res = {}
for i, (b, n, s) in enumerate(rows):
    for j, k in enumerate(("src", "road", "wide")):
        a = ax[i, j]
        a.set_xticks([]); a.set_yticks([])
        p = T / f"{b}_{s}_{k}.png"
        if not p.exists():
            a.text(0.5, 0.5, "source frame not stored\n(native 1928x1208 CARLA camera, not dumped)" if k == "src" else "missing", ha="center", va="center", transform=a.transAxes, fontsize=9)
            continue
        im = np.asarray(Image.open(p).convert("RGB"))
        a.imshow(im)
        if k != "src":
            a.axhline(CY[k], color="#ffd400", lw=1.4)
            a.text(4, CY[k] - 4, f"openpilot nominal horizon, row {CY[k]:.0f}", color="#ffd400", fontsize=7, va="bottom")
            if k == "wide":
                f, _ = sky_fraction(im, CY[k])
                res[f"{b}_{s}"] = f
                a.set_xlabel(f"heuristic sky share {f * 100:.0f}% (geometric above-horizon share 59%)", fontsize=7, labelpad=1)
        if j == 0:
            a.set_ylabel(f"{n}\n{s}", fontsize=9)
        if i == 0:
            a.set_title({"src": "source camera frame", "road": "openpilot road input (f=910, 31.4 deg)", "wide": "openpilot wide input (f=455, 59.5 deg)"}[k], fontsize=10)
fig.savefig(OUTP, dpi=100, bbox_inches="tight")
json.dump(res, open(Path(OUTP).with_suffix(".sky.json"), "w"), indent=1)
print(OUTP, res)
