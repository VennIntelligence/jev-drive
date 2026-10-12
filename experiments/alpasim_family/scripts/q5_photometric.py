"""Box: Q5 of plans/2026-10-12-quality-prereg.md: the k = 0 render (driver dump, SH30_DUMP) against the real nuPlan CAM_F0 frame of the
same instant (asset frame 18; frames 16-20 searched, the best edge match kept and reported). Gray, per-image mean / sd normalised;
q5_l1 = mean absolute difference, q5_ncc = gradient-magnitude NCC (sigma 2), both on the frame minus a 60 px border.

usage: python q5_photometric.py <driver-logs/dump dir> <assets dir> <sensor_blobs/test dir> <out.csv>     (AlpaSim venv: cv2)
"""
import csv
import glob
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def grad(g):
    x = cv2.GaussianBlur(g, (0, 0), 2)
    return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def norm(g):
    return (g - g.mean()) / (g.std() + 1e-6)


def main():
    dump, assets, blobs, out = map(Path, sys.argv[1:5])
    rows = []
    for f in sorted(glob.glob(str(dump / "s*_k0.npz"))):
        z = np.load(f)
        scene = str(z["scene"])
        ren = cv2.cvtColor(cv2.imdecode(z["jpeg"], 1), cv2.COLOR_BGR2GRAY).astype(np.float32)[60:-60, 60:-60]
        fi = list(pickle.load(open(assets / scene / "video_scene_dict.pkl", "rb")).values())[0]["frame_infos"]
        best = None
        for i in range(16, 21):
            p = blobs / fi[i]["cams"]["CAM_F0"]["data_path"]
            if not p.exists():
                continue
            real = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)[60:-60, 60:-60]
            n = ncc(grad(ren), grad(real))
            if best is None or n > best[0]:
                best = (n, i, float(np.abs(norm(ren) - norm(real)).mean()))
        if best:
            rows.append(dict(scene=scene, q5_ncc=round(best[0], 4), q5_l1=round(best[2], 4), q5_frame=best[1]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "->", out)


if __name__ == "__main__":
    main()
