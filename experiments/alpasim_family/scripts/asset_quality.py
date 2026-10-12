"""Box: cheap per-asset reconstruction-quality measures for every public MTGS asset (plans/2026-10-12-quality-prereg.md, Q1-Q4).

  Q1 pose_t_m / pose_r_deg   median residual of the reconstruction's ego2global against ego2global_original over the scene window
                             (frames 18 .. 68), both re-anchored at frame 18 (how much the reconstruction moved the logged poses)
  Q2 fyfx_dev                |fy / fx - 1| of the colmap-refined CAM_F0 intrinsics (the render is anisotropic by that much)
  Q3 k1_dev                  |k1_colmap - k1_raw|
  Q4 bg_bytes                size of the background checkpoint (a proxy for the Gaussian count)

usage: python asset_quality.py <assets dir> <out.csv>     (AlpaSim venv)
"""
import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

F0, N = 18, 51


def _rel(E):
    A = np.linalg.inv(E[0])
    return A[None] @ E


def one(p: Path) -> dict:
    v = list(pickle.load(open(p, "rb")).values())[0]
    fi = v["frame_infos"][F0:F0 + N]
    E = np.array([np.asarray(f["ego2global"], float) for f in fi])
    has = all("ego2global_original" in f for f in fi)
    if has:
        O = np.array([np.asarray(f["ego2global_original"], float) for f in fi])
        D = np.linalg.inv(_rel(O)) @ _rel(E)
        t = np.linalg.norm(D[:, :3, 3], axis=1)
        r = np.degrees(np.arccos(np.clip((np.trace(D[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    else:                                            # older bundles carry no original poses
        t = r = np.full(len(fi), np.nan)
    c = v["frame_infos"][0]["cams"]["CAM_F0"]
    K = np.asarray(c["colmap_param"]["cam_intrinsic"], float)
    bg = next((p.parent / "background").glob("*.ckpt"))
    return dict(scene=p.parent.name, has_original=int(has), pose_t_m=float(np.median(t[1:])), pose_t_max=float(np.max(t)), pose_r_deg=float(np.median(r[1:])),
                fyfx_dev=abs(K[1, 1] / K[0, 0] - 1), k1_dev=abs(c["colmap_param"]["distortion"][0] - c["distortion"][0]),
                bg_bytes=bg.stat().st_size)


if __name__ == "__main__":
    ps = sorted(Path(sys.argv[1]).glob("*/video_scene_dict.pkl"))
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(one, ps, chunksize=16))
    with open(sys.argv[2], "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows({k: (round(x, 6) if isinstance(x, float) else x) for k, x in r.items()} for r in rows)
    print(len(rows), "->", sys.argv[2])
