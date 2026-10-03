"""PSNR of render vs real per camera and frame shift (frame alignment / camera mapping check). Project venv or envs/waymo, CPU:
  rvr_ds_check.py <dataset> <scene> ..."""
import os, sys
from pathlib import Path
import numpy as np

D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ds = sys.argv[1]
for s in sys.argv[2:]:
    R = np.load(D / "runs/real_vs_render" / ds / s / "real.npy", mmap_mode="r")
    G = np.load(D / "runs/real_vs_render" / ds / s / "render.npy", mmap_mode="r")
    ps = lambda a, b: float(-10 * np.log10(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2) / 255 ** 2))  # noqa: E731
    for c in range(R.shape[1]):
        row = {sh: np.median([ps(R[min(max(k + sh, 0), len(R) - 1), c], G[k, c]) for k in range(10, len(R) - 10, 12)]) for sh in (-2, -1, 0, 1, 2)}
        print(ds, s, "cam", c, "PSNR by frame shift", {k: round(v, 1) for k, v in row.items()}, flush=True)
    # camera permutation check on the front render
    print("  front render vs real cams 0/1/2:", [round(float(np.median([ps(R[k, j], G[k, 0]) for k in range(10, len(R) - 10, 12)])), 1) for j in range(R.shape[1])])
