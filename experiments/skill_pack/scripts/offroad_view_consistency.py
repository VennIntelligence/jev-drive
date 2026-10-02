"""Is the synthesized history consistent with the stated ego motion? (plan Q1, 'look at the history frames')

For every navhard token and every consecutive key-frame pair (k -> k+1, 0.5 s apart) the frame k of the road view is re-projected
(road-plane warp of jevdrive.op_interp) to the pose of frame k+1 under four ego-motion hypotheses and compared with the real
frame k+1 (mean |Y| difference over the lower half of the image, where the road is):
  stated    the NAVSIM pose history as given
  no_yaw    same translation, rotation removed
  flip_yaw  same translation, rotation sign flipped
  still     no motion at all
A history whose rotation is consistent with its images has err(stated) < err(flip_yaw) and err(no_yaw) when the rotation is large
enough to see. Output: view_consistency.csv (per token) under the run dir.   navsim2 env, CPU.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import offroad_lib as L  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

_G = {}


def _init():
    _G["frames"] = np.load(L.D / "runs/navsim_zs/openpilot/navhard_two_stage/frames.npy", mmap_mode="r")
    _G["idx"] = L.index()


def work(i):
    e = _G["idx"][i]
    keys = np.asarray(_G["frames"][i])             # (4, 2, 6, 128, 256)
    cam = np.asarray(e["cams"][-1]["CAM_F0"]["t"], float)
    pose = np.asarray(e["pose"], float)
    import cv2
    out = dict(token=e["token"], stage=e["stage"], map=e["map"], cmd=int(np.argmax(e["cmd"][-1])),
               yaw_rate_t0=float((pose[3, 2] - pose[2, 2]) / 0.5), dyaw_total=float(pose[3, 2] - pose[0, 2]))
    errs = {k: 0.0 for k in ("stated", "no_yaw", "flip_yaw", "still")}
    for k in range(3):
        a, b = pose[k], pose[k + 1]
        hyp = {"stated": b, "no_yaw": np.r_[b[:2], a[2]], "flip_yaw": np.r_[b[:2], a[2] - (b[2] - a[2])], "still": a}
        Yb = I.unpack(keys[k + 1, 0])[0].astype(np.float32)
        for name, dst in hyp.items():
            w = I.warp_frame(keys[k, :1].repeat(2, 0), cam, dst, a)[0]     # road view only (both views are filled, the 2nd unused)
            Yw = I.unpack(w)[0].astype(np.float32)
            errs[name] += float(np.abs(Yw[128:] - Yb[128:]).mean()) / 3
    out.update({f"err_{k}": v for k, v in errs.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    n = len(L.index())
    ids = list(range(n))[::max(1, n // a.limit)] if a.limit else range(n)
    with mp.get_context("fork").Pool(a.procs, initializer=_init) as pool:
        rows = pool.map(work, ids, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(L.OUT / ("view_consistency_debug.csv" if a.limit else "view_consistency.csv"), index=False)
    print(df.groupby("stage")[["err_stated", "err_no_yaw", "err_flip_yaw", "err_still"]].mean())
