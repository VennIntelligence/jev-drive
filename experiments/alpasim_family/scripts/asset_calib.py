"""Box: CAM_F0 calibration as stored in every MTGS asset (video_scene_dict.pkl), i.e. what the renderer renders with
(colmap_param K + distortion, else cam_intrinsic + distortion; the adapter's own preference) next to what the driver is told
(the same K, rig_to_camera, but never the distortion: plugins/mtgs/server/artifact_adapter.py get_available_cameras_from_data_source).

usage: python asset_calib.py <assets dir> <out.csv>     (AlpaSim venv: needs pyquaternion for the pickles)
"""
import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def one(p: Path) -> dict:
    d = pickle.load(open(p, "rb"))
    nested = isinstance(d, dict) and "frame_infos" not in d
    v = list(d.values())[0] if nested else d
    c = v["frame_infos"][0]["cams"]["CAM_F0"]
    src = c["colmap_param"] if "colmap_param" in c else c
    K, D = np.asarray(src["cam_intrinsic"], float), np.asarray(src["distortion"], float)
    K0, D0 = np.asarray(c["cam_intrinsic"], float), np.asarray(c["distortion"], float)
    q = c["sensor2ego_rotation"]
    q = np.asarray(getattr(q, "elements", q), float)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    f = R[:, 2]                                   # optical axis in the rig frame
    t = np.asarray(c["sensor2ego_translation"], float)
    return dict(scene=p.parent.name, vehicle=v.get("vehicle_name", ""), nested=int(nested), colmap=int("colmap_param" in c),
                hw=f"{c.get('height', '')}x{c.get('width', '')}",
                fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], k1=D[0], k2=D[1], p1=D[2], p2=D[3], k3=D[4],
                fx_raw=K0[0, 0], fy_raw=K0[1, 1], k1_raw=D0[0], k2_raw=D0[1], k3_raw=D0[4],
                yaw_deg=np.degrees(np.arctan2(f[1], f[0])), pitch_deg=np.degrees(np.arcsin(-f[2])),
                tx=t[0], ty=t[1], tz=t[2])


if __name__ == "__main__":
    ps = sorted(Path(sys.argv[1]).glob("*/video_scene_dict.pkl"))
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(one, ps, chunksize=16))
    with open(sys.argv[2], "w", newline="") as fh:
        wr = csv.DictWriter(fh, rows[0].keys())
        wr.writeheader()
        wr.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(len(rows), "assets ->", sys.argv[2])
