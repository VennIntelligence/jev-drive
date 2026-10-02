"""Export the recorded route of every HUGSIM scene scored in the zero-shot exam (run on the box, numpy only).

Writes one JSON {scene: {"xz": [[x, z], ...], "yaw": [...], "cmd": [...]}} where xz is the camera position on the
ground plane (x right, z forward, as infos.pkl ego_pos[0], ego_pos[2]) and yaw is the camera heading in the same
convention as jevdrive.hugsim_zs.ego_pose2d (theta = arctan2(R[0, 2], R[2, 2]), positive = turning right).
    python spin_export_routes.py <scored_op.csv> <out.json>
"""
import csv
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

D = Path("/root/autodl-tmp/ujs/datasets/hugsim/scenes")


def load(ds, scene):
    p = D / ds / scene / "ground_param.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    with zipfile.ZipFile(D / ds / f"{scene}.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("ground_param.pkl"))
        return pickle.load(io.BytesIO(z.read(name)))


def main(csv_path, out):
    res = {}
    for r in csv.DictReader(open(csv_path)):
        key = (r["dataset"], r["scene"])
        if key[1] in res:
            continue
        gp = load(*key)
        cam = np.asarray(gp[0])
        yaw = np.arctan2(cam[:, 0, 2], cam[:, 2, 2])
        res[key[1]] = {"dataset": key[0], "xz": np.round(cam[:, [0, 2], 3], 3).tolist(),
                       "yaw": np.round(yaw, 4).tolist(), "cmd": [int(c) for c in np.ravel(gp[1])]}
    json.dump(res, open(out, "w"))
    print(len(res), "scenes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
