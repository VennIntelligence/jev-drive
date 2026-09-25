#!/usr/bin/env python
"""Survey of the HUGSIM scenarios for the zero-shot exam (todos/2026-09-25-hugsim-exam): one row per scenario yaml
with the start state, the inserted actors, the recorded route (length, heading change, turn direction, command
mix) and, for nuScenes, the city (left- or right-hand traffic). Reads ground_param.pkl straight from the scene zips.

    $DATA_DIR/envs/hugsim/bin/python scripts/hugsim/zs_survey.py --out research/results/hugsim-exam/scenarios.csv
"""
import argparse
import csv
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import yaml

D = Path(os.environ.get("DATA_DIR", Path.home() / "data"))
H = D / "datasets" / "hugsim"
DATASETS = ("nuscenes", "waymo", "kitti360", "pandaset")


def nusc_locations():
    root = D / "datasets" / "nuscenes" / "v1.0-trainval"
    logs = {r["token"]: r["location"] for r in json.loads((root / "log.json").read_text())}
    return {r["name"]: logs[r["log_token"]] for r in json.loads((root / "scene.json").read_text())}


def route(ds, scene):
    """Recorded ego (front camera) route: xz, heading (+ right), commands; from the scene zip."""
    zp = H / "scenes" / ds / f"{scene}.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith("ground_param.pkl"))
        cam_poses, _, cmds = pickle.load(io.BytesIO(z.read(name)))
    xz = cam_poses[:, [0, 2], 3]
    fwd = cam_poses[:, [0, 2], 2]                     # optical axis in world (x, z)
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1]))  # + = right, as the simulator's theta
    return xz, yaw, np.asarray(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loc = nusc_locations()
    rows, cache = [], {}
    for ds in DATASETS:
        for f in sorted((H / "scenarios" / ds).glob("*.yaml")):
            c = yaml.safe_load(f.read_text())
            scene, mode = c["scene_name"], c["mode"]
            if (ds, scene) not in cache:
                cache[(ds, scene)] = route(ds, scene)
            xz, yaw, cmds = cache[(ds, scene)]
            seg = np.linalg.norm(np.diff(xz, axis=0), axis=1)
            dyaw = np.degrees(yaw[-1] - yaw[0])
            actors = c.get("plan_list") or []
            ahead = [p for p in actors if abs(float(p[0])) < 2.0 and 0 < float(p[1]) < 40]   # in the ego lane, ahead
            rows.append(dict(
                dataset=ds, scenario=f.stem, scene=scene, mode=mode, difficulty=mode.split("_")[0],
                start_velo=c.get("start_velo"), start_ab=c.get("start_ab"), start_euler=c.get("start_euler"),
                hd_map=bool(c.get("load_HD_map")), n_actors=len(actors),
                n_ahead=len(ahead), ahead_min_b=min((float(p[1]) for p in ahead), default=""),
                ahead_min_v=min((float(p[4]) for p in ahead), default=""),
                planners="|".join(sorted({str(p[6]) for p in actors})),
                route_m=round(float(seg.sum()), 1), heading_change_deg=round(float(dyaw), 1),
                max_abs_heading_deg=round(float(np.degrees(np.abs(yaw - yaw[0]).max())), 1),
                turn="left" if dyaw < -30 else "right" if dyaw > 30 else "straight",
                cmd_right=round(float((cmds == 0).mean()), 3), cmd_left=round(float((cmds == 1).mean()), 3),
                cmd_straight=round(float((cmds == 2).mean()), 3), n_route_poses=len(xz),
                location=loc.get(scene, "") if ds == "nuscenes" else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} scenarios -> {a.out}")


if __name__ == "__main__":
    main()
