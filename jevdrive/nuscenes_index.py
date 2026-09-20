"""Step 1a: keyframe table and ego trajectories straight from the nuScenes JSON tables.

We skip the devkit's NuScenes class: it loads every annotation table, which is slow on trainval.
Outputs (under processed/nuscenes/<version>/):
  keyframes.parquet  one row per CAM_FRONT keyframe: scene, split, sample/sd tokens, timestamp, pose, image path
  cam_front.parquet  every CAM_FRONT frame, keyframes and sweeps alike (~12 Hz), for video backbones that
                     want a clip rather than a frame; only usable once sweeps/CAM_FRONT is extracted
  ego_traj.parquet   per-scene ego pose at LIDAR_TOP rate (20 Hz), yaw unwrapped within each scene
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import orjson
import pandas as pd

from .common import dataroot, get_logger, processed_dir

log = get_logger(__name__)
SPLITS = {"v1.0-mini": ("mini_train", "mini_val"), "v1.0-trainval": ("train", "val")}


def _table(version: str, name: str) -> pd.DataFrame:
    return pd.DataFrame(orjson.loads((dataroot() / version / f"{name}.json").read_bytes()))


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw (rad, counter-clockwise about +z) from (w, x, y, z) quaternions."""
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def scene_splits(version: str) -> dict[str, str]:
    from nuscenes.utils.splits import create_splits_scenes  # heavy import, only needed here
    s = create_splits_scenes()
    return {scene: split for split, key in zip(("train", "val"), SPLITS[version]) for scene in s[key]}


def build_index(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = ["scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor"]
    with ThreadPoolExecutor(len(names)) as ex:
        t = dict(zip(names, ex.map(lambda n: _table(version, n), names)))

    channel = t["calibrated_sensor"].merge(t["sensor"], left_on="sensor_token", right_on="token", suffixes=("", "_s"))
    sd = t["sample_data"].merge(channel[["token", "channel"]].rename(columns={"token": "calibrated_sensor_token"}))
    sd = sd[sd.channel.isin(["CAM_FRONT", "LIDAR_TOP"])]
    scene = t["sample"][["token", "scene_token"]].merge(
        t["scene"][["token", "name"]].rename(columns={"token": "scene_token", "name": "scene"}))
    sd = sd.merge(scene.rename(columns={"token": "sample_token"}), on="sample_token")

    pose = t["ego_pose"].rename(columns={"token": "ego_pose_token"})
    xyz, quat = np.stack(pose.translation.to_numpy()), np.stack(pose.rotation.to_numpy())
    pose = pd.DataFrame({"ego_pose_token": pose.ego_pose_token, "x": xyz[:, 0], "y": xyz[:, 1],
                         "yaw": quat_to_yaw(quat)})
    sd = sd.merge(pose, on="ego_pose_token")

    traj = (sd[sd.channel == "LIDAR_TOP"][["scene", "timestamp", "x", "y", "yaw"]]
            .drop_duplicates(["scene", "timestamp"]).sort_values(["scene", "timestamp"], ignore_index=True))
    traj["yaw"] = traj.groupby("scene").yaw.transform(np.unwrap)

    split = scene_splits(version)
    cam = sd[sd.channel == "CAM_FRONT"]
    cam = cam.assign(split=cam.scene.map(split).fillna("test"), path=cam.filename)[
        ["scene", "split", "sample_token", "token", "timestamp", "x", "y", "yaw", "path", "is_key_frame"]]
    cam = cam.rename(columns={"token": "sd_token"}).sort_values(["scene", "timestamp"], ignore_index=True)
    kf = cam[cam.is_key_frame].drop(columns="is_key_frame").reset_index(drop=True)
    return kf, traj, cam


def run(version: str) -> pd.DataFrame:
    kf, traj, cam = build_index(version)
    out = processed_dir(version)
    kf.to_parquet(out / "keyframes.parquet")
    cam.to_parquet(out / "cam_front.parquet")
    traj.to_parquet(out / "ego_traj.parquet")
    log.info("%s: %d scenes, %d CAM_FRONT keyframes (%s), %d CAM_FRONT frames in all (%.1f Hz), "
             "%d trajectory poses -> %s", version, kf.scene.nunique(), len(kf),
             kf.split.value_counts().to_dict(), len(cam),
             len(cam) / max(1e-9, (cam.groupby("scene").timestamp.max() - cam.groupby("scene").timestamp.min())
                            .sum() * 1e-6), len(traj), out)
    return kf
