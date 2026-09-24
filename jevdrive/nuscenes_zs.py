"""nuScenes open-loop planning zero-shot exam (todos/2026-09-24-zeroshot-exam/nuscenes-physicalai.md).

  index     val keyframes with 6 future keyframes in the same scene (the UniAD/VAD "valid" samples), with the GT
            future of the LIDAR_TOP point in the t0 ego frame, future agent boxes, the VAD driving command, and per
            scene the ~12 Hz frame list of every camera (samples + sweeps) and the 20 Hz ego poses
  geometry  nuScenes pinhole cameras (images are undistorted) in camgeom's WOD calib format, for the same
            rotation-only reprojection the WOD-E2E exam uses
  metrics   L2 and collision rate under the VAD / ST-P3 and the BEV-Planner conventions
  convert   model outputs (rear-axle trajectory + heading) -> the LIDAR_TOP point at the GT times

Frames: nuScenes ego = rear axle, x forward, y left, z up (the same as Alpamayo's rig and WOD's vehicle frame).
The index is built in the project venv (pandas, orjson); everything else is numpy-only so the model venvs import it.
"""
import pickle
from pathlib import Path

import numpy as np

from .common import data_dir, dataroot

VERSION = "v1.0-trainval"
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")
N_FUT = 6                                   # 0.5 ... 3.0 s, one keyframe each
EGO_L, EGO_W, EGO_DX = 4.084, 1.85, 0.5     # ST-P3 / VAD ego box (length, width, centre offset ahead of the point)
CMD_NAMES = ("right", "left", "straight")   # VAD command order
NAV_TEXT = {"left": "Turn left", "straight": "Continue straight", "right": "Turn right"}   # WOD/NAVSIM templates
AGENT_PREFIX = ("vehicle.", "human.pedestrian.")


def root(*parts) -> Path:
    d = data_dir() / "runs" / "nusc_zs" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path() -> Path:
    return data_dir() / "processed" / "nusc_zs" / "val_index.pkl"


def load_index() -> dict:
    with open(index_path(), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- rotations

def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(R) -> np.ndarray:
    R = np.asarray(R)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def rot_z(yaw) -> np.ndarray:
    yaw = np.asarray(yaw, np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1
    return R


def slerp_R(t_src, R_src, t_dst) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp
    t_dst = np.clip(t_dst, t_src[0], t_src[-1])
    return Slerp(t_src, Rotation.from_matrix(R_src))(t_dst).as_matrix()


# ---------------------------------------------------------------- index (project venv)

def _tables(names):
    import orjson
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    def one(n):
        return pd.DataFrame(orjson.loads((dataroot() / VERSION / f"{n}.json").read_bytes()))
    with ThreadPoolExecutor(len(names)) as ex:
        return dict(zip(names, ex.map(one, names)))


def build_index(log=print) -> dict:
    """Val scenes only. Returns {"samples": [entry], "scenes": {scene: {...}}}; see module docstring."""
    from .nuscenes_index import scene_splits
    t = _tables(["scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "sample_annotation",
                 "instance", "category", "log"])
    split = scene_splits(VERSION)
    scenes = t["scene"][t["scene"].name.map(split) == "val"]
    log(f"{len(scenes)} val scenes")
    sensor = t["sensor"].set_index("token").channel
    cs = t["calibrated_sensor"].set_index("token")
    cs_channel = cs.sensor_token.map(sensor)
    sd = t["sample_data"]
    sd = sd.assign(channel=sd.calibrated_sensor_token.map(cs_channel))
    pose = t["ego_pose"].set_index("token")
    smp = t["sample"].set_index("token")
    loc = t["log"].set_index("token").location
    cat = t["instance"].set_index("token").category_token.map(t["category"].set_index("token").name)
    ann = t["sample_annotation"]
    ann = ann.assign(cat=ann.instance_token.map(cat))
    ann = ann[ann.cat.str.startswith(AGENT_PREFIX) & ((ann.num_lidar_pts + ann.num_radar_pts) > 0)]
    ann_by_sample = {k: g for k, g in ann.groupby("sample_token")}

    sd_scene = sd.merge(smp[["scene_token"]], left_on="sample_token", right_index=True)
    out_scenes, samples = {}, []
    for _, sc in scenes.iterrows():
        s_sd = sd_scene[sd_scene.scene_token == sc.token]
        lid = s_sd[s_sd.channel == "LIDAR_TOP"].sort_values("timestamp")
        P = pose.loc[lid.ego_pose_token]
        lid_cs = cs.loc[lid.calibrated_sensor_token.iloc[0]]
        cams = {}
        for c in CAMS:
            f = s_sd[s_sd.channel == c].sort_values("timestamp")
            ccs = cs.loc[f.calibrated_sensor_token.iloc[0]]
            cams[c] = {"t": f.timestamp.to_numpy(np.int64), "path": f.filename.tolist(),
                       "key": f.is_key_frame.to_numpy(bool),
                       "K": np.asarray(ccs.camera_intrinsic, np.float64),
                       "R": quat_wxyz_to_R(ccs.rotation), "t_ego": np.asarray(ccs.translation, np.float64)}
        out_scenes[sc["name"]] = {
            "location": loc[sc.log_token], "cams": cams,
            "pose_t": lid.timestamp.to_numpy(np.int64), "pose_xyz": np.stack(P.translation.to_numpy()),
            "pose_R": np.stack([quat_wxyz_to_R(q) for q in P.rotation]),
            "lidar_xyz": np.asarray(lid_cs.translation, np.float64)}
        # keyframes of the scene, in order
        keys = []
        tok = sc.first_sample_token
        while tok:
            keys.append(tok)
            tok = smp.next[tok]
        kt = smp.loc[keys].timestamp.to_numpy(np.int64)
        sc_d = out_scenes[sc["name"]]
        key_lid = lid[lid.is_key_frame].set_index("sample_token")
        for i, tok in enumerate(keys):
            e = {"token": tok, "scene": sc["name"], "i": i, "t0": int(kt[i]), "hist_s": (kt[i] - kt[0]) * 1e-6,
                 "location": sc_d["location"], "valid": i + N_FUT < len(keys)}
            if e["valid"]:
                pl = pose.loc[key_lid.loc[keys[i:i + N_FUT + 1]].ego_pose_token]
                p = np.stack(pl.translation.to_numpy())
                R = np.stack([quat_wxyz_to_R(q) for q in pl.rotation])
                R0t = R[0].T
                d = sc_d["lidar_xyz"]
                lidar = (p[1:] - p[0]) @ R0t.T + (R0t @ R[1:] @ d) - d          # LIDAR_TOP point, t0 ego frame
                e["fut_t"] = (kt[i + 1:i + N_FUT + 1] - kt[i]) * 1e-6
                e["gt"] = lidar[:, :2].astype(np.float32)
                e["gt_rear"] = ((p[1:] - p[0]) @ R0t.T)[:, :2].astype(np.float32)
                e["gt_yaw"] = yaw_of(R0t @ R[1:]).astype(np.float32)
                y3 = e["gt"][-1, 1]
                e["cmd"] = "right" if y3 <= -2 else "left" if y3 >= 2 else "straight"   # VAD converter rule
                boxes = []
                for tk in keys[i + 1:i + N_FUT + 1]:
                    g = ann_by_sample.get(tk)
                    if g is None or not len(g):
                        boxes.append(np.zeros((0, 5), np.float32))
                        continue
                    c = (np.stack(g.translation.to_numpy()) - p[0]) @ R0t.T
                    yaw = np.array([yaw_of(R0t @ quat_wxyz_to_R(q)) for q in g.rotation])
                    wlh = np.stack(g["size"].to_numpy())
                    boxes.append(np.c_[c[:, :2], wlh[:, 1], wlh[:, 0], yaw].astype(np.float32))   # x y l w yaw
                e["boxes"] = boxes
            samples.append(e)
    log(f"{len(samples)} val keyframes, {sum(e['valid'] for e in samples)} valid (6 future keyframes)")
    return {"samples": samples, "scenes": out_scenes}


# ---------------------------------------------------------------- per-sample inputs (numpy only)

def ego_at(scene: dict, t_us) -> tuple[np.ndarray, np.ndarray]:
    """Ego pose (xyz, R) at arbitrary times, linear / slerp between the 20 Hz LIDAR_TOP poses (clamped at the ends)."""
    t = np.atleast_1d(np.asarray(t_us, np.float64))
    ts = scene["pose_t"].astype(np.float64)
    xyz = np.stack([np.interp(t, ts, scene["pose_xyz"][:, k]) for k in range(3)], -1)
    return xyz, slerp_R(ts, scene["pose_R"], t)


def frame_at(scene: dict, cam: str, t_us: float) -> int:
    """Index of the frame of `cam` nearest to t_us."""
    t = scene["cams"][cam]["t"]
    j = int(np.searchsorted(t, t_us))
    return min((k for k in (j - 1, j) if 0 <= k < len(t)), key=lambda k: abs(t[k] - t_us))


def latest_frame(scene: dict, cam: str, t_us: float) -> int:
    """Index of the last frame of `cam` at or before t_us (0 if none)."""
    return max(int(np.searchsorted(scene["cams"][cam]["t"], t_us, side="right")) - 1, 0)


def cam_calib(scene: dict, cam: str) -> dict:
    """nuScenes pinhole (OpenCV axes, no distortion: nuScenes images are undistorted) in camgeom's WOD format:
    intrinsic [fu fv cu cv k1 k2 p1 p2 k3], extrinsic vehicle_from_camera with camera axes x optical, y left, z up."""
    c = scene["cams"][cam]
    K = c["K"]
    M = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])     # WOD camera axes expressed in OpenCV axes (columns)
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = c["R"] @ M, c["t_ego"]
    return {"intrinsic": [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0, 0], "extrinsic": E, "width": 1600,
            "height": 900}


def alpamayo_history(scene: dict, t0: int) -> tuple[np.ndarray, np.ndarray]:
    """Alpamayo egomotion: 16 steps @10 Hz, t0-1.5 ... t0, xyz and full rotation in the t0 rear-axle frame."""
    xyz, R = ego_at(scene, t0 + (np.arange(16) - 15) * 100_000)
    R0t = R[-1].T
    return ((xyz - xyz[-1]) @ R0t.T).astype(np.float32), (R0t @ R).astype(np.float32)


def cv_speed(scene: dict, t0: int, dt: float = 0.5) -> float:
    """Current speed from the ego poses over the last dt seconds (clamped at the scene start)."""
    xyz, _ = ego_at(scene, [t0 - dt * 1e6, t0])
    return float(np.linalg.norm(xyz[1, :2] - xyz[0, :2]) / dt)


def to_lidar_point(t_src, rear_xy, yaw, lidar_xy, t_dst) -> tuple[np.ndarray, np.ndarray]:
    """Rear-axle trajectory (t_src, xy, heading) in the t0 ego frame -> LIDAR_TOP point (t_dst, 2) and heading:
    p(t) = rear(t) + R(yaw_t) d - d. The origin (0, 0, yaw 0) is prepended at t = 0 before interpolating."""
    t = np.r_[0.0, t_src]
    xy = np.r_[[[0.0, 0.0]], np.asarray(rear_xy, np.float64)]
    yw = np.unwrap(np.r_[0.0, np.asarray(yaw, np.float64)])
    x, y, h = (np.interp(t_dst, t, v) for v in (xy[:, 0], xy[:, 1], yw))
    d = np.asarray(lidar_xy, np.float64)[:2]
    c, s = np.cos(h), np.sin(h)
    return np.stack([x + c * d[0] - s * d[1] - d[0], y + s * d[0] + c * d[1] - d[1]], -1), h


# ---------------------------------------------------------------- metrics

def _corners(x, y, l, w, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([1, 1, -1, -1]) * 0.5
    dy = np.array([1, -1, -1, 1]) * 0.5
    lx, ly = np.multiply.outer(l, dx), np.multiply.outer(w, dy)
    return np.stack([x[..., None] + c[..., None] * lx - s[..., None] * ly,
                     y[..., None] + s[..., None] * lx + c[..., None] * ly], -1)          # (..., 4, 2)


def _overlap(a, b) -> np.ndarray:
    """Separating-axis test between rectangles a (4, 2) and b (n, 4, 2) -> (n,) bool."""
    out = np.ones(len(b), bool)
    for poly in (a[None].repeat(len(b), 0), b):
        for k in range(2):
            edge = poly[:, k + 1] - poly[:, k]
            axis = np.stack([-edge[:, 1], edge[:, 0]], -1)                                 # (n, 2)
            pa, pb = np.einsum("nd,cd->nc", axis, a), np.einsum("nd,ncd->nc", axis, b)
            out &= ~((pa.max(1) < pb.min(1)) | (pb.max(1) < pa.min(1)))
    return out


def traj_yaw(xy: np.ndarray) -> np.ndarray:
    """Heading of each point from the displacement since the previous point (origin first); a step shorter
    than 0.1 m keeps the previous heading (0 at the start)."""
    prev = np.r_[[[0.0, 0.0]], xy[:-1]]
    d = xy - prev
    out, h = np.zeros(len(xy)), 0.0
    for k in range(len(xy)):
        if np.hypot(*d[k]) >= 0.1:
            h = np.arctan2(d[k, 1], d[k, 0])
        out[k] = h
    return out


def collisions(xy: np.ndarray, boxes: list, with_yaw: bool) -> np.ndarray:
    """(6,) bool: does the ego box at each step overlap any GT agent box of that future keyframe.
    ST-P3 / VAD: axis-aligned (t0 heading), centred EGO_DX ahead of the point; BEV-Planner: heading from the
    trajectory."""
    yaw = traj_yaw(xy) if with_yaw else np.zeros(len(xy))
    out = np.zeros(len(xy), bool)
    for k, b in enumerate(boxes):
        if not len(b):
            continue
        cx, cy = xy[k, 0] + EGO_DX * np.cos(yaw[k]), xy[k, 1] + EGO_DX * np.sin(yaw[k])
        ego = _corners(np.array(cx), np.array(cy), np.array(EGO_L), np.array(EGO_W), np.array(yaw[k]))
        out[k] = _overlap(ego, _corners(b[:, 0], b[:, 1], b[:, 2], b[:, 3], b[:, 4])).any()
    return out


def per_sample(pred: np.ndarray, samples: list) -> dict:
    """pred (N, 6, 2) LIDAR_TOP points at each sample's fut_t. Returns per-sample arrays:
    err (N, 6) L2 per step; col_vad (N, 6), col_bevp (N, 6) collision indicators with the GT-collides steps masked
    (a step counts only if the GT box at that step does not collide, as ST-P3's evaluate_coll)."""
    gt = np.stack([e["gt"] for e in samples])
    err = np.linalg.norm(pred - gt, axis=-1)
    col = {}
    for name, wy in (("col_vad", False), ("col_bevp", True)):
        c = np.zeros(err.shape, bool)
        for n, e in enumerate(samples):
            c[n] = collisions(pred[n], e["boxes"], wy) & ~collisions(e["gt"].astype(np.float64), e["boxes"], wy)
        col[name] = c
    return {"err": err, **col}


def horizons(ps: dict) -> dict:
    """Per-sample values at 1/2/3 s: L2 averaged over the steps up to t (VAD / ST-P3 / BEV-Planner) and at the step
    t (UniAD paper); collision as the step mean up to t (VAD) and as any collision up to t (BEV-Planner)."""
    out = {}
    for s, k in ((1, 2), (2, 4), (3, 6)):
        out[f"l2_{s}s"] = ps["err"][:, :k].mean(1)
        out[f"l2pt_{s}s"] = ps["err"][:, k - 1]
        out[f"col_vad_{s}s"] = ps["col_vad"][:, :k].mean(1) * 100
        out[f"col_bevp_{s}s"] = ps["col_bevp"][:, :k].any(1) * 100.0
    return out
