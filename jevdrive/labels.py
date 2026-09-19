"""Step 1b: future-turn labels and past ego-state features per keyframe.

Label: yaw change over the next HORIZON seconds, > +THRESH deg -> left, < -THRESH deg -> right, else straight.
Ego state: the pose is sampled every DT seconds over the past HORIZON seconds (linear interpolation of the
20 Hz trajectory, yaw unwrapped per scene), giving speed and yaw rate per interval and the accelerations between them.
Keyframes without a full HORIZON of trajectory before and after them are dropped.
"""
import numpy as np
import pandas as pd

from .common import CLASSES, get_logger, processed_dir

log = get_logger(__name__)
HORIZON, DT, THRESH_DEG = 2.0, 0.5, 5.0
HARD_YAW_RATE = 1.0  # deg/s; |current yaw rate| below this counts as "not turning yet"
N = int(HORIZON / DT)
EGO_COLS = ([f"speed_{i}" for i in range(N)] + [f"yaw_rate_{i}" for i in range(N)]
            + [f"accel_{i}" for i in range(N - 1)])


def _scene_labels(kf: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    tt = traj.timestamp.to_numpy() * 1e-6
    t0 = kf.timestamp.to_numpy() * 1e-6
    offsets = np.arange(-N, N + 1) * DT  # past ... now ... future
    ts = t0[:, None] + offsets
    x, y, yaw = (np.interp(ts, tt, traj[c].to_numpy()) for c in ("x", "y", "yaw"))

    past = slice(0, N + 1)
    speed = np.hypot(np.diff(x[:, past]), np.diff(y[:, past])) / DT
    yaw_rate = np.degrees(np.diff(yaw[:, past])) / DT
    accel = np.diff(speed) / DT
    dyaw = np.degrees(yaw[:, -1] - yaw[:, N])

    out = kf[["scene", "split", "sample_token", "timestamp"]].copy()
    out["has_past"], out["has_future"] = ts[:, 0] >= tt[0] - 1e-3, ts[:, -1] <= tt[-1] + 1e-3
    out["dyaw_future_deg"] = dyaw
    out["label"] = np.select([dyaw > THRESH_DEG, dyaw < -THRESH_DEG], [0, 2], 1)
    out[EGO_COLS] = np.concatenate([speed, yaw_rate, accel], axis=1)
    out["hard"] = np.abs(yaw_rate[:, -1]) < HARD_YAW_RATE
    return out


def build_labels(kf: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    by_scene = dict(tuple(traj.groupby("scene")))
    return pd.concat([_scene_labels(g, by_scene[s]) for s, g in kf.groupby("scene")], ignore_index=True)


def run(version: str) -> pd.DataFrame:
    d = processed_dir(version)
    lab = build_labels(pd.read_parquet(d / "keyframes.parquet"), pd.read_parquet(d / "ego_traj.parquet"))
    n_all, n_past, n_future = len(lab), (~lab.has_past).sum(), (~lab.has_future).sum()
    lab = lab[lab.has_past & lab.has_future].drop(columns=["has_past", "has_future"]).reset_index(drop=True)
    lab.to_parquet(d / "labels.parquet")

    log.info("labels: kept %d / %d keyframes; dropped %d with < %.0fs past, %d with < %.0fs future",
             len(lab), n_all, n_past, HORIZON, n_future, HORIZON)
    dist = pd.crosstab(lab.split, lab.label.map(dict(enumerate(CLASSES))), margins=True)
    hard = pd.crosstab(lab[lab.hard].split, lab[lab.hard].label.map(dict(enumerate(CLASSES))), margins=True)
    log.info("class distribution:\n%s\nhard subset (|yaw rate now| < %.1f deg/s):\n%s", dist, HARD_YAW_RATE, hard)
    return lab
