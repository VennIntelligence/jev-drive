"""nq4 P3 scene selection: WOD segments (and nuScenes mini scenes) with a pedestrian in the ego corridor.

Rule registered in todos/2026-09-26-night-queue-4.md, [P3] 18:50 entry, before any reconstruction:
corridor = logged future path clipped to 30 m (P1 helpers, scripts/nq4_p1_count.py), |lateral| <= 4 m,
pedestrians only. A segment qualifies with one target event whose first corridor frame f0 lies in
[3.0, 16.5] s, ego speed at f0 >= 2 m/s, the target in the front camera (4 <= x <= 30 m, |bearing| <= 22 deg)
for >= 1 s continuously inside [f0 - 1, f0 + 2] s, <= 5 pedestrian tracks entering the corridor inside the
render window [f0 - 3, f0 + 2] s, and daytime. Qualifying segments are sorted by crc32(name); the first 10 win.

  python -m jevdrive.nq4_p3_select wod  --out <dir>     # WOD v2 parquet under $DATA_DIR/datasets/waymo_perception/v2
  python -m jevdrive.nq4_p3_select nusc --out <dir>     # nuScenes v1.0-mini under $DATA_DIR/datasets/nuscenes
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive.common import n_cpus

DATA = Path(__import__("os").environ.get("DATA_DIR", Path.home() / "data"))
_spec = importlib.util.spec_from_file_location("p1", Path(__file__).resolve().parents[1] / "scripts/nq4_p1_count.py")
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

MIN_SPEED, VIS_X, VIS_BEARING, VIS_MIN_S, MAX_TRACKS = 2.0, (4.0, 30.0), math.radians(22.0), 1.0, 5
WIN_PRE, WIN_POST, VIS_PRE, VIS_POST = 3.0, 2.0, 1.0, 2.0
N_PICK = 10


def crc(s: str) -> int:
    return zlib.crc32(s.encode())


def corridor_mask(xy: np.ndarray, yaw: np.ndarray, peds_local: list[np.ndarray]) -> list[np.ndarray]:
    """Per frame i: bool mask over that frame's pedestrians (vehicle-frame xy) of 'inside the 30 m / 4 m corridor'."""
    cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    out = []
    for i, pts in enumerate(peds_local):
        if len(pts) == 0:
            out.append(np.zeros(0, bool))
            continue
        path = p1._extend_path_to_30m(p1._path_to_30m(xy, cum, i), float(yaw[i]))
        s, lat = p1._project_to_path(pts, path, float(yaw[i]))
        out.append((s >= 0) & (s <= p1.MAX_PATH_M) & (lat <= p1.MAX_LATERAL_M))
    return out


def longest_run(t: np.ndarray, dt: float) -> float:
    """Longest continuous span (s) of sorted sample times with gaps <= 1.5 dt."""
    if len(t) == 0:
        return 0.0
    best, start = 0.0, t[0]
    for a, b in zip(t[:-1], t[1:]):
        if b - a > 1.5 * dt:
            best, start = max(best, a - start), b
    return max(best, t[-1] - start) + dt  # a sample covers one period


def select_scene(name: str, t: np.ndarray, xy: np.ndarray, yaw: np.ndarray, peds: pd.DataFrame, day: bool,
                 dt: float, f0_range: tuple[float, float]) -> dict:
    """peds: columns frame (index into t), track, x, y (vehicle frame at that frame). Returns one summary row."""
    speed = np.zeros(len(t))
    speed[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) / (t[2:] - t[:-2])
    speed[0], speed[-1] = speed[1], speed[-2]
    by_frame = [g for _, g in peds.groupby("frame")] if len(peds) else []
    frames = {int(g.frame.iat[0]): g for g in by_frame}
    local = [frames[i][["x", "y"]].to_numpy() if i in frames else np.zeros((0, 2)) for i in range(len(t))]
    masks = corridor_mask(xy, yaw, local)
    inside = pd.concat([frames[i][m] for i, m in enumerate(masks) if i in frames and m.any()]) if any(m.any() for m in masks) else peds.iloc[:0]
    row = dict(name=name, day=day, n_frames=len(t), n_ped_tracks=peds.track.nunique() if len(peds) else 0,
               n_corridor_tracks=inside.track.nunique() if len(inside) else 0, n_events=0, n_targets=0)
    events = []
    for tr, g in inside.groupby("track"):
        f = np.sort(g.frame.to_numpy())
        starts = f[np.r_[True, np.diff(f) > 1]]
        events += [(tr, int(s)) for s in starts]
    row["n_events"] = len(events)
    fail = {"window": 0, "speed": 0, "visible": 0, "crowd": 0}
    targets = []
    for tr, f0 in events:
        t0 = t[f0] - t[0]
        if not f0_range[0] <= t0 <= f0_range[1]:
            fail["window"] += 1; continue
        if speed[f0] < MIN_SPEED:
            fail["speed"] += 1; continue
        g = peds[(peds.track == tr) & (t[peds.frame] >= t[f0] - VIS_PRE) & (t[peds.frame] <= t[f0] + VIS_POST)]
        vis = g[(g.x >= VIS_X[0]) & (g.x <= VIS_X[1]) & (np.abs(np.arctan2(g.y, g.x)) <= VIS_BEARING)]
        if longest_run(np.sort(t[vis.frame.to_numpy()]), dt) < VIS_MIN_S:
            fail["visible"] += 1; continue
        win = inside[(t[inside.frame] >= t[f0] - WIN_PRE) & (t[inside.frame] <= t[f0] + WIN_POST)]
        n_del = win.track.nunique()
        if n_del > MAX_TRACKS:
            fail["crowd"] += 1; continue
        targets.append(dict(track=tr, f0=f0, t0=round(float(t0), 2), v0=round(float(speed[f0]), 2), n_delete=n_del,
                            delete_tracks=sorted(map(str, win.track.unique()))))
    row.update({f"fail_{k}": v for k, v in fail.items()})
    row["n_targets"] = len(targets)
    if targets:
        tg = min(targets, key=lambda d: crc(f"{name}/{d['track']}/{d['f0']}"))
        row.update(target_track=tg["track"], f0=tg["f0"], t0=tg["t0"], v0=tg["v0"], n_delete=tg["n_delete"],
                   delete_tracks=json.dumps(tg["delete_tracks"]))
    row["qualifies"] = bool(targets) and day
    return row


# ---------------------------------------------------------------- WOD v2
V2 = DATA / "datasets/waymo_perception/v2"
C = "[LiDARBoxComponent]."


def _wod_one(args: tuple[str, str]) -> dict:
    split, seg = args
    pose = pd.read_parquet(V2 / split / "vehicle_pose" / f"{seg}.parquet").sort_values("key.frame_timestamp_micros")
    ts = pose["key.frame_timestamp_micros"].to_numpy()
    T = np.stack(pose["[VehiclePoseComponent].world_from_vehicle.transform"].to_numpy()).reshape(-1, 4, 4)
    box = pd.read_parquet(V2 / split / "lidar_box" / f"{seg}.parquet",
                          columns=["key.frame_timestamp_micros", "key.laser_object_id", C + "box.center.x", C + "box.center.y", C + "type"])
    box = box[box[C + "type"] == 2]
    fidx = {v: i for i, v in enumerate(ts)}
    peds = pd.DataFrame({"frame": box["key.frame_timestamp_micros"].map(fidx), "track": box["key.laser_object_id"],
                         "x": box[C + "box.center.x"], "y": box[C + "box.center.y"]}).dropna()
    peds["frame"] = peds.frame.astype(int)
    stats = pd.read_parquet(V2 / split / "stats" / f"{seg}.parquet", columns=["[StatsComponent].time_of_day"])
    day = bool((stats["[StatsComponent].time_of_day"] == "Day").all())
    row = select_scene(seg, (ts - ts[0]) / 1e6, T[:, :2, 3], np.arctan2(T[:, 1, 0], T[:, 0, 0]), peds, day, 0.1, (3.0, 16.5))
    row["split"] = split
    return row


def run_wod(out: Path) -> pd.DataFrame:
    lst = (V2 / "sceneflow_list.txt").read_text().split()
    sf = {Path(u).name.removeprefix("segment-").removesuffix("_with_camera_labels.tfrecord"): u for u in lst if u.endswith(".tfrecord")}
    jobs = [(sp, p.stem) for sp in ("training", "validation") for p in sorted((V2 / sp / "lidar_box").glob("*.parquet"))]
    with ProcessPoolExecutor(min(8, n_cpus())) as ex:
        rows = list(ex.map(_wod_one, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    df["in_sceneflow"] = df.name.isin(sf)
    df["gcs"] = df.name.map(sf)
    df["qualifies"] &= df.in_sceneflow
    df["crc"] = df.name.map(crc)
    df = df.sort_values(["qualifies", "crc"], ascending=[False, True]).reset_index(drop=True)
    df["picked"] = False
    df.loc[df.index[df.qualifies][:N_PICK], "picked"] = True
    return df


# ---------------------------------------------------------------- nuScenes mini
def run_nusc(out: Path, version: str = "v1.0-mini") -> pd.DataFrame:
    root = DATA / "datasets/nuscenes"
    J = {k: pd.DataFrame(json.loads((root / version / f"{k}.json").read_text()))
         for k in ("scene", "sample", "sample_data", "ego_pose", "sample_annotation", "instance", "category")}
    cat = J["instance"].merge(J["category"][["token", "name"]].rename(columns={"token": "category_token", "name": "cat"}), on="category_token")
    ped_inst = set(cat[cat.cat.str.startswith("human.pedestrian")].token)
    sd = J["sample_data"][J["sample_data"].is_key_frame & J["sample_data"].filename.str.contains("LIDAR_TOP")]
    sd = sd.merge(J["ego_pose"][["token", "translation", "rotation"]].rename(columns={"token": "ego_pose_token", "translation": "ego_t", "rotation": "ego_q"}), on="ego_pose_token")
    smp = J["sample"].merge(sd[["sample_token", "ego_t", "ego_q"]].rename(columns={"sample_token": "token"}), on="token")
    ann = J["sample_annotation"][J["sample_annotation"].instance_token.isin(ped_inst)]
    rows = []
    for _, sc in J["scene"].iterrows():
        s = smp[smp.scene_token == sc.token].sort_values("timestamp").reset_index(drop=True)
        t = (s.timestamp.to_numpy() - s.timestamp.iat[0]) / 1e6
        xy = np.stack(s.ego_t.to_numpy())[:, :2]
        yaw = np.array([p1._yaw_from_quaternion(q) for q in s.ego_q])
        a = ann[ann.sample_token.isin(s.token)].merge(s[["token"]].reset_index().rename(columns={"token": "sample_token", "index": "frame"}), on="sample_token")
        w = np.stack(a.translation.to_numpy())[:, :2] - xy[a.frame] if len(a) else np.zeros((0, 2))
        c, si = np.cos(yaw[a.frame]), np.sin(yaw[a.frame])
        peds = pd.DataFrame({"frame": a.frame.to_numpy(), "track": a.instance_token.to_numpy(),
                             "x": c * w[:, 0] + si * w[:, 1], "y": -si * w[:, 0] + c * w[:, 1]})
        day = "night" not in sc.description.lower()
        row = select_scene(sc["name"], t, xy, yaw, peds, day, 0.5, (2.0, 16.0))
        row.update(split=version, description=sc.description)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["crc"] = df.name.map(crc)
    df = df.sort_values(["qualifies", "crc"], ascending=[False, True]).reset_index(drop=True)
    df["picked"] = False
    df.loc[df.index[df.qualifies][:1], "picked"] = True
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["wod", "nusc"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = run_wod(a.out) if a.what == "wod" else run_nusc(a.out)
    df.drop(columns=["crc"]).to_csv(a.out / f"selection_{a.what}.csv", index=False)
    funnel = {"segments": len(df), "with_corridor_ped": int((df.n_corridor_tracks > 0).sum()),
              "with_target": int((df.n_targets > 0).sum()), "day": int(df.day.sum()),
              "qualifies": int(df.qualifies.sum()), "picked": int(df.picked.sum())}
    for k in ("window", "speed", "visible", "crowd"):
        funnel[f"events_failing_{k}"] = int(df[f"fail_{k}"].sum())
    if "in_sceneflow" in df:
        funnel["in_sceneflow"] = int(df.in_sceneflow.sum())
    (a.out / f"selection_{a.what}_funnel.json").write_text(json.dumps(funnel, indent=1))
    print(json.dumps(funnel))
    cols = [c for c in ("split", "name", "target_track", "t0", "v0", "n_delete", "n_corridor_tracks", "description") if c in df]
    print(df[df.picked][cols].to_string())


if __name__ == "__main__":
    main()
