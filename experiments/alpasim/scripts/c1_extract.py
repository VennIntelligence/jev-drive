"""C1 extraction: compact per-scene records out of finished AlpaSim runs, for the zero-score review and the arbitration study.
Run on the box with AlpaSim's own venv ($DATA_DIR/third_party/alpasim/.venv/bin/python); reads only, writes one pickle per call.

  logs    every scene of a run: simulated ego / actor poses, the logged trajectories of all objects (the logged ego first), every
          driver response, the ego states and routes the driver was sent, the 2 Hz metric series, the controller trace, the driver's
          own drive.jsonl records and the summary row
            c1_extract.py logs --run <run dir> --out <pkl>
  map     privileged, analysis only: road-area polygons and lane centre lines within --radius of the logged ego path, in the rollout's
          local frame, through the runtime's own scene loader (the map the offroad scorer saw)
            c1_extract.py map --run <run dir> --scenes <txt> --out <pkl>
  frames  CAM_F0 JPEGs the driver received (11 per scene), downscaled, for chosen scenes
            c1_extract.py frames --run <run dir> --scenes <txt> --out <pkl> [--width 960]
  msgs    the serialized driver-side messages of chosen scenes, for the offline replay (c1_replay.py)
            c1_extract.py msgs --run <run dir> --scenes <txt> --out <dir>
"""
import argparse
import asyncio
import csv
import glob
import io
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def yaw(q) -> float:
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def p3(p) -> list:
    return [p.vec.x, p.vec.y, yaw(p.quat)]


def traj(t) -> np.ndarray:
    """A common.Trajectory -> (n, 4) t_us, x, y, yaw."""
    return np.array([[q.timestamp_us, *p3(q.pose)] for q in t.poses], np.float64).reshape(-1, 4)


def asl_of(run, scene) -> str:
    return glob.glob(f"{run}/rollouts/{scene}/*/rollout.asl")[0]


async def _read(f, want):
    from alpasim_utils.logs import async_read_pb_log
    async for e in async_read_pb_log(f):
        k = e.WhichOneof("log_entry")
        if k in want:
            yield k, getattr(e, k)


def one_log(a) -> tuple:
    run, scene = a

    async def go():
        o = dict(actor_t=[], actors={}, drive=[], ego_obs=[], route=[], ctrl_req=[])
        async for k, m in _read(asl_of(run, scene), {"rollout_metadata", "actor_poses", "driver_return", "driver_request", "driver_ego_trajectory",
                                                    "route_request", "traffic_session_request", "driver_session_request"}):
            if k == "rollout_metadata":
                o["session"] = m.session_metadata.session_uuid
                o["size"] = {x.actor_id: (x.aabb.size_x, x.aabb.size_y, x.actor_label) for x in m.actor_definitions.actor_aabb}
                o["force_gt_us"] = int(getattr(m, "force_gt_duration_us", 0) or 0)
            elif k == "actor_poses":
                o["actor_t"].append(int(m.timestamp_us))
                for x in m.actor_poses:
                    o["actors"].setdefault(x.actor_id, []).append([int(m.timestamp_us), *p3(x.actor_pose)])
            elif k == "driver_request":
                o["drive"].append(dict(now=int(m.time_now_us), query=int(m.time_query_us)))
            elif k == "driver_return":
                o["drive"][-1]["traj"] = traj(m.trajectory)
            elif k == "driver_ego_trajectory":
                d = m.dynamic_states[-1] if len(m.dynamic_states) else None
                o["ego_obs"].append(dict(traj=traj(m.trajectory), v=[d.linear_velocity.x, d.linear_velocity.y] if d else None,
                                         a=[d.linear_acceleration.x, d.linear_acceleration.y] if d else None,
                                         w=d.angular_velocity.z if d else None))
            elif k == "route_request":
                o["route"].append(dict(t=int(m.route.timestamp_us), wp=np.array([[w.x, w.y] for w in m.route.waypoints], np.float32)))
            elif k == "traffic_session_request":
                o["logged"] = [dict(id=x.object_id, static=bool(x.is_static), size=(x.aabb.size_x, x.aabb.size_y), traj=traj(x.trajectory))
                               for x in m.logged_object_trajectories]
            elif k == "driver_session_request":
                rig = m.rollout_spec.vehicle
                o["rig"] = str(rig)[-600:] if not rig.available_cameras else None
        o["actors"] = {k: np.array(v) for k, v in o["actors"].items()}
        return o
    o = asyncio.run(go())
    import polars as pl
    m = pl.read_parquet(glob.glob(f"{run}/rollouts/{scene}/*/metrics.parquet")[0]).select(
        "name", pl.col("timestamps_us").cast(pl.Int64), "values")
    o["metrics"] = {n[0]: np.array([g["timestamps_us"].to_list(), g["values"].to_list()]).T for n, g in m.group_by("name")}
    c = Path(run) / f"controller/alpasim_controller_{o['session']}.csv"
    if c.exists():
        rows = list(csv.reader(open(c)))
        o["ctrl_cols"], o["ctrl"] = rows[0], np.array(rows[1:], np.float64)
    return scene, o


def cmd_logs(a):
    run = a.run.rstrip("/")
    S = json.loads((Path(run) / "aggregate/results-summary.json").read_text())["rollouts"]
    scenes = sorted(s for s in os.listdir(f"{run}/rollouts") if glob.glob(f"{run}/rollouts/{s}/*/rollout.asl") and glob.glob(f"{run}/rollouts/{s}/*/metrics.parquet"))
    with ProcessPoolExecutor(a.jobs) as ex:
        out = dict(ex.map(one_log, [(run, s) for s in scenes], chunksize=4))
    rec = {}
    for line in open(f"{run}/driver-logs/drive.jsonl"):
        r = json.loads(line)
        if r["kind"] == "drive":
            r.pop("ms", None)
            rec.setdefault(r["scene"], []).append(r)
    for r in S:
        if r["clipgt_id"] in out:
            out[r["clipgt_id"]]["summary"] = r
    for s, v in rec.items():
        if s in out:
            out[s]["rec"] = sorted(v, key=lambda r: r["k"])
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print("logs", len(out), "scenes ->", a.out, os.path.getsize(a.out) >> 20, "MB")


def cmd_map(a):
    from alpasim_runtime.config import SimulatorConfig, UserSimulatorConfig  # noqa: F401
    from alpasim_runtime.scene_loader import build_scene_loader
    from omegaconf import OmegaConf
    user = OmegaConf.merge(OmegaConf.structured(UserSimulatorConfig), OmegaConf.load(f"{a.run}/generated-user-config-0.yaml"))
    loader = build_scene_loader(user)
    out = {}
    for s in Path(a.scenes).read_text().split():
        ds = loader.get_data_source(s)
        vm = ds.map

        async def ego():
            async for k, m in _read(asl_of(a.run, s), {"traffic_session_request"}):
                return traj(m.logged_object_trajectories[0].trajectory)
        g = asyncio.run(ego())
        c = np.r_[g[len(g) // 2, 1:3], 0.0]
        areas = []
        for ar in vm.get_road_areas_within(c, a.radius):
            areas.append(dict(ext=np.asarray(ar.exterior_polygon.points[:, :2], np.float32),
                              holes=[np.asarray(h.points[:, :2], np.float32) for h in ar.interior_holes]))
        lanes = [dict(id=ln.id, center=np.asarray(ln.center.points[:, :2], np.float32),
                      left=None if ln.left_edge is None else np.asarray(ln.left_edge.points[:, :2], np.float32),
                      right=None if ln.right_edge is None else np.asarray(ln.right_edge.points[:, :2], np.float32),
                      next=sorted(ln.next_lanes), adj=sorted(ln.adj_lanes_left | ln.adj_lanes_right))
                 for ln in vm.get_lanes_within(c, a.radius)]
        out[s] = dict(areas=areas, lanes=lanes, location=ds.scene.location)
        print(s, len(areas), "areas", len(lanes), "lanes", flush=True)
        ds._map = None
        loader._provider._cache.pop(s, None)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print("map", len(out), "scenes ->", a.out, os.path.getsize(a.out) >> 20, "MB")


def one_frames(a) -> tuple:
    run, scene, width = a
    from PIL import Image

    async def go():
        fr = {}
        async for k, m in _read(asl_of(run, scene), {"driver_camera_image"}):
            im = m.camera_image
            if im.logical_id == "CAM_F0":
                b = io.BytesIO()
                x = Image.open(io.BytesIO(im.image_bytes))
                x.resize((width, round(width * x.height / x.width))).save(b, "JPEG", quality=80)
                fr[int(im.frame_end_us)] = b.getvalue()
        return fr
    return scene, asyncio.run(go())


def cmd_frames(a):
    with ProcessPoolExecutor(a.jobs) as ex:
        out = dict(ex.map(one_frames, [(a.run, s, a.width) for s in Path(a.scenes).read_text().split()]))
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print("frames", len(out), "scenes ->", a.out, os.path.getsize(a.out) >> 20, "MB")


def one_msgs(a) -> str:
    run, scene, out = a
    KIND = {"driver_session_request", "driver_camera_image", "driver_ego_trajectory", "route_request", "driver_request"}

    async def go():
        m_ = []
        async for k, m in _read(asl_of(run, scene), KIND):
            if k != "driver_camera_image" or m.camera_image.logical_id == "CAM_F0":
                m_.append((k, m.SerializeToString()))
        return m_
    pickle.dump(asyncio.run(go()), open(f"{out}/{scene}.pkl", "wb"), protocol=4)
    return scene


def cmd_msgs(a):
    """The driver-side messages of chosen scenes in arrival order (CAM_F0 images only), serialized, one pickle per scene: the input of
    c1_replay.py. --out is a directory."""
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        n = len(list(ex.map(one_msgs, [(a.run, s, a.out) for s in Path(a.scenes).read_text().split()])))
    print("msgs", n, "scenes ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("logs", cmd_logs), ("map", cmd_map), ("frames", cmd_frames), ("msgs", cmd_msgs)):
        p = sub.add_parser(n)
        p.add_argument("--run", required=True), p.add_argument("--out", required=True), p.add_argument("--scenes")
        p.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 4)), p.add_argument("--radius", type=float, default=90.0)
        p.add_argument("--width", type=int, default=960)
        p.set_defaults(fn=f)
    a = ap.parse_args()
    a.fn(a)
