"""Render probe: does rendering the PAI NuRec scene behind the log in time, or off the log path, break the image?

Reads a finished human rollout's ASL (rollout.asl: the runtime's own log of every batch_render_rgb request and the scene's
recorded ego trajectory, RolloutMetadata.ego_rig_recorded_ground_truth_trajectory) and sends edited copies of the runtime's
front-camera request (render_rgb, as the runtime sends it for one camera) straight to a renderer (the stack's NRE image,
serve-grpc) over gRPC: same scene, camera, intrinsics,
resolution, shutter, JPEG quality, ego mask. The camera pose is the logged ego pose composed with the request's own
rig_to_camera (sensor_pose = ego_pose @ rig_to_camera, alpasim_runtime/services/sensorsim_service.py).

Conditions, base time t (s after the first rendered frame), lags D (s):
  log        ego = log(t),      frame time t,     actors at t        (control)
  log_late   ego = log(t + D),  frame time t + D, actors at t + D    (control: the log itself later on)
  lag        ego = log(t),      frame time t + D, actors at t + D    (the human behind the log: what the runtime sends)
  lag_time   ego = log(t),      frame time t + D, actors at t        (only the timestamp moves)
  lag_actors ego = log(t),      frame time t,     actors at t + D    (only the actors move)
  shift      ego = log(t) moved left by S m in the rig frame, frame time t, actors at t
Actors at time T = the dynamic_objects of the logged request whose frame ends nearest T (log replay, local frame).

Runs in alpasim-base (grpc, alpasim_grpc, PIL):  render_probe.py --asl <rollout.asl> --addr 127.0.0.1:6800 --out <dir>
Writes <out>/<cond>.jpg, <out>/grid.png (labelled), <out>/probe.json.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import grpc
import numpy as np
from alpasim_grpc.v0 import logging_pb2, sensorsim_pb2, sensorsim_pb2_grpc


def read_asl(path):
    meta, reqs = None, {}
    with open(path, "rb") as f:
        while (h := f.read(4)):
            (n,) = struct.unpack(">L", h)
            e = logging_pb2.LogEntry.FromString(f.read(n))
            kind = e.WhichOneof("log_entry")
            if kind == "rollout_metadata":
                meta = e.rollout_metadata
            elif kind == "render_request":                       # one camera: the runtime calls render_rgb
                reqs[int(e.render_request.frame_end_us)] = e.render_request
            elif kind == "batch_render_request":
                for it in e.batch_render_request.items:
                    if "front" in it.camera_name or "F0" in it.camera_name:
                        reqs[int(it.request.frame_end_us)] = it.request
    return meta, reqs


def q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(m):
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    return np.array([w, x, y, z])


def to_T(p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = q2m([p.quat.w, p.quat.x, p.quat.y, p.quat.z]), [p.vec.x, p.vec.y, p.vec.z]
    return T


def set_pose(p, T):
    q = m2q(T[:3, :3])
    p.vec.x, p.vec.y, p.vec.z = map(float, T[:3, 3])
    p.quat.w, p.quat.x, p.quat.y, p.quat.z = map(float, q)


def log_pose(traj, t_us):
    """Recorded ego pose (4x4, local frame) at t_us: linear in position, nearest-sample rotation slerped by matrix blend."""
    ts = np.array([p.timestamp_us for p in traj.poses], dtype=np.int64)
    i = int(np.clip(np.searchsorted(ts, t_us), 1, len(ts) - 1))
    a, b = to_T(traj.poses[i - 1].pose), to_T(traj.poses[i].pose)
    s = float(np.clip((t_us - ts[i - 1]) / max(1, ts[i] - ts[i - 1]), 0, 1))
    T = a.copy()
    T[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
    u, _, vt = np.linalg.svd((1 - s) * a[:3, :3] + s * b[:3, :3])
    T[:3, :3] = u @ vt
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asl", required=True)
    ap.add_argument("--addr", default="127.0.0.1:6800")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=float, default=3.0)
    ap.add_argument("--lags", default="2,5,10")
    ap.add_argument("--shifts", default="1,2,4")
    ap.add_argument("--thumb", type=int, default=480, help="grid tile width px")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, reqs = read_asl(a.asl)
    traj = meta.ego_rig_recorded_ground_truth_trajectory
    ends = np.array(sorted(reqs), dtype=np.int64)
    t0 = int(ends[0])

    def req_near(T):
        return reqs[int(ends[np.argmin(np.abs(ends - T))])]

    base = req_near(t0 + int(a.t * 1e6))
    rig_to_cam = [np.linalg.inv(to_T(getattr(base.ego_pose, k))) @ to_T(getattr(base.sensor_pose, k)) for k in ("start_pose", "end_pose")]
    span = int(base.frame_end_us - base.frame_start_us)
    tb = int(base.frame_end_us)

    def make(ego_end_us, frame_end_us, actors_end_us, shift_m=0.0):
        r = sensorsim_pb2.RGBRenderRequest()
        r.CopyFrom(base)
        r.frame_end_us, r.frame_start_us = frame_end_us, frame_end_us - span
        for k, te, R2C in (("start_pose", ego_end_us - span, rig_to_cam[0]), ("end_pose", ego_end_us, rig_to_cam[1])):
            E = log_pose(traj, te)
            E[:3, 3] += E[:3, :3] @ np.array([0.0, shift_m, 0.0])            # rig y = left
            set_pose(getattr(r.ego_pose, k), E)
            set_pose(getattr(r.sensor_pose, k), E @ R2C)
        del r.dynamic_objects[:]
        r.dynamic_objects.extend(req_near(actors_end_us).dynamic_objects)
        return r

    conds = [("log", make(tb, tb, tb))]
    for d in [float(x) for x in a.lags.split(",")]:
        td = tb + int(d * 1e6)
        conds += [(f"log_late_{d:g}s", make(td, td, td)), (f"lag_{d:g}s", make(tb, td, td)),
                  (f"lag_time_{d:g}s", make(tb, td, tb)), (f"lag_actors_{d:g}s", make(tb, tb, td))]
    conds += [(f"shift_{s}m", make(tb, tb, tb, float(s))) for s in a.shifts.split(",")]

    stub = sensorsim_pb2_grpc.SensorsimServiceStub(grpc.insecure_channel(a.addr, options=[("grpc.max_receive_message_length", 1 << 28)]))
    info = {"asl": a.asl, "scene": base.scene_id, "t": a.t, "frame_end_us": tb, "log_range_s": [(traj.poses[0].timestamp_us - t0) / 1e6,
            (traj.poses[-1].timestamp_us - t0) / 1e6], "render_range_s": [0, (int(ends[-1]) - t0) / 1e6], "conds": {},
            # frame check: during the force-GT replay (first frames) the runtime's ego pose is the log's
            "first_frame_ego_vs_log_m": float(np.linalg.norm(to_T(reqs[t0].ego_pose.end_pose)[:3, 3] - log_pose(traj, t0)[:3, 3])),
            "base_frame_ego_vs_log_m": float(np.linalg.norm(to_T(base.ego_pose.end_pose)[:3, 3] - log_pose(traj, tb)[:3, 3]))}
    imgs = {}
    for name, r in conds:
        try:
            ret = stub.render_rgb(r, timeout=300)
            it = sensorsim_pb2.BatchRGBRenderReturnItem(success=True, result=ret)
        except grpc.RpcError as e:
            it = sensorsim_pb2.BatchRGBRenderReturnItem(success=False, error_message=str(e.details()))
        info["conds"][name] = {"ok": it.success, "err": it.error_message, "n_actors": len(r.dynamic_objects),
                               "ego_xy": [round(r.ego_pose.end_pose.vec.x, 2), round(r.ego_pose.end_pose.vec.y, 2)],
                               "frame_s": round((r.frame_end_us - t0) / 1e6, 2)}
        if it.success:
            (out / f"{name}.jpg").write_bytes(it.result.image_bytes)
            imgs[name] = it.result.image_bytes
        print(name, info["conds"][name], flush=True)
    (out / "probe.json").write_text(json.dumps(info, indent=1))

    from PIL import Image, ImageDraw
    lags = [x for x in a.lags.split(",")]
    rows = [["log"] + [f"shift_{s}m" for s in a.shifts.split(",")]] + \
           [[f"log_late_{d}s", f"lag_{d}s", f"lag_time_{d}s", f"lag_actors_{d}s"] for d in lags]
    W = a.thumb
    first = Image.open(io.BytesIO(next(iter(imgs.values()))))
    H = int(W * first.height / first.width)
    ncol = max(len(r) for r in rows)
    grid = Image.new("RGB", (ncol * W, len(rows) * (H + 22)), "white")
    dr = ImageDraw.Draw(grid)
    for i, row in enumerate(rows):
        for j, name in enumerate(row):
            x, y = j * W, i * (H + 22)
            if name in imgs:
                grid.paste(Image.open(io.BytesIO(imgs[name])).convert("RGB").resize((W, H)), (x, y + 22))
            c = info["conds"].get(name, {})
            dr.text((x + 4, y + 4), f"{name}  (frame t={c.get('frame_s')} s, {c.get('n_actors')} actors)", fill="black")
    grid.save(out / "grid.png", optimize=True)
    print("DONE", out / "grid.png", flush=True)


if __name__ == "__main__":
    main()
