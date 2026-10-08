"""WA-JEPA as an AlpaSim nuPlan-track driver: egodriver.EgodriverService around experiments/alpasim/lib/wajepa_core.py (their shipped
NAVSIM agent, unmodified). Same service shape as sh30_driver.py: every `drive` runs the model on what the session has received (no cached
plan, no fallback); incomplete inputs or an inference exception abort the call with a gRPC error and are counted.

What the simulator gives -> what WA-JEPA is fed:
  cameras   CAM_L0 / CAM_F0 / CAM_R0 / CAM_B0 JPEGs as rendered (nuPlan rig, 1920 x 1080; the other 4 are dropped undecoded), decoded and
            area-resized to 512 x 256 at submit time (wajepa_core.decode); keyframes matched by `frame_end_us` on the 0.5 s grid
  ego       local -> rig poses at the keyframe times -> x, y, yaw in the rig frame of t0 (rig = rear axle, as the shipped samples
            assume); velocity / acceleration from DynamicState with sh30_driver's guard for the unrotated first sample (`state_rotated`)
  command   the shipped samples' rule on the route (navsim_transfuser_challenge.navigation.command_from_route, imported)
  output    their 8 rear-axle poses at 0.5 s -> local frame at the t0 pose -> 10 Hz trajectory with the LTF sample's functions
  cold      WAJ_COLD=repeat (default; WA-JEPA's own closed-loop padding) or cv (see wajepa_core)

Environment: ALPASIM_DRIVER_HOST / _PORT, ALPASIM_SRC, ALPASIM_DRIVER_LOG_DIR (drive.jsonl, images.jsonl), WAJ_COLD, WAJ_DEVICE,
WAJ_DUMP (number of sessions whose fed frames are saved to <log dir>/dump: decision 3 of each, all decisions of the first 3), WAJ_REPO / WAJ_CKPT / WAJ_CFG (wajepa_core).
Run with envs/wajepa, cwd = the WA-JEPA checkout, PYTHONPATH=<wajepa>:<navsim>:  python experiments/alpasim/lib/wajepa_driver.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("ALPASIM_SRC") or Path(os.environ.get("DATA_DIR", "/nonexistent")) / "third_party/alpasim")
try:
    import alpasim_grpc  # noqa: F401
    API = alpasim_grpc.API_VERSION_MESSAGE
except Exception:                                    # the checkout's stubs without its package metadata (the box)
    for k in [k for k in sys.modules if k.split(".")[0] == "alpasim_grpc"]:
        del sys.modules[k]
    _pkg = types.ModuleType("alpasim_grpc")
    _pkg.__path__ = [str(SRC / "src/grpc/alpasim_grpc")]
    sys.modules["alpasim_grpc"] = _pkg
    from alpasim_grpc.v0 import common_pb2 as _c
    try:
        import tomllib
    except ModuleNotFoundError:                      # python 3.10 (envs/wajepa)
        import tomli as tomllib
    API = _c.VersionId.APIVersion(**dict(zip(("major", "minor", "patch"), map(int, tomllib.loads(
        (SRC / "src/grpc/pyproject.toml").read_text())["project"]["version"].split(".")))))
sys.path.insert(0, str(SRC / "e2e_challenge/sample_submission_simscale_navsim_transfuser"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grpc  # noqa: E402
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc  # noqa: E402
from navsim_transfuser_challenge.navigation import command_from_route  # noqa: E402
from navsim_transfuser_challenge.trajectory import build_trajectory_from_plan, make_cached_plan, yaw_from_quat  # noqa: E402

import wajepa_core as C  # noqa: E402

LOG = logging.getLogger("wajepa")
STEP_US, TOL_US = 500_000, 2_000


def rot(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _xy(p) -> np.ndarray:
    return np.array([p.pose.vec.x, p.pose.vec.y])


class Session:
    def __init__(self, req, n: int):
        cams = {c.logical_id for c in req.rollout_spec.vehicle.available_cameras}
        if miss := [c for c in C.CAMS if c not in cams]:
            raise ValueError(f"cameras {miss} not in the rollout's cameras {sorted(cams)}")
        self.n, self.scene = n, req.debug_info.scene_id if req.HasField("debug_info") else ""
        self.frames, self.poses, self.states = {}, {}, {}       # frames: ts -> {cam: RGB}
        self.cmd = np.array([0, 0, 0, 1], np.float32)
        self.lock = threading.Lock()
        self.count = dict(drive=0, inference=0, inference_error=0, input_error=0, state_rotated=0, cold=0)

    def velocity(self, ts: int):
        """sh30_driver.Session.velocity: the state rotated by the pose yaw when that matches the pose-difference velocity better."""
        v, a, w = self.states[ts]
        near = [t for t in self.poses if t != ts and abs(t - ts) <= STEP_US + TOL_US]
        if not near:
            return v, a, w, False
        o = min(near, key=lambda t: (abs(t - ts), t))
        Rz = rot(yaw_from_quat(self.poses[ts].pose.quat))
        fd = Rz.T @ ((_xy(self.poses[ts]) - _xy(self.poses[o])) / ((ts - o) * 1e-6))
        if np.linalg.norm(Rz @ v - fd) + 0.5 < np.linalg.norm(v - fd):
            return Rz @ v, Rz @ a, w, True
        return v, a, w, False

    def complete(self, ts) -> bool:
        return ts in self.poses and len(self.frames.get(ts, ())) == len(C.CAMS)


class Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, core: C.Core, log_dir: Path, dump: int = 0):
        self.core, self.dump, self.dir = core, dump, log_dir
        self.sessions, self.lock, self.gpu, self.nsess = {}, threading.Lock(), threading.Lock(), 0
        log_dir.mkdir(parents=True, exist_ok=True)
        self.out = open(log_dir / "drive.jsonl", "a", buffering=1)
        self.img = open(log_dir / "images.jsonl", "a", buffering=1)
        if dump:
            (log_dir / "dump").mkdir(exist_ok=True)

    def _s(self, uuid, ctx) -> Session:
        s = self.sessions.get(uuid)
        if s is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s

    def start_session(self, req, ctx):
        try:
            with self.lock:
                s = self.sessions[req.session_uuid] = Session(req, self.nsess)
                self.nsess += 1
        except ValueError as e:
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        self.out.write(json.dumps({"kind": "start", "t": time.time(), "session": req.session_uuid, "n": s.n, "scene": s.scene}) + "\n")
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self.lock:
            s = self.sessions.pop(req.session_uuid, None)
        if s is not None:
            self.out.write(json.dumps({"kind": "close", "t": time.time(), "session": req.session_uuid, "scene": s.scene, **s.count}) + "\n")
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        s, im = self._s(req.session_uuid, ctx), req.camera_image
        if im.logical_id not in C.CAMS:
            return common_pb2.Empty()
        t0 = time.perf_counter()
        try:
            rgb = C.decode(im.image_bytes)
        except Exception as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{im.logical_id} decode failed: {e!r}")
        ts = int(im.frame_end_us)
        with s.lock:
            s.frames.setdefault(ts, {})[im.logical_id] = rgb
            for k in sorted(s.frames)[:-5]:
                del s.frames[k]
        self.img.write(json.dumps({"session": req.session_uuid, "cam": im.logical_id, "ts": ts, "bytes": len(im.image_bytes),
                                   "pack_ms": round(1e3 * (time.perf_counter() - t0), 2)}) + "\n")
        return common_pb2.Empty()

    def submit_egomotion_observation(self, req, ctx):
        s, P, D = self._s(req.session_uuid, ctx), req.trajectory.poses, req.dynamic_states
        if len(D) != len(P):
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{len(P)} poses with {len(D)} dynamic states")
        with s.lock:
            for p, d in zip(P, D):
                ts = int(p.timestamp_us)
                s.poses[ts] = p
                s.states[ts] = (np.array([d.linear_velocity.x, d.linear_velocity.y]),
                                np.array([d.linear_acceleration.x, d.linear_acceleration.y]), float(d.angular_velocity.z))
        return common_pb2.Empty()

    def submit_route(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        s.cmd = command_from_route(req.route)
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, req, ctx):
        self._s(req.session_uuid, ctx)
        return common_pb2.Empty()

    def drive(self, req, ctx):
        s, now, tq = self._s(req.session_uuid, ctx), int(req.time_now_us), int(req.time_query_us)
        t_in = time.perf_counter()
        with s.lock:
            s.count["drive"] += 1
            ts = [t for t in s.frames if t <= now and s.complete(t)]
            t0 = max(ts) if ts else None
            if t0 is None or now - t0 > TOL_US:
                s.count["input_error"] += 1
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, f"no complete {len(C.CAMS)}-camera frame and ego pose at Drive time {now} "
                                                               f"(latest {t0})")
            keys = [t0]
            while len(keys) < 4 and (k := next((t for t in s.frames if abs(t - (keys[-1] - STEP_US)) <= TOL_US and s.complete(t)), None)):
                keys.append(k)
            keys = keys[::-1]
            anchor, yaw0 = s.poses[t0], yaw_from_quat(s.poses[t0].pose.quat)
            R0 = rot(yaw0).T
            pose = np.array([[*(R0 @ (_xy(s.poses[k]) - _xy(anchor))), yaw_from_quat(s.poses[k].pose.quat) - yaw0] for k in keys])
            pose[:, 2] = (pose[:, 2] + np.pi) % (2 * np.pi) - np.pi
            dyn = [s.velocity(k) for k in keys]
            s.count["state_rotated"] += int(dyn[-1][3])
            s.count["cold"] += len(keys) < 4
            frames, cmd = [dict(s.frames[k]) for k in keys], np.asarray(s.cmd, np.float32).copy()
        t_q = time.perf_counter()
        try:
            with self.gpu:
                t_g = time.perf_counter()
                o = self.core.plan(frames, pose, np.array([d[0] for d in dyn]), dyn[-1][1], cmd, yaw_rate=dyn[0][2])
        except Exception as e:
            s.count["inference_error"] += 1
            LOG.exception("inference failed, session %s t %d", req.session_uuid, now)
            ctx.abort(grpc.StatusCode.INTERNAL, f"WA-JEPA inference failed: {e!r}")
        s.count["inference"] += 1
        traj = build_trajectory_from_plan(make_cached_plan(t0, anchor, o["poses"]), anchor, now, tq)
        t_out = time.perf_counter()
        v, a = dyn[-1][0], dyn[-1][1]
        rec = {"kind": "drive", "t": time.time(), "session": req.session_uuid, "scene": s.scene, "now": now, "t0": t0, "k": s.count["drive"] - 1,
               "n_keys": len(keys), "cmd": int(np.argmax(cmd)), "rotated": [bool(d[3]) for d in dyn],
               "ego": [float(v[0]), float(v[1]), float(a[0]), float(a[1])], "hist": o["hist"].round(4).tolist(),
               "anchor": [anchor.pose.vec.x, anchor.pose.vec.y, yaw0], "poses": o["poses"].round(4).tolist(), "n_out": len(traj.poses),
               "ms": {**{k: round(x, 2) for k, x in o["ms"].items()}, "prep": round(1e3 * (t_q - t_in), 2), "wait": round(1e3 * (t_g - t_q), 2),
                      "total": round(1e3 * (t_out - t_in), 2)}}
        self.out.write(json.dumps(rec) + "\n")
        if s.n < self.dump and (rec["k"] == 3 or s.n < 3):      # decision 3 = a NAVSIM token's t0; every decision of the first 3 sessions
            np.savez(self.dir / "dump" / f"s{s.n:02d}_k{rec['k']}.npz", poses=o["poses"], hist=o["hist"], scene=s.scene,
                                **{f"{c}_{j}": f[c] for j, f in enumerate(frames) for c in C.CAMS})
        return egodriver_pb2.DriveResponse(trajectory=traj)

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id=f"jev-{self.core.tag}-{self.core.cold}", git_hash=os.environ.get("WAJ_GIT_HASH", "local"),
                                    grpc_api_version=API)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    log_dir = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))
    t0 = time.time()
    core = C.Core(os.environ.get("WAJ_DEVICE", "cuda"), os.environ.get("WAJ_COLD", "repeat"))
    z = {c: np.zeros(C.HW + (3,), np.uint8) for c in C.CAMS}
    for m in (1, 4):                                    # warm-up before the port opens
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]))
    LOG.info("%s (%s) ready in %.1f s, VRAM %.2f GiB", core.tag, core.cold, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30)
    drv = Driver(core, log_dir, int(os.environ.get("WAJ_DUMP", "0")))
    server = grpc.server(ThreadPoolExecutor(max_workers=int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "8"))))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(drv, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    server.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: server.stop(grace=0.0))
    LOG.info("listening on %s:%d", host, port)
    try:
        server.wait_for_termination()
    finally:
        (log_dir / "vram.json").write_text(json.dumps({"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                       "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}))


if __name__ == "__main__":
    main()
