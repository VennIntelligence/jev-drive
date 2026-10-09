"""SH30 as an AlpaSim nuPlan-track driver: egodriver.EgodriverService around experiments/alpasim/lib/sh30_core.py.

Every `drive` call runs the model on what the session has received (no cached plan, no straight-line fallback); a call whose inputs
are incomplete, or whose inference raises, is aborted with a gRPC error and counted, so a failure shows as a failed rollout.

What the simulator gives -> what the core is fed (docs/alpasim.md, "What a nuPlan-track driver actually receives"):
  camera    CAM_F0 only (the other 7 are dropped undecoded); JPEG -> openpilot road + wide frames at submit time, with the session's
            own calibration (`opencv_pinhole_param`, `rig_to_camera`); keyframes are matched by `frame_end_us` on the 0.5 s grid
  ego       local -> rig poses at the keyframe times -> x, y, yaw in the rig frame of t0; velocity / acceleration from DynamicState.
            A state whose velocity disagrees with the pose difference but agrees after a rotation by the pose yaw is used rotated
            (the first sample of a rollout arrives that way) and counted (`state_rotated`)
  command   the shipped samples' rule on the route (navsim_transfuser_challenge.navigation.command_from_route, imported)
  output    8 rear-axle poses at 0.5 s -> the rollout's local frame at the pose of t0 -> 10 Hz trajectory, with the shipped LTF
            sample's own functions (navsim_transfuser_challenge.trajectory, imported unchanged; Apache-2.0, NVIDIA)
  traffic   right-hand convention unless SH30_LHT=1 (the driver is not told the map)

Yaw damping (decision 205): SH30_MOTION = the share of the ego's own turning shown to the model in the synthesised slots (sh30_core.py;
default 1 = unchanged). SH30_MOTION_GATE=route applies it only while the route message is a straight line that passes the ego
(`route_line`: the road ahead is straight, so any turning is the ego's own); elsewhere the model sees all of it.

Environment: ALPASIM_DRIVER_HOST / ALPASIM_DRIVER_PORT, ALPASIM_SRC (AlpaSim checkout: gRPC stubs and the LTF sample), SH30_TAG
(op_parity run tag, default SH30-F-s0), SH30_COLD (backwarp | zero), SH30_SYNTH (cpu | gpu slot warp, sh30_core.py), SH30_DEVICE, ALPASIM_DRIVER_LOG_DIR (drive.jsonl: one record per
call with inputs, plan, stage times; images.jsonl), SH30_DUMP (number of sessions whose model frames and JPEGs are saved to <log dir>/dump).
Run with envs/op-train:  python experiments/alpasim/lib/sh30_driver.py
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
    import alpasim_grpc  # noqa: F401  (an installed package: the submission image)
    API = alpasim_grpc.API_VERSION_MESSAGE
except Exception:                                    # the checkout's stubs without its package metadata (the box)
    for k in [k for k in sys.modules if k.split(".")[0] == "alpasim_grpc"]:
        del sys.modules[k]
    _pkg = types.ModuleType("alpasim_grpc")
    _pkg.__path__ = [str(SRC / "src/grpc/alpasim_grpc")]
    sys.modules["alpasim_grpc"] = _pkg
    from alpasim_grpc.v0 import common_pb2 as _c
    import tomllib
    API = _c.VersionId.APIVersion(**dict(zip(("major", "minor", "patch"), map(int, tomllib.loads(
        (SRC / "src/grpc/pyproject.toml").read_text())["project"]["version"].split(".")))))
sys.path.insert(0, str(SRC / "e2e_challenge/sample_submission_simscale_navsim_transfuser"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grpc  # noqa: E402
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc  # noqa: E402
from navsim_transfuser_challenge.navigation import command_from_route  # noqa: E402
from navsim_transfuser_challenge.trajectory import build_trajectory_from_plan, make_cached_plan, yaw_from_quat  # noqa: E402

import sh30_core as C  # noqa: E402

LOG = logging.getLogger("sh30")
CAM, STEP_US, TOL_US = "CAM_F0", 500_000, 2_000


def quat_R(q) -> np.ndarray:
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def route_line(wp, tol: float = 0.5, reach: float = 2.5, max_deg: float = 20.0):
    """Route waypoints (n, 2) in the rig frame (NaN = padding) -> (straight, direction rad, signed offset m of the ego from the line).
    straight: at least 3 waypoints within `tol` m of their own least-squares line, that line passes within `reach` m of the ego and
    points within `max_deg` of the ego heading. Uses the route's shape only, so the ego's own yaw does not change the answer."""
    wp = np.asarray(wp, np.float64).reshape(-1, 2)
    wp = wp[~np.isnan(wp).any(1)]
    if len(wp) < 3:
        return False, 0.0, 0.0
    c = wp.mean(0)
    u = np.linalg.svd(wp - c)[2][0]
    u = u if u[0] >= 0 else -u
    n = np.array([-u[1], u[0]])
    dev, off, phi = float(np.abs((wp - c) @ n).max()), float(-c @ n), float(np.arctan2(u[1], u[0]))
    return dev <= tol and abs(off) <= reach and abs(np.degrees(phi)) <= max_deg, phi, off


def rot(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Session:
    def __init__(self, req, n: int):
        cams = {c.logical_id: c for c in req.rollout_spec.vehicle.available_cameras}
        if CAM not in cams:
            raise ValueError(f"{CAM} not in the rollout's cameras {sorted(cams)}")
        c = cams[CAM]
        if c.intrinsics.WhichOneof("camera_param") != "opencv_pinhole_param":
            raise ValueError(f"{CAM}: unsupported camera model {c.intrinsics.WhichOneof('camera_param')}")
        p = c.intrinsics.opencv_pinhole_param
        k, tg = list(p.radial_coeffs) + [0.0] * 6, list(p.tangential_coeffs) + [0.0] * 2
        if any(k[3:6]) or any(p.thin_prism_coeffs):
            raise ValueError(f"{CAM}: rational / thin-prism distortion is not supported")
        t = c.rig_to_camera.vec
        self.cam = C.camera([[p.focal_length_x, 0, p.principal_point_x], [0, p.focal_length_y, p.principal_point_y], [0, 0, 1]],
                            quat_R(c.rig_to_camera.quat), [t.x, t.y, t.z], [k[0], k[1], tg[0], tg[1], k[2]])
        self.n, self.scene = n, req.debug_info.scene_id if req.HasField("debug_info") else ""
        self.frames, self.jpeg, self.poses, self.states = {}, {}, {}, {}
        self.cmd = np.array([0, 0, 0, 1], np.float32)
        self.lock = threading.Lock()
        self.count = dict(drive=0, inference=0, inference_error=0, input_error=0, state_rotated=0, cold=0)

    def velocity(self, ts: int):
        """Body velocity (2,), acceleration (2,), yaw rate at a pose time; the state rotated by the pose yaw when that matches the
        pose-difference velocity better (see the module docstring)."""
        v, a, w = self.states[ts]
        near = [t for t in self.poses if t != ts and abs(t - ts) <= STEP_US + TOL_US]
        if not near:
            return v, a, w, False
        o = min(near, key=lambda t: (abs(t - ts), t))
        yaw = self.poses[ts].pose
        fd = rot(yaw_from_quat(yaw.quat)).T @ ((_xy(self.poses[ts]) - _xy(self.poses[o])) / ((ts - o) * 1e-6))
        Rz = rot(yaw_from_quat(yaw.quat))
        if np.linalg.norm(Rz @ v - fd) + 0.5 < np.linalg.norm(v - fd):
            return Rz @ v, Rz @ a, w, True
        return v, a, w, False


def _xy(p) -> np.ndarray:
    return np.array([p.pose.vec.x, p.pose.vec.y])


class Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, core: C.Core, log_dir: Path, dump: int = 0, lht: bool = False, gate: str = ""):
        self.core, self.lht, self.dump, self.dir, self.gate = core, lht, dump, log_dir, gate
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
        self.out.write(json.dumps({"kind": "start", "t": time.time(), "session": req.session_uuid, "n": s.n, "scene": s.scene,
                                   "cam": {k: np.asarray(v, float).round(6).tolist() for k, v in s.cam.items()}}) + "\n")
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self.lock:
            s = self.sessions.pop(req.session_uuid, None)
        if s is not None:
            self.out.write(json.dumps({"kind": "close", "t": time.time(), "session": req.session_uuid, "scene": s.scene, **s.count}) + "\n")
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        s, im = self._s(req.session_uuid, ctx), req.camera_image
        if im.logical_id != CAM:
            return common_pb2.Empty()
        t0 = time.perf_counter()
        try:
            fr = C.pack(im.image_bytes, s.cam)
        except Exception as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{CAM} decode failed: {e!r}")
        ts = int(im.frame_end_us)
        with s.lock:
            s.frames[ts] = fr
            for k in sorted(s.frames)[:-4]:
                del s.frames[k]
            if s.n < self.dump:
                s.jpeg[ts] = bytes(im.image_bytes)
        self.img.write(json.dumps({"session": req.session_uuid, "ts": ts, "start_us": int(im.frame_start_us), "bytes": len(im.image_bytes),
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
        s.route0 = [req.route.waypoints[0].x, req.route.waypoints[0].y] if req.route.waypoints else None
        s.line = route_line([[w.x, w.y] for w in req.route.waypoints])
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, req, ctx):
        self._s(req.session_uuid, ctx)
        return common_pb2.Empty()

    def motion(self, s) -> float:
        """The share of the ego's turning shown to the model at this decision (module docstring)."""
        return self.core.motion if self.gate != "route" or getattr(s, "line", (False,))[0] else 1.0

    def drive(self, req, ctx):
        s, now, tq = self._s(req.session_uuid, ctx), int(req.time_now_us), int(req.time_query_us)
        t_in = time.perf_counter()
        with s.lock:
            s.count["drive"] += 1
            ts = [t for t in s.poses if t <= now and t in s.frames]
            t0 = max(ts) if ts else None
            if t0 is None or now - t0 > TOL_US:
                s.count["input_error"] += 1
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, f"no CAM_F0 frame and ego pose at Drive time {now} (latest pair {t0})")
            keys = [t0]
            while len(keys) < 4 and (k := next((t for t in s.frames if abs(t - (keys[-1] - STEP_US)) <= TOL_US and t in s.poses), None)):
                keys.append(k)
            keys = keys[::-1]
            anchor, yaw0 = s.poses[t0], yaw_from_quat(s.poses[t0].pose.quat)
            R0 = rot(yaw0).T
            pose = np.array([[*(R0 @ (_xy(s.poses[k]) - _xy(anchor))), yaw_from_quat(s.poses[k].pose.quat) - yaw0] for k in keys])
            pose[:, 2] = (pose[:, 2] + np.pi) % (2 * np.pi) - np.pi
            dyn = [s.velocity(k) for k in keys]
            s.count["state_rotated"] += int(dyn[-1][3])
            s.count["cold"] += len(keys) < 4
            frames, cmd = [s.frames[k] for k in keys], np.asarray(s.cmd, np.float32).copy()
        t_q = time.perf_counter()
        try:
            with self.gpu:
                t_g = time.perf_counter()
                mo = self.motion(s)
                o = self.core.plan(frames, pose, np.array([d[0] for d in dyn]), dyn[-1][1], cmd, s.cam["t"], yaw_rate=dyn[0][2], lht=self.lht, motion=mo)
        except Exception as e:
            s.count["inference_error"] += 1
            LOG.exception("inference failed, session %s t %d", req.session_uuid, now)
            ctx.abort(grpc.StatusCode.INTERNAL, f"SH30 inference failed: {e!r}")
        s.count["inference"] += 1
        plan = make_cached_plan(t0, anchor, o["poses"])
        traj = build_trajectory_from_plan(plan, anchor, now, tq)
        t_out = time.perf_counter()
        rec = {"kind": "drive", "t": time.time(), "session": req.session_uuid, "scene": s.scene, "now": now, "t0": t0, "k": s.count["drive"] - 1,
               "n_keys": len(keys), "n_slots": int(o["valid"].sum()), "cmd": int(np.argmax(cmd)), "route0": getattr(s, "route0", None),
               "motion": mo, "line": [bool(getattr(s, "line", (0, 0, 0))[0]), *np.round(getattr(s, "line", (0, 0.0, 0.0))[1:], 4).tolist()],
               "rotated": [bool(d[3]) for d in dyn], "ego": o["ego"].round(5).tolist(), "hist": o["hist"].round(4).tolist(),
               "anchor": [anchor.pose.vec.x, anchor.pose.vec.y, yaw0], "poses": o["poses"].round(4).tolist(), "n_out": len(traj.poses),
               "ms": {**{k: round(v, 2) for k, v in o["ms"].items()}, "prep": round(1e3 * (t_q - t_in), 2), "wait": round(1e3 * (t_g - t_q), 2),
                      "total": round(1e3 * (t_out - t_in), 2)}}
        self.out.write(json.dumps(rec) + "\n")
        if s.n < self.dump:
            np.savez_compressed(self.dir / "dump" / f"s{s.n:02d}_k{rec['k']}.npz", cur=o["cur"], valid=o["valid"], mu=o["mu"], poses=o["poses"],
                                ego=o["ego"], cam_t=s.cam["t"], scene=s.scene, jpeg=np.frombuffer(s.jpeg.get(t0, b""), np.uint8))
        return egodriver_pb2.DriveResponse(trajectory=traj)

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id=f"jev-{self.core.tag}-{self.core.cold}", git_hash=os.environ.get("SH30_GIT_HASH", "local"),
                                    grpc_api_version=API)


def warm_workers(fn) -> ThreadPoolExecutor:
    """The gRPC worker pool (ALPASIM_DRIVER_GRPC_WORKERS threads) with `fn` (one inference) already run in every thread: the CUDA libraries
    initialise per thread, which costs the first `drive` a thread serves 1.35 s in the submission image (torch +cu126, RTX 3090;
    results/tokyo_image_smoke.md), paid here before the port opens instead of inside the first rollouts."""
    n = int(os.environ.get("ALPASIM_DRIVER_GRPC_WORKERS", "8"))
    pool, bar, lock = ThreadPoolExecutor(max_workers=n), threading.Barrier(n), threading.Lock()

    def one(_):
        bar.wait()                                       # every worker thread exists before any of them returns to the pool
        with lock:
            fn()
    t0 = time.time()
    list(pool.map(one, range(n)))
    LOG.info("%d worker threads warmed in %.1f s", n, time.time() - t0)
    return pool


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    log_dir = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))
    t0 = time.time()
    core = C.Core(os.environ.get("SH30_TAG", "SH30-F-s0"), os.environ.get("SH30_DEVICE", "cuda"), os.environ.get("SH30_COLD", "backwarp"),
                  float(os.environ.get("SH30_MOTION", "1")), os.environ.get("SH30_SYNTH", "cpu"))
    z = np.zeros(C.FRAME, np.uint8)
    for m in (1, 4, 4):                                 # warm-up: both slot counts compiled before the port opens
        core.plan([z] * m, np.zeros((m, 3)), np.zeros((m, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])
    LOG.info("%s (%s, motion %.2f, synth %s) ready in %.1f s, VRAM %.2f GiB", core.tag, core.cold, core.motion, core.synth, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30)
    drv = Driver(core, log_dir, int(os.environ.get("SH30_DUMP", "0")), os.environ.get("SH30_LHT", "0") == "1", os.environ.get("SH30_MOTION_GATE", ""))
    server = grpc.server(warm_workers(lambda: core.plan([z] * 4, np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])))
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
