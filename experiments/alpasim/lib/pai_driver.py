"""The SH30 family as an AlpaSim PAI-track driver: egodriver.EgodriverService around experiments/alpasim/lib/pai_core.py. Serving-side
adaptation of the nuPlan-track checkpoints (sh30_driver.py), no training. What the PAI simulator gives -> what the model is fed:

  camera    PAI_CAM (default camera_front_wide_120fov; the other five are dropped undecoded), the session's own CameraSpec (f-theta)
            scaled to the delivered JPEG -> openpilot road + wide frames at submit time. 10 Hz, so the 8 policy slots are rendered frames
            (frame_end_us within 30 ms of t0 - 1.4 .. t0 at 0.2 s); a slot the session has no frame for is filled by PAI_COLD
            (backwarp | zero, sh30_core's cold-start rules)
  ego       local -> rig poses, interpolated to t0 - 1.5 / 1.0 / 0.5 / 0 s (missing ones: constant-velocity back-extrapolation);
            velocity / acceleration from DynamicState, with sh30_driver's guard for a state that arrives unrotated
  command   the shipped NAVSIM samples' rule on the route (navigation.command_from_route, imported)
  output    8 rear-axle poses at 0.5 s -> the rollout's local frame at the pose of t0 -> 10 Hz trajectory, the LTF sample's functions.
            The rig origin is taken as the rear axle (the lever arm of the plan is the camera's x, y in the rig frame)
  rate      one inference per PAI_EVERY `drive` calls (default 1 = every call, 10 Hz); between inferences the cached plan is returned
  traffic   right-hand convention unless PAI_LHT=1 (the driver is not told the country)

A `drive` whose inputs are incomplete or whose inference raises is aborted with a gRPC error and counted (no straight-line fallback).
Environment: ALPASIM_DRIVER_HOST / ALPASIM_DRIVER_PORT, SH30_TAG (op_parity run tag), SH30_DEVICE, ALPASIM_DRIVER_LOG_DIR (drive.jsonl:
one record per call; start.jsonl: the full rollout spec of every session), PAI_DUMP (number of sessions whose model frames and source
JPEG are saved once per second to <log dir>/dump), JEV_VCONT / JEV_LEAD (the served speed profile: serve_fix.py; both off by default).
Run inside the nuPlan submission image with this directory's pai_*.py mounted (docs/alpasim.md, "PAI track on the Tokyo box").
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np

import sh30_driver as S
from sh30_driver import common_pb2, egodriver_pb2, egodriver_pb2_grpc, grpc

import pai_core as PC
import serve_fix as FX
from pai_core import C, I

LOG = logging.getLogger("pai")
CAM = os.environ.get("PAI_CAM", "camera_front_wide_120fov")
STEP_US, KEEP_US, TOL_US = 100_000, 1_650_000, 30_000
T_KEY_US = np.round(I.T_KEY * 1e6).astype(np.int64)


def camera_spec(c) -> dict:
    """A sensorsim CameraSpec -> the plain dict pai_core.project reads."""
    kind, p = c.WhichOneof("camera_param"), getattr(c, c.WhichOneof("camera_param"))
    spec = {"c": [p.principal_point_x, p.principal_point_y], "native": [c.resolution_w, c.resolution_h],
            "windshield": c.WhichOneof("external_distortion") or ""}
    if kind == "ftheta_param":
        cde = p.linear_cde
        spec.update(model="ftheta", fw=list(p.angle_to_pixeldist_poly), bw=list(p.pixeldist_to_angle_poly), max_angle=p.max_angle,
                    cde=[cde.linear_c or 1.0, cde.linear_d, cde.linear_e])
    elif kind == "opencv_fisheye_param":
        spec.update(model="fisheye", f=[p.focal_length_x, p.focal_length_y], k=list(p.radial_coeffs), max_angle=p.max_angle)
    else:
        spec.update(model="pinhole", f=[p.focal_length_x, p.focal_length_y], k=list(p.radial_coeffs), p=list(p.tangential_coeffs))
    return spec


class Session:
    def __init__(self, req, n: int):
        cams = {c.logical_id: c for c in req.rollout_spec.vehicle.available_cameras}
        if CAM not in cams:
            raise ValueError(f"{CAM} not in the rollout's cameras {sorted(cams)}")
        c = cams[CAM]
        t = c.rig_to_camera.vec
        self.spec, self.R, self.cam_t = camera_spec(c.intrinsics), S.quat_R(c.rig_to_camera.quat), [t.x, t.y, t.z]
        self.n, self.scene = n, req.debug_info.scene_id if req.HasField("debug_info") else ""
        self.fix = FX.new(req.rollout_spec.vehicle, t.x)
        self.maps, self.frames, self.jpeg, self.poses, self.states = None, {}, {}, {}, {}
        self.cmd, self.route0, self.plan, self.anchor = np.array([0, 0, 0, 1], np.float32), None, None, None
        self.lock = threading.Lock()
        self.count = dict(drive=0, inference=0, inference_error=0, input_error=0, state_rotated=0, cold=0, images=0)

    def pose_at(self, ts: int):
        """(x, y, yaw) of the rig in the local frame at ts, linear between the two nearest egomotion samples; None outside them."""
        t = np.array(sorted(self.poses), np.int64)
        if not len(t) or ts < t[0] - TOL_US or ts > t[-1] + TOL_US:
            return None
        xy = np.array([S._xy(self.poses[k]) for k in t])
        yaw = np.unwrap([S.yaw_from_quat(self.poses[k].pose.quat) for k in t])
        return np.array([np.interp(ts, t, xy[:, 0]), np.interp(ts, t, xy[:, 1]), np.interp(ts, t, yaw)])

    def velocity(self, ts: int):
        """Body velocity (2,), acceleration (2,), yaw rate of the egomotion sample nearest ts; rotated by the pose yaw when that
        matches the pose-difference velocity better (sh30_driver.Session.velocity)."""
        k = min(self.states, key=lambda t: abs(t - ts))
        v, a, w = self.states[k]
        p1, p0 = self.pose_at(k), self.pose_at(k - STEP_US)
        if p1 is None or p0 is None or k - STEP_US < min(self.poses):
            return v, a, w, False
        Rz = S.rot(p1[2])
        fd = Rz.T @ ((p1[:2] - p0[:2]) / (STEP_US * 1e-6))
        if np.linalg.norm(Rz @ v - fd) + 0.5 < np.linalg.norm(v - fd):
            return Rz @ v, Rz @ a, w, True
        return v, a, w, False


class Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, core: C.Core, log_dir: Path, cold: str, every: int = 1, dump: int = 0, lht: bool = False):
        self.core, self.cold, self.every, self.dump, self.lht, self.dir = core, cold, every, dump, lht, log_dir
        self.sessions, self.lock, self.gpu, self.nsess = {}, threading.Lock(), threading.Lock(), 0
        log_dir.mkdir(parents=True, exist_ok=True)
        self.out = open(log_dir / "drive.jsonl", "a", buffering=1)
        self.start = open(log_dir / "start.jsonl", "a", buffering=1)
        if dump:
            (log_dir / "dump").mkdir(exist_ok=True)

    def _s(self, uuid, ctx) -> Session:
        s = self.sessions.get(uuid)
        if s is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s

    def start_session(self, req, ctx):
        from google.protobuf.json_format import MessageToDict
        try:
            with self.lock:
                s = self.sessions[req.session_uuid] = Session(req, self.nsess)
                self.nsess += 1
        except ValueError as e:
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        self.start.write(json.dumps({"t": time.time(), "session": req.session_uuid, "n": s.n, "scene": s.scene, "cam": CAM,
                                     "spec": s.spec, "R": np.round(s.R, 6).tolist(), "cam_t": s.cam_t,
                                     "rollout_spec": MessageToDict(req.rollout_spec)}) + "\n")
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self.lock:
            s = self.sessions.pop(req.session_uuid, None)
        if s is not None:
            self.out.write(json.dumps({"kind": "close", "t": time.time(), "session": req.session_uuid, "scene": s.scene, **s.count,
                                       "wh": s.maps.wh if s.maps else None, "coverage": s.maps.coverage if s.maps else None}) + "\n")
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        s, im = self._s(req.session_uuid, ctx), req.camera_image
        if im.logical_id != CAM:
            return common_pb2.Empty()
        try:
            ycc = PC.decode(im.image_bytes)
            if s.maps is None or s.maps.wh != ycc.shape[1::-1]:
                s.maps = PC.Maps(s.spec, s.R, ycc.shape[1::-1])
            fr = s.maps(ycc)
        except Exception as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{CAM} decode failed: {e!r}")
        ts = int(im.frame_end_us)
        with s.lock:
            s.frames[ts] = fr
            s.count["images"] += 1
            for k in [k for k in s.frames if k < ts - KEEP_US]:
                del s.frames[k]
            if s.n < self.dump:
                s.jpeg = {ts: bytes(im.image_bytes)}
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
            for k in [k for k in s.poses if k < max(s.poses) - 2 * KEEP_US]:
                del s.poses[k], s.states[k]
        return common_pb2.Empty()

    def submit_route(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        s.cmd = S.command_from_route(req.route)
        s.route0 = [req.route.waypoints[0].x, req.route.waypoints[0].y] if req.route.waypoints else None
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, req, ctx):
        self._s(req.session_uuid, ctx)
        return common_pb2.Empty()

    def drive(self, req, ctx):
        s, now, tq = self._s(req.session_uuid, ctx), int(req.time_now_us), int(req.time_query_us)
        t_in = time.perf_counter()
        with s.lock:
            k = s.count["drive"]
            s.count["drive"] += 1
            infer = s.plan is None or k % self.every == 0
            ts = [t for t in s.frames if t <= now + TOL_US]
            t0 = max(ts) if ts else None
            p0 = s.pose_at(t0) if ts else None
            if p0 is None or now - t0 > STEP_US + TOL_US:
                s.count["input_error"] += 1
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, f"no {CAM} frame with an ego pose at Drive time {now} (latest frame {t0})")
            if infer:
                keys = [p for p in (s.pose_at(t0 + int(d)) for d in T_KEY_US) if p is not None]      # a suffix: poses only start late
                R0 = S.rot(p0[2]).T
                pose = np.array([[*(R0 @ (p[:2] - p0[:2])), (p[2] - p0[2] + np.pi) % (2 * np.pi) - np.pi] for p in keys])
                dyn = [s.velocity(t0 + int(d)) for d in T_KEY_US[4 - len(keys):]]
                P, V = C.fill_history(pose, np.array([d[0] for d in dyn]), dyn[0][2])
                frames, cmd = dict(s.frames), np.asarray(s.cmd, np.float32).copy()
                s.count["state_rotated"] += int(dyn[-1][3])
        if infer:
            try:
                t_f = time.perf_counter()
                cur, valid, real = PC.slots(frames, t0, P, V, s.cam_t, self.cold)
                t_q = time.perf_counter()
                with self.gpu:
                    t_g = time.perf_counter()
                    o = PC.plan(self.core, cur, valid, P, V, dyn[-1][1], cmd, s.cam_t, self.lht)
            except Exception as e:
                s.count["inference_error"] += 1
                LOG.exception("inference failed, session %s t %d", req.session_uuid, now)
                ctx.abort(grpc.StatusCode.INTERNAL, f"inference failed: {e!r}")
            s.count["inference"] += 1
            s.count["cold"] += int(not real.all())
            fx = FX.apply(s.fix, o, float(np.hypot(*V[-1])), float(dyn[-1][1][0]), t0)
            near = s.poses[min(s.poses, key=lambda t: abs(t - t0))]      # the anchor: the interpolated pose of t0 as a PoseAtTime
            s.anchor = common_pb2.PoseAtTime(timestamp_us=t0, pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=float(p0[0]), y=float(p0[1]), z=near.pose.vec.z),
                quat=common_pb2.Quat(w=float(np.cos(p0[2] / 2)), x=0.0, y=0.0, z=float(np.sin(p0[2] / 2)))))
            s.plan = S.make_cached_plan(t0, s.anchor, o["poses"])
        traj = S.build_trajectory_from_plan(s.plan, s.anchor, now, tq)
        t_out = time.perf_counter()
        rec = {"kind": "drive", "t": time.time(), "session": req.session_uuid, "scene": s.scene, "now": now, "tq": tq, "t0": t0, "k": k,
               "infer": infer, "n_out": len(traj.poses), "total_ms": round(1e3 * (t_out - t_in), 2)}
        if infer:
            rec.update(n_keys=len(keys), n_real=int(real.sum()), cmd=int(np.argmax(cmd)), route0=s.route0, rotated=[bool(d[3]) for d in dyn],
                       ego=o["ego"].round(5).tolist(), hist=P.round(4).tolist(), anchor=p0.round(4).tolist(), poses=o["poses"].round(4).tolist(),
                       ms={**{a: round(b, 2) for a, b in o["ms"].items()}, "prep": round(1e3 * (t_f - t_in), 2),
                           "slots": round(1e3 * (t_q - t_f), 2), "wait": round(1e3 * (t_g - t_q), 2)})
            if fx is not None:
                rec.update(fix=fx, poses_model=o["poses_model"].round(4).tolist())
            if s.n < self.dump and k % 10 == 0:
                np.savez_compressed(self.dir / "dump" / f"s{s.n:02d}_k{k:03d}.npz", cur=cur, valid=valid, real=real, mu=o["mu"],
                                    poses=o["poses"], ego=o["ego"], cam_t=s.cam_t, scene=s.scene, now=now, t0=t0,
                                    jpeg=np.frombuffer(s.jpeg.get(t0, b""), np.uint8))
        self.out.write(json.dumps(rec) + "\n")
        return egodriver_pb2.DriveResponse(trajectory=traj)

    def get_version(self, req, ctx):
        return common_pb2.VersionId(version_id=f"jev-pai-{self.core.tag}-{self.cold}{FX.SUFFIX}", git_hash=os.environ.get("SH30_GIT_HASH", "local"),
                                    grpc_api_version=S.API)


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    host, port = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0"), int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))
    log_dir = Path(os.environ.get("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver"))
    t0 = time.time()
    core = C.Core(os.environ.get("SH30_TAG", "P2H10-F-s0"), os.environ.get("SH30_DEVICE", "cuda"))
    core.lead_out = FX.LEAD
    LOG.info("serving: JEV_VCONT %g, JEV_LEAD %d", FX.VCONT, FX.LEAD)
    z, e = np.zeros((8,) + C.FRAME, np.uint8), np.zeros
    warm = lambda: PC.plan(core, z, np.ones(8, bool), e((4, 3)), e((4, 2)), e(2), np.array([0, 1, 0, 0]), [1.7, 0.0, 1.5])  # noqa: E731
    warm(), warm()
    LOG.info("%s ready in %.1f s, VRAM %.2f GiB", core.tag, time.time() - t0, torch.cuda.max_memory_allocated() / 2**30)
    drv = Driver(core, log_dir, os.environ.get("PAI_COLD", "backwarp"), int(os.environ.get("PAI_EVERY", "1")),
                 int(os.environ.get("PAI_DUMP", "0")), os.environ.get("PAI_LHT", "0") == "1")
    server = grpc.server(S.warm_workers(warm))
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
