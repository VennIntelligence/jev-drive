"""A human with the phone wheel as an AlpaSim driver: egodriver.EgodriverService (both tracks), no model.

Each `drive(time_now, time_query)`: the latest ego pose (local -> rig, rig origin = rear axle) and |DynamicState velocity| (the
magnitude: the first sample can arrive unrotated, docs/alpasim.md), the phone (WheelClient.state(), non-blocking) -> a steering
angle and an acceleration -> a kinematic bicycle rolled 4 s ahead at 0.1 s from the current pose and speed, inputs held -> 41
local-frame poses with absolute timestamps on the anchor's 0.1 s grid at or after `now`: the shape and horizon the model drivers
return (pai_driver / sh30_driver: the LTF sample's build_trajectory_from_plan over a 4 s plan, 10 Hz). The official MPC tracks it.

  steer     delta = s * min(DELTA_MAX, atan(L * AY_MAX / v^2)): full tilt = 0.55 rad road-wheel angle at parking speed
            (radius 4.7 m), above ~4.4 m/s the lock shrinks so full tilt always means AY_MAX = 7 m/s^2 of lateral acceleration
            (speed-sensitive steering: the same thumb motion is sane at 3 and at 25 m/s)
  pedals    a = 3.0 t - 8.0 b - (0.25 + 0.0008 v^2): 3 m/s^2 full throttle (0-100 km/h in ~10 s), 8 m/s^2 full brake (0.8 g),
            coasting slows by 0.25-0.6 m/s^2 (rolling + drag); speed clamped to [0, 36] m/s, no reverse
  L         0.56 x ego box length (5.21 m PAI, 5.18 m nuPlan -> 2.9 m), clamped to [2.5, 3.3]
  stale     no phone frame for 0.3 s -> wheel_client gives throttle 0, brake 1: the plan brakes at 8 m/s^2 to a stop

Start: the world is held inside the first `drive` at or after HUMAN_HOLD_S of the session (the force-GT replay: PAI 1.7 s, nuPlan
0.5 s, so the hold sits where control passes to the driver) until the start signal, at most HUMAN_WAIT_MAX_S (240):
HUMAN_START=click (default: a left click in the viewer touches <live>/start; with no viewer heartbeat in <live>/viewer.json for
5 s the first throttle press is accepted too), throttle (first throttle press > 0.2), none. Pacing: after the start a call sleeps so
sim time never runs ahead of wall time x HUMAN_REALTIME (1.0; 0 = unpaced), as phone_wheel's human_agent does; when the stack is
slower than that nothing is slept (slow motion) and more than 0.5 s behind re-anchors (`late`). Wall s per sim s, the unpaced
figure (without the sleeps) and the sleeps are logged and shown.

Logs (ALPASIM_DRIVER_LOG_DIR): drive.jsonl, one `drive` record per call (phone input, applied delta / accel, speed, input age,
stale, pose, wall) and one `close` record per session (sim s, wall s, wall per sim s after the start, stale calls, wait s);
start.jsonl (scene, cameras, box). Live (HUMAN_LIVE): state.json + front.jpg / left.jpg / right.jpg (the received JPEGs, atomic
rename) for scripts/human_viewer.py. Env: ALPASIM_DRIVER_HOST / _PORT, HUMAN_WHEEL_PORT (47800; fake_wheel.py uses 47801),
HUMAN_TRACK, HUMAN_N_SCENES, HUMAN_SIM_S (scene length for the progress bar). Needs grpc, alpasim_grpc, numpy (the driver image).
"""
from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phone_wheel" / "lib"))
from wheel_client import WheelClient  # noqa: E402

LOG = logging.getLogger("human")
DELTA_MAX, AY_MAX, A_THR, A_BRK, V_MAX, DT, N_OUT = 0.55, 7.0, 3.0, 8.0, 36.0, 0.1, 41
FRONT = ("CAM_F0", "camera_front_wide_120fov")
SIDES = {"left": ("CAM_L0", "camera_cross_left_120fov"), "right": ("CAM_R0", "camera_cross_right_120fov")}
E = os.environ.get
RT = float(E("HUMAN_REALTIME", "1.0"))
HOLD_US, WAIT_MAX, START = int(1e6 * float(E("HUMAN_HOLD_S", "0"))), float(E("HUMAN_WAIT_MAX_S", "240")), E("HUMAN_START", "click")
LIVE = Path(E("HUMAN_LIVE", "/tmp/alpasim-human-live"))


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def controls(st: dict, v: float, L: float):
    """Phone state + speed -> (road-wheel angle rad, acceleration m/s^2)."""
    lock = min(DELTA_MAX, math.atan(L * AY_MAX / max(v, 0.5) ** 2))
    delta = float(np.clip(st["steer"], -1, 1)) * lock
    a = A_THR * st["throttle"] - A_BRK * st["brake"] - (0.25 + 8e-4 * v * v) * (v > 0.05)
    return -delta, float(a)          # phone +1 = right; rig y is left, so a right turn is a negative yaw rate


def rollout(x, y, yaw, v, delta, a, L, n=N_OUT):
    """Kinematic bicycle about the rear axle, inputs held, (n, 4) x, y, yaw, v at 0.1 s steps from the current state."""
    out = np.empty((n, 4))
    for k in range(n):
        out[k] = x, y, yaw, v
        v1 = min(max(v + a * DT, 0.0), V_MAX)
        vm = 0.5 * (v + v1)
        yaw1 = yaw + vm * math.tan(delta) / L * DT
        x, y = x + vm * math.cos(0.5 * (yaw + yaw1)) * DT, y + vm * math.sin(0.5 * (yaw + yaw1)) * DT
        yaw, v = yaw1, v1
    return out


def put(path: Path, data: bytes):
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Session:
    def __init__(self, req, n):
        veh = req.rollout_spec.vehicle
        self.n, self.scene = n, req.debug_info.scene_id if req.HasField("debug_info") else ""
        self.L = float(np.clip(0.56 * veh.bounding_box.size_x, 2.5, 3.3)) if veh.bounding_box.size_x else 2.9
        self.cams = [c.logical_id for c in veh.available_cameras]
        self.poses, self.frames, self.route = {}, {}, []      # ts -> (x, y, z, yaw, speed); name -> (ts, jpeg); latest route (rig frame)
        self.lock = threading.Lock()
        self.t_first = self.started = None
        self.wait_s, self.k, self.n_stale, self.last = 0.0, 0, 0, None
        self.anchor, self.slept, self.n_late = None, 0.0, 0
        self.written = {}


class Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
    def __init__(self, log_dir: Path):
        self.wheel, self.wlock = WheelClient(port=int(E("HUMAN_WHEEL_PORT", "47800"))), threading.Lock()
        self.sessions, self.nsess, self.lock = {}, 0, threading.Lock()
        log_dir.mkdir(parents=True, exist_ok=True)
        LIVE.mkdir(parents=True, exist_ok=True)
        self.out = open(log_dir / "drive.jsonl", "a", buffering=1)
        self.start = open(log_dir / "start.jsonl", "a", buffering=1)
        self.meta = {"track": E("HUMAN_TRACK", "?"), "n_scenes": int(E("HUMAN_N_SCENES", "0")), "sim_total": float(E("HUMAN_SIM_S", "0")),
                     "start_mode": START, "run": E("HUMAN_RUN", "")}

    def _s(self, uuid, ctx) -> Session:
        s = self.sessions.get(uuid)
        if s is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {uuid}")
        return s

    def _wheel(self):
        with self.wlock:                                           # one socket, several gRPC threads
            return self.wheel.state()

    def start_session(self, req, ctx):
        with self.lock:
            s = self.sessions[req.session_uuid] = Session(req, self.nsess)
            self.nsess += 1
        self.start.write(json.dumps({"t": time.time(), "session": req.session_uuid, "n": s.n, "scene": s.scene, "cams": s.cams,
                                     "L": s.L}) + "\n")
        LOG.info("session %d %s, cameras %s, wheelbase %.2f m", s.n, s.scene, s.cams, s.L)
        return common_pb2.SessionRequestStatus()

    def close_session(self, req, ctx):
        with self.lock:
            s = self.sessions.pop(req.session_uuid, None)
        if s is not None and s.last is not None:
            (w0, t0), (w1, t1) = s.started or s.last[:2], s.last[:2]
            sim = (t1 - t0) / 1e6
            rec = {"kind": "close", "t": time.time(), "session": req.session_uuid, "n": s.n, "scene": s.scene, "drives": s.k,
                   "sim_s": round((t1 - s.t_first) / 1e6, 2), "driven_sim_s": round(sim, 2), "driven_wall_s": round(w1 - w0, 2),
                   "wall_per_sim": round((w1 - w0) / sim, 3) if sim > 0 else None, "stale": s.n_stale, "wait_s": round(s.wait_s, 1),
                   "slept_s": round(s.slept, 2), "unpaced_per_sim": round((w1 - w0 - s.slept) / sim, 3) if sim > 0 else None,
                   "late": s.n_late, "realtime": RT}
            self.out.write(json.dumps(rec) + "\n")
            LOG.info("close %s", rec)
            self._live(s, {"waiting": False, "closed": True})
        return common_pb2.Empty()

    def submit_image_observation(self, req, ctx):
        s, im = self._s(req.session_uuid, ctx), req.camera_image
        name = "front" if im.logical_id in FRONT else next((k for k, v in SIDES.items() if im.logical_id in v), None)
        if name:
            with s.lock:
                if im.frame_end_us >= s.frames.get(name, (-1,))[0]:
                    s.frames[name] = (int(im.frame_end_us), bytes(im.image_bytes))
        return common_pb2.Empty()

    def submit_egomotion_observation(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        with s.lock:
            for p, d in zip(req.trajectory.poses, req.dynamic_states):
                q, t = p.pose.quat, p.pose.vec
                s.poses[int(p.timestamp_us)] = (t.x, t.y, t.z, yaw_of(q), math.hypot(d.linear_velocity.x, d.linear_velocity.y))
            for k in sorted(s.poses)[:-20]:
                del s.poses[k]
        return common_pb2.Empty()

    def submit_route(self, req, ctx):
        s = self._s(req.session_uuid, ctx)
        s.route = [[round(w.x, 2), round(w.y, 2)] for w in req.route.waypoints if not (math.isnan(w.x) or math.isnan(w.y))]
        return common_pb2.Empty()

    def submit_recording_ground_truth(self, req, ctx):
        return common_pb2.Empty()

    def _live(self, s: Session, extra: dict):
        """state.json + the newest JPEGs into the live dir for the viewer; never raises."""
        try:
            for name, (ts, jpg) in list(s.frames.items()):
                if s.written.get(name) != ts:
                    put(LIVE / f"{name}.jpg", jpg)
                    s.written[name] = ts
            st = {**self.meta, "t": time.time(), "running": True, "scene": s.scene, "scene_i": s.n + 1, "cams": s.cams,
                  "frames": sorted(s.frames), "route": s.route, **extra}
            put(LIVE / "state.json", json.dumps(st).encode())
        except Exception as e:  # the view must never stop the car
            LOG.warning("live write failed: %r", e)

    def _released(self, t_hold: float) -> bool:
        st = self._wheel()
        thr = not st["stale"] and st["raw_throttle"] > 0.2
        if START == "throttle":
            return thr
        if START == "none":
            return True
        try:
            if (LIVE / "start").stat().st_mtime > t_hold:
                return True
        except OSError:
            pass
        try:
            viewer = time.time() - json.loads((LIVE / "viewer.json").read_text())["t"] < 5
        except (OSError, ValueError, KeyError):
            viewer = False
        return thr and not viewer                                   # no viewer running: the throttle path

    def _hold(self, s: Session, base: dict):
        t_hold = time.time()
        while time.time() - t_hold < WAIT_MAX:
            st = self._wheel()
            self._live(s, {**base, **self._panel(st), "waiting": True, "wait_s": round(time.time() - t_hold, 1)})
            if self._released(t_hold):
                break
            time.sleep(0.05)
        s.wait_s = time.time() - t_hold
        LOG.info("session %d started after %.1f s", s.n, s.wait_s)

    @staticmethod
    def _panel(st):
        return {"steer_in": round(st["steer"], 3), "throttle_in": round(st["throttle"], 3), "brake_in": round(st["brake"], 3),
                "raw_throttle": round(st["raw_throttle"], 3), "age_ms": round(1e3 * min(st["age_s"], 99.0), 1), "stale": st["stale"]}

    def drive(self, req, ctx):
        s, now, tq = self._s(req.session_uuid, ctx), int(req.time_now_us), int(req.time_query_us)
        t_in = time.time()
        with s.lock:
            ts = [t for t in s.poses if t <= now + 50_000]
            if not ts:
                ctx.abort(grpc.StatusCode.FAILED_PRECONDITION, f"no ego pose at Drive time {now}")
            ta = max(ts)
            x, y, z, yaw, v = s.poses[ta]
        if s.t_first is None:
            s.t_first = now
        sim_t = (now - s.t_first) / 1e6
        base = {"sim_t": round(sim_t, 2), "speed": round(v, 2)}
        if s.started is None and now - s.t_first >= HOLD_US:
            self._hold(s, base)
            s.started = s.anchor = (time.time(), now)
        slept = 0.0
        if s.started is not None and RT > 0:                       # pace: sim time never runs ahead of wall time x HUMAN_REALTIME
            wall = time.time()
            target = s.anchor[0] + (now - s.anchor[1]) / 1e6 / RT
            if target > wall:
                slept = target - wall
                time.sleep(slept)
            elif wall - target > 0.5:                              # far behind: re-anchor, never burst to catch up
                s.anchor, s.n_late = (wall, now), s.n_late + 1
        s.slept += slept
        st = self._wheel()                                         # read after the sleep: freshest input
        delta, a = controls(st, v, s.L)
        s.n_stale += st["stale"] and s.started is not None
        traj_xyv = rollout(x, y, yaw, v, delta, a, s.L)
        k0 = max(0, math.ceil((now - ta) / (DT * 1e6) - 1e-9))
        traj = common_pb2.Trajectory()
        for i in range(k0, N_OUT):
            px, py, pyaw, _ = traj_xyv[i]
            traj.poses.append(common_pb2.PoseAtTime(timestamp_us=ta + int(round(i * DT * 1e6)), pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=float(px), y=float(py), z=float(z)),
                quat=common_pb2.Quat(w=math.cos(pyaw / 2), x=0.0, y=0.0, z=math.sin(pyaw / 2)))))
        wall = time.time()
        s.last = (wall, now)
        wps = None
        if s.started and now > s.started[1]:
            wps = round((wall - s.started[0]) / ((now - s.started[1]) / 1e6), 3)
        rec = {"kind": "drive", "t": round(wall, 3), "session": req.session_uuid, "scene": s.scene, "k": s.k, "now": now, "tq": tq,
               "ta": ta, **self._panel(st), "delta": round(delta, 4), "accel": round(a, 3), "v": round(v, 3), "pose": [round(x, 3), round(y, 3),
               round(yaw, 5)], "started": s.started is not None, "slept_ms": round(1e3 * slept, 1), "n_out": len(traj.poses), "ms": round(1e3 * (wall - t_in), 2)}
        self.out.write(json.dumps(rec) + "\n")
        s.k += 1
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        plan = ((traj_xyv[::4, :2] - [x, y]) @ R.T).round(2).tolist()               # rig frame of the current pose, 0.4 s apart
        self._live(s, {**base, **self._panel(st), "waiting": False, "delta_deg": round(math.degrees(delta), 1), "accel": round(a, 2),
                       "plan": plan, "wall_per_sim": wps, "realtime": RT, "late": s.n_late,
                       "unpaced_per_sim": round(((wall - s.started[0]) - s.slept) / ((now - s.started[1]) / 1e6), 3) if wps else None, "started": s.started is not None})
        return egodriver_pb2.DriveResponse(trajectory=traj)

    def get_version(self, req, ctx):
        try:
            import alpasim_grpc
            api = alpasim_grpc.API_VERSION_MESSAGE
        except Exception:
            api = None
        return common_pb2.VersionId(version_id="jev-human-phone", git_hash=E("HUMAN_GIT", "local"), grpc_api_version=api)


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host, port = E("ALPASIM_DRIVER_HOST", "127.0.0.1"), int(E("ALPASIM_DRIVER_PORT", "6790"))
    drv = Driver(Path(E("ALPASIM_DRIVER_LOG_DIR", "/tmp/alpasim-driver")))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(drv, server)
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    server.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: server.stop(grace=0.0))
    LOG.info("realtime %.2f, hold %.1f s, start %s, wheel port %s, live %s", RT, HOLD_US / 1e6, START, E("HUMAN_WHEEL_PORT", "47800"), LIVE)
    LOG.info("listening on %s:%d", host, port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
