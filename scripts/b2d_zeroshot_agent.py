#!/usr/bin/env python
"""Bench2Drive leaderboard agent for the zero-shot exam of open driving models (Alpamayo 1.5, openpilot).
Pre-registration: todos/2026-09-24-zeroshot-exam/bench2drive.md. Python 3.8, envs/carla.

The agent owns the sensors and the actuation; the model lives in scripts/zeroshot_policy_server.py and is
reached over a unix socket. Per tick (20 Hz):

  1. read GNSS/IMU/speed of this exact frame and every camera set that has arrived (cameras render at the
     model's own rate via sensor_tick: Alpamayo 10 Hz, openpilot 5 Hz);
  2. update a GNSS+compass+odometry pose (scripts/b2d_controller_adapter.PoseFilter, sensors only);
  3. on a planning camera set (Alpamayo: every 5th set = 2 Hz; openpilot: every set = 5 Hz) send the model its
     inputs and block for the answer - the simulator is synchronous, so the model's latency costs wall time,
     not driving quality (no latency is simulated);
  4. hand the returned rear-axle path, resampled to 20 points at 0.25 s, to the repo's fixed controller
     (scripts/b2d_controller.py, preset from the config) and apply its throttle/steer/brake.

Route information reaches the model only as the model would get it in a car: Alpamayo gets a nav sentence
("Turn left in 30m") from the next junction command of the route, openpilot a turn/lane-change desire.

Config JSON (--agent-config): {"model": "alpamayo"|"lebowski", "socket": path, "plan_every": 5|1,
"controller_config": path, "controller_preset": "carla", "seed": 0, "dump_every": 0}.
"""
import json
import math
import os
import socket
import sys
import time
from collections import deque

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.timer import GameTime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zeroshot_rigs as rigs  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from b2d_controller import Controller  # noqa: E402
from b2d_controller_adapter import FrameRouter, GPSProjector, PoseFilter, controller_speed  # noqa: E402

DELTA = 0.05
LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGE_LEFT, CHANGE_RIGHT = 1, 2, 3, 4, 5, 6
NAV_RANGE_M = 60.0      # Alpamayo: announce a turn this far ahead (its horizon is 6.4 s)
DESIRE_RANGE_M = 20.0   # openpilot: turn desire this far before the junction
DESIRE_NONE, DESIRE_TURN_LEFT, DESIRE_TURN_RIGHT, DESIRE_LC_LEFT, DESIRE_LC_RIGHT = 0, 1, 2, 3, 4


def get_entry_point():
    return "ZeroShotAgent"


def resample(t_src, xy, t_dst):
    return np.stack([np.interp(t_dst, t_src, xy[:, k]) for k in range(2)], -1)


class Route(object):
    """Dense route with monotonic progress and a look-ahead on its junction commands."""

    def __init__(self, world_plan):
        self.xy = np.array([[tf.location.x, tf.location.y] for tf, _ in world_plan])
        self.cmd = np.array([int(getattr(c, "value", c)) for _, c in world_plan])
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.i = 0

    def progress(self, xy):
        window = self.xy[self.i:self.i + 80]
        self.i += int(np.argmin(np.linalg.norm(window - xy, axis=1)))
        return self.i

    def next_maneuver(self, kinds):
        """(command, distance along the route to where it starts) of the next command in `kinds`, 0 m if inside."""
        ahead = np.nonzero(np.isin(self.cmd[self.i:], kinds))[0]
        if not len(ahead):
            return None, None
        j = self.i + int(ahead[0])
        return int(self.cmd[j]), float(self.s[j] - self.s[self.i])

    def nav_text(self):
        cmd, d = self.next_maneuver([LEFT, RIGHT, STRAIGHT])
        if cmd in (LEFT, RIGHT) and d <= NAV_RANGE_M:
            return "Turn %s in %dm" % ("left" if cmd == LEFT else "right", int(round(d)))
        return None

    def desire(self):
        cmd, d = self.next_maneuver([LEFT, RIGHT, STRAIGHT, CHANGE_LEFT, CHANGE_RIGHT])
        if cmd in (LEFT, RIGHT) and d <= DESIRE_RANGE_M:
            return DESIRE_TURN_LEFT if cmd == LEFT else DESIRE_TURN_RIGHT
        if cmd in (CHANGE_LEFT, CHANGE_RIGHT) and d <= 0.0:
            return DESIRE_LC_LEFT if cmd == CHANGE_LEFT else DESIRE_LC_RIGHT
        return DESIRE_NONE


class ZeroShotAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        with open(path_to_conf_file.split("+")[0]) as fh:
            self.cfg = json.load(fh)
        self.model = self.cfg["model"]
        self.alpamayo = self.model == "alpamayo"
        self.plan_every = int(self.cfg.get("plan_every", 5 if self.alpamayo else 1))
        self.cam_specs = rigs.alpamayo_sensor_specs() if self.alpamayo else rigs.openpilot_sensor_specs()
        self.cam_tags = [s["id"] for s in self.cam_specs]
        with open(self.cfg["controller_config"]) as fh:
            params = json.load(fh)
        self.rear_offset = float(params.pop("rear_axle_offset_m"))
        for key in ("adapter", "metadata", "preset"):
            params.pop(key, None)
        self.controller = Controller(preset=self.cfg.get("controller_preset", "carla"), **params)
        self.out = os.environ.get("B2D_ATTEMPT_OUT") or self.cfg.get("out") or "."
        os.makedirs(os.path.join(self.out, "frames"), exist_ok=True)
        self.plan_log = open(os.path.join(self.out, "plans.jsonl"), "w", buffering=1)
        self.tick_log = open(os.path.join(self.out, "ticks.jsonl"), "w", buffering=1)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.cfg["socket"])
        wire.send(self.sock, {"cmd": "reset"})
        self.server_meta = wire.recv(self.sock)[0]["server"]
        self.router = FrameRouter(self.cam_tags, retain_frames=32)
        self.cam_sets = deque(maxlen=4)          # (frame, sim time, {tag: BGRA}, {tag: frame})
        self.latest = {t: (-1, None) for t in self.cam_tags}
        self.first_frame = None
        self.poses = deque(maxlen=64)            # (sim time, xy, yaw) rear axle, world
        self.frame_time = {}
        self.pose_filter = self.route = None
        self.n_sets = self.n_plans = self.tick = 0
        self.timings = {"tick_ms": [], "plan_ms": []}
        self.control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        if getattr(self, "_dense_plan", None):
            self._init_route()

    def sensors(self):
        motion = [
            {"type": "sensor.other.imu", "x": -1.4, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": DELTA, "id": "IMU"},
            {"type": "sensor.other.gnss", "x": -1.4, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": DELTA, "id": "GPS"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]
        return [dict(s) for s in self.cam_specs] + motion

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        # Dense plan: the base class downsamples to 50 m, which loses the junction commands' extent.
        self._dense_plan, self._dense_gps = list(global_plan_world_coord), list(global_plan_gps)
        if hasattr(self, "cfg"):
            self._init_route()

    def _init_route(self):
        world = np.array([[tf.location.x, tf.location.y] for tf, _ in self._dense_plan])
        gps = np.array([[p["lat"], p["lon"]] for p, _ in self._dense_gps])
        self.pose_filter = PoseFilter(GPSProjector(gps, world), self.rear_offset, -1.4, .05, .1)
        self.route = Route(self._dense_plan)

    # ---- per tick -------------------------------------------------------------------------------------------
    def __call__(self):
        t_start = time.perf_counter()
        self.tick += 1
        frame, now = int(GameTime.get_frame()), float(GameTime.get_time())
        data = self.router.read(self.sensor_interface, frame)
        self.frame_time[frame] = now
        if self.first_frame is None:
            self.first_frame = frame
        self.frame_time.pop(frame - 200, None)
        imu, speed_raw = data["IMU"][1], float(data["SPEED"][1]["speed"])
        speed, world_gyro = controller_speed(speed_raw), float(imu[5])
        try:
            xy, yaw = self.pose_filter.update(data["GPS"][1], float(imu[6]), speed, world_gyro, now)
            self.poses.append((now, xy, yaw))
            self.route.progress(xy)
        except ValueError:
            self.pose_filter.reset()
        plan_ms = 0.0
        for f, tags in sorted(self.router.frames.items()):
            if f < self.first_frame:
                continue            # rendered while the scenario was being built, before the route started
            for tag in self.cam_tags:
                if tag in tags and f > self.latest[tag][0]:
                    self.latest[tag] = (f, tags[tag][1])
        # A camera set is complete when every camera has delivered a frame newer than the previous set. Cameras
        # at a sensor_tick above 0.05 s occasionally fire one tick late (UE tick-interval carry-over), so a set
        # can span two adjacent frames; its frame is the newest, and the per-camera frames are logged.
        last = self.cam_sets[-1][3] if self.cam_sets else {t: -1 for t in self.cam_tags}
        if all(self.latest[t][0] > last[t] for t in self.cam_tags):
            f = max(self.latest[t][0] for t in self.cam_tags)
            self.cam_sets.append((f, self.frame_time.get(f, now), {t: self.latest[t][1] for t in self.cam_tags},
                                  {t: self.latest[t][0] for t in self.cam_tags}))
            self.n_sets += 1
            if (self.n_sets - 1) % self.plan_every == 0 and self.poses:
                plan_ms += self._plan(speed)
        throttle, steer, brake = self.controller.step(now, speed, -world_gyro)
        self.control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        tick_ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["tick_ms"].append(tick_ms)
        diag = self.controller.diagnostics
        self.tick_log.write(json.dumps({"frame": frame, "t": now, "v": speed, "throttle": float(throttle),
                                        "steer": float(steer), "brake": float(brake), "reason": diag["reason"],
                                        "agent_ms": round(tick_ms, 2), "plan_ms": round(plan_ms, 1)}) + "\n")
        return self.control

    def _history(self, t0):
        """16 rear-axle poses at t0 - 1.5 s ... t0 (10 Hz) in the t0 rig frame (x forward, y left, yaw CCW);
        before the first pose, standstill. Poses are CARLA world (y right of x, yaw clockwise)."""
        t = np.array([p[0] for p in self.poses])
        xy = np.array([p[1] for p in self.poses])
        yaw = np.unwrap(np.array([p[2] for p in self.poses]))
        ts = np.clip(t0 + np.arange(-15, 1) * 0.1, t[0], t[-1])
        hx, hy, hyaw = (np.interp(ts, t, v) for v in (xy[:, 0], xy[:, 1], yaw))
        c0, s0 = math.cos(hyaw[-1]), math.sin(hyaw[-1])
        dx, dy = hx - hx[-1], hy - hy[-1]
        return np.stack([c0 * dx + s0 * dy, s0 * dx - c0 * dy, hyaw[-1] - hyaw], -1).astype(np.float32)

    def _plan(self, speed):
        t_start = time.perf_counter()
        f, t_frame, cams, cam_frames = self.cam_sets[-1]
        dump = ""
        every = int(self.cfg.get("dump_every", 0))
        if every and self.n_plans % every == 0:
            dump = os.path.join(self.out, "frames", "%06d.jpg" % f)
        meta = {"cmd": "plan", "seed": int(self.cfg.get("seed", 0)) + self.n_plans, "dump": dump,
                "speed": speed, "t": t_frame}
        if self.alpamayo:
            sets = list(self.cam_sets)
            sets = [sets[0]] * (4 - len(sets)) + sets   # route start: repeat the first frame
            arrays = {tag: np.stack([s[2][tag] for s in sets]) for tag in self.cam_tags}
            arrays["hist"] = self._history(t_frame)
            meta["nav_text"] = self.route.nav_text()
        else:
            arrays = dict(cams)
            meta["desire"] = self.route.desire()
        wire.send(self.sock, meta, arrays)
        info, out = wire.recv(self.sock)
        if self.alpamayo:
            path = resample(out["t"], out["xy"], np.arange(1, 21) * 0.25)
        else:
            path = resample(out["t"], rigs.openpilot_plan_to_rig(out["pos"]), np.arange(1, 21) * 0.25)
        accepted = self.controller.update(path.astype(float), t_frame)
        self.n_plans += 1
        ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["plan_ms"].append(ms)
        rec = {"frame": f, "t": t_frame, "set_frames": [s[0] for s in self.cam_sets], "cam_frames": cam_frames,
               "speed": speed,
               "accepted": accepted, "path": np.round(path, 3).tolist(), "round_trip_ms": round(ms, 1),
               "pose": [float(v) for v in self.poses[-1][1]] + [float(self.poses[-1][2])],
               "route_index": self.route.i}
        rec.update({k: v for k, v in meta.items() if k in ("nav_text", "desire", "seed", "dump")})
        rec.update(info)
        self.plan_log.write(json.dumps(rec) + "\n")
        return ms

    def destroy(self):
        for fh in (getattr(self, "plan_log", None), getattr(self, "tick_log", None)):
            if fh is not None:
                fh.close()
        if getattr(self, "sock", None) is not None:
            self.sock.close()
        tm = self.timings if hasattr(self, "timings") else {}
        summary = {k: {"n": len(v), "mean_ms": float(np.mean(v)) if v else None,
                       "p50_ms": float(np.median(v)) if v else None} for k, v in tm.items()}
        summary.update(n_plans=getattr(self, "n_plans", 0), n_camera_sets=getattr(self, "n_sets", 0),
                       server=getattr(self, "server_meta", None), config=getattr(self, "cfg", None))
        with open(os.path.join(getattr(self, "out", "."), "agent_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
