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

Config JSON (--agent-config): {"model": "alpamayo"|"lebowski"|"cinque"|"small", "socket": path, "plan_every": 5|1,
"controller_config": path, "controller_preset": "carla", "seed": 0, "dump_every": 0}, plus the openpilot adapter
switches added after the smoke (todos/2026-09-24-zeroshot-exam/openpilot-migration.md); their defaults reproduce
the pre-registered smoke:
  "op_camera_tick"  0.2 (smoke) | 0.05: render both cameras every step, road and wide from the same frame
  "plan_origin"     "camera" (smoke: plan x + 1.78 m) | "rear": rear-axle track, rear = d + p - R(psi) d
  "warmup_s"        0 | 5.0: feed the model this long from the first camera set with the brake held before its
                    plans reach the controller (openpilot is engaged only once modeld has been running)
  "drive"           "model" | "oracle": shadow mode, the route-oracle adapter drives (b2d_controller_adapter.
                    RouteAdapter, as b2d_agent --drive controller) and the model's plans are only logged
  "op_mount"        rig [x, y, z] of the camera pair (default the windshield top, zeroshot_rigs.OP_MOUNT_RIG)
  "controller"      "fixed" (the pre-registered smoke: scripts/b2d_controller.py) | "zoo_pid": Bench2DriveZoo's
                    UniAD/VAD PID, vendored verbatim (scripts/b2d_zoo_pid_wrap.py -> b2d_zoo_pid.py). It is
                    evaluated once per plan and its control held until the next plan, as the 2 Hz AD-MLP agent does
  "engage_s"        0 | 5.0: engage while rolling - the route oracle (RouteAdapter at 3 m/s through the fixed
                    controller) drives from the route start until the model's warm-up is over, then hands over;
                    later stops and resumes are the model's own
  "desire"          true (pre-registered: turn / lane-change desire pulses from route commands) | false
  "junction_handover_m"  0 | d: within d m before a LEFT / RIGHT route command and through it, steering comes from the
                    route oracle while throttle / brake stay the model's controller (a driver steering through the
                    turn with ACC on)
  "plan_forward_only"  false | true: controller zoo_pid reads the plan with its backward segments removed (speed >= 0);
                    see b2d_zoo_pid_wrap.py and todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md
  "zoo_cadence"     "plan" (control_pid once per plan, held: AD-MLP) | "tick": every tick on the age-shifted held plan
                    (UniAD / VAD)
  "zoo_lateral"     "zoo" (control_pid's steer) | "fixed" (P1: steer from the fixed controller's 20 Hz tracking of the raw
                    plan, throttle / brake stay Zoo PID) | "time" (P2: Zoo turn PID on the plan point at "zoo_aim_s",
                    default 1.5 s, no route-target substitution); diagnosis doc section 8
  "partner"         absent | {"ckpt": path, "junctions": true|false}: shared control with the official TCP agent
                    (scripts/b2d_partner.py): TCP drives from standstill and (junctions) through route turns, the model
                    everything else; who drives is logged every tick ("driver", "w_model"); "tcp_only": true makes TCP
                    drive the whole route (reference arm; the model still plans, in shadow)
  "replay"          absent | path of an expert log (scripts/b2d_expert_agent.py; "{route}" is replaced by the route id)
                    | "route": no model and no socket; each
                    planning step returns the expert's own track from where the hero is, at the expert's pace (waits
                    as long as the expert waited; see _replay_path), in the hero's simulator rear-axle frame, i.e. the
                    plan a perfect planner would give, at the model's cadence and through the configured controller -
                    the controller acceptance test
                    (todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md). "route": the route oracle's plan
                    from the sensor pose (a no-model load for profiling)
  "lateral"         "plan" (the fixed controller tracks the plan) | "curvature": exploratory, steer from the
                    model's desired curvature through the bicycle model, longitudinal still from the plan
Ground truth (the hero's rear-axle pose) is logged every tick for evaluation only; it never reaches control.
"""
import json
import math
import os
import socket
import sys
import time
from collections import deque
from queue import Empty

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.timer import GameTime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zeroshot_rigs as rigs  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from b2d_controller import Controller  # noqa: E402
from b2d_controller_adapter import FrameRouter, GPSProjector, PoseFilter, RouteAdapter, controller_speed  # noqa: E402

DELTA = 0.05
LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGE_LEFT, CHANGE_RIGHT = 1, 2, 3, 4, 5, 6
NAV_RANGE_M = 60.0      # Alpamayo: announce a turn this far ahead (its horizon is 6.4 s)
DESIRE_RANGE_M = 20.0   # openpilot: turn desire this far before the junction
DESIRE_NONE, DESIRE_TURN_LEFT, DESIRE_TURN_RIGHT, DESIRE_LC_LEFT, DESIRE_LC_RIGHT = 0, 1, 2, 3, 4
REPLAY_WAIT_TOL_M = 2.0     # "replay": the expert counts as still at s0 while within this arc distance of it


def get_entry_point():
    return "ZeroShotAgent"


class SyncRouter(FrameRouter):
    """FrameRouter that also waits (up to 2 s) for `required` cameras of the current frame. With every-tick openpilot
    cameras plus the TCP partner's three 1600x900 cameras, road / wide packets sometimes arrived after the motion
    sensors of their frame, so that frame's pair was superseded before it was read and 9 % of the Lebowski context
    steps were 5-14 frames instead of 4 (plumbing run, route 2086)."""

    def __init__(self, camera_tags, retain_frames, required=()):
        super().__init__(camera_tags, retain_frames=retain_frames)
        self.required = frozenset(required)

    def read(self, interface, frame):
        if self.required:
            deadline = time.monotonic() + 2.0
            while not (self.MOTION | self.required).issubset(self.frames.get(frame, {})):
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    self._put(interface._data_buffers.get(True, left))
                except Empty:
                    break
        return super().read(interface, frame)


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
        self.op_tick = float(self.cfg.get("op_camera_tick", rigs.OP_CAMERA_TICK))
        self.plan_origin = self.cfg.get("plan_origin", "camera")
        self.warmup_s = float(self.cfg.get("warmup_s", 0.0))
        self.drive = self.cfg.get("drive", "model")
        self.lateral = self.cfg.get("lateral", "plan")
        assert self.plan_origin in ("camera", "rear") and self.drive in ("model", "oracle") \
            and self.lateral in ("plan", "curvature"), self.cfg
        self.op_mount = tuple(self.cfg.get("op_mount", rigs.OP_MOUNT_RIG))
        self.cam_specs = rigs.alpamayo_sensor_specs() if self.alpamayo else rigs.openpilot_sensor_specs(self.op_tick,
                                                                                                        self.op_mount)
        self.cam_tags = [s["id"] for s in self.cam_specs]
        with open(self.cfg["controller_config"]) as fh:
            params = json.load(fh)
        self.vehicle = dict(params)
        self.rear_offset = float(params.pop("rear_axle_offset_m"))
        for key in ("adapter", "metadata", "preset"):
            params.pop(key, None)
        self.controller = Controller(preset=self.cfg.get("controller_preset", "carla"), **params)
        self.zoo = self.zoo_control = self.zoo_target = None
        self.zoo_lateral = self.cfg.get("zoo_lateral", "zoo")
        assert self.zoo_lateral in ("zoo", "fixed", "time"), self.zoo_lateral
        if self.cfg.get("controller", "fixed") == "zoo_pid":
            from b2d_zoo_pid_wrap import ZooPID
            self.zoo = ZooPID(forward_only=self.cfg.get("plan_forward_only", False),
                              cadence=self.cfg.get("zoo_cadence", "plan"),
                              lateral="time" if self.zoo_lateral == "time" else "zoo",
                              aim_s=float(self.cfg.get("zoo_aim_s", 1.5)))
        else:
            assert self.cfg.get("controller", "fixed") in ("fixed", "native"), self.cfg
        self.native = self.cfg.get("controller", "fixed") == "native"
        self.accel_des = None                    # openpilot desired accel, smoothed as modeld does (0.3 s)
        self.accel_meas, self.last_speed = 0.0, None
        self.out = os.environ.get("B2D_ATTEMPT_OUT") or self.cfg.get("out") or "."
        os.makedirs(os.path.join(self.out, "frames"), exist_ok=True)
        self.plan_log = open(os.path.join(self.out, "plans.jsonl"), "w", buffering=1)
        self.tick_log = open(os.path.join(self.out, "ticks.jsonl"), "w", buffering=1)
        self.replay = self.cfg.get("replay")
        self.replay_log = self.replay_t0 = None
        if self.replay:                          # no model: the plan comes from an expert log or the route oracle
            self.sock, self.server_meta = None, {"replay": self.replay}
            if self.replay != "route":       # "{route}" in the path: this route's log (b2d_route.py sets the id)
                self._load_replay(self.replay.format(route=os.environ.get("BENCHMARK_ROUTE_ID", "")))
        else:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(self.cfg["socket"])
            wire.send(self.sock, {"cmd": "reset"})
            self.server_meta = wire.recv(self.sock)[0]["server"]
        self.partner = self.arbiter = None
        pcfg = self.cfg.get("partner")
        if pcfg:
            from b2d_partner import TCPPartner
            self.partner = TCPPartner(pcfg["ckpt"], pcfg["socket"], hero=getattr(self, "hero_actor", None))
            if getattr(self, "_dense_plan", None):      # the evaluator set the route before setup()
                self.partner.set_global_plan(self._dense_gps, self._dense_plan)
        router_tags = self.cam_tags + (self.partner.camera_tags() if self.partner else [])
        every_tick = not self.alpamayo and self.op_tick <= DELTA
        self.router = SyncRouter(router_tags, 32, self.cam_tags if every_tick else ())
        self.cam_sets = deque(maxlen=4)          # (frame, sim time, {tag: BGRA}, {tag: frame})
        self.latest = {t: (-1, None) for t in self.cam_tags}
        self.first_frame = None
        self.first_set_t = None
        self.route_t0, self.engaged, self.warm_now = None, False, True
        self.curvature = None                    # latest desired curvature (1/m, right-positive), lateral=curvature
        self.last_steer = 0.0
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
        own = [dict(s) for s in self.cam_specs] + motion
        if getattr(self, "partner", None) is not None:
            own += self.partner.sensors({s["id"] for s in own})
        return own

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        if getattr(self, "partner", None) is not None:
            self.partner.set_global_plan(global_plan_gps, global_plan_world_coord)
        # Dense plan: the base class downsamples to 50 m, which loses the junction commands' extent.
        self._dense_plan, self._dense_gps = list(global_plan_world_coord), list(global_plan_gps)
        if hasattr(self, "cfg"):
            self._init_route()

    def _init_route(self):
        world = np.array([[tf.location.x, tf.location.y] for tf, _ in self._dense_plan])
        gps = np.array([[p["lat"], p["lon"]] for p, _ in self._dense_gps])
        self.pose_filter = PoseFilter(GPSProjector(gps, world), self.rear_offset, -1.4, .05, .1)
        self.route = Route(self._dense_plan)
        if self.zoo is not None:
            self.zoo.route(self)
        self.engage_s = float(self.cfg.get("engage_s", 0.0))
        self.handover_m = float(self.cfg.get("junction_handover_m", 0.0))
        self.guide = self.drive != "oracle" and (self.engage_s > 0 or self.handover_m > 0)
        assert not self.guide or self.zoo is not None, "engage / junction handover are defined for controller zoo_pid"
        cruise = 3.0 if self.guide else float(self.cfg.get("cruise_mps", 8.0))
        self.oracle = RouteAdapter(world, cruise) if (self.drive == "oracle" or self.guide
                                                      or self.replay == "route") else None
        if hasattr(self, "out"):   # route geometry and commands, for the junction analysis
            with open(os.path.join(self.out, "route.json"), "w") as fh:
                json.dump({"xy": np.round(self.route.xy, 3).tolist(), "cmd": self.route.cmd.tolist()}, fh)

    # ---- per tick -------------------------------------------------------------------------------------------
    def __call__(self):
        t_start = time.perf_counter()
        self.tick += 1
        frame, now = int(GameTime.get_frame()), float(GameTime.get_time())
        if self.replay_t0 is None:
            self.replay_t0 = now
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
            if self.oracle is not None:
                self.oracle.project(xy, yaw, speed, now)
        except ValueError:
            self.pose_filter.reset()
        if self.zoo is not None:   # shipped AD-MLP: the route planner advances on every tick
            self.zoo_target = self.zoo.target(np.asarray(data["GPS"][1]), float(imu[6]))
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
        synced = self.alpamayo or self.op_tick > DELTA or len({self.latest[t][0] for t in self.cam_tags}) == 1
        if synced and all(self.latest[t][0] > last[t] for t in self.cam_tags):
            f = max(self.latest[t][0] for t in self.cam_tags)
            self.cam_sets.append((f, self.frame_time.get(f, now), {t: self.latest[t][1] for t in self.cam_tags},
                                  {t: self.latest[t][0] for t in self.cam_tags}))
            self.n_sets += 1
            if (self.n_sets - 1) % self.plan_every == 0 and self.poses:
                plan_ms += self._plan(speed)
        if self.guide and self.poses and (self.tick - 1) % 4 == 0:   # route oracle, 5 Hz, as b2d_agent
            self.controller.update(self.oracle.trajectory(self.poses[-1][1], self.poses[-1][2]), now)
        zoo_tick = None
        if self.zoo is not None and self.zoo.cadence == "tick" and self.zoo.held is not None and self.poses:
            p, t = self.zoo.held_now(now, (self.poses[-1][1], self.poses[-1][2]))
            z_steer, z_throttle, z_brake, zoo_tick = self.zoo.control(p, t, speed, self.zoo_target)
            self.zoo_control = (z_throttle, z_steer, z_brake)
        o_throttle, o_steer, o_brake = self.controller.step(now, speed, -world_gyro)
        if self.zoo is not None:
            throttle, steer, brake = self.zoo_control or (0.0, 0.0, 1.0)
            reason = "zoo_pid" if self.zoo_control else "no_trajectory"
            if self.zoo_lateral == "fixed":      # P1: the fixed controller's 20 Hz path tracking steers
                steer, reason = o_steer, reason + "+fixed_lateral"
        else:
            throttle, steer, brake = o_throttle, o_steer, o_brake
            reason = self.controller.diagnostics["reason"]
        if self.native:
            throttle, steer, brake, reason = self._native_control(speed, throttle, steer, brake, reason)
        elif self.zoo is None and self.lateral == "curvature" and self.curvature is not None and self.controller.diagnostics["reason"] == "tracking":
            steer = self._curvature_steer(speed)
        if self.guide:
            if self.route_t0 is None:
                self.route_t0 = now
            elapsed = now - self.route_t0
            if not self.engaged and (self.engage_s <= 0 or (elapsed >= self.engage_s - 1e-6 and not self.warm_now)):
                self.engaged = True
            if not self.engaged:
                throttle, steer, brake, reason = o_throttle, o_steer, o_brake, "engage_oracle"
            elif self.handover_m > 0:
                cmd, d = self.route.next_maneuver([LEFT, RIGHT])
                if cmd in (LEFT, RIGHT) and d is not None and d <= self.handover_m:
                    steer, reason = o_steer, reason + "+junction_steer"
        share = None
        if self.partner is not None:
            if self.arbiter is None:
                from b2d_partner import Arbiter
                pc = self.cfg["partner"]
                self.arbiter = Arbiter(self.route.xy, self.route.cmd, bool(pc.get("junctions", True)),
                                       bool(pc.get("tcp_only", False)))
            has_ctrl = self.zoo_control is not None if self.zoo is not None else \
                (self.accel_des is not None if self.native else self.controller.diagnostics["reason"] != "no_trajectory")
            ready = has_ctrl and not self.warm_now
            w, why = self.arbiter.step(DELTA, speed, self.route.i, ready, model_go=has_ctrl and brake <= 0.0)
            share = {"driver": self.arbiter.driver, "w_model": round(w, 2), "partner_why": why}
            if w < 1.0:             # the partner's network runs only when its control has weight (b2d_partner.py)
                t_p = time.perf_counter()
                p_ctrl = self.partner.step(data, now)
                throttle, steer, brake = self.arbiter.mix(w, (throttle, steer, brake), p_ctrl)
                share.update(partner_ctrl=[round(x, 3) for x in p_ctrl],
                             partner_ms=round(1e3 * (time.perf_counter() - t_p), 1))
            else:
                self.partner.advance(data)
        self.last_steer = float(steer)
        self.control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        tick_ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["tick_ms"].append(tick_ms)
        rec = {"frame": frame, "t": now, "v": speed, "throttle": float(throttle), "steer": float(steer),
               "brake": float(brake), "reason": reason, "agent_ms": round(tick_ms, 2),
               "plan_ms": round(plan_ms, 1)}
        if zoo_tick is not None:
            rec["zoo_desired"] = round(zoo_tick["desired_speed"], 3)
        if share is not None:
            rec.update(share)
        rec.update(self._truth())
        self.tick_log.write(json.dumps(rec) + "\n")
        return self.control

    def _truth(self):
        """Rear-axle pose of the hero from the simulator, for evaluation logs only."""
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            tf = CarlaDataProvider.get_hero_actor().get_transform()
            yaw = math.radians(tf.rotation.yaw)
            x = tf.location.x + self.rear_offset * math.cos(yaw)
            y = tf.location.y + self.rear_offset * math.sin(yaw)
            return {"truth": [round(x, 4), round(y, 4), round(yaw, 6)]}
        except Exception as exc:  # noqa: BLE001 - evaluation only, never fatal
            return {"truth_error": str(exc)[:80]}

    def _curvature_steer(self, speed):
        """Exploratory lateral: CARLA steer from the model's desired curvature (right-positive) through the
        kinematic bicycle model and CARLA's speed-dependent steering curve, at the fixed controller's rate limit."""
        curve = np.asarray(self.vehicle["steering_curve"], float)
        scale = float(np.interp(speed * 3.6, curve[:, 0], curve[:, 1]))
        angle = math.atan(float(self.vehicle["wheelbase"]) * self.curvature)
        raw = angle / (math.radians(float(self.vehicle["max_steer_deg"])) * scale)
        step = self.controller.steer_rate * DELTA
        return float(np.clip(np.clip(raw, self.last_steer - step, self.last_steer + step),
                             -self.controller.max_steer, self.controller.max_steer))

    # openpilot's own actuation semantics, pre-specified (not tuned on any route): lateral = desired curvature through
    # the kinematic bicycle model (as openpilot's angle-controlled cars do); longitudinal = desired acceleration as
    # feedforward over the MKZ's approximate full-throttle / full-brake acceleration plus a proportional term on the
    # acceleration error (openpilot's LongControl is feedforward with ki = 0 by default), and a brake hold at
    # standstill when the model asks to stay stopped (openpilot's should_stop).
    NATIVE_A_THROTTLE, NATIVE_A_BRAKE, NATIVE_KP = 3.0, 8.0, 0.1

    def _native_control(self, speed, throttle, steer, brake, reason):
        if self.last_speed is not None:
            a = (speed - self.last_speed) / DELTA
            self.accel_meas += (1 - math.exp(-DELTA / 0.3)) * (a - self.accel_meas)
        self.last_speed = speed
        if self.accel_des is None or self.curvature is None:
            return 0.0, 0.0, 1.0, "native_no_plan"
        steer = self._curvature_steer(speed)
        a = self.accel_des
        if speed < 0.3 and a <= 0.2:
            return 0.0, steer, 1.0, "native_stop"
        pedal = a / (self.NATIVE_A_THROTTLE if a > 0 else self.NATIVE_A_BRAKE) + self.NATIVE_KP * (a - self.accel_meas)
        return float(np.clip(pedal, 0.0, 0.75)), steer, float(np.clip(-pedal, 0.0, 1.0)), "native"

    def _load_replay(self, path):
        """Expert log (scripts/b2d_expert_agent.py): per-tick vehicle-centre pose -> rear-axle track, time from the
        expert's first tick."""
        rows = [json.loads(line) for line in open(path)]
        t = np.array([r["t"] for r in rows], float)
        yaw = np.unwrap(np.radians([r["yaw"] for r in rows]))
        x = np.array([r["x"] for r in rows]) + self.rear_offset * np.cos(yaw)
        y = np.array([r["y"] for r in rows]) + self.rear_offset * np.sin(yaw)
        # The route ends when the expert crosses the finish, still at speed: continue its track straight on at its
        # final speed for 10 s, so the plan near the end does not decelerate into a stop that the expert never made.
        k = min(20, len(t) - 1)
        v_end = float(np.hypot(x[-1] - x[-1 - k], y[-1] - y[-1 - k]) / max(t[-1] - t[-1 - k], 1e-6)) if k else 0.0
        if v_end > 0.5:
            dt = np.arange(1, 201) * DELTA
            t = np.r_[t, t[-1] + dt]
            x, y = np.r_[x, x[-1] + v_end * dt * math.cos(yaw[-1])], np.r_[y, y[-1] + v_end * dt * math.sin(yaw[-1])]
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]    # arc length, non-decreasing
        self.replay_log = (t - t[0], x, y, s)
        self.replay_i = 0

    def _replay_path(self, t_frame, times):
        """The known-good plan a perfect planner would output at t_frame: the expert's own track from where the hero is
        now, at the pace the expert drove it. The hero's simulator rear axle is projected onto the expert's rear-axle
        path (arc s0, searched forward from the last match); the plan starts at the expert time t* when it was at s0
        and returns its positions at t* + times, in the hero's rig frame (x forward, y left). Where the expert stood
        still within REPLAY_WAIT_TOL_M of s0 (a red light, a yield) over [ta, tb], t* = the elapsed time since the
        first tick clipped to [ta, tb], so the plan waits as long as the expert did and no longer; the tolerance
        absorbs the expert creeping a few cm while it waits and a controller stopping short of the expert's stop
        point, either of which froze t* with a tight window (deviation 2). Past the log's end the track goes on
        straight at the expert's final speed (see _load_replay).
        (A plan indexed by elapsed time alone jumps ahead of a lagging car and collapses to one point at the log's
        end, which no planner outputs: deviation 1 of the acceptance doc.)
        "route": the route oracle's trajectory from the sensor pose, as drive=oracle (a no-model load for
        profiling, not a known-good plan)."""
        if self.replay == "route":
            return np.asarray(self.oracle.trajectory(self.poses[-1][1], self.poses[-1][2]), float)
        te, xe, ye, se = self.replay_log
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        tf = CarlaDataProvider.get_hero_actor().get_transform()
        h = math.radians(tf.rotation.yaw)
        c, s = math.cos(h), math.sin(h)
        x0, y0 = tf.location.x + self.rear_offset * c, tf.location.y + self.rear_offset * s
        a, b = max(self.replay_i - 40, 0), min(self.replay_i + 400, len(xe) - 1)
        px, py = xe[a:b + 1], ye[a:b + 1]
        sx, sy = np.diff(px), np.diff(py)
        L2 = np.maximum(sx * sx + sy * sy, 1e-12)
        u = np.clip(((x0 - px[:-1]) * sx + (y0 - py[:-1]) * sy) / L2, 0.0, 1.0) if len(sx) else np.zeros(0)
        if len(u):
            k = int(np.argmin(np.hypot(px[:-1] + u * sx - x0, py[:-1] + u * sy - y0)))
            self.replay_i = a + k
            s0 = se[a + k] + u[k] * math.sqrt(L2[k])
        else:
            s0 = se[-1]
        ta = te[min(np.searchsorted(se, s0 - REPLAY_WAIT_TOL_M, "left"), len(te) - 1)]
        tb = te[max(np.searchsorted(se, s0 + REPLAY_WAIT_TOL_M, "right") - 1, 0)]
        t_star = min(max(t_frame - self.replay_t0, ta), max(ta, tb))
        tq = t_star + np.asarray(times, float)
        dx, dy = np.interp(tq, te, xe) - x0, np.interp(tq, te, ye) - y0
        return np.stack([c * dx + s * dy, s * dx - c * dy], -1)

    def _pose_at(self, t0):
        """Rear-axle world pose (xy, CARLA yaw) interpolated at sim time t0 from the pose history."""
        t = np.array([p[0] for p in self.poses])
        xy = np.array([p[1] for p in self.poses])
        yaw = np.unwrap(np.array([p[2] for p in self.poses]))
        t0 = min(max(t0, t[0]), t[-1])
        return (np.array([np.interp(t0, t, xy[:, 0]), np.interp(t0, t, xy[:, 1])]), float(np.interp(t0, t, yaw)))

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
            meta["desire"] = self.route.desire() if self.cfg.get("desire", True) else DESIRE_NONE
        times = np.arange(1, 21) * 0.25
        if self.replay:
            info, out = {"replay": True}, {}
            path = self._replay_path(t_frame, times)
        else:
            wire.send(self.sock, meta, arrays)
            info, out = wire.recv(self.sock)
            if self.alpamayo:
                path = resample(out["t"], out["xy"], times)
            else:
                yaw = out["yaw"] if self.plan_origin == "rear" else None
                path = resample(out["t"], rigs.openpilot_plan_to_rig(out["pos"], yaw, self.op_mount), times)
        if self.first_set_t is None:
            self.first_set_t = t_frame
        warm = t_frame - self.first_set_t < self.warmup_s - 1e-6
        self.warm_now = warm
        if self.drive == "oracle":
            drive_path = self.oracle.trajectory(self.poses[-1][1], self.poses[-1][2])
        else:
            drive_path = path
            if not warm:
                self.curvature = float(info["curvature"]) if "curvature" in info else None
                if "accel" in info:   # modeld smooths desired accel with a 0.3 s time constant
                    dt = DELTA * self.plan_every * (1 if self.op_tick <= DELTA else self.op_tick / DELTA)
                    a = float(info["accel"])
                    self.accel_des = a if self.accel_des is None else \
                        self.accel_des + (1 - math.exp(-dt / 0.3)) * (a - self.accel_des)
        zoo_meta = None
        if warm:
            accepted = False
        elif self.zoo is not None:
            if self.zoo_lateral == "fixed":   # P1: the fixed controller tracks the raw plan for the steer only
                self.controller.update(np.asarray(drive_path, float), t_frame)
            if self.zoo.cadence == "tick":   # control_pid runs every tick in __call__ on this held plan
                self.zoo.hold(np.asarray(drive_path, float), times, t_frame, self._pose_at(t_frame))
            else:
                p, t = self.zoo.prepare(np.asarray(drive_path, float), times)
                steer, throttle, brake, zoo_meta = self.zoo.control(p, t, speed, self.zoo_target)
                self.zoo_control = (throttle, steer, brake)
            accepted = True
        else:
            accepted = self.controller.update(np.asarray(drive_path, float), t_frame)
        self.n_plans += 1
        ms = 1e3 * (time.perf_counter() - t_start)
        self.timings["plan_ms"].append(ms)
        rec = {"frame": f, "t": t_frame, "set_frames": [s[0] for s in self.cam_sets], "cam_frames": cam_frames,
               "speed": speed,
               "accepted": accepted, "path": np.round(path, 3).tolist(), "round_trip_ms": round(ms, 1),
               "pose": [float(v) for v in self.poses[-1][1]] + [float(self.poses[-1][2])],
               "route_index": self.route.i, "warmup": bool(warm)}
        if zoo_meta is not None:
            rec["zoo_pid"] = zoo_meta
        if not self.alpamayo and not self.replay:
            rec["plan_pos"] = np.round(out["pos"], 3).tolist()
            rec["plan_yaw"] = np.round(out["yaw"], 5).tolist() if "yaw" in out else None
        rec.update(self._truth())
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
