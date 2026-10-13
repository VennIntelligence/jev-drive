#!/usr/bin/env python
"""Bench2Drive leaderboard agent driven by a human with the phone wheel (wheel_client), seeing what openpilot sees.

Every tick it reads WheelClient.state() and returns carla.VehicleControl(steer, throttle, brake, hand_brake, reverse);
a stale wheel (no frame for 0.3 s) gives steer 0, throttle 0, brake 1, handbrake on. Python 3.8, envs/carla.

Sensors: only openpilot's wide camera, exactly as the openpilot agents mount and render it
(scripts/zeroshot_rigs.openpilot_sensor_specs: 1928 x 1208, focal 567 px, windshield top OP_MOUNT_RIG, every tick). The road
camera is the model's second input but is never displayed, so it is not rendered ("road": true adds it back). The preview
(scripts/drive_runtime/preview.LivePreview, env B2D_PREVIEW_DIR) gets the wide frame warped as modeld warps it for the model
(zero calibration, focal 455, 512 x 256, at `scale` x the model resolution), the route command banner, the benchmark route
projected onto flat ground with 10/20/40/80 m marks, ego state and a "human" block (wheel values, input age, route progress,
sim speed). The overlays use the hero's true pose: they are a display for the human, control is the phone's alone.

Pacing: the leaderboard ticks synchronously as fast as it can; the agent sleeps so simulated time never runs ahead of wall
time x `realtime` (1.0 = real time; 0 = unpaced). When the sim is slower than that, nothing is slept and the lag is
reported (wall s per sim s, raw = without the sleeps). The route clock starts at the driver's first throttle press
(`wait_start`, the simulation is held at its first tick, at most `wait_max_s`, below the leaderboard's 300 s agent watchdog).

Config JSON (--agent-config, optional; missing keys take DEFAULT): steer_gain, steer_expo (steer = gain * ((1 - e) s + e s^3),
the page has its own curve, so the default is identity); throttle_gamma, throttle_gain (throttle = min(1, gain * t^gamma), default
0.6 / 1.3: a light touch already pulls, the user found the linear map dull), brake_gamma, brake_gain (default 1.0 / 1.0: the MKZ brake
is already sharp, 0.05 gives about -3.2 m/s^2; the page's pedal gain scales both pedals before sending); realtime, publish_every, scale, wait_start, wait_max_s, road.
Per attempt (B2D_ATTEMPT_OUT): human.jsonl (one line per tick) and human_summary.json.
"""
import json
import math
import os
import sys
import time

import carla
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as CDP
from srunner.scenariomanager.timer import GameTime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, os.path.join(REPO, "scripts")]
import b2d_zeroshot_agent as Z  # noqa: E402  (Route, command codes)
import zeroshot_rigs as rigs  # noqa: E402
from b2d_controller_adapter import world_to_local  # noqa: E402
from wheel_client import WheelClient  # noqa: E402

DEFAULT = dict(steer_gain=1.0, steer_expo=0.0, throttle_gamma=0.6, throttle_gain=1.3, brake_gamma=1.0, brake_gain=1.0, realtime=1.0, publish_every=1, scale=2, wait_start=True, wait_max_s=240.0,
               road=False)
F_MODEL, CX_MODEL, CY_MODEL, W_MODEL, H_MODEL = 455.0, 256.0, 0.5 * (256 + 47.6), 512, 256   # modeld's SBIGMODEL_K frame
CMD_NAMES = {Z.LEFT: "turn left", Z.RIGHT: "turn right", Z.STRAIGHT: "straight through junction",
             Z.LANEFOLLOW: "follow lane", Z.CHANGE_LEFT: "lane change left", Z.CHANGE_RIGHT: "lane change right"}


def get_entry_point():
    return "HumanAgent"


def wide_index(scale):
    """Flat BGRA pixel index of the CARLA wide render for every model-frame pixel (nearest neighbour, as modeld)."""
    w, h = rigs.OP_CAMERA_WH
    f = rigs.OP_FOCAL["wide"]
    u, v = np.meshgrid((np.arange(W_MODEL * scale) + 0.5) / scale - 0.5, (np.arange(H_MODEL * scale) + 0.5) / scale - 0.5)
    x = np.clip(np.rint(f * (u - CX_MODEL) / F_MODEL + w / 2), 0, w - 1).astype(np.int64)
    y = np.clip(np.rint(f * (v - CY_MODEL) / F_MODEL + h / 2), 0, h - 1).astype(np.int64)
    return (y * w + x).ravel()


def project(dev, scale):
    """Camera device-frame points (x forward, y right, z down) -> pixel polyline of the scaled model frame."""
    dev = np.asarray(dev, float).reshape(-1, 3)
    dev = dev[dev[:, 0] > 0.8]
    if not len(dev):
        return []
    u = scale * (F_MODEL * dev[:, 1] / dev[:, 0] + CX_MODEL)
    v = scale * (F_MODEL * dev[:, 2] / dev[:, 0] + CY_MODEL)
    return np.round(np.stack([u, v], -1), 1).tolist()


def pedal(v, gamma, gain):
    return float(min(1.0, gain * max(0.0, min(1.0, v)) ** gamma))


def shape_steer(s, gain, expo):
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.clip(gain * ((1.0 - expo) * s + expo * s ** 3), -1.0, 1.0))


class HumanAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        cfg = {}
        src = (path_to_conf_file or "").split("+")[0]
        if src and os.path.isfile(src):
            with open(src) as fh:
                cfg = json.load(fh)
        self.cfg = dict(DEFAULT, **cfg)
        self.mount = rigs.OP_MOUNT_RIG
        self.cam_specs = [s for s in rigs.openpilot_sensor_specs(rigs.OP_CAMERA_TICK, self.mount)
                          if s["id"] == "OP_WIDE" or self.cfg["road"]]
        self.wheel = WheelClient()
        self.scale = int(self.cfg["scale"])
        self.idx = wide_index(self.scale)
        d = os.environ.get("B2D_PREVIEW_DIR")
        self.preview = None
        if d:
            sys.path.insert(0, os.path.join(REPO, "scripts"))
            from drive_runtime.preview import LivePreview
            self.preview = LivePreview(d)
        self.out = os.environ.get("B2D_ATTEMPT_OUT") or "."
        os.makedirs(self.out, exist_ok=True)
        self.log = open(os.path.join(self.out, "human.jsonl"), "w", buffering=1)
        self.rid = os.environ.get("BENCHMARK_ROUTE_ID", "?")
        self.tick, self.started, self.wait_s, self.hero, self.viz_err = 0, not self.cfg["wait_start"], 0.0, None, 0
        self.anchor = None                       # (wall, sim) of the pacing reference
        self.t_prev_in, self.prev_wait, self.prev_sleep = None, 0.0, 0.0   # previous run_step entry, its start wait and pacing sleep
        self.raw_s = self.slept_s = self.sim_s = 0.0
        self.n_stale = self.n_late = 0
        self.v_hist = []
        self.route = None
        if getattr(self, "_dense_plan", None):
            self.route = Z.Route(self._dense_plan)

    def sensors(self):
        return [dict(s) for s in self.cam_specs]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense_plan = list(global_plan_world_coord)        # the base class keeps a 50 m subsample only
        if hasattr(self, "cfg"):
            self.route = Z.Route(self._dense_plan)

    # ---- per tick ----------------------------------------------------------------------------------------------------
    def _pose(self):
        if self.hero is None:
            self.hero = CDP.get_hero_actor()
        tf, vel = self.hero.get_transform(), self.hero.get_velocity()
        yaw = math.radians(tf.rotation.yaw)
        xy = np.array([tf.location.x + rigs.REAR_AXLE_X * math.cos(yaw), tf.location.y + rigs.REAR_AXLE_X * math.sin(yaw)])
        return xy, yaw, math.hypot(vel.x, vel.y)

    def _control(self):
        st = self.wheel.state()
        steer = shape_steer(st["steer"], float(self.cfg["steer_gain"]), float(self.cfg["steer_expo"]))
        k = self.cfg
        c = carla.VehicleControl(throttle=pedal(st["throttle"], float(k["throttle_gamma"]), float(k["throttle_gain"])), steer=steer,
                                 brake=pedal(st["brake"], float(k["brake_gamma"]), float(k["brake_gain"])), hand_brake=bool(st["handbrake"]),
                                 reverse=bool(st["reverse"]))
        return c, st

    def run_step(self, input_data, timestamp):
        t_in = time.time()
        self.tick += 1
        now = float(timestamp)
        if self.t_prev_in is not None:                 # one tick: simulator + render + transfer + this agent
            self.raw_s += t_in - self.t_prev_in - self.prev_wait - self.prev_sleep
            self.slept_s += self.prev_sleep
            self.sim_s += 0.05
        self.t_prev_in, self.prev_wait, self.prev_sleep = t_in, 0.0, 0.0
        frame = input_data["OP_WIDE"][1]
        if not self.started:                           # hold the world at its first tick until the driver presses throttle
            t0 = time.time()
            while time.time() - t0 < float(self.cfg["wait_max_s"]):
                c, st = self._control()
                self._publish(frame, now, c, st, waiting=True)
                if not st["stale"] and st["raw_throttle"] > 0.2:
                    break
                time.sleep(0.05)
            self.started, self.wait_s = True, round(time.time() - t0, 1)
            self.anchor, self.prev_wait = None, time.time() - t0
        slept = 0.0
        rt = float(self.cfg["realtime"])
        if rt > 0:
            wall = time.time()
            if self.anchor is None:
                self.anchor = (wall, now)
            target = self.anchor[0] + (now - self.anchor[1]) / rt
            if target > wall:
                slept = target - wall
                time.sleep(slept)
            elif wall - target > 0.5:                  # far behind: re-anchor, never burst to catch up
                self.anchor, self.n_late = (wall, now), self.n_late + 1
        self.prev_sleep = slept
        c, st = self._control()                        # read the wheel after the sleep: freshest input
        self.n_stale += st["stale"]
        if self.preview is not None and self.tick % int(self.cfg["publish_every"]) == 0:
            self._publish(frame, now, c, st)
        self.log.write(json.dumps({"t": round(now, 2), "wall": round(time.time(), 3), "steer_in": round(st["steer"], 3),
                                   "steer": round(c.steer, 3), "throttle": round(c.throttle, 3), "brake": round(c.brake, 3),
                                   "rev": c.reverse, "hb": c.hand_brake, "age": round(min(st["age_s"], 99.0), 3),
                                   "stale": st["stale"], "v": round(self.v_hist[-1][1], 2) if self.v_hist else None,
                                   "ri": int(self.route.i) if self.route is not None else None,
                                   "slept_ms": round(1e3 * slept, 1)}) + "\n")
        self.t_prev_exit = time.time()
        return c

    def _publish(self, frame, now, c, st, waiting=False):
        try:
            self._publish_inner(frame, now, c, st, waiting)
        except Exception as e:  # the view must never stop the car
            self.viz_err += 1
            if self.viz_err <= 3:
                print("preview error: %r" % (e,), file=sys.stderr, flush=True)

    def _publish_inner(self, frame, now, c, st, waiting):
        if self.preview is None:
            return
        img = np.asarray(frame).reshape(-1, 4)[self.idx][:, 2::-1].reshape(H_MODEL * self.scale, W_MODEL * self.scale, 3)
        xy, yaw, speed = self._pose()
        if not self.v_hist or self.v_hist[-1][0] != now:
            self.v_hist = (self.v_hist + [(now, speed, yaw)])[-11:]
        (t0, v0, y0), (t1, v1, y1) = self.v_hist[0], self.v_hist[-1]
        dt = max(t1 - t0, 1e-3)
        m = self.mount

        def to_dev(local):                             # rig (rear axle on the ground, x forward, y left) -> camera device frame
            local = np.asarray(local, float).reshape(-1, 2)
            return np.stack([local[:, 0] - m[0], -(local[:, 1] - m[1]), np.full(len(local), m[2])], -1)

        route_px, marks, cmd, dist, prog = [], [], None, None, 0.0
        r = self.route
        if r is not None:
            r.progress(xy)
            s0 = r.s[r.i]
            j = int(np.searchsorted(r.s, s0 + 120.0)) + 1
            route_px = project(to_dev(world_to_local(r.xy[r.i:j], xy, yaw)), self.scale)
            for dm in (10, 20, 40, 80):
                k = int(np.searchsorted(r.s, s0 + dm))
                if k < len(r.xy):
                    p = project(to_dev(world_to_local(r.xy[k:k + 1], xy, yaw)), self.scale)
                    if p:
                        marks.append([dm] + p[0])
            cmd, dist = r.next_maneuver([Z.LEFT, Z.RIGHT, Z.STRAIGHT, Z.CHANGE_LEFT, Z.CHANGE_RIGHT])
            prog = float(s0 / max(r.s[-1], 1e-6))
        sim = max(self.sim_s, 1e-6)
        state = {
            "agent": "human", "running": True, "t": time.time(), "route": self.rid, "step": self.tick, "sim_t": round(now, 2),
            "waiting": waiting, "cmd": CMD_NAMES.get(cmd, "follow lane") if cmd is not None else "follow lane (no command left)",
            "cmd_code": int(cmd) if cmd is not None else 0, "cmd_dist": None if dist is None else round(dist, 1),
            "speed": round(speed, 2), "yaw_rate": round(math.degrees((y1 - y0 + math.pi) % (2 * math.pi) - math.pi) / dt, 2),
            "accel": round((v1 - v0) / dt, 2),
            "steer": round(float(c.steer), 3), "throttle": round(float(c.throttle), 3), "brake": round(float(c.brake), 3),
            "route_px": route_px, "route_marks": marks, "img_scale": self.scale,
            "human": {"steer_in": round(st["steer"], 3), "throttle_in": round(st["raw_throttle"], 3), "reverse": bool(st["reverse"]),
                      "handbrake": bool(st["handbrake"]), "stale": bool(st["stale"]), "age_ms": round(1e3 * min(st["age_s"], 99.0), 1),
                      "progress": round(prog, 4), "route_m": round(float(r.s[-1]), 0) if r is not None else None,
                      "wall_per_sim": round((self.raw_s + self.slept_s) / sim, 3) if self.sim_s else None,
                      "raw_per_sim": round(self.raw_s / sim, 3) if self.sim_s else None, "late": self.n_late,
                      "gain": self.cfg["steer_gain"], "expo": self.cfg["steer_expo"],
                      "thr_gamma": self.cfg["throttle_gamma"], "thr_gain": self.cfg["throttle_gain"], "realtime": self.cfg["realtime"]},
        }
        self.preview.publish(state, {"wide": img})

    def destroy(self):
        sim = max(self.sim_s, 1e-6)
        summary = {"route": self.rid, "ticks": self.tick, "sim_s": round(self.sim_s, 2), "wait_s": self.wait_s,
                   "wall_s": round(self.raw_s + self.slept_s, 2), "raw_s": round(self.raw_s, 2), "slept_s": round(self.slept_s, 2),
                   "wall_per_sim": round((self.raw_s + self.slept_s) / sim, 3), "raw_per_sim": round(self.raw_s / sim, 3),
                   "late_reanchors": self.n_late, "stale_ticks": self.n_stale, "config": self.cfg}
        try:
            with open(os.path.join(self.out, "human_summary.json"), "w") as fh:
                json.dump(summary, fh, indent=1)
            print("human summary: %s" % json.dumps(summary), flush=True)
            self.log.close()
        finally:
            if self.preview is not None:
                self.preview.close()
                self.preview = None
