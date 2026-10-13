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
(`wait_start`: the simulation is held at its first tick for as long as the driver rests; `wait_max_s` > 0 caps it, 0 = no cap).
During the hold the agent resets the leaderboard's agent watchdog (ScenarioManager._agent_watchdog, found on the call stack;
without it the hold falls back to a 240 s cap) and writes `hold_beat` into the attempt's heartbeat.json so b2d_run's stall
check sees a live route.

Config JSON (--agent-config, optional; missing keys take DEFAULT): steer_gain, steer_expo (steer = gain * ((1 - e) s + e s^3),
the page has its own curve, so the default is identity); throttle_gamma, throttle_gain (throttle = min(1, gain * t^gamma), default
0.6 / 1.3: a light touch already pulls, the user found the linear map dull), brake_gamma, brake_gain (default 1.0 / 1.0: the MKZ brake
is already sharp, 0.05 gives about -3.2 m/s^2; the page's pedal gain scales both pedals before sending); realtime, publish_every, scale, wait_start, wait_max_s, road.
Per attempt (B2D_ATTEMPT_OUT): human.jsonl (one line per tick), human_summary.json and metric_info.json (the Bench2Drive base
agent's get_metric_info every driven tick, as the shipped agents log it; read by tools/efficiency_smoothness_benchmark.py).

Stop at the first event (env HUMAN_STOP_ON, default off here; human_drive.sh sets "collision"): a comma list of `collision`
and `offroute` (> 3.5 m from the dense route for 1 s). 0.5 s after the first such event the agent writes stop_event.json and
raises, so the leaderboard ends the route there ("Failed - Agent crashed") and scores DS / RC up to that point; b2d_run counts the
attempt as finished and moves to the next route. Evaluation only (simulator truth), as `stop_on` in
experiments/b2d_monitor/lib/op_arb_viz_agent.py (decision 288).
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

DEFAULT = dict(steer_gain=1.0, steer_expo=0.0, throttle_gamma=0.6, throttle_gain=1.3, brake_gamma=1.0, brake_gain=1.0, realtime=1.0, publish_every=1, scale=2, wait_start=True, wait_max_s=0.0,
               road=False)
F_MODEL, CX_MODEL, CY_MODEL, W_MODEL, H_MODEL = 455.0, 256.0, 0.5 * (256 + 47.6), 512, 256   # modeld's SBIGMODEL_K frame
CMD_NAMES = {Z.LEFT: "turn left", Z.RIGHT: "turn right", Z.STRAIGHT: "straight through junction",
             Z.LANEFOLLOW: "follow lane", Z.CHANGE_LEFT: "lane change left", Z.CHANGE_RIGHT: "lane change right"}


class StopRoute(RuntimeError):
    """Raised on purpose to end the route at its first event (HUMAN_STOP_ON)."""


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
        self.metric_info, self.hero_actor = {}, None
        self.stop_on = {k.strip() for k in os.environ.get("HUMAN_STOP_ON", "").lower().split(",")} & {"collision", "offroute"}
        self.stop_sensor, self.stop_hits, self.stop_event, self.off_t = None, [], None, None
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
            t0 = t_beat = time.time()
            wd = self._agent_watchdog()
            cap = float(self.cfg["wait_max_s"]) or (1e9 if wd is not None else 240.0)
            while time.time() - t0 < cap:
                c, st = self._control()
                self._publish(frame, now, c, st, waiting=True)
                if not st["stale"] and st["raw_throttle"] > 0.2:
                    break
                if time.time() - t_beat > 5.0:
                    t_beat = time.time()
                    self._hold_beat(wd, t_beat - t0)
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
        try:                                           # official comfort metric input (Bench2Drive base agent)
            if self.hero_actor is None:
                self.hero_actor = CDP.get_hero_actor()
            self.metric_info[self.tick] = self.get_metric_info()
        except Exception as e:
            if len(self.metric_info) == 0:
                print("metric_info: %r" % (e,), file=sys.stderr, flush=True)
        if self.stop_on:
            try:
                self._stop_check(now)
            except StopRoute:
                raise
            except Exception as e:  # a broken check must never end a human's route: drop it for this route
                print("stop check disabled: %r" % (e,), file=sys.stderr, flush=True)
                self.stop_on = set()
        c, st = self._control()                        # read the wheel after the sleep: freshest input
        self.n_stale += st["stale"]
        if self.preview is not None and self.tick % int(self.cfg["publish_every"]) == 0:
            self._publish(frame, now, c, st)
        self.log.write(json.dumps({"t": round(now, 2), "wall": round(time.time(), 3), "steer_in": round(st["steer"], 3),
                                   "steer": round(c.steer, 3), "throttle": round(c.throttle, 3), "brake": round(c.brake, 3),
                                   "rev": c.reverse, "hb": c.hand_brake, "age": round(min(st["age_s"], 99.0), 3),
                                   "stale": st["stale"], "rx_hz": st.get("rx_hz"), "v": round(self.v_hist[-1][1], 2) if self.v_hist else None,
                                   "ri": int(self.route.i) if self.route is not None else None,
                                   "slept_ms": round(1e3 * slept, 1)}) + "\n")
        self.t_prev_exit = time.time()
        return c

    def _agent_watchdog(self):
        """The leaderboard's per-tick agent watchdog (ScenarioManager._agent_watchdog), from the caller's frames, or None."""
        f = sys._getframe(1)
        while f is not None:
            wd = getattr(f.f_locals.get("self"), "_agent_watchdog", None)
            if wd is not None and hasattr(wd, "update"):
                return wd
            f = f.f_back
        return None

    def _hold_beat(self, wd, held_s):
        """Keep a resting driver's route alive: reset the agent watchdog and mark heartbeat.json for b2d_run's stall check."""
        try:
            if wd is not None:
                wd.update()
            p = os.path.join(self.out, "heartbeat.json")
            hb = {}
            if os.path.exists(p):
                with open(p) as fh:
                    hb = json.load(fh)
            hb.update(hold_beat=round(held_s, 1), hold_t=time.time())
            with open(p + ".hold", "w") as fh:
                json.dump(hb, fh)
            os.replace(p + ".hold", p)
        except Exception as e:  # never end a route over this
            print("hold beat: %r" % (e,), file=sys.stderr, flush=True)

    def _stop_check(self, now, offroute_m=3.5, offroute_s=1.0, grace_s=0.5):
        hero = self.hero or CDP.get_hero_actor()
        if hero is None:
            return
        if "collision" in self.stop_on and self.stop_sensor is None:
            bp = CDP.get_world().get_blueprint_library().find("sensor.other.collision")
            self.stop_sensor = CDP.get_world().spawn_actor(bp, carla.Transform(), attach_to=hero)
            self.stop_sensor.listen(lambda e: self.stop_hits.append((float(GameTime.get_time()), e.other_actor.type_id)))
        if self.stop_event is None and self.stop_hits:
            t, what = self.stop_hits[0]
            self.stop_event = {"t": round(t, 2), "kind": "collision", "with": what}
        if "offroute" in self.stop_on and self.route is not None:
            loc = hero.get_location()
            d = float(np.min(np.hypot(self.route.xy[:, 0] - loc.x, self.route.xy[:, 1] - loc.y)))
            self.off_t = (self.off_t or now) if d > offroute_m else None
            if self.stop_event is None and self.off_t is not None and now - self.off_t >= offroute_s:
                self.stop_event = {"t": round(now, 2), "kind": "off_route", "dist_m": round(d, 2)}
        if self.stop_event is not None and now - self.stop_event["t"] >= grace_s:
            r = self.route
            self.stop_event["progress"] = round(float(r.s[r.i] / max(r.s[-1], 1e-3)), 3) if r is not None else None
            with open(os.path.join(self.out, "stop_event.json"), "w") as fh:
                json.dump(self.stop_event, fh)
            raise StopRoute("stop_on: route ended at the first %s (%s)" % (self.stop_event["kind"], self.stop_event))

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
                      "rx_hz": st.get("rx_hz"), "stale_reason": st.get("reason"),
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
                   "late_reanchors": self.n_late, "stale_ticks": self.n_stale, "config": self.cfg,
                   "stop_on": sorted(self.stop_on), "stop_event": self.stop_event}
        if self.stop_sensor is not None:
            try:
                self.stop_sensor.stop()
                self.stop_sensor.destroy()
            except Exception:
                pass
            self.stop_sensor = None
        try:
            with open(os.path.join(self.out, "human_summary.json"), "w") as fh:
                json.dump(summary, fh, indent=1)
            with open(os.path.join(self.out, "metric_info.json"), "w") as fh:
                json.dump({str(k): v for k, v in self.metric_info.items()}, fh)
            print("human summary: %s" % json.dumps(summary), flush=True)
            self.log.close()
        finally:
            if self.preview is not None:
                self.preview.close()
                self.preview = None
