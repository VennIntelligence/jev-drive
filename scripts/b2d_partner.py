"""Shared control for zero-shot models in Bench2Drive: a learned partner (TCP) drives the phases the model structurally
cannot handle, the model drives everything else. Pre-registration: todos/2026-09-24-zeroshot-exam/openpilot-migration.md,
section D3. Python 3.8, runs inside the route subprocess (envs/b2d-tcp: torch + carla).

Partner: the official Bench2DriveZoo TCP agent (team_code/tcp_b2d_agent.py, branch tcp/admlp 8a08b07,
PLANNER_TYPE=only_traj, checkpoint tcp_b2d.ckpt sha256 e6573ff1...) run unmodified, with its own three 1600x900
cameras, its own route planner / target point and its own PID and final throttle cap. It is stepped on every tick,
whoever drives, so its internal state is always current.

Arbiter - observable state only (speed, route geometry and the ego's progress along it, time), never outcomes:
  the partner drives while any of
    standstill latch   set when speed < 0.1 m/s has lasted 0.5 s (so also at the route start); cleared when
                       speed >= 1.0 m/s has lasted 1.0 s. (TCP's shipped throttle cap - throttle <= 0.05 above
                       1.5 m/s, 1.0 m/s while turning - keeps it near 1.5-2 m/s, so a 3 m/s release would never fire.)
    junction zone      the ego is between 15 m before the start and 5 m after the end (route arc length) of a
                       LEFT / RIGHT route command                                            (only if junctions=True)
    model warm-up      the model has not finished its warm-up
  the model drives otherwise. A change of driver is blended linearly over 0.5 s (10 ticks) on steer, throttle, brake.
"""
import os
import sys
from pathlib import Path

import numpy as np

STANDSTILL_V, STANDSTILL_S = 0.1, 0.5
RELEASE_V, RELEASE_S = 1.0, 1.0
ZONE_BEFORE_M, ZONE_AFTER_M = 15.0, 5.0
BLEND_TICKS = 10
LEFT, RIGHT = 1, 2


class TCPPartner(object):
    def __init__(self, ckpt, zoo_root=None):
        os.environ.setdefault("PLANNER_TYPE", "only_traj")
        os.environ.setdefault("IS_BENCH2DRIVE", "1")          # the shipped agent reads its 'bev' sensor only then
        os.environ.pop("SAVE_PATH", None)
        root = zoo_root or os.environ.get("B2D_ZOO_ROOT") or str(Path(os.environ["DATA_DIR"]) / "third_party/Bench2DriveZoo")
        if root not in sys.path:
            sys.path.insert(0, root)
        from leaderboard.autoagents.autonomous_agent import Track
        from leaderboard.envs.sensor_interface import SensorInterface
        from team_code.tcp_b2d_agent import TCPAgent
        a = TCPAgent.__new__(TCPAgent)                      # the base __init__ only needs the hero; set its fields
        a.track, a._global_plan, a._global_plan_world_coord = Track.SENSORS, None, None
        a.sensor_interface, a.wallclock_t0 = SensorInterface(), None
        a.setup(ckpt + "+partner")
        self.agent = a
        self.tags = [s["id"] for s in a.sensors()]
        self.last = (0.0, 0.0, 1.0)

    def sensors(self, taken):
        return [dict(s) for s in self.agent.sensors() if s["id"] not in taken]

    def camera_tags(self):
        return [s["id"] for s in self.agent.sensors() if s["type"].startswith("sensor.camera")]

    def set_global_plan(self, gps, world):
        self.agent.set_global_plan(gps, world)

    def step(self, data, timestamp):
        """data: {tag: (frame, value)} with every partner sensor; returns (throttle, steer, brake)."""
        if not all(t in data for t in self.tags):
            return self.last
        c = self.agent.run_step({t: data[t] for t in self.tags}, timestamp)
        self.last = (float(c.throttle), float(c.steer), float(c.brake))
        return self.last


class Arbiter(object):
    def __init__(self, route_xy, route_cmd, junctions=True):
        xy, cmd = np.asarray(route_xy, float), np.asarray(route_cmd)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        self.zones = []
        i = 0
        while junctions and i < len(cmd):
            if cmd[i] in (LEFT, RIGHT):
                j = i
                while j + 1 < len(cmd) and cmd[j + 1] == cmd[i]:
                    j += 1
                self.zones.append((self.s[i] - ZONE_BEFORE_M, self.s[j] + ZONE_AFTER_M))
                i = j + 1
            else:
                i += 1
        self.latch, self.low_t, self.high_t = True, 0.0, 0.0
        self.w = 0.0                                          # weight of the model's control, 0 = partner
        self.driver = "partner"

    def step(self, dt, speed, route_index, model_ready):
        """-> weight of the model's control in [0, 1] and the reason the partner holds (or '')."""
        if speed < STANDSTILL_V:
            self.low_t += dt
            if self.low_t >= STANDSTILL_S:
                self.latch = True
        else:
            self.low_t = 0.0
        if speed >= RELEASE_V:
            self.high_t += dt
            if self.high_t >= RELEASE_S:
                self.latch = False
        else:
            self.high_t = 0.0
        s_now = self.s[min(route_index, len(self.s) - 1)]
        zone = any(a <= s_now <= b for a, b in self.zones)
        why = "warmup" if not model_ready else "standstill" if self.latch else "junction" if zone else ""
        target = 0.0 if why else 1.0
        step = 1.0 / BLEND_TICKS
        self.w = min(target, self.w + step) if target > self.w else max(target, self.w - step)
        self.driver = "model" if self.w >= 1.0 else "partner" if self.w <= 0.0 else "blend"
        return self.w, why

    @staticmethod
    def mix(w, model, partner):
        return tuple(float(w * m + (1.0 - w) * p) for m, p in zip(model, partner))
