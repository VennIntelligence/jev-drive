"""Thin wrapper that drives Bench2DriveZoo's official UniAD/VAD trajectory -> VehicleControl PID for the zero-shot
exam agent (scripts/b2d_zeroshot_agent.py, `"controller": "zoo_pid"`). NumPy only, Python 3.8 (envs/carla).

The controller itself is the verbatim vendored `team_code/pid_controller.py` (scripts/b2d_zoo_pid.py, branch
uniad/vad 498c1f7) and its target point comes from the shipped `RoutePlanner(4.0, 50.0)` (team_code/planner.py of the
pinned Zoo checkout, identical on both branches) on the base class's 50 m route, advanced every tick, exactly as the Zoo agents do.
This file only converts frames and applies the Zoo agents' clip / brake normalisation:

  waypoints  planner path (rig: x forward, y left) sampled at 0.5, 1.0 ... 3.0 s (the UniAD/VAD 6 x 0.5 s format),
             passed as (right, forward), the frame control_pid's angle = 90 deg - atan2(forward, right) assumes;
  target     next route node relative to the GNSS position, rotated by the compass (theta = compass - pi/2, the Zoo
             agents' convention) into (forward, right), then passed as (right, forward) as well;
  speed      the speedometer, m/s.

Geometry of the target: the Zoo agents map GNSS to CARLA world with a Mercator reference found by fsolve; here the
GNSS is mapped with the exact GPS/world fit of the dense route (b2d_controller_adapter.GPSProjector, < 1 mm residual)
and the planner runs on world coordinates. Same point, without the fsolve approximation.

Two adapter switches (todos/2026-09-24-zeroshot-exam/alpamayo-closed-loop-diagnosis.md, section 6); their defaults keep
the behaviour of the first Alpamayo full run and of the openpilot runs:
  forward_only  False | True: segments of the plan whose forward component is negative are dropped before sampling
                (the plan's speed clamped at >= 0: a car braking to a stop stays, it does not reverse). control_pid
                reads speed from waypoint distances regardless of sign, so without this a reversing "stop" plan is
                executed as forward throttle.
  cadence       "plan" (control_pid once per plan, held until the next: the AD-MLP Zoo agent) | "tick": control_pid every
                tick on the held plan, shifted by its age and re-expressed in the current rear-axle frame by odometry
                (the per-tick call of the UniAD / VAD Zoo agents).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

from b2d_zoo_pid import PIDController

ZOO = Path(os.environ.get("B2D_ZOO_ROOT", Path(os.environ.get("DATA_DIR", "")) / "third_party/Bench2DriveZoo"))
if str(ZOO) not in sys.path:
    sys.path.append(str(ZOO))
from team_code.planner import RoutePlanner  # noqa: E402  shipped, identical on tcp/admlp and uniad/vad

WAYPOINT_TIMES = np.arange(1, 7) * 0.5


class ZooPID(object):
    """One per route: `route(agent)` once the plan is set, `target(gps, compass)` every tick, `control(...)` once per
    plan. The agent holds the returned control until the next plan (the 2 Hz AD-MLP Zoo agent's cadence)."""

    def __init__(self, forward_only=False, cadence="plan"):
        assert cadence in ("plan", "tick"), cadence
        self.pid = PIDController()          # shipped defaults: turn 0.75/0.75/0.3 n40, speed 5/0.5/1 n40, ...
        self.planner = self.projector = None
        self.forward_only, self.cadence = bool(forward_only), cadence
        self.held = None                    # cadence "tick": (path with origin, times, plan time, rear-axle world pose)

    def route(self, agent):
        self.projector = agent.pose_filter.projector
        self.planner = RoutePlanner(4.0, 50.0)
        self.planner.set_route(agent._global_plan_world_coord, gps=False)

    def target(self, gps, compass):
        if math.isnan(compass):             # as the Zoo agents
            compass = 0.0
        pos = self.projector.project(np.asarray(gps, float)[:2])
        next_wp, _ = self.planner.run_step(pos)
        theta = compass - np.pi / 2
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        forward, right = R.dot(np.asarray(next_wp[:2], float) - pos)
        return float(forward), float(right)

    def prepare(self, path_xy_left, times):
        """The plan as the controller reads it: origin prepended at t = 0 (no effect on sampling at >= 0.25 s); with
        forward_only, every segment with a negative forward component contributes no motion."""
        p = np.vstack([[0.0, 0.0], np.asarray(path_xy_left, float)])
        t = np.r_[0.0, np.asarray(times, float)]
        if self.forward_only:
            d = np.diff(p, axis=0)
            p = np.vstack([[0.0, 0.0], np.cumsum(d * (d[:, :1] > 0), axis=0)])
        return p, t

    def hold(self, path_xy_left, times, t_plan, pose):
        """cadence "tick": keep the plan made at sim time t_plan from rear-axle world pose (xy, CARLA yaw)."""
        p, t = self.prepare(path_xy_left, times)
        self.held = (p, t, float(t_plan), (np.asarray(pose[0], float), float(pose[1])))

    def held_now(self, now, pose):
        """The held plan at sim time `now` from rear-axle world pose: rig frame (x forward, y left), times from now."""
        p, t, t_plan, ((x0, y0), h0) = self.held
        c0, s0 = math.cos(h0), math.sin(h0)
        wx = x0 + c0 * p[:, 0] + s0 * p[:, 1]           # CARLA world: y right, yaw clockwise; rig y is left
        wy = y0 + s0 * p[:, 0] - c0 * p[:, 1]
        (x1, y1), h1 = np.asarray(pose[0], float), float(pose[1])
        c1, s1 = math.cos(h1), math.sin(h1)
        dx, dy = wx - x1, wy - y1
        return np.stack([c1 * dx + s1 * dy, s1 * dx - c1 * dy], -1), t - (float(now) - t_plan)

    def control(self, path_xy_left, times, speed, target_fr):
        """path_xy_left: (N, 2) rig-frame path at `times` (s). Returns steer, throttle, brake, metadata."""
        p = np.stack([np.interp(WAYPOINT_TIMES, times, path_xy_left[:, k]) for k in range(2)], -1)
        wp = np.stack([-p[:, 1], p[:, 0]], -1)                        # (right, forward)
        tgt = np.array([target_fr[1], target_fr[0]])                  # (right, forward)
        steer, throttle, brake, meta = self.pid.control_pid(wp, np.float64(speed), tgt)
        steer = float(np.clip(float(steer), -1, 1))
        throttle = float(np.clip(float(throttle), 0, 0.75))
        brake = float(np.clip(float(brake), 0, 1))
        if brake > 0:
            brake, throttle = 1.0, 0.0
        return steer, throttle, brake, {k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()}
