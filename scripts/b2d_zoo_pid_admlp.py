"""Bench2DriveZoo's official trajectory -> VehicleControl PID, run as shipped, for the Alpamayo zero-shot exam agent.

Imported unmodified from the pinned Bench2DriveZoo checkout ($B2D_ZOO_ROOT, branch tcp/admlp, 8a08b07):
`ADMLP.model.ADMLP.control_pid` with `ADMLP.model.PIDController` and `ADMLP.config.GlobalConfig` (turn PID
0.75/0.75/0.3 window 40, speed PID 5/0.5/1 window 40, aim point nearest 4 m, target speed = mean waypoint spacing x 2,
brake below 0.4 m/s or above 1.1x, clip_delta 0.25, throttle <= 0.75). This is the same algorithm and gains as
TCP/model.py and the UniAD/VAD team_code/pid_controller.py (vendored separately in scripts/b2d_zoo_pid.py).
Its target point is the shipped AD-MLP route logic: `ADMLPAgent._init` (GPS reference by fsolve,
`RoutePlanner(4.0, 50.0)` on the base class's 50 m route) and `gps_to_location`, advanced every tick.

Needs torch (control_pid takes a tensor), so the route subprocess runs in envs/b2d-tcp (Python 3.8, torch, carla).
Frames: waypoints and target are (forward, right) from the ego GNSS / rear axle, as AD-MLP feeds them.
"""
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

ZOO = Path(os.environ.get("B2D_ZOO_ROOT", Path(os.environ.get("DATA_DIR", "")) / "third_party/Bench2DriveZoo"))
if str(ZOO) not in sys.path:
    sys.path.insert(0, str(ZOO))

import torch  # noqa: E402
from ADMLP.config import GlobalConfig  # noqa: E402
from ADMLP.model import ADMLP, PIDController  # noqa: E402
from team_code.admlp_b2d_agent import ADMLPAgent  # noqa: E402


class ZooPID(object):
    """One per route. `route(agent)` after the global plan is set, `target(...)` every tick, `control(...)` once
    per plan (the 2 Hz AD-MLP agent calls control_pid once per prediction and holds the control in between)."""

    def __init__(self):
        cfg = GlobalConfig()
        self.net = types.SimpleNamespace(
            config=cfg,
            turn_controller=PIDController(K_P=cfg.turn_KP, K_I=cfg.turn_KI, K_D=cfg.turn_KD, n=cfg.turn_n),
            speed_controller=PIDController(K_P=cfg.speed_KP, K_I=cfg.speed_KI, K_D=cfg.speed_KD, n=cfg.speed_n))
        self.agent = self.planner = None

    def route(self, agent):
        ADMLPAgent._init(agent)
        self.agent, self.planner = agent, agent._route_planner

    def target(self, gps, compass):
        """Shipped: next route node in the ego frame (forward, right), from the raw GNSS and compass."""
        if math.isnan(compass):
            compass = 0.0
        pos = ADMLPAgent.gps_to_location(self.agent, gps[:2])
        next_wp, _ = self.planner.run_step(pos)
        theta = compass - np.pi / 2
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return tuple(R.dot(np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])))

    def control(self, waypoints_fr, speed, target):
        """(6, 2) forward/right waypoints at 0.5..3.0 s -> (steer, throttle, brake, metadata) as the AD-MLP agent
        applies them: control_pid, clip, brake > 0 -> 1 with throttle 0. The agent's extra low-speed throttle cap
        (throttle <= 0.05 above 2.5-3 m/s) is not applied; see the deviation note in the exam doc."""
        wp = torch.tensor(np.asarray(waypoints_fr, np.float32)[None])
        steer, throttle, brake, meta = ADMLP.control_pid(self.net, wp, speed, target)
        steer = float(np.clip(float(steer), -1, 1))
        throttle = float(np.clip(float(throttle), 0, 0.75))
        brake = float(np.clip(float(brake), 0, 1))
        if brake > 0:
            brake = 1.0
        if brake > 0.5:
            throttle = 0.0
        return steer, throttle, brake, {k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()}
