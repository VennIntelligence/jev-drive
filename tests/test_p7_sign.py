"""Sign / frame chain of the p7 plan-position executor (lib/op_arb_agent.py _plan -> scripts/b2d_controller.Controller, P7.json)
against a kinematic plant in CARLA's left-handed world (x, y with yaw clockwise seen from above; positive steer = right;
IMU gyro z = +d(CARLA yaw)/dt, b2d_agent.py calibration r = .99995). experiments/op_route_ft/results/p7_sign_check.md.
python -m unittest tests.test_p7_sign -v   (no CARLA: carla / leaderboard / srunner are stubbed)"""
import json
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "lib")]
for _n in ("carla", "leaderboard", "leaderboard.autoagents", "leaderboard.autoagents.autonomous_agent", "srunner",
           "srunner.scenariomanager", "srunner.scenariomanager.timer"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
_aa = sys.modules["leaderboard.autoagents.autonomous_agent"]
_aa.AutonomousAgent = getattr(_aa, "AutonomousAgent", type("AutonomousAgent", (), {}))
_aa.Track = getattr(_aa, "Track", types.SimpleNamespace(SENSORS="SENSORS"))
sys.modules["srunner.scenariomanager.timer"].GameTime = object
import op_arb_agent as A  # noqa: E402
import b2d_zeroshot_agent as Z  # noqa: E402
import zeroshot_rigs as rigs  # noqa: E402
from b2d_controller import Controller  # noqa: E402
from b2d_controller_adapter import world_to_local  # noqa: E402

P7 = json.load(open(REPO / "experiments/b2d_tfv6/results/tfv6-controller/controller-eval/P7.json"))
MOUNT = (1.59, 0.0, 1.86)                       # spec preset camera (interface.B2D_SPEC_MOUNT)
T_OP = np.linspace(0.0, 10.0, 33)               # openpilot plan times (any grid; resampled to TIMES)


def op_arc(k_left, v):
    """openpilot plan of a car whose REAR AXLE drives a constant-curvature arc (no rear slip): camera track relative to the
    camera now, in openpilot's frame x forward, y RIGHT, yaw right-positive."""
    th = k_left * v * T_OP
    if abs(k_left) < 1e-9:
        xr, yr = v * T_OP, 0.0 * T_OP
    else:
        xr, yr = np.sin(th) / k_left, (1 - np.cos(th)) / k_left            # rear axle, y left
    d = MOUNT[0]
    qx, qy = xr + d * np.cos(th) - d, yr + d * np.sin(th)                    # camera minus camera(0), y left
    return np.stack([qx, -qy, 0 * qx], -1), -th


def drive_path(pos, yaw, s_fin):
    """The agent's chain for lat_src op + p7: openpilot_plan_to_rig -> resample to TIMES -> place([0,0] + plan, s_fin)."""
    op_path = Z.resample(T_OP, rigs.openpilot_plan_to_rig(pos, yaw, MOUNT), A.TIMES)
    return op_path, A.place(np.r_[[[0.0, 0.0]], op_path], s_fin)


def ctl():
    p = {k: v for k, v in P7.items() if k not in ("rear_axle_offset_m", "pose_lateral_coefficient_s2_per_m")}
    return Controller(preset="pursuit", **p)


def simulate(k_left, v=5.0, secs=6.0):
    """Closed loop on the CARLA plant: every 4th 20 Hz tick the same ego-relative openpilot arc (a constant-curvature plan is
    the same relative to a car on it). Returns CARLA (x, y, yaw) start / end, mean steer."""
    c, dt = ctl(), 0.05
    x, y, yaw = 10.0, -20.0, math.radians(30.0)
    st = [x, y, yaw]
    L, mx = P7["wheelbase"], math.radians(P7["max_steer_deg"])
    curve = np.asarray(P7["steering_curve"], float)
    steer, steers = 0.0, []
    pos, oyaw = op_arc(k_left, v)
    for i in range(int(secs / dt)):
        t = i * dt
        delta = steer * mx * float(np.interp(v * 3.6, curve[:, 0], curve[:, 1]))   # CARLA: steer > 0 = right
        kap = math.copysign(1.0 / (L / max(math.tan(abs(delta)), 1e-9) + 0.5 * P7["track_width_m"]), delta)   # inner wheel -> centre (Ackermann)
        gyro = v * kap                                                               # +d(CARLA yaw)/dt
        if i:
            x, y, yaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + gyro * dt
        if i % 4 == 0:
            c.update(drive_path(pos, oyaw, v * A.TIMES)[1], t)
        steer = c.step(t, v, -gyro)[1]                                               # as the agent: yaw rate left-positive
        steers.append(steer)
    return st, [x, y, yaw], float(np.mean(steers[20:]))


class Frames(unittest.TestCase):
    def test_conversion_left_is_y_positive(self):
        for k in (0.05, -0.05):
            op_path, path = drive_path(*op_arc(k, 5.0), 5.0 * A.TIMES)
            self.assertGreater(np.sign(k) * op_path[-1, 1], 1.0)          # openpilot y-right left arc -> rig y-left > 0
            self.assertGreater(np.sign(k) * path[-1, 1], 1.0)
            kap = 2 * op_path[3, 1] / (op_path[3] @ op_path[3])           # rear-axle chord curvature at 1 s = k exactly
            self.assertAlmostEqual(kap, k, delta=0.01)

    def test_world_to_local_is_y_left(self):
        yaw = math.radians(30.0)                                           # CARLA left of heading = (sin yaw, -cos yaw)
        left = np.array([[math.sin(yaw), -math.cos(yaw)]]) + [1.0, 2.0]
        self.assertAlmostEqual(world_to_local(left, np.array([1.0, 2.0]), yaw)[0, 1], 1.0)

    def test_first_steer_sign(self):
        for k, sign in ((0.05, -1), (-0.05, +1)):                         # left arc -> CARLA steer < 0
            c = ctl()
            c.step(0.0, 5.0, 0.0)
            c.update(drive_path(*op_arc(k, 5.0), 5.0 * A.TIMES)[1], 0.0)
            for t in (0.05, 0.10, 0.15, 0.20):
                s = c.step(t, 5.0, 0.0)[1]
            self.assertEqual(np.sign(s), sign)


class ClosedLoop(unittest.TestCase):
    def test_arcs_and_straight(self):
        v, secs = 5.0, 6.0
        for k in (0.05, -0.05, 0.0):
            s0, s1, mean_steer = simulate(k, v, secs)
            dyaw = s1[2] - s0[2]
            k_meas = -dyaw / (v * (secs - 0.05))                          # left-positive (CARLA yaw is clockwise)
            end = world_to_local(np.array([s1[:2]]), np.array(s0[:2]), s0[2])[0]
            if k == 0.0:
                self.assertLess(abs(dyaw), 1e-3)
                self.assertLess(abs(end[1]), 0.05)
            else:
                self.assertGreater(np.sign(k) * end[1], 2.0)                # ends on the plan's side
                self.assertEqual(np.sign(mean_steer), -np.sign(k))         # left arc: CARLA steer < 0
                self.assertAlmostEqual(k_meas / k, 1.0, delta=0.3)         # tracks the plan curvature (+24%: P7 rear_slip adds PhysX tyre slip this plant lacks)


class StandstillPlan(unittest.TestCase):
    def test_short_plan_extrapolates_last_chord(self):
        """A standstill plan (camera ~fixed, yaw drifting right by 0.03 rad) maps to a rear axle swinging 5 cm LEFT (rotation about
        the camera); place() extends that 5 cm chord straight to the governor's arc: the executed path points left across the road
        while the plan's heading is right. Not a sign error, the standstill plan carries no direction (results/p7_sign_check.md)."""
        pos = np.zeros((len(T_OP), 3))
        pos[:, 0] = 0.002 * T_OP
        yaw = 0.03 * T_OP / T_OP[-1]
        op_path, path = drive_path(pos, yaw, 1.0 * A.TIMES)               # governor: 1 m/s
        self.assertLess(A.arc(np.r_[[[0.0, 0.0]], op_path])[-1], 0.1)
        aim = A.place(np.r_[[[0.0, 0.0]], op_path], np.array([3.0]))[0]
        self.assertGreater(aim[1], 2.0)                                   # 3 m aim almost purely to the left
