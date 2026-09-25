"""Tests of the Zoo PID wrapper's adapter switches (forward_only, cadence "tick"). envs/carla, needs the Zoo checkout:
    $DATA_DIR/envs/carla/bin/python -m pytest -q scripts/test_b2d_zoo_pid_wrap.py   (no pytest in envs/carla: call the test_* functions)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2d_zoo_pid_wrap import ZooPID  # noqa: E402

TIMES = np.arange(1, 21) * 0.25
RNG = np.random.default_rng(0)


def _path(v=3.0, curv=0.02):
    s = v * TIMES
    return np.stack([np.sin(curv * s) / curv, (1 - np.cos(curv * s)) / curv], -1)


def test_default_prepare_is_identity_for_control():
    for _ in range(20):
        path = _path(RNG.uniform(0, 8), RNG.uniform(-0.1, 0.1) + 1e-6)
        a, b = ZooPID(), ZooPID()
        r1 = a.control(path, TIMES, 2.0, (20.0, 1.0))
        p, t = b.prepare(path, TIMES)
        r2 = b.control(p, t, 2.0, (20.0, 1.0))
        assert r1[:3] == r2[:3] and r1[3]["desired_speed"] == r2[3]["desired_speed"]


def test_forward_only_turns_reverse_stop_plan_into_brake():
    rev = np.stack([-0.2 * TIMES ** 2, np.zeros(20)], -1)           # braking through zero: reverses
    z = ZooPID(forward_only=True)
    p, t = z.prepare(rev, TIMES)
    assert np.allclose(p, 0.0)
    steer, throttle, brake, meta = z.control(p, t, 0.0, (20.0, 0.0))
    assert brake == 1.0 and throttle == 0.0
    steer, throttle, brake, meta = ZooPID().control(rev, TIMES, 0.0, (20.0, 0.0))
    assert throttle > 0                                               # the bug the switch removes


def test_forward_only_keeps_forward_plans():
    path = _path(4.0, 0.05)
    p, t = ZooPID(forward_only=True).prepare(path, TIMES)
    assert np.allclose(p[1:], path)


def test_held_now_age_and_pose():
    z = ZooPID(cadence="tick")
    path = _path(5.0, 1e-6)                                           # straight, 5 m/s
    yaw = 0.7                                                         # CARLA world yaw
    x0 = np.array([10.0, -3.0])
    z.hold(path, TIMES, 1.0, (x0, yaw))
    p, t = z.held_now(1.0, (x0, yaw))
    assert np.allclose(p[1:], path, atol=1e-9) and np.allclose(t[1:], TIMES)
    # 0.3 s later the car has driven 1.5 m along its heading: the plan point at 0.3 + 0.5 s is 2.5 m ahead
    x1 = x0 + 1.5 * np.array([math.cos(yaw), math.sin(yaw)])
    p, t = z.held_now(1.3, (x1, yaw))
    q = np.stack([np.interp(0.5, t, p[:, k]) for k in range(2)])
    assert np.allclose(q, [2.5, 0.0], atol=1e-4)
    # a left point of the plan (y > 0) stays on the left after the transform (CARLA y is right)
    z.hold(np.c_[np.zeros(20), np.ones(20)], TIMES, 0.0, (x0, yaw))
    p, _ = z.held_now(0.0, (x0, yaw))
    assert np.allclose(p[1:, 1], 1.0)


# Logged plan of the first f1 full run, route 1825 at t = 24.65 s (4.0 m/s, ConstructionObstacleTwoWays): the model
# plans to swerve left around the work zone (CoT "Nudge to the left to clear the construction trailer"); the shipped
# Zoo PID steered -0.006 (aim = the 0.5 s point, then the route target was straighter). Diagnosis doc, sections 7-8.
DETOUR = np.array([[1.28, 0.003], [2.745, 0.044], [4.336, 0.214], [5.981, 0.59], [7.639, 1.251], [9.284, 2.167],
                   [10.945, 3.281], [12.67, 4.47], [14.517, 5.627], [16.521, 6.683], [18.711, 7.588], [21.039, 8.337],
                   [23.454, 8.93], [25.91, 9.411], [28.387, 9.81], [30.866, 10.161], [33.334, 10.475],
                   [35.782, 10.762], [38.202, 11.023], [40.585, 11.262]])
DETOUR_V, DETOUR_TARGET = 4.012, (12.0788, 1.1935)       # speed, route target (forward, right) as logged


def test_zoo_lateral_mutes_the_detour():
    z = ZooPID(forward_only=True)
    p, t = z.prepare(DETOUR, TIMES)
    steer, throttle, brake, meta = z.control(p, t, DETOUR_V, DETOUR_TARGET)
    assert abs(steer) < 0.02                                          # as logged (-0.006)


def test_p2_time_aim_steers_left_on_the_detour():
    z = ZooPID(forward_only=True, lateral="time", aim_s=1.5)
    p, t = z.prepare(DETOUR, TIMES)
    ref = ZooPID(forward_only=True)
    s0, th0, b0, _ = ref.control(p, t, DETOUR_V, DETOUR_TARGET)
    steer, throttle, brake, meta = z.control(p, t, DETOUR_V, DETOUR_TARGET)
    assert np.allclose(meta["aim_time"], DETOUR[5])                  # the 1.5 s point, 9.3 m ahead, 2.2 m left
    assert -0.25 < steer < -0.1                                       # left (CARLA steer < 0), first PID step
    assert (throttle, brake) == (th0, b0)                             # longitudinal untouched
    # held at the next plan: the derivative kick is gone, P + I keep steering left
    assert z.control(p, t, DETOUR_V, DETOUR_TARGET)[0] < -0.1


def test_p2_time_aim_near_standstill_and_stop_plans():
    z = ZooPID(forward_only=True, lateral="time")
    creep = np.c_[0.3 * TIMES, 0.05 * TIMES]                          # 0.45 m at 1.5 s: use the first point >= 1 m
    p, t = z.prepare(creep, TIMES)
    assert np.hypot(*z.time_aim(p, t)) >= 1.0 and z.time_aim(p, t)[1] > 0
    stop = np.zeros((20, 2))
    p, t = z.prepare(stop, TIMES)
    assert z.time_aim(p, t) is None
    steer, throttle, brake, meta = z.control(p, t, 0.0, (20.0, 3.0))  # target 3 m right is NOT substituted
    assert steer == 0.0 and brake == 1.0


def test_p1_fixed_lateral_steers_left_on_the_detour():
    import json
    from b2d_controller import Controller
    here = os.path.dirname(os.path.abspath(__file__))
    params = json.load(open(os.path.join(here, "..", "todos/2026-09-22-b2d-controller/results/controller_config.json")))
    params.pop("rear_axle_offset_m")
    for key in ("adapter", "metadata", "preset"):
        params.pop(key, None)
    c = Controller(preset="carla", **params)                          # exactly as b2d_zeroshot_agent.py builds it
    c.step(10.0, DETOUR_V, 0.0)
    assert c.update(DETOUR, 10.0)
    steers = [c.step(10.0 + 0.05 * k, DETOUR_V, 0.0)[1] for k in range(1, 11)]
    assert c.diagnostics["reason"] == "tracking"
    assert steers[-1] < -0.1 and all(s <= 1e-9 for s in steers)       # left from the first tick, rate-limited


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
