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
