"""Low-speed lateral transfer limit (controller side only; the model's inputs and plans are untouched).

Rule: below v1 the commanded steering angle follows the controller's own target through a first-order low-pass whose time
constant is tau0 for v <= v0 and falls linearly to 0 at v1 (alpha = dt / (tau + dt) per step), then the commanded curvature
rate is limited to openpilot's own bound MAX_LATERAL_JERK / max(v, MIN_SPEED)^2 (selfdrive/controls/lib/drive_helpers.py
clip_curvature; jerk = 5.0 m/s^3, MIN_SPEED = 1 m/s). The jerk bound is the principled default; at launch speeds it is far
above anything a car can steer, so the low-pass (tau0, v0, v1) is the tuned part and is reported as such.

Rule parameters (dict, JSON-serialisable): {"jerk": 5.0, "v_floor": 1.0, "tau0": 3.0, "v0": 1.0, "v1": 3.0}; tau0 = 0 turns the
low-pass off. Hooks: hugsim_rate (the HUGSIM tree's traj2control output), B2DFilter (scripts/b2d_zeroshot_agent.py, env LOWSPEED_CTRL).
"""
import math

import numpy as np

DEFAULT = dict(jerk=5.0, v_floor=1.0, tau0=0.0, v0=1.0, v1=3.0)


def full(rule):
    return dict(DEFAULT, **(rule or {}))


def alpha(rule, v, dt):
    r = full(rule)
    if r["tau0"] <= 0 or v >= r["v1"]:
        return 1.0
    f = 1.0 if v <= r["v0"] else (r["v1"] - v) / (r["v1"] - r["v0"])
    return dt / (r["tau0"] * f + dt)


def clip_kappa(rule, v, k_prev, k_new, dt):
    """openpilot clip_curvature: |dkappa| <= MAX_LATERAL_JERK / v^2 per second."""
    r = full(rule)
    if r["jerk"] <= 0:
        return k_new
    m = r["jerk"] / max(v, r["v_floor"]) ** 2 * dt
    return float(np.clip(k_new, k_prev - m, k_prev + m))


def hugsim_rate(rule, v, steer, sr, dt, wheelbase=2.7):
    """Steering rate to apply: the iLQR rate sr scaled by alpha (low-pass on the commanded steering angle), then the curvature-rate clip."""
    tgt = steer + alpha(rule, v, dt) * sr * dt
    k_new = clip_kappa(rule, v, math.tan(steer) / wheelbase, math.tan(tgt) / wheelbase, dt)
    return (math.atan(k_new * wheelbase) - steer) / dt


class B2DFilter:
    """Stateful filter on the normalised CARLA steer command (positive right) of scripts/b2d_controller.Controller.step."""
    WB, MAX_STEER = 2.8604714913890885, math.radians(69.99999237060547)
    CURVE = np.array([[0.0, 1.0], [20.0, 0.9], [60.0, 0.8], [120.0, 0.7]])

    def __init__(self, rule):
        self.rule, self.prev = rule, 0.0

    def __call__(self, st, v, dt):
        a = alpha(self.rule, v, dt)
        out = self.prev + a * (st - self.prev)
        sc = float(np.interp(v * 3.6, self.CURVE[:, 0], self.CURVE[:, 1]))
        k = lambda s: math.tan(-s * self.MAX_STEER * sc) / self.WB  # noqa: E731
        k_new = clip_kappa(self.rule, v, k(self.prev), k(out), dt)
        out = -math.atan(k_new * self.WB) / (self.MAX_STEER * sc)
        self.prev = out
        return out
