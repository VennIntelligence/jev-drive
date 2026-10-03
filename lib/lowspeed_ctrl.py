"""Low-speed lateral transfer limit (controller side only; the model's inputs and plans are untouched).

Rule: below v1 the commanded steering angle follows the controller's own target through a first-order low-pass whose time
constant is tau0 for v <= v0 and falls linearly to 0 at v1 (alpha = dt / (tau + dt) per step), then the commanded curvature
rate is limited to openpilot's own bound MAX_LATERAL_JERK / max(v, MIN_SPEED)^2 (selfdrive/controls/lib/drive_helpers.py
clip_curvature; jerk = 5.0 m/s^3, MIN_SPEED = 1 m/s). The jerk bound is the principled default; at launch speeds it is far
above anything a car can steer, so the low-pass (tau0, v0, v1) is the tuned part and is reported as such.

Rule parameters (dict, JSON-serialisable): {"jerk": 5.0, "v_floor": 1.0, "tau0": 3.0, "v0": 1.0, "v1": 3.0}; tau0 = 0 turns the
low-pass off. Selective variant (second arm, experiments/hugsim/plans/2026-10-04-lowspeed-ctrl-selective-prereg.md): the rule parameters d0, d1 (degrees)
define a gain s(|a|) = clip((|a| - d0) / (d1 - d0), 0, 1) on the plan's 1 s direction a (deadband below d0, ramp to d1, pass-through beyond), applied
below v1 with the same speed taper; plan_select rotates the plan accordingly (HUGSIM tree `lowsel`, env LOWSPEED_SEL). Off when d1 = 0.
Hooks: hugsim_rate (the HUGSIM tree's traj2control output), B2DFilter (scripts/b2d_zeroshot_agent.py, env LOWSPEED_CTRL).
"""
import math

import numpy as np

DEFAULT = dict(jerk=5.0, v_floor=1.0, tau0=0.0, v0=1.0, v1=3.0, d0=0.0, d1=0.0)


def full(rule):
    return dict(DEFAULT, **(rule or {}))


def alpha(rule, v, dt):
    r = full(rule)
    if r["tau0"] <= 0 or v >= r["v1"]:
        return 1.0
    f = 1.0 if v <= r["v0"] else (r["v1"] - v) / (r["v1"] - r["v0"])
    return dt / (r["tau0"] * f + dt)


def taper(rule, v):
    r = full(rule)
    return 1.0 if v <= r["v0"] else max(0.0, (r["v1"] - v) / (r["v1"] - r["v0"]))


def sel_gain(rule, a_deg):
    """Selective rule: gain on a plan direction of a_deg degrees (0 = deadband, 1 = pass-through); 1 when the rule is off."""
    r = full(rule)
    return 1.0 if r["d1"] <= 0 else float(np.clip((abs(a_deg) - r["d0"]) / (r["d1"] - r["d0"]), 0.0, 1.0))


def plan_select(rule, v, plan, idx=1, fwd=1, lat=0):
    """Rotate a plan about the car so its direction at point `idx` (the 1 s point: index 1 for HUGSIM's 0.5 s spacing, 3 for B2D's 0.25 s) becomes
    a * (1 - w + w * s(|a|)), w = speed taper. Columns: fwd / lat (HUGSIM (x right, y forward): lat=0, fwd=1; B2D rig (x forward, y lateral): fwd=0,
    lat=1); the rotation turns the plan toward the forward axis, so the lateral sign convention does not matter. Short 1 s points are left alone."""
    r = full(rule)
    plan = np.asarray(plan, float)
    w = taper(rule, v)
    if r["d1"] <= 0 or w <= 0 or len(plan) <= idx or np.linalg.norm(plan[idx]) < 0.3:
        return plan
    a = math.atan2(plan[idx, lat], plan[idx, fwd])
    d = a * (sel_gain(rule, math.degrees(a)) - 1.0) * w          # rotation to apply (rad)
    c, s = math.cos(d), math.sin(d)
    out = plan.copy()
    out[:, lat] = plan[:, lat] * c + plan[:, fwd] * s
    out[:, fwd] = -plan[:, lat] * s + plan[:, fwd] * c
    return out


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
