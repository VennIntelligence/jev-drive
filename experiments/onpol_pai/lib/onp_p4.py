"""Turning school P4 (experiments/route_follow/plans/2026-10-14-p4-self-branch-prereg.md): the pure numpy pieces of the self-branch
instrument, shared by the serving side (onp_log_driver.py, ONP_CELLS) and the offline checks / reader (experiments/route_follow/scripts/rf_p4.py),
so the rule the driver serves and the rule the check recomputes are one piece of code.

  Route      one scene's route polyline in the rollout's local frame on a 0.5 m grid (u, x, y) with its curvature kappa (rf_p4.py plan:
             AlpaSim's MAP route generated along the logged path, heading over a +-2 m chord, curvature smoothed over 2 m), the strict turn's
             entry / exit stations (rf_pai_follow's rule: 10 % / 90 % of the >= 60 deg heading change within 30 m) and the release
             station (exit + 10 m). station(): monotone windowed projection of the ego, held while the ego is more than FAR m off the route.
  profile    the curvature-limited speed profile along the route from the ego's station and measured speed: inside [entry, release) the
             limit is min(v_target, sqrt(A_LAT / |kappa|)); backward pass braking <= DECEL ahead of every limit; before the entry the
             envelope never drops below v_floor (no crawl); forward pass from the measured speed, accelerating <= ACC and decelerating
             <= DECEL. Returns the arc the profile covers in T_REF s and the planned speed at the entry.
  scale      s_k = clip(profile arc / plan arc over T_REF, S_MIN, 1): the per-decision arc-length scale of the served plan (the plan's own
             path re-sampled, onp_log_driver.branch_poses), never above the plan's own speed.
"""
from __future__ import annotations

import numpy as np

A_LAT, DECEL, ACC, T_REF, S_MIN, S_MAX, FAR, DU, LAT_N = 3.3, 2.5, 1.5, 2.0, 0.4, 1.0, 10.0, 0.25, 30


class Route:
    def __init__(self, d: dict):
        self.u = np.asarray(d["u"], np.float64)
        self.xy = np.c_[np.asarray(d["x"], np.float64), np.asarray(d["y"], np.float64)]
        self.kap = np.asarray(d["kappa"], np.float64)
        self.s_entry, self.s_exit, self.s_release = float(d["s_entry"]), float(d["s_exit"]), float(d["s_release"])
        self.v_e = float(d["v_e"])

    def station(self, p, hint=None) -> tuple:
        """Ego position p (2,) -> (station, distance to the route). Window [hint - 5, hint + 30] m (whole route without a hint); the
        station never decreases and is held while the ego is more than FAR m off the route."""
        u, a, b = self.u, self.xy[:-1], self.xy[1:]
        m = np.ones(len(a), bool) if hint is None else (u[1:] >= hint - 5.0) & (u[:-1] <= hint + 30.0)
        if not m.any():
            m[:] = True
        a, b, u0 = a[m], b[m], u[:-1][m]
        ab = b - a
        l2 = np.maximum((ab ** 2).sum(1), 1e-12)
        t = np.clip(((np.asarray(p, np.float64) - a) * ab).sum(1) / l2, 0.0, 1.0)
        d = np.hypot(*(np.asarray(p, np.float64) - (a + t[:, None] * ab)).T)
        j = int(d.argmin())
        s = float(u0[j] + t[j] * np.sqrt(l2[j]))
        if hint is None:
            return s, float(d[j])
        return (max(hint, s) if d[j] <= FAR else hint), float(d[j])


def profile(R: Route, sigma: float, v0: float, v_target: float, v_floor: float) -> dict:
    """Speed profile from station sigma at measured speed v0 (m/s) -> {arc: m covered in T_REF s, v_ent: planned speed at the entry
    (None once past it), v_min: lowest planned speed over the grid}."""
    L = 3.0 * max(v0, v_target) + 30.0
    u = sigma + DU * np.arange(int(L / DU) + 1)
    k = np.interp(u, R.u, R.kap)
    inside = (u >= R.s_entry) & (u < R.s_release)
    lim = np.where(inside, np.minimum(v_target, np.sqrt(A_LAT / np.maximum(np.abs(k), 1e-4))), np.inf)
    env = lim.copy()
    for i in range(len(u) - 2, -1, -1):
        env[i] = min(env[i], np.sqrt(env[i + 1] ** 2 + 2.0 * DECEL * DU))
    env = np.where(u < R.s_entry, np.maximum(env, v_floor), env)
    v = np.empty_like(u)
    v[0] = v0
    for i in range(len(u) - 1):
        v[i + 1] = max(min(env[i + 1], np.sqrt(v[i] ** 2 + 2.0 * ACC * DU)), np.sqrt(max(v[i] ** 2 - 2.0 * DECEL * DU, 0.0)))
    t = np.r_[0.0, np.cumsum(DU / np.maximum(0.5 * (v[1:] + v[:-1]), 0.05))]
    return dict(arc=float(np.interp(T_REF, t, u - sigma)), v_ent=float(np.interp(R.s_entry, u, v)) if u[0] <= R.s_entry else None,
                v_min=float(v.min()))


def plan_arc(poses, t_ref: float = T_REF) -> float:
    """Arc length of an (8, 3) plan (poses at 0.5 .. 4 s, ego at the origin) through t_ref s."""
    q = np.asarray(poses, np.float64)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], q[:, :2]], axis=0).T))]
    return float(np.interp(t_ref, 0.5 * np.arange(9), S))


def scale(arc_profile: float, arc_plan: float) -> float:
    return 1.0 if arc_plan < 0.1 else float(np.clip(arc_profile / arc_plan, S_MIN, S_MAX))
