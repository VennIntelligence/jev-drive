"""ROUND6 line B (perturbation-recovery rows, plans/2026-10-21-recovery-rows.md): the lateral-offset geometry shared by the rollout driver
(lib/rec_driver.py, which steers the ego along the perturbed path) and the offline row builder (scripts/rec.py, which writes the recovery label).

An offset E (m, + = left of the log's heading) is a function of the log's arc length s. One episode at decision time t0 (rollout clock, us)
with target offset e0 (m) and heading offset psi0 (tan psi0 = dE/ds at t0):
  ramp-in   s in [S(t0 - T_IN), S(t0)]: cubic Hermite from (0, 0) to (e0, tan psi0)       (the ego leaves the log over the past T_IN s)
  recovery  s in [S(t0), S(t0 + T_OUT)]: cubic Hermite from (e0, tan psi0) to (0, 0)    (the driven return to the log)
A label at decision time t (any row, also inside the ramp-in): the cubic Hermite from the driven state (E(t), E'(t)) to (0, 0) over
[S(t), S(t + T_MERGE)], i.e. a merge back onto the human's path within T_MERGE s; label pose j sits at the log's pose of time t + 0.5 (j + 1) s
displaced by E_lab along the log's left normal, yaw + atan(E_lab'), so the arc length per label time is the human's (speed = the human's).
S = cumulative chord length of the log samples (time -> arc by linear interpolation): arc-length parameterised offsets make the heading offset
atan(dE/ds) independent of speed. Zero e0 and psi0 give E = 0 exactly (x + 0.0 = x), so a zero episode reproduces the log path bit for bit.
"""
from __future__ import annotations

import numpy as np

T_IN, T_OUT, T_MERGE = 2_000_000, 2_000_000, 2_000_000         # us: ramp-in, driven recovery, label merge


def arc(t_us, xy):
    """Log samples (n,), (n, 2) -> (t (n,) int64, S (n,) cumulative chord length)."""
    xy = np.asarray(xy, np.float64)[:, :2]
    return np.asarray(t_us, np.int64), np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]


def herm(u, p0, m0, p1, m1, L):
    """Cubic Hermite on u in [0, 1] from value p0 / slope m0 (per metre) to p1 / m1 over arc length L -> (E, dE/ds, d2E/ds2)."""
    u = np.clip(u, 0.0, 1.0)
    u2, u3 = u * u, u * u * u
    a, b = m0 * L, m1 * L                                       # slopes per unit u
    E = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * a + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * b
    d1 = ((6 * u2 - 6 * u) * p0 + (3 * u2 - 4 * u + 1) * a + (-6 * u2 + 6 * u) * p1 + (3 * u2 - 2 * u) * b) / L
    d2 = ((12 * u - 6) * p0 + (6 * u - 4) * a + (-12 * u + 6) * p1 + (6 * u - 2) * b) / (L * L)
    return E, d1, d2


def driven(ts, t, S, eps):
    """The driven offset at times ts (m,) us: eps = [(t0_us, e0, tan psi0), ...] (disjoint windows) -> E, E', E'' (m,) each; 0 outside."""
    ts = np.asarray(ts, np.float64)
    s = np.interp(ts, t, S)
    E, D1, D2 = np.zeros(len(ts)), np.zeros(len(ts)), np.zeros(len(ts))
    for t0, e0, tp in eps:
        sa, s0, sb = np.interp([t0 - T_IN, t0, t0 + T_OUT], t, S)
        m = (ts >= t0 - T_IN) & (ts <= t0)
        if m.any() and s0 > sa:
            E[m], D1[m], D2[m] = herm((s[m] - sa) / (s0 - sa), 0.0, 0.0, e0, tp, s0 - sa)
        m = (ts > t0) & (ts <= t0 + T_OUT)
        if m.any() and sb > s0:
            E[m], D1[m], D2[m] = herm((s[m] - s0) / (sb - s0), e0, tp, 0.0, 0.0, sb - s0)
    return E, D1, D2


def merge(ts, t, S, tr, Er, Dr):
    """The label offset at times ts (m,) of a row at decision time tr whose driven state is (Er, Dr = dE/ds) -> E, E', E''."""
    s = np.interp(np.asarray(ts, np.float64), t, S)
    s0, s1 = np.interp([tr, tr + T_MERGE], t, S)
    L = max(s1 - s0, 1e-6)
    E, D1, D2 = herm((s - s0) / L, Er, Dr, 0.0, 0.0, L)
    past = s > s1
    return np.where(past, 0.0, E), np.where(past, 0.0, D1), np.where(past, 0.0, D2)


def offset(p, E, D1):
    """Poses p (..., 3) x, y, yaw (any frame) displaced by E along the left normal of their yaw, yaw + atan(E')."""
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 0] - np.sin(p[..., 2]) * E, p[..., 1] + np.cos(p[..., 2]) * E, p[..., 2] + np.arctan(D1)], -1)
