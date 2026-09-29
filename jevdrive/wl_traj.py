"""WL action candidates (todos/2026-09-28-wm-loop.md, "动作候选"). NumPy only: imported by the CARLA fork agent
(scripts/wl_fork_agent.py, Python 3.10 envs) and by the offline pipeline (jevdrive/wl.py) alike.

Frames. World = CARLA (x east, y south, yaw in degrees, clockwise positive seen from above). Ego = rear axle,
x forward, y left (scripts/b2d_controller.py's contract). A candidate is a time-stamped ego-frame path at
T = 0.25, 0.50, ... HORIZON s after the fork tick; the agent freezes it in world coordinates at the fork and every
5 Hz hands P7 the part still ahead, re-expressed in the current ego frame.

  op          openpilot Cinque's native plan at the fork (20 points to 5 s, stored stream; extended at constant
              speed and heading beyond 5 s)
  op_slow     op's path, every speed x 0.5
  op_stop     op's path, v(t) = max(0, v0 - 4 t)
  hold        route centre line, v(t) = v0
  brake_hard  route centre line, v(t) = max(0, v0 - 6 t)
  shift_L/R   op's path and speed, plus a lateral offset of +-3.0 m (left positive) with a 2 s cosine transition
WL-2 adds four (todos/2026-09-29-wl2-prereg.md, "候选"):
  brake_mild    op's path, v(t) = max(0, v0 - 2 t)
  shift_L/R_slow  the shift geometry (3.0 m, 2 s cosine) on op's path at half of op's speed (= op_slow's path)
  nudge_L       op's path and speed, plus a left offset of 1.5 m with a 1.5 s cosine transition
"""
import math

import numpy as np

DT, HORIZON = 0.25, 6.0
T = DT * np.arange(1, int(round(HORIZON / DT)) + 1)
ACTIONS = ("op", "op_slow", "op_stop", "hold", "brake_hard", "shift_L", "shift_R")          # WL-1 (and D2's random windows)
ACTIONS11 = ACTIONS + ("brake_mild", "shift_L_slow", "shift_R_slow", "nudge_L")                # WL-2 fork branches
OP_T = 0.25 * np.arange(1, 21)
SHIFT_M, SHIFT_S, STOP_DECEL, BRAKE_DECEL = 3.0, 2.0, 4.0, 6.0
MILD_DECEL, NUDGE_M, NUDGE_S = 2.0, 1.5, 1.5


def world_to_ego(pts, xy, yaw_deg):
    """World points (n, 2) -> ego (x forward, y left); xy = rear-axle world position, yaw_deg = CARLA yaw."""
    d = np.asarray(pts, float) - np.asarray(xy, float)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.column_stack((d[:, 0] * c + d[:, 1] * s, d[:, 0] * s - d[:, 1] * c))


def ego_to_world(pts, xy, yaw_deg):
    p = np.asarray(pts, float)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.column_stack((xy[0] + p[:, 0] * c + p[:, 1] * s, xy[1] + p[:, 0] * s - p[:, 1] * c))


def _arc(path):
    """Cumulative arc length of a polyline that starts at the origin."""
    p = np.vstack([[0.0, 0.0], path])
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))], p


def _along(path, s_query, ext=200.0):
    """Points at arc lengths s_query along a polyline from the origin, extended straight past its end."""
    s, p = _arc(path)
    if s[-1] < 1e-3:                                   # a standing plan: extend along +x
        p, s = np.array([[0.0, 0.0], [ext, 0.0]]), np.array([0.0, ext])
    else:
        d = p[-1] - p[-2] if s[-1] - s[-2] > 1e-6 else p[-1] - p[0]
        d = d / max(np.linalg.norm(d), 1e-9)
        p, s = np.vstack([p, p[-1] + ext * d]), np.r_[s, s[-1] + ext]
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    return np.column_stack((np.interp(s_query, s, p[:, 0]), np.interp(s_query, s, p[:, 1])))


def _speed_profile(v0, decel, t):
    """Arc length at t of v(t) = max(0, v0 - decel t) (decel 0: constant)."""
    if decel <= 0:
        return v0 * t
    ts = v0 / decel
    return np.where(t < ts, v0 * t - 0.5 * decel * t ** 2, 0.5 * v0 * ts)


def op_extended(op_plan, t=T):
    """op's (20, 2) plan to 5 s -> positions at t, the last 1 s's speed and heading carried on to HORIZON."""
    op = np.asarray(op_plan, float).reshape(20, 2)
    s_op, _ = _arc(op)
    s_t = np.interp(t, np.r_[0.0, OP_T], s_op)
    v_end = max(0.0, (s_op[-1] - s_op[-5]) / 1.0)
    s_t = np.where(t > OP_T[-1], s_op[-1] + v_end * (t - OP_T[-1]), s_t)
    return _along(op, s_t), s_t


def _offset(path, sgn, amp, dur, t):
    """`path` (points at times t) moved sideways by sgn * amp (left positive) with a `dur` s cosine transition, along the
    normal of the path's own heading at each point."""
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(t, dur) / dur))
    d = np.gradient(np.vstack([[0.0, 0.0], path]), axis=0)[1:]
    n = np.column_stack((-d[:, 1], d[:, 0])) / np.maximum(np.hypot(*d.T), 1e-6)[:, None]
    n[np.hypot(*d.T) < 1e-6] = (0.0, 1.0)
    return path + sgn * amp * ramp[:, None] * n


def candidates(op_plan, route_ego, v0, t=T):
    """{action: (len(t), 2) ego points at times t}. op_plan: (20, 2) ego plan (None -> the route centre line at v0);
    route_ego: the dense route ahead in ego coordinates (n, 2), starting near the ego; v0: speed (m/s)."""
    T = np.asarray(t, float)
    out = {}
    route = np.asarray(route_ego, float)
    route = route[np.argmin(np.hypot(*route.T)):]                  # from the closest route point on,
    route = route[route[:, 0] > 0.1] if (route[:, 0] > 0.1).sum() >= 2 else route   # the part ahead of the rear axle
    out["hold"] = _along(route, _speed_profile(v0, 0.0, T))
    out["brake_hard"] = _along(route, _speed_profile(v0, BRAKE_DECEL, T))
    if op_plan is None:                                # no openpilot plan (random windows): the route at the current speed
        op_plan = _along(route, v0 * OP_T)
    if op_plan is not None:
        op, s_op = op_extended(op_plan, T)
        out["op"] = op
        out["op_slow"] = _along(np.asarray(op_plan, float).reshape(20, 2), 0.5 * s_op)
        out["op_stop"] = _along(np.asarray(op_plan, float).reshape(20, 2), _speed_profile(v0, STOP_DECEL, T))
        op20 = np.asarray(op_plan, float).reshape(20, 2)
        for side, sgn in (("shift_L", 1.0), ("shift_R", -1.0)):      # offset normal to op's own heading at each point
            out[side] = _offset(op, sgn, SHIFT_M, SHIFT_S, T)
            out[side + "_slow"] = _offset(out["op_slow"], sgn, SHIFT_M, SHIFT_S, T)
        out["brake_mild"] = _along(op20, _speed_profile(v0, MILD_DECEL, T))
        out["nudge_L"] = _offset(op, 1.0, NUDGE_M, NUDGE_S, T)
    return out


FINE = 0.05 * np.arange(1, 121)


def command_actions(op_plan, route_ego, v0, dt=0.2, n=10):
    """{action: (n, 2)} commanded (a, omega) of every candidate on the world model's 0.2 s grid: differences of the
    candidate's instantaneous speed and heading (from a 0.05 s resampling) at 0, 0.2, ... s, in nq4_w's convention for
    logged actions (a = (v_{t+1} - v_t) / 0.2 s, omega = CARLA yaw rate, right turn positive), clipped to +-15."""
    out = {}
    for k, traj in candidates(op_plan, route_ego, v0, FINE).items():
        p = np.vstack([[0.0, 0.0], traj])
        d = np.diff(p, axis=0)
        spd = np.hypot(*d.T) / 0.05
        head = np.arctan2(d[:, 1], d[:, 0])
        for i in range(len(head)):                      # standing still keeps the last heading
            if spd[i] < 0.05:
                head[i] = head[i - 1] if i else 0.0
        head = np.unwrap(head)
        mid = FINE - 0.025
        q = np.arange(n + 1) * dt
        v = np.interp(q, np.r_[0.0, mid], np.r_[v0, spd])
        h = np.interp(q, np.r_[0.0, mid], np.r_[0.0, head])
        out[k] = np.clip(np.stack([np.diff(v) / dt, -np.diff(h) / dt], -1), -15, 15).astype(np.float32)
    return out
