"""Feasible reference for AlpaSim's MPC (lane SPIN-FIX, experiments/steer_long/plans/2026-10-13-spin-fix-prereg.md). Serving side, no learned
component, no simulator ground truth; the model's plan is the target, the trajectory handed to the controller is a reference the car can
follow from its current state without a large steering action.

Why (research/next-round/steer-long.md section 4.2, experiments/steer_long/results/spin_fix.md): AlpaSim's nonlinear MPC tracks positions only
(1.0 .. 2.0 s ahead; weights long 2 / lat 1 / heading 1, acceleration 0.1, steering-change 5 rad^-2, no steering-magnitude, steering-rate or
lateral-acceleration term, a linear-tyre vehicle with no saturation). When the reference asks for more deceleration than braking can give
(the force-GT handoff at 1.7 s: the plan 2-10 m behind the car at 2 s; an abrupt plan stop at speed mid-rollout) the cheapest way to lose the
distance is to steer, and at speed any steering is a spin. The fix is to never hand it such a reference.

  JEV_TRACK=time   (a) re-timing only: the plan's own path (polyline through its 8 points, straight beyond its end), driven from the ego's
                   current speed with the plan's speed as target and the change of speed bounded by [AMIN, AMAX] m/s^2. The path is unchanged.
  JEV_TRACK=full   (a) + (b): the reference is the forward simulation (kinematic, DT_SIM steps) of a path-tracking law on the same path and
                   speed: curvature = path curvature (feedforward) + a second-order correction of the lateral / heading error over the
                   distance max(TAU_LAT x v, D_MIN), clipped to |kappa| <= min(KMAX, ALAT / v^2) and its rate to
                   |dkappa/dt| <= min(JLAT / v^2, KRATE); it starts at the ego's current curvature (yaw rate / speed). The MPC then sees a
                   reference whose steering is bounded by these physical limits at every speed (at 30 m/s and ALAT 4: 0.0127 rad).
  JEV_TRACK_CFG    overrides, e.g. "alat=4,amin=-5,jlat=5" (names: the module constants in lower case).
Unset / 0: off, the driver returns the plan as before, bit for bit.

The returned reference has 41 points at 0.1 s (0 .. 4 s, rig frame at the plan's time); the drivers give it to the controller through the
sample submission's CachedPlan / build_trajectory_from_plan unchanged (only the sample times differ from the 8-point plan).
"""
from __future__ import annotations

import os

import numpy as np

MODE = os.environ.get("JEV_TRACK", "0") or "0"
ON = MODE in ("time", "full")
if MODE not in ("0", "time", "full"):
    raise ValueError(f"JEV_TRACK={MODE!r}: expected time | full | 0")

AMIN, AMAX = -5.0, 3.0          # m/s^2, change of speed of the reference
ALAT, JLAT = 4.0, 5.0           # m/s^2, m/s^3 (openpilot clip_curvature: 3.0 no-roll, jerk 5.0)
KMAX, KRATE = 0.2, 0.1          # 1/m (openpilot's cap; minimum radius 5 m), 1/m/s (about 0.29 rad/s of road-wheel steering, wheelbase 2.85)
TAU_LAT, D_MIN, ZETA = 1.0, 4.0, 0.9
V_MIN = 1.0                     # m/s, floor of v in the curvature limits
DT_SIM, T_END, DT_OUT = 0.05, 4.0, 0.1
for _kv in filter(None, (os.environ.get("JEV_TRACK_CFG", "") or "").split(",")):
    _k, _v = _kv.split("=")
    if _k.upper() not in globals() or _k.upper() in ("MODE", "ON"):
        raise ValueError(f"JEV_TRACK_CFG: unknown key {_k}")
    globals()[_k.upper()] = float(_v)
SUFFIX = f"-track{MODE}" if ON else ""


class Path:
    """The plan's path: polyline origin + 8 points, extended 200 m straight along its last chord; arc-length lookups."""

    def __init__(self, poses):
        p = np.concatenate([np.zeros((1, 2)), np.asarray(poses, np.float64)[:, :2]])
        d = np.diff(p, axis=0)
        ds = np.hypot(*d.T)
        keep = np.r_[True, ds > 0.05]                                     # drop repeated points (stop plans)
        p = p[keep]
        d = np.diff(p, axis=0)
        ds = np.hypot(*d.T)
        if len(ds) == 0:                                                  # a plan that does not move: straight ahead
            p, d, ds = np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([1.0])
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.s = np.r_[0.0, np.cumsum(ds)]
        far = p[-1] + 200.0 * d[-1] / ds[-1]
        self.p, self.s = np.vstack([p, far]), np.r_[self.s, self.s[-1] + 200.0]
        mid = (self.s[:-2] + self.s[1:-1]) / 2                           # heading continuous in s: chord headings at segment midpoints
        self.hs, self.hv = np.r_[0.0, mid, self.s[-1]], np.r_[h[0], h, h[-1]]

    def at(self, s):
        """-> x, y on the polyline, smooth heading, curvature (d heading / ds) at arc length s."""
        x, y = np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])
        h = np.interp(s, self.hs, self.hv)
        i = np.clip(np.searchsorted(self.hs, s, side="right") - 1, 0, len(self.hs) - 2)
        k = (self.hv[i + 1] - self.hv[i]) / max(self.hs[i + 1] - self.hs[i], 1e-6)
        return x, y, h, k


def speeds(poses):
    """Plan speed on its own clock: segment speeds of the 8 points at 0.5 s."""
    p = np.concatenate([np.zeros((1, 2)), np.asarray(poses, np.float64)[:, :2]])
    return np.hypot(*np.diff(p, axis=0).T) / 0.5


def build(poses, v0: float, wz: float, mode: str = "", **kw):
    """Plan (8, 3) rig frame, ego speed and yaw rate -> (t (41,), x, y, yaw (41,)) and a log record. mode / kw (lower-case constant names)
    override the environment's configuration (offline replays)."""
    mode = mode or MODE
    g = {k: kw.get(k.lower(), globals()[k]) for k in ("AMIN", "AMAX", "ALAT", "JLAT", "KMAX", "KRATE", "TAU_LAT", "D_MIN", "ZETA", "V_MIN")}
    AMIN, AMAX, ALAT, JLAT, KMAX, KRATE, TAU_LAT, D_MIN, ZETA, V_MIN = g.values()
    path, vp = Path(poses), speeds(poses)
    n, every = int(round(T_END / DT_SIM)), int(round(DT_OUT / DT_SIM))
    v, s, x, y, th = max(float(v0), 0.0), 0.0, 0.0, 0.0, 0.0
    k = float(np.clip(wz / max(v, V_MIN), -KMAX, KMAX)) if v > 0.5 else 0.0
    out, kpk, apk = [(0.0, 0.0, 0.0)], abs(k), 0.0
    for i in range(n):
        t = i * DT_SIM
        a = float(np.clip((vp[min(int(t / 0.5), 7)] - v) / DT_SIM, AMIN, AMAX))
        apk = min(apk, a)
        if mode == "time":
            v = max(v + a * DT_SIM, 0.0)
            s += v * DT_SIM
            x, y, th, _ = path.at(s)
        else:
            px, py, ph, pk = path.at(s)
            c, sn = np.cos(ph), np.sin(ph)
            ey = -sn * (x - px) + c * (y - py)
            eh = (th - ph + np.pi) % (2 * np.pi) - np.pi
            lam = 1.0 / max(TAU_LAT * v, D_MIN)
            kd = pk - lam * lam * ey - 2 * ZETA * lam * eh
            vv = max(v, V_MIN)
            km = min(KMAX, ALAT / vv ** 2)
            dk = min(JLAT / vv ** 2, KRATE) * DT_SIM
            k = float(np.clip(np.clip(kd, k - dk, k + dk), -km, km))
            kpk = max(kpk, abs(k))
            v = max(v + a * DT_SIM, 0.0)
            ds = v * DT_SIM
            s += ds
            th += k * ds
            x += ds * np.cos(th - k * ds / 2)
            y += ds * np.sin(th - k * ds / 2)
        if (i + 1) % every == 0:
            out.append((x, y, th))
    o = np.array(out)
    t = np.arange(len(o)) * DT_OUT
    return (t, o[:, 0], o[:, 1], o[:, 2]), {"mode": mode, "v0": round(float(v0), 3), "k0": round(float(k), 4), "kpk": round(kpk, 4),
                                           "amin": round(apk, 2), "s4": round(s, 2)}


def cached_plan(CachedPlan, t0_us: int, anchor_xy_yaw, ref):
    """The reference (rig frame at t0) -> the sample submission's CachedPlan in the local frame (anchor = ego pose at t0)."""
    t, x, y, h = ref
    ax, ay, a0 = anchor_xy_yaw
    c, s = np.cos(a0), np.sin(a0)
    return CachedPlan(int(t0_us), np.asarray(t, np.float64), np.stack([ax + c * x - s * y, ay + s * x + c * y], 1), a0 + np.asarray(h))


def poses8(ref):
    """The reference at 0.5 .. 4 s as (8, 3), for the drive.jsonl record."""
    t, x, y, h = ref
    i = np.arange(5, 41, 5)
    return np.stack([x[i], y[i], h[i]], 1)
