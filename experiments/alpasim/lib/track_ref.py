"""Feasible reference for AlpaSim's MPC (lane SPIN-FIX, experiments/steer_long/plans/2026-10-13-spin-fix-prereg.md). Serving side, no learned
component, no simulator ground truth; the model's plan is the target, the trajectory handed to the controller is a reference the car can
follow from its current state without a large steering action.

Why (research/next-round/steer-long.md section 4.2, experiments/steer_long/results/spin_fix.md): AlpaSim's nonlinear MPC tracks positions only
(1.0 .. 2.0 s ahead; weights long 2 / lat 1 / heading 1, acceleration 0.1, steering-change 5 rad^-2, no steering-magnitude, steering-rate or
lateral-acceleration term, a linear-tyre vehicle with no saturation). When the reference asks for more deceleration than braking can give
(the force-GT handoff at 1.7 s: the plan 2-10 m behind the car at 2 s; an abrupt plan stop at speed mid-rollout) the cheapest way to lose the
distance is to steer, and at speed any steering is a spin. The fix is to never hand it such a reference.

  JEV_TRACK=time   (a) re-timing only: the plan's own path (polyline through its 8 points, straight beyond its end), driven from the ego's
                   current speed and acceleration with the plan's speed as target, the change of speed bounded by [AMIN, AMAX] m/s^2 and
                   its jerk by JLON. The path is unchanged.
  JEV_TRACK=full   (a) + (b): the reference is the forward simulation (kinematic, DT_SIM steps) of a path-tracking law on the same path and
                   speed: curvature = path curvature (feedforward) + a second-order correction of the lateral / heading error over the
                   distance max(TAU_LAT x v, D_MIN), clipped to |kappa| <= min(ALAT / v^2, KMAX below 3.5 m/s .. KDYN above 5 m/s) with
                   v = max(v_ref, v0 - V_DROP t), and its rate to
                   |dkappa/dt| <= min(JLAT / v^2, KRATE), and (VCURV) the target speed also capped by the path's curvature ahead
                   (v^2 |kappa| <= ALAT reachable with braking at AMIN), so a plan that turns while the car is still faster than the plan
                   slows for the turn instead of widening it; it starts at the clipped desired curvature (starting from the ego's measured curvature instead
                   made the receding-horizon reference lag every turn: run 1 of results/spin_fix.md). The MPC then sees a
                   reference whose steering is bounded by these physical limits at every speed (at 30 m/s and ALAT 7: 0.022 rad; a spin command is 0.2-1.1 rad).
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
JLON = 10.0                     # m/s^3, its jerk, starting from the ego's measured acceleration: a plan whose speed flickers between
                                 # calls does not hand the MPC a new deceleration step every 0.1 s (run 5: a straight, feasible reference
                                 # still drew a 0.69 rad command from such a step)
ALAT, JLAT = 4.0, 10.0          # m/s^2, m/s^3 (openpilot clip_curvature: 3.0, jerk 5.0); ALAT 7 left 16 / 38 rollouts above 6 m/s^2 (run 3)
YAW_TOL = 0.35                  # rad, plan yaw output against its chords
LATCAP = 1.0                    # full: 0 = no curvature / curvature-rate clipping (ablation; VCURV still uses ALAT / KDYN)
VCURV = 1.0                     # full: 1 = also slow for the path's curvature ahead (v^2 |kappa| <= ALAT, braking at AMIN), 0 = off
KMAX, KRATE = 0.2, 0.1          # 1/m (openpilot's cap; minimum radius 5 m), 1/m/s (about 0.29 rad/s of road-wheel steering, wheelbase 2.85)
KDYN, V_KIN = 0.075, (3.5, 5.0)  # 1/m at and above V_KIN[1] m/s (steering ~0.21 rad: the dynamic model's side slip stays < 0.15), KMAX at and
                                 # below V_KIN[0], linear between: AlpaSim's vehicle switches to the linear-tyre model at 5 m/s; (4.0, 5.5) / 0.085
                                 # left 8 / 38 rollouts with side slip >= 0.15 just above 5 m/s (run 4)
V_DROP = 2.0                     # m/s^2: the curvature cap uses max(v, v0 - V_DROP t), the car slows less than the reference asks
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
    """The plan's path: polyline origin + 8 points, extended 200 m straight along its last chord; arc-length lookups. Heading at a node =
    the plan's own yaw output (the tangent; the origin is 0) unless it disagrees with the mean of the adjacent chords by > YAW_TOL, then that
    mean; linear in s between nodes, so curvature is piecewise constant and the heading does not lag the path by half a segment."""

    def __init__(self, poses):
        q = np.asarray(poses, np.float64)
        p, yw = np.concatenate([np.zeros((1, 2)), q[:, :2]]), np.r_[0.0, q[:, 2]]
        ds = np.hypot(*np.diff(p, axis=0).T)
        keep = np.r_[True, ds > 0.05]                                     # drop repeated points (stop plans)
        p, yw = p[keep], yw[keep]
        d = np.diff(p, axis=0)
        ds = np.hypot(*d.T)
        if len(ds) == 0:                                                  # a plan that does not move: straight ahead
            p, yw, d, ds = np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), np.array([[1.0, 0.0]]), np.array([1.0])
        ch = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        mean = np.r_[ch[0], (ch[:-1] + ch[1:]) / 2, ch[-1]]
        yw = mean + (yw - mean + np.pi) % (2 * np.pi) - np.pi             # the yaw output brought next to the chords
        hn = np.where(np.abs(yw - mean) <= YAW_TOL, yw, mean)
        hn[0] = 0.0 if abs(mean[0]) <= YAW_TOL else mean[0]
        self.s = np.r_[0.0, np.cumsum(ds)]
        far = p[-1] + 200.0 * d[-1] / ds[-1]
        self.p, self.s = np.vstack([p, far]), np.r_[self.s, self.s[-1] + 200.0]
        self.hs, self.hv = self.s, np.r_[hn, ch[-1]]

    def at(self, s):
        """-> x, y on the polyline, heading, curvature (d heading / ds) at arc length s."""
        x, y = np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])
        h = np.interp(s, self.hs, self.hv)
        i = np.clip(np.searchsorted(self.hs, s, side="right") - 1, 0, len(self.hs) - 2)
        k = (self.hv[i + 1] - self.hv[i]) / np.maximum(self.hs[i + 1] - self.hs[i], 1e-6)
        return x, y, h, k


def speeds(poses):
    """Plan speed on its own clock: segment speeds of the 8 points at 0.5 s."""
    p = np.concatenate([np.zeros((1, 2)), np.asarray(poses, np.float64)[:, :2]])
    return np.hypot(*np.diff(p, axis=0).T) / 0.5


def build(poses, v0: float, wz: float, mode: str = "", a0: float = 0.0, **kw):
    """Plan (8, 3) rig frame, ego speed, yaw rate (logged only), acceleration -> (t (41,), x, y, yaw (41,)) and a log record. mode / kw (lower-case constant names)
    override the environment's configuration (offline replays)."""
    mode = mode or MODE
    g = {k: kw.get(k.lower(), globals()[k]) for k in ("AMIN", "AMAX", "ALAT", "JLAT", "KMAX", "KRATE", "TAU_LAT", "D_MIN", "ZETA", "V_MIN", "VCURV", "KDYN", "V_DROP", "JLON", "LATCAP")}
    AMIN, AMAX, ALAT, JLAT, KMAX, KRATE, TAU_LAT, D_MIN, ZETA, V_MIN, VCURV, KDYN, V_DROP, JLON, LATCAP = g.values()
    kcap = lambda v: np.minimum(np.interp(v, V_KIN, (KMAX, KDYN)), ALAT / np.maximum(v, V_MIN) ** 2)  # noqa: E731
    path, vp = Path(poses), speeds(poses)
    vlim = None
    if mode == "full" and VCURV > 0:                                      # curvature speed limit, backward pass with braking at AMIN
        sg = np.arange(0.0, path.s[-1], 0.5)
        ka = np.abs(path.at(sg)[3])
        va = np.minimum(np.sqrt(ALAT / np.maximum(ka, 1e-4)), np.where(ka <= KDYN, 1e3, np.interp(ka, (KDYN, KMAX), V_KIN[::-1])))
        for j in range(len(sg) - 2, -1, -1):
            va[j] = min(va[j], np.sqrt(va[j + 1] ** 2 - 2 * AMIN * 0.5))
        vlim = (sg, va)
    n, every = int(round(T_END / DT_SIM)), int(round(DT_OUT / DT_SIM))
    v, s, x, y, th = max(float(v0), 0.0), 0.0, 0.0, 0.0, 0.0
    k = None                                                              # first step: the clipped desired curvature
    out, kpk, apk, a = [(0.0, 0.0, 0.0)], 0.0, 0.0, float(np.clip(a0, AMIN, AMAX))
    for i in range(n):
        t = i * DT_SIM
        vt = vp[min(int(t / 0.5), 7)]
        if vlim is not None:
            vt = min(vt, float(np.interp(s, *vlim)))
        a = float(np.clip(np.clip((vt - v) / DT_SIM, a - JLON * DT_SIM, a + JLON * DT_SIM), AMIN, AMAX))
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
            vv = max(v, V_MIN, v0 - V_DROP * t)
            km = float(kcap(vv))
            dk = min(JLAT / vv ** 2, KRATE) * DT_SIM
            k = float(np.clip(kd if k is None else np.clip(kd, k - dk, k + dk), -km, km)) if LATCAP > 0 else float(kd)
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
    return (t, o[:, 0], o[:, 1], o[:, 2]), {"mode": mode, "v0": round(float(v0), 3), "k_end": round(float(k or 0.0), 4), "kpk": round(kpk, 4),
                                           "amin": round(apk, 2), "s4": round(s, 2), "wz": round(float(wz), 4)}


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
