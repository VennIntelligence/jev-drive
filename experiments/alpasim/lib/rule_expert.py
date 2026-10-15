"""RULE EXPERT: a map-based rule planner that replaces the model's plan in sh30_driver.py (nuPlan track) and pai_driver.py (PAI track);
the AlpaSim runtime, controller, route messages and scoring are unchanged (experiments/route_follow/plans/2026-10-15-rule-expert-prereg.md).
PRIVILEGED (map, log speeds and replayed agents read by scene id): a teacher / probe only, never in a submission image.

Per scene a pack (experiments/route_follow/scripts/rf_expert.py pack, <JEV_EXPERT>/<scene>.npz, numpy only) holds:
  path      the route-lane centreline chain (rf_corridor.SceneMap.match on the log prefix + AlpaSim's MAP route generated along the log;
            ambiguous lane candidates resolved toward the route points; a lane change blended over [cut - 15, cut + 15] m), 0.5 m
            stations, extended 30 m back and 150 m forward along its end headings
  turns     (entry, exit, deg) on the path: >= 45 deg net heading change within 30 m, entry / exit at 10 / 90 % of the change
  vlog      the human log's speed when it passed each station (NaN where the log never passed)
  agents    the replayed objects' logged boxes (ds.traffic_objects: t, x, y, yaw, length, width; the simulator replays exactly these)
Per decision (rear-axle pose, speed, time):
  speed     cruise = max(vlog, V_FLOOR) where known, else V_DEF (the trajdata map has no speed limits); from DECEL_D m before every turn entry
            linear in distance down to V_E, V_E through the turn, then up at A_ACC; capped by v^2 |kappa| <= ALAT on the path's curvature;
            a braking envelope at A_DEC makes the profile reachable
  yield     the plan's ego box (sides + SIDE) against every replayed box ahead of it (centre ahead of the ego box centre), 0.1 s steps over
            YIELD_T s: at the first contact the profile stops STOP_M before the contact pose (up to 3 passes for later contacts)
  path      pure pursuit from the rear axle on the chain, lookahead max(LA_MIN, LA_T * v), curvature within KMAX, kinematic bicycle at 0.1 s
  output    the 8 rear-axle poses at 0.5 .. 4.0 s in the anchor frame (the model's plan format) + an info dict (logged as drive.jsonl `expert`)
Stop lines and traffic lights are ignored (prereg).

Environment: JEV_EXPERT (pack directory; unset = off), JEV_EXPERT_LA (lookahead time, s, default 1.2), JEV_EXPERT_EV (turn entry speed, m/s,
default 4.6).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

PACK = os.environ.get("JEV_EXPERT", "") or ""
ON = bool(PACK)
LA_T = float(os.environ.get("JEV_EXPERT_LA", "") or 1.2)
V_E = float(os.environ.get("JEV_EXPERT_EV", "") or 4.6)
SUFFIX = f"-xp{LA_T:g}e{V_E:g}" if ON else ""

LA_MIN, KMAX = 6.0, 0.2                      # m; 1/m (Pacifica: about 5 m minimum radius at the rear axle)
DECEL_D, V_FLOOR, V_DEF = 26.0, 3.0, 12.0    # m (decision 293); m/s; m/s (no speed limit in the map)
A_ACC, A_DEC, A_MAX, ALAT = 1.5, 3.0, 6.0, 4.0
YIELD_T, STOP_M, SIDE = 3.0, 1.5, 0.25
DT, NSTEP, OUT = 0.1, 40, np.arange(5, 41, 5)  # sim steps; output steps = 0.5 .. 4.0 s
HORIZON = 140.0                              # m of path the speed envelope looks at

_lock = threading.Lock()
_scenes: dict = {}
_sess: dict = {}


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


class Scene:
    def __init__(self, f: Path):
        z = np.load(f)
        self.P, self.S = z["path"].astype(float), z["s"].astype(float)
        self.turns = z["turns"].astype(float).reshape(-1, 3)
        self.vlog = z["vlog"].astype(float)
        self.dims = z["dims"].astype(float)                     # rear (< 0), front, half width about the rear axle
        off = z["obj_off"]
        self.obj = [(z["obj_t"][off[i]:off[i + 1]].astype(float), z["obj_xyh"][off[i]:off[i + 1]].astype(float), z["obj_lw"][i].astype(float))
                    for i in range(len(off) - 1)]
        self.oid = [str(x) for x in z["obj_id"]]
        self.t_lo = np.array([o[0][0] if len(o[0]) else np.inf for o in self.obj])
        self.t_hi = np.array([o[0][-1] if len(o[0]) else -np.inf for o in self.obj])
        self.ds = float(np.median(np.diff(self.S)))
        self.kap = self.curvature()
        self.vbase = self.profile(V_E)

    def curvature(self):
        """|kappa| of the path from the heading change over +-2.5 m, max-filtered over +-2.5 m."""
        n = max(int(round(2.5 / self.ds)), 1)
        d = self.P[2 * n:] - self.P[:-2 * n]
        h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        k = np.zeros(len(self.P))
        k[n:-n] = np.abs(np.gradient(h, self.ds))
        k[:n], k[-n:] = k[n], k[-n - 1]
        w = np.lib.stride_tricks.sliding_window_view(np.pad(k, n, mode="edge"), 2 * n + 1).max(1)
        return w

    def profile(self, ve):
        S = self.S
        vc = np.where(np.isfinite(self.vlog), np.maximum(self.vlog, V_FLOOR), V_DEF)
        v = vc.copy()
        for en, ex, _ in self.turns:
            sa = en - DECEL_D
            va = float(np.interp(sa, S, vc))
            m = (S >= sa) & (S < en)
            v[m] = np.minimum(v[m], va + (ve - va) * (S[m] - sa) / DECEL_D)
            m = (S >= en) & (S <= ex)
            v[m] = np.minimum(v[m], ve)
            m = S > ex
            v[m] = np.minimum(v[m], np.sqrt(ve * ve + 2 * A_ACC * (S[m] - ex)))
        return np.minimum(v, np.sqrt(ALAT / np.maximum(self.kap, 1e-4)))

    def project(self, xy, lo=None, hi=None):
        """Point -> (station, signed lateral left +, index of the segment) on the path within [lo, hi]."""
        i0 = 0 if lo is None else max(int(np.searchsorted(self.S, lo)) - 1, 0)
        i1 = len(self.S) - 1 if hi is None else min(int(np.searchsorted(self.S, hi)) + 1, len(self.S) - 1)
        if i1 <= i0:
            i0, i1 = max(i1 - 1, 0), max(i1, 1)
        A, B = self.P[i0:i1], self.P[i0 + 1:i1 + 1]
        d = B - A
        L2 = np.maximum((d * d).sum(1), 1e-12)
        u = np.clip(((xy - A) * d).sum(1) / L2, 0, 1)
        q = A + u[:, None] * d
        e = ((q - xy) ** 2).sum(1)
        j = int(np.argmin(e))
        lat = float(np.sign(d[j, 0] * (xy[1] - A[j, 1]) - d[j, 1] * (xy[0] - A[j, 0])) * np.sqrt(e[j]))
        return float(self.S[i0 + j] + u[j] * np.sqrt(L2[j])), lat

    def at(self, s):
        return np.array([np.interp(s, self.S, self.P[:, 0]), np.interp(s, self.S, self.P[:, 1])])

    def boxes(self, t_us):
        """Replayed boxes at times t_us (m,) -> centre / heading (n, m, 3), half length / width (n, 2), valid (n, m), ids."""
        sel = np.flatnonzero((self.t_hi >= t_us[0]) & (self.t_lo <= t_us[-1]))
        X = np.full((len(sel), len(t_us), 3), np.nan)
        V = np.zeros((len(sel), len(t_us)), bool)
        HW = np.zeros((len(sel), 2))
        for r, i in enumerate(sel):
            T, xyh, lw = self.obj[i]
            ok = (t_us >= T[0] - 1) & (t_us <= T[-1] + 1)
            if len(T) == 1:
                X[r, ok] = xyh[0]
            else:
                X[r, :, 0], X[r, :, 1] = np.interp(t_us, T, xyh[:, 0]), np.interp(t_us, T, xyh[:, 1])
                X[r, :, 2] = np.interp(t_us, T, xyh[:, 2])
            V[r] = ok
            HW[r] = lw / 2
        return X, HW, V, [self.oid[i] for i in sel]


def scene(name: str):
    with _lock:
        if name not in _scenes:
            f = Path(PACK) / f"{name}.npz"
            _scenes[name] = Scene(f) if f.exists() else None
        return _scenes[name]


def corners(x, y, h, front, rear, hw):
    """Box corners (..., 4, 2) of rectangles about (x, y, h) spanning [rear, front] x [-hw, hw] in their own frame."""
    c, s = np.cos(h)[..., None], np.sin(h)[..., None]
    u = np.stack(np.broadcast_arrays(front, front, rear, rear), -1) if np.ndim(front) else np.array([front, front, rear, rear])
    v = np.stack(np.broadcast_arrays(hw, -hw, -hw, hw), -1) if np.ndim(hw) else np.array([hw, -hw, -hw, hw])
    return np.stack([x[..., None] + c * u - s * v, y[..., None] + s * u + c * v], -1)


def overlap(A, B):
    """Corner sets A (..., 4, 2), B (..., 4, 2) of rectangles -> overlap (...) by the separating axis test."""
    out = np.ones(np.broadcast_shapes(A.shape[:-2], B.shape[:-2]), bool)
    for Q in (A, B):
        for e in (Q[..., 1, :] - Q[..., 0, :], Q[..., 3, :] - Q[..., 0, :]):
            pa, pb = (A * e[..., None, :]).sum(-1), (B * e[..., None, :]).sum(-1)
            out &= (pa.max(-1) >= pb.min(-1)) & (pb.max(-1) >= pa.min(-1))
    return out


def simulate(sc: Scene, x, y, h, v, s0, vt):
    """Pure pursuit + speed control along the path from the rear-axle state; vt(s) the allowed speed -> states (NSTEP + 1, 5): x, y, h, v, s."""
    out = np.zeros((NSTEP + 1, 5))
    out[0] = x, y, h, v, s0
    s = s0
    for n in range(1, NSTEP + 1):
        ld = max(LA_MIN, LA_T * v)
        q = sc.at(s + ld)
        dx, dy = q[0] - x, q[1] - y
        al = np.arctan2(dy, dx) - h
        k = float(np.clip(2 * np.sin(al) / max(np.hypot(dx, dy), 1e-3), -KMAX, KMAX))
        vn = max(0.0, v + float(np.clip((float(np.interp(s, sc.S, vt)) - v) / DT, -A_MAX, A_ACC)) * DT)
        d = 0.5 * (v + vn) * DT
        hm = h + 0.5 * k * d
        x, y, h, v = x + d * np.cos(hm), y + d * np.sin(hm), h + k * d, vn
        s = sc.project(np.array([x, y]), s - 2.0, s + d + 4.0)[0]
        out[n] = x, y, h, v, s
    return out


def envelope(sc: Scene, s0, stop):
    """The allowed speed on the path stations: the base profile, 0 from `stop` on, and the A_DEC braking envelope backwards."""
    vt = sc.vbase.copy()
    if stop is not None:
        vt[sc.S >= stop - sc.ds] = 0.0                       # from the grid point before: no creep once the ego is at the stop
    i0, i1 = max(int(np.searchsorted(sc.S, s0)) - 2, 0), min(int(np.searchsorted(sc.S, s0 + HORIZON)), len(vt) - 1)
    lim = 2 * A_DEC * sc.ds
    for i in range(i1 - 1, i0 - 1, -1):
        vt[i] = min(vt[i], np.sqrt(vt[i + 1] ** 2 + lim))
    return vt


def contact(sc: Scene, st, t0):
    """First contact of the plan's ego box with a replayed box ahead within YIELD_T -> (step, object id) or None."""
    m = int(round(YIELD_T / DT))
    t = t0 + 1e6 * DT * np.arange(m + 1)
    X, HW, V, ids = sc.boxes(t)
    if not len(ids):
        return None
    rear, front, hw = sc.dims
    E = corners(st[:m + 1, 0], st[:m + 1, 1], st[:m + 1, 2], front, rear, hw + SIDE)                       # (m+1, 4, 2)
    O = corners(X[..., 0], X[..., 1], X[..., 2], HW[:, None, 0], -HW[:, None, 0], HW[:, None, 1])          # (n, m+1, 4, 2)
    hit = overlap(E[None], O) & V & np.isfinite(X[..., 0])
    c = 0.5 * (rear + front)
    ch, sh = np.cos(st[:m + 1, 2]), np.sin(st[:m + 1, 2])
    ahead = (X[..., 0] - st[None, :m + 1, 0]) * ch + (X[..., 1] - st[None, :m + 1, 1]) * sh > c
    hit &= ahead & ~hit[:, :1]                                                                                # touching now: nothing to yield
    if not hit.any():
        return None
    n, j = np.argwhere(hit)[np.argmin(np.argwhere(hit)[:, 1])]
    return int(j), ids[int(n)]


def plan(session: str, scene_id: str, t0: int, pose, v: float):
    """-> (poses (8, 3) in the anchor frame, info dict). A scene without a pack, or any failure, raises (the driver aborts the call)."""
    sc = scene(scene_id)
    if sc is None:
        raise FileNotFoundError(f"no expert pack for scene {scene_id} in {PACK}")
    x, y, h = map(float, pose)
    hint = _sess.get(session)
    s0, lat = sc.project(np.array([x, y]), *((hint - 10.0, hint + 40.0) if hint is not None else (None, None)))
    _sess[session] = s0
    stop, hits = None, []
    for _ in range(3):
        st = simulate(sc, x, y, h, v, s0, envelope(sc, s0, stop))
        c = contact(sc, st, t0)
        if c is None:
            break
        new = max(float(st[c[0], 4]) - STOP_M, s0)
        hits.append([round(c[0] * DT, 1), c[1]])
        if stop is not None and new >= stop - 0.1:
            break
        stop = new if stop is None else min(stop, new)
    R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    q = st[OUT]
    xy = (q[:, :2] - [x, y]) @ R.T
    poses = np.c_[xy, wrap(q[:, 2] - h)]
    nxt = [t for t in sc.turns if t[1] >= s0]
    info = dict(s=round(s0, 2), lat=round(lat, 3), v=round(v, 3), vt=round(float(np.interp(s0, sc.S, sc.vbase)), 3),
                v_plan=np.round(q[:, 3], 3).tolist(), stop=None if stop is None else round(stop - s0, 2), hits=hits,
                turn=None if not nxt else [round(nxt[0][0] - s0, 1), round(nxt[0][1] - s0, 1), round(float(nxt[0][2]), 1)])
    return poses, info


def close(session: str):
    _sess.pop(session, None)
