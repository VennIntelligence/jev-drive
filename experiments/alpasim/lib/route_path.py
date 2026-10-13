"""JEV_ROUTE: the served path is the carried AlpaSim route, the speed profile stays the plan's (experiments/route_only,
plans/2026-10-13-route-only-prereg.md). Off by default; the driver's path is unchanged when off.

  JEV_ROUTE=own   the 8 served poses (0.5 .. 4 s) are placed on the route path at the cumulative arc lengths of the served plan
                  (after JEV_VCONT / JEV_LEAD if those are on): openpilot-style "longitudinal from the model, lateral from the route"
  JEV_ROUTE=bridge  as `own`, but the gap is bridged instead of kept as the model plan (prereg Amendment 1): when the memory starts
                  ahead of the ego, the blend runs from the ego pose straight into the route at route arc max(0, Lb - g) from its start
                  (the route start when g >= Lb), so every pose is on the route path (the junction inside the gap is crossed by the
                  Hermite curve, not by the model)
  JEV_ROUTE=log   privileged reference: the same path, arc lengths = the human log's travelled distance from t0 to t0 + 0.5 k s, read from
                  AlpaSim's recording ground truth (needs runtime.simulation_config.send_recording_ground_truth=true); past the log end the
                  log's last speed is held

Route path at a decision (rear-axle frame of t0, x forward, y left), from the route memory of experiments/route_mem/lib/rmem.py (every
route message of the episode carried by the ego poses and merged; the same code the ROUTE-MEM branch is fed from):
  s_e       the ego's projection on the merged route polyline (negative = the memory starts ahead of the ego: the first decisions, whose only
            message starts ~40 m ahead)
  gap       g = max(0, -s_e). g <= GAP_TOL: the path starts at the ego (0, 0, heading 0). Otherwise the uncovered part is the model plan's
            own path up to arc g, and the path continues from that point (with the plan's tangent there)
  blend     a cubic Hermite curve from the start point / heading into the route at route arc max(s_e, 0) + Lb with the route's tangent,
            tangent lengths = the chord; Lb = clip(BLEND_T x v_ego, BLEND_MIN, BLEND_MAX) = clip(2 s x v, 8 m, 30 m): the join lies at or
            beyond the 2 s pose, so the 1-2 s window the AlpaSim MPC tracks sees one smooth curve, never a step to the centreline
  route     the merged polyline from the join to its end
A pose whose target arc lies inside the gap or beyond the path end keeps the model plan (fallback; for `log` the plan's path sampled at the
log arc, extended along its last segment). Yaw of a route-placed pose = the path tangent. Logged per decision as drive.jsonl `route`.
"""
from __future__ import annotations

import os

import numpy as np

MODE = os.environ.get("JEV_ROUTE", "off").strip().lower() or "off"
assert MODE in ("off", "own", "log", "bridge"), f"JEV_ROUTE={MODE}: off | own | log | bridge"
ON, NEED_GT = MODE != "off", MODE == "log"
SUFFIX = "" if not ON else f"-route{MODE}"
BLEND_T, BLEND_MIN, BLEND_MAX, GAP_TOL = 2.0, 8.0, 30.0, 0.5
T_POSE = 0.5 * np.arange(1, 9)


def _clean(P):
    P = np.asarray(P, np.float64).reshape(-1, 2)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 2:
        return P, None
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]
    P = P[keep]
    return P, (np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))] if len(P) > 1 else None)


def _at(P, cum, s, extend=False):
    """Points and tangent headings at arc lengths s on polyline P (cum = its arc lengths); beyond the ends along the end segments."""
    s = np.atleast_1d(np.asarray(s, np.float64))
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(P) - 2)
    d = P[j + 1] - P[j]
    L = np.maximum(cum[j + 1] - cum[j], 1e-9)
    u = d / L[:, None]
    return P[j] + (s - cum[j])[:, None] * u, np.arctan2(u[:, 1], u[:, 0])


def project(M, cum):
    """Arc position of the origin on polyline M (extrapolated before the start / after the end) and its distance to the polyline."""
    d = np.diff(M, axis=0)
    L = cum[1:] - cum[:-1]
    u = d / L[:, None]
    t = np.clip(-(M[:-1] * u).sum(1), 0.0, L)
    q = M[:-1] + t[:, None] * u
    r = np.hypot(q[:, 0], q[:, 1])
    i = int(np.argmin(r))
    s = cum[i] + t[i]
    if i == 0 and t[0] <= 0.0:
        s = float(-(M[0] * u[0]).sum())
    elif i == len(L) - 1 and t[i] >= L[i]:
        s = cum[-1] + float(-(M[-1] * u[-1]).sum())
    return float(s), float(r[i])


def hermite(A, hA, B, hB, n=24):
    D = float(np.hypot(*(B - A)))
    t = np.linspace(0.0, 1.0, n)[:, None]
    m0, m1 = D * np.array([np.cos(hA), np.sin(hA)]), D * np.array([np.cos(hB), np.sin(hB)])
    return (2 * t**3 - 3 * t**2 + 1) * A + (t**3 - 2 * t**2 + t) * m0 + (-2 * t**3 + 3 * t**2) * B + (t**3 - t**2) * m1


def plan_arcs(poses):
    xy = np.r_[[[0.0, 0.0]], np.asarray(poses, np.float64)[:, :2]]
    return np.cumsum(np.hypot(*np.diff(xy, axis=0).T))


def log_arcs(gt_t, gt_xy, t0):
    """Human-log travelled distance from t0 to t0 + 0.5 .. 4 s (gt_t in us, gt_xy any frame); past the log end at its last speed."""
    gt_t = np.asarray(gt_t, np.float64)
    o = np.argsort(gt_t)
    gt_t, gt_xy = gt_t[o], np.asarray(gt_xy, np.float64)[o]
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(gt_xy, axis=0).T))]
    q = t0 + 1e6 * np.r_[0.0, T_POSE]
    s = np.interp(q, gt_t, cum)
    if q[-1] > gt_t[-1] and len(gt_t) > 1:
        k = max(0, len(gt_t) - 6)
        v = (cum[-1] - cum[k]) / max((gt_t[-1] - gt_t[k]) * 1e-6, 1e-6)
        s = s + v * np.clip(q - gt_t[-1], 0, None) * 1e-6
    if q[0] < gt_t[0]:
        return None
    return s[1:] - s[0]


def serve(poses, M, v0, arcs=None):
    """poses (8, 3) served plan; M merged route polyline (n, 2) in the t0 frame or None; v0 ego speed; arcs (8,) target arc lengths
    (None = the plan's own) -> (poses (8, 3), info dict, path (k, 2) at ~2 m for the log)."""
    poses = np.asarray(poses, np.float64)
    own = arcs is None
    c = plan_arcs(poses) if own else np.asarray(arcs, np.float64)
    info = dict(mode=MODE, cov=0, gap=None, s_e=None, off=None, ahead=None, lb=None)
    if M is None:
        return poses, info | dict(why="no_route"), None
    M, cum = _clean(M)
    if cum is None or cum[-1] < 1.0:
        return poses, info | dict(why="no_route"), None
    s_e, off = project(M, cum)
    info.update(s_e=round(s_e, 2), off=round(off, 3), ahead=round(float(cum[-1] - max(s_e, 0.0)), 2))
    if s_e > cum[-1] - 1.0:
        return poses, info | dict(why="behind_end"), None
    P, Pc = _clean(np.r_[[[0.0, 0.0]], poses[:, :2]])        # the plan's own path
    g = max(0.0, -s_e)
    bridge = MODE == "bridge" and g > GAP_TOL
    if g <= GAP_TOL or bridge:
        A, hA, head, gA = np.zeros(2), 0.0, np.zeros((0, 2)), 0.0
    else:
        if Pc is None or Pc[-1] <= g:
            return _fallback(poses, P, Pc, c, own), info | dict(gap=round(g, 2), why="gap"), None
        A, hA = (x[0] for x in _at(P, Pc, g))
        head = np.r_[P[Pc < g], A[None]]
        gA = g
    Lb = float(np.clip(BLEND_T * v0, BLEND_MIN, BLEND_MAX))
    sj = min(max(0.0, Lb - g) if bridge else max(s_e, 0.0) + Lb, cum[-1])
    B, hB = (x[0] for x in _at(M, cum, sj))
    H = hermite(A, hA, B, hB)
    tail = M[cum > sj + 1e-6]
    path, pc = _clean(np.r_[head, H, tail])
    q = c >= gA - 1e-9 if gA > 0 else np.ones(8, bool)
    q &= c <= pc[-1]
    out = _fallback(poses, P, Pc, c, own)
    if q.any():
        X, h = _at(path, pc, c[q])
        out[q, :2] = X
        out[q, 2] = np.unwrap(np.r_[0.0, h])[1:]
    info.update(cov=int(q.sum()), gap=round(g, 2), lb=round(Lb, 2), why="bridge" if bridge else "ok", dlat=np.round(_dlat(poses, out), 3).tolist())
    ss = np.arange(0.0, min(pc[-1], 90.0), 2.0)
    return out, info, np.round(_at(path, pc, ss)[0], 2)


def _fallback(poses, P, Pc, c, own):
    """The model plan: unchanged for `own`; for `log` its path sampled at the log arcs (extended along the last segment)."""
    if own or Pc is None:
        return poses.copy()
    out = poses.copy()
    out[:, :2] = _at(P, Pc, c)[0]
    out[:, 2] = np.interp(c, plan_arcs(poses), poses[:, 2])
    return out


def _dlat(a, b):
    """Lateral (y in the t0 frame) difference served - plan at 1, 2, 4 s."""
    return (b[[1, 3, 7], 1] - a[[1, 3, 7], 1])


if __name__ == "__main__":                                           # self-test
    straight = np.c_[np.arange(-10, 90, 4.0), np.zeros(25) + 0.6]
    poses = np.c_[5 * T_POSE, 0.3 * T_POSE, np.zeros(8)]             # plan drifting left at 5 m/s
    o, i, p = serve(poses, straight, 5.0)
    assert i["why"] == "ok" and i["cov"] == 8, i
    assert np.allclose(plan_arcs(o), plan_arcs(poses), atol=0.05), (plan_arcs(o), plan_arcs(poses))
    assert abs(o[-1, 1] - 0.6) < 0.05 and abs(o[0, 1]) < 0.3, o              # joins the centreline, no jump at 0.5 s
    gap = np.c_[np.arange(42, 82, 4.21), np.zeros(10)]
    o, i, _ = serve(poses, gap, 5.0)
    assert i["why"] == "gap" and np.array_equal(o, poses), i
    fast = np.c_[12 * T_POSE, np.zeros(8), np.zeros(8)]
    o, i, _ = serve(fast, gap, 12.0)
    assert i["why"] == "ok" and 0 < i["cov"] < 8 and np.array_equal(o[: 8 - i["cov"]], fast[: 8 - i["cov"]]), i
    r, th = 15.0, np.linspace(-0.5, np.pi / 2, 300)                   # left turn ahead: straight 20 m then a quarter circle
    turn = np.r_[np.c_[np.linspace(-6, 20, 30), np.zeros(30)], np.c_[20 + r * np.sin(th[th > 0]), r - r * np.cos(th[th > 0])]]
    o, i, _ = serve(np.c_[8 * T_POSE, np.zeros(8), np.zeros(8)], turn, 8.0)
    assert i["cov"] == 8 and o[-1, 1] > 4 and o[-1, 2] > 0.7, (o, i)
    MODE = "bridge"
    o, i, _ = serve(np.c_[3 * T_POSE, np.zeros(8), np.zeros(8)], np.c_[30 + 4.21 * np.arange(10), 5.0 + np.zeros(10)], 3.0)
    assert i["why"] == "bridge" and i["cov"] == 8 and o[-1, 1] > 1.0 and np.allclose(plan_arcs(o), 3 * T_POSE, atol=0.05), (o, i)
    MODE = "own"
    a = log_arcs(np.arange(0, 20e6, 1e5), np.c_[np.arange(200) * 0.4, np.zeros(200)], 5e6)
    assert np.allclose(a, 4 * T_POSE), a
    o2, i2, _ = serve(np.c_[8 * T_POSE, np.zeros(8), np.zeros(8)], turn, 8.0, arcs=a)
    assert np.allclose(plan_arcs(o2), a, atol=0.1)
    print("route_path selftest ok", i)
