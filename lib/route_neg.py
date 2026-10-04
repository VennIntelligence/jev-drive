"""Negative route polylines: a route that contradicts the scene. The correct behaviour is to keep following the logged path (target = the
hindsight / logged future of the frame, or the original model's plan when distilling), not to execute the route.

Kinds (spec in experiments/op_route_cmd/README.md and results/negatives.md):
  N1 exit    at a junction approach the route turns into an exit class that the road does not have (map: not in the ego lane's exits;
             tier A also not in any lane of the ego roadblock, tier B = another lane has it, needs a lane change, may be a legal route)
  N2 side    on a road with no junction ahead the route turns 90 deg into a side road that the map does not have (driveways and parking-lot
             openings are not in the map: always needs a visual check)
  N3 wrong   the route runs 4-7 m to the oncoming side of the road (wrong way), then continues parallel
  N4 uturn   the route makes a U-turn on a through road
Every function takes the driven path in the ego frame (x forward, y left, dense, first row the origin) and returns a dict like `route_poly.hindsight`.
"""
from __future__ import annotations

import numpy as np

import route_poly as RP

KINDS = ("N1_exit", "N2_side", "N3_wrong", "N4_uturn")


def _cut(path, s):
    """Driven path truncated at arc length s; returns (points, end point, unit heading at the end)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = float(min(s, cum[-1]))
    n = int(np.searchsorted(cum, s))
    p = np.vstack([path[:n], [np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])]])
    d = p[-1] - p[max(len(p) - 1 - max(int(2.0 / 0.5), 1), 0)]
    return p, p[-1], d / max(np.linalg.norm(d), 1e-9)


def splice_turn(path, s_turn, angle_deg, radius, tail=170.0, step=0.5):
    """Driven path up to s_turn, then an arc of `angle_deg` (left positive; +-180 = U-turn) with `radius`, then straight on."""
    p, e, h = _cut(path, s_turn)
    sg = 1.0 if angle_deg >= 0 else -1.0
    th = np.arange(step, np.radians(abs(angle_deg)) * radius + 1e-9, step) / radius
    nrm = sg * np.array([-h[1], h[0]])                       # towards the turn side
    arc = e + radius * (np.sin(th)[:, None] * h + (1 - np.cos(th))[:, None] * nrm)
    ang = sg * th[-1] if len(th) else 0.0
    c, s = np.cos(ang), np.sin(ang)
    h2 = np.array([c * h[0] - s * h[1], s * h[0] + c * h[1]])
    last = arc[-1] if len(arc) else e
    straight = last + np.arange(step, tail, step)[:, None] * h2
    return np.vstack([p, arc, straight])


def lateral_shift(path, offset, s0, ramp=20.0):
    """Path moved sideways by `offset` m (left positive) with a smooth ramp over `ramp` m starting at arc length s0."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    t = np.gradient(path, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], -1)
    u = np.clip((cum - s0) / ramp, 0, 1)
    return path + (offset * (3 * u ** 2 - 2 * u ** 3))[:, None] * nrm


def _label(path, kind, **meta):
    h = RP.hindsight(path)
    h.update(kind=kind, **meta)
    return h


def make(kind, path, rng, *, s_branch=None, missing=None, lht=False, plen=None):
    """One negative of `kind` from the driven path (ego frame, dense). Returns None when the path cannot carry it.

    s_branch  distance (m) to the branching point (N1); missing  the absent exit class 'left' / 'right' / 'straight' (N1)
    lht       left-hand traffic (N3: the oncoming side is then to the right)
    """
    plen = plen if plen is not None else float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if kind == "N1_exit":
        if s_branch is None or missing is None or plen < s_branch + 5:
            return None
        if missing == "straight":                             # straight into a T: the road ahead is simply not there, the route runs on
            return _label(splice_turn(path, s_branch, 0.0, 10.0), kind, missing=missing, s_turn=float(s_branch), angle=0.0, radius=0.0)
        ang, r = (90.0 if missing == "left" else -90.0), float(rng.uniform(9.0, 15.0))
        return _label(splice_turn(path, s_branch, ang, r), kind, missing=missing, s_turn=float(s_branch), angle=ang, radius=r)
    if kind == "N2_side":
        if plen < 90:
            return None
        s_t, sg = float(rng.uniform(20.0, min(60.0, plen - 40.0))), float(rng.choice([-1.0, 1.0]))
        r = float(rng.uniform(9.0, 15.0))
        return _label(splice_turn(path, s_t, sg * 90.0, r), kind, missing="side_road", s_turn=s_t, angle=sg * 90.0, radius=r)
    if kind == "N3_wrong":
        s0 = float(rng.uniform(0.0, 15.0))
        off = float(rng.uniform(4.0, 7.0)) * (-1.0 if lht else 1.0)
        return _label(lateral_shift(path, off, s0), kind, missing="oncoming_lane", s_turn=s0, angle=0.0, radius=off)
    if kind == "N4_uturn":
        if plen < 60:
            return None
        s_t, sg = float(rng.uniform(15.0, min(45.0, plen - 15.0))), float(rng.choice([-1.0, 1.0]))
        r = float(rng.uniform(5.0, 8.0))
        return _label(splice_turn(path, s_t, sg * 180.0, r), kind, missing="u_turn", s_turn=s_t, angle=sg * 180.0, radius=r)
    raise ValueError(kind)
