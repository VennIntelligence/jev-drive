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


def _label(path, kind, s_free=0.0, **meta):
    """Hindsight fields of the negative plus `dense` (0.5 m grid, ego frame) and `s_free`: arc length at which the deviation from the driven road is complete
    (the arc has ended / the lateral ramp is over); the map screens look only at the path after it."""
    h = RP.hindsight(path)
    h.update(kind=kind, dense=RP.poly_resample(path, 0.5)[0].astype(np.float32), s_free=float(s_free), **meta)
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
            return _label(splice_turn(path, s_branch, 0.0, 10.0), kind, s_free=s_branch, missing=missing, s_turn=float(s_branch), angle=0.0, radius=0.0)
        ang, r = (90.0 if missing == "left" else -90.0), float(rng.uniform(9.0, 15.0))
        return _label(splice_turn(path, s_branch, ang, r), kind, s_free=s_branch + np.radians(abs(ang)) * r, missing=missing, s_turn=float(s_branch), angle=ang, radius=r)
    if kind == "N2_side":
        if plen < 90:
            return None
        s_t, sg = float(rng.uniform(20.0, min(60.0, plen - 40.0))), float(rng.choice([-1.0, 1.0]))
        r = float(rng.uniform(9.0, 15.0))
        return _label(splice_turn(path, s_t, sg * 90.0, r), kind, s_free=s_t + np.pi / 2 * r, missing="side_road", s_turn=s_t, angle=sg * 90.0, radius=r)
    if kind == "N3_wrong":
        s0 = float(rng.uniform(0.0, 15.0))
        off = float(rng.uniform(4.0, 7.0)) * (-1.0 if lht else 1.0)
        return _label(lateral_shift(path, off, s0), kind, s_free=s0 + 20.0, missing="oncoming_lane", s_turn=s0, angle=0.0, radius=off)
    if kind == "N4_uturn":
        if plen < 60:
            return None
        s_t, sg = float(rng.uniform(15.0, min(45.0, plen - 15.0))), float(rng.choice([-1.0, 1.0]))
        r = float(rng.uniform(5.0, 8.0))
        return _label(splice_turn(path, s_t, sg * 180.0, r), kind, s_free=s_t + np.pi * r, missing="u_turn", s_turn=s_t, angle=sg * 180.0, radius=r)
    raise ValueError(kind)


# ------------------------------------------------------------------ map screens (nuPlan map API, lazy imports; box env navsim2)
# A negative enters training only if the map says the route is truly impossible. All screens work on the dense negative path (ego frame) and the
# ego pose (x, y, yaw) in the map frame. Rules (user-approved 2026-10-05, results/negatives.md):
#   N2 / N4  no drivable area (lane, connector, intersection, roadblock, carpark, drivable_area, pudo) within CLEAR_M of the exit leg
#   N3       the shifted path lies on lanes of the opposite travel direction or off every drivable layer for most of its length, never on same-direction lanes
#   N1       tier A only (no lane of the roadblock has the exit); tier B is emitted separately as `lane_change_needed`
CLEAR_M = 6.0                 # N2 / N4 clearance of the exit leg to any drivable polygon
LEG_FROM, LEG_LEN = 8.0, 40.0  # exit leg = path from s_free + 8 m (clear of the road the turn leaves) over 40 m
N3_STEP = 5.0                 # N3 sampling step along the shifted path (m)
N3_SAME_MAX, N3_BAD_MIN = 0.10, 0.50  # N3 keep: share of same-direction lane points <= 0.10 and share of (opposite + off) points >= 0.50

_DRIVABLE = ("LANE", "LANE_CONNECTOR", "INTERSECTION", "ROADBLOCK", "ROADBLOCK_CONNECTOR", "CARPARK_AREA")   # vector layers; the raster adds generic drivable areas


class DrivableRaster:
    """nuPlan raster `DRIVABLE_AREA` (0.1 m, whole map; roads, intersections, carparks and generic drivable areas that have no vector object) from a memory-mapped
    .npy so that many workers share one page cache. `build(m, path)` writes the file once; the 0.5 GB arrays are never loaded per process."""

    def __init__(self, path):
        self.a = np.load(path, mmap_mode="r")
        self.t = np.load(str(path)[:-4] + "_t.npy")

    @staticmethod
    def build(m, path):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        lay = m.get_raster_map_layer(L.DRIVABLE_AREA)
        np.save(str(path)[:-4] + "_t.npy", np.array([lay.transform[0, 0], lay.transform[0, 3], lay.transform[1, 1], lay.transform[1, 3]]))
        np.save(path, np.asarray(lay.data, np.uint8))

    def px(self, xy):
        """map (x, y) -> (row, col) pixel indices"""
        return (self.t[2] * xy[:, 1] + self.t[3]).astype(int), (self.t[0] * xy[:, 0] + self.t[1]).astype(int)

    def inside(self, xy):
        r, c = self.px(np.asarray(xy, float))
        ok = (r >= 0) & (c >= 0) & (r < self.a.shape[0]) & (c < self.a.shape[1])
        out = np.zeros(len(r), bool)
        out[ok] = self.a[r[ok], c[ok]] > 0
        return out

    def clearance(self, xy, cap=15.0):
        """min distance (m) from the points to the nearest drivable pixel (capped at `cap`; 0 if a point is inside)."""
        from scipy.ndimage import distance_transform_edt
        r, c = self.px(np.asarray(xy, float))
        ok = (r >= 0) & (c >= 0) & (r < self.a.shape[0]) & (c < self.a.shape[1])        # points off the raster extent: nothing drivable there
        if not ok.any():
            return cap
        r, c = r[ok], c[ok]
        k = int(cap * abs(self.t[0]))
        r0, r1, c0, c1 = max(r.min() - k, 0), min(r.max() + k + 1, self.a.shape[0]), max(c.min() - k, 0), min(c.max() + k + 1, self.a.shape[1])
        crop = np.asarray(self.a[r0:r1, c0:c1]) > 0
        if not crop.any():
            return cap
        if crop[r - r0, c - c0].any():
            return 0.0
        return float(min(distance_transform_edt(~crop)[r - r0, c - c0].min() / abs(self.t[0]), cap))


def to_global(p, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([pose[0] + c * p[:, 0] - s * p[:, 1], pose[1] + s * p[:, 0] + c * p[:, 1]], -1)


def _near(m, centre, radius, names):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    return m.get_proximal_map_objects(Point2D(float(centre[0]), float(centre[1])), float(radius), [getattr(L, n) for n in names])


def _arc(p):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def leg_clearance(m, pose, neg, ras=None):
    """Min distance (m) from the exit leg of a turning negative to any drivable polygon (and to the drivable raster when given); NaN when the path has no leg."""
    import shapely
    d = neg["dense"]
    s = _arc(d)
    k = (s >= neg["s_free"] + LEG_FROM) & (s <= neg["s_free"] + LEG_FROM + LEG_LEN)
    if k.sum() < 4:
        return float("nan")
    g = to_global(d[k], pose)
    line = shapely.LineString(g)
    polys = [o.polygon for v in _near(m, g.mean(0), LEG_LEN / 2 + CLEAR_M + 15.0, _DRIVABLE).values() for o in v]
    c = float(shapely.distance(line, np.array(polys, dtype=object)).min()) if polys else 99.0
    return min(c, ras.clearance(g[::2])) if ras is not None else c


def oncoming_profile(m, pose, neg, ras=None):
    """N3: shares of the sampled points (after the lateral ramp) that are on lanes of the same direction as the ego path / on opposite-direction lanes /
    on a lane at another angle (cross road) / on another drivable layer without a lane / off every drivable layer. Returns a dict (shares sum to 1)."""
    import shapely
    from nuplan.common.actor_state.state_representation import Point2D
    from shapely.strtree import STRtree
    d, s = neg["dense"], _arc(neg["dense"])
    t = np.gradient(d, axis=0)
    psi = np.arctan2(t[:, 1], t[:, 0]) + pose[2]
    k = np.flatnonzero(s >= neg["s_free"])
    if len(k):
        k = k[np.unique(np.round(s[k] / N3_STEP).astype(int), return_index=True)[1]]
    if len(k) < 4:
        return None
    g = to_global(d[k], pose)
    objs = _near(m, g.mean(0), (s[k[-1]] - s[k[0]]) / 2 + 20.0, _DRIVABLE)
    pts = shapely.points(g)
    lanes = [o for q, v in objs.items() if q.name in ("LANE", "LANE_CONNECTOR") for o in v]
    best = np.full(len(k), 9.0)                                         # smallest heading difference over the lanes containing the point
    if lanes:
        pi, li = STRtree([o.polygon for o in lanes]).query(pts, predicate="within")
        for a, b in zip(pi, li):
            h = lanes[b].baseline_path.get_nearest_pose_from_position(Point2D(*g[a])).heading
            best[a] = min(best[a], abs((h - psi[k[a]] + np.pi) % (2 * np.pi) - np.pi))
    on = best < 9.0
    cls = np.full(len(k), 4)                                            # 0 same, 1 opposite, 2 cross, 3 area, 4 off
    cls[on & (best < np.radians(60))] = 0
    cls[on & (best > np.radians(120))] = 1
    cls[on & (best >= np.radians(60)) & (best <= np.radians(120))] = 2
    rest = [o.polygon for q, v in objs.items() if q.name not in ("LANE", "LANE_CONNECTOR") for o in v]
    inside = np.zeros(len(k), bool)
    if rest:
        inside[STRtree(rest).query(pts, predicate="within")[0]] = True
    if ras is not None:
        inside |= ras.inside(g)
    cls[(~on) & inside] = 3
    sh = np.bincount(cls, minlength=5) / len(k)
    return dict(same=sh[0], opp=sh[1], cross=sh[2], area=sh[3], off=sh[4], n=len(k))


def screen(m, pose, neg, tier="", ras=None):
    """Map screen of one negative. Returns (use, label, info): label 'ok' / 'lane_change_needed' / 'rej_<why>'; info = the numbers behind the decision
    (clear_m for the turning kinds incl. N1, same / opp / cross / area / off for N3)."""
    kind = neg["kind"]
    if kind == "N1_exit":
        info = dict(clear_m=leg_clearance(m, pose, neg, ras))
        return (False, "lane_change_needed", info) if tier == "B" else (True, "ok", info)   # the straight cap is applied by the sampler over the whole set
    if kind in ("N2_side", "N4_uturn"):
        c = leg_clearance(m, pose, neg, ras)
        info = dict(clear_m=c)
        if not np.isfinite(c):
            return False, "rej_no_leg", info
        return (True, "ok", info) if c > CLEAR_M else (False, "rej_drivable_near_leg", info)
    pr = oncoming_profile(m, pose, neg, ras)
    if pr is None:
        return False, "rej_short", {}
    if pr["same"] > N3_SAME_MAX:
        return False, "rej_same_direction_lane", pr
    if pr["opp"] + pr["off"] < N3_BAD_MIN:
        return False, "rej_ambiguous_area_or_cross", pr
    return True, "ok", pr
