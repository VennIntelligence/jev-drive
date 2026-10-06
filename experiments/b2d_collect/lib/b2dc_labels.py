"""Training labels of a collected B2D clip (experiments/b2d_collect), in op_parity's conventions. NumPy (+ scipy / cv2 / carla for the
drivable raster). Every function takes the clip's raw logs (ego.npz, route.npz, meta.json) so the closed loop (and a later DAgger driver)
can compute the same inputs from the same quantities.

Frames: CARLA world is left-handed (x, y, yaw clockwise in degrees). Labels are right-handed like NAVSIM's AgentInput: X = x, Y = -y,
heading = -yaw (rad); poses are the REAR AXLE (actor origin + REAR_AXLE_X along the heading, scripts/zeroshot_rigs.REAR_AXLE_X), in the
rear-axle frame of the sample's tick t0 (x forward, y left, yaw left-positive).

Per tick i (20 Hz):
  fut (8, 3)        poses at t0 + 0.5 .. 4.0 s (NaN past the clip end); op_parity's `fut`
  hist (4, 3)       poses at t0 - 1.5, -1.0, -0.5, 0 s, oldest first (before the clip start: the first tick, as WA-JEPA's buffer clamps)
  vel / acc (4, 2)  body-frame velocity / acceleration at the same times (each in its own body frame), CARLA's get_velocity / acceleration
  cmd (4,)          NAVSIM one-hot [left, straight, right, unknown]: route_command(), from the route only (re-densified with CARLA's
                    plan, junction turns labelled from geometry: Route.from_geometry)
  ego (20,)         lib/parity_adapter.ego_features(hist, vel, acc, cmd)
  turn_next / turn_dist   the next LEFT / RIGHT junction RoadOption along the route (1 / 2, 0 none) and its distance (m), for other lookaheads
  route_poly (16, 2), route_mask   route ahead as 10 m vertices from the rear axle (vertex 0 = origin), op_route_ft's nav-polyline input
  act_kappa, act_accel  curvature (1/m, left +) and longitudinal acceleration (m/s^2) the car has at t0 + LAT_DELAY (openpilot lateralDelay),
                    from the logged heading / speed (central differences over +-0.1 s); NaN below 1 m/s (curvature)
  ctl (3,), wheel   PDM-Lite's control (steer, throttle, brake) and the front-left wheel angle (deg) at t0
  pdm_path (64, 2)  PDM-Lite's remaining route (its lane-shifted steering path, 1 m spacing) in the t0 frame
  speed, target_speed
SDF (every `sdf_stride` ticks): sdf (k, 128, 96) float16 drivable signed distance (m, + inside) on op_probe's grid (x -8..56, y -24..24, 0.5 m
cell centres, lib/drivable_hinge.py); drivable = CARLA lanes of type Driving, Parking, Bidirectional (lane polygons on a 0.25 m world raster).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REAR_AXLE_X = -1.388633220199954                  # scripts/zeroshot_rigs.REAR_AXLE_X (vehicle.lincoln.mkz_2020)
DT = 0.05
FUT_T = np.arange(1, 9) * 0.5
HIST_T = np.array([-1.5, -1.0, -0.5, 0.0])
LAT_DELAY = 0.2                                   # s, openpilot lateralDelay (experiments/op_route_ft/scripts/rft.py LAT_DELAY)
TURN_DEG = 30.0                                   # junction heading change that makes a LEFT / RIGHT command
CMD_LOOKAHEAD = 30.0                              # m: a LEFT / RIGHT junction option within this distance -> left / right (see the plan)
X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 128, 96   # lib/drivable_hinge.py / op_probe grid
DRIVABLE = ("Driving", "Parking", "Bidirectional")
FINE = 0.25


# ---------------------------------------------------------------- frames
def rear_rh(loc, yaw_deg):
    """CARLA actor locations (n, 2+) and yaw (deg) -> right-handed rear-axle (X, Y) and heading (rad)."""
    loc, yaw = np.asarray(loc, float), np.radians(np.asarray(yaw_deg, float))
    x = loc[..., 0] + REAR_AXLE_X * np.cos(yaw)
    y = loc[..., 1] + REAR_AXLE_X * np.sin(yaw)
    return np.stack([x, -y], -1), -yaw


def to_local(xy, ref_xy, ref_h):
    """World right-handed points (..., 2) -> frame at (ref_xy, ref_h): x forward, y left."""
    d = np.asarray(xy, float) - ref_xy
    c, s = math.cos(ref_h), math.sin(ref_h)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def body(vec_carla, yaw_deg):
    """CARLA world vectors (n, 3) -> body frame (n, 2): forward, left."""
    v = np.asarray(vec_carla, float)
    h = -np.radians(np.asarray(yaw_deg, float))
    vx, vy = v[:, 0], -v[:, 1]
    return np.stack([np.cos(h) * vx + np.sin(h) * vy, -np.sin(h) * vx + np.cos(h) * vy], -1)


# ---------------------------------------------------------------- route
class Route:
    """The leaderboard's dense route (route.npz: CARLA xyzyaw, RoadOption ints) with arc length, in right-handed world coordinates."""

    def __init__(self, xyzyaw, option):
        P = np.asarray(xyzyaw, float)
        self.xy = np.stack([P[:, 0], -P[:, 1]], -1)
        self.opt = np.asarray(option, int)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        turn = (self.opt == 1) | (self.opt == 2)
        starts = np.where(turn & ~np.r_[False, turn[:-1]])[0]          # first point of each LEFT / RIGHT run
        self.turn_s, self.turn_o = self.s[starts], self.opt[starts]

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["xyzyaw"], z["option"])

    @classmethod
    def from_geometry(cls, xyz_carla, jid, turn_deg=TURN_DEG):
        """Route from a dense planner path (CARLA x, y, z; junction id per point, -1 outside) with the turn labels taken from geometry: each
        junction traversal is LEFT / RIGHT when the heading 5 m after its exit differs from the heading 5 m before its entry by more than
        turn_deg (left positive in the right-handed frame), else STRAIGHT. The leaderboard's own RoadOptions are not used: in the agent's
        plan they mislabel junction turns (10-route stage: a 108 deg T-junction turn as STRAIGHT, a 148 deg left turn as RIGHT)."""
        P = np.asarray(xyz_carla, float)
        jid = np.asarray(jid, int)
        xy = np.stack([P[:, 0], -P[:, 1]], -1)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        opt = np.full(len(P), 4, int)
        inj = jid >= 0
        i = 0
        while i < len(P):
            if not inj[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(P) and jid[j + 1] == jid[i]:
                j += 1
            a = max(0, int(np.searchsorted(s, s[i] - 5.0)) - 1)
            b = min(len(P) - 1, int(np.searchsorted(s, s[j] + 5.0)))

            def hd(k):
                k0, k1 = max(0, k - 2), min(len(P) - 1, k + 2)
                d = xy[k1] - xy[k0]
                return math.atan2(d[1], d[0])
            dh = math.degrees(wrap(hd(b) - hd(a)))
            opt[i:j + 1] = 1 if dh > turn_deg else 2 if dh < -turn_deg else 3
            i = j + 1
        yaw = -np.degrees(np.unwrap([0.0] + [math.atan2(*(xy[k + 1] - xy[k])[::-1]) for k in range(len(P) - 1)]))
        return cls(np.c_[P[:, :3], yaw], opt)

    def progress(self, xy, hint=None):
        """Arc length of the route point nearest to xy (searched within 30 m of `hint` when given)."""
        if hint is None:
            i = int(np.argmin(np.linalg.norm(self.xy - xy, axis=1)))
        else:
            lo, hi = np.searchsorted(self.s, [hint - 30.0, hint + 30.0])
            i = lo + int(np.argmin(np.linalg.norm(self.xy[lo:hi + 1] - xy, axis=1)))
        return float(self.s[i])

    def next_turn(self, s):
        """(option, distance) of the LEFT / RIGHT run being driven at progress s (distance <= 0: its start is behind) or of the next one
        ahead; (0, inf) if none."""
        i = min(int(np.searchsorted(self.s, s)), len(self.s) - 1)
        o = self.opt[i]
        if o in (1, 2):
            j = i
            while j > 0 and self.opt[j - 1] == o:
                j -= 1
            return int(o), float(self.s[j] - s)
        k = np.where(self.turn_s > s)[0]
        if not len(k):
            return 0, float("inf")
        return int(self.turn_o[k[0]]), float(self.turn_s[k[0]] - s)

    def poly(self, s, ref_xy, ref_h, n=16, step=10.0):
        """Route ahead from progress s as n vertices every `step` m in the ref frame (vertex 0 = ref origin) and its mask."""
        q = s + np.arange(1, n) * step
        ok = q <= self.s[-1]
        pts = np.stack([np.interp(q, self.s, self.xy[:, k]) for k in (0, 1)], -1)
        out = np.zeros((n, 2), np.float32)
        out[1:] = to_local(pts, ref_xy, ref_h)
        return out, np.r_[True, ok]


def route_command(route: Route, s, lookahead=CMD_LOOKAHEAD):
    """NAVSIM one-hot [left, straight, right, unknown] at route progress s: left / right when the next LEFT / RIGHT junction option starts
    within `lookahead` m (or is being driven), else straight. The single definition the collector labels and a closed loop must share."""
    o, d = route.next_turn(s)
    cmd = np.zeros(4, np.float32)
    cmd[0 if (o == 1 and d <= lookahead) else 2 if (o == 2 and d <= lookahead) else 1] = 1.0
    return cmd


# ---------------------------------------------------------------- per-tick labels
def tick_labels(ego: dict, route: Route) -> dict:
    """All per-tick labels of one clip (see the module docstring)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lib"))
    import parity_adapter as PA
    n = len(ego["t"])
    xy, h = rear_rh(ego["loc"][:, :2], ego["rot"][:, 1])
    h = np.unwrap(h)
    v_b, a_b = body(ego["vel"], ego["rot"][:, 1]), body(ego["acc"], ego["rot"][:, 1])
    speed = np.asarray(ego["speed"], float)
    idx = np.arange(n)
    k_f = np.rint(FUT_T / DT).astype(int)
    k_h = np.rint(HIST_T / DT).astype(int)

    def poses(i, ks, clamp):
        j = i + ks
        bad = (j < 0) | (j >= n)
        jj = np.clip(j, 0, n - 1)
        p = np.zeros((len(ks), 3))
        p[:, :2] = to_local(xy[jj], xy[i], h[i])
        p[:, 2] = h[jj] - h[i]
        if not clamp:
            p[bad] = np.nan
        return p, jj, bad

    fut = np.zeros((n, 8, 3), np.float32)
    hist = np.zeros((n, 4, 3), np.float32)
    vel = np.zeros((n, 4, 2), np.float32)
    acc = np.zeros((n, 4, 2), np.float32)
    cmd = np.zeros((n, 4), np.float32)
    hist_ok = np.zeros(n, bool)
    turn_next, turn_dist = np.zeros(n, np.int8), np.zeros(n, np.float32)
    rpoly, rmask = np.zeros((n, 16, 2), np.float32), np.zeros((n, 16), bool)
    prog = np.zeros(n, np.float32)
    s_hint = None
    for i in idx:
        fut[i] = poses(i, k_f, False)[0]
        hist[i], jj, bad = poses(i, k_h, True)
        hist_ok[i] = not bad.any()
        vel[i], acc[i] = v_b[jj], a_b[jj]
        s = route.progress(xy[i], s_hint)
        s_hint = s
        prog[i] = s
        cmd[i] = route_command(route, s)
        o, d = route.next_turn(s)
        turn_next[i], turn_dist[i] = o, d
        rpoly[i], rmask[i] = route.poly(s, xy[i], h[i])
    ego_f = PA.ego_features(hist, vel, acc, cmd)
    # actions at t0 + LAT_DELAY: heading rate / speed (central differences over +-2 ticks)
    hr = np.gradient(h, DT)
    hr = np.convolve(np.pad(hr, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dl = int(round(LAT_DELAY / DT))
    jd = np.minimum(idx + dl, n - 1)
    kappa = np.where(speed[jd] >= 1.0, hr[jd] / np.maximum(speed[jd], 1e-3), np.nan).astype(np.float32)
    a_lon = np.convolve(np.pad(a_b[:, 0], 2, mode="edge"), np.ones(5) / 5, mode="valid")[jd].astype(np.float32)
    pdm = np.full((n, 64, 2), np.nan, np.float32)
    ah = np.asarray(ego["ahead"], float)
    ok = np.isfinite(ah[..., 0])
    w = np.stack([ah[..., 0], -ah[..., 1]], -1)
    for i in idx:
        if ok[i].any():
            pdm[i, ok[i]] = to_local(w[i, ok[i]], xy[i], h[i])
    return dict(fut=fut, hist=hist, hist_ok=hist_ok, vel=vel, acc=acc, cmd=cmd, ego=ego_f, turn_next=turn_next, turn_dist=turn_dist,
                route_poly=rpoly, route_mask=rmask, progress=prog, act_kappa=kappa, act_accel=a_lon, ctl=np.asarray(ego["ctl_expert"], np.float32),
                wheel=np.asarray(ego["wheel"], np.float32), pdm_path=pdm, speed=speed.astype(np.float32),
                target_speed=np.asarray(ego["target_speed"], np.float32), xy_world=xy.astype(np.float64), heading=h.astype(np.float64),
                fut_ok=np.isfinite(fut[:, -1, 0]))


# ---------------------------------------------------------------- drivable raster
_lanes = {}


def lane_points(town: str, xodr: str, step=0.5):
    """All lane-centre samples of drivable lane types: (x, y CARLA, yaw deg, width) arrays, cached per process."""
    if town not in _lanes:
        import carla
        m = carla.Map(town, Path(xodr).read_text())
        types = {getattr(carla.LaneType, t) for t in DRIVABLE}
        rows = [(w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw, w.lane_width)
                for w in m.generate_waypoints(step) if w.lane_type in types]
        _lanes[town] = np.asarray(rows, np.float64)
    return _lanes[town]


def world_sdf(lanes: np.ndarray, lo, hi, step=0.5):
    """Signed distance (m, + inside) on a FINE world raster over [lo, hi] (CARLA x, y) from lane-centre samples stamped as rectangles of
    their lane width and 1.6 x the sample step (so tight curves leave no gap). Returns (sdf (H, W) float32 with row = x, col = y, lo)."""
    import cv2
    from scipy.ndimage import distance_transform_edt
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    H, W = (np.ceil((hi - lo) / FINE).astype(int) + 1)
    sel = (lanes[:, 0] > lo[0] - 10) & (lanes[:, 0] < hi[0] + 10) & (lanes[:, 1] > lo[1] - 10) & (lanes[:, 1] < hi[1] + 10)
    L = lanes[sel]
    yaw = np.radians(L[:, 2])
    f = np.stack([np.cos(yaw), np.sin(yaw)], -1) * (0.8 * step)
    r = np.stack([-np.sin(yaw), np.cos(yaw)], -1) * (L[:, 3:4] / 2)
    c = L[:, :2]
    quads = np.stack([c + f + r, c + f - r, c - f - r, c - f + r], 1)                     # (n, 4, 2) CARLA x, y
    pix = np.round((quads[..., ::-1] - lo[::-1]) / FINE * 16).astype(np.int32)         # (col = y, row = x), 4 fractional bits
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, list(pix), 1, lineType=cv2.LINE_8, shift=4)
    inside = mask.astype(bool)
    d_in = distance_transform_edt(inside) * FINE
    d_out = distance_transform_edt(~inside) * FINE
    return np.where(inside, d_in, -d_out).astype(np.float32), lo


def grid_world(xy_rh, h):
    """Cell centres of op_probe's grid for a rear-axle pose (right-handed) -> CARLA world x, y (NH, NW, 2)."""
    gx = X0 + (np.arange(NH) + 0.5) * RES
    gy = Y0 + (np.arange(NW) + 0.5) * RES
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    c, s = math.cos(h), math.sin(h)
    X = xy_rh[0] + c * GX - s * GY
    Y = xy_rh[1] + s * GX + c * GY
    return np.stack([X, -Y], -1)


def sample_sdf(sdf, lo, pts):
    """Bilinear sample of the world raster at CARLA points (..., 2)."""
    from scipy.ndimage import map_coordinates
    r = (pts[..., 0] - lo[0]) / FINE
    c = (pts[..., 1] - lo[1]) / FINE
    return map_coordinates(sdf, [r.ravel(), c.ravel()], order=1, mode="nearest").reshape(pts.shape[:-1])


def clip_sdf(lanes, xy_rh, heading, ticks):
    """SDF labels (len(ticks), NH, NW) float16, clipped to +-20 m, for the given ticks of one clip."""
    carla_xy = np.stack([xy_rh[:, 0], -xy_rh[:, 1]], -1)
    lo, hi = carla_xy.min(0) - 75.0, carla_xy.max(0) + 75.0
    sdf, lo = world_sdf(lanes, lo, hi)
    out = np.zeros((len(ticks), NH, NW), np.float16)
    for k, i in enumerate(ticks):
        out[k] = np.clip(sample_sdf(sdf, lo, grid_world(xy_rh[i], heading[i])), -20, 20)
    return out


def footprint(meta: dict) -> np.ndarray:
    """Hero footprint corners about the rear axle (x forward, y left), lib/drivable_hinge.CORNERS layout."""
    e, o = meta["hero"]["extent"], meta["hero"]["offset"]
    front, rear = o[0] + e[0] - REAR_AXLE_X, o[0] - e[0] - REAR_AXLE_X
    return np.array([[front, e[1]], [front, -e[1]], [rear, e[1]], [rear, -e[1]]])


_maps = {}


def junction_ids(town: str, xodr: str, xyz) -> np.ndarray:
    """Junction id (or -1) of the lane waypoint nearest to each CARLA point (offline carla.Map, cached per process)."""
    import carla
    if town not in _maps:
        _maps[town] = carla.Map(town, Path(xodr).read_text())
    m = _maps[town]
    out = np.full(len(xyz), -1, int)
    for k, (x, y, z) in enumerate(np.asarray(xyz, float)):
        w = m.get_waypoint(carla.Location(x=x, y=y, z=z))
        if w is not None and w.is_junction:
            out[k] = w.get_junction().id
    return out


def load_clip(clip: Path, xodr=None):
    """(ego, Route, meta). The route is the agent's plan (route.npz, the leaderboard's ~1 m interpolation); with `xodr` (town -> OpenDrive
    path) its junction turns are labelled from geometry (Route.from_geometry) instead of the plan's RoadOptions."""
    clip = Path(clip)
    ego = dict(np.load(clip / "ego.npz"))
    meta = json.loads((clip / "meta.json").read_text())
    if xodr is None:
        return ego, Route.load(clip / "route.npz"), meta
    kp = np.load(clip / "route.npz")["xyzyaw"][:, :3]
    return ego, Route.from_geometry(kp, junction_ids(meta["town"], str(xodr(meta["town"])), kp)), meta
