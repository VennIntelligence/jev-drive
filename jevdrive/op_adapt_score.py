"""op-adapt round 2, package S: the rule-based plan scorer S_jev (todos/2026-09-29-op-adapt-r2-prereg.md, section 2.1).

    S_jev(tau) = NC * DAC * DDC * (5 P + 5 TTC + 2 C) / 12

Every candidate plan tau of a slot s is rolled out as rear-axle positions on a 10 Hz grid over H = 4 s (no LQR tracking)
and checked against the slot's scored actor set A(s) (their recorded futures: non-reactive) and the map:
  NC   no at-fault collision = WL-2's cg: the ego box overlaps an actor box, or the in-lane gap of jevdrive.wl._gap_front
       (|y| <= 1.75 m ahead, front bumper to the actor's reference point) is < 2 m; both only while the ego moves
       (>= 0.5 m/s), and an overlap only counts when it touches the front half of the ego box
  DAC  all four footprint corners on the drivable surface at every step
  DDC  NAVSIM v2 driving-direction compliance, item by item as navsim's pdm_scorer: the box centre's displacement while
       it is not inside an on-route lane (and not in an intersection), summed over a 1 s window (11 poses), max over the
       horizon: < 2 m -> 1, < 6 m -> 0.5, else 0. On-route lanes: NAVSIM = the route's lanes (devkit, exact); CARLA and
       nuScenes (no roadblock route) = the lanes whose travel direction agrees with the ego's heading (the tangential form
       the prereg wrote); bidirectional lanes agree with both
  P    progress: box-centre projection onto the reference path at the horizon end, over max(5 m, best safe candidate)
       with PDM's rule "nobody reaches 5 m -> all 1"; safe = NC * DAC * DDC = 1; yield exemption: when op or hold fails NC
       because of a pedestrian, a cyclist or a moving (>= 0.5 m/s) vehicle, only candidates within 1 m of the reference
       path are safe candidates
  TTC  PDM's time-to-collision: the ego box pushed along its heading at its speed for 0 / 0.3 / 0.6 / 0.9 s against the
       actor boxes at that time; counts when the actor is ahead (< 30 deg) or the ego is in several lanes / off-road / in
       an intersection and the actor is not behind (> 150 deg); a track that is hit but not counted is ignored afterwards
  C    PDM comfort: navsim's pdm_comfort_metrics.ego_is_comfortable (Savitzky-Golay estimates and bounds) on the rollout
Top(s) = {k : S_k >= max S - 0.1}; a slot with max S = 0 or less than 3 s of usable horizon is not valid.

Frames: every geometry here is right-handed (CARLA's world is mirrored, y -> -y, yaw -> -yaw, by the slot builders). A slot
lives in its own ego frame (rear axle, x forward, y left); the map is queried in world coordinates. openpilot plans are
in openpilot's calibrated frame (x forward, y right, origin at the camera), `cam_x` ahead of the rear axle.

Maps (`TileMap`): 0.2 m raster in 256 x 256 tiles, only tiles near the routes; `PolyMap`: NAVSIM's metric-cache geometry.
Consumers: `load_scores(domain)`, `score_plans(...)`; the slot builders, checks and production are in
jevdrive/op_adapt_score_data.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT, H, NT = 0.1, 4.0, 41
TS = DT * np.arange(NT)
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
N21 = 21                                                   # T_IDXS points <= 4 s (0 ... 3.906 s)
CANDS = ("op", "op_L", "op_R", "op_slow", "op_stop", "hold", "brake_hard", "brake_mild", "shift_L", "shift_R",
         "shift_L_slow", "shift_R_slow", "nudge_L")
CANDS_OFF = CANDS + ("rej",)
DELTA, H_MIN = 0.1, 3.0
GAP_MIN, LANE_HALF, SELF_EXCL, V_FAULT = 2.0, 1.75, 2.0, 0.5          # NC (jevdrive.wl._gap_front / cg)
PROG_MIN, MOVING_V, KEEP_LAT = 5.0, 0.5, 1.0                           # P
DDC_WIN, DDC_OK, DDC_BAD = 10, 2.0, 6.0                                # int(1.0 s / 0.1 s) poses back, thresholds (m)
TTC_STEPS, STOP_V, AHEAD, BEHIND = (0, 3, 6, 9), 5e-3, math.radians(30), math.radians(150)
MAX_LON_ACC, MIN_LON_ACC, MAX_LAT_ACC = 2.40, -4.05, 4.89               # navsim pdm_comfort_metrics
MAX_JERK, MAX_LON_JERK, MAX_YAW_ACC, MAX_YAW_RATE = 8.37, 4.13, 1.93, 0.95
REJ_S = 3.0
HEAD_V = 0.2                                               # heading held below this speed (the devkit LQR's stopping velocity)
VEH, PED, CYC, STATIC = 0, 1, 2, 3
RES, TILE = 0.2, 256
TM = RES * TILE
F_DRIVE, F_LANE, F_JUNC, F_BIDIR = 1, 2, 4, 8


@dataclass(frozen=True)
class Ego:
    """hl / hw: half length / width; rc: rear axle -> box centre; gap_o: the _gap_front origin ahead of the rear axle
    (CARLA: the vehicle transform); gap_f: that origin -> front bumper, as _gap_front subtracts it."""
    hl: float
    hw: float
    rc: float
    gap_o: float
    gap_f: float


EGO = {"carla": Ego(2.446, 0.918, 1.383, 1.389, 2.4),       # lincoln.mkz_2020 (runs' meta.json), WL's 2.4 m bumper
       "nus": Ego(2.042, 0.865, 1.38, 1.38, 2.042),         # Renault Zoe, nuScenes ego_pose = rear axle
       "nav": Ego(2.588, 1.1485, 1.461, 1.461, 2.588)}      # nuPlan get_pacifica_parameters()


@dataclass
class Actors:
    """Scored actors on the slot's 10 Hz grid, ego frame. c: (NT, N, 2) box centres; h: (NT, N) headings; hl / hw: (N,);
    ref: (NT, N, 2) the gap reference point (CARLA: the actor location, else the box centre); valid: (NT, N);
    speed: (NT, N); kind: (N,) VEH / PED / CYC / STATIC; vis: (N,) in the camera-visible set A(s)."""
    c: np.ndarray
    h: np.ndarray
    hl: np.ndarray
    hw: np.ndarray
    ref: np.ndarray
    valid: np.ndarray
    speed: np.ndarray
    kind: np.ndarray
    vis: np.ndarray

    def subset(self, m: np.ndarray) -> "Actors":
        return Actors(self.c[:, m], self.h[:, m], self.hl[m], self.hw[m], self.ref[:, m], self.valid[:, m],
                      self.speed[:, m], self.kind[m], self.vis[m])

    @staticmethod
    def empty() -> "Actors":
        z = np.zeros((NT, 0))
        return Actors(np.zeros((NT, 0, 2)), z, np.zeros(0), np.zeros(0), np.zeros((NT, 0, 2)), z.astype(bool), z,
                      np.zeros(0, int), np.zeros(0, bool))


@dataclass
class Slot:
    """One scene s. pose: (x, y, yaw) of the rear axle in the (right-handed) world; n: usable poses (<= NT, t = 0 incl.);
    ref: (M, 2) progress reference path in the ego frame (CARLA route / real: op's path / NAVSIM: PDM centre line)."""
    ego: Ego
    pose: tuple
    v0: float
    n: int
    actors: Actors
    ref: np.ndarray
    mapq: object
    route: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


# ================================================================ geometry

def obb_overlap(ca, ha, la, wa, cb, hb, lb, wb) -> np.ndarray:
    """Strict overlap of oriented boxes (centres (..., 2), headings, half lengths, half widths; broadcast), by the
    separating-axis test on the four box axes."""
    d = np.asarray(cb) - np.asarray(ca)
    dx, dy = d[..., 0], d[..., 1]
    ca_, sa_ = np.cos(ha), np.sin(ha)
    cb_, sb_ = np.cos(hb), np.sin(hb)
    c, s = np.abs(np.cos(hb - ha)), np.abs(np.sin(hb - ha))
    sep = (np.abs(dx * ca_ + dy * sa_) > la + lb * c + wb * s) | (np.abs(-dx * sa_ + dy * ca_) > wa + lb * s + wb * c)
    sep |= (np.abs(dx * cb_ + dy * sb_) > lb + la * c + wa * s) | (np.abs(-dx * sb_ + dy * cb_) > wb + la * s + wa * c)
    return ~sep


def box_corners(c, h, hl, hw) -> np.ndarray:
    """(..., 4, 2) corners in navsim's BBCoordsIndex order: front left, rear left, rear right, front right."""
    u = np.stack([np.cos(h), np.sin(h)], -1)[..., None, :]
    v = np.stack([-np.sin(h), np.cos(h)], -1)[..., None, :]
    sl = np.array([1.0, -1.0, -1.0, 1.0])[:, None]
    sw = np.array([1.0, 1.0, -1.0, -1.0])[:, None]
    return np.asarray(c)[..., None, :] + sl * hl * u + sw * hw * v


def rollout(p: np.ndarray, v0: float | np.ndarray | None = None) -> dict:
    """Ego states of rear-axle positions p (..., T, 2) on the 0.1 s grid: heading from the path tangent (held while the
    car moves slower than 0.2 m/s), speed from the arc length, acceleration, yaw rate and yaw acceleration."""
    p = np.asarray(p, float)
    g = np.gradient(p, DT, axis=-2)
    mv = np.hypot(g[..., 0], g[..., 1]) > HEAD_V         # standing jitter (a logged stop) must not turn the car around
    head = np.arctan2(g[..., 1], g[..., 0])
    idx = np.where(mv, np.arange(p.shape[-2]), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    head = np.where(mv.any(-1, keepdims=True), np.take_along_axis(head, idx, -1), 0.0)
    first = np.argmax(mv, -1)[..., None]                   # leading standing poses take the first moving heading
    head = np.where(np.arange(p.shape[-2]) < first, np.take_along_axis(head, first, -1), head)
    head = np.unwrap(head, axis=-1)
    s = np.concatenate([np.zeros(p.shape[:-2] + (1,)), np.cumsum(np.hypot(*np.moveaxis(np.diff(p, axis=-2), -1, 0)), -1)], -1)
    v = np.gradient(s, DT, axis=-1)
    if v0 is not None:
        v[..., 0] = v0
    a = np.gradient(v, DT, axis=-1)
    w = np.gradient(head, DT, axis=-1)
    return {"p": p, "h": head, "v": v, "a": a, "w": w, "al": np.gradient(w, DT, axis=-1), "s": s}


def project(pts: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(arc length of the nearest point on the polyline, distance to it) for points (..., 2)."""
    path = np.asarray(path, float)
    a, b = path[:-1], path[1:]
    ab = b - a
    L = np.hypot(ab[:, 0], ab[:, 1])
    s0 = np.r_[0.0, np.cumsum(L)]
    q = np.asarray(pts, float)[..., None, :]
    t = np.clip(((q - a) * ab).sum(-1) / np.maximum(L ** 2, 1e-12), 0, 1)
    d = np.hypot(*np.moveaxis(q - (a + t[..., None] * ab), -1, 0))
    j = np.argmin(d, -1)
    return np.take_along_axis(s0[:-1] + t * L, j[..., None], -1)[..., 0], np.take_along_axis(d, j[..., None], -1)[..., 0]


def to_world(xy, pose) -> np.ndarray:
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.asarray(xy, float)
    return np.stack([x0 + xy[..., 0] * c - xy[..., 1] * s, y0 + xy[..., 0] * s + xy[..., 1] * c], -1)


def to_ego(xy, pose) -> np.ndarray:
    x0, y0, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, float) - (x0, y0)
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], -1)


# ================================================================ maps

class TileMap:
    """Tiled raster (module docstring). flags (n, 256, 256) uint8 [row = y pixel, col = x pixel]; lane (n, 2, 256, 256)
    int32 lane ids (0 none); head (n, 2, 256, 256) uint8 lane travel direction * 255 / 2 pi (255 none)."""

    def __init__(self, ij, flags, lane, head):
        self.ij, self.flags, self.lane, self.head = np.asarray(ij, np.int64), flags, lane, head
        k = self._key(self.ij[:, 0], self.ij[:, 1])
        self.order = np.argsort(k)
        self.keys = k[self.order]

    @staticmethod
    def _key(i, j):
        return (np.asarray(i, np.int64) + (1 << 20)) << 22 | (np.asarray(j, np.int64) + (1 << 20))

    @classmethod
    def load(cls, path) -> "TileMap":
        """<name>.npz, memory-mapped through an uncompressed sibling dir <name>/ (made on first use) so that pool
        workers share one copy in the page cache."""
        import os
        import shutil
        from pathlib import Path
        path = Path(path)
        d = path.with_suffix("")
        if not (d / "head.npy").exists():
            tmp = d.with_name(f"{d.name}.tmp{os.getpid()}_{np.random.randint(1 << 30)}")
            try:
                tmp.mkdir(parents=True)
                with np.load(path) as z:
                    for k in ("ij", "flags", "lane", "head"):
                        np.save(tmp / f"{k}.npy", z[k])
                tmp.rename(d)
            except OSError:                                     # another worker won the race
                shutil.rmtree(tmp, ignore_errors=True)
        return cls(*(np.load(d / f"{k}.npy", mmap_mode="r") for k in ("ij", "flags", "lane", "head")))

    def lookup(self, xy):
        """flags (...), lane ids (..., 2), heading (..., 2) rad (nan none) at world points (..., 2)."""
        xy = np.asarray(xy, float)
        i, j = np.floor(xy[..., 0] / TM).astype(np.int64), np.floor(xy[..., 1] / TM).astype(np.int64)
        k = self._key(i, j)
        pos = np.clip(np.searchsorted(self.keys, k), 0, max(len(self.keys) - 1, 0))
        ok = (self.keys[pos] == k) if len(self.keys) else np.zeros(k.shape, bool)
        t = self.order[pos] if len(self.keys) else pos
        px = np.clip(((xy[..., 0] - i * TM) / RES).astype(np.int64), 0, TILE - 1)
        py = np.clip(((xy[..., 1] - j * TM) / RES).astype(np.int64), 0, TILE - 1)
        fl = np.where(ok, self.flags[t, py, px], 0) if len(self.keys) else np.zeros(k.shape, np.uint8)
        ln = np.where(ok[..., None], np.stack([self.lane[t, 0, py, px], self.lane[t, 1, py, px]], -1), 0) if len(self.keys) \
            else np.zeros(k.shape + (2,), np.int32)
        hd = np.stack([self.head[t, 0, py, px], self.head[t, 1, py, px]], -1) if len(self.keys) else np.full(k.shape + (2,), 255)
        hd = np.where(ok[..., None] & (hd != 255), hd * (2 * np.pi / 255), np.nan)
        return fl, ln, hd

    def areas(self, corners, centre, rear, heading) -> dict:
        """Per pose: off-road (a corner off the drivable surface), in several lanes (navsim's EgoAreaIndex rule on lane
        ids), centre / rear axle in a junction, centre not in a lane agreeing with the heading (oncoming)."""
        fc, lc, _ = self.lookup(corners)                   # (..., 4), (..., 4, 2)
        f0, _, hd = self.lookup(centre)
        fr, _, _ = self.lookup(rear)
        ids = lc.reshape(lc.shape[:-2] + (8,))
        nz = ids != 0
        mx = np.where(nz, ids, -1).max(-1)
        mn = np.where(nz, ids, np.iinfo(np.int32).max).min(-1)
        several = nz.any(-1) & (mx != mn)
        full = np.zeros(several.shape, bool)
        for s in range(2):                                 # a lane holding all four corners must hold corner 0
            idj = lc[..., 0, s]
            full |= (idj != 0) & (lc[..., 1:, :] == idj[..., None, None]).any(-1).all(-1)
        agree = (np.cos(hd - np.asarray(heading)[..., None]) > 0).any(-1) | ((f0 & F_BIDIR) > 0)
        return {"offroad": ((fc & F_DRIVE) == 0).any(-1), "multi": several & ~full, "junc_c": (f0 & F_JUNC) > 0,
                "junc_r": (fr & F_JUNC) > 0, "oncoming": ~agree}


class PolyMap:
    """NAVSIM metric-cache geometry (shapely): drivable = ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK_AREA
    polygons, lanes = LANE / LANE_CONNECTOR polygons (route: on-route), intersection = INTERSECTION polygons."""

    def __init__(self, drivable, lanes, route_mask, intersection):
        import shapely
        self.drive = shapely.union_all(drivable)
        self.lanes = list(lanes)
        self.route = shapely.union_all([g for g, r in zip(self.lanes, route_mask) if r]) if any(route_mask) else None
        self.inter = shapely.union_all(intersection) if len(intersection) else None
        for g in [self.drive, self.route, self.inter] + self.lanes:
            if g is not None:
                shapely.prepare(g)

    @staticmethod
    def _in(g, xy) -> np.ndarray:
        import shapely
        xy = np.asarray(xy, float)
        return np.zeros(xy.shape[:-1], bool) if g is None else shapely.contains_xy(g, xy[..., 0], xy[..., 1])

    def areas(self, corners, centre, rear, heading) -> dict:
        inl = np.stack([self._in(g, corners) for g in self.lanes], -1) if self.lanes else np.zeros(corners.shape[:-1] + (0,), bool)
        cnt = inl.sum(-2)                                   # (..., n_lanes): corners per lane
        several = (cnt > 0).sum(-1) > 1
        full = (cnt == 4).any(-1)
        return {"offroad": ~self._in(self.drive, corners).all(-1), "multi": several & ~full,
                "junc_c": self._in(self.inter, centre), "junc_r": self._in(self.inter, rear),
                "oncoming": ~self._in(self.route, centre)}


# ================================================================ metrics

def _geom(st: dict, ego: Ego, n: int) -> dict:
    p, h = st["p"][..., :n, :], st["h"][..., :n]
    u = np.stack([np.cos(h), np.sin(h)], -1)
    c = p + ego.rc * u
    return {"p": p, "h": h, "u": u, "c": c, "v": st["v"][..., :n], "corners": box_corners(c, h, ego.hl, ego.hw)}


def nc(g: dict, ego: Ego, A: Actors, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(NC (K,), failing actors (K, N), failing actors that earn the yield exemption (K, N)): an at-fault overlap with the front half of the ego box, or the _gap_front gap
    < 2 m, at a pose where the ego moves >= 0.5 m/s."""
    if A.c.shape[1] == 0:
        z = np.zeros((g["p"].shape[0], 0), bool)
        return np.ones(g["p"].shape[0], bool), z, z
    h, u, mov = g["h"][..., None], g["u"], (g["v"] >= V_FAULT)[..., None]
    val = A.valid[None, :n]
    cf = (g["p"] + (ego.rc + ego.hl / 2) * u)[:, :, None]
    ov = obb_overlap(cf, h, ego.hl / 2, ego.hw, A.c[None, :n], A.h[None, :n], A.hl, A.hw)
    rel = A.ref[None, :n] - (g["p"] + ego.gap_o * u)[:, :, None]
    x = rel[..., 0] * u[:, :, None, 0] + rel[..., 1] * u[:, :, None, 1]
    y = -rel[..., 0] * u[:, :, None, 1] + rel[..., 1] * u[:, :, None, 0]
    gap = (x > 0) & (np.abs(y) <= LANE_HALF) & (np.hypot(x, y) > SELF_EXCL) & (x - ego.gap_f < GAP_MIN)
    fail = (ov | gap) & val & mov                           # (K, n, N)
    kind = A.kind[None, None]
    moving = ((kind == PED) | (kind == CYC)) | ((kind == VEH) & (A.speed[None, :n] >= MOVING_V))
    return ~fail.any((1, 2)), fail.any(1), (fail & moving).any(1)


def ttc(g: dict, ego: Ego, A: Actors, n: int, ar: dict) -> np.ndarray:
    """PDM's TTC (navsim pdm_scorer._calculate_ttc) on the rollout; tracks overlapping the ego at t = 0 start ignored."""
    K = g["p"].shape[0]
    out = np.ones(K, bool)
    if A.c.shape[1] == 0 or n <= max(TTC_STEPS):
        return out
    ign = obb_overlap(g["c"][:, 0, None], g["h"][:, 0, None], ego.hl, ego.hw, A.c[None, 0], A.h[None, 0], A.hl, A.hw) & A.valid[None, 0]
    area = ar["multi"] | ar["offroad"] | ar["junc_r"]
    for ti in range(n - max(TTC_STEPS)):
        run = g["v"][:, ti] >= STOP_V
        for dt in TTC_STEPS:
            tc = ti + dt
            cc = g["c"][:, ti] + g["v"][:, ti, None] * dt * DT * g["u"][:, ti]
            hit = obb_overlap(cc[:, None], g["h"][:, ti, None], ego.hl, ego.hw, A.c[None, tc], A.h[None, tc], A.hl, A.hw)
            hit &= A.valid[None, tc] & ~ign & run[:, None]
            if not hit.any():
                continue
            d = A.c[None, tc] - g["p"][:, ti, None]
            ang = np.abs((np.arctan2(d[..., 1], d[..., 0]) - g["h"][:, ti, None] + np.pi) % (2 * np.pi) - np.pi)
            fault = (ang < AHEAD) | (area[:, ti, None] & (ang <= BEHIND))
            out &= ~(hit & fault).any(1)
            ign |= hit & ~fault
    return out


def _savgol(y, window, poly, deriv=0):
    from scipy.signal import savgol_filter
    return np.round(savgol_filter(y, window_length=min(window, y.shape[-1]), polyorder=poly, deriv=deriv, delta=DT, axis=-1), 8)


def comfort(st: dict, ego: Ego, n: int) -> np.ndarray:
    """navsim ego_is_comfortable on the rollout (rear-axle body frame: vx = v, vy = 0, ax = a, ay = v w; centre-shifted
    x / y accelerations for the longitudinal / lateral bounds, as state_array_to_center_state_array)."""
    v, a, w, al, h = (st[k][..., :n] for k in ("v", "a", "w", "al", "h"))
    if n < 5:                                              # too short for the filters (such a slot is not valid anyway)
        return np.ones(v.shape[:-1], bool)
    ay = v * w
    lon, lat = _savgol(a - w ** 2 * ego.rc, 8, 2), _savgol(ay + al * ego.rc, 8, 2)
    mag = _savgol(np.hypot(a, ay), 8, 2)
    ok = ((lon > MIN_LON_ACC) & (lon < MAX_LON_ACC)).all(-1) & (np.abs(lat) < MAX_LAT_ACC).all(-1)
    ok &= (np.abs(_savgol(mag, 15, 2, 1)) < MAX_JERK).all(-1) & (np.abs(_savgol(lon, 15, 2, 1)) < MAX_LON_JERK).all(-1)
    ok &= (np.abs(_savgol(h, 5, 3, 2)) < MAX_YAW_ACC).all(-1) & (np.abs(_savgol(h, 5, 2, 1)) < MAX_YAW_RATE).all(-1)
    return ok


def ddc(g: dict, ar: dict) -> np.ndarray:
    """navsim _calculate_driving_direction_compliance: oncoming box-centre progress over a 1 s window, three levels."""
    step = np.zeros(g["c"].shape[:-1])
    step[..., 1:] = np.hypot(*np.moveaxis(np.diff(g["c"], axis=-2), -1, 0))
    step = np.where(ar["oncoming"] & ~ar["junc_c"], step, 0.0)
    cs = np.concatenate([np.zeros(step.shape[:-1] + (1,)), np.cumsum(step, -1)], -1)
    i = np.arange(step.shape[-1])
    win = cs[..., i + 1] - cs[..., np.maximum(0, i - DDC_WIN)]
    D = win.max(-1)
    return np.where(D < DDC_OK, 1.0, np.where(D < DDC_BAD, 0.5, 0.0)), D


def raw_metrics(slot: Slot, p: np.ndarray, actors: dict | None = None) -> dict:
    """Every per-candidate term for rear-axle rollouts p (K, NT, 2) in the slot's ego frame. actors: {"vis": mask, ...}
    extra actor subsets to score NC / TTC with (default: the visible set only under the key "")."""
    n, ego = slot.n, slot.ego
    st = rollout(p)                                        # v, a from the candidate's own arc-length curve (§2.1)
    g = _geom(st, ego, n)
    yaw = slot.pose[2]
    ar = slot.mapq.areas(to_world(g["corners"], slot.pose), to_world(g["c"], slot.pose), to_world(g["p"], slot.pose), g["h"] + yaw)
    dd, D = ddc(g, ar)
    s_ref, _ = project(g["c"][:, [0, -1]], slot.ref)
    _, lat = project(g["p"], slot.ref)
    out = {"DAC": ~ar["offroad"].any(-1), "DDC": dd, "D_onc": D, "C": comfort(st, ego, n),
           "prog": np.maximum(s_ref[:, 1] - s_ref[:, 0], 0.0), "lat": lat.max(-1)}
    for tag, m in (actors or {"": slot.actors.vis}).items():
        A = slot.actors.subset(m)
        out["NC" + tag], out["fail" + tag], out["failmov" + tag] = nc(g, ego, A, n)
        out["TTC" + tag] = ttc(g, ego, A, n, ar)
    return out


def finalize(m: dict, names, tag: str = "", keep=("op", "hold"), prog_norm: float | None = None) -> dict:
    """P, S, Top, valid from raw metrics of one slot's candidate set. prog_norm given: the stored normaliser (score_plans)."""
    safe = m["NC" + tag] & m["DAC"] & (m["DDC"] == 1.0)
    idx = [names.index(k) for k in keep if k in names]
    exempt = bool(m["failmov" + tag][idx].any()) if prog_norm is None else False
    if prog_norm is None:
        cand = safe & (m["lat"] <= KEEP_LAT) if exempt else safe
        best = float(m["prog"][cand].max()) if cand.any() else 0.0
        prog_norm = best if best > PROG_MIN else 0.0
    P = np.minimum(1.0, m["prog"] / prog_norm) if prog_norm > 0 else np.ones_like(m["prog"])
    S = m["NC" + tag] * m["DAC"] * m["DDC"] * (5 * P + 5 * m["TTC" + tag] + 2 * m["C"]) / 12
    top = S >= S.max() - DELTA
    return {"P": P, "S": S, "top": top, "exempt": exempt, "prog_norm": prog_norm, "safe": safe}


def score_slot(slot: Slot, p: np.ndarray, names, all_actors: bool = True) -> dict:
    """Full S_jev of one slot's candidate set: raw metrics + P / S / Top, for the visible set ("") and all actors ("_all")."""
    subsets = {"": slot.actors.vis}
    if all_actors:
        subsets["_all"] = np.ones(len(slot.actors.vis), bool)
    m = raw_metrics(slot, p, subsets)
    out = {**m, **finalize(m, list(names))}
    if all_actors:
        f = finalize(m, list(names), "_all")
        out.update({"S_all": f["S"], "top_all": f["top"]})
    out["valid"] = bool(out["S"].max() > 0) and slot.n - 1 >= round(H_MIN / DT)
    out["h"] = (slot.n - 1) * DT
    return out


# ================================================================ candidates

def op_to_rear(xy, cam_x: float, yaw=None) -> np.ndarray:
    """openpilot plan positions (x fwd, y right, the camera's track) -> rear axle (x fwd, y left): rear = d + p - R(psi) d
    with d = (cam_x, 0) and psi = -yaw (navsim_zs.openpilot_to_navsim without the camera's lateral offset)."""
    xy = np.asarray(xy, float)
    psi = np.zeros(xy.shape[:-1]) if yaw is None else -np.asarray(yaw, float)
    return np.stack([xy[..., 0] + cam_x - cam_x * np.cos(psi), -xy[..., 1] - cam_x * np.sin(psi)], -1)


def rear_to_op(xy, cam_x: float, psi=None) -> np.ndarray:
    """Inverse of op_to_rear; psi: the rear-axle heading (left positive)."""
    xy = np.asarray(xy, float)
    psi = np.zeros(xy.shape[:-1]) if psi is None else np.asarray(psi, float)
    return np.stack([xy[..., 0] - cam_x + cam_x * np.cos(psi), -(xy[..., 1] + cam_x * np.sin(psi))], -1)


def plan_at(plan: np.ndarray, t: np.ndarray, cam_x: float) -> np.ndarray:
    """openpilot plan (33, >= 2; channel 11 = yaw when present) positions at times t, rear-axle ego frame."""
    plan = np.asarray(plan, float)
    yaw = np.interp(t, T_IDXS, plan[:, 11]) if plan.shape[1] >= 12 else None
    return op_to_rear(np.stack([np.interp(t, T_IDXS, plan[:, 0]), np.interp(t, T_IDXS, plan[:, 1])], -1), cam_x, yaw)


def rej_path(op_rear: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """`rej`: from the (offset) start, a 3 s cosine merge onto the centre line, then along it; the arc-length profile is
    op's. op_rear: (NT, 2) op's rollout; centre: (M, 2) centre line in the same ego frame."""
    from .wl_traj import _along
    s_op = np.r_[0.0, np.cumsum(np.hypot(*np.diff(op_rear, axis=0).T))]
    s0, d0 = project(np.zeros(2), centre)
    seg = np.diff(centre, axis=0)
    cs = np.r_[0.0, np.cumsum(np.hypot(*seg.T))]
    i = int(np.clip(np.searchsorted(cs, s0) - 1, 0, len(seg) - 1))
    nrm = np.array([-seg[i, 1], seg[i, 0]]) / max(np.hypot(*seg[i]), 1e-9)
    side = float(np.sign(-(centre[i] @ nrm)) or 1.0)      # which side of the line the start (origin) lies on
    # centre-line points from the projection on, then extended straight
    rest = np.vstack([centre[i] + (s0 - cs[i]) * seg[i] / max(np.hypot(*seg[i]), 1e-9), centre[i + 1:]])
    base = _along(rest - rest[0], s_op) + rest[0]
    tg = np.gradient(base, axis=0)
    nn = np.stack([-tg[:, 1], tg[:, 0]], -1) / np.maximum(np.hypot(*tg.T), 1e-9)[:, None]
    ramp = 0.5 * (1 - np.cos(np.pi * np.minimum(TS, REJ_S) / REJ_S))
    return base + (side * d0 * (1 - ramp))[:, None] * nn


def candidates(teacher: dict, v0: float, cam_x: float, route: np.ndarray | None = None, centre: np.ndarray | None = None,
               names=CANDS) -> tuple[np.ndarray, np.ndarray]:
    """(rear-axle rollouts (K, NT, 2) on the 0.1 s grid, traj (K, 21, 4) in openpilot's frame at T_IDXS[:21]) of the
    candidate set. teacher: {"op", "op_L", "op_R"} openpilot plans (33, 15) (MDN means; (33, >= 2) positions also work).
    WL-2's 11 come from jevdrive.wl_traj.candidates on op's (20, 2) 0.25 s plan (same code); route None: op's path
    (real side). `op` / `op_L` / `op_R` are the plans themselves, and their traj rows are the plan's own (x, y, vx, ax)."""
    from .wl_traj import OP_T, candidates as wl_candidates
    op20 = plan_at(teacher["op"], OP_T, cam_x)
    wl = wl_candidates(op20, op20 if route is None else route, v0, TS[1:])
    P, trj = [], []
    for k in names:
        if k in ("op", "op_L", "op_R"):
            pl = np.asarray(teacher[k], float)
            P.append(plan_at(pl, TS, cam_x))
            if pl.shape[1] >= 7:
                trj.append(np.stack([pl[:N21, 0], pl[:N21, 1], pl[:N21, 3], pl[:N21, 6]], -1))
                continue
        elif k == "rej":
            P.append(rej_path(P[names.index("op")], centre))
        else:
            P.append(np.vstack([[0.0, 0.0], wl[k]]))
        st = rollout(P[-1][None])
        tq = T_IDXS[:N21]
        xy = rear_to_op(np.stack([np.interp(tq, TS, P[-1][:, 0]), np.interp(tq, TS, P[-1][:, 1])], -1), cam_x,
                        np.interp(tq, TS, st["h"][0]))
        trj.append(np.column_stack([xy, np.interp(T_IDXS[:N21], TS, st["v"][0]), np.interp(T_IDXS[:N21], TS, st["a"][0])]))
    return np.stack(P), np.stack(trj).astype(np.float32)


# ================================================================ consumers

FIELDS = ("S", "P", "prog", "NC", "DAC", "DDC", "TTC", "C", "top", "S_all", "top_all")


def load_scores(domain: str, root=None) -> dict:
    """R/score/<domain>.npz as a dict (see the S section of tmp/2026-09-30-op-adapt-r2-build.md)."""
    from pathlib import Path
    from .common import data_dir
    with np.load(Path(root or data_dir() / "runs" / "op_adapt_r2") / "score" / f"{domain}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def score_plans(domain: str, uids, plans, frame: str = "op", root=None) -> dict:
    """Score arbitrary plans against stored slots: plans (m, 33, >= 2) openpilot plans (frame "op") or (m, NT, 2)
    rear-axle rollouts (frame "rear"). Same context (actors, map, reference path) and the slot's stored progress
    normaliser. Returns {S, P, prog, NC, DAC, DDC, TTC, C, NC_all, TTC_all, S_all} each (m,)."""
    from .op_adapt_score_data import SlotContext
    ctx = SlotContext(domain, root)
    norm = dict(zip(*ctx.scores(("uid", "prog_norm"))))
    rows = {k: [] for k in ("S", "P", "prog", "NC", "DAC", "DDC", "TTC", "C", "NC_all", "TTC_all", "S_all")}
    for uid, pl in zip(np.asarray(uids), plans):
        s = ctx.slot(int(uid))
        p = plan_at(pl, TS, s.meta["cam_x"])[None] if frame == "op" else np.asarray(pl, float)[None]
        m = raw_metrics(s, p, {"": s.actors.vis, "_all": np.ones(len(s.actors.vis), bool)})
        f, fa = finalize(m, ["x"], prog_norm=float(norm[int(uid)])), finalize(m, ["x"], "_all", prog_norm=float(norm[int(uid)]))
        for k, v in (("S", f["S"]), ("P", f["P"]), ("S_all", fa["S"])):
            rows[k].append(float(v[0]))
        for k in ("prog", "NC", "DAC", "DDC", "TTC", "C", "NC_all", "TTC_all"):
            rows[k].append(float(m[k][0]))
    return {k: np.asarray(v, np.float32) for k, v in rows.items()}
