"""ONPOL-PAI label geometry (numpy only): the logged ego path and what is measured against it, actor boxes on the agent_labels layout,
road-edge segments, wait-line crossings. Conventions and every output array: experiments/onpol_pai/results/rows_format.md.
Poses are (x, y, yaw); "local" = the rollout's local frame, "rig frame at a" = x forward, y left at the pose a."""
import numpy as np

NEVER = np.iinfo(np.int64).max
NT, K, NE, NP = 9, 32, 96, 121                   # box times (t0 + 0.5 j s), boxes per row, edge segments per row, path samples per row
T_BOX = (np.arange(NT) * 500_000).astype(np.int64)
T_FUT = (np.arange(1, 9) * 500_000).astype(np.int64)
PATH_ARC = np.arange(-20.0, 101.0)               # arcs of `path` relative to s_ego
EDGE_R, EDGE_DZ, EDGE_STEP, EDGE_W = 40.0, 3.0, 1.0, 0.1
STOP_AHEAD, STOP_ZONE, V_LOG_STOP, STOP_EXT, NL = 100.0, 15.0, 1.0, 30.0, 4      # lc.py: STOP_ZONE, V_LOG_STOP, EXT, NL
CLASSES = ("vehicle", "generic_object", "pedestrian", "bicycle")                 # agent_labels' classes
CLS = {"person": 2, "rider": 3, "": 1, **{k: 0 for k in ("automobile", "heavy_truck", "trailer", "bus", "train_or_tram_car", "other_vehicle")}}
WL_KIND = {"STOP": 1, "YIELD": 2}                # any other wait-line type: 3


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_rig(xy, a):
    """Local points (..., 2) -> the rig frame at pose a (3,)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(xy, np.float64) - a[:2]
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def pose_to_rig(p, a):
    """Local poses (..., 3) -> the rig frame at pose a; the yaw is not wrapped (it stays continuous along an unwrapped input)."""
    p = np.asarray(p, np.float64)
    return np.concatenate([to_rig(p[..., :2], a), p[..., 2:3] - a[2]], -1)


def box_sdf(p, c, hl, hw):
    """Signed distance of points (..., 2) to the axis-aligned box centred at c = (cx, 0) with half extents hl, hw."""
    q = np.abs(np.asarray(p) - np.array([c, 0.0])) - np.array([hl, hw])
    return np.hypot(*np.maximum(q, 0).T).T + np.minimum(q.max(-1), 0)


class Path:
    """A timed pose track t_us (L,), xyyaw (L, 3) as a path. Vertices = the samples that advance more than MIN_STEP from the previous
    vertex (`sv` arc, `pv`, `hv` heading, `vv` = the minimum speed of the samples a vertex stands for, so a logged stop reads about 0):
    a parked log's position noise is not path. Per sample: `s` = arc of its vertex, `v` = speed from the raw positions."""
    MIN_STEP = 0.05

    def __init__(self, t_us, xyyaw):
        self.t, self.xy, self.yaw = np.asarray(t_us, np.int64), np.asarray(xyyaw, np.float64)[:, :2], np.unwrap(np.asarray(xyyaw, np.float64)[:, 2])
        keep = [0]
        for i in range(1, len(self.xy)):
            if np.hypot(*(self.xy[i] - self.xy[keep[-1]])) > self.MIN_STEP:
                keep.append(i)
        keep = np.array(keep)
        self.pv, self.hv = self.xy[keep], self.yaw[keep]
        self.sv = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.pv, axis=0).T))]
        self.len = float(self.sv[-1])
        self.s = self.sv[np.searchsorted(keep, np.arange(len(self.xy)), "right") - 1]
        ts = self.t * 1e-6
        self.v = np.hypot(np.gradient(self.xy[:, 0], ts), np.gradient(self.xy[:, 1], ts)) if len(ts) > 1 else np.zeros(len(ts))
        self.vv = np.minimum.reduceat(self.v, keep)

    def at(self, a):
        """Arc(s) -> pose(s) (..., 3) on the path; straight along the end headings beyond either end."""
        a = np.asarray(a, np.float64)
        c = np.clip(a, 0.0, self.len)
        h = np.interp(c, self.sv, self.hv)
        d = a - c
        return np.stack([np.interp(c, self.sv, self.pv[:, 0]) + d * np.cos(h), np.interp(c, self.sv, self.pv[:, 1]) + d * np.sin(h), h], -1)

    def speed(self, a):
        return np.interp(np.clip(a, 0.0, self.len), self.sv, self.vv)

    def at_time(self, t_us):
        """Time(s) -> (pose(s) (..., 3), arc(s), inside the track's time span)."""
        t = np.asarray(t_us, np.int64)
        f = lambda y: np.interp(t, self.t, y)  # noqa: E731
        return np.stack([f(self.xy[:, 0]), f(self.xy[:, 1]), f(self.yaw)], -1), f(self.s), (t >= self.t[0]) & (t <= self.t[-1])

    def project(self, p, lo=-np.inf, hi=np.inf):
        """Point (2,) -> (arc, signed lateral offset, left +) of its nearest point on the path among arcs [lo, hi]. Past an end the
        path continues straight along its end heading (arc < 0 or > len; the offset is then the perpendicular one)."""
        p, sv, pv = np.asarray(p, np.float64), self.sv, self.pv
        if len(sv) < 2:                                                       # a log that never moves
            u = np.array([np.cos(self.hv[0]), np.sin(self.hv[0])])
            d = p - pv[0]
            return float(d @ u), float(u[0] * d[1] - u[1] * d[0])
        A, u = pv[:-1], np.diff(pv, axis=0)
        L = np.hypot(u[:, 0], u[:, 1])
        t = np.clip(((p - A) * u).sum(1) / L ** 2, 0.0, 1.0)
        d2 = ((p - A - t[:, None] * u) ** 2).sum(1)
        m = (sv[1:] >= lo) & (sv[:-1] <= hi)
        if not m.any():
            m[len(m) - 1 if lo > sv[-1] else 0] = True
        i = int(np.where(m, d2, np.inf).argmin())
        for end, j, sg in ((i == len(L) - 1 and t[i] == 1.0, -1, 1.0), (i == 0 and t[i] == 0.0, 0, -1.0)):
            if end:
                e = np.array([np.cos(self.hv[j]), np.sin(self.hv[j])])
                d = p - pv[j]
                al = float(d @ e)
                if sg * al > 0:
                    return (self.len if j else 0.0) + al, float(e[0] * d[1] - e[1] * d[0])
        return float(sv[i] + t[i] * L[i]), float((u[i, 0] * (p[1] - A[i, 1]) - u[i, 1] * (p[0] - A[i, 0])) / L[i])

    def track(self, P, back=5.0, fwd=40.0):
        """Points (N, 2) visited in time order -> (arc (N,), offset (N,)): each projection is searched from `back` behind to `fwd`
        ahead of the previous one, so a path that comes back near itself is not jumped across."""
        out, s = np.zeros((len(P), 2)), 0.0
        for i, p in enumerate(P):
            out[i] = self.project(p, s - back, s + fwd)
            s = out[i, 0]
        return out[:, 0], out[:, 1]

    def first_at(self, arc):
        """Arc(s) -> time (us, int64) at which the track first reaches it; NEVER if it does not."""
        arc = np.atleast_1d(np.asarray(arc, np.float64))
        k = np.searchsorted(self.s, arc, "left")
        out = np.full(len(arc), NEVER, np.int64)
        out[k == 0] = self.t[0]
        m = (k > 0) & (k < len(self.s))
        km = k[m]
        f = (arc[m] - self.s[km - 1]) / np.maximum(self.s[km] - self.s[km - 1], 1e-9)
        out[m] = np.round(self.t[km - 1] + f * (self.t[km] - self.t[km - 1])).astype(np.int64)
        return out


def scorer_lateral(xy, p):
    """AlpaSim GroundTruthScorer._lateral_distance_to_gt for a point p against the logged box-centre polyline xy (L, 2), on the raw
    coordinates as the scorer has them: the distance to the polyline, or, when the nearest point is its end, the component
    perpendicular to its last segment (whatever direction that segment has: a parked log ends in position noise)."""
    A, u = xy[:-1], np.diff(xy, axis=0)
    L2 = (u ** 2).sum(1)
    t = np.clip(((p - A) * u).sum(1) / np.maximum(L2, 1e-18), 0.0, 1.0)
    d = np.hypot(*(p - A - t[:, None] * u).T)
    i = int(d.argmin())
    arc, tot, n = np.sqrt(L2[:i]).sum() + t[i] * np.sqrt(L2[i]), np.sqrt(L2).sum(), np.sqrt(L2[-1])
    if arc < tot - 1e-9 or n <= 0 or tot <= 0:
        return float(d[i])
    return float(abs((p - xy[-1])[0] * u[-1, 1] - (p - xy[-1])[1] * u[-1, 0]) / n)


def imit(path: Path, t0, s_ego, e0, dpsi0, v0, anchor):
    """The pre-registered recovery target of one row -> (poses (8, 3) in the rig frame at `anchor`, ok (8,)); rows_format.md."""
    tk = t0 + T_FUT
    (_, sl, ok), sl0 = path.at_time(tk), path.at_time(t0)[1]
    d = sl - sl0
    D = max(8.0, 2.0 * float(v0))

    def curve(dd):
        q, w = path.at(s_ego + dd), np.clip(1.0 - dd / D, 0.0, None)
        return q[:, :2] + (e0 * w)[:, None] * np.stack([-np.sin(q[:, 2]), np.cos(q[:, 2])], -1), q[:, 2], w
    xy, h, w = curve(d)
    g = curve(d + 0.1)[0] - curve(np.maximum(d - 0.1, 0.0))[0]
    psi = np.where(d < 0.5, h + dpsi0 * w, h + wrap(np.arctan2(g[:, 1], g[:, 0]) - h))
    return pose_to_rig(np.c_[xy, psi], anchor), ok & (s_ego + d <= path.len)


# ---------------------------------------------------------------- simulator actors -> agent_labels boxes
def actor_tracks(actors: dict, size: dict, actor_t):
    """c1_extract.one_log's `actors` {id: (n, 4) t, x, y, yaw} and `size` {id: (length, width, label)} on the rollout's time grid
    -> ids (A,), track (A, T, 3) with NaN where the actor has no pose (yaw unwrapped per actor), dims (A, 2), labels (A,)."""
    tg = np.asarray(actor_t, np.int64)
    ids = sorted(k for k in actors if k != "EGO")
    X = np.full((len(ids), len(tg), 3), np.nan)
    for i, k in enumerate(ids):
        a = actors[k]
        j = np.searchsorted(tg, a[:, 0].astype(np.int64))
        X[i, j, :2], X[i, j, 2] = a[:, 1:3], np.unwrap(a[:, 3])
    return np.array(ids), X, np.array([size[k][:2] for k in ids], np.float32).reshape(-1, 2), np.array([size[k][2] for k in ids])


def boxes(tg, X, dims, t0, anchor):
    """Rows t0 (N,), anchor (N, 3) -> box (N, 9, K, 5), ok (N, 9, K), actor index (N, K) (-1 = empty): the K actors nearest to the rig
    origin at t0 followed over t0 + 0.5 j s, linear between the two neighbouring simulator steps (a time is valid when the actor has
    both; nothing is extrapolated past the rollout's last step)."""
    N, A = len(t0), len(X)
    box, ok, idx = np.zeros((N, NT, K, 5), np.float32), np.zeros((N, NT, K), bool), np.full((N, K), -1, np.int32)
    if not A:
        return box, ok, idx
    tq = np.asarray(t0, np.int64)[:, None] + T_BOX                                       # (N, 9)
    f = np.interp(tq, tg, np.arange(len(tg), dtype=np.float64))
    i0 = np.clip(np.floor(f).astype(np.int64), 0, len(tg) - 2)
    w = (f - i0)[None, ..., None]
    a, b = X[:, i0], X[:, i0 + 1]                                                        # (A, N, 9, 3)
    with np.errstate(invalid="ignore"):
        P = np.where(w < 1e-6, a, np.where(w > 1 - 1e-6, b, (1 - w) * a + w * b))
    P[:, (tq < tg[0]) | (tq > tg[-1])] = np.nan
    for n in range(N):
        p = P[:, n]                                                                      # (A, 9, 3)
        d = np.hypot(*(p[:, 0, :2] - anchor[n, :2]).T)
        sel = np.argsort(np.where(np.isfinite(d), d, np.inf), kind="stable")[:K]
        sel = sel[np.isfinite(d[sel])]
        q = pose_to_rig(p[sel], anchor[n])                                               # (k, 9, 3)
        q[..., 2] -= 2 * np.pi * np.round(q[:, :1, 2] / (2 * np.pi))                    # yaw at t0 in (-pi, pi], continuous after it
        v = np.isfinite(q[..., 0])
        box[n, :, :len(sel), :3] = np.where(v[..., None], q, 0.0).transpose(1, 0, 2)
        box[n, :, :len(sel), 3:] = np.where(v[..., None], dims[sel][:, None], 0.0).transpose(1, 0, 2)
        ok[n, :, :len(sel)], idx[n, :len(sel)] = v.T, sel
    return box, ok, idx


# ---------------------------------------------------------------- road edges
def resample(poly, step=EDGE_STEP):
    """Polyline (n, 2 | 3) -> the same line at equal steps of at most `step` m (plan-view arc)."""
    poly = np.asarray(poly, np.float64)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(poly[:, :2], axis=0).T))]
    g = np.linspace(0.0, s[-1], max(1, int(np.ceil(s[-1] / step))) + 1)
    return np.stack([np.interp(g, s, poly[:, j]) for j in range(poly.shape[1])], -1)


def edge_segments(xyz, off):
    """Concatenated resampled polylines -> segments (m, 5): centre x, y, yaw, length, centre z."""
    a, b = xyz[:-1], xyz[1:]
    keep = np.ones(len(a), bool)
    keep[np.asarray(off[1:-1], np.int64) - 1] = False                                    # no segment across two polylines
    keep &= np.hypot(*(b[:, :2] - a[:, :2]).T) > 1e-6
    a, b = a[keep], b[keep]
    d = b[:, :2] - a[:, :2]
    return np.c_[(a[:, :2] + b[:, :2]) / 2, np.arctan2(d[:, 1], d[:, 0]), np.hypot(d[:, 0], d[:, 1]), (a[:, 2] + b[:, 2]) / 2]


def edge_rows(seg, anchor, z_ego):
    """Segments (m, 5), rows anchor (N, 3), ego z (N,) -> edge_box (N, NE, 5), edge_ok (N, NE): nearest first, within EDGE_R, same level."""
    N = len(anchor)
    box, ok = np.zeros((N, NE, 5), np.float32), np.zeros((N, NE), bool)
    for n in range(N):
        d = np.hypot(*(seg[:, :2] - anchor[n, :2]).T)
        m = (d <= EDGE_R) & ~(np.abs(seg[:, 4] - z_ego[n]) > EDGE_DZ)                    # NaN z (no elevation in the map) passes
        j = np.flatnonzero(m)
        j = j[np.argsort(d[j], kind="stable")[:NE]]
        box[n, :len(j)] = np.c_[to_rig(seg[j, :2], anchor[n]), wrap(seg[j, 2] - anchor[n, 2]), seg[j, 3], np.full(len(j), EDGE_W)]
        ok[n, :len(j)] = True
    return box, ok


# ---------------------------------------------------------------- wait lines
def crossings(path: Path, lines, front):
    """Wait lines [(polyline (n, 2), type, implicit)] crossed by the logged path or by its STOP_EXT m straight extension -> arrays over
    the crossings, by arc: arc, kind, implicit, tcross (the log's front bumper reaches the line), tstop (the log is slower than
    V_LOG_STOP with the rear axle in [arc - STOP_ZONE, arc - front + 1.5] before crossing; lc_ctx.py's rule). Crossings within 1 m
    of arc are one line (STOP before YIELD before the rest)."""
    pv = np.vstack([path.pv, path.at(path.len + STOP_EXT)[:2]])
    sv = np.r_[path.sv, path.len + STOP_EXT]
    A, u = pv[:-1], np.diff(pv, axis=0)
    hits = []
    for poly, ty, imp in lines:
        poly = np.asarray(poly, np.float64)[:, :2]
        for C, D in zip(poly[:-1], poly[1:]):
            w, r = D - C, C - A
            den = u[:, 0] * w[1] - u[:, 1] * w[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t, q = (r[:, 0] * w[1] - r[:, 1] * w[0]) / den, (r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]) / den
            for i in np.flatnonzero((np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (q >= 0) & (q <= 1)):
                hits.append((float(sv[i] + t[i] * (sv[i + 1] - sv[i])), WL_KIND.get(str(ty).upper(), 3), bool(imp)))
    hits.sort(key=lambda h: (h[1], h[0]))
    keep = []
    for h in hits:
        if all(abs(h[0] - k[0]) > 1.0 for k in keep):
            keep.append(h)
    keep.sort()
    arc = np.array([h[0] for h in keep], np.float64)
    tc = path.first_at(arc - front) if len(arc) else np.zeros(0, np.int64)
    ts = np.full(len(arc), NEVER, np.int64)
    for j, a in enumerate(arc):
        z = (path.s >= a - STOP_ZONE) & (path.s <= a - front + 1.5) & (path.v < V_LOG_STOP) & (path.t < tc[j])
        if z.any():
            ts[j] = path.t[np.argmax(z)]
    return dict(arc=arc, kind=np.array([h[1] for h in keep], np.int8), implicit=np.array([h[2] for h in keep], bool), tcross=tc, tstop=ts)


def stop_rows(X, t0, s_ego, front):
    """crossings() output -> the per-row arrays: `wl_*` every line ahead (front - 0.5 m < arc - s_ego <= STOP_AHEAD), nearest first;
    `stop_*` the eligible ones (the log stops for the line and has not crossed it at t0)."""
    N = len(t0)
    o = {k: np.full((N, NL), v, d) for k, v, d in (("stop_s", np.nan, np.float32), ("stop_tcross_us", NEVER, np.int64), ("stop_kind", 0, np.int8),
                                                   ("wl_s", np.nan, np.float32), ("wl_tcross_us", NEVER, np.int64), ("wl_tstop_us", NEVER, np.int64),
                                                   ("wl_kind", 0, np.int8), ("wl_implicit", False, bool))}
    for n in range(N):
        r = X["arc"] - s_ego[n]
        a = np.flatnonzero((r > front - 0.5) & (r <= STOP_AHEAD))
        e = a[(X["tstop"][a] != NEVER) & (t0[n] < X["tcross"][a])][:NL]
        a = a[:NL]
        o["wl_s"][n, :len(a)], o["wl_tcross_us"][n, :len(a)], o["wl_tstop_us"][n, :len(a)] = r[a], X["tcross"][a], X["tstop"][a]
        o["wl_kind"][n, :len(a)], o["wl_implicit"][n, :len(a)] = X["kind"][a], X["implicit"][a]
        o["stop_s"][n, :len(e)], o["stop_tcross_us"][n, :len(e)], o["stop_kind"][n, :len(e)] = r[e], X["tcross"][e], X["kind"][e]
    return o
