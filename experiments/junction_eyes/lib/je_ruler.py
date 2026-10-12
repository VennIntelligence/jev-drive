"""junction_eyes rulers (plans/2026-10-12-prereg-A.md, amendment 1 items 3-4, amendment 2): R-A (route-conditioned arc-length ruler) and
R-B (48 x 48 BEV grid at 0.5 m: 0 off-road / 1 drivable / 2 oncoming) from a trajdata VectorMap and a logged rear-axle path.

One code path for every source. Frames: world (or any frame the map and the path share); labels in the token's rear-axle frame, x forward,
y LEFT. Used by scripts/je0.py (labels) and by the step-1 read protocol.

  m = LocalMap(vm, lo, hi)                     lanes (with both edges) and road area inside a bbox; junction-lane flags
  r = m.route(LP, S, H)                        the de-duplicated lane sequence the logged path follows, sampled every STEP m
  ra = ra_label(r, LP, S, T, anchor, cam, t)   (5, 3, 2): C / L / R at SS metres of route arc ahead of the camera's projection
  rb = m.rb_label(anchor)                      (48, 48) uint8, [x index, y index], cell centres GX x GY
"""
from __future__ import annotations

import numpy as np

SS = (5.0, 10.0, 15.0, 20.0, 30.0)
NRA = len(SS) * 3 * 2                                   # 30 numbers: s-major, points C, L, R, coords x, y
GX = np.arange(48) * 0.5 + 0.25                         # rear-axle frame cell centres (m)
GY = np.arange(48) * 0.5 - 11.75
STEP, BACK, AHEAD, GAP, JUMP = 1.0, 20.0, 80.0, 4.0, 6.0
WMIN, WMAX = 1.5, 8.0
HEAD_MAX, JUNC, ONC = np.radians(45.0), np.radians(30.0), np.radians(120.0)
OVERLAP = 1.0                                          # m^2 of polygon overlap that makes two crossing lanes junction lanes
T_WIN = 15e6                                           # us: the anchor is projected on the logged path within +- 15 s of t


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def in_ego(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, float).reshape(-1, 2) - np.asarray(a[:2], float)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def to_world(a, X):
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, float).reshape(-1, 2)
    return np.asarray(a[:2], float) + np.c_[c * X[:, 0] - s * X[:, 1], s * X[:, 0] + c * X[:, 1]]


def proj(P, p):
    """Nearest point of polyline P (n, 2) to p -> (segment index, t in [0, 1], distance)."""
    d = P[1:] - P[:-1]
    t = np.clip(((p - P[:-1]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    q = P[:-1] + t[:, None] * d
    e = np.hypot(*(q - p).T)
    i = int(np.argmin(e))
    return i, float(t[i]), float(e[i])


def seg_proj(C, X):
    """Points X (m, 2) on polyline C (n, 2) -> projected points (m, 2), segment heading (m,)."""
    A, d = C[:-1], C[1:] - C[:-1]
    dd = np.maximum((d * d).sum(1), 1e-12)
    t = np.clip((((X[:, None, :] - A[None]) * d[None]).sum(-1)) / dd[None], 0, 1)          # (m, nseg)
    Q = A[None] + t[..., None] * d[None]
    j = np.argmin(((Q - X[:, None, :]) ** 2).sum(-1), 1)
    return Q[np.arange(len(X)), j], np.arctan2(d[j, 1], d[j, 0])


def ray_hit(c, n, E):
    """Signed parameters lam where the lines c_i + lam n_i (m, 2) meet polyline E (k, 2) -> (m, nseg) with NaN where they miss."""
    Q0, d = E[:-1], E[1:] - E[:-1]
    den = n[:, None, 0] * d[None, :, 1] - n[:, None, 1] * d[None, :, 0]
    w = Q0[None] - c[:, None]
    lam = (w[..., 0] * d[None, :, 1] - w[..., 1] * d[None, :, 0]) / np.where(np.abs(den) < 1e-12, np.nan, den)
    mu = (w[..., 0] * n[:, None, 1] - w[..., 1] * n[:, None, 0]) / np.where(np.abs(den) < 1e-12, np.nan, den)
    return np.where((mu >= -1e-9) & (mu <= 1 + 1e-9), lam, np.nan)


def logged_path(T, P, ext=60.0):
    """Rear-axle path (T us, P (n, 2)) -> LP (k, 2), arc S, vertex heading H (unwrapped), vertex time TT; straight extension of `ext` m."""
    Y_end = np.arctan2(*(P[-1] - P[max(0, len(P) - 4)])[::-1]) if len(P) > 1 else 0.0
    LP = np.vstack([P, P[-1] + np.outer(np.arange(2.0, ext + 0.01, 2.0), [np.cos(Y_end), np.sin(Y_end)])])
    TT = np.r_[T, np.full(len(LP) - len(P), T[-1])]
    keep = np.r_[True, np.hypot(*np.diff(LP, axis=0).T) > 1e-3]
    LP, TT = LP[keep], TT[keep]
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(LP, axis=0).T))]
    seg = np.unwrap(np.arctan2(*np.diff(LP, axis=0).T[::-1]))
    return LP, S, np.r_[seg, seg[-1]], TT


class LocalMap:
    def __init__(self, vm, lo, hi):
        import shapely
        from trajdata.maps.vec_map_elements import MapElementType as MT
        self.sh = shapely
        inb = lambda X: bool(((X[:, 0] >= lo[0]) & (X[:, 0] <= hi[0]) & (X[:, 1] >= lo[1]) & (X[:, 1] <= hi[1])).any())  # noqa: E731
        self.C, self.L, self.R, polys = [], [], [], []
        for ln in vm.elements.get(MT.ROAD_LANE, {}).values():
            if ln.left_edge is None or ln.right_edge is None or ln.center is None:
                continue
            C, L, R = (np.asarray(e.points, float)[:, :2] for e in (ln.center, ln.left_edge, ln.right_edge))
            if len(C) < 2 or len(L) < 2 or len(R) < 2 or not (inb(C) or inb(L) or inb(R)):
                continue
            p = shapely.Polygon(np.r_[L, R[::-1]]).buffer(0)
            if p.is_empty:
                continue
            self.C.append(C), self.L.append(L), self.R.append(R), polys.append(p)
        self.poly = polys
        for p in polys:
            shapely.prepare(p)
        self.tree = shapely.STRtree(polys) if polys else None
        self.junc = np.zeros(len(polys), bool)
        if self.tree is not None:
            for i, p in enumerate(polys):
                for j in self.tree.query(p, predicate="intersects"):
                    j = int(j)
                    if j <= i:
                        continue
                    x = p.intersection(polys[j])
                    if x.area <= OVERLAP:
                        continue
                    c = np.array(x.representative_point().coords)[0][None]
                    hi_, hj = seg_proj(self.C[i], c)[1][0], seg_proj(self.C[j], c)[1][0]
                    if abs(wrap(hi_ - hj)) > JUNC:
                        self.junc[i] = self.junc[j] = True
        areas = []
        for ra in vm.elements.get(MT.ROAD_AREA, {}).values():
            E = np.asarray(ra.exterior_polygon.points, float)[:, :2]
            if len(E) > 2 and inb(E):
                areas.append(shapely.Polygon(E, [np.asarray(h.points, float)[:, :2] for h in ra.interior_holes]).buffer(0))
        self.road = shapely.union_all(areas) if areas else None
        if self.road is not None:
            shapely.prepare(self.road)

    # ------------------------------------------------------------ R-A
    def route(self, LP, S, H):
        """Samples every STEP m of the logged path -> dict(s path arc, c (n, 2) route centre, l / r boundary points (NaN if none),
        t tangent heading, ok assigned, run id (-1 = none)): the lane sequence the path follows (amendment 1 item 3)."""
        sg = np.arange(0.0, S[-1], STEP)
        P = np.c_[np.interp(sg, S, LP[:, 0]), np.interp(sg, S, LP[:, 1])]
        h = np.interp(sg, S, H)
        n = len(sg)
        c, l, r, t = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full(n, np.nan)
        best = np.full(n, np.inf)
        lane = np.full(n, -1)
        if self.tree is not None and n:
            ii, jj = self.tree.query(self.sh.points(P), predicate="within")
            for j in np.unique(jj):
                m = ii[jj == j]
                q, hd = seg_proj(self.C[j], P[m])
                dh = np.abs(wrap(hd - h[m]))
                b = (dh < HEAD_MAX) & (dh < best[m])
                m, q, hd, dh = m[b], q[b], hd[b], dh[b]
                best[m], lane[m], c[m], t[m] = dh, j, q, hd
        for j in np.unique(lane[lane >= 0]):
            m = np.flatnonzero(lane == j)
            nv = np.c_[-np.sin(t[m]), np.cos(t[m])]
            lam_l = ray_hit(c[m], nv, self.L[j])
            lam_r = ray_hit(c[m], nv, self.R[j])
            ll = np.nanmin(np.where(lam_l > 0, lam_l, np.nan), 1) if lam_l.size else np.full(len(m), np.nan)
            rr = np.nanmax(np.where(lam_r < 0, lam_r, np.nan), 1) if lam_r.size else np.full(len(m), np.nan)
            w = ll - rr
            okw = np.isfinite(w) & (w >= WMIN) & (w <= WMAX)
            l[m] = np.where(okw[:, None], c[m] + ll[:, None] * nv, np.nan)
            r[m] = np.where(okw[:, None], c[m] + rr[:, None] * nv, np.nan)
        ok = lane >= 0
        run = np.full(n, -1)
        k, last, prev = -1, -np.inf, None
        for i in range(n):
            if not ok[i]:
                continue
            if sg[i] - last > GAP or (prev is not None and np.hypot(*(c[i] - c[prev])) > JUMP):
                k += 1
            run[i], last, prev = k, sg[i], i
        return dict(s=sg, c=c, l=l, r=r, t=t, ok=ok, run=run)

    # ------------------------------------------------------------ R-B
    def rb_label(self, a):
        """(48, 48) uint8 classes around pose a = (x, y, yaw) (amendment 1 item 4)."""
        X, Y = np.meshgrid(GX, GY, indexing="ij")
        W = to_world(a, np.c_[X.ravel(), Y.ravel()])
        out = np.zeros(len(W), np.uint8)
        if self.road is None:
            return out.reshape(48, 48)
        road = self.sh.contains_xy(self.road, W[:, 0], W[:, 1])
        junc, onc = np.zeros(len(W), bool), np.zeros(len(W), bool)
        if self.tree is not None:
            box = self.sh.box(*W.min(0), *W.max(0))
            for j in self.tree.query(box, predicate="intersects"):
                j = int(j)
                inside = self.sh.contains_xy(self.poly[j], W[:, 0], W[:, 1])
                if not inside.any():
                    continue
                if self.junc[j]:
                    junc |= inside
                else:
                    m = np.flatnonzero(inside)
                    hd = seg_proj(self.C[j], W[m])[1]
                    onc[m[np.abs(wrap(hd - a[2])) > ONC]] = True
        out[road] = 1
        out[road & onc & ~junc] = 2
        return out.reshape(48, 48)


def _interp_finite(s, S, V, maxgap=2.0):
    """V (n, 2) on arc S at s, using only finite rows; NaN if the finite rows around s are more than maxgap apart from s."""
    f = np.isfinite(V[:, 0])
    if not f.any():
        return np.full(2, np.nan)
    Sf, Vf = S[f], V[f]
    i = np.searchsorted(Sf, s)
    lo, hi = max(i - 1, 0), min(i, len(Sf) - 1)
    if min(abs(Sf[lo] - s), abs(Sf[hi] - s)) > maxgap or s < Sf[0] - maxgap or s > Sf[-1] + maxgap:
        return np.full(2, np.nan)
    return np.array([np.interp(s, Sf, Vf[:, 0]), np.interp(s, Sf, Vf[:, 1])])


def ra_label(rt, LP, S, TT, a, cam, t):
    """R-A label of a decision: rt = LocalMap.route of the logged path (LP, S, vertex times TT), a = rear-axle pose (x, y, yaw) in the
    map frame, cam = camera position in the rear-axle frame, t = decision time (us, the path's time base) -> (5, 3, 2) float32 (NaN =
    undefined), the anchor's logged-path arc s0 and the camera's route arc."""
    out = np.full((len(SS), 3, 2), np.nan, np.float32)
    w = np.flatnonzero(np.abs(TT - t) <= T_WIN)
    if len(w) < 2:
        w = np.arange(len(LP))
    w = np.arange(w[0], min(w[-1] + 2, len(LP)))
    i, u, _ = proj(LP[w], np.asarray(a[:2], float))
    s0 = S[w[i]] + u * (S[w[i] + 1] - S[w[i]])
    sg, run = rt["s"], rt["run"]
    win = (sg >= s0 - BACK) & (sg <= s0 + AHEAD) & (run >= 0)
    if not win.any():
        return out, s0, np.nan
    k0 = int(np.argmin(np.abs(sg - s0)))
    cand = np.flatnonzero(win & (sg >= sg[k0] - 0.5 * STEP))
    if run[k0] >= 0 and win[k0]:
        rid = run[k0]
    elif len(cand) and sg[cand[0]] - s0 <= GAP:
        rid = run[cand[0]]
    else:
        return out, s0, np.nan
    m = np.flatnonzero(win & (run == rid))
    if len(m) < 2:
        return out, s0, np.nan
    c, l, r, th = rt["c"][m], rt["l"][m], rt["r"][m], rt["t"][m]
    dS = np.maximum(0.0, (np.diff(c, axis=0) * np.c_[np.cos(th[1:]), np.sin(th[1:])]).sum(1))
    Sc = np.r_[0.0, np.cumsum(dS)]
    pc = to_world(a, np.asarray(cam[:2], float)[None])[0]
    j, uu, _ = proj(c, pc)
    s_cam = Sc[j] + uu * (Sc[j + 1] - Sc[j])
    for q, s in enumerate(SS):
        x = s_cam + s
        if x > Sc[-1] + 1e-6:
            continue
        C = np.array([np.interp(x, Sc, c[:, 0]), np.interp(x, Sc, c[:, 1])])
        out[q, 0] = in_ego(a, C[None])[0]
        Lp, Rp = _interp_finite(x, Sc, l), _interp_finite(x, Sc, r)
        if np.isfinite(Lp).all() and np.isfinite(Rp).all() and WMIN <= np.hypot(*(Lp - Rp)) <= WMAX:
            out[q, 1], out[q, 2] = in_ego(a, Lp[None])[0], in_ego(a, Rp[None])[0]
    return out, s0, s_cam
