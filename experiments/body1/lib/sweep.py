"""BODY1 swept-footprint labels (plans/2026-10-10-body1-prereg.md, 2.2): the ego box along a queried trajectory against logged agent boxes at
matching times (with their motion) and against the drivable boundary. numpy, vectorised over (states, queries, 41 steps, objects).

Conventions (all in the token's LOGGED rear-axle frame at t0, x forward, y left):
  query      (..., 8, 3) x, y, yaw at 0.5 .. 4 s in the state's own frame; `off` (..., 2) = (dy, dpsi) of the state's t0 pose in the logged frame
             (zeros on the log; the `off` column of the ot1 / yr1 / bd1 caches). `dense` maps to the logged frame and interpolates linearly to
             0.1 s from the state's own origin (41 steps), the way lib/drivable_hinge.py and the driver serve a plan.
  agents     box (N, 9, K, 5) = x, y, yaw, L, W at t0 + 0, 0.5 .. 4 s (lib/agent_hinge.py labels); interpolated linearly between the two
             neighbouring annotated times, a step is valid when both are (AgentHinge's rule). Velocity = finite difference of the dense centres.
  boundary   drivable SDF raster (128, 96), x -8 .. 56 m, y -24 .. 24 m, 0.5 m cells, positive inside (opb_labels.py); sampled bilinearly at
             the 4 footprint corners with border clamp (torch grid_sample, align_corners False); steps with a corner outside the raster are
             not labelled (`cov`).
  ego box    nuPlan Pacifica about the rear axle: front 4.049, rear -1.127, half width 1.1485 (5.176 x 2.297 m).

Contact = separating-axis test of the two rectangles (exact, also for crossing boxes without a corner inside); clearance = min corner-to-box
signed distance both ways (exact when separated), clamped to <= 0 on contact. Checked against swv1_lib.Rollout.sweep (shapely) and
AgentHinge.distances / drivable_hinge.Hinge.margins by experiments/body1/scripts/bd1_rows.py check.
"""
import numpy as np

FRONT, REAR, HALF_W = 4.049, -1.127, 1.1485
C_OFF, HALF_L = (FRONT + REAR) / 2, (FRONT - REAR) / 2
X0, Y0, RES, NH, NW = -8.0, -24.0, 0.5, 128, 96
NS, DT = 41, 0.1
TAU = np.arange(NS) * DT
BIG = np.float32(99.0)                     # clearance / margin / time filler where nothing is defined
SGN = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], np.float32)   # corner signs (front-left, front-right, rear-right, rear-left)


def interp_matrix() -> np.ndarray:
    """(41, 9): linear interpolation of [origin, 8 poses at 0.5 s] onto the 0.1 s grid (identical to lib/drivable_hinge.interp_matrix)."""
    M = np.zeros((NS, 9), np.float32)
    for k in range(NS):
        j = min(k // 5, 7)
        f = (k - 5 * j) / 5.0
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


M = interp_matrix()
MV = M > 0


def to_log(P, off):
    """Poses (..., K, 3) in the state's frame -> the logged t0 frame; off (..., 2) = (dy, dpsi). Inverse of ot_rows.to_frame."""
    dy, dp = off[..., None, 0], off[..., None, 1]
    c, s = np.cos(dp), np.sin(dp)
    return np.stack([c * P[..., 0] - s * P[..., 1], dy + s * P[..., 0] + c * P[..., 1], P[..., 2] + dp], -1)


def dense(P, off=None):
    """Query poses (..., 8, 3) -> (..., 41, 3) logged-frame rear-axle poses at 0.1 s, starting at the state's own t0 pose."""
    P = np.asarray(P, np.float32)
    off = np.zeros(P.shape[:-2] + (2,), np.float32) if off is None else np.asarray(off, np.float32)
    o = np.stack([np.zeros_like(off[..., 0]), off[..., 0], off[..., 1]], -1)[..., None, :]
    return np.einsum("kj,...jc->...kc", M, np.concatenate([o, to_log(P, off)], -2))


def arc(d):
    """Dense poses (..., 41, 3) -> cumulative arc length (..., 41)."""
    s = np.hypot(np.diff(d[..., 0], axis=-1), np.diff(d[..., 1], axis=-1))
    return np.concatenate([np.zeros_like(s[..., :1]), np.cumsum(s, -1)], -1)


def dense_boxes(box, valid):
    """Agent labels (N, 9, K, 5), (N, 9, K) -> (N, 41, K, 5) boxes at 0.1 s, validity (N, 41, K), velocity (N, 41, K, 2) m/s."""
    b = np.einsum("kj,njoc->nkoc", M, box.astype(np.float32))
    v = ~(np.einsum("kj,njo->nko", MV.astype(np.float32), (~valid).astype(np.float32)) > 0.5)
    seg = (b[:, 5::5, :, :2] - b[:, :-5:5, :, :2]) / 0.5                       # (N, 8, K, 2): constant within an annotation interval
    vel = seg[:, np.minimum(np.arange(NS) // 5, 7)]
    return b, v, np.where(v[..., None], vel, 0.0).astype(np.float32)


def ego_centre(d):
    """Rear-axle poses (..., 3) -> box-centre x, y."""
    return d[..., 0] + C_OFF * np.cos(d[..., 2]), d[..., 1] + C_OFF * np.sin(d[..., 2])


def corners(cx, cy, yaw, hl, hw):
    """Box params (...,) -> corner x, y (..., 4)."""
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    u, v = SGN[:, 0] * np.asarray(hl)[..., None], SGN[:, 1] * np.asarray(hw)[..., None]
    return cx[..., None] + c * u - s * v, cy[..., None] + s * u + c * v


def box_sdf(px, py, cx, cy, yaw, hl, hw):
    """Signed distance of points to oriented boxes (broadcasting)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = px - cx, py - cy
    qx, qy = np.abs(c * dx + s * dy) - hl, np.abs(-s * dx + c * dy) - hw
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def sat(ax, ay, ayaw, ahl, ahw, bx, by, byaw, bhl, bhw):
    """Rectangle overlap by the separating-axis test (broadcasting) -> bool."""
    ca, sa, cb, sb = np.cos(ayaw), np.sin(ayaw), np.cos(byaw), np.sin(byaw)
    dx, dy = bx - ax, by - ay
    cc, ss = np.abs(ca * cb + sa * sb), np.abs(sa * cb - ca * sb)              # |cos|, |sin| of the relative heading
    return ((np.abs(ca * dx + sa * dy) <= ahl + bhl * cc + bhw * ss) & (np.abs(-sa * dx + ca * dy) <= ahw + bhl * ss + bhw * cc)
            & (np.abs(cb * dx + sb * dy) <= bhl + ahl * cc + ahw * ss) & (np.abs(-sb * dx + cb * dy) <= bhw + ahl * ss + ahw * cc))


def clearance(ex, ey, eyaw, ox, oy, oyaw, ohl, ohw, ehl=HALF_L, ehw=HALF_W):
    """Ego box (centre ex, ey, heading eyaw) against object boxes (broadcasting) -> (signed clearance, contact)."""
    qx, qy = corners(ex, ey, eyaw, np.broadcast_to(np.float32(ehl), ex.shape), np.broadcast_to(np.float32(ehw), ex.shape))
    d1 = box_sdf(qx, qy, ox[..., None], oy[..., None], oyaw[..., None], ohl[..., None], ohw[..., None]).min(-1)
    px, py = corners(ox, oy, oyaw, ohl, ohw)
    d2 = box_sdf(px, py, ex[..., None], ey[..., None], eyaw[..., None], ehl, ehw).min(-1)
    hit = sat(ex, ey, eyaw, ehl, ehw, ox, oy, oyaw, ohl, ohw)
    d = np.minimum(d1, d2)
    return np.where(hit, np.minimum(d, 0), np.maximum(d, 0)), hit


def ego_frame(d, px, py):
    """Points (..., ) in the frame of the ego box centre of dense poses d (...,3): (lon, lat)."""
    ex, ey = ego_centre(d)
    c, s = np.cos(d[..., 2]), np.sin(d[..., 2])
    return c * (px - ex) + s * (py - ey), -s * (px - ex) + c * (py - ey)


def nearest_lon(d, b):
    """Longitudinal coordinate (ego box-centre frame) of the point of each object box nearest to the ego box centre. d (..., 1, 3), b (..., K, 5)."""
    ex, ey = ego_centre(d)
    c, s = np.cos(b[..., 2]), np.sin(b[..., 2])
    dx, dy = ex - b[..., 0], ey - b[..., 1]
    u, v = np.clip(c * dx + s * dy, -b[..., 3] / 2, b[..., 3] / 2), np.clip(-s * dx + c * dy, -b[..., 4] / 2, b[..., 4] / 2)
    return ego_frame(d, b[..., 0] + c * u - s * v, b[..., 1] + s * u + c * v)[0]


def _take(a, i):
    """a (..., T, K), i (..., K) step index -> a at that step per object (..., K)."""
    return np.take_along_axis(a, i[..., None, :], -2)[..., 0, :]


def agent_labels(d, b, v, vel, cls):
    """d (S, Q, 41, 3) dense queries; b (S, 41, K, 5), v (S, 41, K), vel (S, 41, K, 2) from dense_boxes; cls (S, K) -> dict of (S, Q) arrays:
    hit (bool; contact with a counted object), t / s (time s, arc length m of the first counted contact; BIG if none), cls (class of the struck
    object, -1), obj (its slot, -1), rear (bool: some object ran into the ego's rear half while faster; such objects are not counted),
    clr (min clearance over all valid objects and steps, <= 0 on contact, BIG if no valid object), lat (signed lateral gap m of the
    min-clearance object at that step: + left, - right, 0 when it overlaps the ego's lateral extent), t0 (bool: counted contact already at t = 0)."""
    S, Q = d.shape[:2]
    bb, vv = b[:, None], v[:, None]                                            # (S, 1, 41, K, ...)
    ex, ey = ego_centre(d)
    c, hit = clearance(ex[..., None], ey[..., None], d[..., 2:3], bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3] / 2, bb[..., 4] / 2)
    hit &= vv
    c = np.where(vv, c, BIG)
    anyk = hit.any(2)                                                          # (S, Q, K)
    first = hit.argmax(2)                                                      # first contact step per object
    # rear-end rule at the object's first contact: nearest point of the object behind the ego box centre, object faster along the ego heading
    si, qi, ki = np.arange(S)[:, None, None], np.arange(Q)[None, :, None], np.arange(b.shape[2])[None, None]
    bf, vf = b[si, first, ki], vel[si, first, ki]                              # (S, Q, K, 5 / 2) object box and velocity at its first contact
    dfk = d[si, qi, first]                                                     # (S, Q, K, 3) ego pose at that step
    lon = nearest_lon(dfk, bf)
    ev = np.gradient(d[..., :2], DT, axis=2)                                   # (S, Q, 41, 2) ego velocity
    evk = ev[si, qi, first]
    hx, hy = np.cos(dfk[..., 2]), np.sin(dfk[..., 2])
    rear_k = anyk & (lon < 0) & (vf[..., 0] * hx + vf[..., 1] * hy > evk[..., 0] * hx + evk[..., 1] * hy + 0.1)
    cnt = anyk & ~rear_k
    tf = np.where(cnt, first, NS)
    k1 = tf.argmin(-1)                                                         # (S, Q) struck object
    f1 = np.take_along_axis(tf, k1[..., None], -1)[..., 0]
    pos = f1 < NS
    s_arc = arc(d)
    fc = np.minimum(f1, NS - 1)
    out = dict(hit=pos, t=np.where(pos, fc * DT, BIG).astype(np.float32), s=np.where(pos, np.take_along_axis(s_arc, fc[..., None], -1)[..., 0], BIG).astype(np.float32),
               cls=np.where(pos, cls[np.arange(S)[:, None], k1], -1).astype(np.int8), obj=np.where(pos, k1, -1).astype(np.int8),
               rear=rear_k.any(-1), t0=pos & (f1 == 0))
    flat = c.reshape(S, Q, -1)
    j = flat.argmin(-1)
    cm = np.take_along_axis(flat, j[..., None], -1)[..., 0]
    jt, jk = j // c.shape[3], j % c.shape[3]
    bm, dm = b[np.arange(S)[:, None], jt, jk], d[np.arange(S)[:, None], np.arange(Q)[None], jt]     # (S, Q, 5), (S, Q, 3)
    px, py = corners(bm[..., 0], bm[..., 1], bm[..., 2], bm[..., 3] / 2, bm[..., 4] / 2)
    lat = ego_frame(dm[..., None, :], px, py)[1]
    lo, hi = lat.min(-1), lat.max(-1)
    out |= dict(clr=cm.astype(np.float32), lat=np.where(cm >= BIG, 0, np.where(lo > HALF_W, lo - HALF_W, np.where(hi < -HALF_W, hi + HALF_W, 0))).astype(np.float32))
    return out


def sdf_at(sdf, x, y):
    """sdf (S, 128, 96); points x, y (S, ...) -> bilinear sample with border clamp (S, ...), inside-raster mask."""
    S = len(sdf)
    fx, fy = (x - X0) / RES - 0.5, (y - Y0) / RES - 0.5
    inside = (x >= X0) & (x <= X0 + NH * RES) & (y >= Y0) & (y <= Y0 + NW * RES)
    fx, fy = np.clip(fx, 0, NH - 1), np.clip(fy, 0, NW - 1)
    i0, j0 = np.minimum(fx.astype(np.int64), NH - 2), np.minimum(fy.astype(np.int64), NW - 2)
    a, b = (fx - i0).astype(np.float32), (fy - j0).astype(np.float32)
    si = np.arange(S).reshape((S,) + (1,) * (x.ndim - 1))
    g = lambda i, j: sdf[si, i, j].astype(np.float32)  # noqa: E731
    return (g(i0, j0) * (1 - a) * (1 - b) + g(i0 + 1, j0) * a * (1 - b) + g(i0, j0 + 1) * (1 - a) * b + g(i0 + 1, j0 + 1) * a * b), inside


def corner_margins(d, sdf):
    """d (S, Q, 41, 3), sdf (S, 128, 96) -> SDF at the 4 footprint corners (S, Q, 41, 4), inside-raster mask (same shape)."""
    ex, ey = ego_centre(d)
    cx, cy = corners(ex, ey, d[..., 2], np.broadcast_to(np.float32(HALF_L), ex.shape), np.broadcast_to(np.float32(HALF_W), ex.shape))
    return sdf_at(sdf, cx, cy)


def boundary_labels(d, sdf):
    """-> dict of (S, Q): hit (a footprint corner at SDF < 0 on a covered step), t, s (first such step; BIG), margin (min corner SDF over covered
    steps; BIG if none), cov (number of the 41 steps whose 4 corners lie inside the raster, uint8), t0 (contact already at t = 0)."""
    m, ins = corner_margins(d, sdf)
    ok = ins.all(-1)                                                           # (S, Q, 41)
    step = np.where(ok, m.min(-1), BIG)
    hit = step < 0
    f = hit.argmax(-1)
    pos = hit.any(-1)
    return dict(hit=pos, t=np.where(pos, f * DT, BIG).astype(np.float32), s=np.where(pos, np.take_along_axis(arc(d), f[..., None], -1)[..., 0], BIG).astype(np.float32),
                margin=step.min(-1).astype(np.float32), cov=ok.sum(-1).astype(np.uint8), t0=pos & (f == 0))


def labels(q, off, box, valid, cls, sdf):
    """Queries q (S, Q, 8, 3) in the states' frames, off (S, 2); agent labels box (S, 9, K, 5), valid (S, 9, K), cls (S, K); sdf (S, 128, 96)
    -> {"a_<key>": agent labels, "b_<key>": boundary labels}, every value (S, Q)."""
    d = dense(q, np.broadcast_to(np.asarray(off, np.float32)[:, None], q.shape[:2] + (2,)))
    b, v, vel = dense_boxes(box, valid)
    A, B = agent_labels(d, b, v & (cls >= 0)[:, None], vel, cls), boundary_labels(d, sdf)
    return {f"a_{k}": x for k, x in A.items()} | {f"b_{k}": x for k, x in B.items()}
