"""LONG-PROBE (experiments/long_contact): is the LONGITUDINAL consequence of the student's own plan readable from what the driver has at
inference time? Labels, route geometry and the input-ladder head on top of BODY1 arm S0 (experiments/body1/lib, imported, not edited).

Frames: every map / log quantity is stored in the token's LOGGED rear-axle frame at t0 (x forward, y left), the frame of body1's agent labels;
a state of the ot1 / yr1 families starts at `off` = (dy, dpsi) in that frame (sweep.dense maps its queries there).

Route = the ego's hindsight path (the driven path of the next HOR log frames, 30 s at 2 Hz) plus EXT m straight along the last heading,
resampled at 1 m from the logged origin to NR - 1 m (beyond the route's end it continues straight; `rlen` marks how far it is known).
  vlog / tlog (NR,)  the logged ego's speed / time at its first arrival at that arc (nan beyond the hindsight path)
  lines              stop polygons the route crosses (map layer STOP_LINE, type 1 stop sign / 2 traffic-light stop line), first-crossing arc d,
                     light state of the route's connector at the 9 keyframes t0 + 0.5 k (TL), the logged driver's stop / crossing times
  turn               the first route turn of >= 45 deg (curvature runs as in lowboard_diag/scripts/stoplbl_label.py), its curvature peak and
                     the logged speed there
Labels (registered in plans/2026-10-11-long-probe-prereg.md; `a_lab` etc. in {1 positive, 0 negative, -1 no label}):
  A  the own sweep (body1 sweep: ego box at 0.1 s against the logged boxes at matching times) contacts an A-eligible object (a vehicle whose
     centre lies within LAT_IN of the route at t0, ahead of the ego front, aligned with the route or stopped) after t = 0, and the same
     arc-length profile driven along the route (the counterfactual "speed profile on the lane") contacts the same object.
  B  the own plan's front bumper passes an eligible stop line (where the logged driver stops or whose light is red in the window) while the
     stop is mandatory: TL red at the crossing keyframe; stop sign without having slowed below V_STOPPED first.
  C  the own plan reaches the curvature peak of the first >= 45 deg route turn within 4 s at a speed above the logged driver's there by
     C_ABS m/s and the factor C_REL.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "experiments/body1/lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import b1 as B  # noqa: E402
import sweep as SW  # noqa: E402

NR, EXT, HOR = 131, 30.0, 60                    # route grid (1 m, 0 .. 130 m), straight extension (m), hindsight frames (30 s)
NL, NK = 4, 9                                    # stop lines kept per token, light-state keyframes (t0 + 0.5 k, k = 0 .. 8)
LAT_IN, LAT_ROUTE = 1.5, 3.5                     # A: object centre within 1.5 m of the route; B / C: plan point within 3.5 m of the route
HEAD_IN, V_OBJ_STOP = 45.0, 0.5                  # A: object heading within 45 deg of the route heading, or slower than 0.5 m/s
STOP_ZONE, V_LOG_STOP = 15.0, 1.0                # logged driver "stops at the line": speed < 1 m/s with the rear axle in [d - 15, d - FRONT + 1.5]
V_STOPPED = 0.5                                  # B, stop sign: the plan has stopped when its speed fell below 0.5 m/s before the line
C_ABS, C_REL, C_BAND, C_VLOG_MIN, TURN_DEG = 1.5, 1.25, 0.5, 1.0, 45.0
QM = np.r_[0, 1, -1, -1, np.tile([0, 0, 1, 1], 5)].astype(np.int8)      # (24,) student model a query is built on (-1: ship / log)
LONG = ("A", "B", "C")
SPEED_BINS, SPEED_NAMES = (2.0, 5.0, 10.0), ("< 2", "2-5", "5-10", ">= 10")


def root() -> Path:
    return B.data_dir() / "runs" / "long_contact"


def ctx_path() -> Path:
    return root() / "ctx" / "ctx.npz"


def load_ctx(mmap=False) -> dict:
    z = np.load(ctx_path())
    return {k: z[k] for k in z.files}


def speed_band(v):
    return np.digitize(np.asarray(v, float), SPEED_BINS)


# ---------------------------------------------------------------- route geometry (numpy, vectorised over states)
def project(R, P):
    """R (S, NR, 2) route grid at 1 m; P (S, ..., 2) points -> (arc (S, ...) m along the route, signed lateral offset (S, ...) m, left +).
    Nearest grid vertex, refined on its two neighbouring segments."""
    S = len(R)
    sh = P.shape[1:-1]
    p = P.reshape(S, -1, 2)
    d2 = ((p[:, :, None, :] - R[:, None, :, :]) ** 2).sum(-1)                  # (S, n, NR)
    m = d2.argmin(-1)
    best_s, best_l, best_d = None, None, None
    for a in (-1, 0):                                                           # segments [m-1, m] and [m, m+1]
        i0 = np.clip(m + a, 0, NR - 2)
        A = np.take_along_axis(R, i0[..., None].repeat(2, -1), 1)
        Bp = np.take_along_axis(R, (i0 + 1)[..., None].repeat(2, -1), 1)
        u = Bp - A
        L2 = np.maximum((u ** 2).sum(-1), 1e-9)
        t = np.clip(((p - A) * u).sum(-1) / L2, 0.0, 1.0)
        q = A + t[..., None] * u
        dd = ((p - q) ** 2).sum(-1)
        lat = (u[..., 0] * (p[..., 1] - A[..., 1]) - u[..., 1] * (p[..., 0] - A[..., 0])) / np.sqrt(L2)
        s = i0 + t * np.sqrt(L2)
        if best_d is None:
            best_s, best_l, best_d = s, lat, dd
        else:
            w = dd < best_d
            best_s, best_l, best_d = np.where(w, s, best_s), np.where(w, lat, best_l), np.where(w, dd, best_d)
    return best_s.reshape((S,) + sh).astype(np.float32), best_l.reshape((S,) + sh).astype(np.float32)


def pose_at(R, H, s):
    """Route grid R (S, NR, 2), heading H (S, NR); arc s (S, ...) -> poses (S, ..., 3) on the route (straight beyond the grid's end)."""
    S = len(R)
    sh = s.shape[1:]
    ss = s.reshape(S, -1)
    i0 = np.clip(np.floor(ss).astype(np.int64), 0, NR - 2)
    f = np.clip(ss - i0, 0.0, None)[..., None]                                  # > 1 only past the last vertex: straight continuation
    A = np.take_along_axis(R, i0[..., None].repeat(2, -1), 1)
    Bp = np.take_along_axis(R, (i0 + 1)[..., None].repeat(2, -1), 1)
    xy = A + f * (Bp - A)
    h0, h1 = np.take_along_axis(H, i0, 1), np.take_along_axis(H, i0 + 1, 1)
    yaw = h0 + np.clip(f[..., 0], 0, 1) * (h1 - h0)
    return np.concatenate([xy, yaw[..., None]], -1).reshape((S,) + sh + (3,)).astype(np.float32)


def front_point(d):
    """Dense rear-axle poses (..., 3) -> front-bumper centre x, y."""
    return d[..., 0] + SW.FRONT * np.cos(d[..., 2]), d[..., 1] + SW.FRONT * np.sin(d[..., 2])


def path_speed(d):
    """Dense poses (..., 41, 3) at 0.1 s -> speed (..., 41) m/s at every step (mean of the adjacent 0.1 s segments; one-sided at the ends)."""
    v = np.hypot(np.diff(d[..., 0], axis=-1), np.diff(d[..., 1], axis=-1)) / SW.DT
    return np.concatenate([v[..., :1], 0.5 * (v[..., 1:] + v[..., :-1]), v[..., -1:]], -1)


# ---------------------------------------------------------------- per-token A eligibility (state independent)
def a_eligible(box0, valid0, cls, vel0, R, H, rlen):
    """t0 agent boxes (S, K, 5), validity (S, K), class (S, K), velocity (S, K, 2); route -> eligible (S, K) bool, lateral offset (S, K).
    Eligible = vehicle, valid at t0, centre within LAT_IN of the route and on its known part, nearest point ahead of the ego box front,
    heading within HEAD_IN of the route heading there or speed < V_OBJ_STOP."""
    s, lat = project(R, box0[..., :2])
    hr = np.take_along_axis(H, np.clip(np.round(s).astype(np.int64), 0, NR - 1), 1)
    dh = np.degrees(np.abs(np.angle(np.exp(1j * (box0[..., 2] - hr)))))
    slow = np.hypot(vel0[..., 0], vel0[..., 1]) < V_OBJ_STOP
    o = np.zeros((len(box0), 1, 3), np.float32)
    ahead = SW.nearest_lon(o, box0) > SW.HALF_L
    ok = valid0 & (cls == 0) & (np.abs(lat) <= LAT_IN) & (s <= rlen[:, None]) & ahead & ((dh < HEAD_IN) | slow)
    return ok, lat


# ---------------------------------------------------------------- labels of one chunk of rows
def labels(q, off, box, valid, cls, X, a_hit, a_t0, qok):
    """q (S, Q, 8, 3) queries in the states' frames, off (S, 2); body1 agent labels box (S, 9, K, 5), valid (S, 9, K), cls (S, K);
    X = context arrays of the states' tokens (`load_ctx` rows); a_hit / a_t0 / qok (S, Q) from the body1 row unit.
    -> dict of (S, Q) arrays: a_lab, a_sub (1 stopped, 2 moving), a_t; b_lab, b_kind (1 stop sign, 2 light), b_t; c_lab, c_dv, c_t;
       and per state: a_elig_n, b_ahead (an eligible line within 60 m), c_ahead (eligible turn peak within 60 m)."""
    S, Q = q.shape[:2]
    ok_ctx = X["ok"]
    R, H, rlen = X["route"].astype(np.float32), X["rhead"].astype(np.float32), X["rlen"].astype(np.float32)
    offb = np.broadcast_to(np.asarray(off, np.float32)[:, None], (S, Q, 2))
    d = SW.dense(q, offb)                                                       # (S, Q, 41, 3) logged frame
    b, v, vel = SW.dense_boxes(box, valid)
    v &= (cls >= 0)[:, None]
    # ---- A
    elig, _ = a_eligible(box[:, 0], valid[:, 0] & (cls >= 0), cls, vel[:, 0], R, H, rlen)
    ve = v & elig[:, None]
    ex, ey = SW.ego_centre(d)
    bb, vv = b[:, None], ve[:, None]
    _, hit = SW.clearance(ex[..., None], ey[..., None], d[..., 2:3], bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3] / 2, bb[..., 4] / 2)
    hit &= vv                                                                   # (S, Q, 41, K)
    s_own = SW.arc(d)
    s0, _ = project(R, np.stack([np.zeros((S,), np.float32), np.asarray(off, np.float32)[:, 0]], -1)[:, None])
    dc = pose_at(R, H, s0[:, :1, None] + s_own)                                  # counterfactual: own arc profile along the route
    cx, cy = SW.ego_centre(dc)
    _, hcf = SW.clearance(cx[..., None], cy[..., None], dc[..., 2:3], bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3] / 2, bb[..., 4] / 2)
    hcf &= vv
    own_any = hit[:, :, 1:].any(2)                                              # (S, Q, K) contact after t = 0
    t0hit = hit[:, :, 0].any(-1)
    both = own_any & hcf.any(2)
    first = np.where(hit[:, :, 1:].any(2), hit[:, :, 1:].argmax(2) + 1, SW.NS)  # (S, Q, K) first own contact step
    tf = np.where(both, first, SW.NS)
    k1 = tf.argmin(-1)
    f1 = np.take_along_axis(tf, k1[..., None], -1)[..., 0]
    a_pos = (f1 < SW.NS) & ~t0hit
    spd = np.hypot(vel[..., 0], vel[..., 1])                                    # (S, 41, K)
    sp1 = spd[np.arange(S)[:, None], np.minimum(f1, SW.NS - 1), k1]
    a_lab = np.where(a_pos, 1, np.where(~a_hit & ~t0hit & ~own_any.any(-1), 0, -1)).astype(np.int8)
    a_lab[~qok | ~ok_ctx[:, None]] = -1
    out = dict(a_lab=a_lab, a_sub=np.where(a_pos, np.where(sp1 < V_OBJ_STOP, 1, 2), 0).astype(np.int8),
               a_t=np.where(a_pos, f1 * SW.DT, SW.BIG).astype(np.float32), a_elig_n=elig.sum(1).astype(np.int8))
    # ---- B
    fx, fy = front_point(d)
    sf, lf = project(R, np.stack([fx, fy], -1))                                 # (S, Q, 41)
    on = np.abs(lf) <= LAT_ROUTE
    vp = path_speed(d)                                                          # (S, Q, 41)
    cm = np.minimum.accumulate(vp[..., 1:], -1)                                 # cm[k - 1] = min speed over steps 1 .. k
    inf2 = np.full(vp.shape[:-1] + (2,), np.inf, np.float32)
    vmin_before = np.concatenate([inf2, cm[..., :-1]], -1)                      # [st] = min speed over steps 1 .. st - 1 (inf for st <= 1)
    b_lab = np.zeros((S, Q), np.int8)
    b_unk = np.zeros((S, Q), bool)
    b_t, b_kind = np.full((S, Q), SW.BIG, np.float32), np.zeros((S, Q), np.int8)
    reach = sf[..., -1]
    for j in range(NL):
        ty, dl, el = X["ln_type"][:, j], X["ln_d"][:, j], X["ln_elig"][:, j]
        if not el.any():
            continue
        cr = on & (sf >= dl[:, None, None]) & el[:, None, None]
        has = cr.any(-1)
        st = np.where(has, cr.argmax(-1), SW.NS)                                # (S, Q) crossing step
        tc = st * SW.DT
        kf = np.clip(np.round(tc / 0.5).astype(np.int64), 0, NK - 1)
        light = np.take_along_axis(np.broadcast_to(X["ln_st"][:, j][:, None, :], (S, Q, NK)), kf[..., None], -1)[..., 0]
        vmn = np.take_along_axis(vmin_before, np.minimum(st, SW.NS - 1)[..., None], -1)[..., 0]
        viol = has & np.where((ty == 2)[:, None], light == 1, (ty == 1)[:, None] & (vmn >= V_STOPPED))
        unk = has & (ty == 2)[:, None] & (light < 0)
        new = viol & (tc < b_t)
        b_t, b_kind = np.where(new, tc, b_t), np.where(new, ty[:, None], b_kind)
        b_lab |= viol.astype(np.int8)
        b_unk |= unk
    known = (reach + 0.0 <= rlen[:, None]) | (b_lab == 1)                       # a negative needs the route known to the plan's front reach
    b_lab = np.where(b_lab == 1, 1, np.where(b_unk | ~known, -1, 0)).astype(np.int8)
    b_lab[~qok | ~ok_ctx[:, None]] = -1
    ahead = (X["ln_elig"] & (X["ln_d"] <= 60.0)).any(1)
    out |= dict(b_lab=b_lab, b_kind=b_kind, b_t=b_t, b_ahead=ahead)
    # ---- C
    sr, lr = project(R, d[..., :2])
    spk, vlog_pk, tok = X["tn_spk"], X["tn_vpk"], X["tn_ok"]
    elig_c = tok & (vlog_pk >= C_VLOG_MIN)                                      # the logged driver drives through the peak (no stop in the turn)
    at = (sr >= spk[:, None, None]) & (np.abs(lr) <= LAT_ROUTE) & tok[:, None, None]
    reached = at.any(-1)
    stp = np.where(reached, at.argmax(-1), SW.NS)
    vpk = np.take_along_axis(vp, np.minimum(stp, SW.NS - 1)[..., None], -1)[..., 0]
    dv = vpk - np.nan_to_num(vlog_pk)[:, None]
    fast = reached & elig_c[:, None] & (dv >= C_ABS) & (vpk >= C_REL * np.nan_to_num(vlog_pk)[:, None])
    neg = ~reached | (elig_c[:, None] & (dv <= C_BAND))                         # peak not reached within 4 s, or not faster than the log
    c_lab = np.where(fast, 1, np.where(neg, 0, -1)).astype(np.int8)            # -1: the band (C_BAND, C_ABS), or a peak where the log stops
    c_lab[~qok | ~ok_ctx[:, None]] = -1
    out |= dict(c_lab=c_lab, c_dv=np.where(reached & elig_c[:, None], dv, np.nan).astype(np.float32),
                c_t=np.where(reached, stp * SW.DT, SW.BIG).astype(np.float32), c_ahead=elig_c & (spk <= 60.0))
    return out
