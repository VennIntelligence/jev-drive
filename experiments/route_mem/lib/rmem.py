"""ROUTE-MEM (experiments/route_mem, plans/2026-10-12-route-mem-prereg.md): the route memory a driver keeps of the AlpaSim route messages it
has received, and the adapter branch that reads it.

Memory (numpy, shared by the offline row builders and the served driver, so both compute the same numbers):
  batches   every route message of the episode so far: the waypoints (rig frame of the pose at the message time, NaN padding dropped), each
            carried into the current rear-axle frame through the ego poses (local frame -> current frame, rigid)
  merge     all batches sample the same route polyline (AlpaSim builds it once per rollout and only extends it), oldest first: the merged
            polyline is the oldest batch followed by every newer batch's points that lie beyond its end (along its last segment)
  window    the ego's arc position on the merged polyline (its projection; extrapolated along the first / last segment when it lies before /
            beyond it), then N_PT = 40 points at the arc offsets OFFS = -6, -4, .., +72 m: valid where the polyline covers the offset
  features  per point [valid, x / 20, y / 20, cos(heading), sin(heading)] in the current rear-axle frame (x forward, y left), zeros where
            invalid -> DIM = 200 numbers. A first decision has only its own batch, which starts ~42 m ahead: the near points are invalid.

Branch (torch): RouteParityAdapter = parity_adapter.ParityAdapter (unchanged weights and forward on the first 20 ego numbers) + RouteHead, a
small transformer decoder whose 32 slot queries read [null, ego token, the valid route points]; its output layer starts at zero, so the
branch adds exactly 0 until trained and an init from any P2 checkpoint is that checkpoint. The fed row is ego (20) ++ route features (200);
`present` (ego[:, 0]) multiplies both biases (anchor rows stay the shipped model).

Ordered layout (route v2, experiments/route_follow/plans/2026-10-15-route-v2-prereg.md Amendment 2, rm_train --rm-order; sidecar head
`order: true`):
  anchor    the ego's arc on the merged route is PROGRESS-anchored instead of projected: the newest route message's first waypoint lies L0 =
            42.105 m of route arc ahead of the ego's projection at that message (AlpaSim's resampling), so s_e = arc(that waypoint) - L0 + the
            chord driven since. The projection mislabels the slots while the ego is short of the memory's start and the route bends in
            between (decision 299: the start shown a median 4.5 m too near, de9db5dc 14 m for 41 m); the progress anchor puts every point at
            its route arc. Same 40 slots (-6 .. +72 m, 2 m), same point features.
  scalars   [d_turn / 60, d_turn known, min(trav, 200) / 100] appended: d_turn = arc offset of the first valid slot in 0 .. +60 m whose heading
            differs >= 30 deg from the reference heading (the slot at +0 m when valid, else the ego's own heading) and still differs 4 m on
            (or the memory ends there; a merge kink is not a turn); none found: 60 when the +60 m slot is valid, else unknown (value 0, bit 0).
            trav = chords between the poses of the route messages up to t0, then to the anchor. DIM_O = 40 x 5 + 3 = 203.
  one rule  memory_o (offline rows, rm_build / rf_cfpool) and Tracker(order=True) (served) are the same function of the messages and poses;
            to_order / to_order_t give the scalars of any point block (the trainer's fakes are built on the ordered block, then re-scored).
  branch    RouteHead(order=True): the 40 point tokens as before + a zero-initialised linear map of the scalars added to the ego token
            (order_state() converts an old head: the scalar map at 0, every other tensor kept).

Anchored layout (2026-10-17, results/align_audit.md "Minimal fix" C + D, check results/anchor_check.md; rm_train --rm-anchor, sidecar head
`order: true, anchor: true`; the ordered slot format, so RouteHead(order=True) reads it; the projected and ordered layouts never use this):
  merge_a   the fold-back fix: a newer batch extends the memory from the ARC of the memory's end on that batch (its nearest point on the batch's
            segments that run within 90 deg of the memory's last segment, if within A_ON m): the batch's points more than TOL further along are
            appended. A batch that starts beyond the memory's end (a gap) is appended whole; one the memory's end lies beyond adds nothing, and
            so does one with no segment running with the memory's end (it lies on the approach of a loop the memory has passed). merge decides
            "beyond the end" by the direction of the last segment, which at loops points at the newer batch's first points and appends the
            whole batch (the polyline folds back 33-37 m).
  anchor    the car's arc from AlpaSim's own progress, corrected for its trim, carried by the car's driven chord. Per message k with waypoints:
            C_k = arc of its first waypoint on the merged polyline (as merged up to k) - L0 (memory_o's progress anchor). AlpaSim resamples 20
            points every SPC = 80 / 19 m from the car's projection and drops those whose CHORD sum is under 40 m (_trim_route_start): on a
            sharp corner close ahead m more points go, the first waypoint lies m x SPC further out and the message holds N_FULL - m points
            (N_FULL = 10; a message is also short when the route ends or a segment leaves 3.5-4.5 m, which cuts its far end instead). So the
            car's arc is one of c_k(m) = C_k - m x SPC, m = 0 .. N_FULL - n_k: the first such message takes m = 0 (a far end cut by a route
            end leaves the same short message as a trim, e.g. 9 of RSG's 13 one-short first messages on public 1491, and nothing in it tells
            them apart: the ordered layout's assumption); every later one takes the hypothesis nearest the previous anchor + the chord the car
            drove since if it explains that within A_TRIM m (ties: the smaller m), else keeps the previous m (clipped to the message's range):
            a legit jump of the car's projection against its chord at a far-end cut is otherwise read as a trim. A full message is C_k
            itself. At the decision: s_K + the chord driven since message K (chords along every message pose with a known pose, then to the
            decision pose). Route arc legitimately outruns the driven chord at the inside of a turn by up to 7 m per 0.5 s, so the anchor is
            never capped by the chord (anchor_check.md: the cap min(C_k, s_{k-1} + chord + 1 m) lags there).
  scalars   memory_o's (d_turn / 60, known, min(trav, 200) / 100; to_order of the point block); DIM_O = 203.
  one rule  memory_a (offline rows: rm_build nav / pool / anchor, rm_align anchor) and Tracker(anchor=True) (served: sh30_driver and pai_driver
            set SERVE_ANCHOR from the checkpoint's sidecar, head_anchor) are the same function of the messages and their poses.

Window layout (2026-10-22, never trained: the 10-22 base retrain was dropped, experiments/round9/results/route_check/README.md; rm_train --rm-win, sidecar head `order: true, anchor: true,
win: {...}, cmd: "heading"`; the anchored layout's merge and arc, only the slot offsets and the command change; every other layout untouched):
  slots     the 40 slots at offsets -BACK .. L(v), evenly spaced, L(v) = clip(T v + C, L_MIN, L_MAX) with v = the fed ego speed (10 x the ego
            row's vx / 10 column, float32, as the training rows store it): T 5 s (the round9 draft's value), C 20 m, L_MIN 40 m, L_MAX 78 m (an AlpaSim
            route message reaches 80 m from the car's projection, RouteGenerator.ROUTE_LOOKAHEAD_DISTANCE_M, so nothing beyond ~80 m ever
            exists at serve), BACK 6 m: 1.18 m spacing below 4 m/s, 2.15 m from 11.6 m/s on. Point features as before (the positions carry the
            spacing; the head's slot embeddings are learnt on this layout from the factory weights).
  scalars   memory_a's three (d_turn / 60, known, trav / 100), from the fixed-grid block (the anchored block at OFFS: unchanged definition).
  command   cmd_heading (40 m, 20 deg) on the float16 fixed-grid block: one-hot (4,) appended after the scalars -> DIM_W = 203 + 4. The
            trainer feeds the first DIM_O numbers to the head and the command to the ego row's command columns of the AlpaSim-route pool rows
            (round7's CMD_PARTS; navtrain rows keep NAVSIM's command); the drivers do the same with the served command (no JEV_CMD needed).
  one rule  memory_w + compose_w (offline rows: rm_build win, round9 nb.py pai) and Tracker(anchor=True, win=...) (served) are the same
            function of the messages, their poses and the fed speed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_PT, NF = 40, 5
OFFS = -6.0 + 2.0 * np.arange(N_PT)              # arc offsets (m) from the ego's projection on the merged route
DIM = N_PT * NF
I20 = int(np.flatnonzero(OFFS == 20.0)[0])       # the +20 m point (coverage flag of local_map / decision 279)
TOL = 1.0                                        # a newer batch's point extends the merged polyline when it lies > TOL m beyond its end
EGO_DIM = 20
SIDECAR = "route_mem.json"                       # next to ckpt-final.pt: the checkpoint carries the branch (its config)
I0 = int(np.flatnonzero(OFFS == 0.0)[0])         # the ego's projection
I60 = int(np.flatnonzero(OFFS == 60.0)[0])
N_PT_O = N_PT                                    # ordered layout: the same 40 slots (-6 .. +72 m), progress-anchored (Amendment 2)
L0 = 80.0 / 19.0 * 10                            # AlpaSim RouteGenerator: 20 points every 80 / 19 m from the ego's projection, < 40 m dropped
OFFS_O = OFFS[:N_PT_O]
NSC = 3                                          # d_turn / 60, d_turn known, travelled / 100
DIM_O = N_PT_O * NF + NSC
TURN_DEG, TURN_CAP, TRAV_CAP = 30.0, 60.0, 200.0
LAT_S = (10.0, 20.0, 30.0, 40.0)                 # --rm-latfeat: arcs of the lateral-offset features
LAT_I = [int(np.flatnonzero(OFFS == s_)[0]) for s_ in LAT_S]


def lat_feats(P):
    """Ordered point block (B, N_PT, NF) tensor -> (B, 8): the route's signed lateral offset (left +, ego heading frame) at 10 / 20 / 30 / 40 m
    of route arc / 10, clipped to +-4, then the 4 validity bits (an invalid slot gives 0, 0). A function of the fed block, so the served
    and trained values are the same numbers."""
    import torch
    v = (P[:, LAT_I, 0] > 0.5).float()
    return torch.cat([(P[:, LAT_I, 2] * 2.0).clamp(-4, 4) * v, v], 1)
SERVE_ORDER = False                              # Tracker() default in a driver: set from the served checkpoint's sidecar (head_order)
SERVE_ANCHOR = False                             # anchored layout: Tracker() default in a driver, set from the sidecar (head_anchor)
A_ON = 2.0                                       # merge_a: the memory's end lies on a newer batch when within A_ON m of its segments
SPC, N_FULL = 80.0 / 19.0, 10                    # memory_a: AlpaSim's waypoint spacing; waypoints of a message nothing extra was trimmed from
A_TRIM = 1.5                                     # memory_a: a short message changes the trim count only if a hypothesis explains the step within this
WIN = dict(T=5.0, C=20.0, L_MIN=40.0, L_MAX=78.0, BACK=6.0)   # window layout (module docstring): L(v) = clip(T v + C, L_MIN, L_MAX), slots -BACK .. L(v)
SERVE_WIN = None                                 # window layout: Tracker() default in a driver, the sidecar head's `win` dict (head_win); None = off
NCMD = 4                                         # window layout: the command one-hot appended after the scalars
DIM_W = DIM_O + NCMD


def win_offs(v, win=None):
    """Fed speed(s) v (m/s; (...,)) -> the window layout's slot offsets (..., N_PT) (module docstring "Window layout")."""
    w = WIN if win is None else win
    L = np.clip(w["T"] * np.maximum(np.asarray(v, np.float64), 0.0) + w["C"], w["L_MIN"], w["L_MAX"])
    return -w["BACK"] + (np.asarray(L)[..., None] + w["BACK"]) * np.linspace(0.0, 1.0, N_PT)


def speed_of(e4) -> float:
    """The ego row's vx / 10 column (float32, parity_adapter.ego_features) -> the fed speed (m/s) the window layout uses."""
    return 10.0 * float(np.float32(e4))


def to_local(a, X):
    """Rear-axle frame points X (n, 2) of pose a = (x, y, yaw) -> local frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    X = np.asarray(X, np.float64).reshape(-1, 2)
    return np.c_[a[0] + c * X[:, 0] - s * X[:, 1], a[1] + s * X[:, 0] + c * X[:, 1]]


def in_ego(a, X):
    """Local frame points X (n, 2) -> rear-axle frame of pose a = (x, y, yaw)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(X, np.float64).reshape(-1, 2) - np.asarray(a[:2], np.float64)
    return np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]


def merge(batches):
    """Batches (list of (m, 2) polylines in one frame, oldest first; NaN rows dropped) -> merged (M, 2) polyline or None."""
    M = None
    for B in batches:
        B = np.asarray(B, np.float64).reshape(-1, 2)
        B = B[np.isfinite(B).all(1)]
        if not len(B):
            continue
        if M is None:
            M = B.copy()
            continue
        if len(M) < 2:                                     # a one-point memory: keep the newer batch if it is longer
            M = B.copy() if len(B) > 1 else M
            continue
        u = M[-1] - M[-2]
        u = u / max(np.hypot(*u), 1e-9)
        j = np.flatnonzero((B - M[-1]) @ u > TOL)
        if len(j):
            M = np.vstack([M, B[j[0]:]])
    return M


def window(M, prog=None, info=None, s_fix=None, offs=None):
    """Merged polyline (M, 2) in the current rear-axle frame -> (N_PT, NF) features (module docstring). prog = (p, c): the ordered layout's
    progress anchor instead of the ego's projection: p = the newest route message's first waypoint (rear-axle frame), which AlpaSim places
    L0 m of route arc ahead of the ego's projection at that message, c = the chord the ego has driven since -> s_e = arc(p) - L0 + c.
    s_fix: the anchored layout's arc (memory_a) used as s_e. info (dict, optional) receives s_geo (the projection's arc), s_e (the arc used)
    and M0 (the merged polyline's first point). offs: the slot offsets (window layout, win_offs), default OFFS."""
    F = np.zeros((N_PT, NF), np.float32)
    if M is None or len(M) < 2:
        return F
    d = np.diff(M, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    keep = L > 1e-6
    if not keep.any():
        return F
    M = np.vstack([M[:1], M[1:][keep]])
    d, L = d[keep], L[keep]
    u = d / L[:, None]
    cum = np.r_[0.0, np.cumsum(L)]
    t = np.clip(-(M[:-1] * u).sum(1), 0.0, L)                                 # projection of the origin on every segment
    q = M[:-1] + t[:, None] * u
    i = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    s_e = cum[i] + t[i]
    if i == 0 and t[0] <= 0.0:                                                # before the memory: extrapolate along the first segment
        s_e = float(-(M[0] * u[0]).sum())
    elif i == len(L) - 1 and t[i] >= L[i]:                                    # beyond its end: along the last segment
        s_e = cum[-1] + float(-((M[-1]) * u[-1]).sum())
    if prog is not None:                                                      # ordered layout: the progress anchor
        pq, ch = np.asarray(prog[0], np.float64), float(prog[1])
        tq = np.clip(((pq - M[:-1]) * u).sum(1), 0.0, L)
        j = int(np.argmin(np.hypot(*(M[:-1] + tq[:, None] * u - pq).T)))
        s_e = cum[j] + tq[j] - L0 + ch
    if info is not None:
        info.update(s_geo=float(cum[i] + t[i]) if not ((i == 0 and t[0] <= 0.0) or (i == len(L) - 1 and t[i] >= L[i])) else None, s_e=float(s_e), M0=M[0].copy())
        if info["s_geo"] is None:
            info["s_geo"] = float(-(M[0] * u[0]).sum()) if i == 0 and t[0] <= 0.0 else cum[-1] + float(-((M[-1]) * u[-1]).sum())
        if prog is None:
            info["s_geo"] = float(s_e)
    if s_fix is not None:                                                     # anchored layout (memory_a): the arc it tracked
        s_e = float(s_fix)
        if info is not None:
            info["s_e"] = s_e
    s = s_e + (OFFS if offs is None else offs)
    ok = (s >= 0.0) & (s <= cum[-1])
    if not ok.any():
        return F
    j = np.clip(np.searchsorted(cum, s[ok], side="right") - 1, 0, len(L) - 1)
    P = M[j] + (s[ok] - cum[j])[:, None] * u[j]
    F[ok] = np.c_[np.ones(ok.sum()), P[:, 0] / 20.0, np.clip(P[:, 1] / 20.0, -4, 4), u[j, 0], u[j, 1]].astype(np.float32)
    return F


def memory(batches_local, anchor):
    """Batches already in the local frame (oldest first) and the current rear-axle pose -> (DIM,) features."""
    M = merge(batches_local)
    return window(None if M is None else in_ego(anchor, M)).reshape(-1)


def pts(f):
    """Features of either layout (..., DIM | DIM_O) -> the point block (..., N_PT | N_PT_O, NF)."""
    f = np.asarray(f)
    n = N_PT if f.shape[-1] == DIM else N_PT_O
    assert f.shape[-1] in (DIM, DIM_O), f.shape
    return f[..., :n * NF].reshape(*f.shape[:-1], n, NF)


def memory_o(batches_local, anchor, last_pose, trav, info=None):
    """Ordered layout (DIM_O,): batches already in the local frame (oldest first; the last one = the newest message with waypoints), the
    current rear-axle pose, the local pose of that newest message (its first waypoint anchors the ego's arc) and the travelled arc."""
    B = [np.asarray(b, np.float64).reshape(-1, 2) for b in batches_local]
    B = [b[np.isfinite(b).all(1)] for b in B]
    B = [b for b in B if len(b)]
    M = merge(B)
    if M is None:
        return to_order(np.zeros(DIM, np.float32), trav)
    a = np.asarray(anchor, np.float64)
    prog = (in_ego(a, B[-1][:1])[0], float(np.hypot(*(a[:2] - np.asarray(last_pose, np.float64)[:2]))))
    return to_order(window(in_ego(a, M), prog, info).reshape(-1), trav)


# ---------------------------------------------------------------- anchored layout (module docstring "Anchored layout")
def _clean(B):
    """Points (m, 2) -> float64, NaN rows and repeated points (zero-length segments) dropped."""
    B = np.asarray(B, np.float64).reshape(-1, 2)
    B = B[np.isfinite(B).all(1)]
    return B[np.r_[True, np.hypot(*np.diff(B, axis=0).T) > 1e-6]] if len(B) > 1 else B


def _on(P, q, u=None, strict=False):
    """Polyline P (m >= 2, 2; no zero-length segment), point q (2,), direction u (2,) or None -> (arc of q's nearest point on P, its distance,
    side): among the segments within 90 deg of u (all when u is None or none is; strict: None when none is); side -1 = q lies before P's
    start, +1 beyond its end."""
    d = np.diff(P, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    v = d / L[:, None]
    tr = ((np.asarray(q, np.float64) - P[:-1]) * v).sum(1)
    t = np.clip(tr, 0.0, L)
    D = np.hypot(*(P[:-1] + t[:, None] * v - q).T)
    if u is not None and (v @ u > 0).any():
        D = np.where(v @ u > 0, D, np.inf)
    elif u is not None and strict:
        return None
    i = int(np.argmin(D))
    side = -1 if i == 0 and tr[0] < 0 else 1 if i == len(L) - 1 and tr[i] > L[i] else 0
    return float(L[:i].sum() + t[i]), float(D[i]), side


def merge_a(batches, sizes=None):
    """merge with the fold-back fix (module docstring "Anchored layout"): batches (oldest first, one frame) -> merged (M, 2) polyline or None.
    sizes (list, optional) receives the merged polyline's length after each batch (0 while there is none)."""
    M = None
    for B in batches:
        B = _clean(B)
        if len(B) and (M is None or len(M) < 2):                         # a first (or one-point) memory, as in merge
            M = B.copy() if M is None or len(B) > 1 else M
        elif len(B):
            u = (M[-1] - M[-2]) / np.hypot(*(M[-1] - M[-2]))
            if len(B) == 1:                                               # one point: appended when it lies beyond the memory's end
                if _on(M, B[0])[2] == 1 and (B[0] - M[-1]) @ u > TOL:
                    M = np.vstack([M, B])
            else:
                r = _on(B, M[-1], u, strict=True)                         # None: no segment of the batch runs with the memory's end
                if r is not None and r[2] == 0 and r[1] <= A_ON:          # the memory's end lies on the batch: append what lies beyond it
                    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(B, axis=0).T))]
                    j = np.flatnonzero(cum > r[0] + TOL)
                    if len(j):
                        M = np.vstack([M, B[j[0]:]])
                elif r is not None and (r[2] == -1 or (r[2] == 0 and (_on(M, B[0], B[1] - B[0], strict=True) or (0, 0, 0))[2] == 1)):
                    M = np.vstack([M, B])                                 # the batch starts beyond the memory's end: a gap
        if sizes is not None:
            sizes.append(0 if M is None else len(M))
    return M


def memory_a(batches_local, poses, anchor, info=None):
    """Anchored layout (DIM_O,) (module docstring): every route message up to the decision, oldest first, as the waypoints in the local frame
    (a (0, 2) array for a message without any) and its local pose (x, y, yaw); anchor = the current rear-axle pose. info (dict, optional)
    receives window's s_geo / s_e / M0 and s_msg: per message with waypoints (index, C_k, extra spacings m, s_k)."""
    Me, sf, trav = _anchored(batches_local, poses, anchor, info)
    if Me is None:
        return to_order(np.zeros(DIM, np.float32), trav)
    return to_order(window(Me, info=info, s_fix=sf).reshape(-1), trav)


def memory_w(batches_local, poses, anchor, v, win=None, info=None):
    """Window layout's raw blocks (module docstring "Window layout"): memory_a's inputs + the fed speed v -> (Fw (DIM,) the block at
    win_offs(v), F0 (DIM,) the anchored block at OFFS (memory_a's), trav). compose_w makes the fed numbers."""
    Me, sf, trav = _anchored(batches_local, poses, anchor, info)
    if Me is None:
        return np.zeros(DIM, np.float32), np.zeros(DIM, np.float32), trav
    return window(Me, s_fix=sf, offs=win_offs(v, win)).reshape(-1), window(Me, info=info, s_fix=sf).reshape(-1), trav


def compose_w(Fw, F0, trav):
    """Window layout: blocks (..., DIM) and trav (...,) -> (..., DIM_W) float32 = the window block, memory_a's scalars of the fixed-grid block,
    cmd_heading of the float16 fixed-grid block (the same numbers offline and served)."""
    F0 = np.asarray(F0, np.float32)
    return np.concatenate([np.asarray(Fw, np.float32), to_order(F0, trav)[..., N_PT_O * NF:], cmd_heading(F0.astype(np.float16))], -1).astype(np.float32)


def _anchored(batches_local, poses, anchor, info=None):
    """memory_a's merge and arc: -> (merged polyline in the rear-axle frame of anchor or None, the arc s_e it tracked, trav)."""
    B = [_clean(b) for b in batches_local]
    Q = [np.asarray(p, np.float64)[:2] for p in poses]
    a = np.asarray(anchor, np.float64)
    trav = travelled(Q, a) if Q else 0.0
    sizes = []
    M = merge_a(B, sizes)
    if M is None or len(M) < 2:
        return None, None, trav
    path = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.array(Q), axis=0).T))]       # chord driven from the first message's pose
    s = pk = None
    m = 0
    hist = []
    for k, b in enumerate(B):
        if not len(b) or sizes[k] < 2:
            continue
        c = _on(M[:sizes[k]], b[0], b[1] - b[0] if len(b) > 1 else None)[0] - L0
        hyp = c - SPC * np.arange(max(0, N_FULL - len(b)) + 1)                    # m more points trimmed (a short message)
        if s is None:
            m = 0
        else:                                                                     # a new count only when it explains the step within A_TRIM
            e = np.abs(hyp - (s + path[k] - pk))
            m = int(np.argmin(e)) if e.min() <= A_TRIM else min(m, len(hyp) - 1)
        s, pk = float(hyp[m]), path[k]
        hist.append((k, c, m, s))
    if info is not None:
        info["s_msg"] = hist
    return in_ego(a, M), s + trav - pk, trav


def summary(f):
    """Features (..., DIM | DIM_O) -> dict of numpy arrays: n valid points, nearest valid offset (nan if none), +20 m covered."""
    P = pts(f)
    v = P[..., 0] > 0.5
    first = np.where(v.any(-1), OFFS[np.argmax(v, -1)], np.nan)
    return dict(n=v.sum(-1), first=first, c20=v[..., I20])


def turn_dist(f):
    """Old-layout features (..., DIM) -> (...,) arc offset (m) of the next heading change >= TURN_DEG within 0 .. +60 m (module docstring),
    TURN_CAP when none and +60 m is covered, nan when unknown."""
    P = np.asarray(f, np.float32).reshape(*np.shape(f)[:-1], N_PT, NF)
    v = P[..., 0] > 0.5
    h = np.arctan2(P[..., 4], P[..., 3])
    ref = np.where(v[..., I0], h[..., I0], 0.0)
    dh = np.abs((h - ref[..., None] + np.pi) % (2 * np.pi) - np.pi)
    c = v & (dh >= np.radians(TURN_DEG))
    c[..., :-2] &= c[..., 2:] | ~v[..., 2:]                                    # persistent over 4 m (a merge kink is one segment, two points)
    c[..., :I0] = False
    c[..., I60 + 1:] = False
    return np.where(c.any(-1), OFFS[np.argmax(c, -1)], np.where(v[..., I60], TURN_CAP, np.nan))


CMD_LOOK, CMD_DEG = 40.0, 20.0                   # round7 joint arm (change 3, decision 432): the command from the route's heading change


def cmd_heading(f):
    """Old-layout features (..., DIM), as fed (training rows: the stored float16 features; the PAI driver rounds its fed memory to float16
    first, so both sides see the same numbers) -> (..., 4) NAVSIM one-hot [left, straight, right, unknown]. The route's own heading change
    against the +0 m slot's heading (the ego's heading in the near gap, as turn_dist): the first slot in 0 .. CMD_LOOK m whose change is
    >= CMD_DEG and persists 4 m on (turn_dist's merge-kink rule) gives LEFT (change > 0) or RIGHT; none (or no valid slot at all) gives
    STRAIGHT, never unknown. Replaces the shipped rule (first waypoint >= 5 m, about 42 m ahead, |y| > 2 m),
    which turns gentle curvature and the ego's own pose into turn commands (route_mem_curv.md)."""
    P = np.asarray(f, np.float32).reshape(*np.shape(f)[:-1], N_PT, NF)
    v = P[..., 0] > 0.5
    h = np.arctan2(P[..., 4], P[..., 3])
    ref = np.where(v[..., I0], h[..., I0], 0.0)
    d = (h - ref[..., None] + np.pi) % (2 * np.pi) - np.pi
    c = v & (np.abs(d) >= np.radians(CMD_DEG))
    c[..., :-2] &= c[..., 2:] | ~v[..., 2:]
    c &= (OFFS >= 0.0) & (OFFS <= CMD_LOOK)
    j = np.argmax(c, -1)
    dj = np.take_along_axis(d, j[..., None], -1)[..., 0]
    k = np.where(c.any(-1), np.where(dj > 0, 0, 2), 1)
    return np.eye(4, dtype=np.float32)[k]


def to_order(f, trav):
    """Old-layout features (..., DIM) and travelled arc (...,) -> ordered features (..., DIM_O) float32."""
    f = np.asarray(f, np.float32)
    d = turn_dist(f)
    ok = np.isfinite(d)
    tr = np.minimum(np.nan_to_num(np.asarray(trav, np.float32)), TRAV_CAP) / 100.0
    sc = np.stack(np.broadcast_arrays(np.where(ok, d / TURN_CAP, 0.0), ok.astype(np.float32), tr), -1)
    return np.concatenate([f[..., :N_PT_O * NF], sc], -1).astype(np.float32)


def to_order_t(f, trav):
    """torch twin of to_order (the trainer's on-the-fly negatives): f (n, DIM), trav (n,) -> (n, DIM_O)."""
    import torch
    P = f.float().view(-1, N_PT, NF)
    v = P[..., 0] > 0.5
    h = torch.atan2(P[..., 4], P[..., 3])
    ref = torch.where(v[:, I0], h[:, I0], torch.zeros_like(h[:, I0]))
    dh = ((h - ref[:, None] + np.pi) % (2 * np.pi) - np.pi).abs()
    c = v & (dh >= float(np.radians(TURN_DEG)))
    c[:, :-2] &= c[:, 2:] | ~v[:, 2:]
    c[:, :I0] = False
    c[:, I60 + 1:] = False
    offs = torch.as_tensor(OFFS, dtype=torch.float32, device=f.device)
    first = offs[c.int().argmax(-1)]
    cap = torch.full_like(first, TURN_CAP)
    d = torch.where(c.any(-1), first, torch.where(v[:, I60], cap, torch.full_like(first, float("nan"))))
    ok = torch.isfinite(d)
    sc = torch.stack([torch.where(ok, d / TURN_CAP, torch.zeros_like(d)), ok.float(), trav.float().nan_to_num().clamp(max=TRAV_CAP) / 100.0], -1)
    return torch.cat([f.float()[:, :N_PT_O * NF], sc], -1)


def travelled(poses, anchor=None):
    """xy (m, 2) of the route messages' poses so far (oldest first) [+ the anchor pose] -> sum of the chords (m)."""
    Q = [np.asarray(q, np.float64)[:2] for q in poses] + ([np.asarray(anchor, np.float64)[:2]] if anchor is not None else [])
    return float(np.hypot(*np.diff(np.array(Q), axis=0).T).sum()) if len(Q) > 1 else 0.0


def transform_rows(f, off):
    """Features (n, DIM) in the logged frame -> the perturbed frame of yaw-rate rows (ot_rows.off_hinge: off = (dy, dpsi) of the row's t0 pose
    in the logged frame)."""
    P = np.asarray(f, np.float32).reshape(-1, N_PT, NF).copy()
    dy, dp = np.asarray(off, np.float64)[:, :1], np.asarray(off, np.float64)[:, 1:]
    c, s = np.cos(dp), np.sin(dp)
    x, y = P[..., 1] * 20.0, P[..., 2] * 20.0 - dy
    v = P[..., 0] > 0.5
    P[..., 1] = np.where(v, (c * x + s * y) / 20.0, 0)
    P[..., 2] = np.where(v, np.clip((-s * x + c * y) / 20.0, -4, 4), 0)
    hc, hs = P[..., 3].copy(), P[..., 4].copy()
    P[..., 3] = np.where(v, c * hc + s * hs, 0)
    P[..., 4] = np.where(v, -s * hc + c * hs, 0)
    return P.reshape(-1, DIM)


class Tracker:
    """Served memory of one driver session: route messages (rig-frame waypoints at their time) and the ego poses -> features at a decision."""

    def __init__(self, order=None, anchor=None, win=None):
        self.routes = {}
        self.order = SERVE_ORDER if order is None else bool(order)
        self.anchor = SERVE_ANCHOR if anchor is None else bool(anchor)        # anchored layout (memory_a; sidecar head `anchor`)
        self.win = SERVE_WIN if win is None else (win or None)                # window layout (memory_w; sidecar head `win`), None = off
        assert self.win is None or self.anchor, "the window layout is the anchored layout's merge and arc"

    def add(self, ts: int, wp):
        self.routes[int(ts)] = np.asarray(wp, np.float64).reshape(-1, 2)

    def features(self, t0: int, pose_of, anchor, v=None):
        """pose_of(ts) -> (x, y, yaw) local rig pose or None; anchor = the pose at t0. Messages after t0 or without a pose are skipped.
        v: the fed speed (speed_of of the ego row), required by the window layout (-> (DIM_W,))."""
        B, Q, last = [], [], None
        BA, PA = [], []                                                        # anchored layout: every message with a pose
        for ts in sorted(self.routes):
            if ts > t0:
                continue
            p = pose_of(ts)
            wp = self.routes[ts]
            wp = wp[np.isfinite(wp).all(1)]
            if p is not None:
                Q.append(np.asarray(p, np.float64)[:2])
            if p is not None and len(wp):
                B.append(to_local(p, wp))
                last = p
            if self.anchor and p is not None:
                BA.append(to_local(p, wp) if len(wp) else np.zeros((0, 2)))
                PA.append(p)
        if self.win is not None:
            assert v is not None, "window layout: the fed speed"
            return compose_w(*memory_w(BA, PA, anchor, v, self.win))
        if self.anchor:
            return memory_a(BA, PA, anchor)
        if not self.order:
            return memory(B, anchor).astype(np.float32)
        return memory_o(B, anchor, last if last is not None else anchor, travelled(Q, anchor) if Q else 0.0)

    def polyline(self, t0: int, pose_of, anchor):
        """The merged route polyline (M, 2) in the rear-axle frame of anchor (same batches as `features`), or None
        (experiments/alpasim/lib/route_path.py: JEV_ROUTE serves this path)."""
        B = []
        for ts in sorted(self.routes):
            p, wp = (pose_of(ts) if ts <= t0 else None), self.routes[ts]
            wp = wp[np.isfinite(wp).all(1)]
            if p is not None and len(wp):
                B.append(to_local(p, wp))
        M = merge_a(B) if self.anchor else merge(B)                            # an anchored tracker serves the fold-back fix here too
        return None if M is None else in_ego(anchor, M)


# ---------------------------------------------------------------- torch branch
def _torch():
    import torch
    import torch.nn as nn
    import parity_adapter as PA

    class RouteHead(nn.Module):
        def __init__(self, d=128, layers=2, heads=4, T=32, D=512, order=False, latfeat=False, anchor=False, win=None, cmd=None):
            super().__init__()
            assert order or not anchor, "anchor: the ordered slot format (order=True)"
            assert win is None or anchor, "win: the anchored layout's slot format"   # window layout: no tensor of its own (fed DIM_O numbers)
            self.anchor = bool(anchor)                       # the anchored layout feeds the ordered format: no tensor of its own
            self.n_pt = N_PT_O if order else N_PT
            self.dim = DIM_O if order else DIM
            self.pt = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
            self.pos = nn.Parameter(0.02 * torch.randn(self.n_pt, d))
            self.ego = nn.Sequential(nn.Linear(EGO_DIM - 1, d), nn.GELU(), nn.Linear(d, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.q = nn.Parameter(0.02 * torch.randn(T, d))
            layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            if order:                                        # the scalars: zero-initialised, added to the ego token
                self.sc = nn.Linear(NSC, d)
                nn.init.zeros_(self.sc.weight)
                nn.init.zeros_(self.sc.bias)
            self.latfeat = bool(latfeat)
            if self.latfeat:                                 # --rm-latfeat: lateral offsets of the route at 10 / 20 / 30 / 40 m, zero-initialised
                assert order, "latfeat: the ordered layout (slots at their true arc)"
                self.lat = nn.Linear(2 * len(LAT_I), d)
                nn.init.zeros_(self.lat.weight)
                nn.init.zeros_(self.lat.bias)

        def forward(self, ego, f):
            B, d = ego.shape[0], self.q.shape[1]
            if self.dim == DIM:
                P = f.float().view(B, N_PT, NF)
                e = self.ego(ego[:, 1:].float())
            else:
                P = f.float()[:, :self.n_pt * NF].reshape(B, self.n_pt, NF)
                e = self.ego(ego[:, 1:].float()) + self.sc(f.float()[:, self.n_pt * NF:])
                if self.latfeat:
                    e = e + self.lat(lat_feats(P))
            pad = P[..., 0] < 0.5
            mem = torch.cat([self.null.expand(B, 1, d), e[:, None], self.pt(P) + self.pos[None]], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=ego.device), pad], 1)
            h = self.dec(self.q[None].expand(B, -1, -1), mem, memory_key_padding_mask=pad)
            return self.out(self.norm(h))

    class RouteParityAdapter(PA._torch()):
        """ParityAdapter (use_ego only) + RouteHead on the route features appended to the ego row."""

        def __init__(self, **kw):
            super().__init__(use_ego=True, use_side=False)
            self.route = RouteHead(**kw)

        def forward(self, ego, side=None, side_mask=None, lead=None):
            ego = ego.float()
            e, f = ego[:, :EGO_DIM], ego[:, EGO_DIM:]
            b = super().forward(e)
            if f.shape[1] == self.route.dim:
                b = b + self.route(e if self.in_tf is None else self.in_tf(e), f) * e[:, :1, None]
            return b

    return RouteParityAdapter


def attach(model, **kw):
    """Swap a P2 PModel's adapter for the route branch, keeping every existing adapter tensor (the branch starts at an exact zero)."""
    import torch
    old = model.adapter
    assert old is not None and old.use_ego and not old.use_side and not getattr(old, "use_lead", False), "route branch: a P2 adapter"
    dev = next(old.parameters()).device
    g = torch.random.get_rng_state()
    torch.manual_seed(1234)                                  # the branch's init draws do not touch the training streams
    new = _torch()(**kw).to(dev)
    torch.random.set_rng_state(g)
    miss = new.load_state_dict(old.state_dict(), strict=False)
    assert not miss.unexpected_keys and all(k.startswith("route.") for k in miss.missing_keys), miss
    new.in_tf = old.in_tf
    model.adapter = new
    return model


def order_state(sd: dict, latfeat: bool = False) -> dict:
    """An old-layout adapter state (with `route.*`) -> the ordered head's: every tensor kept (the same 40 slots), the scalar map at 0
    (and the --rm-latfeat map at 0)."""
    import torch
    out = dict(sd)
    k = next(k for k in sd if k.endswith("route.pos"))
    pre, d = k[:-len("pos")], sd[k].shape[1]
    if pre + "sc.weight" not in out:
        out[pre + "sc.weight"] = torch.zeros(d, NSC, dtype=sd[k].dtype)
        out[pre + "sc.bias"] = torch.zeros(d, dtype=sd[k].dtype)
    if latfeat and pre + "lat.weight" not in out:
        out[pre + "lat.weight"] = torch.zeros(d, 2 * len(LAT_I), dtype=sd[k].dtype)
        out[pre + "lat.bias"] = torch.zeros(d, dtype=sd[k].dtype)
    return out


def head_order(tag: str) -> bool:
    import pp_train as T
    p = T.proot("runs", tag) / SIDECAR
    return bool(p.exists() and json.loads(p.read_text()).get("head", {}).get("order", False))


def head_anchor(tag: str) -> bool:
    """The checkpoint's sidecar head has `anchor` (the anchored layout, memory_a): the drivers set SERVE_ANCHOR from it."""
    import pp_train as T
    p = T.proot("runs", tag) / SIDECAR
    return bool(p.exists() and json.loads(p.read_text()).get("head", {}).get("anchor", False))


def head_win(tag: str):
    """The checkpoint's sidecar head `win` (the window layout's dict; the drivers set SERVE_WIN from it), else None."""
    import pp_train as T
    p = T.proot("runs", tag) / SIDECAR
    return json.loads(p.read_text()).get("head", {}).get("win") if p.exists() else None


def dim_of(tag: str) -> int:
    return DIM_O if head_order(tag) else DIM


def has_branch(tag: str) -> bool:
    import pp_train as T
    return (T.proot("runs", tag) / SIDECAR).exists()


def load_pmodel(tag: str, dev):
    """pp_train.load_pmodel for a checkpoint with the route branch (its sidecar), else pp_train.load_pmodel itself."""
    import torch
    import pp_train as T
    if not has_branch(tag):
        return T.load_pmodel(tag, dev)
    side = json.loads((T.proot("runs", tag) / SIDECAR).read_text())
    ck = torch.load(T.proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
    m = T.PModel(ck["model"]["arm"]).to(dev)
    attach(m, **side.get("head", {}))
    if side.get("reflex") is not None:                       # STACK reflex module (lib/reflex.py): the adapter state carries rx.*
        import reflex as RX
        RX.attach(m, rx=side["reflex"].get("rx"))
    m.load_state(ck["model"])
    return m.eval()


def reflex_of(tag: str) -> dict | None:
    """The sidecar's `reflex` entry (lib/reflex.py SIDE_KEY: the module's config) of a checkpoint, None without the module."""
    import pp_train as T
    p = T.proot("runs", tag) / SIDECAR
    return json.loads(p.read_text()).get("reflex") if p.exists() else None


def write_sidecar(tag: str, head: dict, extra: dict):
    import pp_train as T
    (T.proot("runs", tag) / SIDECAR).write_text(json.dumps(dict(head=head, n_pt=N_PT, nf=NF, offs=OFFS.tolist()) | extra, indent=1))


def empty() -> np.ndarray:
    return np.zeros(DIM, np.float32)


if __name__ == "__main__":                                   # self-test: straight and curved routes, carried batches, perturbed frames
    a0 = np.array([0.0, 0.0, 0.0])
    B0 = np.c_[np.arange(42.1, 81, 4.21), np.zeros(10)]
    f = memory([B0], a0).reshape(N_PT, NF)
    assert f[OFFS < 42, 0].sum() == 0 and f[(OFFS >= 43) & (OFFS <= 72), 0].all(), f[:, 0]
    B1 = np.c_[np.arange(42.1, 81, 4.21) + 50, np.zeros(10)]
    f = memory([B0, B1], np.array([50.0, 0.0, 0.0])).reshape(N_PT, NF)
    assert f[OFFS >= -6, 0].all() and np.allclose(f[:, 1] * 20, OFFS) and np.allclose(f[:, 2], 0)
    r, th = 30.0, np.linspace(0, np.pi / 2, 200)                             # quarter circle, left turn
    C = np.c_[r * np.sin(th), r - r * np.cos(th)]
    f = memory([C], np.array([0.0, 0.0, 0.0])).reshape(N_PT, NF)
    v = f[:, 0] > 0
    assert np.allclose(np.hypot(f[v, 1] * 20, f[v, 2] * 20 - r), r, atol=0.05) and (f[v, 2] >= -1e-6).all()
    off = np.array([[0.5, 0.05]])
    g = transform_rows(memory([B0, B1], np.array([50.0, 0.0, 0.0]))[None], off).reshape(N_PT, NF)
    x, y = g[:, 1] * 20, g[:, 2] * 20                                        # back to the logged frame: x c - y s, dy + x s + y c
    c, s = np.cos(0.05), np.sin(0.05)
    assert np.allclose(0.5 + x * s + y * c, 0, atol=1e-4) and np.allclose(x * c - y * s, OFFS, atol=1e-4)
    o = to_order(memory([C], np.array([0.0, 0.0, 0.0]))[None], np.array([0.0]))[0]           # quarter circle r 30 m: 30 deg at arc ~15.7 m
    assert o.shape == (DIM_O,) and abs(o[-3] * TURN_CAP - 16.0) < 1e-6 and o[-2] == 1 and o[-1] == 0, o[-3:]
    o = to_order(memory([B0], a0)[None], np.array([12.5]))[0]                                # gap: first point at +43 m, straight, 60 m covered
    assert o[-3] == 1.0 and o[-2] == 1 and abs(o[-1] - 0.125) < 1e-6 and (pts(o)[OFFS < 42, 0] == 0).all()
    # progress anchor: a route that turns left 90 deg (r 10 m) 20 m ahead; the first message's points start at arc L0 (on the exit leg)
    th = np.linspace(0, np.pi / 2, 400)
    Rt = np.r_[np.c_[np.linspace(-10, 20, 300), np.zeros(300)], np.c_[20 + 10 * np.sin(th), 10 - 10 * np.cos(th)], np.c_[np.full(800, 30.0), 10 + np.linspace(0.01, 80, 800)]]
    sR = np.r_[0, np.cumsum(np.hypot(*np.diff(Rt, axis=0).T))]
    s0 = float(np.interp(0.0, Rt[:, 0], sR))                                                 # the ego's arc on the route
    W = np.c_[np.interp(s0 + L0 + 80 / 19 * np.arange(10), sR, Rt[:, 0]), np.interp(s0 + L0 + 80 / 19 * np.arange(10), sR, Rt[:, 1])]
    og, oo = memory([W], a0).reshape(N_PT, NF), memory_o([W], a0, a0, 0.0)
    first_g, first_o = OFFS[np.argmax(og[:, 0] > 0.5)], OFFS[np.argmax(pts(oo)[:, 0] > 0.5)]
    assert abs(first_o - L0) <= 2.0 and first_g < L0 - 5, (first_g, first_o)                # projection: too near; progress: at its arc
    v = pts(oo)[:, 0] > 0.5
    assert np.allclose(np.hypot(pts(oo)[v, 1] * 20 - np.interp(OFFS[v] + s0, sR, Rt[:, 0]), pts(oo)[v, 2] * 20 - np.interp(OFFS[v] + s0, sR, Rt[:, 1])), 0, atol=0.1)
    print(f"progress anchor: first slot {first_o:.0f} m (true {L0:.1f}), projection {first_g:.0f} m")
    assert np.isnan(turn_dist(np.zeros(DIM))) and (to_order(np.zeros((2, DIM)), np.zeros(2))[:, -3:] == 0).all()
    tr = Tracker(order=True)
    tr.add(0, B0)
    tr.add(1, B0)
    ps = {0: (0.0, 0.0, 0.0), 1: (3.0, 4.0, 0.0)}
    o = tr.features(1, ps.get, np.array([3.0, 4.0, 0.0]))
    assert o.shape == (DIM_O,) and abs(o[-1] - 0.05) < 1e-6, o[-3:]
    Bl = [to_local(ps[0], B0), to_local(ps[1], B0)]
    assert np.array_equal(o, memory_o(Bl, np.array([3.0, 4.0, 0.0]), ps[1], 5.0))
    assert abs(travelled([(0, 0), (3, 4)], (6, 8)) - 10.0) < 1e-9
    # anchored layout. Straight route along x (car arc = x): messages at 2 Hz from the car's arc, i0 points trimmed as AlpaSim does
    msg = lambda s, i0=10: np.c_[s + 80 / 19 * np.arange(i0, 20), np.zeros(20 - i0)]   # noqa: E731
    pz = [np.array([s, 0.0, 0.0]) for s in (0.0, 3.0, 6.0)]
    mz = [msg(0.0), msg(3.0), msg(6.0)]
    for bb in ([B0], [B0, B1], [B0, B0], mz):                                     # no fold: merge_a = merge
        assert np.array_equal(merge(bb), merge_a(bb)), "merge_a = merge without a fold"
    fa = memory_a(mz, pz, pz[-1])
    assert fa.shape == (DIM_O,) and np.allclose(fa, memory_o(mz, pz[-1], pz[-1], 6.0), atol=1e-5), "memory_a = memory_o without an offset"
    lead = lambda f: OFFS[np.argmax(pts(f)[:, 0] > 0.5)] - pts(f)[np.argmax(pts(f)[:, 0] > 0.5), 1] * 20.0   # noqa: E731  (label - true x)
    fx = memory_o([mz[0], mz[1], msg(6.0, 11)], pz[-1], pz[-1], 6.0)               # the newest message trimmed one spacing late
    fy = memory_a([mz[0], mz[1], msg(6.0, 11)], pz, pz[-1])
    assert abs(lead(fx) + SPC) < 0.05 and abs(lead(fy)) < 0.05, (lead(fx), lead(fy))   # ordered 4.2 m off, anchored at the car
    for bb, m_ in (([msg(0.0, 11)], 0), ([mz[0], mz[1][:8]], 0), ([mz[0], msg(3.0, 12)], 2)):   # a short first message (m = 0: a trim reads
        inf = {}                                                                 # SPC ahead), a later far end cut by two points, a later two-spacing trim
        f_ = memory_a(bb, pz[:len(bb)], pz[len(bb) - 1], inf)
        assert inf["s_msg"][-1][2] == m_ and abs(lead(f_) + (SPC if len(bb) == 1 else 0.0)) < 0.05, (inf["s_msg"], lead(f_))
    tr = Tracker(anchor=True)
    for t_, p_, b in zip((0, 500_000, 1_000_000), pz, mz):
        tr.add(t_, in_ego(p_, b))
    ps = dict(zip((0, 500_000, 1_000_000), pz))
    assert np.array_equal(tr.features(1_000_000, ps.get, pz[-1]), memory_a([to_local(p_, in_ego(p_, b)) for p_, b in zip(pz, mz)], pz, pz[-1]))
    # U-turn (r 6 m at route arc 60 .. 78.8): message 1 ends 5 m into the return leg, message 2 (5 m later) starts on the approach
    th = np.linspace(0, np.pi, 300)
    U = np.r_[np.c_[np.linspace(-60, 0, 601), np.zeros(601)], np.c_[6 * np.sin(th[1:]), 6 - 6 * np.cos(th[1:])], np.c_[-np.linspace(0, 60, 601)[1:], np.full(600, 12.0)]]
    sU = np.r_[0, np.cumsum(np.hypot(*np.diff(U, axis=0).T))]
    at = lambda s: np.c_[np.interp(s, sU, U[:, 0]), np.interp(s, sU, U[:, 1])]   # noqa: E731
    st = lambda P: np.array([sU[np.argmin(np.hypot(*(U - p).T))] for p in np.atleast_2d(P)])   # noqa: E731
    pu = [np.r_[at(s)[0], 0.0] for s in (3.85, 8.85)]
    mu = [at(s + 80 / 19 * np.arange(10, 20)) for s in (3.85, 8.85)]
    assert np.diff(st(merge(mu))).min() < -20 and np.diff(st(merge_a(mu))).min() > -0.5, "merge folds back at the U-turn, merge_a does not"
    P_ = pts(memory_a(mu, pu, pu[1]))
    v = P_[:, 0] > 0.5
    err = np.abs(st(to_local(pu[1], P_[v, 1:3] * 20.0)) - 8.85 - OFFS[v]).max()          # slot point's route arc - (car arc + label)
    assert v.sum() >= 15 and err < 1.0, (v.sum(), err)
    # a tight U-turn (r 3 m at arc 50 .. 59.4) inside the first message: a later 3-point message on the approach runs against the memory's
    # end on the return leg, so it adds nothing (without the strict direction test merge_a found the end before the batch: a gap, appended)
    th3 = np.linspace(0, np.pi, 200)
    V = np.r_[np.c_[np.linspace(-50, 0, 501), np.zeros(501)], np.c_[3 * np.sin(th3[1:]), 3 - 3 * np.cos(th3[1:])], np.c_[-np.linspace(0, 60, 601)[1:], np.full(600, 6.0)]]
    sV = np.r_[0, np.cumsum(np.hypot(*np.diff(V, axis=0).T))]
    atV = lambda s: np.c_[np.interp(s, sV, V[:, 0]), np.interp(s, sV, V[:, 1])]   # noqa: E731
    mv = [atV(0.0 + 80 / 19 * np.arange(10, 20)), atV(3.0 + 80 / 19 * np.arange(10, 13))]
    uv = merge_a(mv[:1])[-1] - merge_a(mv[:1])[-2]
    assert _on(mv[1], mv[0][-1], uv)[2] == -1 and _on(mv[1], mv[0][-1], uv, strict=True) is None and len(merge_a(mv)) == len(merge(mv)) == 10
    # the trim count changes only when a hypothesis explains the step within A_TRIM: the projection outran the chord by 2.3 m while the far
    # end lost a point (route end) -> m stays 0 (the nearest hypothesis, m = 1, misses by 1.9 m)
    inf = {}
    memory_a([mz[0], mz[1], np.c_[8.3 + 80 / 19 * np.arange(10, 19), np.zeros(9)]], pz, pz[-1], inf)
    assert inf["s_msg"][-1][2] == 0, inf["s_msg"]
    print(f"anchored layout: merge_a = merge without a fold, no fold-back at a U-turn (slots within {err:.2f} m of their arc); a one-spacing "
          f"trim {lead(fx):+.2f} m (ordered) -> {lead(fy):+.2f} m (anchored)")
    # window layout: the fixed-grid block is memory_a's, the window block sits at win_offs(v) along the same arc, the command is cmd_heading's
    assert np.allclose(win_offs([0.0, 5.0, 10.0, 30.0])[:, -1], [40.0, 45.0, 70.0, 78.0]) and np.allclose(win_offs(0.0)[0], -6.0)
    for bb, pp, v_ in ((mz, pz, 3.0), (mz, pz, 12.0), (mu, pu, 25.0), ([np.zeros((0, 2))], pz[:1], 9.0)):
        Fw, F0, tv_ = memory_w(bb, pp, pp[-1], v_)
        fw = compose_w(Fw, F0, tv_)
        assert fw.shape == (DIM_W,) and np.array_equal(fw[N_PT * NF:DIM_O], memory_a(bb, pp, pp[-1])[N_PT * NF:])
        assert np.array_equal(F0, memory_a(bb, pp, pp[-1])[:N_PT * NF]) and np.array_equal(fw[DIM_O:], cmd_heading(F0.astype(np.float16)))
        Pw = Fw.reshape(N_PT, NF)
        if Pw[:, 0].any() and bb is mz:                                          # straight route along x: slot x = car arc + offset
            vv = Pw[:, 0] > 0.5
            assert vv.any() and np.allclose(Pw[vv, 1] * 20.0, win_offs(v_)[vv], atol=1e-4), (Pw[:, 1] * 20, win_offs(v_))
    thq = np.linspace(0, np.pi / 2, 300)                                          # straight to x 45, then a left quarter circle (r 30 m)
    Q = np.r_[np.c_[np.linspace(-10, 45, 551)[:-1], np.zeros(550)], np.c_[45 + 30 * np.sin(thq), 30 - 30 * np.cos(thq)]]
    sQ = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    atQ = lambda s_: np.c_[np.interp(s_, sQ, Q[:, 0]), np.interp(s_, sQ, Q[:, 1])]   # noqa: E731
    pq = [np.array([x_, 0.0, 0.0]) for x_ in (0.0, 10.0, 20.0)]
    mq = [atQ(x_ + 10 + 80 / 19 * np.arange(10, 20)) for x_ in (0.0, 10.0, 20.0)]
    fq = compose_w(*memory_w(mq, pq, pq[-1], 8.0))                              # the turn starts 25 m ahead of the car: LEFT
    fs = compose_w(*memory_w(mq[:1], pq[:1], pq[0], 8.0))                        # 45 m ahead: STRAIGHT
    assert fq[DIM_O] == 1.0 and fs[DIM_O + 1] == 1.0, (fq[DIM_O:], fs[DIM_O:])
    tr = Tracker(anchor=True, win=WIN)
    for t_, p_, b in zip((0, 500_000, 1_000_000), pz, mz):
        tr.add(t_, in_ego(p_, b))
    assert np.array_equal(tr.features(1_000_000, ps.get, pz[-1], v=12.0), compose_w(*memory_w([to_local(p_, in_ego(p_, b)) for p_, b in zip(pz, mz)], pz, pz[-1], 12.0)))
    print(f"window layout: offsets at 0 / 10 / 30 m/s end at {win_offs([0, 10, 30])[:, -1]}, fixed-grid block = memory_a's, command LEFT on the r 30 m turn")
    try:
        import torch
        F = np.stack([memory([C], np.array([0.0, 0.0, 0.0])), memory([B0], a0), memory([B0, B1], np.array([50.0, 0.0, 0.0])), np.zeros(DIM, np.float32)])
        tv = np.array([0.0, 12.5, 50.0, 3.0], np.float32)
        assert np.allclose(to_order_t(torch.as_tensor(F), torch.as_tensor(tv)).numpy(), to_order(F, tv), atol=1e-6)
        print("to_order_t == to_order")
    except ImportError:
        pass
    print("rmem selftest ok", summary(memory([B0], a0)))
