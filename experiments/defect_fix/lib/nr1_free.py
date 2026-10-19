"""NR1FREE: one-sided longitudinal consequence term on the pool's S rows (rm_train --on-free W; NR1 prereg supplement 15; practice plan item 5,
research/next-round/practice-plan.md). No target speed: the plan's 4 s arc is penalised only when it is below FRAC (0.8) x the human's 4 s arc
AND the way ahead is free. Nothing when a logged object is in the way (the agent hinge handles a lead). Carried by nr1_lat (the pool's one
--on-lon module): `lon_free` = 10 x W x term, the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every NR1 term.

Reference, per row and 0.5 s step j: s_ref_j = min(arc(fut)_j, arc(fut_reg)_j): the smaller of the human's logged travel (tab `fut_reg`, the
log's own speed from the same log time) and the same speed profile made reachable from the visited state (tab `fut`: the ego's speed blended
into the log's over the first 2 s; nr1_lon). The reference is never above what the human did nor above what the ego could reach from its state.
Rows: the S rows (tab mode `self`) with finite fut / fut_reg and agent labels; trainval imitation rows get 0.

Free (on the detached plan; no gradient): for every step j = 1..8 the ego box (nuPlan Pacifica about the rear axle, front edge moved forward by
the safety distance D_j = D0 + TH x v_ref_j, v_ref_j = the reference's speed on segment j) is placed along the plan's path (the 8 poses from the
origin, extended straight past the last pose along its heading) at NS arc positions from min(s_plan_j, s_ref_j) to max(s_plan_j, s_ref_j), each
at the path heading there. The row is free when no logged object box at time j (agent.npz: vehicle / generic object / pedestrian / bicycle,
valid at j, centre ahead of the placed plain box's centre) overlaps any placed box (separating-axis test, exact for rectangles).

Fire (the trigger is the 4 s arc): free & s_ref_8 >= S_MIN & s_plan_8 < FRAC x s_ref_8. Value on fire rows only: the sum over the 8 steps of
smooth L1 (beta 0.5) of relu(FRAC x s_ref_j - s_plan_j) (per step, as nr1_lon sums over the poses: the 4 s arc alone would move only the last
pose). Gradient: each pose moves only along its own heading (nr1_lon.lon_only; none across it, none on the yaw), as nr1_lon.

NR1FREE-M (prereg supplement 16; rm_train --on-free-mode move --on-free-frac 0.95): the same term, the band FRAC tightened and the rows limited to
moving states: the ego's t0 speed (tab `speed`) >= V_MOVE and the human neither near a stop nor braking (every 0.5 s segment speed of `fut_reg`
>= max(V_MOVE, HUM_KEEP x his first segment's)); folded into `ok` at load (no gradient). Default mode "all", FRAC 0.8 = supplement 15 as it was.
Supplement 19 (rm_train --on-free-clear CMIN,CTH): a stricter free check, the safety distance D_j = max(D0 + TH x v_ref_j, CMIN, CTH x v_ref_j):
no logged object within max(CMIN m, CTH s x v) ahead of the bumper along the sweep. Default 0,0 = off (no operation added).

NR1STAND (prereg supplement 17; rm_train --on-stand W --on-stand-margin M): the mirror term on standing states. Rows: S rows where the visited
state stands (tab `speed` < V_STAND) and the human stands all 4 s (arc(fut_reg)_8 < H_STAND); no free check. Cap b_j = max(arc(fut)_j,
arc(fut_reg)_j) + M (the larger human path, so neither of his paths fires). Fire: s_plan_8 > b_8. Value: sum over the 8 steps of smooth L1
(beta 0.5) of relu(s_plan_j - b_j); gradient only along each pose's heading. No target speed, nothing inside the cap.
STAND-R (prereg supplement 20; rm_train --on-stand-roll V): the same term also on rolling rows, V_STAND <= ego t0 speed < V_ROLL, the human
standing all 4 s and a finite station gap g = s_h(t + 4 s) - s_e (the human's standing station minus the visited state's, from
<pool>/stand_gap.npz, scripts/stand_r.py gap = stop_label's h4c). Cap c_j = max(b_j, max(g, 0) + M): a rolling ego behind the human may
roll up to his standing station + M, one level with or past it gets b_j (stop within what `fut`'s 2 s blend reaches). V_ROLL = 0 (default):
nothing loaded, the pack and the term exactly as supplement 17. nr1_lat logs the rolling rows' part apart as lon_standr.

NR1STOPPT (prereg supplement 18; rm_train --on-stoppt W): "do not pass the human's stop point" on approach states. Labels from the log
(stop_label, built offline into <pool>/stoppt.npz by scripts/stoppt.py labels): rest = the human's station, relative to the ego's, at the first
log time tau >= t with v_h(tau) < V_REST and s_h(tau) >= s_e (clock-consistent: a point he left before t does not count); rows: S rows, ego t0
speed >= V_APP, the human has not passed rest + M_STOP at t + 4 s nor 4 s after he was where the ego is, no logged object at the stop (label
file). Cap = max(rest + M_STOP, v0^2 / (2 A_MAX)), the same at every step. Fire: s_plan_8 > cap. Value: sum over the 8 steps of smooth L1
(beta 0.5) of relu(s_plan_j - cap); gradient only along each pose's heading. No target speed, nothing below the cap.

TURN-CAP (prereg supplement 21; rm_train --on-turncap W): a one-sided upper limit on moving S rows in sharp turns (decision 395: the base
model accelerates out of tight turns and plans wider when fast). Rows: S rows, ego t0 speed (tab `speed`) >= V_TC, the human moving (logged
4 s arc of `fut_reg` >= H_TC) and his logged path turning: the heading range of the path over [s_e - TC_BACK, s_e + TC_AHEAD] (1 m samples,
<pool>/turn_cap.npz from scripts/turn_cap.py label, the rows meta `gpath`) >= TURN_TC deg. Cap c_j = K_TC x max(arc(fut)_j, arc(fut_reg)_j):
never below either human arc (`fut` gives the ego 2 s to blend from its own speed to the human's). Fire: s_plan_8 > c_8. Value: sum over the
8 steps of smooth L1 (beta 0.5) of relu(s_plan_j - c_j) (= stand_terms with this cap); gradient only along each pose's heading.
"""
from __future__ import annotations

import numpy as np

FRAC, S_MIN, D0, TH, NS, BETA = 0.8, 1.0, 2.0, 0.5, 8, 0.5
MODE, V_MOVE, HUM_KEEP = "all", 3.0, 0.8
CLEAR_MIN, CLEAR_TH = 0.0, 0.0                    # supplement 19's stricter free check (0, 0 = off)
FRONT, REAR, HALF_W = 4.049, 1.127, 1.1485        # nuPlan Pacifica about the rear axle (= lib/agent_hinge.py)
DT = 0.5
V_STAND, H_STAND, MARGIN = 0.5, 0.5, 1.0
V_ROLL = 0.0                                      # supplement 20: upper ego speed of the rolling STAND rows (0 = off)
V_REST, V_APP, M_STOP, A_MAX, H_FUT, LOOK = 0.5, 0.5, 1.0, 4.0, 4.0, 30.0
V_TC, H_TC, TURN_TC, K_TC, TC_BACK, TC_AHEAD = 2.0, 4.0, 45.0, 1.05, 10, 20     # supplement 21 (TURN-CAP)


def cum_arc(P):
    """(N, 8, >=2) -> (N, 8) cumulative polyline arc from the origin (numpy; = nr1_lon.human_arc)."""
    P = np.concatenate([np.zeros((len(P), 1, 2)), np.asarray(P, np.float64)[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)


def configure(frac=0.8, mode="all", clear=(0.0, 0.0)):
    global FRAC, MODE, CLEAR_MIN, CLEAR_TH
    assert mode in ("all", "move") and 0 < frac <= 1 and min(clear) >= 0, (frac, mode, clear)
    FRAC, MODE, (CLEAR_MIN, CLEAR_TH) = float(frac), str(mode), map(float, clear)
    return dict(frac=FRAC, mode=MODE) | (dict(clear_min=CLEAR_MIN, clear_th=CLEAR_TH) if CLEAR_MIN or CLEAR_TH else {})


def seg_speed(P):
    """(N, 8, >=2) -> (N, 8) speed on each 0.5 s segment of the polyline from the origin."""
    return np.diff(cum_arc(P), axis=1, prepend=0.0) / DT


def moving(v0, v_hum):
    """ego t0 speed (N,), the human's segment speeds (N, 8) -> (N,) bool: the rows supplement 16 lets fire (module docstring)."""
    v_hum = np.asarray(v_hum, np.float64)
    fin = np.isfinite(v_hum).all(1)
    vh = np.nan_to_num(v_hum)
    return fin & (np.asarray(v0) >= V_MOVE) & (vh.min(1) >= np.maximum(V_MOVE, HUM_KEEP * vh[:, 0]))


def ref_arc(fut, fut_reg):
    """-> s_ref (N, 8) = min of the two cumulative arcs, ok (N,) finite."""
    a, b = cum_arc(fut), cum_arc(fut_reg)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return np.nan_to_num(np.minimum(a, b)).astype(np.float32), ok


def load(pool_dir, names, dev, ah):
    """-> pack: s_ref (N, 8), ok (N,) on dev for the pool rows (S rows with labels); `ah` = the pool's lib/agent_hinge.AgentHinge (boxes)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: tab rows != the pool's"
    s_ref, fin = ref_arc(z["fut"], z["fut_reg"])
    S = z["mode"].astype(str) == "self" if "mode" in z.files else np.ones(len(fin), bool)
    ok = fin & S & ah.ok.cpu().numpy()
    mv = moving(z["speed"], seg_speed(z["fut_reg"])) if MODE == "move" else np.ones(len(ok), bool)
    ok_all, ok = ok, ok & mv
    rec = dict(mode=MODE, moving_of_labelled=int(ok.sum()), labelled_all=int(ok_all.sum()),
               rows=int(len(ok)), s_rows=int(S.sum()), labelled=int(ok.sum()), ref_below_smin=int((ok & (s_ref[:, -1] < S_MIN)).sum()),
               ref_is_reg=int((ok & (cum_arc(z["fut_reg"])[:, -1] < cum_arc(z["fut"])[:, -1])).sum()), frac=FRAC, d0=D0, th=TH)
    return dict(s_ref=torch.as_tensor(s_ref, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


def _place(P, s):
    """P (n, 8, 3) detached plan poses, s (n, Q) arc positions -> x, y, yaw (n, Q) on the polyline from the origin, extended straight past
    the last pose along its heading; the yaw is the pose yaw interpolated by arc (the last pose's past the end)."""
    import torch
    n = len(P)
    P0 = torch.cat([torch.zeros_like(P[:, :1]), P], 1)                                   # (n, 9, 3)
    seg = torch.hypot(*(P0[:, 1:, :2] - P0[:, :-1, :2]).unbind(-1))                     # (n, 8)
    C = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)                      # (n, 9)
    i = torch.searchsorted(C.contiguous(), s.contiguous(), right=True).clamp(1, 8)       # segment i-1 .. i
    c0, c1 = C.gather(1, i - 1), C.gather(1, i)
    t = ((s - c0) / (c1 - c0).clamp_min(1e-6)).clamp(0, 1)
    g = lambda k, j: P0[..., k].gather(1, j)  # noqa: E731
    x = g(0, i - 1) + t * (g(0, i) - g(0, i - 1))
    y = g(1, i - 1) + t * (g(1, i) - g(1, i - 1))
    yaw = g(2, i - 1) + t * (g(2, i) - g(2, i - 1))
    past = (s - C[:, -1:]).clamp_min(0)                                                  # beyond the last pose
    ye = P[:, -1:, 2].expand(n, s.shape[1])
    x, y = x + past * torch.cos(ye), y + past * torch.sin(ye)
    yaw = torch.where(past > 0, ye, yaw)
    return x, y, yaw


def _sat(ax, ay, ayaw, ahl, ahw, bx, by, byaw, bhl, bhw):
    """Overlap of oriented boxes A (centre ax, ay, heading ayaw, half extents) and B (broadcast), separating-axis test -> bool."""
    import torch
    dx, dy = bx - ax, by - ay
    ca, sa, cb, sb = torch.cos(ayaw), torch.sin(ayaw), torch.cos(byaw), torch.sin(byaw)
    sep = torch.zeros_like(dx, dtype=torch.bool)
    for ux, uy in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        ra = ahl * (ca * ux + sa * uy).abs() + ahw * (-sa * ux + ca * uy).abs()
        rb = bhl * (cb * ux + sb * uy).abs() + bhw * (-sb * ux + cb * uy).abs()
        sep = sep | ((dx * ux + dy * uy).abs() > ra + rb)
    return ~sep


def free(P, s_plan, s_ref, box, val):
    """P (n, 8, 3) plan poses (detached), s_plan / s_ref (n, 8), box (n, 9, K, 5) x, y, yaw, length, width in the row's rear-axle frame at t0
    (times 0, 0.5 .. 4 s), val (n, 9, K) -> (n,) bool: no logged object in the extended sweep at any step (module docstring)."""
    import torch
    n = len(P)
    lo, hi = torch.minimum(s_plan, s_ref), torch.maximum(s_plan, s_ref)                  # (n, 8)
    f = torch.linspace(0, 1, NS, device=P.device)
    s = (lo[..., None] + (hi - lo)[..., None] * f).reshape(n, 8 * NS)                    # (n, 8 NS), step-major
    x, y, yaw = (q.view(n, 8, NS) for q in _place(P, s))
    v = torch.diff(s_ref, dim=1, prepend=torch.zeros_like(s_ref[:, :1])) / DT
    D = D0 + TH * v
    if CLEAR_MIN or CLEAR_TH:                                                           # supplement 19
        D = torch.maximum(D, (CLEAR_TH * v).clamp_min(CLEAR_MIN))
    D = D[..., None]                                                                    # (n, 8, 1)
    c, sn = torch.cos(yaw), torch.sin(yaw)
    off = (FRONT + D - REAR) / 2                                                        # extended box centre ahead of the rear axle
    ex, ey = x + c * off, y + sn * off
    ehl = (FRONT + D + REAR) / 2
    b = box[:, 1:]                                                                      # (n, 8, K, 5) at 0.5 .. 4 s
    ok = val[:, 1:]
    pc = (FRONT - REAR) / 2                                                             # plain box centre
    lon = c[..., None] * (b[..., 0][:, :, None] - (x + c * pc)[..., None]) + sn[..., None] * (b[..., 1][:, :, None] - (y + sn * pc)[..., None])
    hit = _sat(ex[..., None], ey[..., None], yaw[..., None], ehl[..., None], HALF_W, b[..., 0][:, :, None], b[..., 1][:, :, None],
               b[..., 2][:, :, None], b[..., 3][:, :, None] / 2, b[..., 4][:, :, None] / 2)          # (n, 8, NS, K)
    hit = hit & ok[:, :, None] & (lon > 0)
    return ~hit.flatten(1).any(1)


def terms(poses, s_ref, ok, box, val, parts=False):
    """poses (n, 8, 3) rear-axle x, y, yaw -> (n,) raw term (module docstring); parts: also fire, free, slow masks and s_plan_8."""
    import torch
    import torch.nn.functional as F
    import nr1_lon
    p = poses.float()
    xl, yl = nr1_lon.lon_only(p[..., 0], p[..., 1], p[..., 2])
    sp = nr1_lon.arc(xl, yl)
    with torch.no_grad():
        slow = ok & (s_ref[:, -1] >= S_MIN) & (sp[:, -1] < FRAC * s_ref[:, -1])
        fr = torch.zeros_like(slow)
        if slow.any():
            fr[slow] = free(p[slow].detach(), sp[slow].detach(), s_ref[slow], box[slow], val[slow])
        fire = slow & fr
    d = torch.relu(FRAC * s_ref - sp)
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1) * fire.to(d.dtype)
    return (v, dict(fire=fire, free=fr, slow=slow, s_plan=sp[:, -1].detach())) if parts else v


def term(poses, rows, pack, ah):
    return terms(poses, pack["s_ref"][rows], pack["ok"][rows], ah.box[rows], ah.val[rows])


def configure_stand(margin=1.0, roll=0.0):
    global MARGIN, V_ROLL
    assert margin > 0 and (roll == 0 or roll > V_STAND), (margin, roll)
    MARGIN, V_ROLL = float(margin), float(roll)
    return dict(stand_margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND) | (dict(v_roll=V_ROLL) if V_ROLL else {})


def rolling(v0, hum4, gap):
    """supplement 20 rows: the human stands all 4 s, the ego rolls at V_STAND <= v0 < V_ROLL, the station gap is finite."""
    v0 = np.asarray(v0)
    return np.isfinite(hum4) & (np.nan_to_num(hum4, nan=1e9) < H_STAND) & (v0 >= V_STAND) & (v0 < V_ROLL) & np.isfinite(gap)


def roll_cap(cap, gap):
    """cap (n, 8) = b_j, gap (n,) -> c_j = max(b_j, max(g, 0) + MARGIN)."""
    return np.maximum(cap, (np.maximum(np.asarray(gap, np.float64), 0.0) + MARGIN)[:, None]).astype(np.float32)


def standing(v0, hum4):
    """visited-state t0 speed (N,), the human's logged 4 s arc (N,) -> (N,) bool: the rows supplement 17 lets fire."""
    return np.isfinite(hum4) & (np.asarray(v0) < V_STAND) & (np.nan_to_num(hum4, nan=1e9) < H_STAND)


def stand_cap(fut, fut_reg):
    """-> cap (N, 8) = max of the two cumulative human arcs + MARGIN, ok (N,) finite."""
    a, b = cum_arc(fut), cum_arc(fut_reg)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return (np.nan_to_num(np.maximum(a, b)) + MARGIN).astype(np.float32), ok


def stand_load(pool_dir, names, dev):
    """-> pack: cap (N, 8), ok (N,) on dev for the pool's standing S rows."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: tab rows != the pool's"
    cap, fin = stand_cap(z["fut"], z["fut_reg"])
    S = z["mode"].astype(str) == "self" if "mode" in z.files else np.ones(len(fin), bool)
    hum4 = cum_arc(z["fut_reg"])[:, -1]
    st = standing(z["speed"], hum4)
    ok = fin & S & st
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), standing=int(ok.sum()), ego_slow=int((S & fin & (z["speed"] < V_STAND)).sum()),
               margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND)
    if not V_ROLL:
        return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)
    g = np.load(Path(pool_dir) / "stand_gap.npz")                       # supplement 20: scripts/stand_r.py gap
    assert np.array_equal(g["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: stand_gap rows != the pool's"
    roll = fin & S & rolling(z["speed"], hum4, g["gap"]) & ~st
    cap[roll] = roll_cap(cap[roll], g["gap"][roll])
    rec |= dict(rolling=int(roll.sum()), rolling_behind_b8=int((roll & (g["gap"] + MARGIN > cap[:, -1] - 1e-6)).sum()), v_roll=V_ROLL)
    return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok | roll, device=dev), roll=torch.as_tensor(roll, device=dev), rec=rec)


def stand_terms(poses, cap, ok, parts=False):
    """poses (n, 8, 3) -> (n,) raw STAND term (module docstring); parts: also the fire mask and s_plan_8."""
    import torch
    import torch.nn.functional as F
    import nr1_lon
    p = poses.float()
    xl, yl = nr1_lon.lon_only(p[..., 0], p[..., 1], p[..., 2])
    sp = nr1_lon.arc(xl, yl)
    with torch.no_grad():
        fire = ok & (sp[:, -1] > cap[:, -1])
    d = torch.relu(sp - cap)
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1) * fire.to(d.dtype)
    return (v, dict(fire=fire, s_plan=sp[:, -1].detach())) if parts else v


def stand_term(poses, rows, pack):
    return stand_terms(poses, pack["cap"][rows], pack["ok"][rows])


def stand_term_split(poses, rows, pack):
    """supplement 20: the STAND term -> (standing rows' part, rolling rows' part); their sum is stand_term."""
    v = stand_term(poses, rows, pack)
    r = pack["roll"][rows].to(v.dtype)
    return v * (1 - r), v * r


def stop_label(t, s_e, T, S, look=LOOK):
    """One state: log time t (s), the ego's station s_e on the logged path; the human's log times T (s, increasing) and stations S
    (non-decreasing) -> dict(rest (m from s_e, inf: none), t_rest, t_h (when he was at s_e), h4c = s_h(t + 4) - s_e, h4s = s_h(t_h + 4) - s_e,
    vmin8 (his lowest speed over the 8 s after t_h), path_ok)."""
    T, S = np.asarray(T, np.float64), np.maximum.accumulate(np.asarray(S, np.float64))
    sh = lambda tau: np.interp(tau, T, S)  # noqa: E731
    ok = len(T) >= 2 and T[0] <= t <= T[-1] and S[0] - 1.0 <= s_e
    i = int(np.searchsorted(S, s_e, "left"))                          # the first log sample at or past s_e (S has flat standing runs)
    if s_e > S[-1]:
        th = np.inf
    elif i == 0:
        th = float(T[0])
    else:
        th = float(T[i - 1] + (T[i] - T[i - 1]) * (s_e - S[i - 1]) / max(S[i] - S[i - 1], 1e-9))
    g = np.arange(t, min(t + look, T[-1]) + 1e-9, 0.1)
    v = (sh(g + 0.05) - sh(g - 0.05)) / 0.1
    hit = np.flatnonzero((v < V_REST) & (sh(g) >= s_e) & (g <= T[-1] - 0.05))
    rest, tr = (float(sh(g[hit[0]]) - s_e), float(g[hit[0]])) if len(hit) else (np.inf, np.nan)
    gs = np.arange(th, th + 8.0 + 1e-9, 0.1) if np.isfinite(th) else np.zeros(0)
    gs = gs[gs <= T[-1] - 0.05]
    vmin8 = float(((sh(gs + 0.05) - sh(gs - 0.05)) / 0.1).min()) if len(gs) and th + 8.0 <= T[-1] else np.nan
    return dict(rest=rest, t_rest=tr, t_h=th, h4c=float(sh(t + H_FUT) - s_e), h4s=float(sh(th + H_FUT) - s_e) if np.isfinite(th) else np.nan,
                vmin8=vmin8, path_ok=bool(ok))


def stoppt_ok(lab, v0, lead):
    """Label dict of stop_label + the ego's t0 speed + the lead flag -> (ok, reason): the rows supplement 18 lets fire."""
    if not lab["path_ok"]:
        return False, "no_path"
    if not np.isfinite(lab["rest"]):
        return False, "no_rest"
    if v0 < V_APP:
        return False, "ego_slow"
    lim = lab["rest"] + M_STOP
    if lab["h4c"] > lim:
        return False, "leaves_clock"
    if not np.isfinite(lab["h4s"]) or lab["h4s"] > lim:
        return False, "leaves_station"
    if lead:
        return False, "lead"
    return True, ""


def stoppt_cap(rest, v0):
    rest, v0 = np.asarray(rest, np.float64), np.asarray(v0, np.float64)
    return np.maximum(np.nan_to_num(rest, posinf=1e6) + M_STOP, v0 ** 2 / (2 * A_MAX)).astype(np.float32)


def stoppt_load(pool_dir, names, dev):
    """-> pack: cap (N,), ok (N,) on dev from <pool_dir>/stoppt.npz (scripts/stoppt.py labels; names = the pool's tab names in order)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "stoppt.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: stoppt.npz rows != the pool's"
    ok, cap = z["ok"].astype(bool), stoppt_cap(z["rest"], z["v0"])
    rec = dict(rows=int(len(ok)), stoppt_rows=int(ok.sum()), cap_raised=int((ok & (cap > np.nan_to_num(z["rest"], posinf=1e6) + M_STOP + 1e-4)).sum()),
               m_stop=M_STOP, a_max=A_MAX, v_app=V_APP, v_rest=V_REST)
    return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


def stoppt_terms(poses, cap, ok, parts=False):
    """poses (n, 8, 3), cap (n,), ok (n,) -> (n,) raw STOPPT term (module docstring); parts: also the fire mask and s_plan_8."""
    import torch
    import torch.nn.functional as F
    import nr1_lon
    p = poses.float()
    xl, yl = nr1_lon.lon_only(p[..., 0], p[..., 1], p[..., 2])
    sp = nr1_lon.arc(xl, yl)
    with torch.no_grad():
        fire = ok & (sp[:, -1] > cap)
    d = torch.relu(sp - cap[:, None])
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1) * fire.to(d.dtype)
    return (v, dict(fire=fire, s_plan=sp[:, -1].detach())) if parts else v


def stoppt_term(poses, rows, pack):
    return stoppt_terms(poses, pack["cap"][rows], pack["ok"][rows])


def turn_range(path):
    """path (n, m, 2): the human's logged path at 1 m spacing over [s_e - TC_BACK, s_e + TC_AHEAD] -> (n,) heading range (deg, max - min of
    the unwrapped segment headings) and (n,) net heading change (deg, last - first segment; left +). Segments shorter than 0.2 m (a clamped
    path end) carry the previous heading; NaN rows stay NaN."""
    P = np.asarray(path, np.float64)
    d = np.diff(P, axis=1)
    sl = np.hypot(d[..., 0], d[..., 1])
    h = np.where(sl > 0.2, np.arctan2(d[..., 1], d[..., 0]), np.nan)
    for j in range(1, h.shape[1]):
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j - 1], h[:, j])
    for j in range(h.shape[1] - 2, -1, -1):                            # a clamped start: take the first real heading
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j + 1], h[:, j])
    bad = np.isnan(h).any(1) | ~np.isfinite(P).all((1, 2))
    h = np.degrees(np.unwrap(np.nan_to_num(h), axis=1))
    rng, net = h.max(1) - h.min(1), h[:, -1] - h[:, 0]
    return np.where(bad, np.nan, rng), np.where(bad, np.nan, net)


def turncap_rows(v0, hum4, turn):
    """ego t0 speed (N,), the human's logged 4 s arc (N,), the path's heading range (N, deg) -> (N,) bool: the rows supplement 21 lets fire."""
    hum4, turn = np.nan_to_num(np.asarray(hum4, np.float64), nan=-1.0), np.nan_to_num(np.asarray(turn, np.float64), nan=-1.0)
    return (np.asarray(v0) >= V_TC) & (hum4 >= H_TC) & (turn >= TURN_TC)


def turncap_cap(fut, fut_reg):
    """-> cap (N, 8) = K_TC x max of the two cumulative human arcs, ok (N,) finite."""
    a, b = cum_arc(fut), cum_arc(fut_reg)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return (K_TC * np.nan_to_num(np.maximum(a, b))).astype(np.float32), ok


def turncap_load(pool_dir, names, dev):
    """-> pack: cap (N, 8), ok (N,) on dev for the pool's moving S rows in sharp turns (labels <pool_dir>/turn_cap.npz)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: tab rows != the pool's"
    g = np.load(Path(pool_dir) / "turn_cap.npz")                       # scripts/turn_cap.py label
    assert np.array_equal(g["names"].astype(str), np.asarray(names).astype(str)), "nr1_free: turn_cap rows != the pool's"
    cap, fin = turncap_cap(z["fut"], z["fut_reg"])
    S = z["mode"].astype(str) == "self" if "mode" in z.files else np.ones(len(fin), bool)
    ok = fin & S & turncap_rows(z["speed"], cum_arc(z["fut_reg"])[:, -1], g["turn"])
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), turncap_rows=int(ok.sum()), right=int((ok & (g["net"] < 0)).sum()),
               left=int((ok & (g["net"] >= 0)).sum()), v_tc=V_TC, h_tc=H_TC, turn_tc=TURN_TC, k_tc=K_TC, window=[-TC_BACK, TC_AHEAD])
    return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


def turncap_term(poses, rows, pack):
    return stand_terms(poses, pack["cap"][rows], pack["ok"][rows])


if __name__ == "__main__":                     # known-answer check (CPU)
    import sys
    import torch
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    t = torch.linspace(0.5, 4.0, 8)
    go = lambda v: torch.stack([v * t, torch.zeros(8), torch.zeros(8)], -1)[None]  # noqa: E731
    s_ref = (2.0 * t)[None]
    ok = torch.tensor([True])
    K = 4
    box, val = torch.zeros(1, 9, K, 5), torch.zeros(1, 9, K, dtype=torch.bool)
    v0, vs = terms(go(2.0), s_ref, ok, box, val), terms(go(1.0), s_ref, ok, box, val)
    assert v0 == 0 and vs > 0.5, (v0, vs)                                            # at the human: 0; at half his speed: fires
    assert terms(go(1.7), s_ref, ok, box, val) == 0                                   # 0.85 x: inside the 0.8 band
    assert terms(go(3.0), s_ref, ok, box, val) == 0                                   # faster: one-sided, nothing
    lead = box.clone()
    lead[0, :, 0] = torch.tensor([4.049 + 3.0 + 2.25, 0.0, 0.0, 4.5, 2.0])            # a car 3 m ahead of the bumper, standing
    lv = val.clone()
    lv[0, :, 0] = True
    assert terms(go(1.0), s_ref, ok, lead, lv) == 0                                    # lead in the way: nothing
    side = lead.clone()
    side[0, :, 0, 1] = 3.5                                                             # the same car one lane over: free, fires
    assert terms(go(1.0), s_ref, ok, side, lv) > 0.5
    far = lead.clone()
    far[0, :, 0, 0] = 30.0                                                             # 25 m ahead: beyond the sweep (4 s at 2 m/s + D)
    assert terms(go(1.0), s_ref, ok, far, lv) > 0.5
    behind = lead.clone()
    behind[0, :, 0, 0] = -6.0
    assert terms(go(1.0), s_ref, ok, behind, lv) > 0.5                                 # behind: not counted
    cross = lead.clone()                                                               # a car crossing 5 m ahead (plus-shaped overlap)
    cross[0, :, 0] = torch.tensor([4.049 + 5.0, 0.0, 1.5708, 6.0, 2.0])
    assert terms(go(1.0), s_ref, ok, cross, lv) == 0
    stand = torch.zeros(1, 8, 3)
    assert terms(stand, s_ref, ok, box, val) > 1.0                                     # standing plan, free road, human goes: fires
    q = go(1.0).clone().requires_grad_(True)                                           # gradient along the heading only, none on the yaw
    th = torch.linspace(0.1, 0.9, 8)
    q = torch.stack([1.0 * t * torch.cos(th), 1.0 * t * torch.sin(th), th + 0.05], -1)[None].requires_grad_(True)
    terms(q, s_ref, ok, box, val).sum().backward()
    g = q.grad[0]
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    assert g[:, 2].abs().max() == 0 and across.abs().max() < 1e-6 and g[:, :2].abs().max() > 0, g
    vh = np.array([[8.0] * 8, [8.0] * 7 + [6.0], [8.0] * 7 + [2.5], [3.5] * 8, [8.0] * 8, [np.nan] * 8])
    assert moving(np.array([8.0, 8.0, 8.0, 3.5, 2.0, 8.0]), vh).tolist() == [True, False, False, True, False, False]   # cruise / brake / stop / slow / ego slow / nan
    configure(0.95, "move")
    assert terms(go(1.85), s_ref, ok, box, val) > 0 and terms(go(1.95), s_ref, ok, box, val) == 0                    # 0.925 x fires at 0.95, 0.975 x not
    lead20 = lead.clone()
    lead20[0, :, 0, 0] = 4.049 + 15.0 + 2.25                                           # a standing car 15 m ahead of the bumper
    assert terms(go(1.85), s_ref, ok, lead20, lv) > 0                                  # 2 m + 0.5 s x v: free, fires
    configure(0.95, "move", (20.0, 3.0))
    assert terms(go(1.85), s_ref, ok, lead20, lv) == 0 and terms(go(1.85), s_ref, ok, box, val) > 0   # within 20 m: blocked; empty road fires
    lead20[0, :, 0, 0] = 4.049 + 8.0 + 30.0                                            # 30 m beyond the 4 s reference position: free
    assert terms(go(1.85), s_ref, ok, lead20, lv) > 0
    configure()
    assert (CLEAR_MIN, CLEAR_TH) == (0.0, 0.0)
    capz = torch.full((1, 8), MARGIN)                                                  # the human stands: cap = 1 m at every step
    assert stand_terms(stand, capz, ok) == 0 and stand_terms(go(0.2), capz, ok) == 0  # standing / 0.8 m in 4 s: inside the cap
    assert stand_terms(go(1.0), capz, ok) > 0.5 and stand_terms(go(1.0), capz, ~ok) == 0   # 4 m: fires; not a standing row: 0
    q = torch.stack([1.0 * t * torch.cos(th), 1.0 * t * torch.sin(th), th + 0.05], -1)[None].requires_grad_(True)
    stand_terms(q, capz, ok).sum().backward()
    g = q.grad[0]
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    assert g[:, 2].abs().max() == 0 and across.abs().max() < 1e-6 and g[:, :2].abs().max() > 0, g
    along = torch.cos(q[0, :, 2].detach()) * g[:, 0] + torch.sin(q[0, :, 2].detach()) * g[:, 1]
    assert (along >= 0).all()                                                          # descent pulls the poses back (shorter)
    hum = np.array([[0.0] * 8, [0.0] * 7 + [0.3]])
    c_, o_ = stand_cap(hum[:, :, None].repeat(2, 2) * [1, 0], hum[:, :, None].repeat(2, 2) * [1, 0])
    assert o_.all() and np.allclose(c_[:, -1], [1.0, 1.3])
    assert standing(np.array([0.1, 0.1, 0.8, 0.1]), np.array([0.2, 0.7, 0.2, np.nan])).tolist() == [True, False, False, False]
    # STOPPT: a human who stops 20 m ahead at 2 s and stays; an ego at his station-matched state
    T_ = np.arange(0, 20.01, 0.1)
    S_ = np.where(T_ < 2, 10.0 * T_ - 2.5 * T_ ** 2, 10.0)                             # 10 m/s braking at 5 m/s^2 to rest at 10 m, t = 2 s
    lab = stop_label(0.0, 0.0, T_, S_)
    assert abs(lab["rest"] - 10.0) < 0.2 and abs(lab["t_rest"] - 2.0) < 0.15 and lab["h4c"] <= 10.0 + 1e-6 and stoppt_ok(lab, 10.0, False)[0], lab
    assert stoppt_ok(lab, 0.2, False) == (False, "ego_slow") and stoppt_ok(lab, 10.0, True) == (False, "lead")
    lab_go = stop_label(0.0, 0.0, T_, np.where(T_ < 3, S_, 10.0 + 5.0 * (T_ - 3)))      # he leaves at 3 s: 15 m at 4 s > 11 m
    assert stoppt_ok(lab_go, 10.0, False) == (False, "leaves_clock"), lab_go
    lab_past = stop_label(5.0, 15.0, T_, np.where(T_ < 3, S_, 10.0 + 5.0 * (T_ - 3)))   # at t = 5 he is moving; the point he left is behind
    assert not np.isfinite(lab_past["rest"]), lab_past
    lab_early = stop_label(0.0, 6.0, T_, S_)                                            # the ego ahead of him: rest 4 m ahead of the ego
    assert abs(lab_early["rest"] - 4.0) < 0.2 and stoppt_ok(lab_early, 10.0, False)[0], lab_early
    cap = torch.as_tensor(stoppt_cap([10.0, 10.0, 2.0], [5.0, 5.0, 10.0]))               # third: 2 m ahead at 10 m/s -> 12.5 m (A_MAX)
    assert abs(float(cap[2]) - 12.5) < 1e-4 and abs(float(cap[0]) - 11.0) < 1e-6
    ok3 = torch.tensor([True, False, True])
    P3 = torch.cat([go(2.0), go(5.0), go(2.0)])                                          # 8 m, 20 m, 8 m in 4 s
    v3, pr = stoppt_terms(P3, cap, ok3, parts=True)
    assert v3[0] == 0 and v3[1] == 0 and v3[2] == 0 and not pr["fire"].any()             # under the cap / not a row / under the raised cap
    assert stoppt_terms(go(5.0), cap[:1], ok3[:1]) > 1.0                                  # 20 m against 11 m: fires
    q = torch.stack([4.0 * t * torch.cos(th), 4.0 * t * torch.sin(th), th + 0.05], -1)[None].requires_grad_(True)
    stoppt_terms(q, cap[:1], ok3[:1]).sum().backward()
    g = q.grad[0]
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    along = torch.cos(q[0, :, 2].detach()) * g[:, 0] + torch.sin(q[0, :, 2].detach()) * g[:, 1]
    assert g[:, 2].abs().max() == 0 and across.abs().max() < 1e-6 and (along >= 0).all() and along.max() > 0, g
    # STAND-R (supplement 20): rolling rows' caps
    assert configure_stand(1.0) == dict(stand_margin=1.0, v_stand=V_STAND, h_stand=H_STAND) and V_ROLL == 0
    configure_stand(1.0, 3.0)
    assert rolling(np.array([0.3, 0.5, 2.9, 3.0, 2.0, 2.0]), np.array([0.1, 0.1, 0.1, 0.1, 0.6, 0.1]),
                   np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan])).tolist() == [False, True, True, False, False, False]
    v0 = 2.0                                                                           # fut: 2 m/s blended to 0 over 2 s -> 2 m, cap b ~ 3 m
    tt = np.arange(41) * 0.1
    vv = np.maximum(v0 + (0.0 - v0) * np.minimum(tt / 2.0, 1.0), 0.0)
    D = np.r_[0.0, np.cumsum((vv[1:] + vv[:-1]) * 0.05)][5::5]
    fut = np.stack([D, np.zeros(8), np.zeros(8)], -1)[None]
    b, _ = stand_cap(fut, np.zeros_like(fut))
    assert abs(b[0, -1] - 3.0) < 0.01, b
    ok1 = torch.tensor([True])
    for g_, lo, hi in ((0.0, 2.5, 8.0), (10.0, 9.0, 12.0), (-3.0, 2.5, 8.0)):          # g = 0: b; g = 10 m: 11 m; g = -3 (past him): b
        c = torch.as_tensor(roll_cap(b, [g_]))
        assert stand_terms(go(lo / 4.0), c, ok1) == 0 and stand_terms(go(hi / 4.0), c, ok1) > 0, (g_, c)
    assert np.allclose(roll_cap(b, [-3.0]), b) and np.allclose(roll_cap(b, [10.0])[0, 0], 11.0)
    configure_stand(1.0)
    # TURN-CAP (supplement 21): turn label on a 1 m path, rows, cap and fire
    s_ = np.arange(-TC_BACK, TC_AHEAD + 1, dtype=np.float64)              # 31 samples, the ego at index TC_BACK
    straight = np.stack([s_, 0 * s_], -1)
    R_ = 8.0                                                             # a 90 deg right turn of radius 8 m starting 2 m ahead
    th_ = np.clip((s_ - 2.0) / R_, 0, np.pi / 2)
    xr = np.where(s_ < 2, s_, 2 + R_ * np.sin(th_) + np.maximum(s_ - 2 - R_ * np.pi / 2, 0) * np.cos(th_))
    yr = np.where(s_ < 2, 0.0, -R_ * (1 - np.cos(th_)) - np.maximum(s_ - 2 - R_ * np.pi / 2, 0) * np.sin(th_))
    clamp = straight.copy()
    clamp[:4] = clamp[4]                                                 # the log starts 6 m behind the ego: clamped samples
    rg, nt = turn_range(np.stack([straight, np.stack([xr, yr], -1), np.stack([xr, -yr], -1), clamp]))
    assert abs(rg[0]) < 1e-6 and abs(rg[1] - 90) < 2 and nt[1] < -85 and nt[2] > 85 and abs(rg[3]) < 1e-6, (rg, nt)
    assert turncap_rows(np.array([3.0, 1.5, 3.0, 3.0, 3.0]), np.array([10.0, 10.0, 3.0, 10.0, np.nan]),
                        np.array([90.0, 90.0, 90.0, 40.0, 90.0])).tolist() == [True, False, False, False, False]   # row / ego slow / human slow / mild / nan
    vv = np.maximum(4.0 + (3.0 - 4.0) * np.minimum(tt / 2.0, 1.0), 0.0)  # ego 4 m/s, the human 3 m/s: fut = 4 -> 3 over 2 s = 13 m in 4 s
    Dt = np.r_[0.0, np.cumsum((vv[1:] + vv[:-1]) * 0.05)][5::5]
    futt = np.stack([Dt, np.zeros(8), np.zeros(8)], -1)[None]
    regt = np.stack([3.0 * t.numpy(), np.zeros(8), np.zeros(8)], -1)[None]  # his own 3 m/s: 12 m
    ct, okc = turncap_cap(futt, regt)
    assert okc.all() and abs(ct[0, -1] - 1.05 * 13.0) < 0.02 and np.allclose(ct[0], 1.05 * np.maximum(Dt, regt[0, :, 0]), atol=1e-4), ct
    ct = torch.as_tensor(ct)
    assert stand_terms(go(4.0), ct, ok1) > 0 and stand_terms(go(3.25), ct, ok1) == 0      # 16 m against 13.65 m fires; 13 m (his reach) not
    assert stand_terms(torch.as_tensor(futt, dtype=torch.float32), ct, ok1) == 0 and stand_terms(torch.as_tensor(regt, dtype=torch.float32), ct, ok1) == 0
    assert stand_terms(go(4.0), ct, ~ok1) == 0                                          # not a row: 0
    q = torch.stack([4.0 * t * torch.cos(th), 4.0 * t * torch.sin(th), th + 0.05], -1)[None].requires_grad_(True)
    stand_terms(q, ct, ok1).sum().backward()
    g = q.grad[0]
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    along = torch.cos(q[0, :, 2].detach()) * g[:, 0] + torch.sin(q[0, :, 2].detach()) * g[:, 1]
    assert g[:, 2].abs().max() == 0 and across.abs().max() < 1e-6 and (along >= 0).all() and along.max() > 0, g   # shorter, shape untouched
    print("nr1_free known-answer ok", float(vs))
