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

NR1STAND (prereg supplement 17; rm_train --on-stand W --on-stand-margin M): the mirror term on standing states. Rows: S rows where the visited
state stands (tab `speed` < V_STAND) and the human stands all 4 s (arc(fut_reg)_8 < H_STAND); no free check. Cap b_j = max(arc(fut)_j,
arc(fut_reg)_j) + M (the larger human path, so neither of his paths fires). Fire: s_plan_8 > b_8. Value: sum over the 8 steps of smooth L1
(beta 0.5) of relu(s_plan_j - b_j); gradient only along each pose's heading. No target speed, nothing inside the cap.

NR1STOPPT (prereg supplement 18; rm_train --on-stoppt W): "do not pass the human's stop point" on approach states. Labels from the log
(stop_label, built offline into <pool>/stoppt.npz by scripts/stoppt.py labels): rest = the human's station, relative to the ego's, at the first
log time tau >= t with v_h(tau) < V_REST and s_h(tau) >= s_e (clock-consistent: a point he left before t does not count); rows: S rows, ego t0
speed >= V_APP, the human has not passed rest + M_STOP at t + 4 s nor 4 s after he was where the ego is, no logged object at the stop (label
file). Cap = max(rest + M_STOP, v0^2 / (2 A_MAX)), the same at every step. Fire: s_plan_8 > cap. Value: sum over the 8 steps of smooth L1
(beta 0.5) of relu(s_plan_j - cap); gradient only along each pose's heading. No target speed, nothing below the cap.
"""
from __future__ import annotations

import numpy as np

FRAC, S_MIN, D0, TH, NS, BETA = 0.8, 1.0, 2.0, 0.5, 8, 0.5
MODE, V_MOVE, HUM_KEEP = "all", 3.0, 0.8
FRONT, REAR, HALF_W = 4.049, 1.127, 1.1485        # nuPlan Pacifica about the rear axle (= lib/agent_hinge.py)
DT = 0.5
V_STAND, H_STAND, MARGIN = 0.5, 0.5, 1.0
V_REST, V_APP, M_STOP, A_MAX, H_FUT, LOOK = 0.5, 0.5, 1.0, 4.0, 4.0, 30.0


def cum_arc(P):
    """(N, 8, >=2) -> (N, 8) cumulative polyline arc from the origin (numpy; = nr1_lon.human_arc)."""
    P = np.concatenate([np.zeros((len(P), 1, 2)), np.asarray(P, np.float64)[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)


def configure(frac=0.8, mode="all"):
    global FRAC, MODE
    assert mode in ("all", "move") and 0 < frac <= 1, (frac, mode)
    FRAC, MODE = float(frac), str(mode)
    return dict(frac=FRAC, mode=MODE)


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
    D = (D0 + TH * v)[..., None]                                                        # (n, 8, 1)
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


def configure_stand(margin=1.0):
    global MARGIN
    assert margin > 0, margin
    MARGIN = float(margin)
    return dict(stand_margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND)


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
    st = standing(z["speed"], cum_arc(z["fut_reg"])[:, -1])
    ok = fin & S & st
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), standing=int(ok.sum()), ego_slow=int((S & fin & (z["speed"] < V_STAND)).sum()),
               margin=MARGIN, v_stand=V_STAND, h_stand=H_STAND)
    return dict(cap=torch.as_tensor(cap, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


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
    configure()
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
    print("nr1_free known-answer ok", float(vs))
