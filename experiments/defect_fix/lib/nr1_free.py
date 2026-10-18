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
"""
from __future__ import annotations

import numpy as np

FRAC, S_MIN, D0, TH, NS, BETA = 0.8, 1.0, 2.0, 0.5, 8, 0.5
FRONT, REAR, HALF_W = 4.049, 1.127, 1.1485        # nuPlan Pacifica about the rear axle (= lib/agent_hinge.py)
DT = 0.5


def cum_arc(P):
    """(N, 8, >=2) -> (N, 8) cumulative polyline arc from the origin (numpy; = nr1_lon.human_arc)."""
    P = np.concatenate([np.zeros((len(P), 1, 2)), np.asarray(P, np.float64)[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)


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
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), labelled=int(ok.sum()), ref_below_smin=int((ok & (s_ref[:, -1] < S_MIN)).sum()),
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
    print("nr1_free known-answer ok", float(vs))
