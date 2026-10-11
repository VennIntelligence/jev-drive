"""ONPOL-NUPLAN self-consequence L: the four longitudinal hinges of research/self-consequence/design.md 3.2 on the student's own plan at an
on-policy state, in the on-policy versions of plans/2026-10-11-onpol-prereg.md ("Self-consequence L"). Analytic terms only: labels + own plan.

Trainer hook (scripts/on_train.py --on-lon):
    pack = on_lon.load(pool_dir, names, device)    # the pack of scripts/on_lon_pack.py (pack_path), re-ordered to the trainer's row names
                                                   # (on_lon.load_pack(path, device) reads a pack in its own order)
    L = on_lon.loss(poses_rear, rows, pack)        # poses_rear (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's own frame (with grad),
                                                   # rows (n,) pool row indices -> {"lonA", "lonB", "lonC", "lonP"}: (n,) per-row losses
Units: lonA, lonB, lonP in metres, lonC in m/s. The trainer applies lambda 10 and the hinge-only row weight. Only the items in pack["items"]
are returned (the prereg's 0.2 % activation rule drops an item at pack time: scripts/on_lon_pack.py --items).

Every label of a row lives in the ROW's own rear-axle frame at its time t (x forward, y left): the route (the scene's hindsight route of
scripts/on_ctx.py, re-expressed), the simulator's actor boxes (the pool's agent_labels layout). Arcs are metres along the route from the
route's origin (2 s before the scene token on the logged path).

The plan enters through `timing_only` (the mirror of body1 loss43.shape_only): values unchanged, the gradient of every pose keeps only its
component along the pose's own (detached) heading, the yaw gets none. L can move poses along the path (the speed profile), not sideways.

Terms (t on the 41-step dense grid 0 .. 4 s; s_f / l_f = arc / lateral offset of the front-bumper point, s_r / l_r of the rear axle):
  A  lead contact (m).  Per A-eligible actor j at the row's time (lc.a_eligible on the simulator boxes: vehicle, valid, centre within 1.5 m of
     the route, ahead of the ego front, aligned with the route or slower than 0.5 m/s; the 4 nearest by rear arc): sigma_j(t) = min arc of its 4
     corners, lambda_j(t) = lateral offset of its centre. g_j(t) = sigma_j(t) - s_f(t); m_j = min(2.0, max(g_j(0) - 0.5, 0)) with g_j(0) the gap
     at the row's own pose; h_A = mean_t max_j [valid_j(t) & |lambda_j(t)| <= 1.5 & t > 0 & g_j(0) > 0 & |l_f(t)| <= 3.5] relu(m_j - g_j(t)).
     DEVIATION (audit): the gate |l_f(t)| <= 3.5 on the plan's own front point, as B and C have it (a row may sit up to 4 m beside the route).
  B  stop line (m).  Traffic-light lines only (stop-sign lines are masked), eligible in the context (ln_elig) and not already passed by the
     front bumper at the row's time (d_l >= s_f(0); DEVIATION from the prereg's formula, which would otherwise pin a car that is already past
     the line): mu_l(t) = the light of the route's connector is red at the log frame nearest t (by timestamp); theta_l = max(d_l - 1.0, s_f(0));
     h_B = mean_t max_l mu_l(t) [not cleared_l(t)] [|l_f(t)| <= 3.5] [s_f(t) <= rlen] relu(s_f(t) - theta_l), cleared_l(t) = the front bumper was
     beyond d_l at some step t' <= t that was not a must-stop step. DEVIATION (audit): without it a plan that crosses on green is charged for
     every later step once the light turns red (the LONG-PROBE label reads the light at the crossing; design.md had the log-relaxed theta).
  C  speed into the first >= 45 deg route turn (m/s).  Eligible when the logged driver passes the curvature peak s_pk at >= 1 m/s.
     v_cap(s) = max(v_ref + 1.0, 1.15 v_ref), v_ref(s) = max(the route's logged speed at the two grid vertices around s (ctx `vlog`, speed at the
     first arrival), the logged speed at the peak); h_C = mean_t [s_pk - 10 <= s_r(t) <= s_pk + 3] [|l_r(t)| <= 3.5] [|psi(t) - H(s_pk)| <= 30 deg] relu(v(t) - v_cap).
     DEVIATIONS from design.md: v_ref comes from the context's logged speed by arc everywhere, not from the dense logged future of the row
     (an on-policy row has no logged future of its own); and it is floored at the log's speed at the peak (audit): where the log waited at a
     line before the turn its speed by arc is 0, which would cap a row that arrives on green at 1 m/s.
  P  progress guard (m).  h_P = mean_k relu(0.95 D_k - 0.5 - s_plan(t_k)), k = 1 .. 8, D_k = the prereg's D_k as Amendment 2 redefined it
     (on_rows.py meta `D`: the speed blends from the ego's own into the log's over 2 s; the pack's default, on_lon_pack.py --p-ref amended;
     `--p-ref log` = the log's travel s_log(t + 0.5 k) - s_log(t), which asks a slow row for a speed jump), s_plan = the plan's own arc length; only on OPEN rows. design.md's "open" (`p_open_design`):
     no A-eligible actor, no eligible stop line of either type and no eligible turn peak within 60 m ahead of the row along the route.
     DEVIATION (audit; the pack's default `p_open`, scripts/on_lon_pack.py --p-open strict): additionally no stop line of any kind (eligible or
     not) within 60 m ahead, and no actor of any class whose centre is within 4.5 m of the route and 0 .. 60 m ahead of the row at any of the
     9 label times (the smoke preview showed a plan that leaves the route towards a vehicle 3 m beside it); and in the loss a pose is charged
     only while it is within 3.5 m of the route (|l_r(t_k)| <= 3.5, as B and C). Away from the log's timeline the log's progress is not safe by construction, and in ON-SELF nothing else acts along the
     path (S's hinges are shape-only), so P must not push where the labels cannot see what is ahead.
Known limits: only the first >= 45 deg turn of a scene's route carries C; a line or actor the context cannot label gives no B / A.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

FRONT = 4.049
NS, DT = 41, 0.1
NA, NL = 4, 4
M_A0, M_A_BACK, M_B = 2.0, 0.5, 1.0
LAT_IN, LAT_ROUTE = 1.5, 3.5
C_PRE, C_POST, C_HEAD, C_ADD, C_MUL, C_VLOG_MIN = 10.0, 3.0, math.radians(30.0), 1.0, 1.15, 1.0
P_FRAC, P_SLACK, OPEN_M, P_ACT_LAT = 0.95, 0.5, 60.0, 4.5
BIG = 1e3
ITEMS = ("A", "B", "C", "P")
EPS = 1e-12


def interp_matrix() -> torch.Tensor:
    """(41, 9): [origin, 8 poses at 0.5 s] -> the 0.1 s grid (body1 sweep.interp_matrix)."""
    M = torch.zeros(NS, 9)
    for k in range(NS):
        j = min(k // 5, 7)
        f = (k - 5 * j) / 5.0
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


_M = interp_matrix()


# ---------------------------------------------------------------- operators (torch mirrors of long_contact/lib/lc.py and body1/lib/sweep.py)
def timing_only(x, y, psi):
    """Same values; d(x~, y~) / d(x, y) = u u^T with u = (cos sg(psi), sin sg(psi)): a pose can only move along its own heading.
    The caller passes psi on detached."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    ux, uy = torch.cos(pd), torch.sin(pd)
    a = ux * (x - xd) + uy * (y - yd)                                           # 0 in value, carries the along-heading gradient
    return xd + ux * a, yd + uy * a


def dense(x, y, psi):
    """Plan poses (n, 8) each in the row's frame -> (n, 41, 3) at 0.1 s from the row's own origin (sweep.dense with off = 0)."""
    o = x.new_zeros(len(x), 1)
    P = torch.stack([torch.cat([o, x], 1), torch.cat([o, y], 1), torch.cat([o, psi], 1)], -1)
    return torch.einsum("kj,njc->nkc", _M.to(x), P)


def project(R, P):
    """R (n, NR, 2) route grid at 1 m, P (n, T, 2) points -> arc (n, T) m along the route, signed lateral offset (n, T), left +.
    lc.project: nearest grid vertex (not differentiated), refined on its two neighbouring segments (differentiated)."""
    NR = R.shape[1]
    with torch.no_grad():
        m = ((P[:, :, None, :] - R[:, None, :, :]) ** 2).sum(-1).argmin(-1)
    bs = bl = bd = None
    for a in (-1, 0):
        i0 = (m + a).clamp(0, NR - 2)
        A = R.gather(1, i0[..., None].expand(-1, -1, 2))
        Bp = R.gather(1, (i0 + 1)[..., None].expand(-1, -1, 2))
        u = Bp - A
        L2 = (u ** 2).sum(-1).clamp_min(1e-9)
        Ln = L2.sqrt()
        t = (((P - A) * u).sum(-1) / L2).clamp(0.0, 1.0)
        q = A + t[..., None] * u
        dd = ((P - q) ** 2).sum(-1)
        lat = (u[..., 0] * (P[..., 1] - A[..., 1]) - u[..., 1] * (P[..., 0] - A[..., 0])) / Ln
        s = i0 + t * Ln
        if bd is None:
            bs, bl, bd = s, lat, dd
        else:
            w = dd < bd
            bs, bl, bd = torch.where(w, s, bs), torch.where(w, lat, bl), torch.where(w, dd, bd)
    return bs, bl


def front_point(d):
    return torch.stack([d[..., 0] + FRONT * torch.cos(d[..., 2]), d[..., 1] + FRONT * torch.sin(d[..., 2])], -1)


def path_speed(d):
    """Dense poses (n, 41, 3) -> speed (n, 41) m/s (lc.path_speed: mean of the adjacent 0.1 s segments, one-sided at the ends)."""
    v = torch.sqrt((d[:, 1:, 0] - d[:, :-1, 0]) ** 2 + (d[:, 1:, 1] - d[:, :-1, 1]) ** 2 + EPS) / DT
    return torch.cat([v[:, :1], 0.5 * (v[:, 1:] + v[:, :-1]), v[:, -1:]], 1)


def plan_arc(x, y):
    """Plan poses (n, 8) -> the plan's cumulative arc length at the 8 poses, from the row's origin."""
    o = x.new_zeros(len(x), 1)
    dx, dy = torch.diff(torch.cat([o, x], 1), dim=1), torch.diff(torch.cat([o, y], 1), dim=1)
    return torch.sqrt(dx ** 2 + dy ** 2 + EPS).cumsum(1)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


# ---------------------------------------------------------------- the loss
def terms(poses_rear, rows, pack, margin_a=None, theta_b=None):
    """-> dict of per-row (n,) hinges for all four items + the per-step tensors (for the gates and previews).
    margin_a / theta_b override the stored A margin (a scalar) / B threshold offset (theta = d_l - theta_b) for the label-agreement gate."""
    if poses_rear.dtype in (torch.float16, torch.bfloat16):                     # arcs reach 260 m: never in half precision
        poses_rear = poses_rear.float()
    x, y, psi = poses_rear.unbind(-1)
    xt, yt = timing_only(x, y, psi)
    d = dense(xt, yt, psi.detach())
    g = lambda k: pack[k][rows]  # noqa: E731
    R, rlen, ok = g("route"), g("rlen"), g("ok")
    sf, lf = project(R, front_point(d))
    sr, lr = project(R, d[..., :2])
    tpos = (torch.arange(NS, device=x.device) > 0)[None, None]
    # A
    gap = g("a_sig") - sf[:, None]                                              # (n, NA, 41)
    m = g("a_m")[..., None] if margin_a is None else torch.full_like(g("a_m"), margin_a)[..., None]
    act = g("a_val") & (g("a_lam").abs() <= LAT_IN) & tpos & (g("a_g0") > 0)[..., None] & (lf.abs() <= LAT_ROUTE)[:, None]
    sA = torch.where(act, torch.relu(m - gap), torch.zeros_like(gap)).amax(1)
    # B
    th = g("b_th") if theta_b is None else g("b_d") - theta_b
    on = ((lf.abs() <= LAT_ROUTE) & (sf <= rlen[:, None]))[:, None]
    over = torch.relu(sf[:, None] - th[..., None])                              # (n, NL, 41)
    mu = g("b_mu")
    cleared = ((sf.detach()[:, None] > g("b_d")[..., None]) & ~mu).cummax(-1)[0]  # beyond the line at a step that was not a must-stop step
    sB = torch.where(mu & on & ~cleared, over, torch.zeros_like(over)).amax(1)
    # C
    v = path_speed(d)
    cap = g("c_cap")
    NR = cap.shape[1]
    i0 = sr.detach().floor().long().clamp(0, NR - 1)
    vcap = torch.maximum(cap.gather(1, i0), cap.gather(1, (i0 + 1).clamp(max=NR - 1)))
    spk = g("c_spk")[:, None]
    w = (g("c_ok")[:, None] & (sr >= spk - C_PRE) & (sr <= spk + C_POST) & (lr.abs() <= LAT_ROUTE)
         & (wrap(d[..., 2] - g("c_hpk")[:, None]).abs() <= C_HEAD))
    sC = torch.where(w, torch.relu(v - vcap), torch.zeros_like(v))
    # P
    D = g("p_D")
    sP = torch.where(g("p_open")[:, None] & (D > -BIG) & (lr[:, 5::5].abs() <= LAT_ROUTE), torch.relu(P_FRAC * D - P_SLACK - plan_arc(xt, yt)), torch.zeros_like(D))
    z = ok.to(sA.dtype)
    return dict(lonA=sA.mean(-1) * z, lonB=sB.mean(-1) * z, lonC=sC.mean(-1) * z, lonP=sP.mean(-1) * z,
                steps=dict(A=sA, B=sB, C=sC, P=sP, sf=sf, lf=lf, sr=sr, lr=lr, v=v, vcap=vcap, d=d))


def loss(poses_rear, rows, pack):
    """The trainer hook: per-row (n,) losses of the items the pack carries."""
    T = terms(poses_rear, rows, pack)
    return {f"lon{i}": T[f"lon{i}"] for i in pack["items"]}


def to_pack(Z: dict, device="cpu", items=ITEMS) -> dict:
    """numpy pack arrays (build_rows / the saved npz) -> tensors on the device."""
    keys = ("route", "rlen", "ok", "a_sig", "a_lam", "a_val", "a_m", "a_g0", "b_d", "b_th", "b_mu", "c_ok", "c_spk", "c_hpk", "c_cap", "p_D", "p_open")
    P = {k: torch.as_tensor(np.asarray(Z[k])).to(device) for k in keys}
    P["items"] = tuple(items)
    return P


def _bare(n: str) -> str:
    return n[3:] if n.startswith("on:") else n


def load_pack(path, device="cpu", names=None) -> dict:
    """The pack npz of scripts/on_lon_pack.py -> tensors. names (N,) = the trainer's pool row order: the pack is re-ordered to it and must cover it
    (row ids are compared without an `on:` prefix)."""
    z = np.load(path, allow_pickle=False)
    items = tuple(str(x) for x in z["items"]) if "items" in z.files else ITEMS
    Z = {k: z[k] for k in z.files}
    if names is not None:
        n0 = len(Z["names"])
        pos = {_bare(n): i for i, n in enumerate(Z["names"].tolist())}
        want = [_bare(n) for n in np.asarray(names).tolist()]
        miss = [n for n in want if n not in pos]
        assert len(pos) == n0 and not miss, f"{path}: {len(miss)} pool rows without longitudinal labels, e.g. {miss[:2]}"
        o = np.array([pos[n] for n in want])
        Z = {k: (v[o] if k != "items" and v.ndim and len(v) == n0 else v) for k, v in Z.items()}
    return to_pack(Z, device, items)


PACK_NAME = "lon.npz"


def pack_path(pool_dir) -> Path:
    """<POOL>/lon.npz if present, else the rows dir of scripts/on_rows.py: cache/on1_<N> -> runs/onpol_nuplan/rows/<N>/lon.npz."""
    pool_dir = Path(pool_dir)
    p = pool_dir / PACK_NAME
    if p.exists():
        return p
    return pool_dir.parents[2] / "onpol_nuplan" / "rows" / pool_dir.name.split("_", 1)[-1] / PACK_NAME


def load(pool_dir, names, dev) -> dict:
    """on_train.py's hook (plans/on_pool_format.md)."""
    return load_pack(pack_path(pool_dir), dev, names)


# ---------------------------------------------------------------- pack builder (numpy; everything in the rows' own frames)
def _lc():
    exp = Path(__file__).resolve().parents[2]
    for p in (exp / "long_contact/lib", exp / "body1/lib"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import lc
    return lc


def build_rows(R, H, rlen, box, valid, cls, ln_type, ln_d, ln_elig, ln_red, tn_ok, tn_spk, tn_vpk, vlog, D, ok=None):
    """Per-row label tensors of a chunk of S rows. All geometry in each row's own rear-axle frame at its time.
      R (S, NR, 2), H (S, NR), rlen (S,)      route grid, heading, known length
      box (S, 9, K, 5), valid (S, 9, K), cls (S, K)   actor boxes at t + 0, 0.5 .. 4 s (agent_labels layout; class 0 = vehicle, -1 = empty slot)
      ln_type / ln_d / ln_elig (S, NL), ln_red (S, NL, 41) bool   stop lines; red at the log frame nearest each dense step of THIS row
      tn_ok, tn_spk, tn_vpk (S,), vlog (S, NR)  first >= 45 deg turn and the route's logged speed by arc
      D (S, 8)                                 the log's travel over t .. t + 0.5 k (nan where the log has ended)
    The route's NR must equal lc.NR in this process (lc.project reads the module constant)."""
    LC = _lc()
    SW = LC.SW
    S, NR = R.shape[:2]
    assert NR == LC.NR, f"route grid {NR} != lc.NR {LC.NR}"
    ok = np.ones(S, bool) if ok is None else ok
    R, H, rlen = np.nan_to_num(R).astype(np.float32), np.nan_to_num(H).astype(np.float32), np.nan_to_num(rlen).astype(np.float32)   # finite also on not-ok rows
    ar = np.arange(S)
    # the row's own pose on the route
    p0 = np.zeros((S, 2, 2), np.float32)
    p0[:, 1, 0] = FRONT
    s0, l0 = LC.project(R, p0)
    sr0, sf0 = s0[:, 0], s0[:, 1]
    # ---- A
    b, v, vel = SW.dense_boxes(box.astype(np.float32), valid)
    v &= (cls >= 0)[:, None]
    elig, _ = LC.a_eligible(box[:, 0].astype(np.float32), valid[:, 0] & (cls >= 0), cls, vel[:, 0], R, H, rlen)
    cx, cy = SW.corners(box[:, 0, :, 0], box[:, 0, :, 1], box[:, 0, :, 2], box[:, 0, :, 3] / 2, box[:, 0, :, 4] / 2)      # (S, K, 4)
    sc0, _ = LC.project(R, np.stack([cx, cy], -1).astype(np.float32))
    sig0 = np.where(elig, sc0.min(-1), np.inf)
    pick = np.argsort(sig0, 1, kind="stable")[:, :NA]                           # (S, NA) nearest eligible first
    pe = np.take_along_axis(elig, pick, 1)
    bp = np.take_along_axis(b, pick[:, None, :, None], 2)                       # (S, 41, NA, 5)
    vp = np.take_along_axis(v, pick[:, None, :], 2) & pe[:, None]
    qx, qy = SW.corners(bp[..., 0], bp[..., 1], bp[..., 2], bp[..., 3] / 2, bp[..., 4] / 2)                              # (S, 41, NA, 4)
    sig = LC.project(R, np.stack([qx, qy], -1).astype(np.float32))[0].min(-1)   # (S, 41, NA)
    lam = LC.project(R, bp[..., :2].astype(np.float32))[1]
    a_val = vp.transpose(0, 2, 1)
    a_sig = np.where(a_val, sig.transpose(0, 2, 1), 0).astype(np.float32)
    a_lam = np.where(a_val, lam.transpose(0, 2, 1), BIG).astype(np.float32)
    a_g0 = np.where(pe, a_sig[:, :, 0] - sf0[:, None], -BIG).astype(np.float32)
    a_m = np.minimum(M_A0, np.maximum(a_g0 - M_A_BACK, 0.0)).astype(np.float32)
    # ---- B (traffic-light lines only)
    dl = np.nan_to_num(ln_d, nan=BIG).astype(np.float32)
    b_use = ln_elig & (ln_type == 2) & np.isfinite(ln_d) & (dl >= sf0[:, None])
    b_mu = ln_red & b_use[..., None]
    b_th = np.where(b_use, np.maximum(dl - M_B, sf0[:, None]), BIG).astype(np.float32)
    # ---- C
    c_ok = tn_ok & (np.nan_to_num(tn_vpk) >= C_VLOG_MIN) & np.isfinite(tn_spk)
    spk = np.nan_to_num(tn_spk).astype(np.float32)
    hpk = H[ar, np.clip(np.round(spk).astype(np.int64), 0, NR - 1)]
    vl = np.where(np.isfinite(vlog), vlog.astype(np.float32), -np.inf)
    vref = np.where(np.isfinite(vl), np.maximum(vl, np.where(c_ok, np.nan_to_num(tn_vpk), 0.0).astype(np.float32)[:, None]), -np.inf)
    c_cap = np.where(np.isfinite(vref), np.maximum(vref + C_ADD, C_MUL * vref), BIG).astype(np.float32)
    # ---- P
    ahead = lambda s: (s >= sr0[:, None] - C_POST) & (s <= sr0[:, None] + OPEN_M)  # noqa: E731
    has_a = (pe & (a_g0 <= OPEN_M)).any(1)
    has_b = (ln_elig & np.isfinite(ln_d) & ahead(dl)).any(1)
    has_c = c_ok & ahead(spk[:, None])[:, 0]
    p_open_design = ~(has_a | has_b | has_c)
    any_line = ((ln_type > 0) & np.isfinite(ln_d) & ahead(dl)).any(1)
    vk = valid & (cls >= 0)[:, None]                                            # (S, 9, K) every labelled actor at the 9 label times
    sa, la = LC.project(R, box[..., :2].reshape(S, -1, 2).astype(np.float32))
    sa, la = sa.reshape(vk.shape), la.reshape(vk.shape)
    any_actor = (vk & (np.abs(la) <= P_ACT_LAT) & (sa >= sr0[:, None, None]) & (sa <= sr0[:, None, None] + OPEN_M)).any((1, 2))
    p_open = p_open_design & ~any_line & ~any_actor
    p_D = np.where(np.isfinite(D), D, -2 * BIG).astype(np.float32)
    return dict(route=R, rhead=H, rlen=rlen, ok=ok, s0=sr0, l0=l0[:, 0], sf0=sf0,
                a_sig=a_sig, a_lam=a_lam, a_val=a_val, a_m=a_m, a_g0=a_g0, a_slot=np.where(pe, pick, -1).astype(np.int8),
                b_d=dl, b_th=b_th, b_mu=b_mu, b_use=b_use, c_ok=c_ok, c_spk=spk, c_hpk=hpk.astype(np.float32), c_cap=c_cap,
                p_D=p_D, p_open=p_open, p_open_design=p_open_design, has_a=has_a, has_b=has_b, has_c=has_c, any_line=any_line, any_actor=any_actor)
