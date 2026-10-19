"""TIGHT-IN (NR1 prereg supplement 22, plans/2026-10-20-nr1-supp22-tightin.md; rm_train --on-tightin W): in tight turns, do not plan outside
the human's logged line (decision 397: B2 runs ~1 m wide in every sharp turn and exits where the road is narrow; garage turns in early and
stays inside). Carried by nr1_lat (the pool's one --on-lon module): `lon_tightin` (lateral part) and `lon_tightina` (arc part), each
10 x W x its part; the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every NR1 term.

Rows (folded into `ok` at load, no gradient): the pool's S rows with finite fut / fut_reg / gpath, and all of
  ego t0 speed (tab `speed`) >= V_TI; the human moving: arc(fut_reg)_8 >= H_TI (a human stopping in the turn belongs to STAND / STOPPT);
  the logged path (rows meta `gpath`, 1 m stations s0 - 15 .. s0 + 70, GI = the ego's station) turns: the heading change between the
  segments [s0, s0 + 1] and [s0 + 19, s0 + 20] is >= TURN_TI deg; and it is tight: kmax_1m over [s0, s0 + 20] >= K_TI (the max of the
  heading change between consecutive 1 m segments averaged over 2 m; = turn_sep.kmax_1m on gpath[GI - 1 : GI + 23]).
  sigma = the sign of that heading change (+1 left, -1 right).
Lateral part: the path = gpath stations LO .. HI m (ego frame). Every plan pose is projected onto the path polyline; d_j = its signed offset
  (left +); poses whose nearest point is a path end (behind LO or beyond HI) do not count. r = max(0, -sigma d_0), d_0 = the origin's (the
  ego's) offset: the reference is the more outward of the human line and the ego's own offset, so the human's line, any line inside it and a
  plan that keeps the ego's present outward offset are all free. o_j = -sigma d_j - r (outward +). Value: sum over the counted poses of
  smooth L1 (beta 0.5) of relu(o_j - M_TI). Gradient only across each pose's own heading (nr1_lat.shape_only, as the lateral terms).
Arc part: TURN-CAP's cap on the same rows: c_j = K_ARC x max(arc(fut)_j, arc(fut_reg)_j) (never below either human arc); value
  nr1_free.stand_terms (sum of smooth L1 of relu(s_plan_j - c_j)); gradient only along each pose's heading.
Labels: <pool>/tight_in.npz (scripts/tight_in.py label): names, path (N, NP, 2) = gpath[GI + LO : GI + HI + 1] (NaN off the S rows),
kmax, dh (deg). W = 0 (default): nothing loaded, nothing added.
"""
from __future__ import annotations

import numpy as np

V_TI, H_TI, TURN_TI, K_TI, M_TI, K_ARC, BETA = 2.0, 4.0, 15.0, 0.108, 0.2, 1.05, 0.5
GI, LO, HI = 15, -1, 25                           # gpath index of the ego's station; the lateral path's stations (m)
NP = HI - LO + 1


def kmax_1m(path, i0, i1):
    """path (n, m, 2) at 1 m stations -> (n,) max |curvature| (rad/m) over i0 .. i1 (= turn_sep.kmax_1m)."""
    P = np.asarray(path, np.float64)
    d = np.diff(P, axis=1)
    h = np.where(np.hypot(d[..., 0], d[..., 1]) > 0.2, np.arctan2(d[..., 1], d[..., 0]), np.nan)
    for j in range(1, h.shape[1]):
        h[:, j] = np.where(np.isnan(h[:, j]), h[:, j - 1], h[:, j])
    h = np.unwrap(np.nan_to_num(h), axis=1)
    k = np.diff(h, axis=1)
    k2 = 0.5 * (k[:, 1:] + k[:, :-1])
    return np.abs(k2[:, i0:i1]).max(1)


def geom(gpath):
    """gpath (n, 86, 2) -> kmax (n,) over [s0, s0 + 20] m, dh (n,) deg (left +) between segments [s0, s0 + 1] and [s0 + 19, s0 + 20];
    NaN where the path is not finite."""
    g = np.asarray(gpath, np.float64)
    bad = ~np.isfinite(g[:, GI - 1:GI + HI + 1]).all((1, 2))
    g = np.nan_to_num(g)
    km = kmax_1m(g[:, GI - 1:GI + 23], 0, 20)
    a = g[:, GI + 1] - g[:, GI]
    b = g[:, GI + 20] - g[:, GI + 19]
    dh = np.degrees(np.angle(np.exp(1j * (np.arctan2(b[:, 1], b[:, 0]) - np.arctan2(a[:, 1], a[:, 0])))))
    return np.where(bad, np.nan, km), np.where(bad, np.nan, dh)


def rows(v0, hum4, km, dh):
    """-> (N,) bool: the rows supplement 22 lets fire (NaN = no)."""
    hum4, km, dh = (np.nan_to_num(np.asarray(x, np.float64), nan=-1.0) for x in (hum4, km, dh))
    return (np.asarray(v0) >= V_TI) & (hum4 >= H_TI) & (np.abs(dh) >= TURN_TI) & (km >= K_TI)


def offset(xy, path):
    """xy (n, m, 2), path (n, p, 2) -> signed offset (n, m) (left +) to the polyline, and (n, m) bool: the nearest point is an end vertex
    (the point lies behind the first or beyond the last station)."""
    import torch
    A, B = path[:, :-1], path[:, 1:]
    ab = B - A                                                               # (n, p-1, 2)
    L2 = ab.pow(2).sum(-1).clamp_min(1e-6)
    rel = xy[:, :, None] - A[:, None]                                        # (n, m, p-1, 2)
    u_ = (rel * ab[:, None]).sum(-1) / L2[:, None]
    u = u_.clamp(0, 1)
    c = A[:, None] + u[..., None] * ab[:, None]
    dist = (xy[:, :, None] - c).norm(dim=-1)
    j = dist.argmin(-1, keepdim=True)
    cross = ab[:, None, :, 0] * rel[..., 1] - ab[:, None, :, 1] * rel[..., 0]
    d = (torch.sign(cross) * dist).gather(-1, j).squeeze(-1)
    uj = u_.gather(-1, j).squeeze(-1)
    jj = j.squeeze(-1)
    end = ((jj == 0) & (uj < 0)) | ((jj == ab.shape[1] - 1) & (uj > 1))
    return d, end


def lat_terms(poses, path, sigma, ok, parts=False):
    """poses (n, 8, 3) ego frame, path (n, NP, 2), sigma (n,), ok (n,) -> (n,) lateral part; parts: also the fire mask and max outward."""
    import torch
    import torch.nn.functional as F
    from nr1_lat import shape_only
    p = poses.float()
    xs, ys = shape_only(p[..., 0], p[..., 1], p[..., 2])
    P = torch.nan_to_num(path.float())
    d, end = offset(torch.stack([xs, ys], -1), P)
    with torch.no_grad():
        d0, _ = offset(torch.zeros_like(P[:, :1]), P)
        r = torch.relu(-sigma * d0[:, 0])
    o = -sigma[:, None] * d - r[:, None]
    e = torch.relu(o - M_TI) * (~end).to(o.dtype)
    with torch.no_grad():
        fire = ok & (e.amax(1) > 0)
    v = F.smooth_l1_loss(e, torch.zeros_like(e), beta=BETA, reduction="none").sum(1) * fire.to(e.dtype)
    if parts:
        return v, dict(fire=fire, out=(o.detach() * (~end)).amax(1))
    return v


def terms(poses, cap, path, sigma, ok, parts=False):
    """-> (lateral (n,), arc (n,)); parts: also dict(fire, fire_lat, fire_arc, out, s_plan)."""
    import nr1_free
    vl, pl = lat_terms(poses, path, sigma, ok, parts=True)
    va, pa = nr1_free.stand_terms(poses, cap, ok, parts=True)
    if parts:
        return vl, va, dict(fire=pl["fire"] | pa["fire"], fire_lat=pl["fire"], fire_arc=pa["fire"], out=pl["out"], s_plan=pa["s_plan"])
    return vl, va


def cap_of(fut, fut_reg):
    """-> cap (N, 8) = K_ARC x max of the two cumulative human arcs, ok (N,) finite."""
    import nr1_free
    a, b = nr1_free.cum_arc(fut), nr1_free.cum_arc(fut_reg)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return (K_ARC * np.nan_to_num(np.maximum(a, b))).astype(np.float32), ok


def load(pool_dir, names, dev):
    """-> pack: cap (N, 8), path (N, NP, 2), sigma (N,), ok (N,) on dev (labels <pool_dir>/tight_in.npz)."""
    import torch
    from pathlib import Path
    import nr1_free
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_tight: tab rows != the pool's"
    g = np.load(Path(pool_dir) / "tight_in.npz")                            # scripts/tight_in.py label
    assert np.array_equal(g["names"].astype(str), np.asarray(names).astype(str)), "nr1_tight: tight_in rows != the pool's"
    assert g["path"].shape[1:] == (NP, 2), "nr1_tight: path stations"
    cap, fin = cap_of(z["fut"], z["fut_reg"])
    S = z["mode"].astype(str) == "self" if "mode" in z.files else np.ones(len(fin), bool)
    fin &= np.isfinite(g["path"]).all((1, 2))
    ok = fin & S & rows(z["speed"], nr1_free.cum_arc(z["fut_reg"])[:, -1], g["kmax"], g["dh"])
    sig = np.where(np.nan_to_num(g["dh"]) >= 0, 1.0, -1.0).astype(np.float32)
    rec = dict(rows=int(len(ok)), s_rows=int(S.sum()), tightin_rows=int(ok.sum()), right=int((ok & (sig < 0)).sum()), left=int((ok & (sig > 0)).sum()),
               v_ti=V_TI, h_ti=H_TI, turn_ti=TURN_TI, k_ti=K_TI, m_ti=M_TI, k_arc=K_ARC, stations=[LO, HI])
    return dict(cap=torch.as_tensor(cap, device=dev), path=torch.as_tensor(np.nan_to_num(g["path"]).astype(np.float32), device=dev),
                sigma=torch.as_tensor(sig, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec)


def term(poses, rows_, pack):
    return terms(poses, pack["cap"][rows_], pack["path"][rows_], pack["sigma"][rows_], pack["ok"][rows_])


if __name__ == "__main__":                     # known-answer check (CPU)
    import sys
    import torch
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    s = np.arange(-15.0, 71.0)                                                # gpath stations

    def turn(R, sgn, start=2.0):
        """a 90 deg turn of radius R (sgn -1 right, +1 left) starting `start` m ahead, at the gpath stations -> (86, 2)."""
        th = np.clip((s - start) / R, 0, np.pi / 2)
        after = np.maximum(s - start - R * np.pi / 2, 0)
        x = np.where(s < start, s, start + R * np.sin(th) + after * np.cos(th))
        y = np.where(s < start, 0.0, sgn * (R * (1 - np.cos(th)) + after * np.sin(th)))
        return np.stack([x, y], -1)

    straight = np.stack([s, 0 * s], -1)
    G = np.stack([turn(8.0, -1), turn(8.0, 1), turn(12.0, -1), straight])
    km, dh = geom(G)
    assert abs(km[0] - 0.125) < 0.01 and abs(km[1] - 0.125) < 0.01 and km[2] < K_TI and km[3] < 1e-9, km
    assert dh[0] < -85 and dh[1] > 85 and abs(dh[3]) < 1e-6, dh
    assert rows(np.array([3.0, 3.0, 3.0, 1.5, 3.0, 3.0]), np.array([10.0, 10.0, 10.0, 10.0, 3.0, np.nan]), km[[0, 2, 3, 0, 0, 0]],
                dh[[0, 2, 3, 0, 0, 0]]).tolist() == [True, False, False, False, False, False]   # row / wide turn / straight / ego slow / human slow / nan
    Rt = turn(8.0, -1)
    path = torch.as_tensor(Rt[GI + LO:GI + HI + 1], dtype=torch.float32)[None]
    sig = torch.tensor([-1.0])
    ok1 = torch.tensor([True])

    def along(st, lat=0.0, P=Rt):
        """poses at stations st on the path P, `lat` m to the left (left +), heading along the path."""
        st = np.asarray(st, np.float64)
        x, y = np.interp(st, s, P[:, 0]), np.interp(st, s, P[:, 1])
        hx, hy = np.interp(st + 0.5, s, P[:, 0]) - np.interp(st - 0.5, s, P[:, 0]), np.interp(st + 0.5, s, P[:, 1]) - np.interp(st - 0.5, s, P[:, 1])
        yaw = np.arctan2(hy, hx)
        return torch.as_tensor(np.stack([x - np.sin(yaw) * lat, y + np.cos(yaw) * lat, yaw], -1), dtype=torch.float32)[None]

    st8 = np.arange(1, 9) * 1.5                                              # 1.5 .. 12 m: through the turn
    v, pr = lat_terms(along(st8), path, sig, ok1, parts=True)
    assert v == 0 and not pr["fire"].any(), (v, pr)                        # on the human line: 0
    assert lat_terms(along(st8, -1.0), path, sig, ok1) == 0                  # 1 m inside a right turn (to its right): free
    v, pr = lat_terms(along(st8, 0.5), path, sig, ok1, parts=True)
    assert v > 0 and pr["fire"].all() and abs(float(pr["out"]) - 0.5) < 0.02, (v, pr)   # 0.5 m wide (left of a right turn): fires
    assert lat_terms(along(st8, 0.15), path, sig, ok1) == 0                 # 0.15 m wide: inside the 0.2 m margin
    assert lat_terms(along(st8, 0.5), path, sig, ~ok1) == 0                 # not a row: 0
    wide = torch.stack([torch.as_tensor(st8, dtype=torch.float32), torch.zeros(8), torch.zeros(8)], -1)[None]
    assert lat_terms(wide, path, sig, ok1) > 1.0                             # straight on through a right turn: fires
    assert lat_terms(wide * torch.tensor([1.0, -1.0, -1.0]), torch.as_tensor(turn(8.0, 1)[GI + LO:GI + HI + 1], dtype=torch.float32)[None],
                     torch.tensor([1.0]), ok1) > 1.0                         # mirrored left turn: the same
    # the ego already 0.6 m wide (the path shifted 0.6 m to the right of the origin): keeping that offset is free, 0.9 m (0.3 more) fires
    Rs = Rt + np.array([0.0, -0.6])
    sh = torch.as_tensor(Rs[GI + LO:GI + HI + 1], dtype=torch.float32)[None]
    st_ = np.arange(1, 9) * 0.25                                             # 0.25 .. 2 m: still on the straight part
    assert lat_terms(along(st_, 0.6, Rs), sh, sig, ok1) == 0 and lat_terms(along(st_, 0.9, Rs), sh, sig, ok1) > 0
    # the ego 0.8 m inside: a plan back onto the human line is free (r = 0, o <= 0)
    Ri = Rt + np.array([0.0, 0.8])
    si = torch.as_tensor(Ri[GI + LO:GI + HI + 1], dtype=torch.float32)[None]
    back = along(st8, np.linspace(-0.7, 0.0, 8), Ri)
    assert lat_terms(back, si, sig, ok1) == 0
    # poses beyond the last station (HI = 25 m) do not count: a plan 30 .. 40 m ahead, 3 m wide
    far = along(np.linspace(30, 40, 8), 3.0, Rt)
    assert lat_terms(far, path, sig, ok1) == 0
    # the gradient: across each pose's heading only, descent moves the pose inward (to the right of a right turn)
    q = along(st8, 0.6).clone().requires_grad_(True)
    lat_terms(q, path, sig, ok1).sum().backward()
    g = q.grad[0]
    yw = q[0, :, 2].detach()
    ac = -torch.sin(yw) * g[:, 0] + torch.cos(yw) * g[:, 1]
    al = torch.cos(yw) * g[:, 0] + torch.sin(yw) * g[:, 1]
    assert g[:, 2].abs().max() == 0 and al.abs().max() < 1e-5 and (ac >= 0).all() and ac.max() > 0, g   # +across grad = descent goes right
    # the arc part (TURN-CAP's cap, the same rows) and the sum
    t = torch.linspace(0.5, 4.0, 8)
    tt = np.arange(41) * 0.1
    vv = np.maximum(4.0 + (3.0 - 4.0) * np.minimum(tt / 2.0, 1.0), 0.0)
    Dt = np.r_[0.0, np.cumsum((vv[1:] + vv[:-1]) * 0.05)][5::5]
    futt = np.stack([Dt, np.zeros(8), np.zeros(8)], -1)[None]
    regt = np.stack([3.0 * t.numpy(), np.zeros(8), np.zeros(8)], -1)[None]
    cap, okc = cap_of(futt, regt)
    assert okc.all() and abs(cap[0, -1] - 1.05 * 13.0) < 0.02
    cap = torch.as_tensor(cap)
    go = lambda v: torch.stack([v * t, torch.zeros(8), torch.zeros(8)], -1)[None]  # noqa: E731
    vl, va, pr = terms(go(4.0), cap, torch.as_tensor(straight[GI + LO:GI + HI + 1], dtype=torch.float32)[None], sig, ok1, parts=True)
    assert vl == 0 and va > 0 and pr["fire"].all() and pr["fire_arc"].all() and not pr["fire_lat"].any()   # fast on a straight path: arc only
    vl, va = terms(go(3.25), cap, torch.as_tensor(straight[GI + LO:GI + HI + 1], dtype=torch.float32)[None], sig, ok1)
    assert vl == 0 and va == 0
    print("nr1_tight known-answer ok")
