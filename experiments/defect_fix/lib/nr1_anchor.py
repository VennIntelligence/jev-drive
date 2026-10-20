"""LON-ANCHOR: a station-anchored two-sided longitudinal term on the pool's S rows (rm_train --on-anchor W; prereg
experiments/defect_fix/plans/2026-10-20-lon-anchor-prereg.md + supplement 1). Carried by nr1_lat (the pool's one --on-lon module):
`lon_anchor` = 10 x W x term, the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every NR1 term.

Target (labels built offline into <pool>/LABELS by scripts/lon_anchor.py labels): the ego is at station s_e on the human's logged
path; t_h = the first log time the human was at s_e; the target at the plan's time points tau_j = 0.5 j s (j = 1..8) is the human's
own travel from there over the same tau: s_t,j = S_h(t_h + tau_j) - s_e (S_h = the human's station, non-decreasing). So the
human's speed profile is started from where the ego is now, not from the same log time (JT's arc term, nr1_lon). No windows.

Rows: S rows (tab mode `self`) whose label is defined: the ego's state inside the log's time span and not behind the human's start,
the human reaching s_e (t_h finite) and the log covering t_h + 4 s. Two exclusions:
  lead   a lead on the human's path (an object centre within LEAD_LAT of the path at LEAD_LO .. LEAD_HI m ahead of s_e) closer to
         the ego at t than the human's lead was at t_h (by more than GAP_TOL; the human's lead unknown counts as farther).
  clock  (supplement 1) the target disagrees with the human's time-aligned motion c_j = S_h(t + tau_j) - S_h(t): the target moves
         >= GO_M over 4 s while the human at the same clock stood (c_8 < STAND_V x 4 s), or the target ends standing (its last
         0.5 s step < STAND_V) while the human at the same clock moved (his last 0.5 s step >= STAND_V).

Term (supplement 1): the plan's progress p_j = the station of pose j projected on the human's path (the row-frame polyline of the
label file, 1 m vertices at s_e - 5 .. s_e + 95 m, straight beyond the log's ends) minus the station of the origin; d_j = p_j - s_t,j;
smooth L1 (beta 0.5) of d summed over the 8 poses: too fast and too slow cost the same. Moving laterally onto or off the human's line
adds no progress (the registered raw plan arc did: G1 of 2026-10-20). Longitudinal only: the projection is taken of nr1_lon.lon_only
poses, so each pose moves only along its own heading, no gradient across it, none on the yaw: path shape untouched. The nearest
segment of each pose is chosen on the detached pose.
"""
from __future__ import annotations

import numpy as np

H_FUT, DT, BETA = 4.0, 0.5, 0.5
LEAD_LAT, LEAD_LO, LEAD_HI, GAP_TOL = 1.8, 2.0, 60.0, 0.5
GO_M, STAND_V = 2.0, 0.5
PATH_A = np.arange(-5.0, 96.0)                  # label path vertices (m, relative to s_e)
LABELS = "lon_anchor_v2.npz"                   # supplement 1 (v1 = lon_anchor.npz: the registered rule, kept)
WHY = ("ok", "not_s", "no_path", "no_reach", "log_end", "lead", "lead_unknown", "clock_go", "clock_stop")


def first_at(s_e, T, S):
    """The first log time (s) at which the human's station S (non-decreasing, at log times T) reaches s_e; inf if never."""
    S = np.maximum.accumulate(np.asarray(S, np.float64))
    if s_e > S[-1]:
        return np.inf
    i = int(np.searchsorted(S, s_e, "left"))
    if i == 0:
        return float(T[0])
    return float(T[i - 1] + (T[i] - T[i - 1]) * (s_e - S[i - 1]) / max(S[i] - S[i - 1], 1e-9))


def target(t, s_e, T, S):
    """One state (log time t s, ego station s_e) -> (s_t (8,) f8 or None, t_h, why)."""
    T, S = np.asarray(T, np.float64), np.maximum.accumulate(np.asarray(S, np.float64))
    if len(T) < 2 or not (T[0] <= t <= T[-1]) or s_e < S[0] - 1.0:
        return None, np.nan, "no_path"
    th = first_at(s_e, T, S)
    if not np.isfinite(th):
        return None, np.inf, "no_reach"
    if th + H_FUT > T[-1] + 1e-6:
        return None, th, "log_end"
    return np.interp(th + DT * np.arange(1, 9), T, S) - s_e, th, "ok"


def clock8(t, T, S):
    """The human's travel from his own position at the same log time t over tau_j (the time-aligned motion)."""
    S = np.maximum.accumulate(np.asarray(S, np.float64))
    return np.interp(t + DT * np.arange(1, 9), T, S) - np.interp(t, T, S)


def clock_skip(s_t, c):
    """Target s_t (8,), the time-aligned human travel c (8,) -> "ok" / "clock_go" / "clock_stop" (module docstring)."""
    if s_t[-1] >= GO_M and c[-1] < STAND_V * H_FUT:
        return "clock_go"
    if (s_t[-1] - s_t[-2]) / DT < STAND_V and (c[-1] - c[-2]) / DT >= STAND_V:
        return "clock_stop"
    return "ok"


def lead_gap(P, a, B):
    """Path points P (m, 2) at arcs a (m,) ahead of s_e (1 m grid), object centres B (k, 2) -> the smallest arc, among the objects
    whose nearest path point lies in [LEAD_LO, LEAD_HI] and within LEAD_LAT of the centre, of that nearest point (inf: none)."""
    m = (a >= LEAD_LO - 1.0) & (a <= LEAD_HI + 1.0)
    if not len(B) or not m.any():
        return np.inf
    d = np.hypot(B[:, None, 0] - P[None, m, 0], B[:, None, 1] - P[None, m, 1])
    j = d.argmin(1)
    an, dn = a[m][j], d[np.arange(len(B)), j]
    hit = (dn < LEAD_LAT) & (an >= LEAD_LO) & (an <= LEAD_HI)
    return float(an[hit].min()) if hit.any() else np.inf


def lead_skip(gap_e, gap_h):
    """-> why ("ok" / "lead" / "lead_unknown"); gap_h NaN = the human's lead could not be measured."""
    if not np.isfinite(gap_e):
        return "ok"
    if np.isnan(gap_h):
        return "lead_unknown"
    return "lead" if gap_e < gap_h - GAP_TOL else "ok"


def load(pool_dir, names, dev):
    """-> pack: s_t (N, 8), ok (N,), pi (N,) index into path (M, 101, 2) on dev from <pool_dir>/LABELS (names = the pool's tab names)."""
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / LABELS)
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), f"nr1_anchor: {LABELS} rows != the pool's"
    ok, s_t, pi = z["ok"].astype(bool), np.nan_to_num(z["s_t"]).astype(np.float32), z["pidx"].astype(np.int64)
    assert np.isfinite(z["s_t"][ok]).all() and (pi[ok] >= 0).all() and np.isfinite(z["path"]).all(), "nr1_anchor: label file"
    why = z["why"].astype(str)
    rec = dict(file=LABELS, rows=int(len(ok)), anchor_rows=int(ok.sum()), why={w: int((why == w).sum()) for w in WHY if (why == w).any()})
    return dict(s_t=torch.as_tensor(s_t, device=dev), ok=torch.as_tensor(ok, device=dev), pi=torch.as_tensor(np.maximum(pi, 0), device=dev),
                path=torch.as_tensor(z["path"].astype(np.float32), device=dev), rec=rec)


def station(x, y, P):
    """Points x, y (n, k) -> station (n, k) of their projection on the polylines P (n, m, 2) (arc from P's first vertex); the segment
    is picked on the detached points, the station is differentiable in x, y (along the segment; past the polyline's ends it continues
    along the end segments)."""
    import torch
    A, U = P[:, :-1], P[:, 1:] - P[:, :-1]                                          # (n, m-1, 2)
    L2 = (U * U).sum(-1).clamp_min(1e-9)
    cum = torch.cat([torch.zeros_like(L2[:, :1]), torch.sqrt(L2).cumsum(1)], 1)        # (n, m) arc of each vertex
    q = torch.stack([x, y], -1)                                                     # (n, k, 2)
    with torch.no_grad():
        w = (((q[:, :, None] - A[:, None]) * U[:, None]).sum(-1) / L2[:, None]).clamp(0, 1)
        f = A[:, None] + w[..., None] * U[:, None]
        j = ((q[:, :, None] - f) ** 2).sum(-1).argmin(-1)                             # (n, k)
    g = lambda T_: T_.gather(1, j[..., None].expand(-1, -1, T_.shape[-1]) if T_.dim() == 3 else j)  # noqa: E731
    Aj, Uj, L2j, cj = g(A), g(U), g(L2), g(cum[:, :-1])
    wj = ((q - Aj) * Uj).sum(-1) / L2j                                               # unclamped: continuous across vertices (a clamp
    return cj + wj * torch.sqrt(L2j)                                                 # kills the gradient at one), linear past the ends


def progress(poses, P):
    """Plan poses (n, 8, 3) -> (n, 8) station gained along P from the origin's station (lon_only poses: along-heading gradient only)."""
    import torch
    import nr1_lon
    p = poses.float()
    xl, yl = nr1_lon.lon_only(p[..., 0], p[..., 1], p[..., 2])
    s = station(torch.cat([torch.zeros_like(xl[:, :1]), xl], 1), torch.cat([torch.zeros_like(yl[:, :1]), yl], 1), P)
    return s[:, 1:] - s[:, :1].detach()


def terms(poses, s_t, ok, P, parts=False):
    import torch
    import torch.nn.functional as F
    d = progress(poses, P) - s_t
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1) * ok.to(d.dtype)
    return (v, d.detach()) if parts else v


def term(poses, rows, pack):
    return terms(poses, pack["s_t"][rows], pack["ok"][rows], pack["path"][pack["pi"][rows]])


if __name__ == "__main__":                     # known-answer checks (CPU)
    import torch
    T = np.arange(0, 20.01, 0.5)
    S = np.minimum(10.0 * T, 100.0)            # 10 m/s, then standing at 100 m from t = 10 s
    st, th, w = target(3.0, 40.0, T, S)        # ego at 40 m at t = 3 s (the human was there at 4 s): his 4 .. 8 s profile
    assert w == "ok" and abs(th - 4.0) < 1e-9 and np.allclose(st, 5.0 * np.arange(1, 9)), (st, th)
    st, th, w = target(3.0, 80.0, T, S)        # ego at 80 m: the human comes to rest at 100 m 2 s later
    assert w == "ok" and np.allclose(st, np.minimum(5.0 * np.arange(1, 9), 20.0)), st
    assert target(17.0, 90.0, T, 5.0 * T)[2] == "log_end" and target(3.0, 150.0, T, S)[2] == "no_reach" and target(25.0, 10.0, T, S)[2] == "no_path"
    Ts = np.arange(0, 30.01, 0.5)
    Ss = np.where(Ts < 10, 0.0, 5.0 * (Ts - 10))                                     # stands 10 s, then 5 m/s
    st, th, _ = target(2.0, 0.05, Ts, Ss)                                            # a few cm crept at t = 2 s: his launch profile ...
    assert clock_skip(st, clock8(2.0, Ts, Ss)) == "clock_go"                          # ... while he still stood at that clock
    st, th, _ = target(3.0, 80.0, T, S)
    assert clock_skip(st, clock8(3.0, T, S)) == "clock_stop"                          # target stops at 100 m, the clock human drives on
    assert clock_skip(st, clock8(9.0, T, S)) == "ok"
    P, a = np.c_[np.arange(0.0, 70.0), np.zeros(70)], np.arange(0.0, 70.0)
    assert lead_gap(P, a, np.array([[30.0, 1.0]])) == 30.0 and lead_gap(P, a, np.array([[30.0, 3.0]])) == np.inf
    assert lead_skip(20.0, 30.0) == "lead" and lead_skip(30.0, 30.2) == "ok" and lead_skip(np.inf, 5.0) == "ok" and lead_skip(5.0, np.nan) == "lead_unknown"
    t = torch.linspace(0.5, 4.0, 8)
    Pt = torch.as_tensor(np.c_[PATH_A, np.full(len(PATH_A), 2.0)], dtype=torch.float32)[None]   # the human line 2 m left of the ego
    s_t = torch.as_tensor(np.minimum(5.0 * np.arange(1, 9), 20.0), dtype=torch.float32)[None]
    pack = dict(s_t=s_t, ok=torch.tensor([True]), pi=torch.tensor([0]), path=Pt)
    r0 = torch.tensor([0])
    e = 2.0 * torch.clamp(1 - s_t[0] / 8.0, min=0)                                     # the ego-start human path: offset faded over 8 m
    hum = torch.stack([s_t[0], 2.0 - e, torch.zeros(8)], -1)[None]
    assert term(hum, r0, pack).abs().max() < 1e-4                                      # lateral rejoin adds no progress (G1 amended)
    jump = torch.stack([s_t[0], torch.full((8,), 2.0), torch.zeros(8)], -1)[None]
    assert term(jump, r0, pack).abs().max() < 1e-4                                      # not even the registered jump
    fast = torch.stack([10.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None].requires_grad_(True)
    v = term(fast, r0, pack)
    v.sum().backward()
    assert v > 10 and (fast.grad[0, :, 0] > -1e-6).all() and fast.grad[0, 4:, 0].min() > 0.9 and fast.grad[0, :, 1:].abs().max() == 0
    slow = torch.stack([1.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None]
    assert term(slow, r0, pack) > 1                                                     # two-sided: too slow costs too
    th_ = torch.linspace(0.1, 0.9, 8)                                                   # a curved plan: no gradient across the heading or on the yaw
    q = torch.stack([3 * t * torch.cos(th_), 3 * t * torch.sin(th_), th_ + 0.05], -1)[None].requires_grad_(True)
    term(q, r0, pack).sum().backward()
    g = q.grad[0]
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    assert torch.isfinite(g).all() and g[:, 2].abs().max() == 0 and across.abs().max() < 1e-5 and g[:, :2].abs().max() > 0, (across, g)
    assert term(fast, r0, dict(pack, ok=torch.tensor([False]))) == 0
    print("nr1_anchor known-answer ok", float(v.detach()))
