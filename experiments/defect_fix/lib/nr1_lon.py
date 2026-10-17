"""NR1HVR two-sided longitudinal imitation on the on-policy pool rows (rm_train --on-arc W; prereg
experiments/defect_fix/plans/2026-10-17-nr1-prereg.md supplement 8). Carried by nr1_lat (the pool's one --on-lon module): nr1_lat.loss adds
`lon_arc` = 10 x W x term, the trainer multiplies by ho_w and sums over the pool rows / n_imit, as every NR1 lateral term.

Target: the arc length of the pool row's own imitation target `fut` (tab.npz; onpol_nuplan on_rows.py, prereg amendment 2): the logged path
from the row's projection on it, travelled at the log's speed at the same log time, blended from the ego's speed at t to the log's over the
first 2 s (v = v_ego + (v_log - v_ego) min(tau / 2 s, 1)), i.e. the human's speed profile made reachable from the visited state. Arc length =
cumulative polyline length from the origin through the 8 poses (0.5 .. 4 s), plan and target alike (wait_cap wt.py human_arc / lib/wc_arc.py).

Exclusions (label `ok` False, term exactly 0): the human (the log's own travel, tab `fut_reg`) slower than 0.5 m/s on any 0.5 s segment of
the horizon: standing still or coming to a stop for a reason the row may not show (d335 / d338: standing rows teach "stay stopped"); a row
with a non-finite target. Rows where the human launches or keeps moving stay in.
JTB (rm_train --on-arc-rows brake; NR1 prereg supplement 10): the stopping rows come in too. Excluded only: the human below 0.5 m/s on every
0.5 s segment of the horizon (standing at t0 and staying still: the serve hold's case) and non-finite targets. So the term also says "brake
with him" where the human decelerates or stops. Row kinds for the monitoring (`kind`, fut_reg segment speeds v_1..v_8): stop (v_1 >= 0.5,
min < 0.5), launch (v_1 < 0.5, max >= 0.5), brake (neither, v_8 < 0.7 v_1), cruise (the rest), still (max < 0.5, excluded).

Term: d = s_plan - s_target per pose, smooth L1 (beta 0.5) of d summed over the 8 poses: too fast and too slow cost the same.
Longitudinal only: each plan pose is moved only along its own heading (lon_only, the complement of nr1_lat.shape_only), so the term's gradient
has no component across the plan's heading and none on the yaw: lateral geometry is untouched (d321).
"""
from __future__ import annotations

import numpy as np

BETA, V_MIN, DT, EPS = 0.5, 0.5, 0.5, 1e-6


def human_arc(fut):
    """(N, 8, >=2) poses -> (N, 8) cumulative polyline arc from the origin (= wait_cap wt.py human_arc)."""
    P = np.concatenate([np.zeros((len(fut), 1, 2)), np.asarray(fut, np.float64)[..., :2]], 1)
    return np.cumsum(np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1)), 1)


KINDS = ("cruise", "brake", "stop", "launch", "still")


def labels(tab, rows="move"):
    """tab (npz-like with fut, fut_reg) -> s_h (N, 8) f4, ok (N,) bool, rec (counts), kind (N,) int8 (index into KINDS, -1 non-finite).
    rows: "move" (NR1HVR / JT: any segment below 0.5 m/s excluded) or "brake" (JTB: only the still-all-horizon rows excluded)."""
    s_h, s_r = human_arc(tab["fut"]), human_arc(tab["fut_reg"])
    seg = np.diff(np.concatenate([np.zeros((len(s_r), 1)), s_r], 1), axis=1) / DT
    fin = np.isfinite(s_h).all(1) & np.isfinite(s_r).all(1)
    stop = seg.min(1) < V_MIN
    with np.errstate(invalid="ignore"):
        still = seg.max(1) < V_MIN
        kind = np.select([still, seg[:, 0] < V_MIN, stop, seg[:, -1] < 0.7 * seg[:, 0]], [4, 3, 2, 1], 0).astype(np.int8)
    kind[~fin] = -1
    assert rows in ("move", "brake"), rows
    ok = fin & (~stop if rows == "move" else ~still)
    rec = dict(rows=int(len(ok)), labelled=int(ok.sum()), nonfinite=int((~fin).sum()), human_stand_or_stop=int((fin & stop).sum()),
               human_standing_4s=int((fin & (s_r[:, -1] < 1.0)).sum()), v_min=V_MIN)
    if rows != "move":
        rec |= dict(rule=rows, kinds={k: int((kind == i).sum()) for i, k in enumerate(KINDS)},
                    labelled_kinds={k: int(((kind == i) & ok).sum()) for i, k in enumerate(KINDS)})
    return np.nan_to_num(s_h).astype(np.float32), ok, rec, kind


def load(pool_dir, names, dev, rows="move"):
    import torch
    from pathlib import Path
    z = np.load(Path(pool_dir) / "tab.npz")
    assert np.array_equal(z["names"].astype(str), np.asarray(names).astype(str)), "nr1_lon: tab rows != the pool's"
    s_h, ok, rec, kind = labels(z, rows)
    return dict(s_h=torch.as_tensor(s_h, device=dev), ok=torch.as_tensor(ok, device=dev), rec=rec, kind=torch.as_tensor(kind, device=dev))


def lon_only(x, y, psi):
    """Poses keep their values; the gradient reaches x, y only along the pose's own heading (none across it, none on psi)."""
    import torch
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    c, s = torch.cos(pd), torch.sin(pd)
    u = c * (x - xd) + s * (y - yd)
    return xd + c * u, yd + s * u


def arc(x, y):
    """(n, 8) -> (n, 8) cumulative arc from the origin; finite gradient at a standing plan (wc_arc.arc)."""
    import torch
    dx = torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1]))
    dy = torch.diff(y, dim=1, prepend=torch.zeros_like(y[:, :1]))
    return (torch.sqrt(dx * dx + dy * dy + EPS) - EPS ** 0.5).cumsum(1)


def terms(poses, s_h, ok):
    """poses (n, 8, 3) rear-axle x, y, yaw -> (n,) raw term (0 where not ok)."""
    import torch
    import torch.nn.functional as F
    p = poses.float()
    xl, yl = lon_only(p[..., 0], p[..., 1], p[..., 2])
    d = arc(xl, yl) - s_h
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1)
    return v * ok.to(v.dtype)


def term(poses, rows, pack):
    return terms(poses, pack["s_h"][rows], pack["ok"][rows])


if __name__ == "__main__":                     # known-answer check (CPU)
    import torch
    t = torch.linspace(0.5, 4.0, 8)
    go = torch.stack([2.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None]
    s_go = (2.0 * t)[None]
    ok = torch.tensor([True])
    assert terms(go, s_go, ok).abs().max() < 1e-3                         # the EPS offset: 1e-3 m per segment
    slow, fast = terms(go * 0.5, s_go, ok), terms(go * 1.5, s_go, ok)       # 50 % too slow and too fast: equal cost (two-sided)
    assert slow > 1 and torch.allclose(slow, fast, rtol=1e-2), (slow, fast)
    assert terms(go * 0.5, s_go, torch.tensor([False])) == 0
    th = torch.linspace(0.1, 0.9, 8)                                         # a curved plan: no gradient across the heading or on the yaw
    q = torch.stack([3 * t * torch.cos(th), 3 * t * torch.sin(th), th + 0.05], -1)[None].requires_grad_(True)
    terms(q, s_go, ok).sum().backward()
    g = q.grad[0]
    assert torch.isfinite(g).all() and g[:, 2].abs().max() == 0
    across = -torch.sin(q[0, :, 2].detach()) * g[:, 0] + torch.cos(q[0, :, 2].detach()) * g[:, 1]
    assert across.abs().max() < 1e-6 and g[:, :2].abs().max() > 0, (across, g)
    z = dict(fut=np.stack([np.c_[2.0 * t.numpy(), np.zeros(8), np.zeros(8)]] * 3), fut_reg=np.stack([np.c_[2.0 * t.numpy(), np.zeros(8), np.zeros(8)],
             np.zeros((8, 3)), np.c_[np.minimum(2.0 * t.numpy(), 3.0), np.zeros(8), np.zeros(8)]]))
    s, okl, rec, kd = labels(z)                                              # moving / standing / stopping within the horizon
    assert okl.tolist() == [True, False, False] and rec["human_standing_4s"] == 1, (okl, rec)
    s, okl, rec, kd = labels(z, "brake")                                     # JTB: the stopping row comes in, the standing one stays out
    assert okl.tolist() == [True, False, True] and kd.tolist() == [0, 4, 2], (okl, kd)
    print("nr1_lon known-answer ok", float(slow), float(fast), rec)
