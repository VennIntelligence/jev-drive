"""WAIT-TRAIN arc-length speed loss (plans/2026-10-17-wait-train-prereg.md, "Loss"): the --on-lon module of onpol_nuplan's trainer
(on_train.py --on-lon wc_arc, pool contract experiments/onpol_nuplan/plans/on_pool_format.md), for the merged pool on1_L18RBW.

    pack = wc_arc.load(pool_dir, names, device)   # <pool>/wc_arc.npz (wt.py pack): tokens = the pool names, s_h (N, 8) the human's arc length
                                                  # at 0.5 .. 4 s, kind (N,) int8: 0 = no term (the L18RB0 rows), 1 = two-sided (W / G rows),
                                                  # 2 = over-the-human only (S rows)
    L = wc_arc.loss(poses_rear, rows, pack)       # -> {"wcarc": (n,) per-row loss}; the trainer applies lambda (--on-lon-lam 10) x ho_w 3,
                                                  # summed over the rows / imitation rows of the batch (as every pool term)

  s_p = the plan's own arc length from the rear axle through its 8 rear-axle poses (0.5 .. 4 s); d = s_p - s_h per pose;
  kind 1: smooth L1 (beta 0.5) of d; kind 2: smooth L1 of relu(d - 1 m); summed over the 8 poses; kind 0: exactly 0.
  Arc length is invariant to rotation, so the gradient only moves poses along the plan's own segments: no path-shape term.
  The segment length is sqrt(dx^2 + dy^2 + 1e-6) - 1e-3 (a finite gradient at a standing plan, 0 for a zero segment).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BETA, STOP_DEAD, EPS = 0.5, 1.0, 1e-6


def load(pool_dir, names, dev):
    z = np.load(Path(pool_dir) / "wc_arc.npz")
    assert np.array_equal(z["tokens"], np.asarray(names)), "wc_arc.npz rows != pool rows"
    s_h, kind = z["s_h"].astype(np.float32), z["kind"].astype(np.int64)
    assert s_h.shape == (len(names), 8) and np.isfinite(s_h[kind > 0]).all() and set(np.unique(kind)) <= {0, 1, 2}
    return dict(s_h=torch.as_tensor(np.nan_to_num(s_h), device=dev), kind=torch.as_tensor(kind, device=dev))


def arc(x, y):
    """(n, 8) rear-axle x, y -> (n, 8) cumulative arc length from the origin."""
    dx = torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1]))
    dy = torch.diff(y, dim=1, prepend=torch.zeros_like(y[:, :1]))
    return (torch.sqrt(dx * dx + dy * dy + EPS) - EPS ** 0.5).cumsum(1)


def terms(poses, s_h, kind):
    d = arc(poses[..., 0], poses[..., 1]) - s_h
    d = torch.where((kind == 2)[:, None], torch.relu(d - STOP_DEAD), d)
    v = F.smooth_l1_loss(d, torch.zeros_like(d), beta=BETA, reduction="none").sum(1)
    return v * (kind > 0).to(v.dtype)


def loss(poses, rows, pack):
    return {"wcarc": terms(poses.float(), pack["s_h"][rows], pack["kind"][rows])}


if __name__ == "__main__":                     # known-answer check (CPU)
    t = torch.linspace(0.5, 4.0, 8)
    stand = torch.zeros(3, 8, 3)
    go = torch.stack([2.0 * t, torch.zeros(8), torch.zeros(8)], -1)[None].repeat(3, 1, 1)
    s_go = (2.0 * t)[None].repeat(3, 1)
    k = torch.tensor([0, 1, 2])
    a = terms(stand, s_go, k)                  # standing plan, human goes: 0 / two-sided penalty / no stop penalty
    b = terms(go, torch.zeros(3, 8), k)        # moving plan, human waits: 0 / penalty / over-only penalty past 1 m
    assert a[0] == 0 and a[1] > 10 and a[2] == 0 and b[0] == 0 and b[1] > 10 and 0 < b[2] < b[1], (a, b)
    p = go.clone().requires_grad_(True)
    terms(p, torch.zeros(3, 8), k).sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad[..., 1].abs().max() == 0 and p.grad[..., 2].abs().max() == 0
    q = stand.clone().requires_grad_(True)
    terms(q, s_go, k).sum().backward()
    assert torch.isfinite(q.grad).all()
    print("wc_arc known-answer ok", a.tolist(), b.tolist())
