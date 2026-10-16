"""STACK candidates (experiments/route_follow/plans/2026-10-16-stack-prereg.md C1): K = 8 structured perturbations of a plan, the same set for
the C2 consequence-head counterfactuals and the trainer's minimal-correction preference term (rm_train --pref).

A plan is (..., 8, 3) rear-axle x, y, yaw at 0.5 .. 4.0 s in the row's own frame (ego at the origin, heading 0).
  lat d     every pose moved along its own normal (-sin yaw, cos yaw) by d * min(t / 2 s, 1) (0 at t = 0, the full d from 2 s on); yaw kept
  gain g    perturb.branch_poses(P, g, 1): every pose yaw and segment course about the ego x g at the same segment lengths (curvature gain)
  arc s     perturb.branch_poses(P, 1, s): the same path re-sampled at s x each pose's arc length (slower), yaw interpolated along the arc
CANDS lists them in a fixed order; candidate 0 of a candidate set is the plan itself (cands(..., with_self=True)).
numpy (cands_np) and torch (cands_t, batched, no gradient needed) agree to <= 1e-5 m (selftest).
"""
from __future__ import annotations

import numpy as np

T = 0.5 * np.arange(1, 9)                                   # pose times (s)
CANDS = (("lat", -1.0), ("lat", -0.5), ("lat", 0.5), ("lat", 1.0), ("gain", 0.8), ("gain", 1.2), ("arc", 0.75), ("arc", 0.9))
K = len(CANDS)
RAMP = np.minimum(T / 2.0, 1.0)


def branch_np(P, g, s):
    """experiments/alpasim/lib/perturb.branch_poses (copied: importing it reads the driver env)."""
    q = np.asarray(P, np.float64)
    if g == 1.0 and s == 1.0:
        return q
    xy = np.r_[[[0.0, 0.0]], q[:, :2]]
    d = np.diff(xy, axis=0)
    ds = np.hypot(*d.T)
    c = g * np.unwrap(np.r_[0.0, np.arctan2(d[:, 1], d[:, 0])])[1:]
    yaw = g * np.unwrap(np.r_[0.0, q[:, 2]])
    xy = np.r_[[[0.0, 0.0]], np.cumsum(ds[:, None] * np.c_[np.cos(c), np.sin(c)], 0)]
    S = np.r_[0.0, np.cumsum(ds)]
    if s != 1.0:
        u = np.maximum.accumulate(S + 1e-9 * np.arange(len(S)))
        w = s * u[1:]
        return np.c_[np.interp(w, u, xy[:, 0]), np.interp(w, u, xy[:, 1]), np.interp(w, u, yaw)]
    return np.c_[xy[1:], yaw[1:]]


def one_np(P, kind, v):
    P = np.asarray(P, np.float64)
    if kind == "lat":
        a = v * RAMP
        return np.c_[P[:, 0] - a * np.sin(P[:, 2]), P[:, 1] + a * np.cos(P[:, 2]), P[:, 2]]
    return branch_np(P, v, 1.0) if kind == "gain" else branch_np(P, 1.0, v)


def cands_np(P, with_self=True):
    """(8, 3) -> (K (+1), 8, 3) float64."""
    C = [one_np(P, k, v) for k, v in CANDS]
    return np.stack(([np.asarray(P, np.float64)] if with_self else []) + C)


def _unwrap_t(a):
    import torch
    d = torch.diff(a, dim=-1)
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return torch.cat([a[..., :1], a[..., :1] + torch.cumsum(d, -1)], -1)


def _branch_t(P, g, s):
    import torch
    z = torch.zeros_like(P[:, :1, :2])
    xy = torch.cat([z, P[:, :, :2]], 1)
    d = xy[:, 1:] - xy[:, :-1]
    ds = torch.linalg.norm(d, dim=-1)
    z1 = torch.zeros_like(P[:, :1, 0])
    c = g * _unwrap_t(torch.cat([z1, torch.atan2(d[..., 1], d[..., 0])], 1))[:, 1:]
    yaw = g * _unwrap_t(torch.cat([z1, P[..., 2]], 1))
    xy = torch.cat([z, torch.cumsum(ds[..., None] * torch.stack([torch.cos(c), torch.sin(c)], -1), 1)], 1)
    if s == 1.0:
        return torch.cat([xy[:, 1:], yaw[:, 1:, None]], -1)
    S = torch.cat([z1, torch.cumsum(ds, 1)], 1)
    u = torch.cummax(S + 1e-9 * torch.arange(S.shape[1], device=P.device, dtype=P.dtype), 1).values
    w = s * u[:, 1:]
    j = torch.searchsorted(u.contiguous(), w.contiguous()).clamp(1, u.shape[1] - 1)
    u0, u1 = u.gather(1, j - 1), u.gather(1, j)
    f = ((w - u0) / (u1 - u0).clamp_min(1e-12)).clamp(0, 1)
    lerp = lambda V: V.gather(1, j - 1) + f * (V.gather(1, j) - V.gather(1, j - 1))  # noqa: E731
    return torch.stack([lerp(xy[..., 0]), lerp(xy[..., 1]), lerp(yaw)], -1)


def cands_t(P, with_self=True):
    """(B, 8, 3) torch -> (B, K (+1), 8, 3), same dtype; no gradient is taken through it (callers detach)."""
    import torch
    out = [P] if with_self else []
    r = torch.as_tensor(RAMP, device=P.device, dtype=P.dtype)[None]
    for kind, v in CANDS:
        if kind == "lat":
            a = v * r
            out.append(torch.stack([P[..., 0] - a * torch.sin(P[..., 2]), P[..., 1] + a * torch.cos(P[..., 2]), P[..., 2]], -1))
        else:
            out.append(_branch_t(P, v, 1.0) if kind == "gain" else _branch_t(P, 1.0, v))
    return torch.stack(out, 1)


def selftest(n=300):
    import torch
    rng = np.random.default_rng(0)
    worst = 0.0
    Ps = []
    for i in range(n):
        v = rng.uniform(0, 15)
        arc = np.cumsum(np.full(8, 0.5 * v) * rng.uniform(0.6, 1.2, 8)) if i % 10 else np.zeros(8)   # stopped plans too
        th = np.linspace(0, rng.uniform(-1.2, 1.2), 8)
        Ps.append(np.c_[arc * np.cos(th), arc * np.sin(th), th + rng.normal(0, 0.05, 8)])
    Ps = np.stack(Ps)
    Ct = cands_t(torch.as_tensor(Ps, dtype=torch.float64)).numpy()
    for i in range(n):
        worst = max(worst, float(np.abs(Ct[i] - cands_np(Ps[i])).max()))
    assert worst <= 1e-5, worst
    return worst


if __name__ == "__main__":
    print("stack_cand selftest max |torch - numpy|", selftest())
