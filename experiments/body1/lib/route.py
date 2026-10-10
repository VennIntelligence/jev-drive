"""BODY1 route tube (prereg Amendment 7): distance of plan poses to the logged path of the same row.

The logged path of a row, in the row's own frame: the polyline through the logged t0 pose and the 8 logged future poses (0.5 .. 4 s), extended
by a ray at each end (backwards along the logged t0 heading, forwards along the heading of the last logged pose), so a plan that is faster
or slower than the log along the path is at distance 0 and only the lateral part counts. On an off-track row (ot1 / yr1 / bd4) the logged
t0 pose is not the origin: it is (0, 0, 0) of the logged frame moved by the row's `off` = (dy, dpsi) (ot_rows.to_frame).

  path_dist_np   numpy, (n, K, 3) plans -> distance (n, K) and signed lateral (n, K; + = left of the logged path)      (readers)
  path_dist      torch, differentiable in x, y -> distance (n, K)                                                      (lib/loss43.py)
  turn_deg       signed logged heading change over the 4 s (deg; + = left turn)
"""
import numpy as np


def log_t0(off: np.ndarray) -> np.ndarray:
    """off (n, 2) = (dy, dpsi) -> the logged t0 pose (n, 3) in the row's own frame."""
    dy, dp = off[:, 0].astype(np.float64), off[:, 1].astype(np.float64)
    return np.stack([-np.sin(dp) * dy, -np.cos(dp) * dy, -dp], -1)


def turn_deg(F: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Logged future (n, 8, 3) in the row's frame -> signed heading change of the log over the 4 s, degrees."""
    y = np.unwrap(np.concatenate([log_t0(off)[:, None, 2], np.nan_to_num(F[..., 2].astype(np.float64))], 1), axis=1)
    return np.degrees(y[:, -1] - y[:, 0])


def path_dist_np(P: np.ndarray, F: np.ndarray, off: np.ndarray):
    """P (n, K, 3) plan poses, F (n, 8, 3) logged future, off (n, 2); all in the row's frame -> (d (n, K) >= 0, lat (n, K) signed, + left)."""
    L = np.concatenate([log_t0(off)[:, None], np.nan_to_num(F.astype(np.float64))], 1)            # (n, 9, 3)
    p = P[..., None, :2].astype(np.float64)                                                       # (n, K, 1, 2)
    A, Bp = L[:, None, :-1, :2], L[:, None, 1:, :2]                                               # (n, 1, 8, 2)
    ab = Bp - A
    l2 = (ab ** 2).sum(-1)
    t = np.clip(((p - A) * ab).sum(-1) / np.maximum(l2, 1e-9), 0.0, 1.0)
    q = A + t[..., None] * ab
    ud = np.stack([np.cos(L[:, :-1, 2]), np.sin(L[:, :-1, 2])], -1)[:, None]                      # heading of the segment's first pose
    u = np.where((l2 > 2.5e-3)[..., None], ab / np.sqrt(np.maximum(l2, 1e-9))[..., None], ud)      # direction: the segment, or the yaw if it is < 5 cm
    u0 = np.stack([np.cos(L[:, 0, 2]), np.sin(L[:, 0, 2])], -1)[:, None, None]                    # rays
    u8 = np.stack([np.cos(L[:, -1, 2]), np.sin(L[:, -1, 2])], -1)[:, None, None]
    a0, a8 = L[:, None, None, 0, :2], L[:, None, None, -1, :2]
    q0 = a0 + np.minimum(((p - a0) * u0).sum(-1), 0.0)[..., None] * u0
    q8 = a8 + np.maximum(((p - a8) * u8).sum(-1), 0.0)[..., None] * u8
    Q = np.concatenate([q, q0, q8], -2)                                                           # (n, K, 10, 2)
    U = np.concatenate([u, np.broadcast_to(u0, q0.shape), np.broadcast_to(u8, q8.shape)], -2)
    r = p - Q
    d = np.hypot(r[..., 0], r[..., 1])
    j = d.argmin(-1)[..., None]
    dm = np.take_along_axis(d, j, -1)[..., 0]
    rj = np.take_along_axis(r, j[..., None], -2)[..., 0, :]
    uj = np.take_along_axis(U, j[..., None], -2)[..., 0, :]
    return dm, np.sign(uj[..., 0] * rj[..., 1] - uj[..., 1] * rj[..., 0]) * dm


def path_dist(x, y, F, off):
    """torch: x, y (n, K) plan positions, F (n, 8, 3) logged future, off (n, 2), the row's frame -> distance (n, K) to the extended logged
    path. The gradient with respect to (x, y) is the unit vector away from the nearest point of the path (0 on the path)."""
    import torch
    dy, dp = off[:, 0], off[:, 1]
    L0 = torch.stack([-torch.sin(dp) * dy, -torch.cos(dp) * dy, -dp], -1)
    L = torch.cat([L0[:, None], F], 1)                                                            # (n, 9, 3)
    p = torch.stack([x, y], -1)[:, :, None]                                                       # (n, K, 1, 2)
    A, Bp = L[:, None, :-1, :2], L[:, None, 1:, :2]
    ab = Bp - A
    t = (((p - A) * ab).sum(-1) / (ab ** 2).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
    q = A + t[..., None] * ab
    u0 = torch.stack([torch.cos(L[:, 0, 2]), torch.sin(L[:, 0, 2])], -1)[:, None, None]
    u8 = torch.stack([torch.cos(L[:, -1, 2]), torch.sin(L[:, -1, 2])], -1)[:, None, None]
    a0, a8 = L[:, None, None, 0, :2], L[:, None, None, -1, :2]
    q0 = a0 + ((p - a0) * u0).sum(-1).clamp_max(0.0)[..., None] * u0
    q8 = a8 + ((p - a8) * u8).sum(-1).clamp_min(0.0)[..., None] * u8
    r = p - torch.cat([q, q0, q8], -2).detach()                                                   # nearest point held fixed: gradient = r / |r|
    return (r.pow(2).sum(-1) + 1e-12).sqrt().amin(-1)
