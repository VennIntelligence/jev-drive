"""ONPOL-LONG on-policy terms K (corridor) and C (lane centre) on the student's own plan (plans/2026-10-12-ol1-prereg.md, "Labels";
research/next-round/closed-loop.md 2.2). Plugged into onpol_nuplan's trainer unedited through its hook (on_train.py --on-lon, pool contract
experiments/onpol_nuplan/plans/on_pool_format.md):

    pack = ol_kc.load(pool_dir, names, device)    # <pool>/corr.npz and <pool>/lane.npz of scripts/ol_labels.py build, in the trainer's row order
    L = ol_kc.loss(poses_rear, rows, pack)        # poses_rear (n, 8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in each row's own frame (with grad),
                                                  # rows (n,) pool row indices -> {"kcorr", "klane"}: (n,) per-row losses in metres
The trainer applies lambda (--on-lon-lam 10) and the hinge-only row weight (ho_w 3), summed over the rows / imitation rows of the batch.

  K  kcorr = mean over the 8 poses of relu(d_k - 1.5 m), d_k = distance of the rear axle to the logged rear-axle path (polyline, 1 m vertices)
  C  klane = mean over the poses at 1 .. 4 s (k = 2 .. 8) of relu(-m(c_k)), c_k = the box centre (rear axle + 1.461 m along the yaw), m = the
     lane-centre margin raster of lane.npz sampled bilinearly (half lane width - offset from the nearest-fitting lane centre - half ego width -
     0.2 m); a centre outside the raster or a row without lanes is not charged
Both are shape-only as S's hinges (body1 loss43.shape_only): values unchanged, the gradient of a rear-axle position keeps only its component
across the pose's own detached heading; the yaw keeps its gradient (C through the box centre). Neither asks for speed.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEAD, CEN = 1.5, 1.461


def shape_only(x, y, psi):
    """= body1 loss43.shape_only (copied: importing loss43 pulls pp_train)."""
    xd, yd, pd = x.detach(), y.detach(), psi.detach()
    nx, ny = -torch.sin(pd), torch.cos(pd)
    s = nx * (x - xd) + ny * (y - yd)
    return xd + nx * s, yd + ny * s


def _rows(f, names):
    z = np.load(f)
    pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
    idx = np.array([pos[t] for t in np.asarray(names).tolist()])
    return z, idx


def load(pool_dir, names, dev):
    pool_dir = Path(pool_dir)
    zc, ic = _rows(pool_dir / "corr.npz", names)
    zl, il = _rows(pool_dir / "lane.npz", names)
    g = torch.as_tensor(zc["gpath"][ic], dtype=torch.float32, device=dev)
    assert torch.isfinite(g).all() and g.shape[1:] == (86, 2), g.shape
    return dict(gpath=g, m=torch.as_tensor(zl["m"][il], dtype=torch.float16, device=dev), lane_ok=torch.as_tensor(zl["ok"][il], device=dev),
                x0=float(zl["x0"]), y0=float(zl["y0"]), res=float(zl["res"]))


def corridor(xs, ys, G):
    """Rear-axle points (n, K) and paths (n, V, 2) -> distance (n, K) to the polyline (sqrt with a floor: finite gradient at 0)."""
    A, AB = G[:, :-1], G[:, 1:] - G[:, :-1]                                      # (n, S, 2)
    P = torch.stack([xs, ys], -1)                                                # (n, K, 2)
    w = P[:, :, None] - A[:, None]                                               # (n, K, S, 2)
    t = ((w * AB[:, None]).sum(-1) / (AB * AB).sum(-1).clamp_min(1e-9)[:, None]).clamp(0, 1)
    d2 = ((w - t[..., None] * AB[:, None]) ** 2).sum(-1).amin(-1)
    return torch.sqrt(d2 + 1e-12)


def lane_margin(cx, cy, pack, rows):
    """Box centres (n, K) -> (margin (n, K), inside (n, K) bool), bilinear on the lane raster (cells at x0 + res (i + 0.5))."""
    m = pack["m"][rows].float()[:, None]                                         # (n, 1, H, W): H along x, W along y
    H, W = m.shape[-2:]
    u = 2 * (cy - pack["y0"]) / (pack["res"] * W) - 1
    v = 2 * (cx - pack["x0"]) / (pack["res"] * H) - 1
    val = F.grid_sample(m, torch.stack([u, v], -1)[:, None], mode="bilinear", padding_mode="border", align_corners=False)[:, 0, 0]
    inside = (u.abs() <= 1) & (v.abs() <= 1) & pack["lane_ok"][rows][:, None]
    return val, inside


def loss(poses, rows, pack):
    x, y, psi = poses[..., 0], poses[..., 1], poses[..., 2]
    xs, ys = shape_only(x, y, psi)
    kcorr = torch.relu(corridor(xs, ys, pack["gpath"][rows]) - DEAD).mean(1)
    cx, cy = xs[:, 1:] + CEN * torch.cos(psi[:, 1:]), ys[:, 1:] + CEN * torch.sin(psi[:, 1:])
    m, inside = lane_margin(cx, cy, pack, rows)
    klane = (torch.relu(-m) * inside).mean(1)
    return {"kcorr": kcorr, "klane": klane}
