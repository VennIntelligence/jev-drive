"""ONPOL-LONG Amendment 14 (arm B'): the oncoming hinge of the on-policy rows, through on_train.py --on-lon ol_onc (PYTHONPATH experiments/onpol_long/lib).

    pack = ol_onc.load(pool_dir, names, device)   # raster file from $OL_ONC (scripts/ol_dirraster.py onc): signed distance to the oncoming region
                                                  # (lanes / roadblocks > 90 deg from the row heading, minus intersections and connectors), + outside
    L = ol_onc.loss(poses_rear, rows, pack)       # -> {"onc": (n,)} mean over the 41 dense steps x 4 footprint corners of relu(0 - sdf)

The same corner sampling as S's road hinge (lib/drivable_hinge.Hinge.margins), margin 0, shape-only like it (loss43's shape flag); rows without a
raster are not charged. OL_ONC_MODE=point (Amendment 17, variant a): the rear axle at the 41 dense steps instead of the 4 footprint corners.
The trainer multiplies by --on-lon-lam (2.5 / 5 = 0.25 / 0.5 of the road hinge's 10) and ho_w, as every --on-lon term.
"""
import os

import numpy as np
import torch

from ol_kc import shape_only


def load(pool_dir, names, dev):
    from drivable_hinge import Hinge
    return dict(h=Hinge(os.environ["OL_ONC"], np.asarray(names), dev, 0.0, bank=True))


def loss(poses, rows, pack):
    h = pack["h"]
    x, y, psi = poses[..., 0], poses[..., 1], poses[..., 2]
    xs, ys = shape_only(x, y, psi)
    if os.environ.get("OL_ONC_MODE", "") == "point":
        P0 = torch.cat([torch.zeros_like(poses[:, :1]), torch.stack([xs, ys, psi], -1)], 1)
        d = torch.einsum("kj,bjc->bkc", h.M, P0)                                  # (B, 41, 3) rear axle, as Hinge.margins interpolates
        import drivable_hinge as DH
        g = torch.stack([(d[..., 1] - DH.Y0) / 24.0 - 1.0, (d[..., 0] - DH.X0) / 32.0 - 1.0], -1)[:, :, None]
        m = torch.nn.functional.grid_sample(h.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
    else:
        m = h.margins(xs, ys, psi, rows)
    v = torch.relu(-m).mean(1)
    return {"onc": v * h.ok[rows].to(v.dtype)}
