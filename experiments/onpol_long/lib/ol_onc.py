"""ONPOL-LONG Amendment 14 (arm B'): the oncoming hinge of the on-policy rows, through on_train.py --on-lon ol_onc (PYTHONPATH experiments/onpol_long/lib).

    pack = ol_onc.load(pool_dir, names, device)   # raster file from $OL_ONC (scripts/ol_dirraster.py onc): signed distance to the oncoming region
                                                  # (lanes / roadblocks > 90 deg from the row heading, minus intersections and connectors), + outside
    L = ol_onc.loss(poses_rear, rows, pack)       # -> {"onc": (n,)} mean over the 41 dense steps x 4 footprint corners of relu(0 - sdf)

The same corner sampling as S's road hinge (lib/drivable_hinge.Hinge.margins), margin 0, shape-only like it (loss43's shape flag); rows without a
raster are not charged. The trainer multiplies by --on-lon-lam (2.5 / 5 = 0.25 / 0.5 of the road hinge's 10) and ho_w, as every --on-lon term.
"""
import os

import torch

from ol_kc import shape_only


def load(pool_dir, names, dev):
    from drivable_hinge import Hinge
    return dict(h=Hinge(os.environ["OL_ONC"], list(names), dev, 0.0, bank=True))


def loss(poses, rows, pack):
    h = pack["h"]
    x, y, psi = poses[..., 0], poses[..., 1], poses[..., 2]
    xs, ys = shape_only(x, y, psi)
    v = torch.relu(-h.margins(xs, ys, psi, rows)).mean(1)
    return {"onc": v * h.ok[rows].to(v.dtype)}
