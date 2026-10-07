"""Drivable hinge on the devkit tracker's replayed footprint (plans/2026-10-07-replay-hinge-prereg.md, arms R / RM).

Same labels, footprint, sampling and normalisation as lib/drivable_hinge.Hinge (41 states x 4 corners, mean relu(margin - sdf)), but the 41 states
are the LQR + bicycle replay of the 8 plan poses (lib/lqr_proxy.replay, initial v0 / a0 of the row's ego status) instead of their linear
interpolation. front_turn_margin > 0 (RM): the two front corners take that margin on rows whose logged 4 s heading change exceeds 20 deg.
"""
import numpy as np
import torch

import lqr_proxy as LP
from drivable_hinge import Hinge, X0, Y0

TURN_DEG = 20.0


class ReplayHinge(Hinge):
    def __init__(self, label_file, tab, dev, margin: float = 0.3, footprint="pacifica", front_turn_margin: float = 0.0):
        """tab: the op_parity cache tab of the rows (names, vel, acc, fut), in row order."""
        super().__init__(label_file, tab["names"], dev, margin, footprint)
        self.v0 = torch.as_tensor(tab["vel"][:, -1, 0], dtype=torch.float64, device=dev)
        self.a0 = torch.as_tensor(tab["acc"][:, -1, 0], dtype=torch.float64, device=dev)
        fut = np.nan_to_num(tab["fut"][:, 7, 2].astype(np.float64))
        self.turn = torch.as_tensor(np.abs(np.degrees(np.arctan2(np.sin(fut), np.cos(fut)))) > TURN_DEG, device=dev)
        self.front_turn_margin = front_turn_margin

    def margins(self, x, y, psi, rows):
        S = LP.replay(torch.stack([x, y, psi], -1).double(), self.v0[rows], self.a0[rows]).float()      # (B, 41, 3)
        C = self.C[self.src[rows]]
        c, s = torch.cos(S[..., 2:3]), torch.sin(S[..., 2:3])
        cx = S[..., :1] + c * C[:, None, :, 0] - s * C[:, None, :, 1]
        cy = S[..., 1:2] + s * C[:, None, :, 0] + c * C[:, None, :, 1]
        xy = torch.stack([cx, cy], -1).reshape(len(S), -1, 2)
        g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
        return torch.nn.functional.grid_sample(self.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]

    def __call__(self, x, y, psi, rows):
        ok = self.ok[rows]
        if not ok.any():
            return x.sum() * 0.0
        r = rows[ok]
        v = self.margins(x[ok], y[ok], psi[ok], r)
        m = torch.full_like(v, self.margin)
        if self.front_turn_margin > 0:
            front = torch.tensor([1.0, 1.0, 0.0, 0.0], device=v.device).repeat(41).bool()           # corners FL, FR, RL, RR per state
            m = torch.where(self.turn[r][:, None] & front[None], torch.full_like(v, self.front_turn_margin), m)
        return torch.relu(m - v).mean()
