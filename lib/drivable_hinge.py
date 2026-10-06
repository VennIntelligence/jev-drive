"""Footprint drivable-area hinge on a plan (op_probe's DAC hinge, as a training loss for op_parity).

Labels: experiments/op_probe/scripts/opb_labels.py (drivable signed-distance raster per NAVSIM token, rear-axle frame at t0, x -8..56 m,
y -24..24 m, 0.5 m cells, + inside). Loss: the 8 plan poses (x, y, yaw at 0.5 .. 4 s, rear axle, left +) are linearly interpolated to 0.1 s, the
4 corners of the nuPlan footprint are placed on every interpolated pose, the SDF is sampled bilinearly (border clamp) and
mean(relu(margin - sdf)) is the hinge. Same geometry as op_probe's decoders (opb_probe.corners_torch / sdf_at).
"""
import numpy as np
import torch

X0, Y0, NH, NW = -8.0, -24.0, 128, 96
# nuPlan ego footprint (Pacifica) about the rear axle: front 4.049 m, rear -1.127 m, half width 1.1485 m
CORNERS = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, 1.1485], [-1.127, -1.1485]])
T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1


def interp_matrix() -> np.ndarray:
    """(41, 9): linear interpolation of [origin, 8 poses] onto the 0.1 s grid."""
    t = np.r_[0, T_POSE]
    M = np.zeros((41, 9))
    for k, tt in enumerate(T_DENSE):
        j = min(np.searchsorted(t, tt, side="right") - 1, 7)
        f = (tt - t[j]) / (t[j + 1] - t[j])
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


class Hinge:
    """Labels aligned to a row order (tokens), on `dev`; __call__(x, y, psi, rows) -> mean footprint-corner hinge over the rows with a label."""

    def __init__(self, label_file, tokens, dev, margin: float = 0.3):
        z = np.load(label_file)
        pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
        idx = np.array([pos.get(t, -1) for t in tokens.tolist()])
        ok = (idx >= 0) & z["ok"][np.maximum(idx, 0)]
        sdf = z["sdf"]
        self.sdf = torch.empty((len(tokens), 1, NH, NW), dtype=torch.float16, device=dev)
        for i in range(0, len(tokens), 8192):                                    # chunked: no second host copy of the full array
            j = np.maximum(idx[i:i + 8192], 0)
            self.sdf[i:i + len(j), 0] = torch.from_numpy(sdf[j]).to(dev)
        self.ok = torch.as_tensor(ok, device=dev)
        self.M = torch.as_tensor(interp_matrix(), dtype=torch.float32, device=dev)
        self.C = torch.as_tensor(CORNERS, dtype=torch.float32, device=dev)
        self.margin = margin
        self.coverage = float(ok.mean())

    def margins(self, x, y, psi, rows):
        """x, y, psi (B, 8) -> sdf at the interpolated footprint corners (B, 164)."""
        P = torch.stack([x, y, psi], -1)
        P0 = torch.cat([torch.zeros_like(P[:, :1]), P], 1)
        d = torch.einsum("kj,bjc->bkc", self.M, P0)
        c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
        cx = d[..., :1] + c * self.C[None, None, :, 0] - s * self.C[None, None, :, 1]
        cy = d[..., 1:2] + s * self.C[None, None, :, 0] + c * self.C[None, None, :, 1]
        xy = torch.stack([cx, cy], -1).reshape(len(P), -1, 2)
        g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
        grid = self.sdf[rows].float()
        return torch.nn.functional.grid_sample(grid, g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]

    def __call__(self, x, y, psi, rows):
        ok = self.ok[rows]
        if not ok.any():
            return x.sum() * 0.0
        v = self.margins(x[ok], y[ok], psi[ok], rows[ok])
        return torch.relu(self.margin - v).mean()
