"""Footprint drivable-area hinge on a plan (op_probe's DAC hinge, as a training loss for op_parity).

Labels: experiments/op_probe/scripts/opb_labels.py (drivable signed-distance raster per NAVSIM token, rear-axle frame at t0, x -8..56 m,
y -24..24 m, 0.5 m cells, + inside). Loss: the 8 plan poses (x, y, yaw at 0.5 .. 4 s, rear axle, left +) are linearly interpolated to 0.1 s, the
4 corners of the nuPlan footprint are placed on every interpolated pose, the SDF is sampled bilinearly (border clamp) and
mean(relu(margin - sdf)) is the hinge. Same geometry as op_probe's decoders (opb_probe.corners_torch / sdf_at).
A label file may also be deduplicated (tokens, ok, ix, sdf_file: token i's raster is row ix[i] of the .npy beside it, memory-mapped;
experiments/round5/scripts/road_pai.py); files without `ix` load as before.
A label file may also carry its own canvas `geom` = (x0, y0, cell_x, cell_y) of its (NH, NW) rasters (round9 S-PAI: the PAI raster window
stretched to x -8..152 m, experiments/round9/plans/2026-10-24-ntv1-spai.md); corners outside such a canvas read +20 (no hinge) instead of
the border value. Without any `geom` file the sampling is the original expression, operation for operation.
"""
from pathlib import Path

import numpy as np
import torch

X0, Y0, NH, NW = -8.0, -24.0, 128, 96
GEOM0 = (X0, Y0, 0.5, 0.5)                                                       # the stored raster's canvas: x0, y0, cell x, cell y
# nuPlan ego footprint (Pacifica) about the rear axle: front 4.049 m, rear -1.127 m, half width 1.1485 m
CORNERS = np.array([[4.049, 1.1485], [4.049, -1.1485], [-1.127, 1.1485], [-1.127, -1.1485]])
# CARLA vehicle.lincoln.mkz_2020 (B2D hero) about the rear axle, the footprint the b2d_collect SDF labels were generated with (sdf.npz `footprint`)
MKZ_CORNERS = np.array([[3.8286, 0.9184], [3.8286, -0.9184], [-1.0638, 0.9184], [-1.0638, -0.9184]])
FOOTPRINTS = {"pacifica": CORNERS, "mkz": MKZ_CORNERS}
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


class RowBank:
    """The rasters of a Hinge held once per distinct label instead of once per Store row: bank[rows] reads and bank[rows] = v writes like the
    dense (n, 1, NH, NW) tensor it replaces (rows: a long tensor; a written row gets a slot of its own). Rows without a label read slot 0,
    as arbitrary as the dense tensor's uninitialised rows. Same values, so the same hinge bit for bit; a full P2H10S run holds 281k rows
    but 103k distinct rasters per label file (the hinge-only families repeat the base tokens): 13.8 GB of card memory -> 5.2 GB."""

    def __init__(self, n, dev):
        self.pos = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.empty((0, 1, NH, NW), dtype=torch.float16, device=dev)

    def _append(self, v):
        self.t = torch.cat([self.t, v]) if len(self.t) else v
        if self.t is not v and self.t.is_cuda:
            torch.cuda.empty_cache()                                             # the replaced block goes back to the card, not to the allocator's cache

    def add(self, rows, slot, rasters):
        """rows (m,) take the slots `slot` (m,) of the new `rasters` (k, NH, NW) (numpy)."""
        self.pos[torch.as_tensor(rows, device=self.t.device)] = torch.as_tensor(slot, device=self.t.device) + len(self.t)
        self._append(torch.from_numpy(rasters).to(self.t.device)[:, None])

    def __getitem__(self, rows):
        return self.t[self.pos[rows]]

    def __setitem__(self, rows, v):
        self.pos[rows] = torch.arange(len(v), device=self.t.device) + len(self.t)
        self._append(v.to(self.t))


class Hinge:
    """Labels aligned to a row order (tokens), on `dev`; __call__(x, y, psi, rows) -> mean footprint-corner hinge over the rows with a label.

    label_file / footprint may be lists of equal length (one per label source, e.g. NAVSIM Pacifica + B2D MKZ): a row takes the first source that
    labels it and uses that source's footprint ("pacifica" | "mkz"). The default (one file, "pacifica") is the original behaviour.
    bank=True keeps the rasters in a RowBank (less card memory, same values); the default is the dense per-row tensor."""

    def __init__(self, label_file, tokens, dev, margin: float = 0.3, footprint="pacifica", bank: bool = False):
        files = label_file if isinstance(label_file, (list, tuple)) else [label_file]
        fps = footprint if isinstance(footprint, (list, tuple)) else [footprint] * len(files)
        assert len(files) == len(fps)
        n = len(tokens)
        self.sdf = RowBank(n, dev) if bank else torch.empty((n, 1, NH, NW), dtype=torch.float16, device=dev)
        ok_all, src, geo = np.zeros(n, bool), np.zeros(n, np.int64), []
        for s, f in enumerate(files):
            z = np.load(f)
            geo.append(tuple(float(v) for v in z["geom"]) if "geom" in z.files else None)
            pos = {t: i for i, t in enumerate(z["tokens"].tolist())}
            idx = np.array([pos.get(t, -1) for t in tokens.tolist()])
            ok = (idx >= 0) & z["ok"][np.maximum(idx, 0)] & ~ok_all
            if "ix" in z.files:                                                  # deduplicated file (round5 road_pai.py): token i reads raster
                sdf = np.load(Path(f).parent / str(z["sdf_file"]), mmap_mode="r")  # ix[i] of the memory-mapped sdf_file next to it
                idx = np.where(idx >= 0, z["ix"][np.maximum(idx, 0)], -1)
            else:
                sdf = z["sdf"]
            rows = np.flatnonzero(ok)
            if bank:                                                             # every distinct label of this file once
                u, inv = np.unique(idx[rows], return_inverse=True)
                self.sdf.add(rows, inv, sdf[u])
                rows = rows[:0]
            for i in range(0, len(rows), 8192):                                  # chunked: no second host copy of the full array
                r = rows[i:i + 8192]
                self.sdf[torch.as_tensor(r, device=dev), 0] = torch.from_numpy(sdf[idx[r]]).to(dev)
            ok_all |= ok
            src[ok] = s
        self.ok = torch.as_tensor(ok_all, device=dev)
        self.src = torch.as_tensor(src, device=dev)
        self.M = torch.as_tensor(interp_matrix(), dtype=torch.float32, device=dev)
        self.C = torch.as_tensor(np.stack([FOOTPRINTS[f] for f in fps]), dtype=torch.float32, device=dev)      # (S, 4, 2)
        self.margin = margin
        self.coverage = float(ok_all.mean())
        self.geo = None if all(g is None for g in geo) else torch.tensor(                  # (S, 5): x0, y0, cell x, cell y, mask outside
            [[*(g or GEOM0), float(g is not None)] for g in geo], dtype=torch.float32, device=dev)

    def _sample(self, xy, rows):
        """xy (B, K, 2) corner positions -> (B, K) SDF of each row's raster (bilinear, border clamp; a `geom` canvas: +20 outside it)."""
        if self.geo is None:
            g = torch.stack([(xy[..., 1] - Y0) / 24.0 - 1.0, (xy[..., 0] - X0) / 32.0 - 1.0], -1)[:, :, None]
            return torch.nn.functional.grid_sample(self.sdf[rows].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
        G = self.geo[self.src[rows]][:, None]                                     # (B, 1, 5)
        gx = (xy[..., 0] - G[..., 0]) / (G[..., 2] * (NH / 2)) - 1.0
        gy = (xy[..., 1] - G[..., 1]) / (G[..., 3] * (NW / 2)) - 1.0
        v = torch.nn.functional.grid_sample(self.sdf[rows].float(), torch.stack([gy, gx], -1)[:, :, None], mode="bilinear", padding_mode="border",
                                            align_corners=False)[:, 0, :, 0]
        return v.masked_fill(((gx.abs() > 1) | (gy.abs() > 1)) & (G[..., 4] > 0), 20.0)

    def margins(self, x, y, psi, rows):
        """x, y, psi (B, 8) -> sdf at the interpolated footprint corners (B, 164)."""
        P = torch.stack([x, y, psi], -1)
        P0 = torch.cat([torch.zeros_like(P[:, :1]), P], 1)
        d = torch.einsum("kj,bjc->bkc", self.M, P0)
        C = self.C[self.src[rows]]                                               # (B, 4, 2)
        c, s = torch.cos(d[..., 2:3]), torch.sin(d[..., 2:3])
        cx = d[..., :1] + c * C[:, None, :, 0] - s * C[:, None, :, 1]
        cy = d[..., 1:2] + s * C[:, None, :, 0] + c * C[:, None, :, 1]
        xy = torch.stack([cx, cy], -1).reshape(len(P), -1, 2)
        return self._sample(xy, rows)

    def __call__(self, x, y, psi, rows):
        ok = self.ok[rows]
        if not ok.any():
            return x.sum() * 0.0
        v = self.margins(x[ok], y[ok], psi[ok], rows[ok])
        return torch.relu(self.margin - v).mean()
