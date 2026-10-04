"""Route-choice adapter for openpilot Cinque (experiments/op_route_ft): the navigation route as a small input, no pixels.

The route is a road-level polyline in the ego rear-axle frame (x forward, y left, vertex 0 = ego origin, 10 m vertices up to 150 m;
`route_poly`'s format). Two encodings of the same polyline:

  bear   distance to the next maneuver and the bearing of the route after it (what a car navigation gives: "in 80 m turn left"):
         [present, has_maneuver, d / 50 (<= 3), sin b, cos b - 1, b / (pi / 2)]; no maneuver within the horizon -> has 0, d 150 m, b 0
  poly   the polyline itself: [present, x_1..15 / 50, y_1..15 / 50, mask_1..15]

`present` = 0 is "no command": the adapter output is multiplied by it, so a zero feature vector is exactly the original model.

RouteAdapter (torch) maps the feature vector to a bias (32, 512) that is added to the hidden tokens of every one of the 9 policy context
frames (the same place as op_adapt_l's IntentAdapter, so op_l_onnx.py's `intent_bias` serving input carries it unchanged); its last layer
starts at zero, so the untrained adapter is the identity. `numpy_bias` evaluates a saved adapter (`adapter.npz`) without torch, for the
closed-loop server.
"""
from __future__ import annotations

import numpy as np

import route_poly as RP

D_SCALE, D_NONE = 50.0, 150.0
MAN_MIN_DEG = 25.0
ENC_DIM = {"bear": 6, "poly": 1 + 15 * 3}


# ---------------------------------------------------------------- features (numpy, train and serve)
def maneuver(poly, pmask, sigma=3.0):
    """Next maneuver on a (noised) vertex polyline: (has, d m, bearing rad). Bearing = heading of the route right after the maneuver relative to
    the ego heading (x axis), so a maneuver already in progress gives the remaining turn. A maneuver = route_poly.maneuvers segment
    (R < 50 m, merged over 8 m) whose exit heading or net angle is >= 25 deg."""
    v = np.asarray(poly, float)[: int(np.asarray(pmask, bool).sum())]
    if len(v) < 2:
        return 0.0, D_NONE, 0.0
    P = RP.smooth_path(v, 0.5, sigma)
    if P is None or len(P) < 8:
        return 0.0, D_NONE, 0.0
    psi = RP.heading(P)
    for m in RP.maneuvers(P, 0.5):
        a1 = min(m["i1"] + 2, len(P) - 1)
        b = float(psi[min(a1 + 6, len(P) - 1)])                    # 3 m past the end of the curved part
        if abs(np.degrees(b)) >= MAN_MIN_DEG or abs(m["angle"]) >= MAN_MIN_DEG:
            return 1.0, float(m["i0"] * 0.5), b
    return 0.0, D_NONE, 0.0


def feat_bear(poly, pmask, rng=None, d_sd=(2.0, 0.1), b_sd_deg=6.0):
    has, d, b = maneuver(poly, pmask)
    if rng is not None and has:
        d = max(0.0, d + rng.normal() * (d_sd[0] + d_sd[1] * d))
        b = b + np.radians(b_sd_deg) * rng.normal()
    return np.array([1.0, has, min(d, D_NONE) / D_SCALE, np.sin(b), np.cos(b) - 1.0, b / (np.pi / 2)], np.float32)


def feat_poly(poly, pmask):
    p = np.asarray(poly, np.float32)[1:16] / D_SCALE
    m = np.asarray(pmask, bool)[1:16]
    p = np.where(m[:, None], p, 0.0)
    return np.concatenate([[1.0], p[:, 0], p[:, 1], m.astype(np.float32)]).astype(np.float32)


def features(enc: str, poly, pmask, rng=None, noise=None):
    """One sample: clean polyline -> (noised when rng is given) feature vector of encoding `enc`."""
    if rng is not None:
        poly, pmask = RP.noise_polyline(poly, pmask, rng, noise or RP.NavNoise())
    if enc == "bear":
        return feat_bear(poly, pmask, rng)
    if enc == "poly":
        return feat_poly(poly, pmask)
    raise KeyError(enc)


def route_poly_from_path(path_xy, k=RP.K):
    """Dense route in the ego frame (first point near the ego) -> clean 10 m vertex polyline (k, 2), mask (k,), vertex 0 = origin.
    For a live route (closed loop): pass the route points ahead of the ego's projection, transformed to the ego rear-axle frame."""
    P, g = RP.poly_resample(np.vstack([[0.0, 0.0], np.asarray(path_xy, float)]), 0.25)
    plen = min(g[-1], RP.HORIZON)
    n = min(int(plen // RP.DS) + 1, k)
    sv = RP.DS * np.arange(n)
    out, m = np.zeros((k, 2), np.float32), np.zeros(k, bool)
    out[:n] = np.stack([np.interp(sv, g, P[:, c]) for c in range(2)], -1)
    m[:n] = True
    return out, m


# ---------------------------------------------------------------- torch adapter
def _torch():
    import torch
    import torch.nn as nn

    class RouteAdapter(nn.Module):
        """features (B, F) -> bias (B, 32, 512) = present * W2 gelu(W1' gelu(W1 f)); W2 starts at zero."""

        def __init__(self, enc: str, hidden=256, T=32, D=512):
            super().__init__()
            self.enc, self.T, self.D = enc, T, D
            self.mlp = nn.Sequential(nn.Linear(ENC_DIM[enc], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.out = nn.Linear(hidden, T * D)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, f):
            b = self.out(self.mlp(f.float())).view(-1, self.T, self.D)
            return b * f[:, :1, None].float()

        def apply(self, H, f):
            """H (B, 9, 32, 512) hidden tokens -> H + bias on every context frame."""
            return H + self(f)[:, None].to(H.dtype)

        def to_npz(self, path):
            sd = {k: v.detach().cpu().float().numpy() for k, v in self.state_dict().items()}
            np.savez(path, enc=np.array(self.enc), **{k.replace(".", "_"): v for k, v in sd.items()})

    return RouteAdapter


def RouteAdapter(enc: str, **kw):
    return _torch()(enc, **kw)


def _gelu(x):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


class NumpyAdapter:
    """A saved adapter (`RouteAdapter.to_npz`) evaluated in numpy: features (F,) -> bias (1, 32, 512) fp16 for the `intent_bias` input."""

    def __init__(self, path):
        z = np.load(path)
        self.enc = str(z["enc"])
        self.W = [(z[f"mlp_{i}_weight"].astype(np.float64), z[f"mlp_{i}_bias"].astype(np.float64)) for i in (0, 2)]
        self.Wo, self.bo = z["out_weight"].astype(np.float64), z["out_bias"].astype(np.float64)

    def bias(self, f):
        f = np.asarray(f, np.float64)
        h = f
        for W, b in self.W:
            h = _gelu(W @ h + b)
        return ((self.Wo @ h + self.bo) * f[0]).reshape(1, 32, 512).astype(np.float16)

    def bias_from_route(self, poly, pmask):
        return self.bias(features(self.enc, poly, pmask))
