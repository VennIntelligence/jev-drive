"""Route polyline: the navigation-level input "which way to go" as an ego-frame polyline (experiments/op_route_cmd).

Shared by the open-loop fine-tune material (experiments/op_route_cmd) and the closed-loop turn calibration
(experiments/op_closed_loop: option C "10 m road polyline + 1 m noise"), so both use one definition of navigation noise.

Frames: ego frame = rear axle, x forward, y left (heading and curvature are counter-clockwise positive, so a left turn is
positive). The CARLA world frame has y to the right; there the same functions give right-positive angles (unchanged
behaviour of the closed-loop scripts, which pass world coordinates).

Clean label (`hindsight`): the path the ego actually drove, resampled at DS = 10 m up to HORIZON = 150 m -> (K = 16, 2) vertices
(vertex 0 = the ego origin), `pmask` marks the vertices that exist (the logged future may end earlier), plus turn statistics.
Noise (`noise_polyline`) is applied at train time to a clean polyline; `NavNoise` holds the parameters. `C_OPTION` is the closed-loop option C
(lateral AR(1) offset only), `noisy_road` the original function of turn_calibration_sparse.py (bit-identical, kept as the reference).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K, DS, HORIZON = 16, 10.0, 150.0
TURN_MIN_DEG = 25.0


# ---------------------------------------------------------------- geometry (moved from experiments/op_closed_loop/scripts)
def poly_resample(xy, step=0.25):
    """Polyline -> uniform arc-length grid (linear interpolation). Returns (P (n, 2), s (n,))."""
    xy = np.asarray(xy, float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    g = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(g, s, xy[:, k]) for k in range(2)], -1), g


def heading(P):
    d = np.gradient(P, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def curvature(P, step=0.5, win=3.0):
    """Signed curvature along the grid: d heading / d s over a +-win/2 m window (counter-clockwise positive)."""
    psi = heading(P)
    k = max(int(round(win / step / 2)), 1)
    out = np.zeros(len(P))
    for i in range(len(P)):
        a, b = max(i - k, 0), min(i + k, len(P) - 1)
        out[i] = (psi[b] - psi[a]) / max((b - a) * step, 1e-6)
    return out


def maneuvers(P, step=0.5, k_th=0.02, gap=8.0, min_deg=8.0):
    """Turn segments of a uniform-grid path: |kappa| > k_th (R < 50 m) merged over gaps < gap m. Returns a list of dicts
    (i0, i1, angle deg signed counter-clockwise positive, rmin m)."""
    k = curvature(P, step)
    on = np.abs(k) > k_th
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if segs and (i - segs[-1][1]) * step < gap:
                segs[-1][1] = j
            else:
                segs.append([i, j])
            i = j + 1
        else:
            i += 1
    psi = heading(P)
    out = []
    for i0, i1 in segs:
        a0, a1 = max(i0 - 2, 0), min(i1 + 2, len(P) - 1)
        ang = np.degrees(psi[a1] - psi[a0])
        if abs(ang) >= min_deg:
            out.append(dict(i0=i0, i1=i1, angle=float(ang), rmin=float(1.0 / max(np.abs(k[i0:i1 + 1]).max(), 1e-6))))
    return out


# ---------------------------------------------------------------- hindsight label
def smooth_path(xy, step=0.5, sigma=1.5):
    """Logged poses (m, 2), first row = ego origin -> uniform grid (n, 2), Gaussian-smoothed, ends extended linearly so the
    smoothing does not pull them inward; None if the path is shorter than 1 m (stationary)."""
    xy = np.asarray(xy, float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]]
    if len(xy) < 2 or np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() < 1.0:
        return None
    P, g = poly_resample(xy, step)
    if len(P) < 3:
        return None
    h = int(3 * sigma / step)
    c = max(int(2.0 / step), 1)

    def tangent(a, b):
        d = P[b] - P[a]
        return d / max(np.linalg.norm(d), 1e-9)

    t0, t1 = tangent(0, min(c, len(P) - 1)), tangent(max(len(P) - 1 - c, 0), len(P) - 1)
    ext0 = P[0] - step * np.arange(h, 0, -1)[:, None] * t0
    ext1 = P[-1] + step * np.arange(1, h + 1)[:, None] * t1
    Q = np.vstack([ext0, P, ext1])
    w = np.exp(-0.5 * (np.arange(-h, h + 1) * step / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(Q, ((h, h), (0, 0)), mode="edge")
    S = np.stack([np.convolve(pad[:, k], w, mode="valid") for k in range(2)], -1)
    return S[h:h + len(P)]


def hindsight(path_xy, step=0.5, sigma=1.5, want_path=False) -> dict:
    """Clean route label from the logged future path (poses in the ego frame at sample time, first row (0, 0), any time spacing).

    poly (K, 2) f32 vertices at s = 0, 10 .. 150 m along the driven path, pmask (K,) bool (vertex exists), plen (path length within the
    horizon, m), and the turn statistics of the smoothed path: `turn_deg` / `turn_s` / `turn_end_s` / `turn_rmin` of the first complete
    turn >= 25 deg that starts more than 2 m ahead (nan if none), `n_turn` such turns within the horizon, `in_turn` = a >= 8 deg
    maneuver is active at s = 0, `max_turn_deg` = largest |net heading change| of any complete maneuver >= 8 deg ahead.
    want_path adds `path`: the smoothed path (n, 2) on the 0.5 m grid up to the horizon (None when stationary).
    """
    out = dict(poly=np.zeros((K, 2), np.float32), pmask=np.zeros(K, bool), plen=0.0, turn_deg=np.nan, turn_s=np.nan, turn_end_s=np.nan,
               turn_rmin=np.nan, n_turn=0, in_turn=False, max_turn_deg=0.0)
    if want_path:
        out["path"] = None
    S = smooth_path(path_xy, step, sigma)
    if S is None:
        out["pmask"][0] = True
        return out
    S[0] = 0.0                                                       # vertex 0 is exactly the ego origin
    S, g = poly_resample(S, step)
    plen = min(g[-1], HORIZON)
    n = int(plen // DS) + 1
    sv = DS * np.arange(n)
    out["poly"][:n] = np.stack([np.interp(sv, g, S[:, k]) for k in range(2)], -1)
    out["pmask"][:n] = True
    out["plen"] = float(plen)
    P = S[: int(plen / step) + 1]
    if want_path:
        out["path"] = P
    if len(P) >= 8:
        ms = maneuvers(P, step)
        # a maneuver that reaches the end of the available path is incomplete (angle underestimated): not counted
        full = [m for m in ms if m["i1"] * step < plen - 3.0]
        out["in_turn"] = any(m["i0"] * step <= 2.0 for m in ms)
        ahead = [m for m in full if m["i0"] * step > 2.0]
        if ahead:
            out["max_turn_deg"] = float(max(abs(m["angle"]) for m in ahead))
        big = [m for m in ahead if abs(m["angle"]) >= TURN_MIN_DEG]
        out["n_turn"] = len(big)
        if big:
            m = big[0]
            out.update(turn_deg=m["angle"], turn_s=m["i0"] * step, turn_end_s=m["i1"] * step, turn_rmin=m["rmin"])
    return out


TURN_BINS = (25, 45, 75, 105, 135, 181)       # deg: slight, 45-75, ~90, sharp, u-turn
TURN_BIN_NAMES = ("25-45", "45-75", "75-105", "105-135", ">=135")


def turn_bin(deg):
    """Index into TURN_BIN_NAMES of |deg| (-1 if < 25 or nan); array in, array out."""
    a = np.abs(np.asarray(deg, float))
    return np.where(np.isnan(a) | (a < TURN_BINS[0]), -1, np.digitize(a, TURN_BINS[1:-1]))


# ---------------------------------------------------------------- navigation noise
@dataclass(frozen=True)
class NavNoise:
    """Navigation-quality noise on a route polyline.

    sigma   sd (m) of the lateral offset of each vertex, AR(1) along the route with correlation length `corr` (m): smooth, not white
    decim   vertex spacing (m) of the noised polyline (road-level map resolution)
    along   sd (m) of the along-track jitter of the interior vertices (0 = off)
    p_drop  probability that an interior vertex is missing (the neighbours are joined by a straight segment; 0 = off)
    """
    sigma: float = 1.0
    corr: float = 30.0
    decim: float = 10.0
    along: float = 1.5
    p_drop: float = 0.05


C_OPTION = NavNoise(along=0.0, p_drop=0.0)     # closed-loop option C of turn_calibration_options.py: 10 m road polyline + 1 m lateral noise


def _vertices(P, g, cfg: NavNoise, rng):
    """Decimated vertices of a dense path P (grid g, 0.25 m) with the noise of `cfg`. The draws of the lateral part come first and are the same
    as the original noisy_road, so cfg = C_OPTION reproduces it bit-for-bit."""
    step = g[1] - g[0]
    keep = np.arange(0, len(P), int(cfg.decim / step))
    keep = np.r_[keep, len(P) - 1] if keep[-1] != len(P) - 1 else keep
    Q, s = P[keep], g[keep]
    a = np.exp(-cfg.decim / cfg.corr)
    e = np.zeros(len(Q))
    e[0] = rng.normal()
    for i in range(1, len(Q)):
        e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * rng.normal()
    if cfg.along > 0 and len(Q) > 2:                               # jitter the interior vertices along the path (order kept)
        s2 = s.copy()
        s2[1:-1] = np.sort(np.clip(s[1:-1] + cfg.along * rng.normal(size=len(Q) - 2), 0.0, g[-1]))
        Q = np.stack([np.interp(s2, g, P[:, k]) for k in range(2)], -1)
    t = np.gradient(Q, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], -1)
    Q = Q + cfg.sigma * e[:, None] * nrm
    if cfg.p_drop > 0 and len(Q) > 2:
        drop = np.r_[False, rng.random(len(Q) - 2) < cfg.p_drop, False]
        Q = Q[~drop]
    return Q


def noisy_road(xy, sigma, rng, decim=10.0, corr=30.0):
    """Closed-loop option C (was turn_calibration_sparse.noisy_road): dense route (m, 2) -> decimated vertices with correlated lateral noise."""
    P, g = poly_resample(xy, 0.25)
    return _vertices(P, g, NavNoise(sigma=sigma, corr=corr, decim=decim, along=0.0, p_drop=0.0), rng)


def noise_polyline(poly, pmask, rng, cfg: NavNoise = NavNoise(), k_out: int = K):
    """Noised copy of one clean polyline (K, 2) / pmask (K,) -> (k_out, 2) f32, mask (k_out,) bool; vertices first, rest padded with zeros.
    The polyline stays anchored at the ego origin (vertex 0 moves only laterally, like option C)."""
    poly, pmask = np.asarray(poly, float), np.asarray(pmask, bool)
    out, om = np.zeros((k_out, 2), np.float32), np.zeros(k_out, bool)
    v = poly[: int(pmask.sum())]
    if len(v) < 2:
        if len(v):
            out[0], om[0] = v[0], True
        return out, om
    P, g = poly_resample(v, 0.25)
    if len(P) < 3:
        out[:len(v)], om[:len(v)] = v[:k_out], True
        return out, om
    Q = _vertices(P, g, cfg, rng)[:k_out]
    out[:len(Q)], om[:len(Q)] = Q, True
    return out, om


def noise_batch(poly, pmask, rng, cfg: NavNoise = NavNoise(), k_out: int = K):
    """(B, K, 2) / (B, K) -> noised (B, k_out, 2) / (B, k_out)."""
    r = [noise_polyline(p, m, rng, cfg, k_out) for p, m in zip(poly, pmask)]
    return np.stack([a for a, _ in r]), np.stack([m for _, m in r])


# ---------------------------------------------------------------- per-sample sidecar (the extra field of a sample table)
FIELDS = ("poly", "pmask", "plen", "turn_deg", "turn_s", "turn_end_s", "turn_rmin", "n_turn", "in_turn", "max_turn_deg",
          "jct_dist", "n_exit", "taken_cls", "cmd")


def attach(route_npz, ids) -> dict:
    """Route fields for the rows of a sample table (`tab["id"]` of op_adapt_H samples, or any id list): arrays aligned to `ids`;
    rows without a route get pmask all False, nan floats, n_exit -1. `route_npz` = $DATA_DIR/processed/op_route_cmd/<source>/route.npz."""
    with np.load(route_npz, allow_pickle=False) as z:
        rid = z["id"]
        pos = {k: i for i, k in enumerate(rid.tolist())}
        idx = np.array([pos.get(str(i), -1) for i in ids])
        ok = idx >= 0
        out = {}
        for f in z.files:
            if f == "id":
                continue
            a = z[f]
            res = np.zeros((len(idx),) + a.shape[1:], a.dtype)
            if a.dtype.kind == "f":
                res[:] = np.nan
            elif a.dtype.kind == "i":
                res[:] = -1
            elif a.dtype.kind in "US":
                res[:] = ""
            if f == "poly":
                res[:] = 0.0
            res[ok] = a[idx[ok]]
            out[f] = res
        out["has_route"] = ok
    return out
