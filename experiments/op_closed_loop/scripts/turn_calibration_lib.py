"""Shared geometry for the turn-calibration analysis (experiments/op_closed_loop/results/turn_calibration.md).

Conventions: curvature is right-positive (CARLA x forward-ish / y right, standard signed curvature in the CARLA frame), like the logged `act_k`.
"""
import numpy as np


def resample(xy, step=0.5, sigma=1.5):
    """Dense route polyline -> uniform arc-length grid, Gaussian-smoothed (sigma metres). Returns (P (n,2), s (n,))."""
    xy = np.asarray(xy, float)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    g = np.arange(0.0, s[-1] + 1e-9, step)
    P = np.stack([np.interp(g, s, xy[:, k]) for k in range(2)], -1)
    if sigma > 0 and len(g) > 8:
        h = int(3 * sigma / step)
        w = np.exp(-0.5 * (np.arange(-h, h + 1) * step / sigma) ** 2)
        w /= w.sum()
        pad = np.pad(P, ((h, h), (0, 0)), mode="edge")
        # extend the ends linearly so the smoothing does not pull them inward
        P = np.stack([np.convolve(pad[:, k], w, mode="valid") for k in range(2)], -1)
    return P, g


def heading(P):
    d = np.gradient(P, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def curvature(P, step=0.5, win=3.0):
    """Signed curvature along the grid (right-positive in the CARLA frame): d heading / d s over a +-win/2 m window."""
    psi = heading(P)
    k = max(int(round(win / step / 2)), 1)
    out = np.zeros(len(P))
    for i in range(len(P)):
        a, b = max(i - k, 0), min(i + k, len(P) - 1)
        out[i] = (psi[b] - psi[a]) / max((b - a) * step, 1e-6)
    return out


def chord_kappa(P, i0, a, step=0.5):
    """Chord curvature 2*y/(x^2+y^2) of the route point a metres ahead of grid index i0, in the route frame at i0 (right-positive)."""
    i1 = min(i0 + int(round(a / step)), len(P) - 1)
    if i1 <= i0:
        return 0.0
    psi = np.arctan2(*(P[min(i0 + 1, len(P) - 1)] - P[max(i0 - 1, 0)])[::-1])
    d = P[i1] - P[i0]
    c, s_ = np.cos(psi), np.sin(psi)
    x, y = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]      # y > 0 = to the right in the CARLA frame
    return 2 * y / max(x * x + y * y, 1e-6)


def maneuvers(P, step=0.5, k_th=0.02, gap=8.0, min_deg=8.0):
    """Turn segments of the route: |kappa| > k_th (R < 50 m) merged over gaps < gap m. Returns a list of dicts
    (i0, i1, angle_deg signed right-positive, rmin_m)."""
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


TYPES = ["straight", "bend 8-25", "turn 25-60", "turn 60-120", "tight >120"]


def turn_type(angle_abs):
    if angle_abs is None or np.isnan(angle_abs):
        return 0
    return 1 if angle_abs < 25 else 2 if angle_abs < 60 else 3 if angle_abs < 120 else 4


def leaderboard_downsample(cmd, sample_factor=50.0, xy=None):
    """Indices kept by the leaderboard's downsample_route (command changes, lane changes, > sample_factor m, the ends)."""
    CHL, CHR = 5, 6   # RoadOption values: CHANGELANELEFT 5, CHANGELANERIGHT 6
    ids, prev, dist = [], None, 0.0
    for i in range(len(cmd)):
        cur = int(cmd[i])
        if prev is None:
            ids.append(i); dist = 0.0
        elif cur in (CHL, CHR):
            ids.append(i); dist = 0.0
        elif prev != cur and prev not in (CHL, CHR):
            ids.append(i); dist = 0.0
        elif dist > sample_factor:
            ids.append(i); dist = 0.0
        elif i == len(cmd) - 1:
            ids.append(i); dist = 0.0
        else:
            dist += float(np.linalg.norm(xy[i] - xy[i - 1]))
        prev = cur
    return ids


def boot_ci(vals_by_cluster, stat, n=2000, seed=0):
    """Cluster bootstrap. vals_by_cluster: list of (arrays tuple) per cluster; stat(list of tuples) -> float."""
    rng = np.random.default_rng(seed)
    m = len(vals_by_cluster)
    if m < 2:
        return float("nan"), float("nan")
    out = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        out.append(stat([vals_by_cluster[i] for i in idx]))
    out = np.array(out)
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))
