"""COL1 shared helpers (analysis side, plain numpy): load col1_pai_extract.py pickles and per-scene geometry.
`actors` / `logged` are AABB-centre poses (t_us, x, y, yaw) in the rollout local frame; driver plans, routes and ego observations are
rig (rear-axle) poses. Simulator state is used for labels only."""
import pickle
import sys

import numpy as np

def load(f):
    """A pickle written under numpy 2, also under numpy 1 (the module alias exists only during the load: shapely checks for it)."""
    if hasattr(np, "_core"):
        return pickle.load(open(f, "rb"))
    import numpy.core as _c
    names = ("numpy._core", "numpy._core.multiarray", "numpy._core.numeric")
    for n in names:
        sys.modules[n] = getattr(_c, n.split(".")[-1], _c)
    try:
        return pickle.load(open(f, "rb"))
    finally:
        for n in names:
            sys.modules.pop(n, None)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def interp(tr, t):
    """tr (n, 4) t_us, x, y, yaw -> (m, 3) at times t (clamped)."""
    t = np.clip(np.atleast_1d(t).astype(float), tr[0, 0], tr[-1, 0])
    return np.c_[np.interp(t, tr[:, 0], tr[:, 1]), np.interp(t, tr[:, 0], tr[:, 2]), np.interp(t, tr[:, 0], np.unwrap(tr[:, 3]))]


def speed(tr, t, dt=0.15e6):
    t = np.atleast_1d(t).astype(float)
    a, b = interp(tr, t - dt), interp(tr, t + dt)
    t0, t1 = np.clip(t - dt, tr[0, 0], tr[-1, 0]), np.clip(t + dt, tr[0, 0], tr[-1, 0])
    return np.hypot(*(b[:, :2] - a[:, :2]).T) / np.maximum((t1 - t0) * 1e-6, 1e-3)


def center_off(o):
    return float(o["start"]["rollout_spec"]["vehicle"]["rigToBoundingBox"]["vec"].get("x", 0.0)) if "start" in o else 1.461


def to_rig(c, off):
    """AABB-centre poses (n, >= 3: x, y, yaw in the last three columns) -> rig poses (n, 3)."""
    c = np.atleast_2d(c)
    return np.c_[c[:, -3] - off * np.cos(c[:, -1]), c[:, -2] - off * np.sin(c[:, -1]), c[:, -1]]


def into(frame, xy):
    """World points (n, 2) -> the frame of pose (x, y, yaw)."""
    return (np.atleast_2d(xy) - frame[:2]) @ rot(frame[2])


def corners(x, y, h, L, W):
    return np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]]) @ rot(h).T + [x, y]


def metric(o, name):
    m = o["metrics"].get(name)
    return None if m is None else m[np.argsort(m[:, 0])]


def first_event(o, flag):
    """First scored sim time (us) at which a zero flag fires, None if never."""
    if flag == "collision_at_fault":
        a, b = metric(o, "collision_front"), metric(o, "collision_lateral")
        t, v = a[:, 0], np.maximum(a[:, 1], b[:, 1])
    else:
        m = metric(o, flag)
        t, v = m[:, 0], m[:, 1]
    ok = (v > 0) & (metric(o, "eval_relevant")[:, 1] > 0)
    return int(t[ok][0]) if ok.any() else None


def lat_to_path(path_xy, p):
    """Signed lateral offset (left positive) of points p (n, 2) from a polyline, and the arc position of the foot -> (n, 2)."""
    path_xy, p = np.asarray(path_xy, float), np.atleast_2d(p)
    a, b = path_xy[:-1], path_xy[1:]
    d = b - a
    L = np.maximum(np.hypot(*d.T), 1e-9)
    s0 = np.r_[0, np.cumsum(L)[:-1]]
    out = []
    for q in p:
        t = np.clip(((q - a) * d).sum(1) / L**2, 0, 1)
        f = a + t[:, None] * d
        i = np.hypot(*(q - f).T).argmin()
        n = np.array([-d[i, 1], d[i, 0]]) / L[i]
        out.append([(q - f[i]) @ n, s0[i] + t[i] * L[i]])
    return np.array(out)
