"""JEV_PERTURB: a by-action perturbation of the policy's own rollout for sh30_driver.py (the self-rewind frontier read,
experiments/route_follow/plans/2026-10-16-rewind-self-prereg.md). Off by default (unset or empty: the driver is bit-identical to without it).

JEV_PERTURB is a file of `<scene id> <k> <g> <s> <n>` lines (`#` comments). For the decisions k .. k + n - 1 of that scene (the session's
drive call count from 0, the record's `k`; 2 Hz on the nuPlan track) the served plan is branch_poses(plan, g, s): every pose yaw and segment
course about the ego multiplied by g at the same segment lengths (curvature gain), then the path re-sampled at s times its arc length (arc
scale: s < 1 = slower). Before k and after k + n the model's own plan is served. A scene not in the file is never perturbed. The model runs
and is logged as always; the record's `poses` is the served (transformed) plan, `poses_model` the model's, `perturb` {g, s, k0, n}.
No state injection, no restart: the simulator is untouched, the perturbation is an action the controller tracks.

branch_poses is experiments/onpol_pai/lib/onp_log_driver.branch_poses (P3's transform), copied because importing that driver has side
effects; selftest() checks the copy against its source.
"""
from __future__ import annotations

import os

import numpy as np


def _table(path: str) -> dict:
    tab = {}
    if path:
        for line in open(path):
            f = line.split("#")[0].split()
            if len(f) >= 5:
                tab[f[0]] = (int(f[1]), float(f[2]), float(f[3]), int(f[4]))
    return tab


FILE = os.environ.get("JEV_PERTURB", "") or ""
ON = bool(FILE)
TAB = _table(FILE)
SUFFIX = "-pert" if ON else ""


def active(scene_id: str, k: int):
    """-> (g, s, k0, n) when decision k of this scene is perturbed, else None."""
    q = TAB.get(scene_id) if ON else None
    if q is None or not (q[0] <= k < q[0] + q[3]):
        return None
    return q[1], q[2], q[0], q[3]


def branch_poses(poses, g: float, s: float):
    """(8, 3) x, y, yaw in the ego frame (ego at the origin, yaw 0) -> the transformed plan. Curvature gain g: every pose's yaw and every
    segment's course (from the previous pose, the origin first) are multiplied by g, the segment lengths kept, the path re-integrated.
    Speed scale s: the (transformed) polyline origin -> pose 8 re-sampled at s times each pose's arc length (same path, shorter), yaw
    interpolated along the arc. g = s = 1 returns the input untouched."""
    q = np.asarray(poses, np.float64)
    if g == 1.0 and s == 1.0:
        return q
    xy = np.r_[[[0.0, 0.0]], q[:, :2]]
    d = np.diff(xy, axis=0)
    ds = np.hypot(*d.T)
    c = g * np.unwrap(np.r_[0.0, np.arctan2(d[:, 1], d[:, 0])])[1:]
    yaw = g * np.unwrap(np.r_[0.0, q[:, 2]])
    xy = np.r_[[[0.0, 0.0]], np.cumsum(ds[:, None] * np.c_[np.cos(c), np.sin(c)], 0)]
    S = np.r_[0.0, np.cumsum(ds)]
    if s != 1.0:
        u = np.maximum.accumulate(S + 1e-9 * np.arange(len(S)))           # strictly increasing for interp (stopped plans)
        w = s * u[1:]
        return np.c_[np.interp(w, u, xy[:, 0]), np.interp(w, u, xy[:, 1]), np.interp(w, u, yaw)]
    return np.c_[xy[1:], yaw[1:]]


def selftest():
    import ast
    import tempfile
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "onpol_pai" / "lib" / "onp_log_driver.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "branch_poses")
    ns = {"np": np}
    exec(compile(ast.Module([fn], []), "onp_branch", "exec"), ns)
    rng = np.random.default_rng(0)
    for _ in range(200):
        arc = np.cumsum(rng.uniform(0, 4, 8))
        P = np.c_[arc * np.cos(np.linspace(0, rng.uniform(-1, 1), 8)), arc * np.sin(np.linspace(0, rng.uniform(-1, 1), 8)), rng.normal(0, .3, 8)]
        for g, s in ((0.7, 0.9), (1.3, 1.1), (1.0, 0.9), (0.85, 1.0)):
            assert np.array_equal(branch_poses(P, g, s), ns["branch_poses"](P, g, s))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# scene k g s n\nsceneA 5 0.7 1.1 2\n")
    t = _table(f.name)
    assert t == {"sceneA": (5, 0.7, 1.1, 2)}
    os.unlink(f.name)
    print("perturb selftest ok")


if __name__ == "__main__":
    selftest()
