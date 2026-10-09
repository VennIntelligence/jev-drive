"""COL1 diagnostic arm of the PAI driver (pai_driver.py, unchanged): the plan's path is kept and re-timed so that the returned trajectory
starts at the ego's current speed and joins the plan's own speed profile after PAI_VCONT seconds (default 1.0).

Why: the served trajectory is a linear resampling of the plan's 0.5 s points, so its first segment has the plan's mean speed over the
first 0.5 s whatever the ego's speed is. On the PAI scenes that mean speed is 3-11 % below the ego's above 16 m/s and 5-28 % above it at
5-12 m/s (results/collisions.md); the position-tracking nonlinear MPC answers the step at 24 and 35 m/s with saturated braking and
steering (two spins at the hand-over). Serving-side only, no training; not a submission default.
Run as pai_driver.py is run (scripts/pai_run.sh with DRV_PY=col1_pai_driver.py).
"""
import os

import numpy as np

import pai_driver as D

TAU = float(os.environ.get("PAI_VCONT", "1.0"))


def retime(poses: np.ndarray, v0: float, tau: float = TAU, dt: float = 0.05) -> np.ndarray:
    """Plan poses (8, 3) x, y, yaw at 0.5 .. 4 s -> the same path sampled where a speed profile that starts at v0 and blends linearly
    into the plan's segment speeds over tau seconds has arrived at 0.5 .. 4 s. Beyond the plan's end the path continues straight."""
    p = np.concatenate([np.zeros((1, 3)), np.asarray(poses, np.float64)])
    ds = np.hypot(*np.diff(p[:, :2], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-6))])
    t = np.arange(dt / 2, 4.0, dt)
    vp = (ds / 0.5)[np.minimum((t / 0.5).astype(int), 7)]
    v = v0 + (vp - v0) * np.minimum(t / max(tau, 1e-3), 1.0)
    sn = np.cumsum(np.maximum(v, 0.0) * dt)[int(round(0.5 / dt)) - 1::int(round(0.5 / dt))]
    far = p[-1] + np.array([60.0 * np.cos(p[-1, 2]), 60.0 * np.sin(p[-1, 2]), 0.0])
    P, S = np.concatenate([p, far[None]]), np.concatenate([s, [s[-1] + 60.0]])
    return np.stack([np.interp(sn, S, P[:, 0]), np.interp(sn, S, P[:, 1]), np.interp(sn, S, np.unwrap(P[:, 2]))], 1)


_plan = D.PC.plan


def plan(core, cur, valid, P, V, acc, cmd, cam_t, lht=False):
    o = _plan(core, cur, valid, P, V, acc, cmd, cam_t, lht)
    o["poses_model"] = o["poses"]
    o["poses"] = retime(o["poses"], float(np.hypot(*np.asarray(V)[-1])))
    return o


D.PC.plan = plan

if __name__ == "__main__":
    D.main()
