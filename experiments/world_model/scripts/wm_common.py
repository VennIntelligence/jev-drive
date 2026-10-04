"""Shared pieces of the WM-vs-reprojection check: paths, the 5 Hz grid, and the kinematic perturbations.

Conventions (physical, WOD): world / anchor frame x forward, y left, yaw counter-clockwise (left +). The world model was trained
on CARLA actions (omega = CARLA yaw rate, right turn positive; a = (v_{t+1} - v_t) / 0.2 s), so `w_actions` flips the sign.
A perturbation is a heading-offset profile delta_j (rad, left +) over the 5 Hz steps j = 1..K: the step j-1 -> j is driven at
heading psi_j + delta_j, so the same pose sequence feeds the reprojection arm (R) and, as (a, omega), the world-model arm (W).
"""
import os
from pathlib import Path

import numpy as np

DD = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
RUN = DD / "runs" / "wm_vs_reproj"
SF = DD / "runs" / "scale_check" / "wod_sf"            # sc_wod_prep.py: one pickle per WOD sceneflow segment (10 Hz, 3 front cameras, pose)
DT = 0.2                                                 # 5 Hz model context step
K = 10                                                   # future steps (the world model's horizon, 2 s)
HIST = 8                                                 # world-model history steps
PRE = 40                                                 # openpilot warm-up frames before the anchor (8 s, longer than its 5 s queue)
YAWS = (1.0, 2.0, -1.0, -2.0)                            # deg
LATS = (0.5, -0.5)                                       # m
BETA_MAX = np.radians(12.0)                              # a lateral offset that needs a larger swerve than this is not run (v < ~4 m/s)


def nominal(poses):
    """poses (K + 1, 3) world (x, y, yaw) -> anchor-frame poses and speeds (K,)."""
    p = np.asarray(poses, float)
    c, s = np.cos(p[0, 2]), np.sin(p[0, 2])
    d = p[:, :2] - p[0, :2]
    xy = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
    yaw = np.unwrap(p[:, 2]) - p[0, 2]
    return np.c_[xy, yaw], np.hypot(*np.diff(xy, axis=0).T) / DT


def perturb(P, delta):
    """P (K + 1, 3) anchor-frame poses, delta (K + 1,) heading offsets (delta[0] = 0) -> the perturbed pose sequence."""
    Q = P.copy()
    for j in range(1, len(P)):
        c, s = np.cos(delta[j]), np.sin(delta[j])
        d = P[j, :2] - P[j - 1, :2]
        Q[j, :2] = Q[j - 1, :2] + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        Q[j, 2] = P[j, 2] + delta[j]
    return Q


def yaw_profile(deg, k=K):
    d = np.full(k + 1, np.radians(deg))
    d[0] = 0.0
    return d


def lat_profile(off, speed, m=3, k=K):
    """Heading offset +beta for the first m steps then 0: the ego ends `off` m (left +) from the nominal path. None if beta > BETA_MAX."""
    path = float(np.sum(speed[:m]) * DT)
    sb = abs(off) / max(path, 1e-6)
    if sb > np.sin(BETA_MAX):
        return None
    d = np.zeros(k + 1)
    d[1: m + 1] = np.sign(off) * np.arcsin(sb)
    return d


def arms(P, speed):
    """{name: pose sequence (K + 1, 3)} of the nominal motion and the perturbations that are feasible at this speed."""
    out = {"nom": P}
    for y in YAWS:
        out[f"yaw{y:+g}"] = perturb(P, yaw_profile(y))
    for o in LATS:
        d = lat_profile(o, speed)
        if d is not None:
            out[f"lat{o:+g}"] = perturb(P, d)
    return out


def w_actions(Q, vf):
    """The world model's future actions for the pose sequence Q (K + 1, 3) with frame speeds vf (K + 1,): a_next[j] = (vf[j + 1] - vf[j]) / DT and
    omega_next[j] = -(yaw[j + 1] - yaw[j]) / DT (CARLA sign: right turn positive), as nq4_w defines them. Returns (K, 2) float32."""
    a = np.diff(vf) / DT
    w = -np.diff(np.unwrap(Q[:, 2])) / DT
    return np.stack([a, w], 1).astype(np.float32)
