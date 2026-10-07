"""Differentiable torch proxy of the NAVSIM devkit's plan replay (navsim PDMSimulator = BatchLQRTracker + BatchKinematicBicycleModel), the
step pdm_score runs before DAC and every other sub-score.

Devkit computation reproduced step by step (navsim/planning/simulation/planner/pdm_planner/simulation/*.py):
  1. [initial state, 8 plan poses at 0.5 .. 4 s] -> 41 states at 0.1 s, linear in x, y and unwrapped heading (InterpolatedTrajectory).
  2. Velocity / curvature profiles (40 each) from the 41 poses by the two regularised least squares fits (jerk 1e-4, curvature rate 1e-2,
     initial curvature 1e-10).
  3. 40 closed-loop steps: stopping P controller (gain 0.5) when v and the reference speed 10 steps ahead are <= 0.2 m/s, else the one-step
     longitudinal LQR (q 10, r 1, horizon 10 x 0.1 s: a = -10/11 (v - v_ref)) and the one-step lateral LQR on [lateral error, heading error,
     steering angle] (Q diag 1, 10, 0; R 1) over the 10-step linearised dynamics with the speed held at v + a k dt; then the bicycle model:
     first-order lag on acceleration (0.2 s) and steering (0.05 s), Euler step, Pacifica wheel base 3.089 m, steering clip pi / 3.
Initial state in the t0 rear-axle frame: (0, 0, 0), v0 / a0 = rear-axle longitudinal velocity / acceleration of the ego status, steering 0
(navsim's ego_status_to_ego_state). Every gain is a constant, so this is the devkit computation itself; the 10-step lateral products are
written as cumulative sums (same values). Gradients flow through the least squares (torch.linalg.solve) and the closed-loop rollout.
"""
import math

import numpy as np
import torch

WHEEL_BASE, DT, HORIZON = 3.089, 0.1, 10
Q_LON, R_LON, Q_LAT, R_LAT = 10.0, 1.0, (1.0, 10.0, 0.0), 1.0
JERK, CURV_RATE, CURV0 = 1e-4, 1e-2, 1e-10
STOP_GAIN, STOP_V = 0.5, 0.2
TAU_A, TAU_D, MAX_STEER = 0.2, 0.05, math.pi / 3
T_POSE, T_DENSE = np.arange(1, 9) * 0.5, np.arange(0, 41) * 0.1


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def interp_matrix() -> np.ndarray:
    """(41, 9): linear interpolation of [origin, 8 poses] onto the 0.1 s grid."""
    t = np.r_[0, T_POSE]
    M = np.zeros((41, 9))
    for k, tt in enumerate(T_DENSE):
        j = min(np.searchsorted(t, tt, side="right") - 1, 7)
        f = (tt - t[j]) / (t[j + 1] - t[j])
        M[k, j], M[k, j + 1] = 1 - f, f
    return M


_CACHE = {}


def _consts(dev, dt):
    k = (dev, dt)
    if k not in _CACHE:
        n = 40
        tri = torch.tril(torch.ones(n, n, dtype=dt, device=dev))                       # (row i, col j) j <= i
        R = torch.zeros(n - 2, n, dtype=dt, device=dev)                               # jerk: differences of the 39 accelerations
        idx = torch.arange(n - 2, device=dev)
        R[idx, idx + 1], R[idx, idx + 2] = -1.0, 1.0
        Qc = CURV_RATE * torch.eye(n, dtype=dt, device=dev)
        Qc[0, 0] = CURV0
        _CACHE[k] = dict(M=torch.as_tensor(interp_matrix(), dtype=dt, device=dev), tri=tri, RtR=JERK * R.T @ R, Qc=Qc,
                         kk=torch.arange(HORIZON, dtype=dt, device=dev))
    return _CACHE[k]


def proposal(P8):
    """(B, 8, 3) plan poses -> (B, 41, 3) devkit proposal states (x, y, unwrapped heading) incl. the origin."""
    C = _consts(P8.device, P8.dtype)
    P0 = torch.cat([torch.zeros_like(P8[:, :1]), P8], 1)
    h = P0[..., 2]
    h = torch.cat([h[:, :1], h[:, :1] + torch.cumsum(wrap(h[:, 1:] - h[:, :-1]), 1)], 1)
    P0 = torch.stack([P0[..., 0], P0[..., 1], h], -1)
    return torch.einsum("kj,bjc->bkc", C["M"], P0)


def profiles(S):
    """(B, 41, 3) proposal states -> velocity (B, 40), curvature (B, 40) as get_velocity_curvature_profiles_with_derivatives_from_poses."""
    C = _consts(S.device, S.dtype)
    n = 40
    d = S[:, 1:] - S[:, :-1]
    dxy, dh = d[..., :2], wrap(d[..., 2])
    hd = S[:, :-1, 2]
    col = torch.stack([torch.cos(hd), torch.sin(hd)], -1).reshape(len(S), 2 * n)                # (B, 80)
    tri2 = C["tri"].repeat_interleave(2, 0)                                                     # (80, 40)
    A = col[..., None] * DT ** 2 * tri2
    A = torch.cat([col[..., None] * DT, A[..., 1:]], -1)
    At = A.transpose(1, 2)
    x = torch.linalg.solve(At @ A + C["RtR"], (At @ dxy.reshape(len(S), 2 * n)[..., None]))[..., 0]
    v = x[:, :1] + torch.cat([torch.zeros_like(x[:, :1]), torch.cumsum(x[:, 1:] * DT, 1)], 1)    # (B, 40)
    Ac = C["tri"] * (v[..., None] * DT ** 2)                                                    # row i scaled by v_i dt^2
    Ac = torch.cat([v[..., None] * DT, Ac[..., 1:]], -1)
    At = Ac.transpose(1, 2)
    xc = torch.linalg.solve(At @ Ac + C["Qc"], (At @ dh[..., None]))[..., 0]
    kap = xc[:, :1] + torch.cat([torch.zeros_like(xc[:, :1]), torch.cumsum(xc[:, 1:] * DT, 1)], 1)
    return v, kap


def replay(P8, v0, a0, steer0=None):
    """(B, 8, 3) plans, (B,) initial speed / acceleration [, steering] -> (B, 41, 3) replayed rear-axle states (x, y, heading), t0 frame."""
    B = len(P8)
    C = _consts(P8.device, P8.dtype)
    S = proposal(P8)
    vp, kp = profiles(S)
    x = torch.zeros(B, dtype=P8.dtype, device=P8.device)
    y, h = torch.zeros_like(x), torch.zeros_like(x)
    v, a = v0.to(P8.dtype), a0.to(P8.dtype)
    dl = torch.zeros_like(x) if steer0 is None else steer0.to(P8.dtype)
    out = [torch.stack([x, y, h], -1)]
    kk = C["kk"]
    q0, q1, q2 = Q_LAT
    for i in range(40):
        xr, yr, hr = S[:, i, 0], S[:, i, 1], S[:, i, 2]
        e0 = -(x - xr) * torch.sin(hr) + (y - yr) * torch.cos(hr)
        p0 = wrap(h - hr)
        ri = min(i + HORIZON, 39)
        v_ref = vp[:, ri]
        kap = kp[:, i:ri]
        if kap.shape[1] < HORIZON:
            kap = torch.cat([kap, kp[:, ri:ri + 1].expand(B, HORIZON - kap.shape[1])], 1)
        stop = (v_ref <= STOP_V) & (v <= STOP_V)
        acc_lqr = -(HORIZON * DT * Q_LON) / ((HORIZON * DT) ** 2 * Q_LON + R_LON) * (v - v_ref)
        acc_cmd = torch.where(stop, -STOP_GAIN * (v - v_ref), acc_lqr)
        vk = v[:, None] + acc_lqr[:, None] * DT * kk[None]                                     # (B, 10) linearisation speeds
        # zero-input error after 10 steps: psi_k = p0 + dt sum_{j<k} v_j (dl / L - kappa_j); e = e0 + dt sum_k v_k psi_k
        dpsi = DT * vk * (dl[:, None] / WHEEL_BASE - kap)
        psi = p0[:, None] + torch.cat([torch.zeros_like(dpsi[:, :1]), torch.cumsum(dpsi[:, :-1], 1)], 1)
        ze = e0 + DT * (vk * psi).sum(1)
        zp = wrap(psi[:, -1] + dpsi[:, -1])
        zd = wrap(dl)
        # input response: b_delta_k = k dt, b_psi_k = sum_{j<k} v_j dt / L * j dt, b_e = sum_k v_k dt b_psi_k
        bps = vk * DT / WHEEL_BASE * (kk[None] * DT)
        bpsi = torch.cat([torch.zeros_like(bps[:, :1]), torch.cumsum(bps[:, :-1], 1)], 1)
        be = (vk * DT * bpsi).sum(1)
        bp = bpsi[:, -1] + bps[:, -1]
        bd = HORIZON * DT
        u = -(q0 * be * ze + q1 * bp * zp + q2 * bd * zd) / (q0 * be ** 2 + q1 * bp ** 2 + q2 * bd ** 2 + R_LAT)
        u = torch.where(stop, torch.zeros_like(u), u)
        # bicycle model with first-order command lag
        a_n = DT / (DT + TAU_A) * (acc_cmd - a) + a
        d_ideal = DT * u + dl
        d_up = DT / (DT + TAU_D) * (d_ideal - dl) + dl
        rate = (d_up - dl) / DT
        x = x + v * torch.cos(h) * DT
        y = y + v * torch.sin(h) * DT
        h = wrap(h + v * torch.tan(dl) / WHEEL_BASE * DT)
        v = v + a_n * DT
        dl = torch.clamp(dl + rate * DT, -MAX_STEER, MAX_STEER)
        a = a_n
        out.append(torch.stack([x, y, h], -1))
    return torch.stack(out, 1)
