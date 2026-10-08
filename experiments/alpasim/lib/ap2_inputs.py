"""The AlpaSim nuPlan-track input standard for AP2 (openpilot Cinque + ego adapter trained for what AlpaSim hands a driver), numpy only:
shared by the route rebuild (scripts/ap2_route.py), the training-row builder (scripts/ap2_train.py), the offline read and the driver.

A rollout is 10 decisions at 2 Hz with no history before t = 0 (docs/alpasim.md), so decision k has m = min(k + 1, 4) keyframes:
  slots    the policy slots at or after the oldest keyframe are real (N_SLOT[m] of the 8 at 0.2 s), the older ones are zero and invalid; the
           image pair of the oldest real slot starts from a zero image (the rule pp_prep already uses for the oldest slot at m = 4)
  poses    the 4 - m missing history poses are the oldest state run backwards at constant body velocity and yaw rate (sh30_core.fill_history)
  ego      what DynamicState holds at that decision (measured on 48 tapped scenes, results/ap2_smoke.md):
             k = 0   the recorded nuPlan velocity / acceleration (= NAVSIM's), delivered rotated by -yaw (the driver rotates it back)
             k = 1   vy = 0, ax = d speed / dt of the track, ay = vx * yaw rate (the controller's state coerced from the recording)
             k >= 2  vy = 0, ax = d vx / dt of the vehicle model, ay = 0
           Training rows take ax from the recording at every k: on logs it is the closest available estimate of the track's d speed / dt
           (0.11 m/s^2 from the central difference of the recorded speeds; a spline or chord through the 2 Hz poses is 0.14 / 0.17 off)
  command  the shipped samples' rule on the route AlpaSim sends (`route_cmd`), or the route itself (`route_feat`, arm R)
"""
from __future__ import annotations

import numpy as np

T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])
N_SLOT = {1: 1, 2: 3, 3: 6, 4: 8}                 # real policy slots (of the 8 at t0 - 1.4 .. t0, 0.2 s apart) with m keyframes
COLD_AT = {1: slice(0, 1), 2: slice(1, 4), 3: slice(4, 10)}       # the real slots of m keyframes in scripts/ap2_prep.py's cold.npy (N, 10, 32, 512)
MIX = (0.1, 0.1, 0.1, 0.7)                        # share of decisions with m = 1, 2, 3, 4 keyframes in a 10-decision rollout
N_WP, ROUTE_DIM = 20, 60                          # route waypoints per message; route_feat width
T12 = np.r_[T_KEY, 0.5 * np.arange(1, 9)]         # the 4 history + 8 future poses of a NAVSIM token


def route_cmd(wp, lat: float = 2.0, look: float = 5.0) -> np.ndarray:
    """wp (..., n, 2) route waypoints in the rig frame, NaN = padding -> (..., 4) one-hot [left, straight, right, unknown]: the shipped
    samples' navsim_transfuser_challenge.navigation.command_from_route (first waypoint at least `look` m away; y > lat left, < -lat right;
    none that far: straight; no waypoint message at all, n = 0: unknown)."""
    wp = np.asarray(wp, np.float64)
    out = np.zeros(wp.shape[:-2] + (4,), np.float32)
    if wp.shape[-2] == 0:
        out[..., 3] = 1
        return out
    far = np.hypot(wp[..., 0], wp[..., 1]) >= look            # NaN compares False, as in the sample
    i = far.argmax(-1)
    y = np.take_along_axis(wp[..., 1], i[..., None], -1)[..., 0]
    c = np.where(far.any(-1), np.where(y > lat, 0, np.where(y < -lat, 2, 1)), 1)
    np.put_along_axis(out, c[..., None], 1.0, -1)
    return out


def route_feat(wp) -> np.ndarray:
    """wp (..., 20, 2) rig-frame waypoints with NaN padding -> (..., 60): per waypoint [valid, (x - 60) / 20, y / 20], zeros when padded."""
    wp = np.asarray(wp, np.float32)
    ok = ~np.isnan(wp[..., 0])
    f = np.stack([ok.astype(np.float32), np.where(ok, (wp[..., 0] - 60.0) / 20.0, 0.0), np.where(ok, np.clip(wp[..., 1] / 20.0, -3, 3), 0.0)], -1)
    return f.reshape(wp.shape[:-2] + (ROUTE_DIM,)).astype(np.float32)


def yaw_rates(pose, fut) -> np.ndarray:
    """pose (N, 4, 3) history and fut (N, 8, 3) future rear-axle poses (x, y, yaw in the t0 frame, 0.5 s apart) -> the track's yaw rate at
    each history key (N, 4): a cubic spline through the 12 unwrapped yaws (AlpaSim reports the recording's yaw rate at k = 0 / 1 to
    0.003 / 0.001 rad/s of the log's central difference). Rows without a logged future (NaN) use the history alone."""
    from scipy.interpolate import CubicSpline
    pose, fut = np.asarray(pose, np.float64), np.asarray(fut, np.float64)
    has = ~np.isnan(fut).any((1, 2))
    psi = np.unwrap(np.concatenate([pose[..., 2], np.nan_to_num(fut[..., 2])], 1), axis=1)
    return np.where(has[:, None], CubicSpline(T12, psi, axis=1)(T_KEY, 1), np.gradient(psi[:, :4], 0.5, axis=1)).astype(np.float32)


def sim_state(vel, acc, w0, m: int):
    """The (vx, vy), (ax, ay) AlpaSim reports at a decision with m keyframes, from a logged token: vel / acc (N, 2) recorded body-frame
    state at t0, w0 (N,) the track's yaw rate at t0 (module docstring). ax stays the recorded one for every m."""
    vel, acc = np.asarray(vel, np.float32).copy(), np.asarray(acc, np.float32).copy()
    if m == 1:
        return vel, acc
    vel[:, 1] = 0.0
    acc[:, 1] = vel[:, 0] * w0 if m == 2 else 0.0
    return vel, acc


def fill_history(pose, vel, w, m: int):
    """Batched sh30_core.fill_history: pose (N, 4, 3), vel (N, 4, 2), w (N, 4) yaw rate per key -> the same arrays with the 4 - m oldest
    keys replaced by the state of key 4 - m run backwards at constant body velocity and yaw rate."""
    P, V = np.asarray(pose, np.float64).copy(), np.asarray(vel, np.float64).copy()
    e = 4 - m
    x0, y0, a0, v, wr = P[:, e, 0], P[:, e, 1], P[:, e, 2], V[:, e], np.asarray(w, np.float64)[:, e]
    for j in range(e):
        dt = T_KEY[j] - T_KEY[e]
        small = np.abs(wr) <= 1e-6
        ws = np.where(small, 1.0, wr)
        s, c = np.where(small, dt, np.sin(ws * dt) / ws), np.where(small, 0.0, (1 - np.cos(ws * dt)) / ws)
        bx, by = s * v[:, 0] - c * v[:, 1], c * v[:, 0] + s * v[:, 1]
        P[:, j] = np.stack([x0 + np.cos(a0) * bx - np.sin(a0) * by, y0 + np.sin(a0) * bx + np.cos(a0) * by, a0 + wr * dt], 1)
        V[:, j] = v
    return P, V
