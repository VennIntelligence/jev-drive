"""The efference memory of the STACK reflex module (lib/reflex.py; plans/2026-10-16-stack-prereg.md 4.5): one rule for the served drivers
(experiments/alpasim/lib/sh30_driver.py, pai_driver.py) and the offline self-state rows (experiments/route_follow/scripts/rf_stack_self.py).

Per session the driver stores every served decision: t0 (us), the plan's frame (the rear-axle anchor x, y, yaw in the rollout's local
frame) and the 8 poses it actually served in that frame (after every serve-side transform). At a decision at t0 with the rear-axle pose
`pose` (local frame) and the measured speed v:
  previous  the newest stored decision with t0_prev <= t0 - LAG + TOL. LAG = 0.5 s, the 2 Hz decision period of the training rows: on the
            nuPlan track (2 Hz) this is decision k - 1 (k - 2 when k - 1 was aborted), on the PAI track (10 Hz) the decision 0.5 s back, so
            the served dt matches the rows' (JEV_REFLEX_LAG overrides, seconds)
  rel       `pose` in the previous decision's frame; dt = (t0 - t0_prev) / 1e6 -> reflex.selfstate(served_prev, rel, dt, v)
  none      no stored decision old enough: reflex.empty() (valid 0)
"""
from __future__ import annotations

import collections
import os

import numpy as np

import reflex as RX

LAG_US = int(round(float(os.environ.get("JEV_REFLEX_LAG", "0.5")) * 1e6))
TOL_US = 50_000
KEEP = 32                                                  # decisions kept per session (3.2 s at 10 Hz; selfstate refuses dt > 2.5 s)


def rel_pose(frame, pose) -> np.ndarray:
    """pose (x, y, yaw) given in the local frame -> the same pose in the frame of `frame` (x, y, yaw, local), yaw wrapped."""
    fx, fy, fa = (float(q) for q in frame)
    px, py, pa = (float(q) for q in pose)
    c, s = np.cos(fa), np.sin(fa)
    dx, dy = px - fx, py - fy
    return np.array([c * dx + s * dy, -s * dx + c * dy, (pa - fa + np.pi) % (2 * np.pi) - np.pi])


class Memory:
    def __init__(self):
        self.d = collections.deque(maxlen=KEEP)

    def add(self, t0_us: int, frame, served):
        """A served decision: t0, its plan frame (x, y, yaw local), the served poses (8, 3) in that frame."""
        self.d.append((int(t0_us), np.asarray(frame, np.float64).reshape(3).copy(), np.array(served, np.float64).reshape(8, 3)))

    def prev(self, t0_us: int):
        return next((e for e in reversed(self.d) if e[0] < t0_us and e[0] <= t0_us - LAG_US + TOL_US), None)

    def features(self, t0_us: int, pose, v: float):
        """-> (selfstate (SF,) float32, info dict: dt s and the previous decision's t0, or {} without one)."""
        e = self.prev(int(t0_us))
        if e is None:
            return RX.empty(), {}
        dt = (int(t0_us) - e[0]) * 1e-6
        return RX.selfstate(e[2], rel_pose(e[1], pose), dt, float(v)), dict(dt=round(dt, 6), t_prev=e[0])
