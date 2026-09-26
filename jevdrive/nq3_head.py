"""The closed-loop bypass head of night queue 3 Q2 (todos/2026-09-26-night-queue-3.md, CL5 / CL5d), numpy only so any
agent env can load it. Written by jevdrive.nq3_q2.export into runs/nq3/q2/closed_loop_head/ (head.npz + manifest.json).

  h = Head(dir)
  traj, mode = h(temporal, ego)   # temporal (b, 512) openpilot `temporal`; ego (b, 100) = p5_exam.ego_input
  traj (b, 20, 2): future at t = 0.25 .. 5 s, rear-axle ego frame, x forward, y left (m); mode (b,) index into MODES
"""
import json
from pathlib import Path

import numpy as np

MODES = ("keep", "stop", "bypass_L", "bypass_R", "wait")


class Head:
    def __init__(self, d):
        d = Path(d)
        self.p = dict(np.load(d / "head.npz"))
        self.man = json.loads((d / "manifest.json").read_text())
        self.arm, self.mode_arm = self.man["trajectory_arm"], self.man["mode_arm"]

    @staticmethod
    def _lin(X, W):
        return X @ W[:-1] + W[-1]

    def __call__(self, temporal, ego):
        p = self.p
        temporal = np.atleast_2d(np.asarray(temporal, np.float32))
        ego = np.atleast_2d(np.asarray(ego, np.float32))
        xe = (ego - p["ego_mu"]) / p["ego_sd"]
        xi = (temporal - p["op_mu"]) / p["op_sd"]
        P = (self._lin(xe, p["We"]) + self._lin(xi, p["Wp"])).reshape(-1, 20, 2)
        z = (temporal - p["z_mu"]) / p["z_sd"] / np.sqrt(float(self.man["scalars"]["z_dim"]))
        logit = self._lin(np.concatenate([xe, xi], 1), p["W2"]) if "W2" in p else None
        if logit is not None and "A3_W" in p:
            logit3 = logit + (z - p["A3_zbar"]) @ p["A3_W"]
        mode = None
        if self.mode_arm == "A3":
            mode = logit3.argmax(1)
        elif self.mode_arm == "A2":
            mode = logit.argmax(1)
        out = P.copy()
        if self.arm == "A1":
            out[..., 1] += (z - p["A1_zbar"]) @ p["A1_W"]
        elif self.arm == "A2":
            out[..., 1] += p["templates"][logit.argmax(1)]
        elif self.arm == "A3":
            out[..., 1] += p["templates"][logit3.argmax(1)]
        return out, mode
