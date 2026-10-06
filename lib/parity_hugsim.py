"""HUGSIM side of the op_parity inputs (experiments/op_parity, lib/parity_adapter.py): what the HUGSIM openpilot agent
(experiments/hugsim/lib/zs_agent.py, opt `parity`) computes per simulator step and sends to the parity bias server
(experiments/op_parity/scripts/pp_hugsim.py serve), which returns the (32, 512) bias the served ONNX adds to all 9 context frames.
NumPy only (envs/hugsim).

Sources and conventions mirror WA-JEPA's HUGSIM client (third_party/wajepa close_loop/hugsim_planner.py):
  command   info["command"] (HUGSIM 0 right / 1 left / 2 straight) -> NAVSIM one-hot [left, straight, right, unknown] through
            their command map [2, 0, 1]
  speed     vx = info["ego_velo"], ax = info["accelerate"], vy = ay = 0 (HUGSIM reports no lateral components)
  history   4 poses at 2 Hz ending now, relative SE(2) to the current pose (x forward, y left, yaw left-positive); before the
            episode start the first state is repeated (their buffer clamps to the oldest frame)
  cameras   CAM_FRONT_LEFT / CAM_FRONT_RIGHT / CAM_BACK (= NAVSIM CAM_L0 / CAM_R0 / CAM_B0) as rendered; the black CAM_BACK of
            KITTI-360 / Waymo is fed as is, as they do
Where this differs from their client, it follows the NAVSIM training rows (op_parity pp_prep) or the openpilot harness:
  pose      rear axle (NAVSIM AgentInput) = the ego (front camera) pose moved back by jevdrive.hugsim_zs.rear_offset, as the
            Alpamayo history in this harness; WA-JEPA uses the ego box centre
  clock     "model" (default): the openpilot harness feeds one 0.25 s simulator step as one 0.2 s context step (dilate x1.25;
            speed x1.25 into the server, plan times x1.25 back out), so the ego inputs are put on that clock too: vx x1.25,
            ax x1.25^2, poses at model times -1.5 / -1.0 / -0.5 / 0 s = -1.875 / -1.25 / -0.625 / 0 s simulated (linear in
            position and heading between simulator steps), side-camera keys at the simulator steps 7 / 5 / 2 / 0 back (the
            nearest, ties to the newer). A model that keeps the speed it is told then keeps the simulated speed after the
            plan is stretched back. "sim": WA-JEPA's clock (stride 2 steps, raw speed and acceleration).
  side      each camera as an openpilot road + wide model-frame pair along its own mounting yaw (level, rotation only),
            packed like the front (jevdrive.hugsim_zs.OpenpilotFrames: BT.601 limited-range YUV); the NAVSIM version is
            jevdrive.navsim_zs.OpenpilotMaps(cam, yaw_deg=cam_yaw_deg(cam))
"""
from __future__ import annotations

import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "scripts")]
import parity_adapter as PA  # noqa: E402
import zeroshot_wire as wire  # noqa: E402
from jevdrive import hugsim_zs as Z  # noqa: E402

NAV_MAP = (2, 0, 1)                                                  # HUGSIM command -> NAVSIM index (WA-JEPA hugsim.command_map)
SIDE = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK")             # order of parity_adapter.SIDE_CAMS (L0, R0, B0)
KEYS_S = np.array([-1.5, -1.0, -0.5, 0.0])                           # history keys on the model clock (NAVSIM 2 Hz), oldest first
SIM_DT = 0.25


def mount_yaw_deg(calib: dict) -> float:
    """Mounting yaw of a camgeom calibration (deg, left positive): heading of its optical axis (extrinsic column 0) in M."""
    R = np.asarray(calib["extrinsic"], np.float64).reshape(4, 4)[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def key_steps(scale: float) -> np.ndarray:
    """Simulator steps back of the 4 history keys: floor(-tau * scale / 0.25), i.e. ties to the newer step."""
    return np.floor(-KEYS_S * scale / SIM_DT + 1e-6).astype(int)


def ego_inputs(hist: "Z.History", info: dict, d: float, scale: float):
    """parity_adapter.ego_features of the current step and the raw pose (4, 3) (rear axle, current frame, oldest first)."""
    t, pos, th = np.asarray(hist.t), np.asarray(hist.pos), np.asarray(hist.th)
    f, _ = Z.fwd_right(th)
    rear = pos - d * f
    tq = t[-1] + KEYS_S * scale                                     # np.interp clamps before t[0]: the first state repeated
    xy = np.stack([np.interp(tq, t, rear[:, k]) for k in (0, 1)], -1)
    hq = np.interp(tq, t, th)
    f0, r0 = Z.fwd_right(th[-1])
    rel = xy - xy[-1]
    pose = np.stack([rel @ f0, -(rel @ r0), -(hq - hq[-1])], -1)    # x forward, y left, yaw left-positive (theta is right-positive)
    v = float(np.ravel(info["ego_velo"])[0]) * scale
    a = float(np.ravel(info.get("accelerate", 0.0))[0]) * scale ** 2
    vel, acc = np.tile([v, 0.0], (4, 1)), np.tile([a, 0.0], (4, 1))
    cmd = np.zeros(4, np.float32)
    cmd[NAV_MAP[int(np.ravel(info["command"])[0])]] = 1.0
    return PA.ego_features(pose, vel, acc, cmd), pose


class ParityInputs:
    """Per scenario: side-camera key frames, ego features and the bias request. `cfg` = the agent opt `parity`:
    {"socket": bias server socket, "clock": "model" | "sim"}; which inputs are sent follows the server (its arm)."""

    def __init__(self, cfg: dict, cal: dict, d: float, dilation: float):
        self.cfg, self.d = dict(cfg), d
        self.clock = self.cfg.get("clock", "model")
        self.scale = float(dilation) if self.clock == "model" else 1.0
        self.back = key_steps(self.scale)
        self.sock = wire.connect_retry(self.cfg["socket"])
        wire.send(self.sock, {"cmd": "reset"}, {})
        self.server = wire.recv(self.sock)[0].get("server", {})
        self.use_side = bool(self.server.get("use_side"))
        self.frames = deque(maxlen=int(self.back.max()) + 1)
        self.yaw = {}
        if self.use_side:
            self.packers = {}
            for c in SIDE:
                self.yaw[c] = mount_yaw_deg(cal[c])
                fr = Z.OpenpilotFrames(cal, cams=(c,))
                self.packers[c] = (fr, fr.rot_index(self.yaw[c]))
        self.coverage = {c: self._coverage(c) for c in self.yaw}

    def _coverage(self, c):
        fr, idx = self.packers[c]
        return {k: float((idx[k] >= 0).mean()) for k in ("road", "wide")}

    def describe(self) -> dict:
        return {"server": self.server, "clock": self.clock, "scale": self.scale, "key_steps_back": self.back.tolist(),
                "side_cams": list(self.yaw), "mount_yaw_deg": {c: round(y, 2) for c, y in self.yaw.items()}, "coverage": self.coverage}

    def bias(self, rgb: dict, info: dict, hist) -> tuple[np.ndarray, dict]:
        """(32, 512) fp16 bias for this step and a log record."""
        t0 = time.perf_counter()
        ego, pose = ego_inputs(hist, info, self.d, self.scale)
        arrays = {"ego": ego}
        if self.use_side:
            self.frames.append(np.stack([fr.pack(rgb, idx) for fr, idx in (self.packers[c] for c in SIDE)]))   # (3, 2, 6, 128, 256)
            n = len(self.frames)
            arrays["side"] = np.stack([self.frames[max(0, n - 1 - b)] for b in self.back], 1)                # (3, 4, 2, 6, 128, 256)
        t1 = time.perf_counter()
        wire.send(self.sock, {"cmd": "bias"}, arrays)
        meta, out = wire.recv(self.sock)
        b = out["bias"]
        rec = {"ego": np.round(ego, 4).tolist(), "bias_rms": round(float(np.sqrt(np.mean(np.square(b.astype(np.float32))))), 5),
               "prep_ms": round(1e3 * (t1 - t0), 1), "rtt_ms": round(1e3 * (time.perf_counter() - t1), 1)}
        return b, rec
