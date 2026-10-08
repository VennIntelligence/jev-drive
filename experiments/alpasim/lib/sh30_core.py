"""SH30 (openpilot Cinque + op_parity adapter arm P2 + drivable hinge) as a causal 2 Hz planner: the NAVSIM serving path of
jevdrive.bench (pp_prep protocol W -> pp_train.PModel fp16 -> op_interp adapter `base`) for ONE decision, with nothing read from a
dataset. Simulator independent; experiments/alpasim/lib/sh30_driver.py wraps it as an AlpaSim EgodriverService.

Per decision at t0, from the CAM_F0 keyframes at t0 - 1.5 / 1.0 / 0.5 / 0 s (as many as exist) and the ego states at those times:
  frames   the 8 policy slots t0 - 1.4 .. t0 at 0.2 s: a keyframe where the slot is one (-1.0, 0), else the nearest keyframe re-projected
           along the ego track through the road plane (jevdrive.op_interp.synth_cpu `warp`); image pair of a slot = (slot - 0.2 s, slot),
           the pair of the oldest slot starts from a zero image (pp_prep: frames before the first slot are zero)
  tokens   Cinque's frozen vision encoder on the 8 pairs -> (8, 32, 512)
  policy   PModel: adapter bias from lib/parity_adapter.ego_features (command, vx, vy, ax, ay, 4 poses) + the trained plan pathway
  export   33 camera-frame plan points -> rear axle through the camera lever arm, linear resampling to 0.5 .. 4 s (8 poses)

Cold start (fewer than 4 keyframes; NAVSIM always has 4, so no rule was trained):
  poses    the missing history poses are the oldest known state run backwards at constant body velocity and yaw rate (both rules)
  backwarp (default) the slots older than the oldest keyframe are that keyframe re-projected to the back-extrapolated poses (op_interp's
           pre-roll warp): 8 valid slots whose image pairs carry the ego motion, as in training
  zero     those slots are zero hidden states marked invalid, the stream-start state of the policy (PModel.forward with n < 8 slots)
"""
from __future__ import annotations

import io
import pathlib
import sys
import time

import numpy as np

_R = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(p) for p in (_R, _R / "lib", _R / "scripts", _R / "experiments/op_adapt_r2/lib", _R / "experiments/op_parity/scripts")
                if str(p) not in sys.path]
import parity_adapter as PA  # noqa: E402
from jevdrive import navsim_zs as Z  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402

FRAME = (2, 6, 128, 256)
SLOT_T = np.round(np.arange(-7, 1) * 0.2, 3)          # the 8 valid policy slots (pp_prep.STEPS on op_lb's 20 Hz clock), oldest first
COLD = ("backwarp", "zero")


def camera(K, R, t, D=(0, 0, 0, 0, 0)) -> dict:
    """A navsim_zs camera dict: K (3, 3), R camera (x right, y down, z forward) -> ego rotation, t camera position in the ego frame,
    D Brown-Conrady (k1, k2, p1, p2, k3)."""
    return {"K": np.asarray(K, np.float32), "R": np.asarray(R, np.float32), "t": np.asarray(t, np.float32), "D": np.asarray(D, np.float32)}


_MAPS = {}


def pack(jpeg, cam: dict) -> np.ndarray:
    """A CAM_F0 JPEG (path or bytes) -> (2, 6, 128, 256) uint8 openpilot road + wide model frames (navsim_zs.OpenpilotMaps)."""
    k = Z.calib_key({"CAM_F0": cam})
    m = _MAPS.get(k) or _MAPS.setdefault(k, Z.OpenpilotMaps(cam))
    return m(m.decode(io.BytesIO(jpeg) if isinstance(jpeg, (bytes, bytearray, memoryview)) else jpeg))


def fill_history(pose, vel, yaw_rate: float):
    """pose (m, 3) x, y, yaw in the t0 frame and body velocities (m, 2) of the m <= 4 newest history keys (oldest first) -> (4, 3), (4, 2):
    the missing older keys are the oldest known state run backwards at constant body velocity and yaw rate."""
    pose, vel = np.asarray(pose, np.float64).reshape(-1, 3), np.asarray(vel, np.float64).reshape(-1, 2)
    e = 4 - len(pose)
    P, V = np.zeros((4, 3)), np.zeros((4, 2))
    P[e:], V[e:] = pose, vel
    w, (x0, y0, a0), v = float(yaw_rate), pose[0], vel[0]
    for j in range(e):
        dt = I.T_KEY[j] - I.T_KEY[e]
        s, c = (np.sin(w * dt) / w, (1 - np.cos(w * dt)) / w) if abs(w) > 1e-6 else (dt, 0.0)
        P[j] = np.r_[np.array([x0, y0]) + I.rot2(a0) @ np.array([s * v[0] - c * v[1], c * v[0] + s * v[1]]), a0 + w * dt]
        V[j] = v
    return P, V


def lattice(keys: np.ndarray, e: int, track, cam_t, cold: str):
    """keys (4, 2, 6, 128, 256) at op_interp.T_KEY, real from index e on -> the slot frames (8, 2, 6, 128, 256) and their validity."""
    valid = SLOT_T >= I.T_KEY[e] - 1e-9
    cur = np.zeros((len(SLOT_T),) + FRAME, np.uint8)
    cur[valid] = I.synth_cpu(keys, "warp", SLOT_T[valid], track, cam_t)
    if cold == "backwarp":
        for j in np.flatnonzero(~valid):
            cur[j] = I.warp_frame(keys[e], cam_t, track(SLOT_T[j]), track(I.T_KEY[e]))
        valid = np.ones_like(valid)
    return cur, valid


class Core:
    def __init__(self, tag: str = "SH30-F-s0", dev: str = "cuda", cold: str = "backwarp"):
        import torch
        import pp_train as T
        from jevdrive import op_adapt as A
        assert cold in COLD, cold
        self.torch, self.A, self.tag, self.cold = torch, A, tag, cold
        self.dev = torch.device(dev)
        self.model = T.load_pmodel(tag, self.dev)
        assert self.model.arm == "P2" and self.model.adapter is not None, f"{tag}: expected an ego-only parity arm, got {self.model.arm}"
        s = self.model.net.slices["plan"].start
        self.pi = slice(s, s + 33 * 15)

    def _sync(self) -> float:
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)
        return time.perf_counter()

    def plan(self, keys, pose, vel, acc, cmd, cam_t, yaw_rate: float = 0.0, lht: bool = False) -> dict:
        """One decision. keys: the m <= 4 newest keyframes (packed, 0.5 s apart, oldest first, the last at t0); pose (m, 3) x, y, yaw of
        the rear axle at those times in the t0 frame; vel (m, 2) body velocities; acc (2,) and cmd (4,) NAVSIM one-hot [L, S, R, unknown]
        at t0; cam_t the camera position in the ego frame; yaw_rate of the oldest state (cold start only).
        -> poses (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in the t0 frame, mu (33, 15), ego (20,), frames, stage times (ms)."""
        torch, A = self.torch, self.A
        t0 = time.perf_counter()
        m = len(keys)
        e = 4 - m
        P, V = fill_history(pose, vel, yaw_rate)
        K = np.zeros((4,) + FRAME, np.uint8)
        K[e:] = np.stack(keys)
        cam_t = np.asarray(cam_t, np.float64)
        cur, valid = lattice(K, e, I.track_navsim(P, V), cam_t, self.cold)
        prev = np.concatenate([np.zeros((1,) + FRAME, np.uint8), cur[:-1]])
        ego = PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32))
        t1 = time.perf_counter()
        with torch.no_grad():
            p, c = (torch.from_numpy(x[valid]).to(self.dev) for x in (prev, cur))
            H = self.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
            t2 = self._sync()
            tc = torch.tensor([[0.0, 1.0] if lht else [1.0, 0.0]], device=self.dev)
            out = self.model(H, torch.from_numpy(ego[None]).to(self.dev), tc).float()
            t3 = self._sync()
            mu = out[0, self.pi].reshape(33, 15).cpu().numpy()
        poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
        t4 = time.perf_counter()
        return {"poses": poses, "mu": mu, "ego": ego, "hist": P, "cur": cur, "valid": valid, "tokens": H[0],
                "ms": {"frames": 1e3 * (t1 - t0), "encode": 1e3 * (t2 - t1), "policy": 1e3 * (t3 - t2), "export": 1e3 * (t4 - t3)}}
