"""SH30 (openpilot Cinque + op_parity adapter arm P2 + drivable hinge) as a causal 2 Hz planner: the NAVSIM serving path of
jevdrive.bench (pp_prep protocol W -> pp_train.PModel fp16 -> op_interp adapter `base`) for ONE decision, with nothing read from a
dataset. Simulator independent; experiments/alpasim/lib/sh30_driver.py wraps it as an AlpaSim EgodriverService.

Per decision at t0, from the CAM_F0 keyframes at t0 - 1.5 / 1.0 / 0.5 / 0 s (as many as exist) and the ego states at those times:
  frames   the 8 policy slots t0 - 1.4 .. t0 at 0.2 s: a keyframe where the slot is one (-1.0, 0), else the nearest keyframe re-projected
           along the ego track through the road plane (jevdrive.op_interp.synth_cpu `warp`); image pair of a slot = (slot - 0.2 s, slot),
           the pair of the oldest slot starts from a zero image (pp_prep: frames before the first slot are zero).
           synth = "cpu" (default, the reference every checkpoint was trained on) or "gpu": the same warp as one batched GPU call
           (op_interp.warp_gpu), keyframes sampled on the card too (pack_gpu); results/lat1_frame_synthesis.md
  tokens   Cinque's frozen vision encoder on the 8 pairs -> (8, 32, 512)
  policy   PModel: adapter bias from lib/parity_adapter.ego_features (command, vx, vy, ax, ay, 4 poses) + the trained plan pathway
  export   33 camera-frame plan points -> rear axle through the camera lever arm, linear resampling to 0.5 .. 4 s (8 poses)

Closed-loop yaw damping (decision 205, off by default): the synthesised slots are the only place the model sees the ego's turning in the
last 0.5 s (the newest image pair is one keyframe warped along the ego track), and its plan continues that yaw rate one to one, as the logs
do. Behind a tracker that executes the plan this is an undamped loop. `motion` < 1 shows the model that share of the turning only.

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
from concurrent.futures import ThreadPoolExecutor

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


_IDX = {}


def pack_gpu(jpeg, cam: dict, dev, decode=None):
    """pack with the model-frame sampling on the GPU -> uint8 tensor (2, 6, 128, 256) on dev, bit-identical to pack: the JPEG is decoded
    on the CPU as there (libjpeg's own YCbCr), OpenpilotMaps.__call__'s nearest sampling and 2x2 chroma mean run on the card.
    decode: another decoder, bytes -> (1080, 1920, 3) uint8 YCbCr tensor on dev (lat1_check.py nvjpeg; not bit-identical)."""
    import torch
    k = Z.calib_key({"CAM_F0": cam})
    m = _MAPS.get(k) or _MAPS.setdefault(k, Z.OpenpilotMaps(cam))
    if (k, dev) not in _IDX:
        _IDX[k, dev] = torch.from_numpy(np.stack(m.idx)).to(dev)
    src = io.BytesIO(jpeg) if isinstance(jpeg, (bytes, bytearray, memoryview)) else jpeg
    ycc = decode(jpeg) if decode else torch.from_numpy(np.array(m.decode(src))).to(dev)
    p = ycc.reshape(-1, 3)[_IDX[k, dev]].view(2, 256, 512, 3)
    Y, uv = p[..., 0], p[..., 1:].float().reshape(2, 128, 2, 256, 2, 2).mean((2, 4)).round().to(torch.uint8)
    return torch.stack([Y[:, 0::2, 0::2], Y[:, 1::2, 0::2], Y[:, 0::2, 1::2], Y[:, 1::2, 1::2], uv[..., 0], uv[..., 1]], 1)


def stack_keys(keys):
    """The m <= 4 newest keyframes (arrays from pack, or tensors from pack_gpu) -> (4, 2, 6, 128, 256) of the same kind, zeros first."""
    if isinstance(keys[0], np.ndarray):
        K = np.zeros((4,) + FRAME, np.uint8)
        K[4 - len(keys):] = np.stack(keys)
        return K
    import torch
    return torch.cat([keys[0].new_zeros((4 - len(keys),) + FRAME), torch.stack(list(keys))])


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


def damp_history(P, V, gain: float):
    """History (4, 3), (4, 2) with its turning scaled by `gain`: gain 1 = as driven, 0 = the same spacing on the x axis of t0 (yaw 0, no
    lateral velocity). Only the warp track of the synthesised slots takes this (Core.motion); the ego features keep the driven history."""
    if gain >= 1.0:
        return P, V
    d = np.r_[np.hypot(*np.diff(P[:, :2], axis=0).T), 0.0]
    Ps = np.c_[-d[::-1].cumsum()[::-1], np.zeros(4), np.zeros(4)]
    Vs = np.c_[np.copysign(np.hypot(V[:, 0], V[:, 1]), V[:, 0]), np.zeros(4)]
    return gain * P + (1 - gain) * Ps, gain * V + (1 - gain) * Vs


_POOL = None


def lattice(keys: np.ndarray, e: int, track, cam_t, cold: str):
    """keys (4, 2, 6, 128, 256) at op_interp.T_KEY, real from index e on -> the slot frames (8, 2, 6, 128, 256) and their validity.
    One thread per warped slot (27 ms each on one core, numpy + cv2 outside the GIL); the frames are those of a serial synth_cpu call."""
    global _POOL
    _POOL = _POOL or ThreadPoolExecutor(len(SLOT_T))
    real = SLOT_T >= I.T_KEY[e] - 1e-9
    valid = real | (cold == "backwarp")
    one = lambda j: (I.synth_cpu(keys, "warp", SLOT_T[j:j + 1], track, cam_t)[0] if real[j] else  # noqa: E731
                     I.warp_frame(keys[e], cam_t, track(SLOT_T[j]), track(I.T_KEY[e])))
    cur = np.zeros((len(SLOT_T),) + FRAME, np.uint8)
    idx = np.flatnonzero(valid)
    cur[idx] = list(_POOL.map(one, idx))
    return cur, valid


def lattice_gpu(keys, e: int, track, cam_t, cold: str, dev):
    """lattice with the slot warps as one batched GPU call (op_interp.warp_gpu) -> uint8 tensor (8, 2, 6, 128, 256) on dev, validity.
    keys: array or tensor. Same source keyframe and poses per slot as lattice; pixel agreement: results/lat1_frame_synthesis.md."""
    import torch
    real = SLOT_T >= I.T_KEY[e] - 1e-9
    valid = real | (cold == "backwarp")
    K = torch.as_tensor(keys, device=dev)
    cur = torch.zeros((len(SLOT_T),) + FRAME, dtype=torch.uint8, device=dev)
    slot, src = [], []
    for j in np.flatnonzero(valid):
        k = np.flatnonzero(np.isclose(I.T_KEY, SLOT_T[j]))
        if real[j] and len(k):
            cur[j] = K[k[0]]
            continue
        i0, i1, s = I.neighbours(SLOT_T[j]) if real[j] else (e, e, 0.0)
        slot.append(j), src.append(i0 if s <= 0.5 else i1)
    if slot:
        cur[slot] = I.warp_gpu(K[src], cam_t, np.stack([track(SLOT_T[j]) for j in slot]), np.stack([track(I.T_KEY[i]) for i in src]))
    return cur, valid


SYNTH = ("cpu", "gpu")


class Core:
    def __init__(self, tag: str = "SH30-F-s0", dev: str = "cuda", cold: str = "backwarp", motion: float = 1.0, synth: str = "cpu"):
        import cv2
        import torch
        import pp_train as T
        cv2.setNumThreads(1)                              # the slot warps are threaded here; cv2's own pool would oversubscribe
        from jevdrive import op_adapt as A
        assert cold in COLD and synth in SYNTH, (cold, synth)
        self.torch, self.A, self.tag, self.cold, self.motion, self.synth = torch, A, tag, cold, float(motion), synth
        self.dev = torch.device(dev)
        self.model = T.load_pmodel(tag, self.dev)
        assert self.model.arm == "P2" and self.model.adapter is not None, f"{tag}: expected an ego-only parity arm, got {self.model.arm}"
        s = self.model.net.slices["plan"].start
        self.pi = slice(s, s + 33 * 15)

    def _sync(self) -> float:
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)
        return time.perf_counter()

    def plan(self, keys, pose, vel, acc, cmd, cam_t, yaw_rate: float = 0.0, lht: bool = False, motion: float | None = None) -> dict:
        """One decision. keys: the m <= 4 newest keyframes (packed, 0.5 s apart, oldest first, the last at t0); pose (m, 3) x, y, yaw of
        the rear axle at those times in the t0 frame; vel (m, 2) body velocities; acc (2,) and cmd (4,) NAVSIM one-hot [L, S, R, unknown]
        at t0; cam_t the camera position in the ego frame; yaw_rate of the oldest state (cold start only); motion: the share of the
        history's turning shown in the synthesised slots (damp_history; default self.motion, 1 = all of it).
        -> poses (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s in the t0 frame, mu (33, 15), ego (20,), frames, stage times (ms)."""
        torch, A = self.torch, self.A
        t0 = time.perf_counter()
        m = len(keys)
        e = 4 - m
        P, V = fill_history(pose, vel, yaw_rate)
        K = stack_keys(keys)
        cam_t = np.asarray(cam_t, np.float64)
        track, gpu = I.track_navsim(*damp_history(P, V, self.motion if motion is None else motion)), self.synth == "gpu"
        cur, valid = lattice_gpu(K, e, track, cam_t, self.cold, self.dev) if gpu else lattice(K, e, track, cam_t, self.cold)
        prev = torch.cat([torch.zeros_like(cur[:1]), cur[:-1]]) if gpu else np.concatenate([np.zeros((1,) + FRAME, np.uint8), cur[:-1]])
        ego = PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32))
        t1 = self._sync() if gpu else time.perf_counter()
        with torch.no_grad():
            p, c = (x[np.flatnonzero(valid)] if gpu else torch.from_numpy(x[valid]).to(self.dev) for x in (prev, cur))
            H = self.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
            t2 = self._sync()
            tc = torch.tensor([[0.0, 1.0] if lht else [1.0, 0.0]], device=self.dev)
            out = self.model(H, torch.from_numpy(ego[None]).to(self.dev), tc).float()
            t3 = self._sync()
            mu = out[0, self.pi].reshape(33, 15).cpu().numpy()
            cur = cur.cpu().numpy() if gpu else cur
        poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
        t4 = time.perf_counter()
        return {"poses": poses, "mu": mu, "ego": ego, "hist": P, "cur": cur, "valid": valid, "tokens": H[0],
                "ms": {"frames": 1e3 * (t1 - t0), "encode": 1e3 * (t2 - t1), "policy": 1e3 * (t3 - t2), "export": 1e3 * (t4 - t3)}}
