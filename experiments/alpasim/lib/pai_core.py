"""The SH30 family (openpilot Cinque + op_parity adapter, sh30_core.py) fed as the AlpaSim PAI track delivers a decision: one camera
of any of the three calibrated models at 10 Hz, so the 8 policy slots (t0 - 1.4 .. t0 at 0.2 s) are frames the simulator rendered and
nothing is synthesised once 1.4 s of history exists. Serving-side only: the checkpoints are those trained for the nuPlan track.

  camera   `spec` (a plain dict built from the session's CameraSpec, pai_driver.camera_spec) -> `project`: f-theta (the PAI cameras),
           OpenCV pinhole (with rational terms) or OpenCV fisheye. The calibration describes the native sensor; pixels are scaled to the
           size of the JPEG actually delivered (e2e_challenge/README.md, "PAI Sensor Contract")
  frames   `Maps`: the openpilot road (focal 910) and wide (focal 455) 512x256 model frames sampled from that one camera along level,
           straight rig axes, nearest pixel, 2x2 chroma mean: navsim_zs.OpenpilotMaps with the projection replaced
  slots    `slots`: real frames where the session has one, else (cold start, or a dropped frame) the nearest real slot re-projected
           along the ego track (op_interp.warp_frame, the `backwarp` rule of sh30_core) or a zero state (`zero`)
  plan     `plan`: the encode / policy / export half of sh30_core.Core.plan, unchanged
"""
from __future__ import annotations

import io
import time

import numpy as np

import sh30_core as C
from sh30_core import I, PA, Z

T_SLOT_US = np.round(C.SLOT_T * 1e6).astype(np.int64)


def _poly(c, x):
    return sum(float(a) * x ** i for i, a in enumerate(c))


def project(rays: np.ndarray, spec: dict, wh) -> tuple[np.ndarray, np.ndarray]:
    """Rays (..., 3) in the camera frame (x right, y down, z forward) -> pixel (u, v) on the delivered image of size wh and the mask
    of rays the camera sees. spec: model ("ftheta" | "pinhole" | "fisheye"), c (2,), native (w, h), and the model's parameters."""
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    rho = np.hypot(x, y)
    th = np.arctan2(rho, z)
    ok = z > 1e-6 if spec["model"] == "pinhole" else np.ones(th.shape, bool)
    if spec["model"] == "ftheta":
        if spec.get("fw"):
            r = _poly(spec["fw"], th)
        else:                                               # only the backward polynomial: invert it on a table
            rr = np.linspace(0, 1.2 * np.hypot(*spec["native"]) / 2, 4096)
            r = np.interp(th, _poly(spec["bw"], rr), rr)
        px, py = r * x / np.maximum(rho, 1e-12), r * y / np.maximum(rho, 1e-12)
        c, d, e = spec.get("cde", (1.0, 0.0, 0.0))
        u, v = c * px + d * py, e * px + py
    elif spec["model"] == "fisheye":
        k = list(spec["k"]) + [0.0] * 4
        r = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
        u, v = spec["f"][0] * r * x / np.maximum(rho, 1e-12), spec["f"][1] * r * y / np.maximum(rho, 1e-12)
    else:
        k, (p1, p2) = list(spec["k"]) + [0.0] * 6, (list(spec["p"]) + [0.0] * 2)[:2]
        a, b = x / np.where(ok, z, 1.0), y / np.where(ok, z, 1.0)
        r2 = a * a + b * b
        rad = (1 + k[0] * r2 + k[1] * r2**2 + k[2] * r2**3) / (1 + k[3] * r2 + k[4] * r2**2 + k[5] * r2**3)
        u = spec["f"][0] * (a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a))
        v = spec["f"][1] * (b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b)
    if spec.get("max_angle", 0) > 0:
        ok = ok & (th <= spec["max_angle"])
    sx, sy = wh[0] / spec["native"][0], wh[1] / spec["native"][1]
    uv = np.stack([(u + spec["c"][0]) * sx, (v + spec["c"][1]) * sy], -1)
    return uv, ok & (uv[..., 0] >= 0) & (uv[..., 0] <= wh[0] - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= wh[1] - 1)


class Maps:
    """One camera (spec, R camera -> rig) and one delivered image size -> the gathers that sample the two openpilot model frames."""

    def __init__(self, spec: dict, R, wh):
        from jevdrive.openpilot.frames import MEDMODEL_K, MODEL_H, MODEL_W, SBIGMODEL_K, VIEW_FROM_DEVICE
        uu, vv = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
        idx, self.coverage, self.wh = [], [], tuple(wh)
        for Km in (MEDMODEL_K, SBIGMODEL_K):
            ray = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T * np.array([1.0, -1.0, -1.0])
            uv, ok = project(ray @ np.asarray(R, np.float64), spec, wh)      # rig axes (x fwd, y left, z up) -> camera frame
            xi = np.clip(np.rint(uv[..., 0]), 0, wh[0] - 1).astype(np.int64)
            yi = np.clip(np.rint(uv[..., 1]), 0, wh[1] - 1).astype(np.int64)
            idx.append(yi * wh[0] + xi)
            self.coverage.append(float(ok.mean()))
        self.idx = [3 * np.stack(idx) + c for c in range(3)]                 # (2, 256, 512) flat byte offsets of Y, Cb, Cr

    def __call__(self, ycc: np.ndarray) -> np.ndarray:
        """YCbCr image (h, w, 3) uint8 -> (2, 6, 128, 256) uint8 [road, wide] (sh30_core.pack_fast's integer chroma mean)."""
        f = ycc.reshape(-1)
        Y, U, V = (f.take(i) for i in self.idx)

        def half(c):
            c = c.astype(np.uint16)
            s = c[:, 0::2, 0::2] + c[:, 1::2, 0::2] + c[:, 0::2, 1::2] + c[:, 1::2, 1::2]
            return ((s + 1 + ((s >> 2) & 1)) >> 2).astype(np.uint8)
        return np.stack([Y[:, 0::2, 0::2], Y[:, 1::2, 0::2], Y[:, 0::2, 1::2], Y[:, 1::2, 1::2], half(U), half(V)], 1)


def decode(jpeg: bytes) -> np.ndarray:
    """JPEG bytes -> libjpeg's own YCbCr (h, w, 3) uint8, as navsim_zs.OpenpilotMaps.decode."""
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg))
    im.draft("YCbCr", im.size)
    return np.asarray(im.convert("YCbCr"))


def slots(frames: dict, t0: int, P, V, cam_t, cold: str, tol_us: int = 30_000):
    """frames {frame time us: packed frame}; history P (4, 3), V (4, 2) at op_interp.T_KEY -> the slot frames (8, 2, 6, 128, 256),
    their validity, and which are real. A slot without a frame within tol_us is the nearest real slot re-projected along the ego
    track (cold = backwarp) or invalid (zero)."""
    ts = np.array(sorted(frames), np.int64)
    near = [ts[np.abs(ts - (t0 + d)).argmin()] for d in T_SLOT_US]
    real = np.array([abs(int(n) - (t0 + int(d))) <= tol_us for n, d in zip(near, T_SLOT_US)])
    cur = np.zeros((len(T_SLOT_US),) + C.FRAME, np.uint8)
    for j in np.flatnonzero(real):
        cur[j] = frames[int(near[j])]
    valid = real.copy()
    if cold == "backwarp" and not real.all():
        track, ri = I.track_navsim(P, V), np.flatnonzero(real)
        for j in np.flatnonzero(~real):
            s = ri[np.abs(ri - j).argmin()]
            cur[j] = I.warp_frame(cur[s], np.asarray(cam_t, np.float64), track(C.SLOT_T[j]), track(C.SLOT_T[s]))
        valid[:] = True
    return cur, valid, real


def plan(core: C.Core, cur: np.ndarray, valid: np.ndarray, P, V, acc, cmd, cam_t, lht: bool = False) -> dict:
    """One decision from the slot frames: sh30_core.Core.plan after its frame synthesis. P (4, 3), V (4, 2) history at op_interp.T_KEY
    in the t0 frame, acc (2,), cmd (4,) one-hot [L, S, R, unknown] -> poses (8, 3) rear-axle x, y, yaw at 0.5 .. 4 s, stage times."""
    torch, A = core.torch, core.A
    t0 = time.perf_counter()
    prev = np.concatenate([np.zeros((1,) + C.FRAME, np.uint8), cur[:-1]])
    ego = PA.ego_features(P, V, np.tile(np.asarray(acc, np.float32), (4, 1)), np.asarray(cmd, np.float32))
    cam_t = np.asarray(cam_t, np.float64)
    with torch.no_grad():
        p, c = (torch.from_numpy(x[valid]).to(core.dev) for x in (prev, cur))
        H = core.model.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(1, int(valid.sum()), *A.H_SHAPE)
        t1 = core._sync()
        tc = torch.tensor([[0.0, 1.0] if lht else [1.0, 0.0]], device=core.dev)
        out = core.model(H, torch.from_numpy(ego[None]).to(core.dev), tc).float()
        mu, ld = out[0, core.pi].reshape(33, 15).cpu().numpy(), core.leads(out)
        t2 = time.perf_counter()
    poses = I.to_rear(mu[:, 0:3], mu[:, 11], I.T_IDXS, cam_t[:2], Z.T_OUT, "lever")
    return {"poses": poses, "mu": mu, "ego": ego, **ld,
            "ms": {"encode": 1e3 * (t1 - t0), "policy": 1e3 * (t2 - t1), "export": 1e3 * (time.perf_counter() - t2)}}
