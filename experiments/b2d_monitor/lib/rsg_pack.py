"""CARLA openpilot road / wide renders -> the (2, 6, 128, 256) uint8 [road, wide] model frames the SH30 family was trained on
(navsim_zs.OpenpilotMaps: BT.601 full-range YCbCr, nearest model pixel, 2x2 chroma mean of the model-resolution samples rounded half to
even; no JPEG round trip). numpy only (Python 3.8 agent and 3.11 server)."""
import math

import numpy as np

MODEL_W, MODEL_H = 512, 256
CAM_WH = (1928, 1208)
FOCAL = {"road": 2648.0, "wide": 567.0}
MODEL_K = {"road": (910.0, MODEL_W / 2, 47.6), "wide": (455.0, MODEL_W / 2, 0.5 * (256 + 47.6))}


def _index(name):
    """Flat source pixel of every model pixel (level, straight calibration: the model ray (u - cx, v - cy, F) seen by the pinhole render),
    rounded as jevdrive.openpilot.frames._nn_index."""
    F, cx, cy = MODEL_K[name]
    w, h = CAM_WH
    f = FOCAL[name]
    u, v = np.meshgrid(np.arange(MODEL_W, dtype=np.float32), np.arange(MODEL_H, dtype=np.float32))
    xi = np.clip(np.rint(np.float32(f / F) * (u - np.float32(cx)) + np.float32(w / 2)), 0, w - 1).astype(np.int64)
    yi = np.clip(np.rint(np.float32(f / F) * (v - np.float32(cy)) + np.float32(h / 2)), 0, h - 1).astype(np.int64)
    return (yi * w + xi).ravel()


class Packer:
    def __init__(self):
        self.idx = {k: _index(k) for k in ("road", "wide")}

    def one(self, bgra, name):
        px = np.asarray(bgra).reshape(-1, 4)[self.idx[name]].astype(np.float32)
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        Y = np.clip(np.rint(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).reshape(MODEL_H, MODEL_W)
        U = np.clip(np.rint(128 - 0.168736 * r - 0.331264 * g + 0.5 * b), 0, 255).reshape(MODEL_H, MODEL_W)
        V = np.clip(np.rint(128 + 0.5 * r - 0.418688 * g - 0.081312 * b), 0, 255).reshape(MODEL_H, MODEL_W)
        half = lambda c: np.rint((c[0::2, 0::2] + c[1::2, 0::2] + c[0::2, 1::2] + c[1::2, 1::2]) / 4.0)  # noqa: E731
        Y = Y.astype(np.uint8)
        return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], half(U).astype(np.uint8), half(V).astype(np.uint8)])

    def __call__(self, road, wide):
        return np.stack([self.one(road, "road"), self.one(wide, "wide")])
