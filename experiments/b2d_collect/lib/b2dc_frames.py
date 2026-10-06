"""openpilot model frames for the B2D collector: CARLA BGRA -> the 6 x 128 x 256 packed YUV420 planes Cinque reads, and their storage as
one H.264 stream per clip. NumPy only (runs inside the leaderboard agent, envs/simlingo, and in the trainer / checks).

Packing is the closed-loop harness's (scripts/zeroshot_policy_server.py OpenpilotModel.pack, the frames every B2D openpilot exam fed the
model): BT.601 limited-range Y at modeld's nearest-neighbour warp indices (jevdrive.openpilot.frames), chroma = the 2 x 2 BGRA block mean at
the half-resolution indices. tests/test_b2dc_frames.py checks bit equality against that method. Sensors are the harness's
(scripts/zeroshot_rigs.openpilot_sensor_specs: 1928 x 1208, road focal 2648, wide focal 567, level, rpy 0).

Storage: a packed pair (road, wide) is a 512 x 512 yuv420p picture (Y rows 0-255 road, 256-511 wide; U / V rows 0-127 road, 128-255
wide), i.e. exactly the model's YUV420 planes with no colour conversion; one picture per 20 Hz tick, libx264 (crf 0 = lossless).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))
from jevdrive.openpilot import frames as opf  # noqa: E402

CAM_WH = (1928, 1208)
FOCAL = {"road": 2648.0, "wide": 567.0}
MODEL_K = {"road": opf.MEDMODEL_K, "wide": opf.SBIGMODEL_K}
H2, W2 = opf.MODEL_H // 2, opf.MODEL_W // 2            # 128, 256
PIC_W, PIC_H = opf.MODEL_W, 2 * opf.MODEL_H             # 512 x 512 stacked picture
FFMPEG = os.environ.get("B2DC_FFMPEG", str(Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs")) / "envs/comma-wm/bin/ffmpeg"))


def indices(name: str, cam_wh=CAM_WH, focal=None):
    """(y_idx (H*W,), quad (4, H/2*W/2)) gather indices into the flat camera image, as OpenpilotModel.__init__ builds them."""
    w, h = cam_wh
    M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, focal or FOCAL[name]), name == "wide", model_K=MODEL_K[name])
    y = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
    uv = opf._nn_index(M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]], np.float32), (W2, H2), (w // 2, h // 2))
    r, c = np.divmod(uv, w // 2)
    quad = np.stack([(2 * r + i) * w + 2 * c + j for i in (0, 1) for j in (0, 1)])
    return y, quad


class Packer:
    """BGRA (H, W, 4) uint8 -> (6, 128, 256) uint8, per camera name."""

    def __init__(self, cam_wh=CAM_WH):
        self.idx = {n: indices(n, cam_wh) for n in FOCAL}

    def __call__(self, bgra: np.ndarray, name: str) -> np.ndarray:
        y_idx, quad = self.idx[name]
        px = bgra.reshape(-1, 4)
        b, g, r = (px[y_idx, k].astype(np.float32) for k in range(3))
        Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(opf.MODEL_H, opf.MODEL_W)
        q = px[quad].astype(np.float32).mean(0)
        b, g, r = q[:, 0], q[:, 1], q[:, 2]
        U = 128 - 0.148 * r - 0.291 * g + 0.439 * b
        V = 128 + 0.439 * r - 0.368 * g - 0.071 * b
        Y, U, V = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, U, V))
        out = np.empty((6, H2, W2), np.uint8)
        out[0], out[1], out[2], out[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
        out[4] = U.reshape(H2, W2)
        out[5] = V.reshape(H2, W2)
        return out


# ---------------------------------------------------------------- packed pair <-> yuv420p picture
def pair_to_yuv(img2: np.ndarray) -> bytes:
    """(2, 6, 128, 256) packed (road, wide) -> one 512 x 512 yuv420p picture (bytes)."""
    Y = np.empty((PIC_H, PIC_W), np.uint8)
    for k in range(2):
        p = img2[k]
        Yk = Y[k * opf.MODEL_H:(k + 1) * opf.MODEL_H]
        Yk[0::2, 0::2], Yk[1::2, 0::2], Yk[0::2, 1::2], Yk[1::2, 1::2] = p[:4]
    U = np.concatenate([img2[0, 4], img2[1, 4]], 0)
    V = np.concatenate([img2[0, 5], img2[1, 5]], 0)
    return Y.tobytes() + U.tobytes() + V.tobytes()


def yuv_to_pairs(buf: np.ndarray) -> np.ndarray:
    """(n, 512*512*3/2) uint8 yuv420p pictures -> (n, 2, 6, 128, 256) packed pairs."""
    n = len(buf)
    ny = PIC_H * PIC_W
    Y = buf[:, :ny].reshape(n, PIC_H, PIC_W)
    U = buf[:, ny:ny + ny // 4].reshape(n, PIC_H // 2, PIC_W // 2)
    V = buf[:, ny + ny // 4:].reshape(n, PIC_H // 2, PIC_W // 2)
    out = np.empty((n, 2, 6, H2, W2), np.uint8)
    for k in range(2):
        Yk = Y[:, k * opf.MODEL_H:(k + 1) * opf.MODEL_H]
        out[:, k, 0], out[:, k, 1], out[:, k, 2], out[:, k, 3] = Yk[:, 0::2, 0::2], Yk[:, 1::2, 0::2], Yk[:, 0::2, 1::2], Yk[:, 1::2, 1::2]
        out[:, k, 4], out[:, k, 5] = U[:, k * H2:(k + 1) * H2], V[:, k * H2:(k + 1) * H2]
    return out


# ---------------------------------------------------------------- H.264 streams
class Encoder:
    """Raw pictures piped into one ffmpeg (libx264). fmt 'yuv420p' (512 x 512 packed pairs) or 'bgra' (any size, e.g. the chase view)."""

    def __init__(self, path, w=PIC_W, h=PIC_H, fmt="yuv420p", crf=0, fps=20, preset="veryfast", gop=40, threads=2):
        self.path = Path(path)
        q = ["-qp", "0"] if crf == 0 else ["-crf", str(crf)]
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{w}x{h}", "-r", str(fps),
               "-i", "-", "-c:v", "libx264", "-preset", preset, *q, "-g", str(gop), "-threads", str(threads), "-pix_fmt", "yuv420p"]
        if fmt == "yuv420p":
            cmd += ["-color_range", "tv"]
        self.proc = subprocess.Popen(cmd + [str(self.path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, data) -> None:
        self.proc.stdin.write(data if isinstance(data, (bytes, bytearray)) else np.ascontiguousarray(data).tobytes())
        self.n += 1

    def close(self) -> int:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg {self.path}: rc {rc}: {self.proc.stderr.read().decode()[-500:]}")
        return self.n


def decode(path, w=PIC_W, h=PIC_H, fmt="yuv420p", frames=None) -> np.ndarray:
    """Whole stream -> (n, bytes per picture) uint8 ('yuv420p') or (n, h, w, 3) ('rgb24'). frames: optional index array to keep."""
    nb = w * h * 3 // 2 if fmt == "yuv420p" else w * h * 3
    raw = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, nb)
    if frames is not None:
        a = a[np.asarray(frames)]
    return a if fmt == "yuv420p" else a.reshape(-1, h, w, 3)


def read_pairs(path, frames=None) -> np.ndarray:
    """A clip's model-frame stream -> (n, 2, 6, 128, 256) packed (road, wide)."""
    return yuv_to_pairs(decode(path, frames=frames))


def horizon_rows() -> dict:
    """Model-frame row of the horizon for a level camera (rpy 0): the principal point row of each model frame."""
    return {n: float(MODEL_K[n][1, 2]) for n in MODEL_K}
