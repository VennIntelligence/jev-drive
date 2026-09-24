"""NAVSIM zero-shot exam of open driving models (todos/2026-09-24-zeroshot-exam/navsim.md).

  index       per-token model inputs (4 history frames @2 Hz: 8-camera paths + calibration, ego poses, velocity,
              acceleration, driving command), written by scripts/navsim_zs_index.py in the devkit venv
  cameras     nuPlan pinhole + Brown distortion; rotation-only reprojection into Alpamayo's f-theta cameras
              (576x320 model images, rig from scripts/zeroshot_rigs.py, shared with the other exams) and into
              openpilot's narrow / wide model frames (nearest sampling of CAM_F0, as openpilot's own warp)
  converters  NAVSIM 2 Hz ego history -> Alpamayo 10 Hz egomotion; model outputs -> NAVSIM 8 poses @2 Hz (x, y, yaw)

Imported by four venvs (project, navsim devkit, alpamayo, openpilot): numpy/scipy only at import time, py3.10+.
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

from .common import data_dir

T_OUT = np.arange(1, 9) * 0.5                        # NAVSIM output: 0.5 ... 4.0 s, 8 poses
T_HIST2 = np.array([-1.5, -1.0, -0.5, 0.0])          # NAVSIM history: 4 frames @2 Hz
T_HIST10 = (np.arange(16) - 15) * 0.1                # Alpamayo egomotion history: -1.5 ... 0 s
ALP_OUT_IDX = np.round(T_OUT / 0.1).astype(int) - 1  # Alpamayo outputs are at 0.1 ... 6.4 s -> exact 2 Hz samples
# NAVSIM driving_command one-hot [left, straight, right, unknown]; same templates as the WOD-E2E exam
NAV_TEXT = {0: "Turn left", 1: "Continue straight", 2: "Turn right"}
NUPLAN_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
NUPLAN_WH = (1920, 1080)
SRC_SCALE = 4        # sources for the 120-degree views are decoded at 1/4 (DCT scaling): 386 px/rad vs 278 needed


def root(*parts) -> Path:
    d = data_dir() / "runs" / "navsim_zs" / Path(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_index(split: str) -> list:
    with open(root("index") / f"{split}.pkl", "rb") as f:
        return pickle.load(f)


def rigs():
    """scripts/zeroshot_rigs.py: the Alpamayo camera rig shared by all three zero-shot exams."""
    spec = importlib.util.spec_from_file_location("zeroshot_rigs", Path(__file__).resolve().parents[1] / "scripts/zeroshot_rigs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- nuPlan cameras

def cams_of(cam_dict: dict, sensor_root) -> dict:
    """OpenScene frame["cams"] (or a synthetic scene's camera_dict) -> {cam: path, R, t, K, D}."""
    return {c: {"path": str(Path(sensor_root) / cam_dict[c]["data_path"]),
                "R": np.asarray(cam_dict[c]["sensor2lidar_rotation"], np.float32),
                "t": np.asarray(cam_dict[c]["sensor2lidar_translation"], np.float32),
                "K": np.asarray(cam_dict[c]["cam_intrinsic"], np.float32),
                "D": np.asarray(cam_dict[c]["distortion"], np.float32)} for c in NUPLAN_CAMS}


def cam_to_ego(cam) -> np.ndarray:
    """nuPlan camera (OpenCV axes: x right, y down, z forward) -> ego (x forward, y left, z up) rotation.
    OpenScene's lidar2ego is the identity, so sensor2lidar is sensor2ego."""
    return np.asarray(cam["R"], np.float64)


def project_nuplan(rays_ego: np.ndarray, cam, scale: int = 1):
    """Ego-frame rays (..., 3) -> pixel (..., 2) in a nuPlan camera image decoded at 1/scale, and validity.
    Brown-Conrady (k1, k2, p1, p2, k3) as OpenCV; rays are cut where the radial polynomial stops being monotonic."""
    K, (k1, k2, p1, p2, k3) = np.asarray(cam["K"], np.float64), np.asarray(cam["D"], np.float64)
    r = rays_ego @ cam_to_ego(cam)                  # rows: R^T ray
    z = r[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    x, y = r[..., 0] / zs, r[..., 1] / zs
    r2 = x * x + y * y
    # d(r * radial(r))/dr = 1 + 3 k1 r^2 + 5 k2 r^4 + 7 k3 r^6 must stay > 0
    rr = np.linspace(0, 3, 3001) ** 2
    bad = np.flatnonzero(1 + 3 * k1 * rr + 5 * k2 * rr ** 2 + 7 * k3 * rr ** 3 <= 0.05)
    r2max = rr[bad[0]] if len(bad) else rr[-1]
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u, v = K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]
    w, h = NUPLAN_WH
    ok = (z > 1e-6) & (r2 < r2max) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uv = np.stack([(u + .5) / scale - .5, (v + .5) / scale - .5], -1)
    return uv.astype(np.float32), ok, z / np.linalg.norm(rays_ego, axis=-1)


def _yaw_pitch_to_R(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera (x right, y down, z forward) -> rig, for a camera looking along yaw/pitch (roll 0), as the rig module."""
    cy, sy, cp, sp = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg)), np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    fwd = np.array([cp * cy, cp * sy, sp])
    left = np.array([-sy, cy, 0.])
    up = np.cross(fwd, left)
    return np.stack([-left, -up, fwd], 1)


def calib_key(cams: dict) -> tuple:
    return tuple(np.round(np.r_[np.ravel(cams[c]["R"]), np.ravel(cams[c]["K"]), np.ravel(cams[c]["D"])], 4).tobytes()
                 for c in sorted(cams))


class AlpamayoMaps:
    """For one nuPlan calibration: per Alpamayo camera, the source cameras and bilinear sample maps that render
    its 576x320 model image (f-theta, rig yaw/pitch, rotation only: parallax from the different mounting points is
    ignored). Each model pixel takes the nuPlan camera that sees its ray closest to the optical axis; pixels no
    camera sees stay black. Wide views sample 1/4-decoded sources, the 30-degree tele samples CAM_F0 at full size."""

    def __init__(self, cams: dict, rig=None):
        rig = rig or rigs()
        (w, h), (mw, mh) = rig.ALPAMAYO_NATIVE_WH, rig.ALPAMAYO_MODEL_WH
        u, v = np.meshgrid((np.arange(mw) + .5) * w / mw - .5, (np.arange(mh) + .5) * h / mh - .5)
        self.views, self.cam_idx = [], []
        for cam in rig.ALPAMAYO_CAMERAS:
            name, idx, _, (yaw, pitch) = cam[:4]
            rays = rig._ftheta_rays(cam, u, v) @ _yaw_pitch_to_R(yaw, pitch).T     # rig = NAVSIM ego axes
            scale = 1 if "tele" in name else SRC_SCALE
            best = np.full(rays.shape[:2], -1.0)
            pick = np.full(rays.shape[:2], -1, int)
            maps = []
            for j, c in enumerate(NUPLAN_CAMS):
                uv, ok, cosang = project_nuplan(rays, cams[c], scale)
                better = ok & (cosang > best)
                best[better], pick[better] = cosang[better], j
                maps.append(uv)
            used = [j for j in range(len(NUPLAN_CAMS)) if (pick == j).any()]
            self.views.append([(NUPLAN_CAMS[j], scale, maps[j][..., 0].copy(), maps[j][..., 1].copy(), pick == j) for j in used])
            self.cam_idx.append(idx)
        self.coverage = [float(sum(m.sum() for *_, m in v) / (mw * mh)) for v in self.views]

    def sources(self) -> set:
        return {(c, s) for v in self.views for c, s, *_ in v}

    def render(self, imgs: dict) -> np.ndarray:
        """imgs[(cam, scale)] RGB uint8 -> (N_cam, 3, 320, 576) uint8 in model camera order."""
        import cv2
        out = []
        for view in self.views:
            o = np.zeros(view[0][2].shape + (3,), np.uint8)
            for c, s, mx, my, m in view:
                o[m] = cv2.remap(imgs[(c, s)], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[m]
            out.append(o.transpose(2, 0, 1))
        return np.stack(out)


class OpenpilotMaps:
    """CAM_F0 -> openpilot narrow (focal 910) and wide (focal 455) 512x256 model frames, calib = the NAVSIM ego
    axes (level, straight), nearest sampling like tinygrad's warp. Y at 512x256, U/V at 256x128 from I420 planes."""

    def __init__(self, cam: dict):
        from .openpilot.frames import MEDMODEL_K, SBIGMODEL_K, VIEW_FROM_DEVICE, MODEL_W, MODEL_H
        self.idx = []
        for Km in (MEDMODEL_K, SBIGMODEL_K):
            per = []
            for W, H, s in ((MODEL_W, MODEL_H, 1), (MODEL_W // 2, MODEL_H // 2, 2)):
                # model pixel centre at plane scale s -> full-res model pixel -> device ray (x fwd, y right, z down)
                uu, vv = np.meshgrid((np.arange(W) + .5) * s - .5, (np.arange(H) + .5) * s - .5)
                ray_dev = np.stack([uu, vv, np.ones_like(uu)], -1) @ np.linalg.inv(Km @ VIEW_FROM_DEVICE).T
                ray_ego = ray_dev * np.array([1., -1., -1.])
                uv, ok, _ = project_nuplan(ray_ego, cam, 1)
                w, h = NUPLAN_WH[0] // s, NUPLAN_WH[1] // s
                xi = np.clip(np.rint((uv[..., 0] + .5) / s - .5), 0, w - 1).astype(np.int64)
                yi = np.clip(np.rint((uv[..., 1] + .5) / s - .5), 0, h - 1).astype(np.int64)
                per.append(((yi * w + xi).ravel(), ok.ravel()))
            self.idx.append(per)
        self.coverage = [float(p[0][1].mean()) for p in self.idx]

    def __call__(self, rgb: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Full-res RGB CAM_F0 (1080, 1920, 3) -> (2, 6, 128, 256) uint8 [narrow, wide], frames_to_tensor packing.
        Rays outside CAM_F0 (none for these FOVs) read black."""
        import cv2
        h, w = NUPLAN_WH[1], NUPLAN_WH[0]
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).ravel()
        planes = (yuv[:h * w], yuv[h * w:h * w * 5 // 4], yuv[h * w * 5 // 4:])
        out = np.empty((2, 6, 128, 256), np.uint8) if out is None else out
        for k, ((iy, oky), (iuv, okuv)) in enumerate(self.idx):
            Y = np.where(oky, planes[0][iy], 16).reshape(256, 512)
            out[k, 0], out[k, 1], out[k, 2], out[k, 3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
            out[k, 4] = np.where(okuv, planes[1][iuv], 128).reshape(128, 256)
            out[k, 5] = np.where(okuv, planes[2][iuv], 128).reshape(128, 256)
        return out


# ---------------------------------------------------------------- history / output converters

def _rot_z(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(yaw.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1
    return R


def alpamayo_history(pose: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NAVSIM ego history (4 poses (x, y, yaw) in the t0 rear-axle frame, 4 body-frame velocities) @2 Hz ->
    Alpamayo egomotion xyz (16, 3), rot (16, 3, 3) @10 Hz in the t0 frame. Cubic Hermite in x/y using the measured
    velocities (rotated into the t0 frame), cubic Hermite in yaw with yaw rates from the Hermite path's heading
    at the knots; z = 0, roll = pitch = 0."""
    from scipy.interpolate import CubicHermiteSpline, CubicSpline
    pose, vel = np.asarray(pose, np.float64), np.asarray(vel, np.float64)
    yaw = np.unwrap(pose[:, 2])
    v0 = np.stack([np.cos(yaw) * vel[:, 0] - np.sin(yaw) * vel[:, 1], np.sin(yaw) * vel[:, 0] + np.cos(yaw) * vel[:, 1]], -1)
    xy = CubicHermiteSpline(T_HIST2, pose[:, :2], v0, axis=0)(T_HIST10)
    yaw10 = CubicSpline(T_HIST2, yaw)(T_HIST10)
    xy[-1], yaw10 = 0.0, yaw10 - yaw10[-1]
    xyz = np.concatenate([xy, np.zeros((16, 1))], 1)
    return xyz.astype(np.float32), _rot_z(yaw10).astype(np.float32)


def alpamayo_to_navsim(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Alpamayo (64, 3) positions + (64, 3, 3) rotations @10 Hz in the t0 rig frame -> NAVSIM (8, 3) x, y, yaw."""
    xyz, rot = np.asarray(xyz)[ALP_OUT_IDX], np.asarray(rot)[ALP_OUT_IDX]
    return np.stack([xyz[:, 0], xyz[:, 1], np.arctan2(rot[:, 1, 0], rot[:, 0, 0])], -1).astype(np.float32)


def openpilot_to_navsim(plan_pos: np.ndarray, plan_yaw: np.ndarray, t_idx: np.ndarray, dev_xy) -> np.ndarray:
    """openpilot plan (33 points, calib frame x fwd / y right, origin at the camera, yaw clockwise) -> NAVSIM
    rear-axle (8, 3) x, y, yaw: rear(t) = d + p(t) - R(psi_t) d with d the camera's (x, y) on the vehicle."""
    p = np.stack([plan_pos[:, 0], -plan_pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(plan_yaw, np.float64)
    d = np.asarray(dev_xy, np.float64)
    Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
    rear = np.concatenate([d + p - Rd, psi[:, None]], 1)
    return np.stack([np.interp(T_OUT, t_idx, rear[:, k]) for k in range(3)], -1).astype(np.float32)


def nav_text(cmd) -> str | None:
    """NAVSIM driving_command one-hot of the current frame -> pre-registered Alpamayo route text (unknown -> None)."""
    k = int(np.argmax(cmd))
    return NAV_TEXT.get(k) if np.asarray(cmd)[k] > 0 else None
