"""HUGSIM zero-shot adapters for Alpamayo 1.5 and openpilot (todos/2026-09-25-hugsim-exam/README.md).
NumPy / SciPy / OpenCV only: imported by the per-scenario agent process in envs/hugsim.

Frames (HUGSIM interface: docs/hugsim.md)
  world  OpenCV axes of the scene's first recorded front camera: x right, y down, z forward.
  ego    the simulator's ego pose = the front camera before the dataset's rig offset (cam_rect); only its yaw
         changes and the kinematic bicycle moves it along its own z axis, so ego z IS the direction of motion.
  M      the ego pose in vehicle axes (x forward, y left, z up), origin at the front camera. openpilot's calib
         frame is M: level, straight ahead, at the camera, so the calibration is rpy = 0.
  rig    Alpamayo's vehicle frame: M moved back to the rear axle by d = CAM_FRONT's forward offset from the
         nuScenes vehicle origin (1.73 m, from CAM_FRONT's v2c), x forward, y left.
  plan   what HUGSIM takes: (N, 2) metres, x right, y forward, origin at the ego (the front camera), point k at
         t = 0.5 k s.
Cameras are reprojected rotation-only (scene at infinity, jevdrive.camgeom), exactly as in the nuScenes exam.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import camgeom as G

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")
CV2V = G.OPENCV_TO_VEHICLE                                        # OpenCV axes -> vehicle axes (x fwd, y left, z up)
WOD_IN_CV = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])       # camgeom camera axes expressed in OpenCV axes
PLAN_DT, PLAN_N = 0.5, 6                                         # 3 s at 2 Hz, as UniAD / VAD send it
NAV_TEXT = {0: "Turn right", 1: "Turn left", 2: "Continue straight"}   # HUGSIM command -> nuScenes-exam template
DESIRE = {0: 2, 1: 1, 2: 0}                                      # -> openpilot desire: turnRight 2, turnLeft 1, none 0
ALP_T = np.arange(1, 65) * 0.1


def rigs():
    """scripts/zeroshot_rigs.py (the Alpamayo virtual rig shared by all zero-shot exams)."""
    p = Path(__file__).resolve().parents[1] / "scripts" / "zeroshot_rigs.py"
    spec = importlib.util.spec_from_file_location("zeroshot_rigs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def yaw_pitch_to_R(yaw_deg, pitch_deg):
    """Camera (x right, y down, z forward) -> rig axes for a camera looking along yaw / pitch (as navsim_zs)."""
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    fwd = np.array([cp * cy, cp * sy, sp])
    left = np.array([-sy, cy, 0.0])
    up = np.cross(fwd, left)
    return np.stack([-left, -up, fwd], 1)


# ---------------------------------------------------------------- calibration

def rect_matrix(camera_yaml) -> np.ndarray:
    """The dataset's cam_rect (configs/sim/<ds>_camera.yaml), as sim_utils.load_camera_cfg builds it."""
    import yaml
    c = yaml.safe_load(Path(camera_yaml).read_text()).get("cam_rect")
    m = np.eye(4)
    if c:
        m[:3, :3] = Rotation.from_euler("XYZ", c["rot"], degrees=True).as_matrix()
        m[:3, 3] = c["trans"]
    return m


def calibs(cam_params: dict, rect: np.ndarray, cams=CAMS) -> dict:
    """info['cam_params'] -> camgeom calibrations in M. A camera renders at ego @ v2c_front @ inv(v2c) @ rect
    (hug_sim._get_obs), so camera -> ego is E = v2c_front @ inv(v2c) @ rect."""
    v2front = np.asarray(cam_params["CAM_FRONT"]["v2c"], np.float64)
    out = {}
    for c in cams:
        p = cam_params[c]
        it = p["intrinsic"]
        E = v2front @ np.linalg.inv(np.asarray(p["v2c"], np.float64)) @ rect
        ext = np.eye(4)
        ext[:3, :3] = CV2V @ E[:3, :3] @ WOD_IN_CV
        ext[:3, 3] = CV2V @ E[:3, 3]
        fx = it["W"] / (2 * np.tan(it["fovx"] / 2))
        fy = it["H"] / (2 * np.tan(it["fovy"] / 2))
        out[c] = {"intrinsic": [fx, fy, it["cx"], it["cy"], 0, 0, 0, 0, 0], "extrinsic": ext,
                  "width": int(it["W"]), "height": int(it["H"])}
    return out


def rear_offset(cam_params: dict) -> float:
    """Forward distance from the nuScenes vehicle origin (rear axle) to CAM_FRONT, from its v2c."""
    t = np.asarray(cam_params["CAM_FRONT"]["v2c"], np.float64)[:3, 3]   # vehicle origin in camera coordinates
    return float(-(CV2V @ t)[0])


# ---------------------------------------------------------------- ego pose on the ground plane

def ego_pose2d(info: dict) -> tuple[np.ndarray, float]:
    """World ground position (x, z) and heading theta (+ right; forward = (sin, cos)) from the full rotation:
    ego_rot's XYZ Euler angles flip to (pi, pi - theta, pi) beyond |theta| = 90 deg."""
    R = Rotation.from_euler("XYZ", info["ego_rot"]).as_matrix()
    return np.array([info["ego_pos"][0], info["ego_pos"][2]], np.float64), float(np.arctan2(R[0, 2], R[2, 2]))


def fwd_right(theta):
    theta = np.asarray(theta, np.float64)
    return np.stack([np.sin(theta), np.cos(theta)], -1), np.stack([np.cos(theta), -np.sin(theta)], -1)


def world_to_plan(pts, pos, theta) -> np.ndarray:
    """World ground points (N, 2) -> HUGSIM plan axes (x right, y forward) at pose (pos, theta)."""
    f, r = fwd_right(theta)
    d = np.asarray(pts, np.float64) - pos
    return np.stack([d @ r, d @ f], -1)


def plan_to_world(plan, pos, theta) -> np.ndarray:
    f, r = fwd_right(theta)
    p = np.asarray(plan, np.float64)
    return pos + p[:, :1] * r + p[:, 1:2] * f


class History:
    """Ego (front camera) poses at the simulator's 4 Hz, and the rear-axle egomotion Alpamayo wants at 10 Hz."""

    def __init__(self):
        self.t, self.pos, self.th, self.v = [], [], [], []

    def add(self, info):
        pos, th = ego_pose2d(info)
        if self.th:                                           # keep theta continuous
            th = self.th[-1] + (th - self.th[-1] + np.pi) % (2 * np.pi) - np.pi
        self.t.append(float(info["timestamp"]))
        self.pos.append(pos)
        self.th.append(th)
        self.v.append(float(info["ego_velo"]))

    def rear_hist(self, d: float, n: int = 16, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Alpamayo egomotion: n steps at dt ending now, rear-axle xyz (z = 0) and rotations in the current rig
        frame. Between simulator steps: linear in position and heading. Before the episode started: the start
        state rolled backwards at its own speed and heading (the car was already moving at start_velo)."""
        t = np.asarray(self.t)
        pos, th = np.asarray(self.pos), np.asarray(self.th)
        f, _ = fwd_right(th)
        rear = pos - d * f
        tq = t[-1] - dt * np.arange(n - 1, -1, -1)
        before = tq < t[0]
        xy = np.stack([np.interp(tq, t, rear[:, k]) for k in (0, 1)], -1)
        hq = np.interp(tq, t, th)
        xy[before] = rear[0] - self.v[0] * (t[0] - tq[before])[:, None] * f[0]
        f0, r0 = fwd_right(th[-1])
        rel = xy - xy[-1]
        x, y = rel @ f0, -(rel @ r0)
        yaw = -(hq - hq[-1])                                  # left-positive in the rig frame
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.zeros((n, 3, 3), np.float32)
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
        return np.stack([x, y, np.zeros(n)], -1).astype(np.float32), rot


# ---------------------------------------------------------------- model inputs

class AlpamayoViews:
    """The four Alpamayo f-theta views (576x320) from the HUGSIM cameras, as nusc_zs_alpamayo.Maps: per model pixel
    the source camera that sees its ray closest to its optical axis, bilinear. `cams` = the cameras with images
    (Waymo / KITTI-360 render the three back cameras as zeros; they are left out)."""

    def __init__(self, cal: dict, cams):
        R = rigs()
        (w, h), (mw, mh) = R.ALPAMAYO_NATIVE_WH, R.ALPAMAYO_MODEL_WH
        u, v = np.meshgrid((np.arange(mw) + .5) * w / mw - .5, (np.arange(mh) + .5) * h / mh - .5)
        sub = {c: cal[c] for c in cams}
        self.views, self.coverage, self.cam_idx = [], [], []
        for cam in R.ALPAMAYO_CAMERAS:
            name, idx, _, (yaw, pitch) = cam[:4]
            rays = R._ftheta_rays(cam, u, v) @ yaw_pitch_to_R(yaw, pitch).T
            src, U, V = G.choose_sources(np, rays, sub)
            self.views.append([(c, U.astype(np.float32), V.astype(np.float32), src == j)
                               for j, c in enumerate(sub) if (src == j).any()])
            self.coverage.append(float((src >= 0).mean()))
            self.cam_idx.append(idx)

    def render(self, rgb: dict) -> np.ndarray:
        import cv2
        out = []
        for view in self.views:
            o = np.zeros(view[0][1].shape + (3,), np.uint8)
            for c, mx, my, m in view:
                o[m] = cv2.remap(rgb[c], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[m]
            out.append(o.transpose(2, 0, 1))
        return np.stack(out)                                   # (4, 3, 320, 576) RGB


class OpenpilotFrames:
    """openpilot road / wide model frames (512x256, calib = M) from the front cameras: nearest neighbour gather
    as modeld's warp, BT.601 limited-range YUV (the comma ISP's range, as the CARLA adapter), chroma 2x2 mean,
    packed 6x128x256 per frame."""

    def __init__(self, cal: dict, cams=("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")):
        sub = {c: cal[c] for c in cams}
        self.cams = list(sub)
        sizes = [(sub[c]["width"], sub[c]["height"]) for c in self.cams]
        self.idx, self.coverage, self.src_frac = {}, {}, {}
        for k in ("road", "wide"):
            src, U, V = G.choose_sources(np, G.pinhole_rays(np, G.OP_K[k], G.OP_W, G.OP_H), sub)
            self.coverage[k] = float((src >= 0).mean())
            self.src_frac[k] = {c: float((src == j).mean()) for j, c in enumerate(self.cams)}
            self.idx[k] = G.nn_gather_index(src, U, V, sizes).ravel()

    def pack(self, rgb: dict) -> np.ndarray:
        cat = np.concatenate([rgb[c].reshape(-1, 3) for c in self.cams] + [np.zeros((1, 3), np.uint8)])
        out = np.empty((2, 6, 128, 256), np.uint8)
        for m, k in enumerate(("road", "wide")):
            px = cat[self.idx[k]].astype(np.float32)           # index -1 -> the appended black pixel
            r, g, b = px[:, 0], px[:, 1], px[:, 2]
            Y = (16 + 0.257 * r + 0.504 * g + 0.098 * b).reshape(G.OP_H, G.OP_W)
            U = (128 - 0.148 * r - 0.291 * g + 0.439 * b).reshape(G.OP_H, G.OP_W)
            V = (128 + 0.439 * r - 0.368 * g - 0.071 * b).reshape(G.OP_H, G.OP_W)
            uv = np.stack([U, V], -1).reshape(128, 2, 256, 2, 2).mean((1, 3))
            Y, uv = (np.clip(np.rint(x), 0, 255).astype(np.uint8) for x in (Y, uv))
            out[m] = np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2], uv[..., 0], uv[..., 1]])
        return out


# ---------------------------------------------------------------- model outputs -> HUGSIM plan

def plan_times(n=PLAN_N, dt=PLAN_DT):
    return dt * np.arange(1, n + 1)


def alpamayo_to_plan(xyz, yaw, d: float, t_src=ALP_T, rigid: bool = True) -> np.ndarray:
    """Alpamayo rear-axle trajectory (rig frame, x fwd, y left; yaw left-positive) -> HUGSIM plan of the front
    camera: cam(t) = rear(t) + R(yaw_t) (d, 0) - (d, 0) (rigid body; rigid=False only shifts the rear track)."""
    t = np.r_[0.0, t_src]
    xy = np.r_[[[0.0, 0.0]], np.asarray(xyz, np.float64)[:, :2]]
    yw = np.unwrap(np.r_[0.0, np.asarray(yaw, np.float64)])
    tq = plan_times()
    x, y, h = (np.interp(tq, t, a) for a in (xy[:, 0], xy[:, 1], yw))
    if rigid:
        x, y = x + d * np.cos(h) - d, y + d * np.sin(h)
    return np.stack([-y, x], -1)


def forward_only(plan) -> np.ndarray:
    """No reverse plans: walk the plan from the ego origin and zero every segment that points backwards relative to
    the direction of travel so far (the ego's forward axis until the first kept segment), then re-accumulate. A
    model's "stop" that integrates to a small reverse track becomes a zero-length plan (stand still) instead of a
    reference behind the car, which iLQR would track by reversing. Forward plans, including sharp turns, are
    unchanged. Returns a new (N, 2) array (x right, y forward)."""
    p = np.asarray(plan, np.float64)
    seg = np.diff(np.r_[[[0.0, 0.0]], p], axis=0)
    ref = np.array([0.0, 1.0])
    for k in range(len(seg)):
        if seg[k] @ ref < 0:
            seg[k] = 0.0
        elif np.linalg.norm(seg[k]) > 1e-6:
            ref = seg[k] / np.linalg.norm(seg[k])
    return np.cumsum(seg, axis=0)


def openpilot_to_plan(plan_pos, t_idxs, dilation: float = 1.0) -> np.ndarray:
    """openpilot plan (calib frame at the camera: x fwd, y right; model time tau) -> HUGSIM plan. The model's
    clock runs `dilation` times faster than the simulator's (one 0.25 s step is fed as one 0.2 s context step),
    so real time t is model time tau * dilation."""
    tau = plan_times() / dilation
    p = np.asarray(plan_pos, np.float64)
    return np.stack([np.interp(tau, t_idxs, p[:, 1]), np.interp(tau, t_idxs, p[:, 0])], -1)
