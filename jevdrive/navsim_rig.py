"""Virtual NAVSIM (nuPlan) cameras rendered from another rig, for NAVSIM-trained models that hard-code nuPlan's
1920 x 1080 images (SparseDriveV2 resizes by the config's H / W, ZTRS crops fixed pixel ranges); top-10 exam T1
(todos/2026-09-26-top10-intersection.md, [T1] 10:05).

A virtual camera has nuPlan's intrinsics and distortion (the models were trained on the raw, distorted nuPlan images)
and nuPlan CAM_F0's orientation turned about z to a chosen yaw. Its pixels come by rotation-only reprojection
(jevdrive.camgeom): first from the source camera it is mapped to, then, where that one cannot see, from whichever
other source camera sees the ray closest to its optical axis; pixels no source sees are black.

Frames: ego = NAVSIM's lidar frame = rear axle, x forward, y left, z up. Source calibrations are camgeom records
(WOD layout: intrinsic [fu fv cu cv k1 k2 p1 p2 k3], extrinsic = vehicle_from_camera with camera axes x optical,
y left, z up). NumPy + OpenCV only: imported by the model envs.
"""
import numpy as np

from . import camgeom as G

W, H = 1920, 1080
# nuPlan camera model as NAVSIM ships it (navsim_logs/test, every CAM_* of the first log): K and OpenCV distortion
K = np.array([[1545.0, 0, 960.0], [0, 1545.0, 560.0], [0, 0, 1]])
DIST = np.array([-0.3561, 0.1725, -0.0021, 0.0005, -0.0523])      # k1 k2 p1 p2 k3
# CAM_F0 sensor2lidar (OpenCV camera axes -> ego) and position, same log; L0 / R0 sit at yaw +-55.2 deg
R_F0 = np.array([[0.0031, -0.0253, 0.9997], [-1.0, -0.0088, 0.0029], [0.0088, -0.9996, -0.0253]])
T_F0 = np.array([1.6701, -0.0259, 1.5226])
WOD_IN_CV = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])     # camgeom camera axes expressed in OpenCV axes
NAMES = ("cam_l0", "cam_f0", "cam_r0")


def _orth(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def virtual(yaw_deg: float, pos) -> dict:
    """One virtual camera: sensor2lidar rotation (OpenCV axes -> ego), translation, K, distortion."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return {"sensor2lidar_rotation": Rz @ _orth(R_F0), "sensor2lidar_translation": np.asarray(pos, np.float64),
            "intrinsics": K.copy(), "distortion": DIST.copy()}


def as_camgeom(v: dict) -> dict:
    """A virtual (or real nuPlan) camera as a camgeom source record."""
    E = np.eye(4)
    E[:3, :3] = v["sensor2lidar_rotation"] @ WOD_IN_CV
    E[:3, 3] = v["sensor2lidar_translation"]
    k1, k2, p1, p2, k3 = v["distortion"]
    Kc = v["intrinsics"]
    return {"intrinsic": [Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2], k1, k2, p1, p2, k3], "extrinsic": E.ravel().tolist(),
            "width": W, "height": H}


def rays(v: dict) -> np.ndarray:
    """(H, W, 3) unit rays in the ego frame through the pixel centres of a distorted virtual camera."""
    import cv2
    u, w = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64), indexing="xy")
    pts = np.stack([u, w], -1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
    xy = cv2.undistortPointsIter(pts, v["intrinsics"], v["distortion"], None, None, crit).reshape(H, W, 2)
    d = np.concatenate([xy, np.ones((H, W, 1))], -1) @ v["sensor2lidar_rotation"].T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def maps(src: list, virt: list, primary: list) -> list:
    """Per virtual camera: (source index or -1, U, V) as float32 / int8. src: camgeom records; primary[i]: the source
    virtual camera i is mapped to (it wins wherever it sees the ray)."""
    out = []
    for v, p in zip(virt, primary):
        r = rays(v)
        s, U, V = G.choose_sources(np, r, {i: c for i, c in enumerate(src)})
        u, w, ok = G.waymo_project(np, r, src[p])
        s, U, V = np.where(ok, p, s), np.where(ok, u, U), np.where(ok, w, V)
        out.append((s.astype(np.int8), U.astype(np.float32), V.astype(np.float32)))
    return out


def render(images: list, mp: list) -> list:
    """RGB uint8 source images -> one (H, W, 3) uint8 image per virtual camera (bilinear, uncovered = 0)."""
    return [G.render_np(s, U, V, images) for s, U, V in mp]


def spline_grid(traj: np.ndarray, dt: float, n: int = 20, step: float = 0.25) -> np.ndarray:
    """(m, k, >=2) poses every `dt` s -> (m, n, 2) on the `step` grid: cubic spline through (0, 0) and the k points,
    straight-line extrapolation from the last two points beyond the model's horizon."""
    from scipy.interpolate import CubicSpline
    m, k = traj.shape[:2]
    t = np.arange(k + 1) * dt
    xy = np.concatenate([np.zeros((m, 1, 2)), traj[..., :2]], 1).astype(np.float64)
    q = np.arange(1, n + 1) * step
    inside = q <= t[-1] + 1e-9
    out = np.empty((m, n, 2))
    out[:, inside] = CubicSpline(t, xy, axis=1)(q[inside])
    vel = (xy[:, -1] - xy[:, -2]) / dt
    out[:, ~inside] = xy[:, -1:, :] + vel[:, None] * (q[~inside] - t[-1])[None, :, None]
    return out.astype(np.float32)
