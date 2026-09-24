"""Rotation-only camera reprojection between rigs: render a virtual camera from WOD-E2E's pinhole cameras.

Every function takes an array module `xp` (numpy or torch) so the same geometry builds numpy index maps for the
openpilot model frames and torch sampling grids on the GPU for Alpamayo. Frames:

  vehicle   x forward, y left, z up (WOD vehicle frame and the PhysicalAI-AV rig frame; both at the rear axle)
  WOD cam   x along the optical axis, y left, z up; extrinsic = vehicle_from_camera (4x4)
  OpenCV    x right, y down, z along the optical axis (PhysicalAI-AV cameras, openpilot view frame)

Translation between cameras is ignored (scene at infinity): with no depth this is the only possible mapping, and
it puts near objects at the wrong place by the parallax between the two mounting points.
"""
import numpy as np

# PhysicalAI-AV (Alpamayo's training rig), per-coefficient median over the 100 clips of calibration chunk 0 of
# nvidia/PhysicalAI-Autonomous-Vehicles @ 33f9bf4 (camera_intrinsics, sensor_extrinsics); rotation = the
# chordal mean of the 100 camera->rig quaternions. f-theta model: theta = bw(r), r = |pixel - (cx, cy)|.
PAI_RIG = {
    "cross_left": dict(index=0, w=1920, h=1080, cx=961.30, cy=542.43,
                       bw=(0.0, 1.079063e-03, 3.572538e-09, 1.922103e-11, 2.744209e-16),
                       quat=(0.692948, -0.143183, 0.140525, -0.692513)),
    "front_wide": dict(index=1, w=1920, h=1080, cx=959.45, cy=541.85,
                       bw=(0.0, 1.080446e-03, 6.44692e-09, 1.845382e-11, 1.188537e-15),
                       quat=(0.500052, -0.503067, 0.500107, -0.496754)),
    "cross_right": dict(index=2, w=1920, h=1080, cx=962.51, cy=541.64,
                        bw=(0.0, 1.079809e-03, 2.268454e-09, 2.318624e-11, -1.181332e-15),
                        quat=(0.143723, -0.685494, 0.698899, -0.144848)),
    "front_tele": dict(index=6, w=1920, h=1080, cx=961.53, cy=544.51,
                       bw=(0.0, 2.711963e-04, 2.191437e-10, -9.174683e-13, 3.455777e-16),
                       quat=(0.499513, -0.50106, 0.500121, -0.499304)),
}
PAI_ORDER = ("cross_left", "front_wide", "cross_right", "front_tele")  # camera index 0, 1, 2, 6 (loader order)

# openpilot model frames (common/transformations/model.py): view frame intrinsics, 512 x 256
OP_W, OP_H = 512, 256
OP_K = {"road": np.array([[910.0, 0, 256.0], [0, 910.0, 47.6], [0, 0, 1]]),
        "wide": np.array([[455.0, 0, 256.0], [0, 455.0, 0.5 * (256 + 47.6)], [0, 0, 1]])}

OPENCV_TO_VEHICLE = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # columns: cam x, y, z in vehicle axes


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _grid(xp, w, h, like=None, scale=1.0):
    kw = {} if xp is np else {"device": like.device if like is not None else "cpu", "dtype": xp.float64}
    u = (xp.arange(w, **kw) + 0.5) / scale - 0.5  # pixel centres of a (scaled) render
    v = (xp.arange(h, **kw) + 0.5) / scale - 0.5
    return xp.meshgrid(u, v, indexing="xy")


def ftheta_rays(xp, cam: dict, scale: float = 1.0, device=None):
    """Unit rays (h, w, 3) in the vehicle frame for a PhysicalAI-AV f-theta camera rendered at `scale`."""
    like = None if xp is np else xp.empty(0, device=device)
    u, v = _grid(xp, round(cam["w"] * scale), round(cam["h"] * scale), like, scale)
    dx, dy = u - cam["cx"], v - cam["cy"]
    r = xp.sqrt(dx * dx + dy * dy)
    theta = sum(c * r ** i for i, c in enumerate(cam["bw"]))
    s = xp.sin(theta) / xp.where(r > 1e-9, r, xp.ones_like(r))
    ray = xp.stack([dx * s, dy * s, xp.cos(theta)], -1)
    R = quat_to_matrix(cam["quat"])  # camera (OpenCV) -> rig
    R = R if xp is np else xp.as_tensor(R, device=device)
    return ray @ R.T


def pinhole_rays(xp, K: np.ndarray, w: int, h: int):
    """Unit rays (h, w, 3) in the vehicle frame for an axis-aligned pinhole (openpilot calib/view frame)."""
    u, v = _grid(xp, w, h)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ OPENCV_TO_VEHICLE.T
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def waymo_project(xp, rays, calib: dict):
    """Project vehicle-frame rays into one WOD camera. calib: intrinsic (9,) [fu fv cu cv k1 k2 p1 p2 k3],
    extrinsic (4, 4) vehicle_from_camera, width, height. Returns pixel u, v and a validity mask (in front of the
    camera, inside the image, and inside the radius where the distortion polynomial is still monotonic)."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = (float(x) for x in calib["intrinsic"])
    R = np.asarray(calib["extrinsic"], np.float64).reshape(4, 4)[:3, :3]  # camera -> vehicle
    R = R if xp is np else xp.as_tensor(R, device=rays.device)
    p = rays @ R                                                           # vehicle -> camera: R^T d
    fwd = p[..., 0]
    safe = xp.where(fwd > 1e-6, fwd, xp.ones_like(fwd))
    x, y = -p[..., 1] / safe, -p[..., 2] / safe
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u, v = fu * xd + cu, fv * yd + cv
    w, h = calib["width"], calib["height"]
    corner = max(((a - cu) / fu) ** 2 + ((b - cv) / fv) ** 2 for a in (0, w) for b in (0, h))
    ok = (fwd > 0.05) & (r2 < 1.3 * corner) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
    return u, v, ok


def choose_sources(xp, rays, calibs: dict):
    """For every ray the WOD camera that sees it closest to its optical axis. calibs: {name: calib}.
    Returns (src index into list(calibs) or -1, u, v), each (h, w)."""
    names = list(calibs)
    best = xp.full(rays.shape[:-1], -2.0) if xp is np else xp.full(rays.shape[:-1], -2.0, device=rays.device,
                                                                     dtype=rays.dtype)
    src = xp.full(rays.shape[:-1], -1) if xp is np else xp.full(rays.shape[:-1], -1, device=rays.device)
    U, V = xp.zeros_like(best), xp.zeros_like(best)
    for i, n in enumerate(names):
        u, v, ok = waymo_project(xp, rays, calibs[n])
        axis = np.asarray(calibs[n]["extrinsic"], np.float64).reshape(4, 4)[:3, 0]
        cos = rays @ (axis if xp is np else xp.as_tensor(axis, device=rays.device))
        take = ok & (cos > best)
        best = xp.where(take, cos, best)
        src = xp.where(take, xp.full_like(src, i), src)
        U, V = xp.where(take, u, U), xp.where(take, v, V)
    return src, U, V


def render_np(src, U, V, images: list, nearest: bool = False) -> np.ndarray:
    """CPU render from choose_sources() output: images[i] is camera i's (H, W, C) uint8 array. Uncovered = 0."""
    import cv2
    out = np.zeros(src.shape + images[0].shape[2:], np.uint8)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    for i, img in enumerate(images):
        m = src == i
        if m.any():
            r = cv2.remap(img, U.astype(np.float32), V.astype(np.float32), interp, borderMode=cv2.BORDER_REPLICATE)
            out[m] = r[m]
    return out


def nn_gather_index(src, U, V, sizes) -> np.ndarray:
    """Flat indices into the concatenation of the cameras' raveled planes (nearest neighbour, as modeld's
    warp), for a plane-at-a-time gather. sizes[i] = (w, h) of camera i. Uncovered pixels -> -1."""
    base = np.concatenate([[0], np.cumsum([w * h for w, h in sizes])[:-1]])
    idx = np.full(src.shape, -1, np.int64)
    for i, (w, h) in enumerate(sizes):
        m = src == i
        x = np.clip(np.rint(U[m]), 0, w - 1).astype(np.int64)
        y = np.clip(np.rint(V[m]), 0, h - 1).astype(np.int64)
        idx[m] = base[i] + y * w + x
    return idx
