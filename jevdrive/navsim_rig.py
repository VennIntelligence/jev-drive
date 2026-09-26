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
# nuPlan CAM_L0 / F0 / R0 sensor2lidar rotations (OpenCV camera axes -> ego) and F0's position, same log. A virtual
# camera keeps its template's mounting (pitch, roll) and is only turned about z to the mapped source camera's yaw.
R_TEMPLATE = {"cam_l0": np.array([[0.8209, 0.0059, 0.5711], [-0.5709, -0.0157, 0.8209], [0.0138, -0.9999, -0.0095]]),
              "cam_f0": np.array([[0.0031, -0.0253, 0.9997], [-1.0, -0.0088, 0.0029], [0.0088, -0.9996, -0.0253]]),
              "cam_r0": np.array([[-0.8236, 0.0132, 0.567], [-0.567, 0.0007, -0.8237], [-0.0112, -0.9999, 0.0069]])}
T_F0 = np.array([1.6701, -0.0259, 1.5226])
WOD_IN_CV = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])     # camgeom camera axes expressed in OpenCV axes
NAMES = ("cam_l0", "cam_f0", "cam_r0")


def _orth(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def optical_yaw(R: np.ndarray) -> float:
    """Yaw (deg) of a camera's optical axis (OpenCV z) in the ego frame."""
    return float(np.degrees(np.arctan2(R[1, 2], R[0, 2])))


def virtual(name: str, yaw_deg: float, pos) -> dict:
    """Virtual camera `name` (cam_l0 / cam_f0 / cam_r0): its nuPlan template turned about z so the optical axis points
    at yaw_deg; sensor2lidar rotation (OpenCV axes -> ego), translation, K, distortion."""
    R0 = _orth(R_TEMPLATE[name])
    a = np.radians(yaw_deg - optical_yaw(R0))
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return {"sensor2lidar_rotation": Rz @ R0, "sensor2lidar_translation": np.asarray(pos, np.float64),
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


_GRID = {}


def _cam_rays(K_: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 unit rays in OpenCV camera axes through the pixel centres of a distorted camera; cached per
    (K, distortion), since every virtual camera shares nuPlan's and only its rotation differs."""
    import cv2
    key = np.r_[np.ravel(K_), np.ravel(dist)].tobytes()
    if key not in _GRID:
        u, w = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64), indexing="xy")
        pts = np.stack([u, w], -1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
        xy = cv2.undistortPointsIter(pts, K_, dist, None, None, crit).reshape(H, W, 2)
        d = np.concatenate([xy, np.ones((H, W, 1))], -1)
        _GRID[key] = (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)
    return _GRID[key]


def rays(v: dict) -> np.ndarray:
    """(H, W, 3) float32 unit rays in the ego frame through the pixel centres of a distorted virtual camera."""
    return _cam_rays(v["intrinsics"], v["distortion"]) @ v["sensor2lidar_rotation"].T.astype(np.float32)


def project(rays: np.ndarray, calib: dict):
    """camgeom.waymo_project with one change: the radius up to which a ray counts as seen is taken from the image
    corners *undistorted* (camgeom bounds the undistorted r^2 by the distorted corner's, which is right for WOD's
    mild lenses but cuts the corners off a strongly barrel-distorted nuPlan source)."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = (float(x) for x in calib["intrinsic"])
    w, h = calib["width"], calib["height"]
    rd2 = max(((a - cu) / fu) ** 2 + ((b - cv) / fv) ** 2 for a in (0, w) for b in (0, h))
    r = np.sqrt(rd2)
    rr = np.linspace(0, 3 * r + 1, 30001)                              # radial model only, first crossing of r_d
    f = rr * (1 + k1 * rr ** 2 + k2 * rr ** 4 + k3 * rr ** 6)
    mono = np.r_[True, np.diff(f) > 0].cumprod().astype(bool)
    hit = np.flatnonzero(mono & (f >= r))
    ru2 = (rr[hit[0]] if len(hit) else rr[mono][-1]) ** 2
    R = np.asarray(calib["extrinsic"], np.float64).reshape(4, 4)[:3, :3]
    p = rays @ R.astype(rays.dtype)
    fwd = p[..., 0]
    safe = np.where(fwd > 1e-6, fwd, 1)
    x, y = -p[..., 1] / safe, -p[..., 2] / safe
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    u = fu * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cu
    v = fv * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cv
    ok = (fwd > 0.05) & (r2 < 1.3 * max(ru2, rd2)) & (u >= -0.5) & (u <= w - 0.5) & (v >= -0.5) & (v <= h - 0.5)
    return u, v, ok


def maps(src: list, virt: list, primary: list) -> list:
    """Per virtual camera: (source index or -1, U, V) as int8 / float32. src: camgeom records; primary[i]: the source
    virtual camera i is mapped to. The primary source wins wherever it sees the ray; elsewhere the source that sees it
    closest to its optical axis (camgeom.choose_sources' rule)."""
    axes = [np.asarray(c["extrinsic"], np.float64).reshape(4, 4)[:3, 0].astype(np.float32) for c in src]
    out = []
    for v, p in zip(virt, primary):
        r = rays(v)
        s = np.full(r.shape[:2], -1, np.int8)
        U, V = np.zeros(r.shape[:2], np.float32), np.zeros(r.shape[:2], np.float32)
        best = np.full(r.shape[:2], -2.0, np.float32)
        for i in [j for j in range(len(src)) if j != p] + [p]:           # primary last: it overrides where it sees
            u, w, ok = project(r, src[i])
            take = ok if i == p else ok & (r @ axes[i] > best)
            if i != p:
                best = np.where(take, r @ axes[i], best)
            s, U, V = np.where(take, i, s), np.where(take, u, U), np.where(take, w, V)
        out.append((s.astype(np.int8), U.astype(np.float32), V.astype(np.float32)))
    return out


PAD = 2          # replicated border around every source in the atlas (cv2 bilinear reads x and x + 1)


def fast_maps(mp: list, sizes: list) -> dict:
    """Precompute render()'s sampling as one fixed-point cv2.remap per virtual camera over an atlas of the sources
    (stacked vertically, each with a PAD-pixel replicated border, and a black block for uncovered pixels). The fixed
    point maps are cv2.convertMaps of the very float maps render() hands cv2.remap, shifted by whole pixels only,
    so every output pixel is bit-identical to render(). sizes[i] = (w, h) of source i."""
    import cv2
    wmax = max(w for w, _ in sizes) + 2 * PAD
    offs, y = [], 0
    for w, h in sizes:
        offs.append(y)
        y += h + 2 * PAD
    black = y
    out = []
    for s, U, V in mp:
        m1, m2 = cv2.convertMaps(U, V, cv2.CV_16SC2)
        m1 = m1.astype(np.int32)
        dy = np.full(s.shape, black + PAD, np.int32)
        dx = np.full(s.shape, PAD, np.int32)
        for i, off in enumerate(offs):
            dy[s == i] = off + PAD
        m1[..., 0] += dx
        m1[..., 1] += dy
        m2 = np.where(s >= 0, m2, 0).astype(np.uint16)
        m1[s < 0] = (PAD, black + PAD)
        out.append((m1.astype(np.int16), m2))
    return {"maps": out, "offs": offs, "shape": (black + 4 * PAD + 2, wmax), "sizes": list(sizes)}


def render_fast(images: list, fm: dict) -> list:
    """render() through fast_maps(): bit-identical output, one remap per virtual camera."""
    import cv2
    atlas = np.zeros(fm["shape"] + images[0].shape[2:], np.uint8)
    for img, off in zip(images, fm["offs"]):
        b = cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        atlas[off:off + b.shape[0], :b.shape[1]] = b
    return [cv2.remap(atlas, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for m1, m2 in fm["maps"]]


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
