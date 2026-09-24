"""openpilot model frames from other camera rigs, simulated on real comma video (comma1M).

What would the model see if the comma road + wide cameras were replaced by another rig (one fisheye, a 3-camera
ring, a higher mount, a coarser sensor)? Every model-frame pixel goes through two stages, so no full rig image is
ever rendered:

  1. physics   rig pixel <- comma source. The rig pixel's ray is looked up in the comma camera that holds the
               model frame's information: fcam (focal 2648) for the road frame (focal 910), ecam (focal 567) for
               the wide frame (focal 455). When the rig camera is coarser than the source, the source is first
               area-averaged to the rig focal (a coarse pixel integrates light over its footprint), then sampled
               bilinearly. A mounting-height change is simulated with the ground plane: road-surface pixels are
               exact, anything standing on the road is flattened onto it; rays above the horizon are unchanged.
  2. adapter   model pixel -> rig pixel, as a migration adapter does it: rotation-only reprojection (scene at
               infinity) through the rig camera whose axis is closest to the ray, nearest neighbour like modeld's
               warp (compile_warp.py), chroma sampled at half the luma coordinates. The adapter's belief about
               the rig (intrinsics, lens model, orientation) may differ from the physical rig: that is how a
               miscalibration or a fisheye treated as a pinhole is simulated.

Frames: calib = openpilot's road-aligned device frame (x forward, y right, z down); view = x right, y down,
z forward. Cam.R is calib_from_view. The comma wide camera is a fisheye that openpilot treats as a pinhole of
focal 567; this module treats it the same way, so "native" reproduces modeld's frames bit for bit and every
other rig is measured against what the model was trained on.
"""
from dataclasses import dataclass, field

import numpy as np

from .frames import MEDMODEL_K, MODEL_H, MODEL_W, SBIGMODEL_K, VIEW_FROM_DEVICE, rot_from_euler

NOMINAL_HEIGHT_M = 1.22  # openpilot's nominal camera height above the road


class Cam:
    """Pinhole or equidistant-fisheye camera; pixel centres at integer coordinates (as modeld's warp)."""

    def __init__(self, w, h, f, cx=None, cy=None, yaw=0.0, pitch=0.0, roll=0.0, model="pinhole", R=None, name=""):
        """yaw right-positive, pitch up-positive, degrees, in the calib frame (openpilot euler convention)."""
        self.w, self.h, self.f, self.model, self.name = int(w), int(h), float(f), model, name
        self.cx = w / 2 if cx is None else float(cx)
        self.cy = h / 2 if cy is None else float(cy)
        self.R = R if R is not None else rot_from_euler(np.radians([roll, pitch, yaw])) @ VIEW_FROM_DEVICE.T

    @classmethod
    def from_K(cls, K, w, h, **kw):
        return cls(w, h, K[0, 0], K[0, 2], K[1, 2], **kw)

    def replace(self, **kw):
        d = dict(w=self.w, h=self.h, f=self.f, cx=self.cx, cy=self.cy, model=self.model, R=self.R, name=self.name)
        d.update(kw)
        return Cam(**d)

    @property
    def axis(self):
        return self.R[:, 2]

    def rays(self, u, v):
        """Unit rays (..., 3) in the calib frame of pixels (u, v)."""
        du, dv = (np.asarray(u, np.float64) - self.cx) / self.f, (np.asarray(v, np.float64) - self.cy) / self.f
        if self.model == "pinhole":
            p = np.stack([du, dv, np.ones_like(du)], -1)
        else:  # equidistant: r = f * theta
            th = np.hypot(du, dv)
            s = np.where(th > 1e-12, np.sin(th) / np.maximum(th, 1e-12), 1.0)
            p = np.stack([du * s, dv * s, np.cos(th)], -1)
        p = p @ self.R.T
        return p / np.linalg.norm(p, axis=-1, keepdims=True)

    def project(self, rays):
        """Calib-frame rays -> pixel u, v and a mask (in front of the lens and inside the image)."""
        p = rays @ self.R
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        if self.model == "pinhole":
            front = z > 1e-6
            zs = np.where(front, z, 1.0)
            u, v = self.f * x / zs + self.cx, self.f * y / zs + self.cy
        else:
            rho = np.hypot(x, y)
            th = np.arctan2(rho, z)
            front = th < np.pi * 0.999
            s = self.f * th / np.maximum(rho, 1e-12)
            u, v = self.cx + s * x, self.cy + s * y
        ok = front & (u >= -0.5) & (u <= self.w - 0.5) & (v >= -0.5) & (v <= self.h - 0.5)
        return u, v, ok


def model_cam(kind):
    return Cam.from_K(MEDMODEL_K if kind == "road" else SBIGMODEL_K, MODEL_W, MODEL_H, R=VIEW_FROM_DEVICE.T,
                      name=f"model_{kind}")


def comma_cams(rpy_calib, wh=(1928, 1208), focal=(2648.0, 567.0)):
    """The comma road (fcam) and wide (ecam) cameras in the calib frame; both share the device orientation, as in
    modeld (liveCalibration's rpyCalib is device_from_calib)."""
    R = rot_from_euler(rpy_calib).T @ VIEW_FROM_DEVICE.T
    return Cam(*wh, focal[0], R=R, name="fcam"), Cam(*wh, focal[1], R=R, name="ecam")


@dataclass
class Rig:
    """A physical rig (`true`), the adapter's belief about it (`belief`, same order), which comma source feeds
    each model frame (`src`: 'fcam' / 'ecam'), and an optional mounting-height change (m, up positive)."""
    true: list
    belief: list | None = None
    src: dict = field(default_factory=lambda: {"road": "fcam", "wide": "ecam"})
    dh: float = 0.0
    use: dict | None = None  # model frame -> candidate rig camera indices (default: all)

    def __post_init__(self):
        self.belief = self.belief or self.true


def _model_grid(kind, chroma):
    """Luma coordinates of the model pixels whose value is looked up (chroma: modeld's (2x, 2y) rule)."""
    if chroma:
        x, y = np.meshgrid(np.arange(MODEL_W // 2) * 2.0, np.arange(MODEL_H // 2) * 2.0)
    else:
        x, y = np.meshgrid(np.arange(MODEL_W, dtype=np.float64), np.arange(MODEL_H, dtype=np.float64))
    return model_cam(kind).rays(x, y)


def _homography(model, cam, chroma):
    """model pixel -> rig pixel for a pinhole rig camera, in float32 exactly as modeld's warp computes it
    (get_warp_matrix, then compile_warp's uv_scale for the chroma planes)."""
    Km = np.array([[model.f, 0, model.cx], [0, model.f, model.cy], [0, 0, 1.0]])
    Kc = np.array([[cam.f, 0, cam.cx], [0, cam.f, cam.cy], [0, 0, 1.0]])
    M = Kc @ cam.R.T @ model.R @ np.linalg.inv(Km)
    if chroma:
        M = M * np.array([[1, 1, .5], [1, 1, .5], [2, 2, 1]])
    M = M.astype(np.float32)
    w, h = (MODEL_W // 2, MODEL_H // 2) if chroma else (MODEL_W, MODEL_H)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy, sw = (M[i, 0] * x + M[i, 1] * y + M[i, 2] for i in range(3))
    return sx / sw, sy / sw, sw > 0


def adapter_lookup(kind, belief, chroma, use=None):
    """Stage 2: for each model pixel, (rig camera index or -1, integer rig pixel x, y) - on the rig camera's
    chroma plane when chroma. `use` restricts the candidate cameras (the comma rig: road <- fcam, wide <- ecam).
    Among candidates the one whose axis is closest to the ray wins; with a single candidate, pixels outside it
    are clipped to its border like modeld's warp, with several they stay uncovered (-1)."""
    use = list(range(len(belief))) if use is None else list(use)
    model = model_cam(kind)
    rays = _model_grid(kind, chroma)
    best = np.full(rays.shape[:-1], -2.0)
    cam = np.full(rays.shape[:-1], -1)
    X, Y = np.zeros_like(best), np.zeros_like(best)
    div = 2.0 if chroma else 1.0
    for i in use:
        c = belief[i]
        if c.model == "pinhole":
            u, v, front = _homography(model, c, chroma)
            u, v = u.astype(np.float64) * div, v.astype(np.float64) * div   # luma units for the bounds check
            ok = front & (u >= -0.5) & (u <= c.w - 0.5) & (v >= -0.5) & (v <= c.h - 0.5)
        else:
            u, v, ok = c.project(rays)
        cos = rays @ c.axis
        take = (ok | (len(use) == 1)) & (cos > best)
        best, cam = np.where(take, cos, best), np.where(take, i, cam)
        X, Y = np.where(take, u / div, X), np.where(take, v / div, Y)
    out_x, out_y = np.zeros(cam.shape, np.int64), np.zeros(cam.shape, np.int64)
    for i in use:
        c = belief[i]
        m = cam == i
        out_x[m] = np.clip(np.rint(X[m]), 0, int(c.w // div) - 1)
        out_y[m] = np.clip(np.rint(Y[m]), 0, int(c.h // div) - 1)
    return cam, out_x, out_y


def ground_shift(rays, dh, h0=NOMINAL_HEIGHT_M):
    """Rays of a camera raised by dh -> rays of the original camera to the same road point (ray unchanged above
    the horizon)."""
    if dh == 0:
        return rays
    rz = rays[..., 2]
    g = rz > 1e-4
    X = rays * np.where(g, (h0 + dh) / np.where(g, rz, 1.0), 0.0)[..., None]
    X[..., 2] -= dh
    X = np.where(g[..., None], X, rays)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


@dataclass
class Part:
    """One (source, prefilter scale) slice of a model frame: pixels `mask` take their value from the source plane
    (area-downscaled by `scale`) at float coordinates (mx, my); nearest when the stage-1 map is the identity."""
    src: str
    scale: float
    mask: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    nearest: bool


def frame_parts(kind, rig: Rig, sources: dict, chroma):
    """Stage 2 then stage 1 for one model frame and one plane type. sources: {'fcam': Cam, 'ecam': Cam}."""
    cam_idx, rx, ry = adapter_lookup(kind, rig.belief, chroma, (rig.use or {}).get(kind))
    src_name = rig.src[kind]
    src = sources[src_name]
    parts = []
    for i, true in enumerate(rig.true):
        m = cam_idx == i
        if not m.any():
            continue
        if true is src and rig.dh == 0:  # the rig camera is the comma camera itself: exact integer lookup
            parts.append(Part(src_name, 1.0, m, rx.astype(np.float32), ry.astype(np.float32), True))
            continue
        # rig pixel centre in luma coordinates -> physical ray -> source luma coordinates
        if chroma:
            u, v = 2.0 * rx + 0.5, 2.0 * ry + 0.5
        else:
            u, v = rx.astype(np.float64), ry.astype(np.float64)
        rays = ground_shift(true.rays(u, v), rig.dh)
        left = m
        # rays the preferred comma camera does not see (a pitched rig, the corners of a wide one) come from the other
        for name in (src_name, "ecam" if src_name == "fcam" else "fcam"):
            s = sources[name]
            su, sv, ok = s.project(rays)
            scale = min(1.0, true.f / s.f)
            if chroma:
                su, sv = (su - 0.5) / 2, (sv - 0.5) / 2
            su, sv = (su + 0.5) * scale - 0.5, (sv + 0.5) * scale - 0.5
            if (left & ok).any():
                parts.append(Part(name, scale, left & ok, su.astype(np.float32), sv.astype(np.float32), False))
            left = left & ~ok
    return parts


class Renderer:
    """Precomputed maps of one rig for one comma segment; __call__ renders packed (2, 6, 128, 256) model frames
    from the decoded source planes {'fcam': (Y, U, V), 'ecam': (Y, U, V)}."""

    def __init__(self, rig: Rig, sources: dict):
        self.parts = {(k, c): frame_parts(k, rig, sources, c) for k in ("road", "wide") for c in (False, True)}
        self.scales = sorted({(p.src, p.scale) for ps in self.parts.values() for p in ps})

    def __call__(self, planes, cache=None):
        import cv2
        cache = {} if cache is None else cache  # (src, scale) -> prefiltered planes, shared across rigs
        for s in self.scales:
            if s not in cache:
                name, scale = s
                Y, U, V = planes[name]
                if scale == 1.0:
                    cache[s] = (Y, U, V)
                else:
                    rs = lambda a: cv2.resize(a, (max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale))),  # noqa: E731
                                              interpolation=cv2.INTER_AREA)
                    cache[s] = (rs(Y), rs(U), rs(V))
        out = np.empty((2, 6, MODEL_H // 2, MODEL_W // 2), np.uint8)
        for j, kind in enumerate(("road", "wide")):
            Y = np.full((MODEL_H, MODEL_W), 16, np.uint8)
            U = np.full((MODEL_H // 2, MODEL_W // 2), 128, np.uint8)
            V = U.copy()
            for p in self.parts[(kind, False)]:
                Y[p.mask] = _sample(cache[(p.src, p.scale)][0], p)
            for p in self.parts[(kind, True)]:
                pl = cache[(p.src, p.scale)]
                U[p.mask], V[p.mask] = _sample(pl[1], p), _sample(pl[2], p)
            o = out[j]
            o[0], o[1], o[2], o[3] = Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]
            o[4], o[5] = U, V
        return out


def _sample(plane, p: Part):
    if p.nearest:
        return plane[p.my[p.mask].astype(np.int64), p.mx[p.mask].astype(np.int64)]
    import cv2
    r = cv2.remap(plane, p.mx, p.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return r[p.mask]
