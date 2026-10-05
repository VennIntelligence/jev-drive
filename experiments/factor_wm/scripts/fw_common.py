"""factor_wm stage 1 engine pieces (plans/2026-10-05-stage1-prereg.md, section 3.1): clip sets, chained logged poses, the depth warp with
wide fill, and the closed-loop ego with openpilot's lateral and longitudinal paths.

Clip = NF packed openpilot model frames (2, 6, 128, 256) on the 5 Hz lattice: frames 0..NH-1 the logged history (frame T0 = t0), then KL logged
future frames. Logged poses of frames T0..T0+KL in the t0 frame (x fwd, y left, yaw left +) are chained from two rows' 5 s futures (t0 and
t0 + 4 s; op_dagger.dg_common.poses_from_fut20 for each). Per-frame metric depth (2, 128, 256) float16 (road, wide; forward distance in m,
road-plane scaled) is cached by fw_depth.py.

Engines (warp of the logged frame at the same time to the ego's offset (dx, dy, dpsi) against the logged pose at that time):
  plane   jevdrive.op_interp.warp_frame (road plane + 60 m sphere), decision 123's R
  depth   per-pixel depth by fixed-point iteration on the source depth (backward warp), plane depth as the start
  estar   depth + wide fill: road-view pixels that fall outside the source road image are taken from the source wide image
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "lib", REPO / "scripts", REPO / "experiments/op_route_ft/scripts", REPO / "experiments/op_adapt_r2/lib",
          REPO / "experiments/op_dagger/scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dg_common as DG  # noqa: E402
from jevdrive import op_interp as I  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

DT = 0.2
NH = 10
T0 = NH - 1
ENGINES = ("plane", "depth", "estar")
DX_CAP = 5.0                       # m, |dx| validity (time-synchronous source)
FAIL_DY = 1.5                      # m
FAIL_PSI = np.radians(15.0)
STALL_V = 1.6                      # m/s
D_MIN, D_MAX = 1.0, 120.0
N_ITER = 5


def root(*p) -> Path:
    d = Path(os.environ.get("FW_ROOT") or data_dir() / "runs" / "factor_wm") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


class Clips:
    def __init__(self, name: str):
        d = root("clips", name)
        self.name = name
        self.imgs = np.load(d / "imgs.npy", mmap_mode="r")
        self.depth = np.load(d / "depth.npy", mmap_mode="r") if (d / "depth.npy").exists() else None
        with np.load(d / "tab.npz", allow_pickle=True) as z:
            self.t = {k: z[k] for k in z.files}
        self.n = len(self.imgs)
        self.kl = self.imgs.shape[1] - NH

    def __len__(self):
        return self.n


# ---------------------------------------------------------------- logged poses
def compose(base, rel):
    """rel (k, 3) poses in the frame of `base` (3,) -> poses in base's parent frame."""
    xy = rel[:, :2] @ DG.rot(base[2]).T + base[:2]
    return np.c_[xy, rel[:, 2] + base[2]]


def chained_poses(fut_rows, kl: int) -> np.ndarray:
    """(kl + 1, 3) logged poses at 0, DT, .., kl DT from the 5 s futures (20, 2) of the rows at 0, 4 s, 8 s, .. (one per 20 steps)."""
    out = [np.zeros((1, 3))]
    base = np.zeros(3)
    for m, f in enumerate(fut_rows):
        k = min(20, kl - 20 * m)
        if k <= 0:
            break
        rel = DG.poses_from_fut20(f, k=k)
        P = compose(base, rel)
        out.append(P[1:])
        base = P[-1]
    return np.concatenate(out)


# ---------------------------------------------------------------- warps
_RAY = {}


def rays(view):
    if view not in _RAY:
        _RAY[view] = I._rays(view)                                     # (H, W, 3) vehicle axes, x = 1
    return _RAY[view]


def plane_lam(view, cam):
    r = rays(view)
    c = np.asarray(cam, float)
    down = -r[..., 2]
    lam = np.where(down > 1e-6, c[2] / np.maximum(down, 1e-6), np.inf)
    return np.minimum(lam, I.D_FAR / np.linalg.norm(r, axis=-1))


def _project(Ps, cam, view):
    """source vehicle-frame points (H, W, 3) -> (mx, my, forward distance) in the source view."""
    ray = Ps - np.asarray(cam, float)
    cv = ray @ I.OPENCV_TO_VEHICLE
    K = I.OP_K[view]
    z = np.where(cv[..., 2] > 0.1, cv[..., 2], np.nan)
    return K[0, 0] * cv[..., 0] / z + K[0, 2], K[1, 1] * cv[..., 1] / z + K[1, 2], z


def depth_map(dview, sview, cam, off, Dsrc):
    """Backward map dest view -> source view pixel coords for an ego offset off = (dx, dy, dpsi) against the source pose, with the source
    view's depth Dsrc (H, W) (forward distance). Fixed point: dest depth scaled by (source depth at the hit) / (implied source depth)."""
    import cv2
    r = rays(dview)
    c = np.asarray(cam, float)
    R = DG.rot(off[2])
    lam = plane_lam(dview, cam)
    for _ in range(N_ITER):
        P = c + lam[..., None] * r
        Ps = np.concatenate([P[..., :2] @ R.T + np.asarray(off[:2]), P[..., 2:]], -1)
        mx, my, z = _project(Ps, cam, sview)
        ds = cv2.remap(Dsrc, np.nan_to_num(mx, nan=-1e4).astype(np.float32), np.nan_to_num(my, nan=-1e4).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ratio = np.where(np.isfinite(z) & (z > 0.1), ds / np.maximum(z, 0.1), 1.0)
        lam = np.clip(lam * np.clip(ratio, 0.5, 2.0), D_MIN, D_MAX)
    P = c + lam[..., None] * r
    Ps = np.concatenate([P[..., :2] @ R.T + np.asarray(off[:2]), P[..., 2:]], -1)
    mx, my, _ = _project(Ps, cam, sview)
    return np.nan_to_num(mx, nan=-1e4).astype(np.float32), np.nan_to_num(my, nan=-1e4).astype(np.float32)


def _remap_yuv(src_packed_view, mx, my):
    import cv2
    Y, U, V = I.unpack(src_packed_view)
    Yw = cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
    Uw = cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Vw = cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Yw, Uw, Vw


def upd(d):
    """cached depth (128, 256) -> (256, 512) float32"""
    import cv2
    return cv2.resize(np.asarray(d, np.float32), (I.W, I.H), interpolation=cv2.INTER_LINEAR)


def warp(engine: str, f: np.ndarray, D, cam, off) -> np.ndarray:
    """Packed frame f (2, 6, 128, 256) taken at the logged pose, seen from the ego offset off. D: (2, 128, 256) depth or None (plane)."""
    off = np.asarray(off, float)
    if not np.any(np.abs(off) > 1e-9):
        return np.asarray(f)
    if engine == "plane":
        return I.warp_frame(np.asarray(f), np.asarray(cam, float), off, np.zeros(3))
    f = np.asarray(f)
    Dr, Dw = upd(D[0]), upd(D[1])
    out = np.empty_like(f)
    for k, (view, Ds) in enumerate((("road", Dr), ("wide", Dw))):
        mx, my = depth_map(view, view, cam, off, Ds)
        Yw, Uw, Vw = _remap_yuv(f[k], mx, my)
        if engine == "estar" and view == "road":
            outside = (mx < 0) | (mx > I.W - 1) | (my < 0) | (my > I.H - 1)
            if outside.any():
                wx, wy = depth_map("road", "wide", cam, off, Dw)
                Y2, U2, V2 = _remap_yuv(f[1], wx, wy)
                Yw = np.where(outside, Y2, Yw)
                oc = outside.reshape(I.H // 2, 2, I.W // 2, 2).any(axis=(1, 3))
                Uw, Vw = np.where(oc, U2, Uw), np.where(oc, V2, Vw)
        out[k] = I.pack(Yw, Uw, Vw)
    return out


def outside_frac(view_src, cam, off, D=None) -> float:
    """Share of road-view pixels whose source falls outside the source road image (plane geometry)."""
    mx, my = I.warp_map("road", np.asarray(cam, float), np.asarray(off, float), np.zeros(3))
    return float(np.mean((mx < 0) | (mx > I.W - 1) | (my < 0) | (my > I.H - 1)))


# ---------------------------------------------------------------- closed-loop ego (time-synchronous)
class Ego:
    """kind: replay (the log) | closed (action curvature -> OpLateral, action acceleration -> OpLongitudinal) | closedlat (lateral closed,
    speed = the log's, decision 132 style). exo (kl + 1,) exogenous heading offset (rad, left +). Offsets against the logged pose at the same time."""

    def __init__(self, pose_log, v_log, kind, exo=None, k0=0.0, a0=0.0, cat=""):
        from op_ctrl import OpLateral, OpLongitudinal
        self.P, self.vl, self.kind, self.cat = pose_log, v_log, kind, cat
        self.kl = len(pose_log) - 1
        self.exo = np.zeros(self.kl + 1) if exo is None else np.asarray(exo, float)
        self.p, self.th, self.v = np.zeros(2), 0.0, float(v_log[T0])
        self.lat = OpLateral({"delay": DG.LAT_DELAY})
        self.lat.sync(k0)
        self.lon = OpLongitudinal()
        self.lon.act = self.lon.last_out = self.lon.prev_cmd = self.lon.real = float(a0)
        self.lon.buf = [float(a0)] * self.lon.n_delay
        self.trace = [(0.0, 0.0, 0.0)]
        self.vs = [self.v]
        self.event, self.t_event = None, None
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pose_log[:, :2], axis=0), axis=1))]
        self.s_log = s

    def advance(self, j, kappa_model, acc_model):
        P = self.P
        if self.kind == "replay" or self.event is not None:
            if self.kind == "replay":
                self.p, self.th, self.v = P[j, :2].copy(), float(P[j, 2]), float(self.vl[T0 + j])
            off = self.trace[-1] if self.event is not None else (0.0, 0.0, 0.0)
            self.trace.append(off)
            self.vs.append(self.v)
            return off
        km, _ = self.lat.step(float(kappa_model), self.v, DT)
        if self.kind == "closed":
            a, _ = self.lon.step(float(acc_model), self.v, DT)
            v_new = max(0.0, self.v + a * DT)
        else:
            v_new = float(self.vl[T0 + j])
        ds = 0.5 * (self.v + v_new) * DT
        dth = -km * ds + (self.exo[j] - self.exo[j - 1])
        mid = self.th + 0.5 * dth
        self.p = self.p + ds * np.array([np.cos(mid), np.sin(mid)])
        self.th += dth
        self.v = v_new
        dxy = DG.rot(-P[j, 2]) @ (self.p - P[j, :2])
        off = (float(dxy[0]), float(dxy[1]), float(np.arctan2(np.sin(self.th - P[j, 2]), np.cos(self.th - P[j, 2]))))
        self.trace.append(off)
        self.vs.append(self.v)
        self._check(j, off)
        return off

    def _check(self, j, off):
        dx, dy, dpsi = off
        ev = None
        vmax4 = max(self.vs[: min(len(self.vs), 21)])
        if abs(dpsi) > FAIL_PSI:
            ev = "heading"
        elif abs(dy) > FAIL_DY:
            ev = "lane"
        elif self.cat == "launch" and j >= 20 and self.s_log[20] > 3.0 and vmax4 < STALL_V:
            ev = "stall"
        elif dx < -DX_CAP:
            ev = "stall" if self.v < STALL_V else "behind"
        elif dx > DX_CAP:
            ev = "ahead"
        if ev:
            self.event, self.t_event = ev, j * DT

    def clamp_off(self):
        dx, dy, dpsi = self.trace[-1]
        return (float(np.clip(dx, -DX_CAP, DX_CAP)), float(np.clip(dy, -FAIL_DY, FAIL_DY)), float(np.clip(dpsi, -FAIL_PSI, FAIL_PSI)))


def kick(kl, deg):
    e = np.full(kl + 1, np.radians(deg))
    e[0] = 0.0
    return e


def swerve(kl, off, v, m=3):
    """op_dagger's swerve (3-step heading pulse that leaves the ego `off` m beside the path), None when infeasible (> 12 deg)."""
    path = float(np.sum(v[T0: T0 + m]) * DT)
    sb = abs(off) / max(path, 1e-6)
    if sb > np.sin(np.radians(12.0)):
        return None
    e = np.zeros(kl + 1)
    e[1: m + 1] = np.sign(off) * np.arcsin(sb)
    return e
