"""op_dagger shared pieces: clip sets, logged poses, the reprojection rollout engine, recovery labels (plans/2026-10-06-dagger-prereg.md).

A clip = 20 packed openpilot model frames (2, 6, 128, 256) on the 5 Hz lattice, frames 0..9 the history (frame 9 = t0), frames 10..19 the logged
future (steps j = 1..K); the logged ego pose of frames 9..19 in the t0 frame (x fwd, y left, yaw left +), the logged speed per frame, the camera mount
(vehicle frame, z above the ground), the traffic convention, and (WOD-E2E clips) the logged 5 s future (20, 2) of frames 9..19 for the labels.

Rollout (one arm of one clip), 5 Hz steps j = 0..K:
  state j     the 10 frames j..j+9; frames after t0 are the logged frame at that time re-projected (jevdrive.op_interp.warp_frame: road plane, 60 m
              sphere) to the ego's offset (dx, dy, dpsi) against the logged pose at that time; frames up to t0 are the logged history
  model       the port (op_adapt_l LModel / op_route_ft RModel), 9 context pairs (k, k+1); outputs -> plan, phi1 (1 s plan direction, deg, left +),
              action[0] -> desired curvature = action[0] / max(1, v)^2 (openpilot sign: right +)
  control     closed: lib/op_ctrl.OpLateral (latActive, clip_curvature, lateralDelay 0.2 s), synced to the logged curvature at t0, at the logged speed;
              the ego heading offset grows by (realised - logged) heading change plus the exogenous perturbation; the step displacement is the
              logged one rotated by the mean heading offset over the step (zero offset = the log)
              open: no control, heading offset = the exogenous profile, displacement rotated by it (= world_model wm_common.perturb, decision 123)
              replay: zero offsets (the engine's identity check)
Validity (decision 123): |dy| <= CAP_Y, |dpsi| <= CAP_PSI on every step up to j, horizon <= K = 10 steps (2 s).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "lib", REPO / "scripts", REPO / "experiments/op_route_ft/scripts", REPO / "experiments/op_adapt_r2/lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jevdrive.common import data_dir  # noqa: E402

DT = 0.2
K = 10
NH = 10                      # model images per state (9 context pairs)
NF = NH + K                  # frames per clip
T0 = NH - 1                  # clip index of t0
CAP_Y = 1.0                  # m
CAP_PSI = np.radians(5.0)    # rad
LAT_DELAY = 0.2              # s, lateralDelay on real logs (model clock = wall clock)
T_FUT = 0.25 * np.arange(1, 21)


def root(*p) -> Path:
    d = Path(os.environ.get("OP_DAGGER_ROOT") or data_dir() / "runs" / "op_dagger") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- clips
class Clips:
    def __init__(self, name: str):
        d = root("clips", name)
        self.name = name
        self.imgs = np.load(d / "imgs.npy", mmap_mode="r")
        with np.load(d / "tab.npz", allow_pickle=True) as z:
            self.t = {k: z[k] for k in z.files}
        self.n = len(self.imgs)

    def __len__(self):
        return self.n


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def poses_from_fut20(fut20: np.ndarray, k: int = K, s_min: float = 0.6) -> np.ndarray:
    """Logged poses (k + 1, 3) at t = 0, DT, .., k DT in the t0 frame from the t0 frame's 5 s future (20, 2) on the 0.25 s grid: positions by
    cubic interpolation in time (P(0) = 0), heading = direction of the chord p(s - s_min / 2) -> p(s + s_min / 2) on the arc-length path
    (the ends clipped into the path); 0 while the ego has moved less than s_min / 2 and the path is shorter than s_min."""
    from scipy.interpolate import CubicSpline
    t = np.r_[0.0, T_FUT]
    P = np.r_[np.zeros((1, 2)), np.asarray(fut20, float)]
    tq = DT * np.arange(k + 1)
    xy = CubicSpline(t, P, axis=0)(tq)
    xy[0] = 0.0
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    ok = np.r_[True, seg > 1e-4]
    s_u, P_u = s[ok], P[ok]
    sq = np.interp(tq, t, s)
    psi = np.zeros(k + 1)
    if s_u[-1] >= s_min:
        for j, sj in enumerate(sq):
            a, b = np.clip([sj - s_min / 2, sj + s_min / 2], 0.0, s_u[-1])
            if b - a < 0.5 * s_min:
                a, b = (max(0.0, b - s_min), b) if b >= s_u[-1] else (a, min(s_u[-1], a + s_min))
            pa = np.array([np.interp(a, s_u, P_u[:, c]) for c in range(2)])
            pb = np.array([np.interp(b, s_u, P_u[:, c]) for c in range(2)])
            psi[j] = np.arctan2(*(pb - pa)[::-1])
        psi -= psi[0]                       # the t0 frame defines heading 0
    return np.c_[xy, psi]


def kappa_log(pose: np.ndarray, v: np.ndarray) -> float:
    """Logged curvature at t0 in openpilot's sign (right +) from the first-step heading change."""
    s = max(np.linalg.norm(pose[1, :2] - pose[0, :2]), 1e-3)
    return float(-(pose[1, 2] - pose[0, 2]) / s) if s > 0.05 else 0.0


# ---------------------------------------------------------------- warps
def warp(f: np.ndarray, cam, off) -> np.ndarray:
    """Packed frame seen from the ego offset by off = (dx, dy, dpsi) against the pose the frame was taken at (exact identity at 0)."""
    from jevdrive import op_interp as I
    off = np.asarray(off, float)
    if not np.any(np.abs(off) > 1e-9):
        return np.asarray(f)
    return I.warp_frame(np.asarray(f), np.asarray(cam, float), off, np.zeros(3))


def drift_offsets(dy: float, dpsi: float, v: float) -> np.ndarray:
    """Static O pair history (op_adapt_h drift): offsets (NH, 3) of the 10 frames ending at the state for a drift that reaches (dy, dpsi) there."""
    from experiments.op_adapt_h.lib import op_adapt_h as H
    t = np.round(-DT * np.arange(NH - 1, -1, -1), 3)
    y, p = H.drift(t, dy, dpsi, v)
    return np.c_[np.zeros(NH), y, p]


# ---------------------------------------------------------------- rollout dynamics (CPU)
class Ego:
    """One rollout's ego against the log. kind: replay | open | closed; exo (K + 1,) exogenous heading offset (rad, left +, exo[0] = 0)."""

    def __init__(self, pose_log: np.ndarray, v: np.ndarray, kind: str, exo=None, k0: float = 0.0):
        from op_ctrl import OpLateral
        self.P, self.v, self.kind = pose_log, v, kind
        self.exo = np.zeros(K + 1) if exo is None else np.asarray(exo, float)
        self.p = np.zeros(2)
        self.o = 0.0                                                  # heading offset against the log
        self.ctl = OpLateral({"delay": LAT_DELAY})
        self.ctl.sync(k0)
        self.trace = [(0.0, 0.0, 0.0)]
        self.kreal = []

    def advance(self, j: int, kappa_model: float | None):
        """Step j - 1 -> j given the model's desired curvature at state j - 1 (right +); returns the offset (dx, dy, dpsi) at j."""
        P = self.P
        d_log = P[j, :2] - P[j - 1, :2]
        s = float(np.linalg.norm(d_log))
        if self.kind == "replay":
            o_new, rot_a = 0.0, 0.0
        elif self.kind == "open":
            o_new = self.exo[j]
            rot_a = o_new
        else:
            km, _ = self.ctl.step(float(kappa_model), float(self.v[T0 + j - 1]), DT)
            self.kreal.append(km)
            o_new = self.o + (-km * s) - (P[j, 2] - P[j - 1, 2]) + (self.exo[j] - self.exo[j - 1])
            rot_a = 0.5 * (self.o + o_new)
        self.p = self.p + rot(rot_a) @ d_log if self.kind != "replay" else P[j, :2].copy()
        self.o = o_new
        dxy = rot(-P[j, 2]) @ (self.p - P[j, :2])
        off = (float(dxy[0]), float(dxy[1]), float(o_new))
        self.trace.append(off)
        return off

    def pose_t0(self) -> np.ndarray:
        """Current ego pose in the t0 frame (for the anchor-source warp of decision 123)."""
        j = len(self.trace) - 1
        return np.r_[self.p, self.P[j, 2] + self.o]


def yaw_exo(deg: float) -> np.ndarray:
    e = np.full(K + 1, np.radians(deg))
    e[0] = 0.0
    return e


def kick_exo(deg: float) -> np.ndarray:
    """Closed loop: a heading kick of deg at step 1 (exo stays, the controller takes it from there)."""
    return yaw_exo(deg)


def swerve_exo(off: float, v: np.ndarray, m: int = 3):
    """Heading +beta for steps 1..m then 0 (decision 123's lateral arm): the ego ends `off` m (left +) from the path. None when beta > 12 deg."""
    path = float(np.sum(v[T0: T0 + m]) * DT)
    sb = abs(off) / max(path, 1e-6)
    if sb > np.sin(np.radians(12.0)):
        return None
    e = np.zeros(K + 1)
    e[1: m + 1] = np.sign(off) * np.arcsin(sb)
    return e


def in_cap(trace) -> np.ndarray:
    """(K + 1,) bool: every step up to j inside the validity cap."""
    t = np.asarray(trace)
    ok = (np.abs(t[:, 1]) <= CAP_Y) & (np.abs(t[:, 2]) <= CAP_PSI)
    return np.logical_and.accumulate(ok)


# ---------------------------------------------------------------- labels
def recovery_hum(fut20: np.ndarray, dy: float, dpsi: float, v: float) -> np.ndarray:
    """(16, 3) imitation target (op_adapt_l grid) of the recovery path back onto the logged path from the offset (dy, dpsi), as layer-3 O pairs."""
    from experiments.op_adapt_h.lib import op_adapt_h as H
    from experiments.op_adapt_l.lib import op_adapt_l as L
    f = H.recover_target(fut20, dy, dpsi, v) if (abs(dy) > 1e-6 or abs(dpsi) > 1e-9) else np.asarray(fut20, np.float32)
    return L.human_targets(np.asarray(f)[None])[0]


def act_label(hum: np.ndarray, v: float):
    """Action target in the shipped head's own scale (op_route_ft rft.act_target): (target action[0], weight)."""
    import rft
    return rft.act_target(hum, v)


# ---------------------------------------------------------------- model (GPU)
def load_policy(tag: str, dev):
    """'shipped' -> the original port; else a dg_train run dir name or a ckpt path (op_route_ft RModel format, action pathway trainable)."""
    import torch
    from experiments.op_adapt_l.lib import op_adapt_l as L
    import rft
    if tag == "shipped":
        return L.load_model(None, dev)
    p = Path(tag)
    if not p.exists():
        p = root("runs", tag, "ckpt-final.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = rft.RModel(L.LCfg(tag, intent="none"), None).to(dev).eval()
    m.load_state(ck["model"])
    return m


class Heads:
    """Readouts of the output vector."""

    def __init__(self, net):
        from jevdrive import op_adapt as A
        self.pi = A.plan_index(net.slices)
        self.a0 = net.slices["action"].start
        self.T = A.T_IDXS

    def __call__(self, out: np.ndarray, v: np.ndarray) -> dict:
        plan = out[:, self.pi].reshape(-1, 33, 15)
        x1 = np.array([np.interp(1.0, self.T, p[:, 0]) for p in plan])
        y1 = np.array([np.interp(1.0, self.T, p[:, 1]) for p in plan])          # right +
        a = out[:, self.a0]
        return dict(phi1=-np.degrees(np.arctan2(y1, np.maximum(x1, 1e-3))), y1=-y1, act=a,
                    kappa=a / np.maximum(1.0, np.asarray(v, float)) ** 2, psi3=-np.degrees(np.array([np.interp(3.0, self.T, p[:, 11]) for p in plan])))
