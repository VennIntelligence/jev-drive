"""Feeding openpilot a benchmark's low-rate camera history (research/openpilot-openloop-integration.md).

openpilot's driving models step a 20 Hz clock and read ego motion from frames 0.2 s apart; NAVSIM gives an agent four
frames at 2 Hz. This module turns the four keyframes (t = -1.5, -1.0, -0.5, 0 s) into a 10 Hz history (the exam's
`ctx1.5` feed: t = -1.5 ... 0, each frame shown for two 20 Hz steps), working on openpilot's own packed YUV420 model
frames (2 views [road, wide] x (6, 128, 256) uint8), so one code path serves every dataset once its keyframes are
rendered. Every method sees only what a benchmark agent is given: the four keyframes and the 2 Hz ego history.

  hold   sample-and-hold, the exam's NAVSIM feed
  blend  linear cross-fade of the two neighbouring keyframes
  warp   the nearest keyframe re-projected to the ego pose at t (cubic Hermite of the 2 Hz poses and velocities) through
         the road plane at the camera height; static world, rays that miss the road within 60 m are put at 60 m
  rife   Practical-RIFE 4.26 (arbitrary t) on RGB model frames
  gimm   GIMM-VFI-R (NeurIPS 2024, continuous t) on RGB model frames

`preroll` extends the history before -1.5 s with the -1.5 s keyframe warped along the ego track extrapolated backwards
at constant velocity (a warm-up for the recurrent state that uses no extra frame).

Output adapters (plan -> benchmark trajectory): camera -> rear axle (the correct lever-arm transform, and the two classic
mistakes), linear or cubic time resampling, and `retime`: the plan re-timed so its initial speed equals the benchmark's
measured ego speed (openpilot has no speed input; on the car the planner starts from CAN speed).

Imported by the openpilot venv (numpy / scipy / cv2 only at import time) and the VFI venv (torch inside functions).
"""
import numpy as np

T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])
T10 = np.round(np.arange(-15, 1) * 0.1, 3)
W, H = 512, 256
OP_K = {"road": np.array([[910.0, 0, 256.0], [0, 910.0, 47.6], [0, 0, 1]]),
        "wide": np.array([[455.0, 0, 256.0], [0, 455.0, 0.5 * (256 + 47.6)], [0, 0, 1]])}
OPENCV_TO_VEHICLE = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])     # columns: cam x, y, z in vehicle axes
D_FAR = 60.0
METHODS = ("hold", "blend", "warp", "rife", "gimm")
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])      # openpilot plan times (openpilot.model.T_IDXS)


def grid(t0: float, dt: float = 0.1) -> np.ndarray:
    """History times t0 ... 0 at dt, oldest first."""
    return np.round(np.arange(round(t0 / dt), 1) * dt, 3)


# ---------------------------------------------------------------- packed YUV420 model frames

def unpack(p):
    """(..., 6, 128, 256) packed -> Y (..., 256, 512), U, V (..., 128, 256); works for numpy and torch."""
    Y = p[..., 0, :, :].new_empty(p.shape[:-3] + (H, W)) if hasattr(p, "new_empty") else np.empty(p.shape[:-3] + (H, W), p.dtype)
    Y[..., 0::2, 0::2], Y[..., 1::2, 0::2], Y[..., 0::2, 1::2], Y[..., 1::2, 1::2] = p[..., 0, :, :], p[..., 1, :, :], p[..., 2, :, :], p[..., 3, :, :]
    return Y, p[..., 4, :, :], p[..., 5, :, :]


def pack(Y, U, V):
    xp = np if isinstance(Y, np.ndarray) else __import__("torch")
    return xp.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2], Y[..., 0::2, 1::2], Y[..., 1::2, 1::2], U, V], -3)


def to_rgb(p):
    """torch uint8 (..., 6, 128, 256) -> float RGB (..., 3, 256, 512) in [0, 1]: JPEG YCbCr (BT.601 full range),
    chroma upsampled by 2x2 replication."""
    import torch
    Y, U, V = (x.float() for x in unpack(p))
    U = U.repeat_interleave(2, -2).repeat_interleave(2, -1) - 128
    V = V.repeat_interleave(2, -2).repeat_interleave(2, -1) - 128
    rgb = torch.stack([Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U], -3)
    return (rgb / 255).clamp(0, 1)


def from_rgb(rgb):
    """float RGB (..., 3, 256, 512) in [0, 1] -> packed uint8, chroma as the 2x2 mean (as the renderers)."""
    import torch
    r, g, b = (255 * rgb).unbind(-3)
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    U = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    V = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    half = lambda c: c.unflatten(-2, (H // 2, 2)).unflatten(-1, (W // 2, 2)).mean((-3, -1))  # noqa: E731
    q = lambda c: c.round().clamp(0, 255).to(torch.uint8)  # noqa: E731
    return pack(q(Y), q(half(U)), q(half(V)))


# ---------------------------------------------------------------- ego track from the 2 Hz history

class EgoTrack:
    """Ego pose (x, y, yaw) in the t0 frame at any t, from 2 Hz samples: cubic Hermite in x/y with the measured
    velocities, cubic spline in yaw (as navsim_zs.alpamayo_history). Outside the samples: constant velocity and yaw
    rate from the first / last sample."""

    def __init__(self, t, xy, yaw, v_world):
        from scipy.interpolate import CubicHermiteSpline, CubicSpline
        self.t, self.xy, self.v = np.asarray(t, float), np.asarray(xy, float), np.asarray(v_world, float)
        self.yaw = np.unwrap(np.asarray(yaw, float))
        self._xy = CubicHermiteSpline(self.t, self.xy, self.v, axis=0)
        self._yaw = CubicSpline(self.t, self.yaw)
        self._w0 = (self.yaw[1] - self.yaw[0]) / (self.t[1] - self.t[0])

    def __call__(self, t: float) -> np.ndarray:
        if t < self.t[0]:            # backwards at constant speed along the first heading, constant yaw rate
            dt = t - self.t[0]
            yaw = self.yaw[0] + self._w0 * dt
            return np.r_[self.xy[0] + self.v[0] * dt, yaw]
        return np.r_[self._xy(t), self._yaw(t)]


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def track_navsim(pose, vel) -> EgoTrack:
    """NAVSIM ego history: poses (4, 3) x, y, yaw in the t0 rear-axle frame, body-frame velocities (4, 2)."""
    pose, vel = np.asarray(pose, float), np.asarray(vel, float)
    vw = np.stack([rot2(pose[k, 2]) @ vel[k] for k in range(4)])
    return EgoTrack(T_KEY, pose[:, :2], pose[:, 2], vw)


def track_wod(past) -> EgoTrack:
    """WOD past_states (16, 6) @4 Hz (x, y, vx, vy, ax, ay in the t0 frame), subsampled to NAVSIM's 2 Hz contract
    (t = -1.5, -1.0, -0.5, 0); yaw from the velocity direction (nearest moving sample when slower than 1 m/s)."""
    past = np.asarray(past, float)[[9, 11, 13, 15]]
    v = past[:, 2:4]
    mov = np.linalg.norm(v, axis=1) >= 1.0
    if mov.any():
        yaw = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        mi = np.flatnonzero(mov)
        yaw = yaw[mi[np.abs(np.arange(4)[:, None] - mi[None]).argmin(1)]]
        yaw = yaw - yaw[-1]
    else:
        yaw = np.zeros(4)
    return EgoTrack(T_KEY, past[:, :2], yaw, v)


# ---------------------------------------------------------------- ego-motion warp through the road plane

_RAYS = {}


def _rays(view: str) -> np.ndarray:
    """(H, W, 3) vehicle-frame rays of the model frame's pixel centres (openpilot calib = vehicle axes)."""
    if view not in _RAYS:
        K = OP_K[view]
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ OPENCV_TO_VEHICLE.T
        _RAYS[view] = r
    return _RAYS[view]


def warp_map(view: str, cam, pose_dst, pose_src):
    """Source-image pixel coordinates (map_x, map_y) (H, W) for every destination pixel: the destination ray hits the
    road plane z = 0 (camera at height cam[2]) or the 60 m sphere, moves with the ego from pose_dst to pose_src."""
    r = _rays(view)
    c = np.asarray(cam, float)
    down = -r[..., 2]
    lam = np.where(down > 1e-6, c[2] / np.maximum(down, 1e-6), np.inf)
    lam = np.minimum(lam, D_FAR / np.linalg.norm(r, axis=-1))
    P = c + lam[..., None] * r                                               # vehicle frame at t_dst
    Pw = P[..., :2] @ rot2(pose_dst[2]).T + pose_dst[:2]
    Ps = (Pw - pose_src[:2]) @ rot2(pose_src[2])                             # R^T (Pw - p)
    ray = np.concatenate([Ps, P[..., 2:]], -1) - c
    cv = ray @ OPENCV_TO_VEHICLE                                             # vehicle -> opencv
    K = OP_K[view]
    z = np.where(cv[..., 2] > 0.1, cv[..., 2], np.nan)
    mx = K[0, 0] * cv[..., 0] / z + K[0, 2]
    my = K[1, 1] * cv[..., 1] / z + K[1, 2]
    return np.nan_to_num(mx, nan=-1e4).astype(np.float32), np.nan_to_num(my, nan=-1e4).astype(np.float32)


def warp_frame(packed_src, cam, pose_dst, pose_src):
    """One packed frame (2, 6, 128, 256) re-projected from pose_src to pose_dst (both views), bilinear, border replicate."""
    import cv2
    out = np.empty_like(packed_src)
    for k, view in enumerate(("road", "wide")):
        mx, my = warp_map(view, cam, pose_dst, pose_src)
        Y, U, V = unpack(packed_src[k])
        Yw = cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # chroma site (i, j) = centre of its 2x2 luma block; its source position in chroma pixel units
        hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
        Uw = cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Vw = cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[k] = pack(Yw, Uw, Vw)
    return out


# ---------------------------------------------------------------- history synthesis

def neighbours(t: float):
    """(i0, i1, s): keyframe indices around t and the fraction s in [0, 1] from i0 to i1."""
    i1 = int(np.searchsorted(T_KEY, t - 1e-9))
    if i1 == 0:
        return 0, 0, 0.0
    i0 = i1 - 1
    return i0, i1, float((t - T_KEY[i0]) / (T_KEY[i1] - T_KEY[i0]))


def synth_cpu(keys: np.ndarray, method: str, times: np.ndarray, track: EgoTrack | None = None, cam=None) -> np.ndarray:
    """keys (4, 2, 6, 128, 256) -> (len(times), 2, 6, 128, 256) for hold / blend / warp; times before -1.5 s (preroll)
    are the -1.5 s keyframe warped along the backwards-extrapolated track (held for hold / blend)."""
    out = np.empty((len(times),) + keys.shape[1:], np.uint8)
    for j, t in enumerate(times):
        k = np.flatnonzero(np.isclose(T_KEY, t))
        if len(k):
            out[j] = keys[k[0]]
            continue
        if t < T_KEY[0]:
            out[j] = keys[0] if method in ("hold", "blend") else warp_frame(keys[0], cam, track(t), track(T_KEY[0]))
            continue
        i0, i1, s = neighbours(t)
        if method == "hold":
            out[j] = keys[i0]
        elif method == "blend":
            out[j] = np.rint((1 - s) * keys[i0].astype(np.float32) + s * keys[i1].astype(np.float32)).astype(np.uint8)
        elif method == "warp":
            src = i0 if s <= 0.5 else i1
            out[j] = warp_frame(keys[src], cam, track(t), track(T_KEY[src]))
        else:
            raise ValueError(method)
    return out


ALIGN = ("none", "rot0", "straight")


def align_history(keys, syn, syn_t, track: EgoTrack, cam, rule: str):
    """Reference-free input rule on the history frames (experiments/skill_pack, history alignment): uses only the ego
    history the benchmark gives. keys (4, ...) at T_KEY, syn (len(syn_t), ...) context frames; the t0 key is never changed.
      rot0      every history frame re-projected from its pose (x, y, yaw) to (x, y, 0): same position, current heading
                (rotation removed, translation kept)
      straight  every history frame replaced by the t0 key warped back along a straight track, (-s(t), 0, 0) with s the
                history's arc length to t0 (rotation and lateral path removed, speed profile kept)"""
    if rule == "none":
        return keys, syn
    T = np.linspace(-1.5, 0.0, 151)
    xy = np.array([track(t)[:2] for t in T])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

    def one(f, t):
        if rule == "rot0":
            p = track(t)
            return warp_frame(f, cam, np.r_[p[:2], 0.0], p)
        s = arc[-1] - np.interp(t, T, arc)
        return warp_frame(keys[3], cam, np.array([-s, 0.0, 0.0]), np.zeros(3))

    k2 = np.array(keys)
    for j in range(3):
        k2[j] = one(keys[j], T_KEY[j])
    s2 = np.stack([one(syn[j], t) for j, t in enumerate(syn_t)])
    return k2, s2


class RIFE:
    """Practical-RIFE 4.26 (hzwer, MIT), weights from HF hzwer/RIFE; batched, arbitrary t."""

    def __init__(self, root, device="cuda"):
        import importlib.util, sys, torch, types
        from pathlib import Path
        root = Path(root)
        sys.modules.setdefault("model", types.ModuleType("model"))
        spec = importlib.util.spec_from_file_location("model.warplayer", root / "Practical-RIFE/model/warplayer.py")
        wl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wl)
        sys.modules["model.warplayer"] = wl
        spec = importlib.util.spec_from_file_location("rife_ifnet", root / "rife_w/RIFEv4.26_0921/IFNet_HDv3.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.net = m.IFNet().to(device).eval()
        sd = torch.load(root / "rife_w/RIFEv4.26_0921/flownet.pkl", map_location=device)
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        self.device = device

    def __call__(self, i0, i1, ss):
        """i0, i1 (B, 3, H, W) float in [0, 1]; ss: fractions in (0, 1) -> [(B, 3, H, W)] per fraction."""
        import torch
        out = []
        with torch.no_grad():
            x = torch.cat([i0, i1], 1)
            for s in ss:
                _, _, merged = self.net(x, torch.full((len(x), 1, 1, 1), s).to(x), [16, 8, 4, 2, 1])
                out.append(merged[-1].clamp(0, 1))
        return out


class GIMM:
    """GIMM-VFI-R with the LPIPS-finetuned checkpoint (S-Lab licence, research use), weights from HF GSean/GIMM-VFI."""

    def __init__(self, root, device="cuda"):
        import os, sys, torch
        from pathlib import Path
        from omegaconf import OmegaConf
        root = Path(root)
        repo = root / "GIMM-VFI"
        ck = repo / "pretrained_ckpt"
        if not ck.exists():
            ck.symlink_to(root / "gimm_w")
        sys.path.insert(0, str(repo / "src"))
        os.chdir(repo)                                   # RAFT loads pretrained_ckpt/raft-things.pth relative to cwd
        from models import create_model
        from utils.config import augment_arch_defaults, load_config
        cfg = load_config(repo / "configs/gimmvfi/gimmvfi_r_arb.yaml")
        self.net, _ = create_model(augment_arch_defaults(cfg.arch))
        sd = torch.load(ck / "gimmvfi_r_arb_lpips.pt", map_location="cpu")
        self.net.load_state_dict(sd["state_dict"], strict=True)
        self.net = self.net.to(device).eval()
        self.device = device

    def __call__(self, i0, i1, ss):
        """All fractions in one forward: the flow between the two frames is estimated once (as the repo's video_Nx)."""
        import torch
        with torch.no_grad():
            xs = torch.stack([i0, i1], 2)
            B, hw = xs.shape[0], xs.shape[-2:]
            coord = [(self.net.sample_coord_input(B, hw, [s], device=xs.device, upsample_ratio=1.0), None) for s in ss]
            out = self.net(xs, coord, t=[torch.full((B,), s, device=xs.device) for s in ss], ds_factor=1.0)
        return [o.clamp(0, 1) for o in out["imgt_pred"]]


def synth_vfi(keys, model, times, batch: int = 64, device="cuda") -> np.ndarray:
    """keys (N, 4, 2, 6, 128, 256) uint8 -> (N, len(times), 2, 6, 128, 256) with a learned interpolator between the two
    neighbouring keyframes of every non-key time (times must lie in [-1.5, 0])."""
    import torch
    N = keys.shape[0]
    out = np.empty((N, len(times)) + keys.shape[2:], np.uint8)
    kt = torch.from_numpy(np.ascontiguousarray(keys)).to(device)
    gaps = {}                                        # (i0, i1) -> [(slot j, s)]
    for j, t in enumerate(times):
        k = np.flatnonzero(np.isclose(T_KEY, t))
        if len(k):
            out[:, j] = keys[:, k[0]]
        else:
            i0, i1, s = neighbours(t)
            gaps.setdefault((i0, i1), []).append((j, s))
    rgb = to_rgb(kt.flatten(0, 2)).view(N, 4, 2, 3, H, W)      # (N, 4 keys, 2 views, 3, H, W)
    for (i0, i1), js in gaps.items():
        a, b = rgb[:, i0].flatten(0, 1), rgb[:, i1].flatten(0, 1)
        res = [[] for _ in js]
        for q in range(0, a.shape[0], batch):
            for r, im in zip(res, model(a[q:q + batch], b[q:q + batch], [s for _, s in js])):
                r.append(from_rgb(im))
        for (j, _), r in zip(js, res):
            out[:, j] = torch.cat(r).view(N, 2, 6, 128, 256).cpu().numpy()
    return out


# ---------------------------------------------------------------- step schedule and output adapters

def schedule(times: np.ndarray, context_rate: bool) -> list[int]:
    """Index of the frame shown at each model step: queued models (small, Cinque) step the 20 Hz clock from the oldest
    frame time to 0, Lebowski its 5 Hz context phases ending at 0 (starting one phase after the oldest frame, as the
    exam); each step shows the latest frame at or before t (wod_openpilot_timeline.schedule, generalised)."""
    dt = 0.2 if context_rate else 0.05
    n = int(round(-times[0] / dt + 1e-6)) if not context_rate else int(np.floor(-times[0] / dt + 1e-6))
    first = 1 if context_rate and np.isclose(n * dt, -times[0]) else 0
    ts = np.round(np.arange(-n + first, 1) * dt, 3)
    return [int(np.searchsorted(times, t + 1e-6) - 1) for t in ts]


def retime(plan_pos, plan_yaw, plan_vel, t_idx, v_meas, r_lim=(0.25, 2.0), v_min=1.0):
    """Re-time the plan so its initial forward speed is the measured ego speed: p'(t) = p(r t), r = v_meas / v_plan(0).
    Left unchanged when either speed is below v_min (a stop or start the model sees and the history cannot rescale)."""
    v0 = float(plan_vel[0, 0])
    if v0 < v_min or v_meas < v_min:
        return plan_pos, plan_yaw, 1.0
    r = float(np.clip(v_meas / v0, *r_lim))
    tq = np.minimum(r * t_idx, t_idx[-1])
    pos = np.stack([np.interp(tq, t_idx, plan_pos[:, k]) for k in range(plan_pos.shape[1])], -1)
    return pos, np.interp(tq, t_idx, plan_yaw), r


def to_rear(plan_pos, plan_yaw, t_idx, dev_xy, t_out, mode: str = "lever", interp: str = "linear"):
    """openpilot plan (calib frame x fwd / y right, origin at the camera, yaw clockwise) -> rear-axle (len(t_out), 3)
    x, y (left), yaw at t_out.
      lever  rear(t) = d + p(t) - R(psi_t) d, d = the camera's (x, y) on the vehicle (navsim_zs.openpilot_to_navsim)
      none   rear(t) = p(t): camera displacement taken as the rear axle's (drops the lever arm on turns)
      shift  rear(t) = p(t) + d: the plan translated by the mounting offset (the B2D adapter bug of decision 33)"""
    p = np.stack([plan_pos[:, 0], -plan_pos[:, 1]], -1).astype(np.float64)
    psi = -np.asarray(plan_yaw, np.float64)
    d = np.asarray(dev_xy, np.float64)
    if mode == "lever":
        Rd = np.stack([np.cos(psi) * d[0] - np.sin(psi) * d[1], np.sin(psi) * d[0] + np.cos(psi) * d[1]], -1)
        xy = d + p - Rd
    elif mode == "none":
        xy = p
    elif mode == "shift":
        xy = p + d
    else:
        raise ValueError(mode)
    rear = np.concatenate([xy, psi[:, None]], 1)
    if interp == "linear":
        return np.stack([np.interp(t_out, t_idx, rear[:, k]) for k in range(3)], -1).astype(np.float32)
    from scipy.interpolate import CubicSpline
    return CubicSpline(t_idx, rear, axis=0)(t_out).astype(np.float32)
