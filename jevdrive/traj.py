"""Trajectory targets, the anchor vocabulary and the trajectory metrics.

Target: the ego path over the next HORIZON seconds sampled at RATE Hz, in the ego frame of the current
keyframe -- origin at the rear axle (that is where nuScenes puts its ego_pose), x forward, y left, metres.
Waymo E2E asks for its 20 waypoints in exactly that frame and unit, so only HORIZON and RATE change between
the two datasets: nuScenes here is 3 s at 2 Hz (6 waypoints), Waymo E2E is 5 s at 4 Hz (20 waypoints).
The first waypoint is at t + 1/RATE, as in the Waymo submission format. Keyframes whose scene does not
contain the whole window are dropped.

Vocabulary: k-means over the training-split targets flattened to 2T metres, plain squared Euclidean distance,
no normalisation -- the space is already metric and isotropic, and the squared Euclidean distance between two
flattened trajectories is T times their mean square displacement, so a cluster minimises exactly the
displacement error its anchor will be scored on. Fitted on training scenes only, so val leaks nowhere.
"""
import numpy as np
import pandas as pd
import torch

from .common import get_logger, processed_dir
from .waymo import _rater_frames  # the trust-region geometry, ported bit-exactly from the official RFS

log = get_logger(__name__)
HORIZON, RATE, VEL_DT = 3.0, 2.0, 0.5  # s, Hz, s (window the current velocity and yaw rate are read over)
LON_MULT, SPEED_REF = 4.0, (1.4, 11.0)  # longitudinal / lateral threshold ratio, and the speed scale's ends
DEV = "cuda"


def n_waypoints(horizon: float = HORIZON, rate: float = RATE) -> int:
    return int(round(horizon * rate))


def _scene_windows(kf: pd.DataFrame, traj: pd.DataFrame, horizon: float, rate: float, vel_dt: float) -> dict:
    """Per keyframe of one scene: future waypoints, current velocity and yaw rate, all in its own ego frame."""
    tt, t0 = traj.timestamp.to_numpy() * 1e-6, kf.timestamp.to_numpy() * 1e-6
    offsets = np.concatenate([[-vel_dt, 0.0], np.arange(1, n_waypoints(horizon, rate) + 1) / rate])
    ts = t0[:, None] + offsets
    x, y, yaw = (np.interp(ts, tt, traj[c].to_numpy()) for c in ("x", "y", "yaw"))
    c, s = np.cos(yaw[:, 1]), np.sin(yaw[:, 1])  # heading now; yaw is already unwrapped per scene
    dx, dy = x - x[:, 1:2], y - y[:, 1:2]
    p = np.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy], -1)
    return {"sample_token": kf.sample_token.to_numpy(), "future": p[:, 2:].astype(np.float32),
            "vel": (-p[:, 0] / vel_dt).astype(np.float32),  # p[:, 0] is where we were vel_dt ago, in this frame
            "yaw_rate": ((yaw[:, 1] - yaw[:, 0]) / vel_dt).astype(np.float32),
            "ok": (ts[:, 0] >= tt[0] - 1e-3) & (ts[:, -1] <= tt[-1] + 1e-3)}


def build(version: str, horizon: float = HORIZON, rate: float = RATE, vel_dt: float = VEL_DT):
    """Labelled keyframes (labels.py: 11-dim past ego state, full past) joined with their future trajectory.
    Returns (rows, future (n, T, 2), velocity (n, 2), yaw rate (n,)), all aligned and filtered."""
    d = processed_dir(version)
    kf, traj = pd.read_parquet(d / "keyframes.parquet"), pd.read_parquet(d / "ego_traj.parquet")
    lab = pd.read_parquet(d / "labels.parquet")
    by_scene = dict(tuple(traj.groupby("scene")))
    parts = [_scene_windows(g, by_scene[s], horizon, rate, vel_dt) for s, g in kf.groupby("scene")]
    w = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    pos = pd.Index(w["sample_token"]).get_indexer(lab.sample_token)
    assert (pos >= 0).all(), "labelled samples must all be keyframes"
    keep = w["ok"][pos]
    lab, pos = lab[keep].reset_index(drop=True), pos[keep]
    log.info("targets: %d / %d labelled keyframes keep a full %.1f s future at %.0f Hz (%d waypoints); "
             "%d scenes (%s)", len(lab), len(keep), horizon, rate, n_waypoints(horizon, rate),
             lab.scene.nunique(), lab.split.value_counts().to_dict())
    return lab, w["future"][pos], w["vel"][pos], w["yaw_rate"][pos]


def const_velocity(vel: np.ndarray, horizon: float, rate: float) -> np.ndarray:
    """Straight line at the current velocity vector, in the current ego frame."""
    t = np.arange(1, n_waypoints(horizon, rate) + 1) / rate
    return vel[:, None, :] * t[None, :, None]


def const_turn_rate(vel: np.ndarray, yaw_rate: np.ndarray, horizon: float, rate: float) -> np.ndarray:
    """CTRV: current speed along a circular arc at the current yaw rate (straight line as yaw rate -> 0)."""
    t = np.arange(1, n_waypoints(horizon, rate) + 1) / rate
    v, w = np.linalg.norm(vel, axis=1)[:, None], yaw_rate[:, None]
    a = w * t
    small = np.abs(w) < 1e-4
    r = np.where(small, 1.0, v / np.where(small, 1.0, w))
    return np.stack([np.where(small, v * t, r * np.sin(a)), np.where(small, 0.0, r * (1 - np.cos(a)))], -1)


def kmeans(X: torch.Tensor, k: int, iters: int = 100, seed: int = 0, tol: float = 1e-5) -> torch.Tensor:
    """Lloyd's algorithm with k-means++ seeding, on the GPU. X (n, d) float32. Returns the centres (k, d)."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = len(X)
    C = torch.empty(k, X.shape[1], device=X.device, dtype=X.dtype)
    C[0] = X[torch.randint(n, (1,), generator=g, device=X.device)]
    d2 = (X - C[0]).square().sum(1)
    for i in range(1, k):  # k-means++: sample the next centre with probability proportional to its squared distance
        C[i] = X[torch.multinomial(d2.clamp_min(0) + 1e-12, 1, generator=g)]
        d2 = torch.minimum(d2, (X - C[i]).square().sum(1))
    for it in range(iters):
        v, a = torch.cdist(X, C).min(1)
        d2 = v.square()
        S = torch.zeros_like(C).index_add_(0, a, X)
        cnt = torch.zeros(k, device=X.device).index_add_(0, a, torch.ones(n, device=X.device))
        empty = (cnt == 0).nonzero().squeeze(1)
        Cn = torch.where(cnt[:, None] > 0, S / cnt.clamp_min(1)[:, None], C)
        Cn[empty] = X[d2.argsort(descending=True)[:len(empty)]]  # empty cluster: take a worst-served point
        shift = (Cn - C).square().sum(1).max()
        C = Cn
        if shift <= tol:
            break
    log.info("k-means K=%d: %d iterations, inertia %.4f m^2/waypoint", k, it + 1, float(d2.mean()) / X.shape[1] * 2)
    return C


def nearest(X: torch.Tensor, C: torch.Tensor, m: int = 1, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """The m nearest anchors of every row: (n, m) ids and (n, m) distances in the flattened space, which are
    sqrt(T) times the RMS displacement between the trajectory and the anchor."""
    ids, err = [], []
    for a in range(0, len(X), chunk):
        d = torch.cdist(X[a:a + chunk], C)
        v, i = d.topk(m, dim=1, largest=False)
        ids.append(i.cpu().numpy())
        err.append(v.cpu().numpy())
    return np.concatenate(ids), np.concatenate(err)


def soft_target(err: np.ndarray, tau: float) -> np.ndarray:
    """Hydra-MDP-style soft imitation target over the m nearest anchors: softmax(-mean displacement / tau)."""
    w = np.exp(-(err - err[:, :1]) / tau)
    return w / w.sum(1, keepdims=True)


def region_for(horizon: float, rate: float) -> tuple[tuple[float, float], ...]:
    """Check times and lateral half-widths of the trust region, (seconds, metres).

    Waymo's Rater Feedback Score checks 3 s and 5 s against +-1.0 m and +-1.8 m lateral (four times that
    longitudinally, and both shrunk to half at walking speed), and floors a prediction that is outside at
    either horizon. Those two points lie on thr(t) = 0.4 t - 0.2, which is what we evaluate at half and full
    horizon on a shorter one: nuScenes' 3 s becomes 1.5 s / 0.4 m and 3 s / 1.0 m. At a 5 s horizon this
    returns the official pair unchanged.
    On nuScenes the region is built around the single logged future, not around three rater trajectories, so
    a miss here is a stand-in for the floored fraction, not RFS."""
    ts = (3.0, 5.0) if horizon >= 5 else (horizon / 2, horizon)
    return tuple((t, 0.4 * t - 0.2) for t in ts)


def region_norm(pred: np.ndarray, ref: np.ndarray, speed: np.ndarray, rate: float, region) -> np.ndarray:
    """Largest normalised trust-region distance from each of `pred` (n, I, T, 2) to `ref` (n, T, 2), measured
    at the region's check times in `ref`'s own longitudinal / lateral frame with the official speed scaling.
    Returns (n, I); <= 1 means the prediction is inside the region, i.e. it would not be floored."""
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    if pred.ndim == 3:
        pred = pred[:, None]
    lng, lat = _rater_frames(ref[:, None])
    v = pred[:, None] - ref[:, None, None]  # (n, 1, I, T, 2)
    k = [int(round(t * rate)) - 1 for t, _ in region]
    d_lng = np.abs((lng[:, :, None] * v).sum(-1))[..., k]  # (n, 1, I, len(region))
    d_lat = np.abs((lat[:, :, None] * v).sum(-1))[..., k]
    lo, hi = SPEED_REF
    scale = np.clip(0.5 + 0.5 * (np.asarray(speed, np.float64) - lo) / (hi - lo), 0.5, 1.0)[:, None]
    lat_thr = scale * np.array([w for _, w in region])
    return np.maximum(d_lng / (lat_thr * LON_MULT)[:, None, None], d_lat / lat_thr[:, None, None]).max(-1)[:, 0]


def region_metrics(preds: np.ndarray, gt: np.ndarray, speed: np.ndarray, rate: float, region,
                   ks=(1, 5, 10)) -> dict[str, np.ndarray]:
    """Per sample: 1.0 when every one of the top-k predictions falls outside the trust region. `miss` (k = 1)
    is what the prediction we would actually submit costs; on Waymo this is the floored fraction."""
    norm = region_norm(preds, gt, speed, rate, region)
    kmax = preds.shape[1]
    return {("miss" if k == 1 else f"miss{k}"): (norm[:, :min(k, kmax)].min(1) > 1.0).astype(float) for k in ks}


def vocab_coverage(anchors: torch.Tensor, gt: np.ndarray, speed: np.ndarray, rate: float, region,
                   budget: int = 2**21) -> np.ndarray:
    """Per sample: 1.0 when no anchor in the whole vocabulary lands inside the trust region. This is the
    coverage floor in the metric's own terms, next to the oracle minADE."""
    A = anchors.reshape(len(anchors), -1, 2).cpu().numpy().astype(np.float64)
    chunk, out = max(8, budget // len(A)), []
    for a in range(0, len(gt), chunk):
        g = gt[a:a + chunk]
        out.append(region_norm(np.broadcast_to(A, (len(g), *A.shape)), g, speed[a:a + chunk], rate, region).min(1))
    return (np.concatenate(out) > 1.0).astype(float)


def sample_metrics(pred: np.ndarray, gt: np.ndarray, rate: float) -> dict[str, np.ndarray]:
    """Per-sample displacement metrics of a single prediction: ADE, FDE and both at every whole second."""
    e = np.linalg.norm(pred - gt, axis=-1)  # (n, T)
    out = {"ade": e.mean(1), "fde": e[:, -1]}
    for s in range(1, int(gt.shape[1] / rate) + 1):
        j = int(round(s * rate))
        out[f"ade@{s}s"], out[f"fde@{s}s"] = e[:, :j].mean(1), e[:, j - 1]
    return out


def min_metrics(preds: np.ndarray, gt: np.ndarray, ks=(1, 5, 10)) -> dict[str, np.ndarray]:
    """minADE / minFDE over the first k of `preds` (n, kmax, T, 2), which must be ordered by score.
    Each is minimised on its own, as in the motion-forecasting literature."""
    e = np.linalg.norm(preds - gt[:, None], axis=-1)  # (n, kmax, T)
    ade, fde = e.mean(2), e[:, :, -1]
    kmax = preds.shape[1]
    return {f"{n}{k}": v[:, :min(k, kmax)].min(1) for k in ks for n, v in (("minade", ade), ("minfde", fde))}


def oracle_metrics(anchors: torch.Tensor, gt: np.ndarray) -> dict[str, np.ndarray]:
    """Best anchor in the whole vocabulary, per sample: the coverage floor of any scorer on this vocabulary.
    ADE and FDE are each minimised on their own, as minADE_K and minFDE_K over the vocabulary."""
    A = anchors.reshape(len(anchors), -1, 2)
    G = torch.as_tensor(gt, device=A.device)
    chunk = max(16, int(2**23 // (len(A) * A.shape[1])))
    ade, fde = [], []
    for a in range(0, len(G), chunk):
        e = (G[a:a + chunk, None] - A).norm(dim=-1)  # (c, K, T)
        ade.append(e.mean(2).amin(1).cpu().numpy())
        fde.append(e[:, :, -1].amin(1).cpu().numpy())
    return {"oracle_ade": np.concatenate(ade), "oracle_fde": np.concatenate(fde)}


def boot_ci(v: np.ndarray, scenes: np.ndarray, b: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile confidence interval of mean(v), resampling whole scenes with replacement."""
    codes, uniq = pd.factorize(scenes)
    s, n = np.bincount(codes, v, len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(len(uniq), size=(b, len(uniq)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))
