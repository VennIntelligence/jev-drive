"""op-adapt r2, arm D: the detection-token plug-in (todos/2026-09-29-op-adapt-r2-prereg.md section 2.2; formats in
tmp/2026-09-30-op-adapt-r2-build.md, "D").

  Yolo          YOLO26x-seg 640 fp16 (decision 45's fast-lane model, Ultralytics' own fused AutoBackend and NMS) on
                native-resolution front-camera images already on the GPU: letterbox on the GPU, per image the top 8 of
                {pedestrian, cyclist, vehicle} by score, with the RoIAlign (1 x 1) of the three neck maps (decision 50's
                1920-d detection appearance)
  geometry      per camera calibration: undistort (Brown-Conrady, fixed point), then the exact rotation homography ideal
                pixel -> openpilot road model frame (focal 910, 512 x 256), fitted on the calibration's own projection
  tokens        (n, 8, TOK_D) float16 + validity mask per image / per trunk row, fields in FIELDS
  DetAdapter    one cross-attention layer on the 32 x 512 hidden tokens `view_39` of every context slot (hidden = query,
                tokens -> MLP -> 512 = key / value, 8 heads), times a scalar gate initialised to 0
  stage4_policy_det   op_adapt.stage4_policy with the adapter between stage 4 and the policy
  load_tok / flatten  token arrays aligned row for row with a trunk cache file / a list of them

Imported by envs/ultralytics (detection) and envs/op-train (tokens, adapter): torch and numpy at import time only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import data_dir

K_TOK, PCA_D, SCORE, IMGSZ = 8, 16, 0.25, 640
CLASSES = ("pedestrian", "cyclist", "vehicle")
WEIGHTS = "yolo26x-seg.pt"
FEAT_D = 384 + 768 + 768
# token fields: class one-hot | score | box (cx, cy, w, h) in the road model frame / (512, 256) | box height / focal | PCA-16
FIELDS = {"cls": slice(0, 3), "score": slice(3, 4), "box": slice(4, 8), "h_over_f": slice(8, 9), "app": slice(9, 9 + PCA_D)}
TOK_D = 9 + PCA_D
ROAD_K = np.array([[910.0, 0, 256.0], [0, 910.0, 47.6], [0, 0, 1]])
ROAD_WH = (512, 256)
BOX_CLIP = (-2.0, 3.0)                   # normalised road coordinates outside the frame are kept, clipped to this range
VIEW_TO_VEHICLE = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])   # camgeom OPENCV_TO_VEHICLE


def root(*p) -> Path:
    return data_dir().joinpath("runs", "op_adapt_r2", "det", *p)


# ================================================================ detection (envs/ultralytics)

class Yolo:
    """YOLO26x-seg on uint8 RGB (B, 3, H, W) CUDA batches. Letterbox = Ultralytics' rect LetterBox (long side 640, pad to
    a multiple of 32, grey 114) done on the GPU with bilinear resize; NMS = the predictor's own (conf 0.25, iou 0.7,
    max_det 300, class-aware over COCO's 80)."""

    def __init__(self, weights: str = WEIGHTS, imgsz: int = IMGSZ):
        from ultralytics import YOLO
        from .fastperc import COCO_MAP, models_dir
        y = YOLO(str(models_dir() / "ultralytics" / weights))
        y.predict([np.zeros((64, 64, 3), np.uint8)], imgsz=imgsz, conf=SCORE, half=True, verbose=False)
        self.m, self.args, self.imgsz = y.predictor.model, y.predictor.args, imgsz
        self.stride = int(max(np.atleast_1d(np.asarray(self.m.stride, dtype=float))))
        lut = torch.full((len(y.names),), -1, dtype=torch.long)
        for i, n in y.names.items():
            if COCO_MAP.get(n) in CLASSES:
                lut[i] = CLASSES.index(COCO_MAP[n])
        self.lut, self.nc = lut.cuda(), len(y.names)

    def letterbox(self, img: torch.Tensor):
        B, _, H, W = img.shape
        r = min(self.imgsz / H, self.imgsz / W)
        nw, nh = round(W * r), round(H * r)
        dw, dh = ((self.imgsz - nw) % self.stride) / 2, ((self.imgsz - nh) % self.stride) / 2
        left, top = round(dw - 0.1), round(dh - 0.1)
        x = img.float()
        if (nh, nw) != (H, W):
            x = F.interpolate(x, (nh, nw), mode="bilinear", align_corners=False)
        x = F.pad(x, (left, round(dw + 0.1), top, round(dh + 0.1)), value=114.0)
        return x.div_(255).half(), r, left, top

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> dict:
        """-> cls int8 (B, 8) (-1 = empty), score f16 (B, 8), box f32 (B, 8, 4) native xyxy, feat f16 (B, 8, 1920)."""
        from torchvision.ops import roi_align
        from ultralytics.utils import nms
        B, _, H, W = img.shape
        x, r, left, top = self.letterbox(img)
        o = self.m(x)
        dets = nms.non_max_suppression(o[0], self.args.conf, self.args.iou, None, False, max_det=self.args.max_det, nc=self.nc)
        feats = o[1]["feats"]
        cls = torch.full((B, K_TOK), -1, dtype=torch.long, device=x.device)
        score = torch.zeros((B, K_TOK), device=x.device)
        lb = torch.zeros((B, K_TOK, 4), device=x.device)
        for b, d in enumerate(dets):
            c = self.lut[d[:, 5].long()]
            k = torch.nonzero(c >= 0).flatten()
            k = k[torch.argsort(d[k, 4], descending=True, stable=True)][:K_TOK]
            n = len(k)
            cls[b, :n], score[b, :n], lb[b, :n] = c[k], d[k, 4].float(), d[k, :4].float()
        valid = cls >= 0
        rois = [lb[b][valid[b]] for b in range(B)]
        feat = torch.zeros((B, K_TOK, FEAT_D), dtype=torch.half, device=x.device)
        if valid.any():
            f = torch.cat([roi_align(m.float(), rois, output_size=1, spatial_scale=m.shape[-1] / x.shape[-1],
                                     sampling_ratio=2, aligned=True).flatten(1) for m in feats], 1)
            feat[valid] = f.half()
        off = torch.tensor([left, top, left, top], device=x.device, dtype=torch.float32)
        box = ((lb - off) / r).clamp_min(0)
        box[..., 0::2], box[..., 1::2] = box[..., 0::2].clamp_max(W), box[..., 1::2].clamp_max(H)
        box[~valid] = 0
        return {"cls": cls.to(torch.int8).cpu().numpy(), "score": score.half().cpu().numpy(),
                "box": box.cpu().numpy(), "feat": feat.cpu().numpy()}


# ================================================================ geometry: native pixel -> road model frame

def _road_rays(step: float = 16.0):
    """Road-frame pixels on a grid far wider than the frame (the native cameras see more) and their vehicle-frame rays."""
    u, v = np.meshgrid(np.arange(-1536, 2048 + 1, step), np.arange(-768, 1024 + 1, step))
    K = ROAD_K
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ VIEW_TO_VEHICLE.T
    return np.stack([u, v], -1).reshape(-1, 2), (ray / np.linalg.norm(ray, axis=-1, keepdims=True)).reshape(-1, 3)


def fit_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalised DLT least squares, src (n, 2) -> dst (n, 2); returns H and the RMS residual in dst pixels."""
    def norm(p):
        m, s = p.mean(0), np.sqrt(2) / np.linalg.norm(p - p.mean(0), axis=1).mean()
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    Ts, Td = norm(src), norm(dst)
    a = np.c_[src, np.ones(len(src))] @ Ts.T
    b = np.c_[dst, np.ones(len(dst))] @ Td.T
    z = np.zeros((len(a), 3))
    A = np.r_[np.c_[a, z, -a * b[:, :1]], np.c_[z, a, -a * b[:, 1:2]]]
    Hn = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    H /= H[2, 2]
    return H, float(np.sqrt(((apply_h(H, src) - dst) ** 2).sum(-1).mean()))


def apply_h(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    """H (3, 3) or (..., 3, 3) broadcast against points p (..., 2)."""
    q = np.einsum("...ij,...j->...i", H, np.concatenate([p, np.ones(p.shape[:-1] + (1,))], -1))
    return q[..., :2] / q[..., 2:3]


def undistort(p: np.ndarray, intr: np.ndarray, iters: int = 30) -> np.ndarray:
    """Native pixels p (..., 2) -> ideal (distortion-free) pixels of the same camera. intr (..., 9) = [fu fv cu cv k1 k2 p1 p2
    k3], the Brown-Conrady model of camgeom.waymo_project and navsim_zs.project_nuplan (OpenCV's), inverted by fixed point."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = np.moveaxis(np.asarray(intr, np.float64), -1, 0)
    xd, yd = (p[..., 0] - cu) / fu, (p[..., 1] - cv) / fv
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / rad
        y = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / rad
    return np.stack([fu * x + cu, fv * y + cv], -1)


def distort(p: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Inverse of `undistort` (the forward model), for the round-trip check."""
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = np.moveaxis(np.asarray(intr, np.float64), -1, 0)
    x, y = (p[..., 0] - cu) / fu, (p[..., 1] - cv) / fv
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return np.stack([fu * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cu,
                     fv * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cv], -1)


def h_wod(calib: dict) -> tuple[np.ndarray, float, np.ndarray]:
    """camgeom (WOD-format) front-camera calibration -> (H ideal pixel -> road, error, intr (9,)); the fit uses the
    calibration without its distortion, so it is an exact rotation homography; `undistort` first. error = max(fit RMS, max
    road-px error of road grid -> distorted native pixel -> undistort -> H) over the grid points the camera sees."""
    from .camgeom import waymo_project
    intr = np.asarray(calib["intrinsic"], np.float64)
    uv, rays = _road_rays()
    U, V, ok = waymo_project(np, rays, {**calib, "intrinsic": list(intr[:4]) + [0.0] * 5})
    H, rms = fit_homography(np.stack([U, V], -1)[ok], uv[ok])
    U, V, ok = waymo_project(np, rays, calib)                       # the chain check: road -> native (distorted) -> road
    back = apply_h(H, undistort(np.stack([U, V], -1)[ok], intr))
    return H, max(rms, float(np.abs(back - uv[ok]).max())), intr


def h_nuplan(cam: dict) -> tuple[np.ndarray, float, np.ndarray]:
    """NAVSIM CAM_F0 record -> (H ideal pixel -> road, RMS, intr (9,)), the rays of navsim_zs.OpenpilotMaps."""
    from .navsim_zs import project_nuplan
    K, D = np.asarray(cam["K"], np.float64), np.asarray(cam["D"], np.float64)
    uv, rays = _road_rays()
    p, ok, _ = project_nuplan(rays, {**cam, "D": np.zeros(5)}, 1)
    H, rms = fit_homography(p.astype(np.float64)[ok], uv[ok])
    intr = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], D]
    p, ok, _ = project_nuplan(rays, cam, 1)                         # the chain check: road -> native (distorted) -> road
    back = apply_h(H, undistort(p.astype(np.float64)[ok], intr))
    return H, max(rms, float(np.abs(back - uv[ok]).max())), intr


# ================================================================ tokens

def fit_pca(feat: np.ndarray, d: int = PCA_D) -> dict:
    """PCA of detection appearance (rows = valid detections), whitened: token = (f - mu) @ V / sd."""
    x = torch.as_tensor(feat, dtype=torch.float64)
    mu = x.mean(0)
    ev, V = torch.linalg.eigh(torch.cov((x - mu).T))
    ev, V = ev.flip(0), V.flip(1)
    return {"mu": mu.float().numpy(), "V": V[:, :d].float().numpy(), "sd": ev[:d].clamp_min(1e-12).sqrt().float().numpy(),
            "explained": (ev[:d] / ev.sum()).numpy(), "n": len(feat)}


def build_tokens(cls, score, box, feat, H, intr, pca) -> tuple[np.ndarray, np.ndarray]:
    """Per image (n rows): cls (n, 8) int8, score (n, 8), box (n, 8, 4) native xyxy, feat (n, 8, 1920), H (n, 3, 3) ideal pixel ->
    road, intr (n, 9) native intrinsics + distortion -> tok (n, 8, TOK_D) float16, mask (n, 8) bool. The four box corners are
    undistorted, mapped to the road frame, and their bounding box taken; height / focal = ideal-pixel height / fv."""
    n = len(cls)
    m = cls >= 0
    tok = np.zeros((n, K_TOK, TOK_D), np.float32)
    tok[..., FIELDS["cls"]] = np.eye(3, dtype=np.float32)[np.maximum(cls, 0)]
    tok[..., 3] = score
    x0, y0, x1, y1 = np.moveaxis(box.astype(np.float64), -1, 0)
    corners = np.stack([np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x0, y1], -1), np.stack([x1, y1], -1)], -2)
    intr = np.asarray(intr, np.float64)
    ideal = undistort(corners, intr[:, None, None])
    q = apply_h(np.asarray(H, np.float64)[:, None, None], ideal)             # (n, 8, 4, 2) road pixels
    lo, hi = q.min(-2), q.max(-2)
    wh = np.array(ROAD_WH, np.float64)
    tok[..., 4:6] = np.clip((lo + hi) / 2 / wh, *BOX_CLIP)
    tok[..., 6:8] = np.clip((hi - lo) / wh, 0, BOX_CLIP[1] - BOX_CLIP[0])
    tok[..., 8] = (ideal[..., 2:, 1].mean(-1) - ideal[..., :2, 1].mean(-1)) / intr[:, 1, None]
    tok[..., FIELDS["app"]] = ((feat.astype(np.float32) - pca["mu"]) @ pca["V"]) / pca["sd"]
    tok[~m] = 0
    return tok.astype(np.float16), m


# ================================================================ loading, aligned with the trunk caches

TOK_DIRS = {"cache-sim": "sim", "cache-off": "off"}           # trunk cache dir -> token dir (round-1 dirs keep their name)


def tok_path(cache_file) -> Path:
    """Trunk cache file (processed/op_adapt/<ds>/<stem>.npz, runs/op_adapt_r2/cache-{sim,off}/<stem>.npy) -> its token file."""
    p = Path(cache_file)
    return root("tok", TOK_DIRS.get(p.parent.name, p.parent.name), p.stem + ".npz")


def load_tok(cache_file) -> tuple[np.ndarray, np.ndarray]:
    """tok (n, 8, TOK_D) float16 and mask (n, 8) bool, row j = trunk row j of `cache_file` (the row's current frame)."""
    z = np.load(tok_path(cache_file))
    return z["tok"], z["mask"]


def flatten(cache_files, out_prefix) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the token arrays of `cache_files` in that order (= a flattened trunk) into memmaps
    <out_prefix>.tok.npy (N, 8, TOK_D) float16 and <out_prefix>.mask.npy (N, 8) bool."""
    parts = [load_tok(f) for f in cache_files]
    N = sum(len(t) for t, _ in parts)
    o = Path(out_prefix)
    tok = np.lib.format.open_memmap(f"{o}.tok.npy", "w+", np.float16, (N, K_TOK, TOK_D))
    msk = np.lib.format.open_memmap(f"{o}.mask.npy", "w+", bool, (N, K_TOK))
    i = 0
    for t, m in parts:
        tok[i:i + len(t)], msk[i:i + len(t)] = t, m
        i += len(t)
    tok.flush(), msk.flush()
    return tok, msk


def gather(tok, mask, g, valid) -> tuple[torch.Tensor, torch.Tensor]:
    """Context tokens of a batch: g (B, 9) global trunk rows (any row where invalid), valid (B, 9) -> tok (B, 9, 8, D),
    mask (B, 9, 8) with every token of an invalid context slot masked."""
    g, valid = np.asarray(g), np.asarray(valid)
    t = torch.from_numpy(np.ascontiguousarray(tok[g.ravel()])).view(*g.shape, K_TOK, TOK_D)
    m = torch.from_numpy(np.ascontiguousarray(mask[g.ravel()])).view(*g.shape, K_TOK) & torch.from_numpy(valid)[..., None]
    return t, m


T_KEY = np.array([-1.5, -1.0, -0.5, 0.0])                     # NAVSIM keyframes (2 Hz)


def nav_tokens(ds: str):
    """navtest / navhard: (tokens (N,), tok (N, 4, 8, TOK_D) f16, mask (N, 4, 8)) for the 4 keyframes of every leaderboard
    token, in the order of runs/op_lb/lb_<ds>/tokens.txt."""
    toks = root(ds, "tokens.txt").read_text().split()
    z = np.load(root("tok", ds, f"{ds}.npz"))
    return np.array(toks), z["tok"].reshape(len(toks), 4, K_TOK, TOK_D), z["mask"].reshape(len(toks), 4, K_TOK)


def hold_ctx(tok4, mask4, times=np.round(np.arange(-8, 1) * 0.2, 3)):
    """Keyframe tokens -> context-slot tokens by sample-and-hold: slot at time t takes the latest keyframe <= t; slots before
    -1.5 s get none (the navtrain cache's 2 Hz rule, also used for the GIMM-filled navtest / navhard contexts)."""
    k = np.searchsorted(T_KEY, np.asarray(times) + 1e-6) - 1
    ok = k >= 0
    t, m = tok4[:, k.clip(0)], mask4[:, k.clip(0)] & ok[None, :, None]
    return np.where(m[..., None], t, 0).astype(tok4.dtype), m


# ================================================================ the adapter (envs/op-train)

class DetAdapter(torch.nn.Module):
    """H (B, S, 32, 512) + tokens (B, S, 8, TOK_D), mask (B, S, 8) -> H + gate * CrossAttn(LN(H), MLP(tokens)), per slot.
    gate is a scalar initialised to 0, so the adapted model equals the original at step 0 (bit for bit: H + 0). The
    attention's output projection keeps PyTorch's default init (not zero, see the build doc: a zero projection times a
    zero gate has zero gradient in both and never trains). Slots with no valid token add exactly 0."""

    def __init__(self, d_tok: int = TOK_D, d: int = 512, heads: int = 8, hidden: int = 256):
        super().__init__()
        self.kv = torch.nn.Sequential(torch.nn.Linear(d_tok, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d),
                                      torch.nn.LayerNorm(d))
        self.q_norm = torch.nn.LayerNorm(d)
        self.attn = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, H: torch.Tensor, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, N, D = H.shape
        q = self.q_norm(H.float()).reshape(B * S, N, D)
        kv = self.kv(tok.float()).reshape(B * S, tok.shape[2], D)
        m = mask.reshape(B * S, -1).bool()
        has = m.any(1)
        y, _ = self.attn(q, kv, kv, key_padding_mask=~(m | ~has[:, None]), need_weights=False)
        y = torch.where(has[:, None, None], y, torch.zeros((), device=y.device))
        return H + (self.gate * y).reshape(B, S, N, D).to(H.dtype)


def stage4_policy_det(net, adapter, trunk, tok, mask, action_t, traffic=None, valid=None) -> dict:
    """op_adapt.stage4_policy with `adapter` (None = the original path) on the 9 hidden states before the policy.
    trunk (B, 9, 1024, 8, 16), tok (B, 9, 8, TOK_D), mask (B, 9, 8)."""
    from . import op_adapt as A
    B = trunk.shape[0]
    H = net.run_batched({A.TRUNK_OUT: trunk.reshape(B * A.CONTEXT, 1, *trunk.shape[2:]).to(net.dtype)}, ["view_39"])["view_39"]
    H = H.reshape(B, A.CONTEXT, *A.H_SHAPE)
    if adapter is not None:
        H = adapter(H, tok.to(H.device), mask.to(H.device))
    return A._policy(net, H, action_t, traffic, valid)
