"""op-adapt H: layer-3 adaptation of openpilot Cinque on two pair types (plans/2026-10-04-op-adapt-H-prereg.md).

  H pairs   the history frames are perturbed (fake ego yaw over the history, the t0 frame repeated, the history dropped) on
            frames whose logged future does not follow the implied motion; target = the logged future
  O pairs   the image is re-projected as if the ego stood at a heading offset (exact up to the camera lever) and a small lateral
            offset (road-plane approximation), reached by a drift over the history; target = a recovery path back onto the
            logged path
  U rows    the same frames unperturbed (real history rotation included), target = the logged future
  D rows    unperturbed frames distilled to the original model (plan consistency + every output head)

Samples come from experiments/op_adapt_h/scripts/h_prep.py (10 packed frames on the 5 Hz lattice, pairs (k, k+1) = the 9 policy
context slots). The model is op_adapt_l's port (`LModel`, intent none): stage 1-3 frozen and run online on the (perturbed)
images, stage 4 and the off-policy plan pathway trainable. Checkpoints keep op_adapt_l's format, so its readouts load them.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from jevdrive import op_adapt as A
from jevdrive import op_interp as I
from jevdrive.common import data_dir
from experiments.op_adapt_l.lib import op_adapt_l as L

DOMS = ("nav", "wod", "carla")
AT = (0.275, 0.525)
NIMG = 10
T_FUT = 0.25 * np.arange(1, 21)
ROLES = {"U": 1, "D": 2, "H": 3, "O": 4}


def hroot(*p) -> Path:
    d = Path(os.environ.get("OP_H_ROOT") or data_dir() / "runs" / "op_adapt_H") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


class Samples:
    """imgs (n, 10, 2, 6, 128, 256) uint8 memmap + the table of one domain."""

    def __init__(self, dom: str):
        d = hroot("samples", dom)
        self.dom = dom
        self.imgs = np.load(d / "imgs.npy", mmap_mode="r")
        with np.load(d / "tab.npz", allow_pickle=True) as z:
            self.t = {k: z[k] for k in z.files}
        self.n = len(self.imgs)
        tea = hroot("teacher") / f"{dom}.npz"
        self.tea = dict(np.load(tea)) if tea.exists() else None

    def rows(self, split=None, bins=None, vmin=None) -> np.ndarray:
        m = np.ones(self.n, bool)
        if split is not None:
            m &= self.t["split"] == split
        if bins is not None:
            m &= np.isin(self.t["bin"], bins)
        if vmin is not None:
            m &= self.t["v0"] >= vmin
        return np.flatnonzero(m)


# ---------------------------------------------------------------- image warps (op_interp.warp_map, as op_common_cause's warp_rot)
def warp(f: np.ndarray, cam, dy: float, dpsi: float) -> np.ndarray:
    """One packed frame (2, 6, 128, 256) re-rendered as seen from the ego displaced by dy (m, left) and yawed by dpsi (rad, left)
    about the vehicle origin. Rotation is exact up to the camera lever; translation uses the road plane / 60 m sphere."""
    import cv2
    if abs(dy) < 1e-6 and abs(dpsi) < 1e-9:
        return f
    out = np.empty_like(f)
    c = np.round(np.asarray(cam, float), 2)
    for k, view in enumerate(("road", "wide")):
        mx, my = I.warp_map(view, c, np.array([0.0, dy, dpsi]), np.zeros(3))
        hx, hy = (mx[0::2, 0::2] + mx[1::2, 1::2]) / 4 - 0.25, (my[0::2, 0::2] + my[1::2, 1::2]) / 4 - 0.25
        Y, U, V = I.unpack(f[k])
        out[k] = I.pack(cv2.remap(Y, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        cv2.remap(np.ascontiguousarray(U), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                        cv2.remap(np.ascontiguousarray(V), hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def hist_rot(imgs, t, valid, cam, omega):
    """Fake ego yaw at rate omega (rad/s, left +) up to t0: frame at source time t gets the extra yaw omega * t (t0 unchanged).
    Same construction as decision 92's rotL / rotR (omega = +-10 deg/s)."""
    return np.stack([warp(f, cam, 0.0, omega * tt) if v else f for f, tt, v in zip(imgs, t, valid)])


def hist_repeat(imgs, valid):
    """The t0 frame on every valid history slot: a static history."""
    return np.where(valid[:, None, None, None, None], imgs[-1][None], imgs)


def hist_single(imgs, slot_valid):
    """The history dropped: only the t0 frame after a zero image (a stream started at t0)."""
    out = np.zeros_like(imgs)
    out[-1] = imgs[-1]
    sv = np.zeros_like(slot_valid)
    sv[-1] = True
    return out, sv


def drift(t, dy, dpsi, v, T=1.6):
    """Offset (lateral m, heading rad) of the ego against the logged path at history times t (<= 0) for a drift that reaches
    (dy, dpsi) at t0: heading error ramps linearly from 0 at -T, the lateral error integrates v * sin(heading error)."""
    tt = np.clip(np.asarray(t, float), -T, 0.0)
    psi = dpsi * (tt + T) / T
    y = dy - v * dpsi / (2 * T) * (T ** 2 - (tt + T) ** 2)            # y(0) = dy, dy/dt = v * psi (small angles)
    return y, psi


def offset_imgs(imgs, t, valid, cam, dy, dpsi, v):
    ys, ps = drift(t, dy, dpsi, v)
    return np.stack([warp(f, cam, y, p) if ok else f for f, y, p, ok in zip(imgs, ys, ps, valid)])


def recover_target(fut20: np.ndarray, dy: float, dpsi: float, v0: float, t_rec=2.5, s_min=8.0) -> np.ndarray:
    """Recovery path in the offset ego frame (20, 2): the logged path p(t) plus a lateral deviation e(s) along its arc length
    that starts at (dy, slope tan dpsi) and returns to (0, 0) at s = S (cubic Hermite), S = max(s_min, t_rec * speed); the
    speed profile stays the logged one."""
    p = np.concatenate([np.zeros((1, 2)), np.asarray(fut20, float)])
    seg = np.diff(p, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    s = np.r_[0.0, np.cumsum(ds)]
    tang = np.where(ds[:, None] > 1e-3, seg / np.maximum(ds[:, None], 1e-9), np.nan)
    tang = np.r_[[[1.0, 0.0]], tang]
    for i in range(1, len(tang)):                                          # stationary segments keep the last direction
        if not np.isfinite(tang[i]).all():
            tang[i] = tang[i - 1]
    n = np.stack([-tang[:, 1], tang[:, 0]], 1)                              # left normal
    vref = max(v0, s[-1] / 5.0)
    S = max(s_min, t_rec * vref)
    u = np.clip(s / S, 0, 1)
    e = dy * (2 * u ** 3 - 3 * u ** 2 + 1) + S * np.tan(dpsi) * (u ** 3 - 2 * u ** 2 + u)
    q = p + e[:, None] * n
    q = q - np.array([0.0, dy])
    c, sn = np.cos(-dpsi), np.sin(-dpsi)
    q = q @ np.array([[c, -sn], [sn, c]]).T
    return q[1:].astype(np.float32)


def fut_yaw_deg(fut20: np.ndarray) -> float:
    """Heading of the logged path at about 3 s (deg, left +) from the 2.5 -> 3.0 s chord; 0 when it barely moves."""
    a, b = fut20[9], fut20[11]
    d = b - a
    return 0.0 if np.linalg.norm(d) < 0.3 else float(np.degrees(np.arctan2(d[1], d[0])))


# ---------------------------------------------------------------- batches
@dataclass
class HCfg:
    name: str
    seed: int = 0
    control: bool = False                 # U-only control: the H / O rows become U rows (no perturbation)
    steps: int = 2500
    roles: dict = field(default_factory=lambda: {"U": 12, "D": 8, "H": 16, "O": 12})
    dom_w: dict = field(default_factory=lambda: {"nav": 0.4, "wod": 0.3, "carla": 0.3})
    hist_p: dict = field(default_factory=lambda: {"rot": 0.6, "repeat": 0.2, "single": 0.2})
    rot_dps: tuple = (5.0, 15.0)          # fake yaw rate range (deg/s), random sign subject to the no-follow rule
    follow_deg: float = 3.0               # |logged heading at 3 s| above this fixes the fake yaw's sign against it
    off_psi_deg: tuple = (1.0, 8.0)
    off_y: tuple = (0.0, 1.0)
    off_vmin: float = 1.0
    repeat_vmin: float = 2.0
    s4: bool = True
    pol: bool = True
    lr: float = 3e-5
    wd: float = 0.01
    warmup: int = 100
    lam_i: float = 1.0
    lam_d: float = 10.0
    lam_c: float = 1.0
    dw: float = 1.0
    w_role: dict = field(default_factory=lambda: {"U": 1.0, "H": 1.0, "O": 1.0})
    eval_every: int = 1000
    ckpt_every: int = 500
    workers: int = 24

    def dump(self):
        return asdict(self)

    def lcfg(self) -> L.LCfg:
        return L.LCfg(name=self.name, seed=self.seed, s4=self.s4, pol=self.pol, intent="none", steps=self.steps)


def perturb_hist(S: Samples, i: int, rng, cfg: HCfg):
    """(imgs, slot_valid, kind) of one H row."""
    t = S.t
    imgs, tt, iv, sv, cam, v0 = np.asarray(S.imgs[i]), t["img_t"][i], t["img_valid"][i], t["slot_valid"][i].copy(), t["cam"][i], t["v0"][i]
    kinds = [k for k in cfg.hist_p if not (k == "repeat" and v0 < cfg.repeat_vmin)]
    p = np.array([cfg.hist_p[k] for k in kinds])
    k = kinds[rng.choice(len(kinds), p=p / p.sum())]
    if k == "rot":
        fy = fut_yaw_deg(t["fut20"][i])
        sign = -np.sign(fy) if abs(fy) > cfg.follow_deg else rng.choice([-1.0, 1.0])
        return hist_rot(imgs, tt, iv, cam, sign * np.radians(rng.uniform(*cfg.rot_dps))), sv, k
    if k == "repeat":
        return hist_repeat(imgs, iv), sv, k
    im, sv = hist_single(imgs, sv)
    return im, sv, k


def row(S: Samples, i: int, role: str, rng, cfg: HCfg) -> dict:
    t = S.t
    sv = t["slot_valid"][i].copy()
    tgt = L.human_targets(t["fut20"][i][None])[0]
    kind = "none"
    if role == "H":
        imgs, sv, kind = perturb_hist(S, i, rng, cfg)
    elif role == "O":
        dpsi = np.radians(rng.uniform(*cfg.off_psi_deg)) * rng.choice([-1.0, 1.0])
        dy = rng.uniform(*cfg.off_y) * rng.choice([-1.0, 1.0])
        v0 = float(t["v0"][i])
        imgs = offset_imgs(np.asarray(S.imgs[i]), t["img_t"][i], t["img_valid"][i], t["cam"][i], dy, dpsi, v0)
        tgt = L.human_targets(recover_target(t["fut20"][i], dy, dpsi, v0)[None])[0]
        kind = "offset"
    else:
        imgs = np.asarray(S.imgs[i])
    tea = S.tea
    return {"imgs": imgs, "valid": sv, "tc": t["tc"][i], "role": ROLES[role], "hum": tgt, "cam": np.float32(t["cam"][i][0]),
            "tgt": tea["out"][i], "tmu": tea["mu"][i], "dom": DOMS.index(S.dom), "kind": kind}


class Batcher(torch.utils.data.Dataset):
    """Item k = one whole batch drawn with rng (seed, k): roles per HCfg, domain per dom_w, rows uniform within the train split."""

    def __init__(self, cfg: HCfg, split="train", n=10 ** 9):
        self.cfg, self.split, self.n = cfg, split, n
        self.S = None

    def __len__(self):
        return self.n

    def _open(self):
        c = self.cfg
        self.S = {d: Samples(d) for d in DOMS if c.dom_w.get(d, 0) > 0}
        self.pool = {}
        for d, S in self.S.items():
            self.pool[(d, "U")] = self.pool[(d, "D")] = self.pool[(d, "H")] = S.rows(self.split)
            self.pool[(d, "O")] = S.rows(self.split, vmin=c.off_vmin)

    def __getitem__(self, k):
        if self.S is None:
            self._open()
        c = self.cfg
        rng = np.random.default_rng([c.seed, k])
        doms = [d for d in c.dom_w if c.dom_w[d] > 0]
        w = np.array([c.dom_w[d] for d in doms])
        rows = []
        for role, cnt in c.roles.items():
            r = "U" if (c.control and role in ("H", "O")) else role
            for d in rng.choice(doms, cnt, p=w / w.sum()):
                p = self.pool[(d, r)]
                rows.append(row(self.S[d], int(p[rng.integers(len(p))]), r, rng, c))
        return collate(rows)


def collate(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        if k == "kind":
            out[k] = [r[k] for r in rows]
        else:
            out[k] = torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows]))
    return out


# ---------------------------------------------------------------- forward from images
def trunks(net, imgs: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """(B, 10, 2, 6, 128, 256) uint8 on the GPU -> (B, 9, 1024, 8, 16) stage-3 outputs of the pairs (k, k+1), no autograd."""
    B = imgs.shape[0]
    prev, cur = imgs[:, :-1].reshape(-1, *imgs.shape[2:]), imgs[:, 1:].reshape(-1, *imgs.shape[2:])
    with torch.no_grad():
        out = [net.run_batched(A.vision_feeds(prev[i:i + chunk], cur[i:i + chunk]), [A.TRUNK_OUT])[A.TRUNK_OUT][:, 0]
               for i in range(0, len(cur), chunk)]
    return torch.cat(out).reshape(B, 9, *out[0].shape[1:])


def forward(model: L.LModel, imgs, valid, tc):
    return model(trunks(model.net, imgs), valid, tc.to(model.net.dtype) if tc.dtype != model.net.dtype else tc)


class Losses:
    """Imitation (U / H / O rows) against the targets on op_adapt_l's 16-point rear-axle grid, plan consistency to the original on
    D rows, normalised distillation of every output head on the unperturbed rows (U, D; perturbed rows change the scene, e.g. the
    lane lines under an offset, so their non-plan heads are not pinned)."""

    def __init__(self, net, cfg: HCfg, tstd: np.ndarray, dev):
        self.cfg = cfg
        self.base = L.Losses(net, cfg.lcfg(), tstd, dev)

    def __call__(self, o, b):
        c, base = self.cfg, self.base
        out = o["outputs"].float()
        plan = base.plan(out)
        role = b["role"]
        L_ = {}
        imit = (role == 1) | (role == 3) | (role == 4)
        w = torch.zeros_like(role, dtype=torch.float32)
        for r, k in (("U", 1), ("H", 3), ("O", 4)):
            w = torch.where(role == k, torch.full_like(w, c.w_role.get(r, 1.0)), w)
        tot = 0.0
        if imit.any():
            d = base.imit_dist(plan[imit], None, b["cam"][imit], b["hum"][imit])
            L_["imit"] = (w[imit] * d).sum() / w[imit].sum()
            for r, k in (("U", 1), ("H", 3), ("O", 4)):
                m = role[imit] == k
                if m.any():
                    L_[f"imit_{r}"] = d[m].mean().detach()
            tot = tot + c.lam_i * L_["imit"]
        dd = role == 2
        if dd.any():
            L_["cons"] = base.imit_dist(plan[dd], b["tmu"][dd].float(), b["cam"][dd]).mean()
            tot = tot + c.dw * c.lam_c * L_["cons"]
        un = (role == 1) | dd
        if un.any():
            e = ((out[un][:, base.di] - b["tgt"][un].float()) / base.tstd).pow(2)
            keep = (role[un] == 2).float()
            num = e[:, ~base.plan_cols].sum(1) + keep * e[:, base.plan_cols].sum(1)
            L_["distill"] = (num / e.shape[1]).mean()
            tot = tot + c.dw * c.lam_d * L_["distill"]
        return tot, L_


# ---------------------------------------------------------------- probe / dev metrics
T_IDXS = A.T_IDXS


def psi3(plan: np.ndarray) -> np.ndarray:
    """Plan heading at 3 s (deg, left +) as op_common_cause's cc_report (plan channel 11 = yaw, openpilot frame)."""
    return -np.degrees(np.array([np.interp(3.0, T_IDXS, p[:, 11]) for p in plan]))


def plan_rear(plan: np.ndarray, cam_x) -> np.ndarray:
    """(n, 33, 15) -> rear-axle (n, 20, 2) on the 0.25 s grid (x fwd, y left)."""
    cam_x = np.broadcast_to(np.asarray(cam_x, float), (len(plan),))
    return np.stack([L.rear_np(p[None], float(c), T_FUT)[0] for p, c in zip(plan, cam_x)])
