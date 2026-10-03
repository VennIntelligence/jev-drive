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
ROLES = {"U": 1, "D": 2, "H": 3, "O": 4, "L": 5, "S": 6, "M": 7}
LAUNCH_DOMS = ("lwod", "lcarla")       # real launch events (h_prep.py prep_lwod / prep_lcarla), table column m


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


def hist_launch(imgs, t, valid, cam, m: int, delta_deg: float, freeze: bool):
    """Launch history (plans/2026-10-04-launch-pairs-prereg.md): a static prefix, then the last m frames moving, plus a fake yaw
    delta that the static prefix carries whole and the moving frames ramp down to 0 at t0 (the de-rotated view of a launch that
    yawed delta since standing). freeze=True makes the prefix static by copying image 9 - m onto the earlier valid images
    (synthetic launch from a moving sample); a real launch keeps its frames."""
    imgs = np.array(imgs)
    k0 = NIMG - 1 - m
    if freeze:
        imgs[:k0][valid[:k0]] = imgs[k0]
    r = np.clip(np.asarray(t, float) / min(float(t[k0]), -1e-6), 0.0, 1.0)
    r[:k0] = 1.0
    d = np.radians(delta_deg)
    return np.stack([warp(f, cam, 0.0, d * q) if v else f for f, q, v in zip(imgs, r, valid)])


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


# ---------------------------------------------------------------- trunk bank
# Stage 1-3 is frozen, so every (sample, perturbation) pair has fixed stage-3 outputs. The bank holds them once per domain:
# per sample the normal input, K fake-yaw histories, the repeated and the dropped history and K offsets, each as the 9 context
# trunks (zeros on invalid slots), with strengths drawn over a wide range; an arm picks its strength window from the table.
BANK_K = 2
BANK_ROT_DPS = (3.0, 20.0)
BANK_OFF_PSI = (1.0, 10.0)
BANK_OFF_Y = (0.0, 1.2)
KINDS = ("normal", "rot", "repeat", "single", "offset")


def bank_plan(S: Samples, seed=0) -> dict:
    """Variant table of a domain: sample, kind, fake yaw rate (deg/s, signed), offset dy (m), dpsi (deg). The fake-yaw sign is
    set against the logged heading at 3 s when that exceeds 3 deg (the future does not follow), else random."""
    rng = np.random.default_rng([seed, DOMS.index(S.dom)])
    rows = []
    for i in range(S.n):
        fy = fut_yaw_deg(S.t["fut20"][i])
        rows.append((i, "normal", 0.0, 0.0, 0.0))
        for _ in range(BANK_K):
            sg = -np.sign(fy) if abs(fy) > 3.0 else rng.choice([-1.0, 1.0])
            rows.append((i, "rot", float(sg * rng.uniform(*BANK_ROT_DPS)), 0.0, 0.0))
        rows += [(i, "repeat", 0.0, 0.0, 0.0), (i, "single", 0.0, 0.0, 0.0)]
        for _ in range(BANK_K):
            rows.append((i, "offset", 0.0, float(rng.uniform(*BANK_OFF_Y) * rng.choice([-1, 1])),
                         float(rng.uniform(*BANK_OFF_PSI) * rng.choice([-1, 1]))))
    c = list(zip(*rows))
    return {"sample": np.array(c[0]), "kind": np.array(c[1]), "rate": np.array(c[2]), "dy": np.array(c[3]), "dpsi": np.array(c[4])}


def variant_imgs(S: Samples, i: int, kind: str, rate=0.0, dy=0.0, dpsi_deg=0.0, m=0):
    """(imgs (10, ...), slot_valid (9,)) of one variant (launch: rate = fake yaw delta in deg, m = moving frames; a real-launch
    domain uses its own m and its frames)."""
    t = S.t
    imgs, sv = np.asarray(S.imgs[i]), t["slot_valid"][i].copy()
    if kind == "normal":
        return imgs, sv
    if kind == "rot":
        return hist_rot(imgs, t["img_t"][i], t["img_valid"][i], t["cam"][i], np.radians(rate)), sv
    if kind == "repeat":
        return hist_repeat(imgs, t["img_valid"][i]), sv
    if kind == "single":
        return hist_single(imgs, sv)
    if kind == "launch":
        real = "m" in t
        return hist_launch(imgs, t["img_t"][i], t["img_valid"][i], t["cam"][i], int(t["m"][i]) if real else int(m), rate, not real), sv
    if kind == "offset":
        return offset_imgs(imgs, t["img_t"][i], t["img_valid"][i], t["cam"][i], dy, np.radians(dpsi_deg), float(t["v0"][i])), sv
    raise KeyError(kind)


# bank2 (plans/2026-10-04-launch-pairs-prereg.md): launch variants and small-rate fake yaw, beside the first bank.
BANK2_DELTA = (0.3, 3.0)              # |fake yaw| of a launch (deg)
BANK2_RATE = (0.3, 2.0)               # |small fake yaw rate| (deg/s)
BANK2_LAUNCH_V = (0.5, 5.0)           # t0 speed of a synthetic launch (m/s)


def bank2_plan(S: Samples, seed=0) -> dict:
    """Variant table of bank2. Real-launch domain: normal + one launch (its own m). Other pools: one small-rate rot on stop / low /
    mid samples, one synthetic launch (m 1-3) on 0.5 <= v0 < 5 m/s. Fake-yaw sign against the logged 3 s heading when |psi| > 3 deg."""
    rng = np.random.default_rng([seed, 7, (DOMS + LAUNCH_DOMS).index(S.dom)])
    real = "m" in S.t
    rows = []
    for i in range(S.n):
        fy = fut_yaw_deg(S.t["fut20"][i])
        sg = lambda: -np.sign(fy) if abs(fy) > 3.0 else rng.choice([-1.0, 1.0])  # noqa: E731
        v0 = float(S.t["v0"][i])
        if real:
            rows += [(i, "normal", 0.0, 0), (i, "launch", float(sg() * rng.uniform(*BANK2_DELTA)), int(S.t["m"][i]))]
            continue
        if S.t["bin"][i] in ("stop", "low", "mid"):
            rows.append((i, "rot", float(sg() * rng.uniform(*BANK2_RATE)), 0))
        if BANK2_LAUNCH_V[0] <= v0 < BANK2_LAUNCH_V[1]:
            rows.append((i, "launch", float(sg() * rng.uniform(*BANK2_DELTA)), int(rng.integers(1, 4))))
    c = list(zip(*rows))
    n = len(rows)
    return {"sample": np.array(c[0]), "kind": np.array(c[1]), "rate": np.array(c[2]), "dy": np.zeros(n), "dpsi": np.zeros(n),
            "m": np.array(c[3])}


# bank3 (round 2, iteration 1): the replayed HUGSIM spins reach |H| 1-15 deg at steps 4-7 (v ~2 m/s), beyond bank2's launch
# window (m <= 3, |delta| <= 3): synthetic launches with m 4-8 moving frames and |delta| 2-12 deg, on low-bin samples only (disk).
BANK3_DELTA = (2.0, 12.0)
BANK3_M = (4, 8)


def bank3_plan(S: Samples, seed=0) -> dict:
    rng = np.random.default_rng([seed, 11, (DOMS + LAUNCH_DOMS).index(S.dom)])
    rows = []
    for i in range(S.n):
        if S.t["bin"][i] != "low":
            continue
        fy = fut_yaw_deg(S.t["fut20"][i])
        sg = -np.sign(fy) if abs(fy) > 3.0 else rng.choice([-1.0, 1.0])
        rows.append((i, "launch", float(sg * rng.uniform(*BANK3_DELTA)), int(rng.integers(BANK3_M[0], BANK3_M[1] + 1))))
    c = list(zip(*rows))
    n = len(rows)
    return {"sample": np.array(c[0]), "kind": np.array(c[1]), "rate": np.array(c[2]), "dy": np.zeros(n), "dpsi": np.zeros(n),
            "m": np.array(c[3])}


class Bank:
    def __init__(self, dom: str, name: str = "bank"):
        d = hroot(name, dom)
        self.T = np.load(d / "trunk.npy", mmap_mode="r")
        with np.load(d / "var.npz", allow_pickle=True) as z:
            self.v = {k: z[k] for k in z.files}


def pick_variant(B: Bank, S: Samples, i: int, role: str, rng, cfg: "HCfg"):
    """Variant index for one row of sample i (None when the sample has no admissible variant for the role)."""
    v = B.v
    lo, hi = B.first[i], B.first[i + 1]
    kinds = v["kind"][lo:hi]
    if role in ("U", "D"):
        return lo + int(np.flatnonzero(kinds == "normal")[0])
    if role in ("L", "M"):
        o = np.flatnonzero(kinds == "launch")
        if not len(o):
            return None
        if role == "L" and "m" in S.t and rng.random() < cfg.launch_plain:     # a real launch without fake yaw (the synthetic ones always have one)
            return lo + int(np.flatnonzero(kinds == "normal")[0])
        return lo + int(o[rng.integers(len(o))])
    if role == "S":
        o = np.flatnonzero(kinds == "rot")
        return lo + int(o[rng.integers(len(o))]) if len(o) else None
    if role == "H":
        opts, w = [], []
        r = np.abs(v["rate"][lo:hi])
        rot = np.flatnonzero((kinds == "rot") & (r >= cfg.rot_dps[0]) & (r <= cfg.rot_dps[1]))
        if len(rot):
            opts.append(rot), w.append(cfg.hist_p.get("rot", 0))
        if S.t["v0"][i] >= cfg.repeat_vmin:
            opts.append(np.flatnonzero(kinds == "repeat")), w.append(cfg.hist_p.get("repeat", 0))
        opts.append(np.flatnonzero(kinds == "single")), w.append(cfg.hist_p.get("single", 0))
        w = np.array(w, float)
        if w.sum() <= 0:
            return None
        o = opts[rng.choice(len(opts), p=w / w.sum())]
        return lo + int(o[rng.integers(len(o))])
    ap, ay = np.abs(v["dpsi"][lo:hi]), np.abs(v["dy"][lo:hi])
    off = np.flatnonzero((kinds == "offset") & (ap >= cfg.off_psi_deg[0]) & (ap <= cfg.off_psi_deg[1]) & (ay >= cfg.off_y[0]) & (ay <= cfg.off_y[1]))
    return lo + int(off[rng.integers(len(off))]) if len(off) else None


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
    rot_dps: tuple = (5.0, 15.0)          # admissible |fake yaw rate| (deg/s) of the bank's rot variants
    off_psi_deg: tuple = (1.0, 8.0)       # admissible |heading offset|
    off_y: tuple = (0.0, 1.0)             # admissible |lateral offset|
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
    w_role: dict = field(default_factory=lambda: {"U": 1.0, "H": 1.0, "O": 1.0, "L": 1.0, "S": 1.0, "M": 1.0})
    dom_w_L: dict = field(default_factory=dict)          # domains of L rows (bank2 launch variants); S rows use dom_w
    dom_w_M: dict = field(default_factory=lambda: {"nav": 0.4, "wod": 0.3, "carla": 0.3})   # M rows: bank3 long launches
    launch_plain: float = 0.25                           # share of L rows on real launches drawn without fake yaw
    lam_p: float = 0.0                                   # pair consistency: plan(perturbed history) to the stop-gradient plan of
    pair_roles: tuple = ("H", "L", "S", "M")             # the same sample's unperturbed history (same t0 image), per these roles
    ckpt_every: int = 500
    workers: int = 10

    def dump(self):
        return asdict(self)

    def lcfg(self) -> L.LCfg:
        return L.LCfg(name=self.name, seed=self.seed, s4=self.s4, pol=self.pol, intent="none", steps=self.steps)


class Batcher(torch.utils.data.Dataset):
    """Item k = one whole batch drawn with rng (seed, k): roles per HCfg, domain per dom_w, samples uniform within the train
    split (O rows: v0 >= off_vmin), the variant per role from the bank."""

    def __init__(self, cfg: HCfg, split="train", n=10 ** 9):
        self.cfg, self.split, self.n = cfg, split, n
        self.S = None

    def __len__(self):
        return self.n

    def _open(self):
        c = self.cfg
        need2 = c.roles.get("L", 0) + c.roles.get("S", 0) > 0
        self.S = {d: Samples(d) for d in DOMS + LAUNCH_DOMS if c.dom_w.get(d, 0) > 0 or c.dom_w_L.get(d, 0) > 0}
        self.B = {d: Bank(d) for d in self.S if d not in LAUNCH_DOMS}
        self.B2 = {d: Bank(d, "bank2") for d in self.S if need2 or d in LAUNCH_DOMS}
        self.B3 = {d: Bank(d, "bank3") for d in DOMS if c.roles.get("M", 0) > 0 and d in self.S}
        self.pool = {}
        for d, S in self.S.items():
            for B in (self.B.get(d), self.B2.get(d), self.B3.get(d)):
                if B is not None:
                    B.first = np.searchsorted(B.v["sample"], np.arange(S.n + 1))
            self.pool[(d, "U")] = self.pool[(d, "D")] = self.pool[(d, "H")] = S.rows(self.split)
            self.pool[(d, "O")] = S.rows(self.split, vmin=c.off_vmin)
            if d in self.B2:                                  # samples with an L / S variant in bank2
                v, tr = self.B2[d].v, set(S.rows(self.split).tolist())
                for r, kind in (("L", "launch"), ("S", "rot")):
                    self.pool[(d, r)] = np.array(sorted(tr & set(v["sample"][v["kind"] == kind].tolist())), int)
            if d in self.B3:
                self.pool[(d, "M")] = np.array(sorted(set(S.rows(self.split).tolist()) & set(self.B3[d].v["sample"].tolist())), int)

    def __getitem__(self, k):
        if self.S is None:
            self._open()
        c = self.cfg
        rng = np.random.default_rng([c.seed, k])
        rows, twins = [], []
        for role, cnt in c.roles.items():
            r = "U" if (c.control and role in ("H", "O", "L", "S", "M")) else role
            dw = c.dom_w_L if r == "L" else c.dom_w_M if r == "M" else {d: x for d, x in c.dom_w.items() if d not in LAUNCH_DOMS} if r in ("H", "O", "S") else c.dom_w
            ds = [d for d in dw if dw[d] > 0]
            pw = np.array([dw[d] for d in ds], float)
            for d in rng.choice(ds, cnt, p=pw / pw.sum()):
                S, p = self.S[d], self.pool[(d, r)]
                B = self.B3[d] if r == "M" else self.B2[d] if (r in ("L", "S") or d in LAUNCH_DOMS) else self.B[d]
                for _ in range(20):
                    i = int(p[rng.integers(len(p))])
                    j = pick_variant(B, S, i, r, rng, c)
                    if j is not None:
                        break
                rows.append(bank_row(S, B, i, j, r))
                if c.lam_p > 0 and r in c.pair_roles:          # the unperturbed twin of the row (same sample, normal variant)
                    B0 = self.B2[d] if d in LAUNCH_DOMS else self.B[d]
                    lo, hi = B0.first[i], B0.first[i + 1]
                    j0 = lo + int(np.flatnonzero(B0.v["kind"][lo:hi] == "normal")[0])
                    twins.append((len(rows) - 1, np.asarray(B0.T[j0]), B0.v["slot_valid"][j0]))
        out = collate(rows)
        if c.lam_p > 0:
            out["pidx"] = torch.tensor([t[0] for t in twins], dtype=torch.long)
            out["trunk0"] = torch.from_numpy(np.stack([t[1] for t in twins])) if twins else out["trunk"][:0]
            out["valid0"] = torch.from_numpy(np.stack([t[2] for t in twins])) if twins else out["valid"][:0]
        return out


def bank_row(S: Samples, B: Bank, i: int, j: int, role: str) -> dict:
    t, v = S.t, B.v
    kind = str(v["kind"][j])
    sv = v["slot_valid"][j]
    if kind == "offset":
        tgt = L.human_targets(recover_target(t["fut20"][i], float(v["dy"][j]), np.radians(float(v["dpsi"][j])), float(t["v0"][i]))[None])[0]
    else:
        tgt = L.human_targets(t["fut20"][i][None])[0]
    return {"trunk": np.asarray(B.T[j]), "valid": sv, "tc": t["tc"][i], "role": ROLES[role], "hum": tgt,
            "cam": np.float32(t["cam"][i][0]), "tgt": S.tea["out"][i], "tmu": S.tea["mu"][i], "dom": (DOMS + LAUNCH_DOMS).index(S.dom), "kind": kind}


def collate(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        if k == "kind":
            out[k] = [r[k] for r in rows]
        else:
            out[k] = torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows]))
    return out


# ---------------------------------------------------------------- forward from images
def trunks(net, imgs: torch.Tensor, chunk: int = 128) -> torch.Tensor:
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

    def __call__(self, o, b, o0=None):
        c, base = self.cfg, self.base
        out = o["outputs"].float()
        plan = base.plan(out)
        role = b["role"]
        L_ = {}
        tot0 = 0.0
        if o0 is not None and len(b["pidx"]):
            p0 = base.plan(o0["outputs"].float()).detach()
            L_["pair"] = base.imit_dist(plan[b["pidx"]], p0, b["cam"][b["pidx"]]).mean()
            tot0 = c.lam_p * L_["pair"]
        imit = (role == 1) | (role >= 3)
        w = torch.zeros_like(role, dtype=torch.float32)
        for r, k in (("U", 1), ("H", 3), ("O", 4), ("L", 5), ("S", 6), ("M", 7)):
            w = torch.where(role == k, torch.full_like(w, c.w_role.get(r, 1.0)), w)
        tot = 0.0
        if imit.any():
            d = base.imit_dist(plan[imit], None, b["cam"][imit], b["hum"][imit])
            L_["imit"] = (w[imit] * d).sum() / w[imit].sum()
            for r, k in (("U", 1), ("H", 3), ("O", 4), ("L", 5), ("S", 6), ("M", 7)):
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
        return tot + tot0, L_


# ---------------------------------------------------------------- probe / dev metrics
T_IDXS = A.T_IDXS


def psi3(plan: np.ndarray) -> np.ndarray:
    """Plan heading at 3 s (deg, left +) as op_common_cause's cc_report (plan channel 11 = yaw, openpilot frame)."""
    return -np.degrees(np.array([np.interp(3.0, T_IDXS, p[:, 11]) for p in plan]))


def plan_rear(plan: np.ndarray, cam_x) -> np.ndarray:
    """(n, 33, 15) -> rear-axle (n, 20, 2) on the 0.25 s grid (x fwd, y left)."""
    cam_x = np.broadcast_to(np.asarray(cam_x, float), (len(plan),))
    return np.stack([L.rear_np(p[None], float(c), T_FUT)[0] for p, c in zip(plan, cam_x)])
