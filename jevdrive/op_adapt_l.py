"""op-adapt L: log-imitation adaptation of openpilot Cinque (todos/2026-10-01-op-adapt-L-prereg.md).

Human log futures are the expert trajectory on three behaviour slices of WOD-E2E train (start from stop, stop onset, turn
onset); WOD routing intent is an optional navigation condition; the original model is the distillation teacher everywhere else.
Built on the op-adapt r2 infrastructure, which is only read: `Domain` (trunk memmaps + sample tables), the teacher arrays, the
distillation scale `tstd`, the exact PyTorch port.

  tables    prep/{wod,wodval,nus}.npz (scripts/op_adapt_l_prep.py): human future, kinematics, intent, slice flags per sample row
  LModel    the port with a chosen trainable set (stage 4 / off-policy plan pathway) and an intent condition
            (none / `ia` learned token embedding added to the hidden tokens / `id` the native desire input)
  Mixer     one batch = imitation rows (start / stop / turn onset) + contrast rows (stay / control / straight-intent) +
            other WOD rows + nuScenes train rows; every row also carries the teacher's outputs
  losses    L_imit (human future, rear-axle x / y / speed), L_distill (r2's normalised MSE on every output, plan block off on
            imitation rows), L_cons (plan pulled to the teacher's plan on non-imitation rows, same units as L_imit)
  metrics   capture / false-trigger / drift / slow / fast rates per row set and paired cluster bootstrap against the original
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import op_adapt as A
from . import op_adapt_r2 as R
from .common import data_dir

CTX = A.CONTEXT
AT = (0.275, 0.525)
T16 = 0.25 * np.arange(1, 17)                        # imitation grid (s)
T8 = 0.5 * np.arange(1, 9)                           # audit grid (s)
CAM_X = {"wod": 1.519, "nus": 1.70}                  # camera ahead of the rear axle (m)
SLICE3 = ("start", "stop", "turn_onset")
CONTRAST = ("stay", "control", "straight_int")
SIG_X, SIG_Y, SIG_V = 0.3 + 0.2 * T16, 0.1 + 0.1 * T16, 1.0     # L_imit / L_cons scales: metres, metres, m/s
POL_NODES = (479, 665)                                # cinque.ort.onnx: the off-policy temporal summarizer + plan hydra
FALSE_START_M, FALSE_TURN_M = 3.0, 2.0               # false-trigger definitions (prereg)


def lroot(*p) -> Path:
    """Lane root $DATA_DIR/runs/op_adapt_L (OP_L_ROOT overrides it: tests)."""
    d = Path(os.environ.get("OP_L_ROOT") or data_dir() / "runs" / "op_adapt_L") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def r2t() -> Path:
    """The r2 tables and teachers (read only)."""
    return Path(data_dir() / "runs" / "op_adapt_r2" / "t")


# ---------------------------------------------------------------- plan geometry
def rear_np(mu: np.ndarray, cam_x: float, ts=T8) -> np.ndarray:
    """(n, 33, >=12) openpilot plan (x fwd, y right at the camera, ch 11 = yaw) -> rear-axle x fwd / y left at ts, (n, K, 2)."""
    mu = np.asarray(mu, np.float64)
    W = R.t_weights(ts).astype(np.float64)
    x, y, psi = mu[..., 0] @ W.T, mu[..., 1] @ W.T, -(mu[..., 11] @ W.T)
    return np.stack([x + cam_x - cam_x * np.cos(psi), -y - cam_x * np.sin(psi)], -1).astype(np.float32)


class Grid:
    """Torch plan geometry on the 16-point imitation grid."""

    def __init__(self, dev):
        self.W = torch.as_tensor(R.t_weights(T16), device=dev)
        self.sx, self.sy = (torch.as_tensor(s, dtype=torch.float32, device=dev) for s in (SIG_X, SIG_Y))

    def rear(self, plan: torch.Tensor, cam_x: torch.Tensor):
        """plan (B, 33, 15), cam_x (B,) -> x, y (B, 16) rear-axle, speed (B, 16) (plan velocity channel)."""
        c = cam_x[:, None]
        x, y, psi, v = (plan[..., k] @ self.W.T for k in (0, 1, 11, 3))
        psi = -psi
        return x + c - c * torch.cos(psi), -y - c * torch.sin(psi), v


def human_targets(fut20: np.ndarray) -> np.ndarray:
    """(n, 20, 2) human future at 0.25 s -> (n, 16, 3): rear-axle x, y and speed (central difference over 0.5 s) at 0.25 ... 4 s."""
    P = np.concatenate([np.zeros_like(fut20[:, :1]), fut20], 1).astype(np.float64)       # P[:, i] = position at 0.25 i s
    v = np.linalg.norm(P[:, 2:18] - P[:, 0:16], axis=-1) / 0.5
    return np.concatenate([P[:, 1:17], v[..., None]], -1).astype(np.float32)


# ---------------------------------------------------------------- tables
def load_tables() -> dict:
    """prep tables as dicts of arrays aligned with the ROWS of their domain (nuScenes: rows without a slot are NaN / False)."""
    out = {}
    for dn in ("wod", "wodval"):
        z = dict(np.load(lroot("prep") / f"{dn}.npz", allow_pickle=True))
        out[dn] = z
    z = dict(np.load(lroot("prep") / "nus.npz", allow_pickle=True))
    dom = R.Domain("nus", r2t())
    n = len(dom)
    full = {}
    for k, v in z.items():
        if k in ("row", "log", "scene", "t", "split"):
            continue
        a = np.zeros((n,) + v.shape[1:], v.dtype) if v.dtype.kind in "bui" else np.full((n,) + v.shape[1:], np.nan, np.float32)
        a[z["row"]] = v
        full[k] = a
    full["in_tab"] = np.zeros(n, bool)
    full["in_tab"][z["row"]] = True
    full["split"] = dom.s.split.to_numpy().astype(str)                # r2's train / dev / val scenes
    full["seq"] = dom.s.group.to_numpy().astype(str)
    out["nus"] = full
    return out


class Data:
    """Domains, teachers and tables of the lane."""

    def __init__(self, names=("wod", "nus"), val=False):
        self.tab = load_tables()
        self.dom, self.tea = {}, {}
        for dn in names:
            root = lroot("t") if dn == "wodval" else r2t()
            self.dom[dn] = R.Domain(dn, root)
            self.tea[dn] = R.load_teacher(dn, root)
        self.full = {dn: (d.ctx >= 0).all(1) for dn, d in self.dom.items()}
        self.cam = {"wod": CAM_X["wod"], "wodval": CAM_X["wod"], "nus": CAM_X["nus"]}
        self.tstd = np.load(r2t() / "teacher" / "tstd.npy")

    def rows(self, dn: str, split: str, flag: str | None = None, need_future=False) -> np.ndarray:
        t = self.tab[dn]
        m = (np.asarray(t["split"]) == split) & self.full[dn]
        if flag is not None:
            m &= t[f"s_{flag}"]
        if need_future:
            m &= t["has"]
        if dn == "nus" and (flag is not None or need_future):
            m &= t["in_tab"]
        return np.flatnonzero(m)

    def intent(self, dn: str, rows) -> np.ndarray:
        return np.asarray(self.tab[dn]["intent"])[rows].astype(np.int64)


# ---------------------------------------------------------------- the model
@dataclass
class LCfg:
    name: str
    seed: int = 0
    s4: bool = True                       # stage 4 trainable
    pol: bool = False                     # off-policy plan pathway trainable
    intent: str = "ia"                    # none | ia (token embedding) | id (native desire input)
    slices: tuple = SLICE3                # imitated slices
    contrast: bool = True                 # stay / control / straight-intent rows in the batch
    dw: float = 1.0                       # multiplier on lam_d and lam_c
    steps: int = 4000
    batch: int = 64
    n_imit: int = 30
    n_contrast: int = 18
    n_other: int = 8
    n_nus: int = 8
    lam_i: float = 1.0
    lam_d: float = 10.0
    lam_c: float = 1.0
    contrast_dw: float = 2.0              # distillation weight of contrast rows relative to other rows
    lr: float = 3e-5
    lr_new: float = 3e-4
    wd: float = 0.01
    warmup: int = 100
    eval_every: int = 2000
    ckpt_every: int = 500
    prefetch: int = 4
    loaders: int = 3

    def dump(self) -> dict:
        return asdict(self)


def pol_weights(model="cinque") -> list[str]:
    """Float initializers (rank >= 1) consumed by the off-policy temporal summarizer and its plan hydra."""
    import onnx
    g = onnx.load(str(A.MODELS_DIR / A.FILES[model]), load_external_data=False).graph
    fl = {t.name for t in g.initializer if t.data_type in (1, 10, 11, 16) and len(t.dims) >= 1}
    return sorted({i for n in g.node[POL_NODES[0]:POL_NODES[1]] for i in n.input if i in fl})


class IntentAdapter(nn.Module):
    """H' = H + E[intent]: one learned (32, 512) embedding per WOD routing intent (1 straight, 2 left, 3 right) added to the
    hidden tokens of every context frame; intent 0 (unknown) adds nothing, so the unknown path is the original model."""

    def __init__(self, n=4, T=32, D=512):
        super().__init__()
        self.E = nn.Parameter(torch.zeros(n, T, D))
        self.register_buffer("on", (torch.arange(n) > 0).float()[:, None, None])

    def forward(self, H, intent):
        return H + (self.E * self.on)[intent][:, None].to(H.dtype)


def desire_from_intent(intent: torch.Tensor, dtype, block=27) -> torch.Tensor:
    """Native desire input carrying the routing intent: a rising-edge pulse of turnLeft (1) for WOD intent 2 / turnRight (2) for
    3 in desire block 27 (1.0 s before t0; the 33 blocks are 0.2 s each, block 32 = now; decisions 66's timing); straight /
    unknown = no desire. -> (B, 1, 33, 8)."""
    B = intent.shape[0]
    x = torch.zeros(B, 1, 33, 8, dtype=dtype, device=intent.device)
    for it, d in ((2, 1), (3, 2)):
        m = intent == it
        if m.any():
            x[m, :, block, d] = 1
    return x


class LModel(nn.Module):
    """Cinque port; trainable set and intent condition per LCfg. `forward(trunk, valid, tc, intent)` mirrors r2's Model."""

    def __init__(self, cfg: LCfg | None = None, dtype=torch.float16):
        super().__init__()
        cfg = cfg or LCfg("O", s4=False, intent="none")
        self.cfg = cfg
        tr = (A.stage4_weights() if cfg.s4 else []) + (pol_weights() if cfg.pol else [])
        self.net = A.load("cinque", dtype, trainable=tr)
        self.adapter = IntentAdapter() if cfg.intent == "ia" else None
        self.s4 = cfg.s4
        self.pol_names = set(pol_weights()) if cfg.pol else set()

    def forward(self, trunk, valid, tc, intent=None, action_t=AT):
        B = trunk.shape[0]
        run = lambda: self.net.run_batched({A.TRUNK_OUT: trunk.reshape(B * CTX, 1, *trunk.shape[2:]).to(self.net.dtype)},  # noqa: E731
                                           ["view_39"])["view_39"]
        if self.s4:
            H = run()
        else:
            with torch.no_grad():
                H = run()
        H = H.reshape(B, CTX, *A.H_SHAPE)
        if self.adapter is not None and intent is not None:
            H = self.adapter(H, intent)
        H = H * valid[:, :, None, None].to(H.dtype)
        f = A.policy_feeds(self.net, H, action_t, tc)
        if self.cfg.intent == "id" and intent is not None:
            f["_to_copy"] = desire_from_intent(intent, self.net.dtype)
        o = self.net.run_batched(f, A.POLICY_OUT)
        return {"outputs": o["outputs"].reshape(B, -1), "select_4": o["select_4"].reshape(B, -1)}

    def trainable(self):
        w = [(k, p) for k, p in self.net.params.items() if p.requires_grad]
        base = [p for k, p in w]
        new = list(self.adapter.parameters()) if self.adapter is not None else []
        return base, new

    def state(self) -> dict:
        return {"net": {k: p.detach().cpu() for k, p in self.net.params.items() if p.requires_grad},
                "adapter": self.adapter.state_dict() if self.adapter is not None else None}

    def load_state(self, st: dict):
        for k, v in st["net"].items():
            self.net.params[k].data.copy_(v)
        if self.adapter is not None and st.get("adapter") is not None:
            self.adapter.load_state_dict(st["adapter"])


def load_model(path: str | Path | None, dev) -> LModel:
    """None -> the original; else a checkpoint written by the trainer."""
    if path is None:
        return LModel(None).to(dev).eval()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    m = LModel(cfg).to(dev).eval()
    m.load_state(ck["model"])
    return m


# ---------------------------------------------------------------- batches
class Mixer:
    """Draws one batch worth of (domain, rows, role) per step. role: 1 imitation, 2 contrast, 0 other."""

    def __init__(self, cfg: LCfg, D: Data, split="train"):
        self.cfg, self.D = cfg, D
        self.pools = {}
        for s in cfg.slices:
            self.pools[("wod", s)] = D.rows("wod", split, s, need_future=True)
        if cfg.contrast:
            for s in CONTRAST:
                self.pools[("wod", s)] = D.rows("wod", split, s, need_future=True)
        excl = np.zeros(len(D.dom["wod"]), bool)
        for k, v in self.pools.items():
            excl[v] = True
        for s in SLICE3:                                    # any imitation slice frame stays out of the "other" pool
            excl[D.rows("wod", split, s, need_future=True)] = True
        allw = D.rows("wod", split)
        self.other = allw[~excl[allw]]
        self.nus = D.rows("nus", split)
        self.n_other = cfg.n_other + (0 if cfg.contrast else cfg.n_contrast)

    def describe(self) -> dict:
        return {f"{k[0]}:{k[1]}": int(len(v)) for k, v in self.pools.items()} | {"wod:other": int(len(self.other)), "nus": int(len(self.nus))}

    def draw(self, rng) -> list[tuple]:
        c = self.cfg
        segs = []
        ns = len(c.slices)
        cnt = np.full(ns, c.n_imit // ns)
        cnt[: c.n_imit - cnt.sum()] += 1
        for s, n in zip(c.slices, cnt):
            p = self.pools[("wod", s)]
            segs.append(("wod", p[rng.integers(len(p), size=n)], 1))
        if c.contrast:
            cnt = np.full(3, c.n_contrast // 3)
            cnt[: c.n_contrast - cnt.sum()] += 1
            for s, n in zip(CONTRAST, cnt):
                p = self.pools[("wod", s)]
                segs.append(("wod", p[rng.integers(len(p), size=n)], 2))
        segs.append(("wod", self.other[rng.integers(len(self.other), size=self.n_other)], 0))
        segs.append(("nus", self.nus[rng.integers(len(self.nus), size=c.n_nus)], 0))
        return segs


def assemble(D: Data, segs: list[tuple]) -> dict:
    xs, vs, tcs, intent, role, tout, tmu, hum, cam = [], [], [], [], [], [], [], [], []
    for dn, rows, r in segs:
        d, t = D.dom[dn], D.tea[dn]
        x, v = d.gather(rows)
        xs.append(x), vs.append(v), tcs.append(d.tc[rows])
        intent.append(D.intent(dn, rows)), role.append(np.full(len(rows), r, np.int64))
        tout.append(np.asarray(t["out"][rows])), tmu.append(np.asarray(t["mu"][rows], np.float32))
        h = np.full((len(rows), 16, 3), np.nan, np.float32)
        if r == 1:
            h = human_targets(np.asarray(D.tab[dn]["fut20"])[rows])
        hum.append(h), cam.append(np.full(len(rows), D.cam[dn], np.float32))
    cat = np.concatenate
    return {"trunk": cat(xs), "valid": cat(vs), "tc": cat(tcs), "intent": cat(intent), "role": cat(role), "tgt": cat(tout),
            "tmu": cat(tmu), "hum": cat(hum), "cam": cat(cam)}


def to_dev(b: dict, dev) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in b.items()}


# ---------------------------------------------------------------- losses
def huber(z: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)


class Losses:
    def __init__(self, net, cfg: LCfg, tstd: np.ndarray, dev):
        self.cfg, self.grid = cfg, Grid(dev)
        self.di = torch.as_tensor(A.distill_index(net.slices), device=dev)
        pi = A.plan_index(net.slices)
        self.pi = torch.as_tensor(pi, device=dev)
        self.plan_cols = torch.as_tensor(np.isin(A.distill_index(net.slices), pi), device=dev)
        self.tstd = torch.as_tensor(tstd, device=dev)

    def plan(self, out):
        return out[:, self.pi].view(-1, 33, 15)

    def imit_dist(self, plan, ref_plan_or_hum, cam, hum=None):
        """(B,) normalised distance mean_t sum_c Huber(e / sigma) between the plan and either a human path (hum (B,16,3)) or a
        reference plan (B, 33, 15), both on the rear-axle 16-point grid."""
        g = self.grid
        x, y, v = g.rear(plan, cam)
        if hum is None:
            hx, hy, hv = g.rear(ref_plan_or_hum, cam)
        else:
            hx, hy, hv = hum[..., 0], hum[..., 1], hum[..., 2]
        return (huber((x - hx) / g.sx) + huber((y - hy) / g.sy) + huber((v - hv) / SIG_V)).mean(1)

    def __call__(self, o: dict, b: dict) -> tuple[torch.Tensor, dict]:
        c = self.cfg
        out = o["outputs"].float()
        plan = self.plan(out)
        role = b["role"]
        imit = role == 1
        L = {}
        if imit.any():
            L["imit"] = self.imit_dist(plan[imit], None, b["cam"][imit], b["hum"][imit]).mean()
        oth = ~imit
        if oth.any():
            L["cons"] = self.imit_dist(plan[oth], b["tmu"][oth], b["cam"][oth]).mean()
        e = ((out[:, self.di] - b["tgt"].float()) / self.tstd).pow(2)                 # (B, Dd)
        keep = (~imit).float()
        num = e[:, ~self.plan_cols].sum(1) + keep * e[:, self.plan_cols].sum(1)
        w = torch.where(role == 2, torch.full_like(keep, c.contrast_dw), torch.ones_like(keep))
        L["distill"] = (w * num / e.shape[1]).sum() / w.sum()
        total = c.lam_i * L.get("imit", 0.0) + c.dw * c.lam_c * L.get("cons", 0.0) + c.dw * c.lam_d * L["distill"]
        return total, L


# ---------------------------------------------------------------- forward on rows (dev eval, readouts)
@torch.no_grad()
def fwd_rows(model: LModel, D: Data, dn: str, rows, dev, bs=192, threads=3, intent=True) -> dict:
    """plan (n, 33, 15), lead, lead_prob on sample rows of a domain; the intent of the row is fed unless intent=False."""
    d = D.dom[dn]
    sl = model.net.slices
    pi = A.plan_index(sl)
    lead = np.arange(sl["lead"].start, sl["lead"].start + 72)
    lp = np.arange(sl["lead_prob"].start, sl["lead_prob"].stop)
    rows = np.asarray(rows, np.int64)
    it = D.intent(dn, rows) if intent else np.zeros(len(rows), np.int64)
    starts = list(range(0, len(rows), bs))

    def load(i):
        r = rows[i:i + bs]
        x, v = d.gather(r)
        return torch.from_numpy(x).pin_memory(), torch.from_numpy(v), torch.from_numpy(d.tc[r]), torch.from_numpy(it[i:i + bs])
    from concurrent.futures import ThreadPoolExecutor
    acc = {"plan": [], "lead": [], "lead_prob": []}
    with ThreadPoolExecutor(threads) as ex:
        futs = {}
        for j, i in enumerate(starts):
            for jj in range(j, min(j + threads + 1, len(starts))):
                futs.setdefault(jj, ex.submit(load, starts[jj]))
            x, v, tc, itn = futs.pop(j).result()
            o = model(x.to(dev, non_blocking=True), v.to(dev), tc.to(dev), itn.to(dev))
            out = o["outputs"].float()
            acc["plan"].append(out[:, pi].view(-1, 33, 15).cpu().numpy())
            acc["lead"].append(out[:, lead].cpu().numpy())
            acc["lead_prob"].append(out[:, lp].cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


# ---------------------------------------------------------------- metrics
def cap_disp(p8, f8):
    return np.linalg.norm(p8[:, 7], axis=-1) >= 0.5 * np.linalg.norm(f8[:, 7], axis=-1)


def cap_stop(p8, f8=None):
    return np.linalg.norm(p8[:, 7] - p8[:, 6], axis=-1) / 0.5 <= 1.0


def cap_lat_end(p8, f8):
    return (np.sign(p8[:, 7, 1]) == np.sign(f8[:, 7, 1])) & (np.abs(p8[:, 7, 1]) >= 0.5 * np.abs(f8[:, 7, 1]))


CAPTURE = {"start": cap_disp, "stop": cap_stop, "turn_onset": cap_lat_end}


def row_metrics(plan: np.ndarray, tab: dict, rows: np.ndarray, cam: float, v0=None) -> dict:
    """Per-row indicator / error arrays of one model's plans on `rows` (all (n,)): capture bits of the three slices (meaningful
    on their slice rows), false-trigger bits (start: |p4s| >= 3 m; stop: plan stands by 4 s; turn: |y4s| >= 2 m), the plan speed
    class at 2 s, and lon / lat ADE against the human future."""
    p8 = rear_np(plan, cam)
    f8 = np.asarray(tab["fut"])[rows].astype(np.float32)
    v0 = np.asarray(tab["v0"])[rows].astype(float) if v0 is None else v0
    e = p8.astype(np.float64) - f8
    v2 = plan[:, :, 3] @ R.t_weights([2.0])[0]
    thr = np.maximum(1.0, 0.2 * v0)
    return {"cap_start": cap_disp(p8, f8), "cap_stop": cap_stop(p8), "cap_turn_onset": cap_lat_end(p8, f8),
            "false_start": np.linalg.norm(p8[:, 7], axis=-1) >= FALSE_START_M, "false_stop": cap_stop(p8),
            "false_turn": np.abs(p8[:, 7, 1]) >= FALSE_TURN_M, "slow": v2 < v0 - thr, "fast": v2 > v0 + thr,
            "lon3": np.abs(e[:, :6, 0]).mean(1), "lat3": np.abs(e[:, :6, 1]).mean(1),
            "ade4": np.linalg.norm(e, axis=-1).mean(1)}


def cluster_boot(groups: np.ndarray, B=2000, seed=0):
    """Cluster indices and the (B, L) multinomial resample weights (shared by every metric of one row set)."""
    ug, gi = np.unique(groups.astype(str), return_inverse=True)
    L = len(ug)
    W = np.random.default_rng(seed).multinomial(L, np.full(L, 1.0 / L), size=B).astype(np.float64)
    return gi, L, W


def boot_mean(x: np.ndarray, gi, L, W) -> tuple:
    """mean and 95% cluster-bootstrap CI of x (n,) [float]."""
    s = np.bincount(gi, weights=x.astype(np.float64), minlength=L)
    n = np.bincount(gi, minlength=L).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ s) / (W @ n)
    return float(s.sum() / max(n.sum(), 1)), tuple(np.nanpercentile(d, [2.5, 97.5]))


def paired_delta(xa: np.ndarray, xo: np.ndarray, groups: np.ndarray, B=2000, seed=0) -> dict:
    """mean(xa) - mean(xo) of one row set with a cluster bootstrap (same resample for both)."""
    if len(xa) == 0:
        return {"n": 0}
    gi, L, W = cluster_boot(groups, B, seed)
    m, ci = boot_mean(xa.astype(float) - xo.astype(float), gi, L, W)
    return {"n": int(len(xa)), "clusters": int(L), "adapt": float(xa.mean()), "orig": float(xo.mean()), "delta": m,
            "lo": float(ci[0]), "hi": float(ci[1])}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
