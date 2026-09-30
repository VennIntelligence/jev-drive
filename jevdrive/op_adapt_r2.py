"""op-adapt round 2 (todos/2026-09-29-op-adapt-r2-prereg.md, v3): arms, data, sampler, losses, model.

Everything the trainer (scripts/op_adapt_r2_train.py) and the readouts (scripts/op_adapt_r2_readout.py) share.
"§" = a section of the pre-registration; the interface with the data (C), scorer (S) and detection (D) packages is
tmp/2026-09-30-op-adapt-r2-build.md.

  storage   every domain = one flat trunk memmap  R2/t/trunk/<domain>.npy  (N, 1024, 8, 16) fp16 (shared page cache
            across the concurrent training processes) + a sample table  R2/t/samples/<domain>.parquet: one row per
            sample (a 5 Hz slot; a sim quad member is its own sample) with ctx0..ctx8 = global trunk rows of its
            9 context frames (oldest first, -1 = zero hidden state), traffic, split, group and the label columns.
  teacher   R2/t/teacher/<domain>.npz: the original model on every sample (same fp16 numeric path as the student):
            distilled outputs (fp16), plan MDN mean / log-std, temporal token.
  arms      ARMS: the §2 table as data; a run = (arm, seed, lambda_s).
  sampler   Mixer: §3.3 mixing (sim / real by sequence, WOD : nuScenes, offset slots and their twins, the parallel
            distillation stream, size / distance oversampling, nuScenes VRU x2) -> one flat batch of sample rows.
  losses    L_aux (round 1), L_pair, L_score, L_dir, L_distill (round 1), L_dplan (A-bhv) on one forward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import op_adapt as A
from .common import data_dir

CTX = A.CONTEXT
TRUNK_SHAPE = (1024, 8, 16)
T4 = np.flatnonzero(A.T_IDXS <= 4.0)                    # 21 plan points (t = 0 .. 3.9 s)
CH = (0, 1, 3, 6)                                       # plan channels x, y, v (vel x), a (acc x) of the (33, 15) mean
V_T = (1.0, 2.0, 3.0, 4.0)                              # L_dir speed checkpoints (s)
DV_T = (1.0, 2.0, 3.0)                                  # L_dplan checkpoints (s)
PX_VIS, PX_MAIN = 68.0, 500.0                           # P5 visibility floor, Q1 main size band (px_eq)
LAT_NEAR, LAT_FAR = 1.75 + 1.0, 1.75 + 3.0              # L_dir gates (m)
SLOW_CANDS = ("op_stop", "brake_hard", "brake_mild")
LAM_D, LAM_B, DELTA = 10.0, 1.0, 0.1
LAM_S_GRID = (0.3, 1.0)
SEQ_PER_ARM = 1_200_000
BATCH = 64                                              # labelled sequences per step (a sim quad = 4)
DOMAINS_REAL = ("wod", "nus")


def r2(*p) -> Path:
    """Run root $DATA_DIR/runs/op_adapt_r2 (OP_R2_ROOT overrides it: self-tests, pilots)."""
    import os
    d = Path(os.environ.get("OP_R2_ROOT") or data_dir() / "runs" / "op_adapt_r2") / Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def t_weights(ts, grid=A.T_IDXS) -> np.ndarray:
    """(len(ts), 33) linear-interpolation weights of the plan's time grid at times ts."""
    W = np.zeros((len(ts), len(grid)), np.float32)
    for i, t in enumerate(ts):
        j = int(np.clip(np.searchsorted(grid, t) - 1, 0, len(grid) - 2))
        w = (t - grid[j]) / (grid[j + 1] - grid[j])
        W[i, j], W[i, j + 1] = 1 - w, w
    return W


# ---------------------------------------------------------------- arms (§2)
@dataclass(frozen=True)
class Arm:
    name: str
    stage4: bool = True                  # stage 4 trainable
    det: bool = False                    # detection-token adapter (package D)
    sim: str = "CK"                      # labelled sim renders ("" = sim only in distillation)
    real: bool = True                    # labelled real (WOD, nuScenes); False = real only in distillation
    offset: bool = True                  # v3 offset slots (+ twins in the distillation stream)
    losses: tuple = ("aux", "pair", "score", "dir", "distill")
    seeds: int = 1
    note: str = ""

    @property
    def score_domains(self):             # where L_score applies (§2.1)
        d = set()
        if "score" in self.losses:
            d |= {f"sim{r}" for r in self.sim}
            if self.real:
                d.add("nus")
            if self.offset:
                d.add("off")
        return d


ARMS = {a.name: a for a in (
    Arm("O", stage4=False, sim="", real=False, offset=False, losses=(), seeds=0, note="original model, reference"),
    Arm("A", seeds=3, note="main: vision-only B, sim + real"),
    Arm("D", det=True, seeds=3, note="main: A + detection-token adapter"),
    Arm("A-real", sim="", losses=("aux", "score", "distill"), note="real only (sim only distilled)"),
    Arm("A-sim", real=False, offset=False, note="sim pairs only (real only distilled)"),
    Arm("A-noC", sim="K", note="no CARLA originals"),
    Arm("A-noK", sim="C", note="no Cosmos"),
    Arm("D-only", stage4=False, det=True, note="detection adapter + aux heads, stage 4 frozen"),
    Arm("A-bhv", offset=False, losses=("aux", "pair", "dplan", "distill"), note="control: v1 L_dplan instead of the scorer"),
    Arm("Z", note="tele-view adapter; opens only if R0 passes (not built)"),
)}
STAGE10 = ("A@0.3", "A@1", "D", "A-real", "A-sim", "A-noC", "A-noK", "D-only", "A-bhv")   # §6 ~10 units


@dataclass
class RunCfg:
    arm: str
    seed: int = 0
    lam_s: float = 1.0
    lam_d: float = LAM_D
    lam_b: float = LAM_B
    seqs: int = SEQ_PER_ARM
    batch: int = BATCH
    lr: float = 3e-5
    lr_new: float = 1e-3
    wd: float = 0.01
    warmup: int = 200
    distill_frac: float = 0.25           # parallel distillation stream, fraction of `batch`
    offset_frac: float = 0.10            # offset slots, fraction of `batch`, taken from the real labelled share
    sim_frac: float = 0.5
    wod_nus: tuple = (0.6, 0.4)
    distill_mix: dict = field(default_factory=lambda: {"wod": 0.4, "nus": 0.3, "nav": 0.3})   # + sim x- in the stream
    pos_rate: float = 0.25               # effective positive rate of the main BCE per domain (pos_weight)
    eval_every: int = 1000
    ckpt_every: int = 500
    max_steps: int = 0                   # 0 = seqs / batch (staged launch: 2 000 steps or 10 %)
    round1: bool = False                 # round-1 compatibility (equivalence check): nus only, aux + distill

    @property
    def steps(self) -> int:
        return self.max_steps or int(np.ceil(self.seqs / self.batch))

    @property
    def tag(self) -> str:
        return f"{self.arm}-s{self.seed}" + (f"-ls{self.lam_s:g}" if "score" in ARMS[self.arm].losses else "")


# ---------------------------------------------------------------- storage
class Domain:
    """Flat trunk memmap + sample table of one domain."""

    def __init__(self, name: str, root: Path | None = None):
        root = root or r2("t")
        self.name = name
        self.T = np.load(root / "trunk" / f"{name}.npy", mmap_mode="r")
        self.s = pd.read_parquet(root / "samples" / f"{name}.parquet")
        self.ctx = self.s[[f"ctx{k}" for k in range(CTX)]].to_numpy(np.int64)
        self.tc = self.s[["tc0", "tc1"]].to_numpy(np.float32)
        self.uid = self.s.uid.to_numpy(np.int64)
        self.pos = pd.Series(np.arange(len(self.s)), index=self.uid)

    def __len__(self):
        return len(self.s)

    def col(self, c, default=np.nan, dtype=None):
        v = self.s[c].to_numpy() if c in self.s else np.full(len(self.s), default)
        return v.astype(dtype) if dtype else v

    def gather(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, 9, 1024, 8, 16) fp16 context trunks (zero rows where ctx = -1) and the (B, 9) validity."""
        c = self.ctx[rows]
        valid = c >= 0
        x = np.zeros((len(rows), CTX) + TRUNK_SHAPE, np.float16)
        f = np.flatnonzero(valid.ravel())
        if len(f):
            g = c.ravel()[f]
            o = np.argsort(g)                                   # sorted reads from the memmap
            x.reshape(-1, *TRUNK_SHAPE)[f[o]] = self.T[g[o]]
        return x, valid


def pack(name: str, index: pd.DataFrame, root: Path | None = None, workers: int = 8) -> None:
    """Build R2/t/trunk/<name>.npy + samples/<name>.parquet from per-stream cache files.
    index: one row per sample with `cache` (npz path, absolute or relative to $DATA_DIR), `row` (slot in that file),
    `lctx` (9 file-local rows, oldest first, -1 = zero hidden) and every other column to keep (uid, split, labels...).
    Only the trunk rows some sample reads are copied."""
    from concurrent.futures import ThreadPoolExecutor
    root = root or r2("t")
    (root / "trunk").mkdir(parents=True, exist_ok=True)
    (root / "samples").mkdir(parents=True, exist_ok=True)
    files = sorted(index.cache.unique())
    lctx = np.stack(index.lctx.to_numpy()).astype(np.int64)
    fid = pd.Series(np.arange(len(files)), index=files)[index.cache].to_numpy()
    order = np.argsort(fid, kind="stable")
    grp = np.split(order, np.cumsum(np.bincount(fid, minlength=len(files)))[:-1])      # sample rows per file
    need = [np.unique(lctx[g][lctx[g] >= 0]) for g in grp]
    base = np.r_[0, np.cumsum([len(n) for n in need])]
    tmp = root / "trunk" / f"{name}.tmp.npy"
    out = np.lib.format.open_memmap(tmp, "w+", np.float16, (int(base[-1]),) + TRUNK_SHAPE)

    def load(k):
        p = Path(files[k])
        p = p if p.is_absolute() else data_dir() / p
        if p.suffix == ".npy":                                  # package C's sim / offset caches
            out[base[k]:base[k + 1]] = np.load(p, mmap_mode="r")[need[k]]
        else:
            with np.load(p) as z:
                out[base[k]:base[k + 1]] = z["trunk"][need[k]]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(load, range(len(files))))
    out.flush()
    g = np.full_like(lctx, -1)
    for k, rows in enumerate(grp):
        loc = lctx[rows]
        g[rows] = np.where(loc >= 0, base[k] + np.searchsorted(need[k], np.maximum(loc, 0)), -1)
    s = index.drop(columns=["lctx"]).reset_index(drop=True).copy()
    for k in range(CTX):
        s[f"ctx{k}"] = g[:, k]
    if "uid" not in s:
        s["uid"] = np.arange(len(s), dtype=np.int64)
    s.to_parquet(root / "samples" / f"{name}.parquet", index=False)
    tmp.replace(root / "trunk" / f"{name}.npy")


# ---------------------------------------------------------------- round-1 caches -> pack indices
def _split_perm(keys, frac_dev, seed=0):
    u = np.array(sorted(set(keys)))
    dev = set(np.random.default_rng(seed).permutation(u)[: int(round(frac_dev * len(u)))])
    return np.array(["dev" if k in dev else "train" for k in keys])


def index_nus(scenes=None) -> pd.DataFrame:
    """nuScenes (round-1 cache processed/op_adapt/nusc, stride 2): every slot of every trainval scene, labels on the
    keyframes, `normal` as round 1 (scripts/op_adapt_train.Store), split train / dev (round-1 50 scenes) / val."""
    from . import op_adapt_data as D
    lab = pd.read_parquet(D.root() / "nusc_labels.parquet")
    L = lab.set_index("token")
    tr = np.array(sorted(lab[lab.split == "train"].scene.unique()))
    dev = set(np.random.default_rng(0).permutation(tr)[:50])            # = scripts/op_adapt_train.split_scenes
    have = {p.stem for p in D.root("nusc").glob("*.npz") if not p.stem.endswith(".tmp")}
    scenes = [s for s in (scenes or sorted(lab.scene.unique())) if s in have]
    off = np.arange(-2 * (CTX - 1), 1, 2)
    rows = []
    for s in scenes:
        f = D.root("nusc") / f"{s}.npz"
        with np.load(f) as z:
            n, tc = len(z["steps"]), z["traffic"]
            key = {int(k): str(t) for k, t in zip(z["key_slot"], z["tokens"])}
        kn = {k: not bool(L.at[t, "vru_wide"]) for k, t in key.items()}
        sp = "val" if L.at[next(iter(key.values())), "split"] == "val" else ("dev" if s in dev else "train")
        for j in range(n):
            k0 = 5 * (j // 5)
            loc = j + off
            rows.append({"cache": str(f), "row": j, "lctx": np.where(loc >= 0, loc, -1), "key": f"{s}/{j}", "scene": s,
                         "token": key.get(j, ""), "normal": kn.get(k0, False) and (j % 5 == 0 or kn.get(k0 + 5, False)),
                         "tc0": float(tc[0]), "tc1": float(tc[1]), "split": sp, "group": s})
    ix = pd.DataFrame(rows)
    k = ix.token != ""
    lk = L.reindex(ix.token[k])
    for c in ("ped_corr", "ped_wide", "vru_wide", "vru_corr"):
        ix[c] = False
        ix.loc[k, c] = lk[c].to_numpy(bool)
    ix["ped_dist"] = np.nan
    ix.loc[k, "ped_dist"] = lk.ped_dist.to_numpy(float)
    ix["labelled"] = k.to_numpy()
    ix["uncertain"] = False
    ix["dist_bin"] = -1
    ix.loc[k, "dist_bin"] = D.dist_bin(lk.reset_index())
    ix["domain"] = "nus"
    return ix


def index_stream(ds: str, labels: pd.DataFrame | None = None, frac_dev: float = 0.0, targets_only=True) -> pd.DataFrame:
    """Stream caches with `names` / `targets` / `stride` (wodtrain, wod val, p5): samples = the target slots."""
    from . import op_adapt_data as D
    rows = []
    for f in sorted(D.root(ds).glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        with np.load(f) as z:
            c, names, tg, tc = int(z["stride"]), z["names"].astype(str), z["targets"], z["traffic"]
        for j in (tg if targets_only else range(len(names))):
            loc = j - c * np.arange(CTX - 1, -1, -1)
            rows.append({"cache": str(f), "row": int(j), "lctx": np.where(loc >= 0, loc, -1), "key": names[j],
                         "stream": f.stem, "tc0": float(tc[0]), "tc1": float(tc[1])})
    ix = pd.DataFrame(rows)
    if ds in ("wodtrain", "wod"):
        ix["group"] = ix.key.str.rsplit("-", n=1).str[0]
    else:
        ix["group"] = ix.stream
    if labels is not None:
        lab = labels.set_index("frame_id")
        lk = lab.reindex(ix.key)
        ix["labelled"] = lk.ped_corr.notna().to_numpy()
        for c in ("ped_corr", "ped_wide", "vru_wide", "vru_corr", "uncertain"):
            ix[c] = lk[c].fillna(False).to_numpy(bool)
        ix["ped_dist"] = lk.ped_dist.to_numpy(float)
        ix["dist_bin"] = lk.dist_bin.fillna(-1).to_numpy(int)
        ix["normal"] = ix.labelled & ~ix.vru_wide & ~ix.uncertain
    ix["split"] = _split_perm(ix.group, frac_dev) if frac_dev else ("val" if ds == "wod" else "test")
    ix["domain"] = {"wodtrain": "wod", "wod": "wodval", "p5": "p5"}.get(ds, ds)
    return ix


def index_nav() -> pd.DataFrame:
    """navtrain (round-1 cache: one file per log, `ctx` per token): distillation only; split 95 / 5 by log (seed 0)."""
    from . import op_adapt_data as D
    lab = pd.read_parquet(D.root() / "navtrain_labels.parquet").set_index("token")
    rows = []
    for f in sorted(D.root("navtrain").glob("*.npz")):
        if f.stem.endswith(".tmp"):
            continue
        with np.load(f) as z:
            ctx, toks, tc = z["ctx"], z["tokens"].astype(str), z["traffic"]
        for i, t in enumerate(toks):
            rows.append({"cache": str(f), "row": int(ctx[i, -1]), "lctx": ctx[i].astype(np.int64), "key": t, "log": f.stem,
                         "tc0": float(tc[i, 0]), "tc1": float(tc[i, 1])})
    ix = pd.DataFrame(rows)
    lk = lab.reindex(ix.key)
    ix["vru_wide"] = lk.vru_wide.fillna(True).to_numpy(bool)
    ix["normal"] = ~ix.vru_wide
    ix["labelled"] = False
    ix["group"] = ix.log
    ix["split"] = _split_perm(ix.log, 0.05)
    ix["domain"] = "nav"
    return ix


def index_c(domain: str) -> pd.DataFrame:
    """Package C's index/<domain>.parquet -> a pack index (cache = `file`, lctx = `ctx`); every column kept.
    Sim rows get `render` (C / K) from the domain name."""
    ix = pd.read_parquet(r2() / "index" / f"{domain}.parquet")
    ix = ix.rename(columns={"file": "cache", "labeled": "labelled"})
    ix["lctx"] = [np.asarray(c, np.int64) for c in ix.ctx]
    ix = ix.drop(columns=["ctx"])
    if domain.startswith("sim"):
        ix["render"] = domain[-1]
    if "group" not in ix:
        ix["group"] = ix.inst.astype(str) if "inst" in ix else ix.key.astype(str)
    return ix


# ---------------------------------------------------------------- teacher
TEACHER_ARRAYS = ("uid", "out", "mu", "logstd", "temporal")


def load_teacher(name: str, root: Path | None = None) -> dict:
    """R2/t/teacher/<domain>/<array>.npy, memory-mapped, row-aligned with the sample table."""
    d = (root or r2("t")) / "teacher" / name
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in TEACHER_ARRAYS if (d / f"{k}.npy").exists()}


def sigma_o(logstd: np.ndarray) -> np.ndarray:
    """(..., 33, 15) MDN log-std -> sigma on the L_score channels (..., 21, 4), openpilot's exp(min(., 11))."""
    return np.exp(np.minimum(np.asarray(logstd, np.float32)[..., T4, :][..., list(CH)], 11.0)).clip(1e-3)


# ---------------------------------------------------------------- scorer tables (package S) and gates
def load_score(name: str) -> dict | None:
    """score/<domain>.npz of package S (build doc §S), or None while it is not delivered."""
    p = r2() / "score" / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["names"] = [str(x) for x in d.get("cands", d.get("names"))]
    d["traj"] = d.get("traj", d.get("cand"))
    d["pos"] = pd.Series(np.arange(len(d["uid"])), index=d["uid"])
    return d


def dir_gates(px_eq, ped_lat, top_eq) -> tuple[np.ndarray, np.ndarray]:
    """§2.1 L_dir gates per sim pair slot. g_dir: visible and >= 500 px_eq and the pedestrian's 4 s true future within
    1.75 + 1.0 m of the op / hold path. g_eq: invisible (< 68 px_eq), or visible but > 1.75 + 3.0 m from the path and
    x+ / x- Top sets equal. Everything else (small-and-near, between the two distances) gets neither."""
    px, lat = np.asarray(px_eq, float), np.asarray(ped_lat, float)
    vis = px >= PX_VIS
    g_dir = vis & (px >= PX_MAIN) & (lat <= LAT_NEAR)
    g_eq = ~vis | (vis & (lat > LAT_FAR) & np.asarray(top_eq, bool))
    return g_dir, g_eq & ~g_dir


# ---------------------------------------------------------------- losses
def huber_d(mu: torch.Tensor, tau: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """d(mu, tau) = mean_t sum_c Huber((mu - tau) / sigma). mu (B, 21, 4), tau (B, K, 21, 4), sig (B, 21, 4) -> (B, K)."""
    z = (mu[:, None] - tau) / sig[:, None]
    return F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0).sum(-1).mean(-1)


def plan_ch(plan: torch.Tensor) -> torch.Tensor:
    """(B, 33, 15) plan -> (B, 21, 4) channels x, y, v, a on T_IDXS <= 4 s."""
    return plan[:, T4][:, :, list(CH)]


def loss_aux(pred, y, pw=None):
    """Round-1 L_aux (scripts/op_adapt_train.loss_aux) with an optional per-sample pos_weight of the main BCE."""
    l = F.binary_cross_entropy_with_logits(pred[:, 0], y[:, 0], pos_weight=pw) + \
        F.binary_cross_entropy_with_logits(pred[:, 1], y[:, 1])
    pos = y[:, 0] > 0.5
    if pos.any():
        l = l + 0.1 * F.smooth_l1_loss(pred[pos, 2], y[pos, 2])
    return l


def loss_pair(z_plus, z_minus):
    """softplus(1 - (z+ - z-)) mean over the given visible pairs."""
    return F.softplus(1 - (z_plus - z_minus)).mean()


def loss_score(mu, cand, top, sig):
    """Winner-take-all over the Top set: min_{k in Top} d(mu, tau_k), mean over rows. mu (B, 21, 4)."""
    d = huber_d(mu, cand, sig)
    return d.masked_fill(~top, float("inf")).amin(1).mean()


def loss_dir(v_plus, v_minus, sig_v, g_dir, mu_plus, mu_minus, sig, g_eq):
    """g_dir * mean_t relu(v+ - v-) / sigma_v + g_eq * d(mu+, sg[mu-]); mean over gated pairs."""
    g_dir, g_eq = g_dir.float(), g_eq.float()
    a = (F.relu(v_plus - v_minus) / sig_v).mean(1)
    b = huber_d(mu_plus, mu_minus.detach()[:, None], sig)[:, 0]
    return (a * g_dir + b * g_eq).sum() / (g_dir + g_eq).clamp(max=1).sum().clamp_min(1)


def loss_dplan(dv_hat, dv_star, sd):
    return F.huber_loss(dv_hat / sd, dv_star / sd, delta=1.0)


def loss_distill(out, tgt, tstd, m):
    e = ((out - tgt) / tstd).pow(2).mean(1)
    return (e * m).sum() / m.sum().clamp_min(1)


# ---------------------------------------------------------------- model
class Model(torch.nn.Module):
    """Cinque port (stage 4 trainable or frozen) + round-1 aux heads + an optional hidden-state hook (package D's
    detection adapter) between stage 4 and the policy. Same fp16 numeric path as round 1 (op_adapt.stage4_policy)."""

    def __init__(self, arm: Arm, dtype=torch.float16, det: torch.nn.Module | None = None):
        super().__init__()
        self.net = A.load("cinque", dtype, trainable=A.stage4_weights() if arm.stage4 else ())
        self.heads = A.AuxHeads()
        self.det = det
        self.stage4_train = arm.stage4

    def forward(self, trunk, valid, tc, det_tok=None, det_mask=None, action_t=(0.275, 0.525)):
        B = trunk.shape[0]
        run = lambda: self.net.run_batched({A.TRUNK_OUT: trunk.reshape(B * CTX, 1, *trunk.shape[2:]).to(self.net.dtype)},  # noqa: E731
                                           ["view_39"])["view_39"]
        if self.stage4_train:
            H = run()
        else:
            with torch.no_grad():
                H = run()
        H = H.reshape(B, CTX, *A.H_SHAPE)
        if self.det is not None:
            H = self.det(H, det_tok, det_mask)
        o = A._policy(self.net, H, action_t, tc, valid)
        o["head_t"], o["head_v"] = self.heads(o["select_4"], o["tokens"])
        return o

    def trainable(self):
        s4 = [p for p in self.net.parameters() if p.requires_grad]
        new = list(self.heads.parameters()) + (list(self.det.parameters()) if self.det is not None else [])
        return s4, new

    def state(self) -> dict:
        return {"stage4": {k: p.detach().cpu() for k, p in self.net.params.items() if p.requires_grad},
                "heads": self.heads.state_dict(), "det": self.det.state_dict() if self.det is not None else None}

    def load_state(self, st: dict):
        for k, v in st["stage4"].items():
            self.net.params[k].data.copy_(v)
        self.heads.load_state_dict(st["heads"])
        if self.det is not None and st.get("det") is not None:
            self.det.load_state_dict(st["det"])


def det_adapter():
    """Package D's adapter (jevdrive/op_adapt_det.py); None until it is delivered."""
    try:
        from . import op_adapt_det as DD
    except ImportError:
        return None
    return DD.make_adapter()


def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def cfg_dict(c: RunCfg) -> dict:
    return asdict(c) | {"arm_def": asdict(ARMS[c.arm]), "steps": c.steps}


# ---------------------------------------------------------------- sampler (§3.3)
def split_counts(n: int, fracs) -> list[int]:
    """Largest-remainder split of n into len(fracs) integer parts proportional to fracs."""
    f = np.asarray(fracs, float) / np.sum(fracs) * n
    k = np.floor(f).astype(int)
    k[np.argsort(-(f - k))[: n - k.sum()]] += 1
    return k.tolist()


class Pool:
    """Rows of one domain with sampling weights (drawn with replacement, like round 1)."""

    def __init__(self, dom: str, rows: np.ndarray, w: np.ndarray | None = None):
        self.dom, self.rows = dom, np.asarray(rows, np.int64)
        w = np.ones(len(self.rows)) if w is None else np.asarray(w, float)
        self.cum = np.cumsum(w) / w.sum() if len(self.rows) else np.zeros(0)
        self.w = w

    def draw(self, rng, n) -> np.ndarray:
        if n <= 0 or not len(self.rows):
            return np.zeros(0, np.int64)
        return self.rows[np.minimum(np.searchsorted(self.cum, rng.random(n), side="right"), len(self.rows) - 1)]


@dataclass
class Seg:
    dom: str
    rows: np.ndarray
    aux: np.ndarray
    distill: np.ndarray
    score: np.ndarray
    npair: int = 0                        # sim segments: the first npair rows are x+, the next npair their x-


def dv_star(d: "Domain") -> np.ndarray:
    """(n, 3) A-bhv targets dv*(1, 2, 3 s) = v+ - v- (package C: dv_star_<t>; NaN beyond the record)."""
    return np.stack([d.col(f"dv_star_{t}", np.nan, float) if f"dv_star_{t}" in d.s else d.col(f"dv{t}", np.nan, float)
                     for t in (1, 2, 3)], 1).astype(np.float32)


def px_weight(px):
    px = np.asarray(px, float)
    return np.where(((px >= 100) & (px < 500)) | ((px >= 500) & (px < 1500)), 2.0, 1.0)


def dist_weight(ped_corr, dist):
    d = np.asarray(dist, float)
    return np.where(np.asarray(ped_corr, bool) & (d >= 10) & (d < 30), 2.0, 1.0)


class Mixer:
    """§3.3 batch composition for one arm. Per step: `batch` labelled sequences (sim share as quads / pairs, real
    share WOD : nuScenes 60 : 40 after the offset slots), + the parallel distillation stream (distill_frac * batch:
    25 % sim x-, 75 % real normal frames WOD : nuScenes : navtrain 40 : 30 : 30) + one twin (unshifted navtrain frame,
    always distilled) per offset slot."""

    def __init__(self, cfg: RunCfg, doms: dict, scores: dict | None = None, split: str = "train"):
        self.cfg, self.arm, self.doms = cfg, ARMS[cfg.arm], doms
        scores = scores or {}
        arm = self.arm
        self.renders = arm.sim
        B = cfg.batch
        n_sim = (int(round(B * cfg.sim_frac)) if arm.real else B) if arm.sim else 0
        self.n_slot = n_sim // (2 * len(arm.sim)) if arm.sim else 0
        n_real = B - n_sim if arm.real else 0
        self.n_off = int(round(cfg.offset_frac * B)) if arm.offset and n_real else 0
        self.n_wod, self.n_nus = split_counts(n_real - self.n_off, cfg.wod_nus) if n_real else (0, 0)
        n_d = int(round(cfg.distill_frac * B))
        self.n_dsim = int(round(0.25 * n_d))
        dm = cfg.distill_mix
        self.n_dreal = dict(zip(dm, split_counts(n_d - self.n_dsim, list(dm.values()))))
        # sim quads: rows of x+ / x- per render, aligned by slot
        self.quad = {}
        if arm.sim:
            b = doms[f"sim{arm.sim[0]}"]
            s = b.s
            base = np.flatnonzero((s.split.to_numpy() == split) & (s.sign.to_numpy() == 1))
            cols = {}
            for r in arm.sim:
                d = doms[f"sim{r}"]
                up = b.uid[base] if r == arm.sim[0] else b.col("other_uid")[base]
                ok = pd.Index(d.uid).get_indexer(up)
                um = np.where(ok >= 0, d.col("twin_uid")[np.maximum(ok, 0)], -1)
                om = pd.Index(d.uid).get_indexer(um)
                cols[r] = (ok, om)
            good = np.logical_and.reduce([(p >= 0) & (m >= 0) for p, m in cols.values()])
            self.quad = {r: (p[good], m[good]) for r, (p, m) in cols.items()}
            px = b.col("px_eq")[base[good]]
            self.slot_w = px_weight(px)
            self.slot_cum = np.cumsum(self.slot_w) / self.slot_w.sum()
            self.n_slots_pool = int(good.sum())
        # real labelled
        self.real = {}
        for dn, n in (("wod", self.n_wod), ("nus", self.n_nus)):
            if n and dn in doms:
                d = doms[dn]
                m = (d.col("split") == split) & d.col("labelled", False, bool)
                rows = np.flatnonzero(m)
                w = dist_weight(d.col("ped_corr", False, bool)[rows], d.col("ped_dist")[rows])
                if dn == "nus":
                    w = w * np.where(self.nus_vru(d, scores.get("nus"))[rows], 2.0, 1.0)
                self.real[dn] = Pool(dn, rows, w)
        self.off = Pool("off", np.flatnonzero(doms["off"].col("split") == split)) if self.n_off else None
        # distillation stream
        self.dsim = []
        if self.n_dsim:
            for r in (arm.sim or "CK"):
                d = doms.get(f"sim{r}")
                if d is not None:
                    self.dsim.append(Pool(f"sim{r}", np.flatnonzero((d.col("split") == split) & (d.col("sign") == -1))))
        self.dreal = {dn: Pool(dn, np.flatnonzero((doms[dn].col("split") == split) & doms[dn].col("normal", False, bool)))
                      for dn in dm if dn in doms}
        self.pw = self.pos_weights()

    @staticmethod
    def nus_vru(d: Domain, sc: dict | None) -> np.ndarray:
        """nuScenes VRU frames (the real L_score rows): rows of S's nus table with valid, else the wide-corridor VRU flag."""
        if sc is not None:
            v = pd.Series(sc["valid"], index=sc["uid"]).reindex(d.uid).fillna(False).to_numpy(bool)
            return v
        return d.col("vru_wide", False, bool)

    def pos_weights(self) -> dict:
        """pos_weight of the main BCE per domain so that the effective (sampled) positive rate is cfg.pos_rate."""
        q = self.cfg.pos_rate
        pw = {}
        for dn, p in self.real.items():
            y = self.doms[dn].col("ped_corr", False, bool)[p.rows] & ~self.doms[dn].col("uncertain", False, bool)[p.rows]
            pe = (p.w * y).sum() / p.w.sum()
            pw[dn] = float(q / (1 - q) * (1 - pe) / pe) if 0 < pe < 1 else 1.0
        for r, (pl, _) in self.quad.items():
            d = self.doms[f"sim{r}"]
            vis = d.col("px_eq")[pl] >= PX_VIS
            y = d.col("ped_corr", False, bool)[pl] & vis
            pe = (self.slot_w * y).sum() / (self.slot_w * (1 + vis)).sum()     # x- members are negatives
            pw[f"sim{r}"] = float(q / (1 - q) * (1 - pe) / pe) if 0 < pe < 1 else 1.0
        return pw

    def draw(self, rng) -> list[Seg]:
        segs, arm = [], self.arm
        if self.quad:
            k = np.minimum(np.searchsorted(self.slot_cum, rng.random(self.n_slot), side="right"), self.n_slots_pool - 1)
            for r, (pl, mi) in self.quad.items():
                dn = f"sim{r}"
                d = self.doms[dn]
                rows = np.r_[pl[k], mi[k]]
                vis = d.col("px_eq")[pl[k]] >= PX_VIS
                segs.append(Seg(dn, rows, np.r_[vis, np.ones(len(k), bool)], np.r_[np.zeros(len(k), bool), np.ones(len(k), bool)],
                                np.r_[np.full(len(k), dn in arm.score_domains), np.zeros(len(k), bool)], npair=len(k)))
        for dn, p in self.real.items():
            rows = p.draw(rng, {"wod": self.n_wod, "nus": self.n_nus}[dn])
            d = self.doms[dn]
            segs.append(Seg(dn, rows, ~d.col("uncertain", False, bool)[rows], d.col("normal", False, bool)[rows],
                            np.full(len(rows), dn in arm.score_domains)))
        if self.off is not None:
            rows = self.off.draw(rng, self.n_off)
            segs.append(Seg("off", rows, np.zeros(len(rows), bool), np.zeros(len(rows), bool), np.ones(len(rows), bool)))
            tw = self.doms["nav"].pos.reindex(self.doms["off"].col("twin_uid")[rows]).to_numpy()
            tw = tw[~np.isnan(tw)].astype(np.int64)
            segs.append(Seg("nav", tw, np.zeros(len(tw), bool), np.ones(len(tw), bool), np.zeros(len(tw), bool)))
        for i, p in enumerate(self.dsim):
            n = split_counts(self.n_dsim, [1] * len(self.dsim))[i]
            rows = p.draw(rng, n)
            segs.append(Seg(p.dom, rows, np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool)))
        for dn, p in self.dreal.items():
            rows = p.draw(rng, self.n_dreal[dn])
            segs.append(Seg(dn, rows, np.zeros(len(rows), bool), np.ones(len(rows), bool), np.zeros(len(rows), bool)))
        return [s for s in segs if len(s.rows)]

    def describe(self) -> dict:
        return {"sim_slots": self.n_slot, "renders": self.renders, "wod": self.n_wod, "nus": self.n_nus, "off": self.n_off,
                "distill_sim": self.n_dsim, "distill_real": self.n_dreal, "pos_weight": self.pw,
                "pool_sizes": {**{"sim_slots": getattr(self, "n_slots_pool", 0)}, **{k: len(p.rows) for k, p in self.real.items()},
                               "off": len(self.off.rows) if self.off else 0}}


# ---------------------------------------------------------------- batch assembly
KMAX = 14


class Assembler:
    """Segments -> one flat CPU batch (trunks, labels, teacher targets, scorer tensors, pair structure)."""

    def __init__(self, doms: dict, teachers: dict, scores: dict, didx: np.ndarray, pw: dict | None = None,
                 sd_dv: np.ndarray | None = None):
        self.doms, self.tea, self.sc, self.pw, self.sd_dv = doms, teachers, scores, pw or {}, sd_dv
        self.Wv, self.Wdv = t_weights(V_T), t_weights(DV_T)
        self.didx = didx
        self.det_fn = None                    # package D: (domain, ctx rows (B, 9)) -> (tokens (B, 9, 8, F), mask)
        self.op_k = {dn: (sc["names"].index("op") if "op" in sc["names"] else -1) for dn, sc in scores.items() if sc}
        self.lat = {}
        for dn, sc in scores.items():
            if dn.startswith("sim") and dn in doms:
                d = doms[dn]
                if sc is not None and "ped_lat" in sc:
                    self.lat[dn] = pd.Series(sc["ped_lat"], index=sc["uid"]).reindex(d.uid).to_numpy(float)
                else:
                    self.lat[dn] = d.col("ped_lat").astype(float)

    def __call__(self, segs: list[Seg]) -> dict:
        xs, vs, tcs, y, pw, aux, dm, tout, tmu, tsig, tlog = [], [], [], [], [], [], [], [], [], [], []
        sc_rows, cand, top, pairs, dtok, dmask = [], [], [], [], [], []
        base = 0
        for s in segs:
            d, t = self.doms[s.dom], self.tea[s.dom]
            x, v = d.gather(s.rows)
            xs.append(x), vs.append(v), tcs.append(d.tc[s.rows])
            if self.det_fn is not None:
                tk, mk = self.det_fn(s.dom, d.ctx[s.rows])
                dtok.append(tk), dmask.append(mk)
            y.append(np.c_[d.col("ped_corr", False, float)[s.rows], d.col("ped_wide", False, float)[s.rows],
                           np.nan_to_num(d.col("ped_dist", np.nan, float)[s.rows]) / 10].astype(np.float32))
            pw.append(np.full(len(s.rows), self.pw.get(s.dom, 1.0), np.float32))
            aux.append(s.aux), dm.append(s.distill)
            tout.append(np.asarray(t["out"][s.rows]))
            mu = np.asarray(t["mu"][s.rows], np.float32)
            lg = np.asarray(t["logstd"][s.rows], np.float32)
            tmu.append(plan_ch(torch.from_numpy(mu)).numpy()), tsig.append(sigma_o(lg)), tlog.append(lg[:, :, 3])
            sc = self.sc.get(s.dom)
            if s.score.any() and sc is not None:
                ok = sc["pos"].reindex(d.uid[s.rows]).to_numpy()
                use = s.score & ~np.isnan(ok)
                use[use] = sc["valid"][ok[use].astype(int)]
                r = ok[use].astype(int)
                c = np.zeros((len(r), KMAX, len(T4), 4), np.float32)
                tp = np.zeros((len(r), KMAX), bool)
                K = sc["traj"].shape[1]
                c[:, :K], tp[:, :K] = sc["traj"][r], sc["top"][r]
                if self.op_k.get(s.dom, -1) >= 0:                  # op = the teacher's own plan channels (exact zero at step 0)
                    c[:, self.op_k[s.dom]] = tmu[-1][use]
                sc_rows.append(base + np.flatnonzero(use)), cand.append(c), top.append(tp)
            if s.npair:
                n = s.npair
                pl, mi = s.rows[:n], s.rows[n:2 * n]
                px = d.col("px_eq")[pl]
                lat = self.lat.get(s.dom, np.full(len(d), np.nan))[pl]
                if sc is not None:
                    a, b = sc["pos"].reindex(d.uid[pl]).to_numpy(), sc["pos"].reindex(d.uid[mi]).to_numpy()
                    ok = ~np.isnan(a) & ~np.isnan(b)
                    teq = np.zeros(n, bool)
                    teq[ok] = (sc["top"][a[ok].astype(int)] == sc["top"][b[ok].astype(int)]).all(1)
                else:
                    teq = np.zeros(n, bool)
                g_dir, g_eq = dir_gates(px, lat, teq)
                dv = dv_star(d)[pl]
                pairs.append({"plus": base + np.arange(n), "minus": base + n + np.arange(n), "vis": px >= PX_VIS,
                              "g_dir": g_dir, "g_eq": g_eq, "dv": dv, "sig_v": (np.exp(np.minimum(tlog[-1][:n], 11)) @ self.Wv.T).clip(1e-3),
                              "render": s.dom})
            base += len(s.rows)
        cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
        b = {"trunk": np.concatenate(xs), "valid": np.concatenate(vs), "tc": np.concatenate(tcs), "y": np.concatenate(y),
             "pw": np.concatenate(pw), "aux": np.concatenate(aux), "distill": np.concatenate(dm), "tgt": np.concatenate(tout),
             "tmu": np.concatenate(tmu), "tsig": np.concatenate(tsig), "sc_rows": cat(sc_rows).astype(np.int64),
             "cand": np.concatenate(cand) if cand else np.zeros((0, KMAX, len(T4), 4), np.float32),
             "top": np.concatenate(top) if top else np.zeros((0, KMAX), bool), "pairs": pairs,
             "doms": np.concatenate([[s.dom] * len(s.rows) for s in segs])}
        if dtok:
            b["det_tok"], b["det_mask"] = np.concatenate(dtok), np.concatenate(dmask)
        return b


def to_dev(b: dict, dev) -> dict:
    o = {}
    for k, v in b.items():
        if k == "pairs":
            o[k] = [{q: (torch.as_tensor(w).to(dev, non_blocking=True) if isinstance(w, np.ndarray) else w) for q, w in p.items()}
                    for p in v]
        elif isinstance(v, np.ndarray) and v.dtype.kind in "biuf":
            o[k] = torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True)
        elif torch.is_tensor(v):
            o[k] = v.to(dev, non_blocking=True)
        else:
            o[k] = v
    return o


# ---------------------------------------------------------------- one step of losses
class Losses:
    def __init__(self, net, arm: Arm, cfg: RunCfg, tstd: torch.Tensor, dev, sd_dv=None):
        self.arm, self.cfg, self.tstd = arm, cfg, tstd
        self.di = torch.as_tensor(A.distill_index(net.slices), device=dev)
        self.pi = torch.as_tensor(A.plan_index(net.slices), device=dev)
        self.Wv = torch.as_tensor(t_weights(V_T), device=dev)
        self.Wdv = torch.as_tensor(t_weights(DV_T), device=dev)
        self.sd_dv = torch.as_tensor(sd_dv if sd_dv is not None else np.ones(3), device=dev, dtype=torch.float32)

    def plan(self, o):
        return o["outputs"].float()[:, self.pi].view(-1, 33, 15)

    def __call__(self, o: dict, b: dict) -> tuple[torch.Tensor, dict]:
        arm, cfg, L = self.arm, self.cfg, {}
        out = o["outputs"].float()
        plan = out[:, self.pi].view(-1, 33, 15)
        if "aux" in arm.losses or cfg.round1:
            m = b["aux"]
            if m.any():
                pw = None if cfg.round1 else b["pw"][m]
                L["aux"] = loss_aux(o["head_t"][m], b["y"][m], pw) + loss_aux(o["head_v"][m], b["y"][m], pw)
        if "distill" in arm.losses or cfg.round1:
            L["distill"] = loss_distill(out[:, self.di], b["tgt"].float(), self.tstd, b["distill"].float())
        if "score" in arm.losses and len(b["sc_rows"]):
            r = b["sc_rows"]
            L["score"] = loss_score(plan_ch(plan[r]), b["cand"], b["top"], b["tsig"][r])
        pairs = b["pairs"]
        if "pair" in arm.losses and pairs:
            lp = []
            for p in pairs:
                v = p["vis"]
                if v.any():
                    pl, mi = p["plus"][v], p["minus"][v]
                    lp.append(loss_pair(o["head_t"][pl, 0], o["head_t"][mi, 0]) + loss_pair(o["head_v"][pl, 0], o["head_v"][mi, 0]))
            if lp:
                L["pair"] = torch.stack(lp).sum()
        vel = plan[:, :, 3]
        if "dir" in arm.losses and pairs:
            ld = []
            for p in pairs:
                if (p["g_dir"] | p["g_eq"]).any():
                    pl, mi = p["plus"], p["minus"]
                    ld.append(loss_dir(vel[pl] @ self.Wv.T, vel[mi] @ self.Wv.T, p["sig_v"], p["g_dir"],
                                       plan_ch(plan[pl]), plan_ch(plan[mi]), b["tsig"][pl], p["g_eq"]))
            if ld:
                L["dir"] = torch.stack(ld).mean()
        if "dplan" in arm.losses and pairs:
            lb = []
            for p in pairs:
                ok = torch.isfinite(p["dv"]).all(1)
                if ok.any():
                    dvh = (vel[p["plus"][ok]] - vel[p["minus"][ok]]) @ self.Wdv.T
                    lb.append(loss_dplan(dvh, p["dv"][ok], self.sd_dv))
            if lb:
                L["dplan"] = torch.stack(lb).mean()
        lam = {"aux": 1.0, "pair": 1.0, "score": cfg.lam_s, "dir": cfg.lam_s, "distill": cfg.lam_d, "dplan": cfg.lam_b}
        total = sum(lam[k] * v for k, v in L.items())
        return total, L


# ---------------------------------------------------------------- forward on sample rows (dev eval, readouts)
@torch.no_grad()
def forward_rows(model: Model, d: Domain, rows, dev, bs: int = 256, det_fn=None, feats=False) -> dict:
    """Model outputs on rows of a domain: plan (n, 33, 15) MDN mean, plan_logstd, head_t / head_v (n, 3), lead (n, 72),
    lead_prob (n, 3); feats=True adds temporal / vision (n, 512)."""
    sl = model.net.slices
    pi = A.plan_index(sl)
    lead = np.arange(sl["lead"].start, sl["lead"].start + 72)
    lp = np.arange(sl["lead_prob"].start, sl["lead_prob"].stop)
    ps = np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
    acc = {}
    rows = np.asarray(rows, np.int64)
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        x, v = d.gather(r)
        tok = det_fn(d.name, d.ctx[r]) if det_fn else (None, None)
        o = model(torch.from_numpy(x).to(dev), torch.from_numpy(v).to(dev), torch.from_numpy(d.tc[r]).to(dev),
                  *[torch.as_tensor(t).to(dev) if t is not None else None for t in tok])
        out = o["outputs"].float()
        res = {"plan": out[:, pi].view(-1, 33, 15), "plan_logstd": out[:, ps].view(-1, 33, 15), "lead": out[:, lead],
               "lead_prob": out[:, lp], "head_t": o["head_t"].float(), "head_v": o["head_v"].float()}
        if feats:
            res |= {"temporal": o["select_4"].float(), "vision": o["mean"].float()}
        for k, t in res.items():
            acc.setdefault(k, []).append(t.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def v_at(plan: np.ndarray, t: float) -> np.ndarray:
    """Plan longitudinal speed (vel x) at time t, (n,)."""
    return plan[:, :, 3] @ t_weights([t])[0]


def slow_flag(plan, v0):
    """B-real slow: plan 2 s speed < current speed - max(1 m/s, 20 %)."""
    v0 = np.asarray(v0, float)
    return v_at(plan, 2.0) < v0 - np.maximum(1.0, 0.2 * v0)


def ego_speed(d: Domain, rows, tea) -> np.ndarray:
    """Current ego speed: the sample table's `ego_speed` / `v0` column, else the original plan's speed at t = 0."""
    for c in ("ego_speed", "v0"):
        if c in d.s:
            return d.col(c)[rows].astype(float)
    return np.asarray(tea["mu"][rows], np.float32)[:, 0, 3].astype(float)


def score_api():
    """Package S's score_plans(domain, uid, plans) or None while it is not delivered."""
    try:
        from . import op_adapt_score as S
        return S.score_plans
    except (ImportError, AttributeError):
        return None


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def probe_auc_oof(X, y, groups, k=5, seed=0) -> float:
    """Out-of-fold AUC of a class-balanced L2 logistic probe on z-scored features, folds by group (dev reference for
    the aux heads: the original model's `temporal` read linearly)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, bool)
    ug = np.array(sorted(set(groups)))
    if len(ug) < k or not 0 < y.sum() < len(y):
        return float("nan")
    fold = pd.Series(np.arange(len(ug)) % k, index=np.random.default_rng(seed).permutation(ug))[groups].to_numpy()
    s = np.full(len(y), np.nan)
    for f in range(k):
        tr, ev = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        s[ev] = m.decision_function((X[ev] - mu) / sd)
    ok = ~np.isnan(s)
    return auc(y[ok], s[ok])


def _cap(rows, n, seed=0):
    rows = np.asarray(rows)
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False)) if len(rows) > n else rows


def dev_eval(model: Model, doms: dict, teachers: dict, scores: dict, arm: Arm, dev, det_fn=None, cap: int = 6000,
             o_probe: dict | None = None) -> dict:
    """Every dev number the §2 lambda_s rule and the §6 checklists read (the adapted model against O = the teacher)."""
    r, sfn = {}, score_api()
    fw = lambda dn, rows: forward_rows(model, doms[dn], rows, dev, det_fn=det_fn)  # noqa: E731
    # drift, null slow rate and speed distribution on real dev normal frames
    drift, slow_a, slow_o, v2a, v2o = [], [], [], [], []
    for dn in ("nus", "wod", "nav"):
        if dn not in doms:
            continue
        d, t = doms[dn], teachers[dn]
        m = (d.col("split") == "dev") & d.col("normal", False, bool)
        if dn != "nav":
            m &= d.col("labelled", False, bool)
        rows = _cap(np.flatnonzero(m), cap)
        if not len(rows):
            continue
        o = fw(dn, rows)
        tm = np.asarray(t["mu"][rows], np.float32)
        dr = A.plan_drift(o["plan"], tm)
        r[f"drift_{dn}_median"], r[f"drift_{dn}_p95"] = float(np.median(dr)), float(np.percentile(dr, 95))
        if dn != "nav":
            drift.append(dr)
            v0 = ego_speed(d, rows, t)
            slow_a.append(slow_flag(o["plan"], v0)), slow_o.append(slow_flag(tm, v0))
            v2a.append(v_at(o["plan"], 2.0)), v2o.append(v_at(tm, 2.0))
    if drift:
        from scipy.stats import ks_2samp
        dr = np.concatenate(drift)
        r["drift_median"], r["drift_p95"] = float(np.median(dr)), float(np.percentile(dr, 95))
        r["null_slow_adapt"], r["null_slow_orig"] = float(np.concatenate(slow_a).mean()), float(np.concatenate(slow_o).mean())
        r["null_slow_delta_pp"] = 100 * (r["null_slow_adapt"] - r["null_slow_orig"])
        r["ks_v2_normal"] = float(ks_2samp(np.concatenate(v2a), np.concatenate(v2o)).statistic)
    # aux-head AUC per domain against the original temporal probe
    for dn in ("nus", "wod", "simC", "simK"):
        if dn not in doms:
            continue
        d = doms[dn]
        m = (d.col("split") == "dev") & d.col("labelled", True, bool) & ~d.col("uncertain", False, bool)
        if dn.startswith("sim"):
            m &= (d.col("sign") == -1) | (d.col("px_eq") >= PX_VIS)
        rows = _cap(np.flatnonzero(m), cap)
        if not len(rows):
            continue
        o = fw(dn, rows)
        y = d.col("ped_corr", False, bool)[rows]
        r[f"aux_auc_{dn}"] = auc(y, o["head_t"][:, 0])
        if o_probe is not None and dn in o_probe:
            r[f"aux_auc_{dn}_orig_probe"] = o_probe[dn]
        elif "temporal" in teachers[dn]:
            r[f"aux_auc_{dn}_orig_probe"] = probe_auc_oof(np.asarray(teachers[dn]["temporal"][rows], np.float32), y,
                                                          d.col("group")[rows])
        if dn.startswith("sim"):
            _dev_sim(r, dn, d, rows, o, teachers[dn], scores.get(dn), arm, sfn)
    # offset slots
    if "off" in doms:
        d = doms["off"]
        sc = scores.get("off")
        if sc is not None:
            op = sc["names"].index("op")
            tr = sc["pos"].reindex(d.uid[d.col("split") == "train"]).dropna().to_numpy(int)
            fail = (sc["DAC"][tr, op] == 0) | (sc["DDC"][tr, op] < 1)
            r["off_train_op_fail"] = float(fail.mean()) if len(tr) else float("nan")
        rows = np.flatnonzero(d.col("split") == "dev")
        if len(rows):
            o = fw("off", rows)
            if sfn is not None and sc is not None:
                s_a = sfn("off", d.uid[rows], o["plan"])
                k = sc["pos"].reindex(d.uid[rows]).to_numpy()
                ok = ~np.isnan(k)
                r["off_dev_pass_adapt"] = float((s_a["DAC"][ok] * (np.asarray(s_a["DDC"])[ok] == 1)).mean())
                r["off_dev_pass_orig"] = float((sc["DAC"][k[ok].astype(int), op] * (sc["DDC"][k[ok].astype(int), op] == 1)).mean())
            tw = doms["nav"].pos.reindex(d.col("twin_uid")[rows]).dropna().to_numpy(int) if "nav" in doms else []
            if len(tw):
                ot = fw("nav", tw)
                dt = A.plan_drift(ot["plan"], np.asarray(teachers["nav"]["mu"][tw], np.float32))
                r["twin_drift_median"] = float(np.median(dt))
    if model.det is not None and hasattr(model.det, "gate"):
        r["det_gate_abs"] = float(model.det.gate.detach().abs().max())
    return r


def _dev_sim(r, dn, d, rows, o, t, sc, arm, sfn):
    """Sim dev: pair accuracy, L_dir violation, S_jev gain where op is not in Top, x- slow-candidate collapse, A-bhv."""
    uid = d.uid[rows]
    pos = pd.Series(np.arange(len(rows)), index=uid)
    plus = np.flatnonzero((d.col("sign")[rows] == 1) & (d.col("px_eq")[rows] >= PX_VIS))
    mi = pos.reindex(d.col("twin_uid")[rows][plus]).to_numpy()
    ok = ~np.isnan(mi)
    pl, mi = plus[ok], mi[ok].astype(int)
    if len(pl):
        r[f"pair_acc_{dn}"] = float((o["head_t"][pl, 0] > o["head_t"][mi, 0]).mean())
    tm = np.asarray(t["mu"][rows], np.float32)
    # L_dir violation on g_dir pairs (all plus rows incl. invisible, to evaluate the gate)
    allp = np.flatnonzero(d.col("sign")[rows] == 1)
    am = pos.reindex(d.col("twin_uid")[rows][allp]).to_numpy()
    k = ~np.isnan(am)
    ap, am = allp[k], am[k].astype(int)
    lat = d.col("ped_lat")[rows][ap] if "ped_lat" in d.s else (
        pd.Series(sc["ped_lat"], index=sc["uid"]).reindex(uid[ap]).to_numpy() if sc is not None and "ped_lat" in sc else np.full(len(ap), np.nan))
    g_dir, _ = dir_gates(d.col("px_eq")[rows][ap], lat, np.zeros(len(ap), bool))
    W = t_weights(V_T)
    if g_dir.any():
        va = ((o["plan"][ap, :, 3] - o["plan"][am, :, 3]) @ W.T).mean(1)
        vo = ((tm[ap, :, 3] - tm[am, :, 3]) @ W.T).mean(1)
        r[f"dir_violation_{dn}_adapt"], r[f"dir_violation_{dn}_orig"] = float((va[g_dir] > 0).mean()), float((vo[g_dir] > 0).mean())
        r[f"dir_n_{dn}"] = int(g_dir.sum())
    if "dplan" in arm.losses:
        dv = dv_star(d)[rows][ap]
        dh = (o["plan"][ap, :, 3] - o["plan"][am, :, 3]) @ t_weights(DV_T).T
        f = np.isfinite(dv) & (np.abs(dv) > 0)
        r[f"dplan_sign_agree_{dn}"] = float((np.sign(dh[f]) == np.sign(dv[f])).mean()) if f.any() else float("nan")
    if sc is None:
        return
    op = sc["names"].index("op")
    k = sc["pos"].reindex(uid).to_numpy()
    has = ~np.isnan(k)
    ki = np.where(has, k, 0).astype(int)
    # S_jev of the native plan where the original plan is not in Top (dev x+)
    xp = np.flatnonzero(has & (d.col("sign")[rows] == 1) & sc["valid"][ki])
    if sfn is not None and len(xp):
        s_a = np.asarray(sfn(dn, uid[xp], o["plan"][xp])["S"], float)
        s_o = sc["S"][ki[xp], op]
        r[f"sjev_delta_{dn}"] = float(np.mean(s_a - s_o))
        notop = ~sc["top"][ki[xp], op]
        r[f"sjev_better_notop_{dn}"] = float((s_a[notop] > s_o[notop]).mean()) if notop.any() else float("nan")
        r[f"sjev_n_notop_{dn}"] = int(notop.sum())
    # x-: nearest Top candidate is a slow / stop candidate
    xm = np.flatnonzero(has & (d.col("sign")[rows] == -1))
    if len(xm):
        slow = np.isin(np.array(sc["names"]), SLOW_CANDS)
        sig = sigma_o(np.asarray(t["logstd"][rows[xm]], np.float32))
        cand = sc["traj"][ki[xm]]
        top = sc["top"][ki[xm]]
        res = {}
        for who, pln in (("adapt", o["plan"][xm]), ("orig", tm[xm])):
            dd = huber_d(torch.from_numpy(plan_ch(torch.from_numpy(pln)).numpy()), torch.from_numpy(cand), torch.from_numpy(sig)).numpy()
            dd[~top] = np.inf
            res[who] = slow[dd.argmin(1)].mean()
        r[f"xminus_slow_nearest_{dn}_adapt"], r[f"xminus_slow_nearest_{dn}_orig"] = float(res["adapt"]), float(res["orig"])


# ---------------------------------------------------------------- lambda_s dev selection (§2)
def select_lambda(devs: dict) -> dict:
    """devs {lam_s: dev dict of A seed 0}. Among configs with dev drift median <= 0.10 m and dev null slow rate
    <= original + 2 pp, the one with the highest mean dev x+ S_jev gain (simC and simK pooled by their means);
    none qualifies -> the smaller drift."""
    rows = []
    for lam, d in devs.items():
        g = [d[k] for k in ("sjev_delta_simC", "sjev_delta_simK") if k in d and np.isfinite(d[k])]
        rows.append({"lam_s": lam, "drift": d.get("drift_median", np.inf), "slow_pp": d.get("null_slow_delta_pp", np.inf),
                     "gain": float(np.mean(g)) if g else float("nan")})
    ok = [x for x in rows if x["drift"] <= 0.10 and x["slow_pp"] <= 2.0]
    if ok and all(np.isfinite(x["gain"]) for x in ok):
        pick, rule = max(ok, key=lambda x: x["gain"]), "qualifying config with the highest dev x+ S_jev gain"
    elif ok and len(ok) == 1:
        pick, rule = ok[0], "only qualifying config (gain not available)"
    else:
        pick, rule = min(rows, key=lambda x: x["drift"]), "no config qualifies (or gain missing): smaller drift"
    return {"lam_s": pick["lam_s"], "rule": rule, "candidates": rows}
