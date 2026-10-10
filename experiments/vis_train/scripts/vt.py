"""vis_train (VT) trainer and read-out paths (plans/2026-10-10-vis-train-prereg.md): does the navtest turn off-road gap close when Cinque's
vision weights are trained on navtrain. Every arm continues SH30-F-s<seed> (P2 + footprint hinge 30 / 0.5, protocol W) on the same row stream
with a constant learning rate after the warmup:

  F    vision frozen, no branch: the same-steps baseline (a plain pp_train P2 checkpoint)
  A0   frozen; the adapter memory channel (ParityAdapter use_side, n_cam = n_t = 1) reads Cinque's own cached t0 token (= branch lr 0)
  A    Cinque itself frozen (its 8 slot tokens come from the cache); a trainable copy of its vision encoder reads the t0 image pair and its
       32 x 512 tokens enter through the memory channel
  B    A + future-feature loss: a 2-layer transformer predicts the frozen t0-slot tokens 1.0 s and 2.0 s ahead from the branch tokens
  C    no branch: the encoder itself trained in place (all 8 slots through the current encoder, gradient through the t0 pair), no anchor
       rows, no distillation of the non-plan heads
  F0   frozen, no anchor rows, no distillation: C's control
  W    A with three t0 views (second wave, prereg amendments 1 / 3 / 5): the shared-weight branch encodes [CAM_F0, CAM_L0, CAM_R0] (F0 from
       the pixel cache, byte-identical to A's input; the side pairs from lib/side_store.py), 3 x 32 tokens through
       ParityAdapter(n_cam = 3, n_t = 1); the branch head reads all three views; the training memory drop masks the three together
A / B / A0 / W carry the branch's own head (the thin decoder of decisions 147 / 160 on [tokens, ego]; decision 204), trained jointly, unused at
inference. Pixels come from the pre-rendered cache (lib/pixel_store.py); nothing is rendered online.

  train   --arm A --seed 0 --steps 30000 [--tag VT-A-s0] [--resume]   snapshots every --snap-every steps -> runs/op_parity/runs/<tag>-k<NN>/,
          the final weights -> <tag>/ckpt-final.pt. Refuses a tag that exists; --resume continues its own interrupted run from <tag>/resume.pt
          (weights + optimizer of the last snapshot step; the row streams are replayed to that step).
  plans   --tag T --data lb_navtest --out F [--mem on|off|shuf|sideoff]   op_lb plan file of a VT checkpoint (what jevdrive.bench exports);
          sideoff (W only) masks the two side views and keeps the F0 view
  ident   --arm A --seed 0                                              identity gate: the arm at SH30's weights (memory masked; C after
          lr-0 steps) against SH30's cached-token plans on the first 2 048 navtest tokens, with the other SH30 seed as the wrong-model control
  tokens  --tag T --datas D ...                                         branch / encoder t0 tokens (N, 32, 512) fp16 -> runs/vis_train/tokens/<tag>/
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, os, re, shutil, time  # noqa: E401,E402
from dataclasses import asdict, dataclass  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import parity_adapter as PA  # noqa: E402
import pixel_store as PX  # noqa: E402
import side_store as SD  # noqa: E402
import pp_train as T  # noqa: E402
import pp_unfreeze as U  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.op_torch import _key  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402

ARMS = {"F": dict(mem=False, vis="", fut=False, anchor=True), "A0": dict(mem=True, vis="", fut=False, anchor=True),
        "A": dict(mem=True, vis="branch", fut=False, anchor=True), "B": dict(mem=True, vis="branch", fut=True, anchor=True),
        "C": dict(mem=False, vis="inplace", fut=False, anchor=False), "F0": dict(mem=False, vis="", fut=False, anchor=False),
        "W": dict(mem=True, vis="branch", fut=False, anchor=True, views=3)}
VIEWS = ("CAM_F0",) + SD.CAMS                  # W: the camera axis of the memory; test-time "side off" keeps only the first
FULL = tuple(f"navtrain_full.s{i}of12" for i in range(12))
RUNS = data_dir() / "runs" / "op_parity" / "runs"
OUT = data_dir() / "runs" / "vis_train"
NPT, TOL_M = 15, 0.03                          # identity gate: the plan points inside 4 s, two fp16 ulps at 32 m (docs/long-runs.md)
ULP2_M = 0.0320                                # two fp16 ulps on the points at 32 .. 64 m are 0.03125 m, just over TOL_M (prereg amendment 1)


@dataclass
class Cfg:
    arm: str
    seed: int = 0
    steps: int = 30000
    batch: int = 64
    d_frac: float = 0.25                   # anchor rows (0 for C / F0)
    lam_i: float = 1.0
    lam_c: float = 3.0
    lam_d: float = 30.0                    # distillation of the non-plan heads (0 for C / F0)
    hinge_lam: float = 30.0
    hinge_margin: float = 0.5
    hinge_labels: str = "runs/op_probe/labels/navtrain_all.npz"
    lr: float = 1e-5                       # policy
    lr_new: float = 1e-4                   # adapter, branch head, predictor
    lr_vis: float = 1e-5
    wd: float = 0.01
    wd_vis: float = 0.0                    # no decay on the vision weights: their displacement from init is then gradient only
    warmup: int = 300                      # then constant
    eval_every: int = 1000
    snap_every: int = 5000
    mem_drop: float = T.MEM_DROP
    lam_head: float = 1.0                  # the branch's own head: imitation + head_hinge_lam x footprint hinge at head_hinge_margin
    head_hinge_lam: float = 10.0
    head_hinge_margin: float = 0.3
    lam_fut: float = 0.5                   # B: smooth-L1 to the standardised frozen tokens fut_k half-seconds ahead
    fut_k: tuple = (2, 4)
    data: tuple = FULL
    split: str = "navsim/op-parity-full"
    init: str = ""                         # SH30-F-s<seed>


# ---------------------------------------------------------------- model
class Head(nn.Module):
    """The thin decoder of decisions 147 / 160 (opb_probe / rep.py cmd_decode): z-scored [32 x 512 tokens, ego] -> 8 poses (x, y, yaw)."""

    def __init__(self, d_in=32 * 512 + PA.EGO_DIM, h=1024):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 24))

    def forward(self, tok, ego):
        return self.net((torch.cat([tok.float().flatten(1), ego.float()], 1) - self.mu) / self.sd).view(-1, 8, 3)


class Predictor(nn.Module):
    """B: [branch t0 tokens (standardised), ego] + a horizon embedding -> the standardised frozen t0-slot tokens of each horizon. No future pose."""

    def __init__(self, d=512, layers=2, heads=8, n_h=2, D=512):
        super().__init__()
        self.register_buffer("mu", torch.zeros(D))
        self.register_buffer("sd", torch.ones(D))
        self.inp = nn.Linear(D, d)
        self.ego = nn.Sequential(nn.Linear(PA.EGO_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.s_emb = nn.Parameter(0.02 * torch.randn(33, d))
        self.h_emb = nn.Parameter(0.02 * torch.randn(n_h, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True), layers,
                                         enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)

    def z(self, tok):
        return (tok.float() - self.mu) / self.sd

    def forward(self, tok, ego):
        B, K = tok.shape[0], self.h_emb.shape[0]
        x = torch.cat([self.inp(self.z(tok)), self.ego(ego.float())[:, None]], 1) + self.s_emb
        x = (x[:, None] + self.h_emb[None, :, None]).reshape(B * K, 33, -1)
        return self.out(self.norm(self.enc(x))[:, :32]).view(B, K, 32, -1)


class VT(nn.Module):
    """Cinque port with the trainable plan pathway (pp_train.PModel's), the parity adapter (with the memory channel on the memory arms) and,
    per arm, trainable vision weights: the branch (A / B: the port's own vision part is the trainable copy; the frozen original is the token
    cache) or the encoder itself (C)."""

    def __init__(self, arm: str, dtype=torch.float16):
        super().__init__()
        self.arm, self.k = arm, ARMS[arm]
        self.vis = U.vision_names("all") if self.k["vis"] else []
        self.net = A.load("cinque", dtype, trainable=L.pol_weights() + self.vis)
        self.vkeys = {_key(w) for w in self.vis}
        mem = self.k["mem"]
        self.nv = self.k.get("views", 1)                                    # views of the branch (W: 3)
        self.adapter = PA.ParityAdapter(use_ego=True, use_side=mem, **(dict(n_cam=self.nv, n_t=1) if mem else {}))
        self.head = Head(self.nv * 32 * 512 + PA.EGO_DIM) if mem else None
        self.pred = Predictor(n_h=2) if self.k["fut"] else None
        self.enc_ckpt, self.enc_compiled = 0, False                         # speed knobs: activation checkpointing chunk of the grad pass; compiled encoder
        self.enc_chunk = 0                                                  # pairs per call of the fast grad pass (0 = the whole batch at once)

    def encode(self, prev, cur, grad=False, chunk=256):
        """(n, 2, 6, 128, 256) uint8 pairs -> (n, 32, 512) tokens of the vision part."""
        fe = getattr(PX, "fast_encode", None)
        if fe is not None and not self.enc_ckpt:
            return fe(self.net, prev, cur, grad=grad, chunk=self.enc_chunk if grad else 0, compiled=self.enc_compiled and grad)
        from torch.utils.checkpoint import checkpoint
        f = lambda p, c: self.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE)  # noqa: E731
        with torch.set_grad_enabled(grad):
            if grad and self.enc_ckpt:
                k = self.enc_ckpt
                return torch.cat([checkpoint(f, prev[i:i + k], cur[i:i + k], use_reentrant=False) for i in range(0, len(cur), k)])
            return torch.cat([f(prev[i:i + chunk], cur[i:i + chunk]) for i in range(0, len(cur), chunk)])

    def branch(self, px, sd=None, grad=False):
        """The branch's memory tokens: (B, 32, 512) of the t0 pair px = (prev, cur); W (sd = the side stores' (prev, cur), each
        (B, 2, 2, 6, 128, 256)): (B, 3, 32, 512) of VIEWS through the same weights. The pass is view-major: its first B pairs are arm A's batch."""
        if sd is None:
            return self.encode(*px, grad=grad)
        p, c = (torch.cat([f[:, None], s], 1).transpose(0, 1).flatten(0, 1) for f, s in zip(px, sd))
        return self.encode(p, c, grad=grad).reshape(self.nv, -1, *A.H_SHAPE).transpose(0, 1)

    def slots(self, prev, cur, grad=False):
        """C: (B, 8, 2, 6, 128, 256) pairs -> (B, 8, 32, 512); the 7 older slots through the same weights without autograd (as U2)."""
        B = cur.shape[0]
        hp = self.encode(prev[:, :7].flatten(0, 1), cur[:, :7].flatten(0, 1)).reshape(B, 7, *A.H_SHAPE)
        return torch.cat([hp, self.encode(prev[:, 7], cur[:, 7], grad=grad)[:, None]], 1)

    def policy(self, front, ego, tc, mem=None, mask=None, inputs_on=True):
        """pp_train.PModel.forward: front (B, 8, 32, 512), mem (B, 32, 512) memory tokens (W: (B, 3, 32, 512)), mask (B, 1) True = memory
        present (W: (B, 1) all views or (B, 3) per view) -> outputs."""
        B, n = front.shape[:2]
        H = torch.cat([front.new_zeros(B, A.CONTEXT - n, *front.shape[2:]), front], 1).to(self.net.dtype)
        valid = torch.zeros(B, A.CONTEXT, dtype=torch.bool, device=H.device)
        valid[:, A.CONTEXT - n:] = True
        if inputs_on:
            H = self.adapter.apply(H, ego, side5(mem) if mem is not None else None, mask)
        H = H * valid[:, :, None, None].to(H.dtype)
        return self.net.run_batched(A.policy_feeds(self.net, H, T.AT, tc.to(self.net.dtype)), ["outputs"])["outputs"].reshape(B, -1)

    def groups(self):
        pol = [p for k, p in self.net.params.items() if p.requires_grad and k not in self.vkeys]
        vis = [p for k, p in self.net.params.items() if p.requires_grad and k in self.vkeys]
        new = [p for m in (self.head, self.pred) if m is not None for p in m.parameters()]
        return pol, vis, list(self.adapter.parameters()), new

    @torch.no_grad()
    def state(self) -> dict:
        """F / F0: pp_train.PModel's format (arm P2: the cached-token parity path loads it). The other arms carry `vt` and no `arm`, so
        that path refuses them (they need the memory tokens / their own encoder: `plans` here)."""
        st = {"net": {k: p.detach().float().cpu() for k, p in self.net.params.items() if p.requires_grad}, "adapter": None,
              "parity": self.adapter.state_dict()}
        if self.arm in ("F", "F0"):
            return st | {"arm": "P2"}
        return st | {"vt": {"arm": self.arm, "head": self.head.state_dict() if self.head else None, "pred": self.pred.state_dict() if self.pred else None}}

    def load_state(self, st, init=False):
        """init: a plain P2 checkpoint (SH30): its adapter fills the shared weights, the memory-channel inputs keep their fresh init."""
        for k, v in st["net"].items():
            self.net.params[k].data.copy_(v)
        r = self.adapter.load_state_dict(st["parity"], strict=not init)
        assert not r.unexpected_keys and all(k.startswith(("side_in.", "cam_emb", "t_emb", "s_emb")) for k in r.missing_keys), r
        for m, k in ((self.head, "head"), (self.pred, "pred")):
            if m is not None and not init:
                m.load_state_dict(st["vt"][k])


def side5(mem):
    """Memory tokens (B, 32, 512) or (B, n_cam, 32, 512) -> the adapter's (B, n_cam, n_t = 1, 32, 512)."""
    return mem[:, None, None] if mem.dim() == 3 else mem[:, :, None]


def mem_mask(n: int, dev, mem: str):
    """Test-time memory mask of n rows: off = everything masked, sideoff = the F0 view kept and the side views masked, else None."""
    if mem == "off":
        return torch.zeros(n, 1, dtype=torch.bool, device=dev)
    return torch.tensor([v == VIEWS[0] for v in VIEWS], device=dev).expand(n, -1) if mem == "sideoff" else None


def ckpt_arm(st: dict) -> str:
    return st["vt"]["arm"] if "vt" in st else "F"


def load_tag(tag: str, dev):
    ck = torch.load(RUNS / tag / "ckpt-final.pt", map_location="cpu", weights_only=False)
    m = VT(ckpt_arm(ck["model"])).to(dev)
    m.load_state(ck["model"])
    return m.eval(), ck


def vis_stage(names) -> dict:
    """Vision initializer -> stem | s0 .. s3 | head | norm (the input mean / std), by the graph order (the unnamed MatMul weights and layer
    scales of a block follow its named depthwise conv)."""
    import onnx
    g = onnx.load(str(A.MODELS_DIR / A.FILES["cinque"]), load_external_data=False).graph
    want, out, cur = set(names), {}, "stem"
    for n in g.node:
        for i in n.input:
            if i in want and i not in out:
                m = re.search(r"encoder\.(stem|head|stages\.\d)", i)
                cur = m.group(1).replace("stages.", "s") if m else cur
                out[i] = "norm" if i.endswith(("._mean", "._std")) else cur
    assert set(out) == want
    return out


class Disp:
    """Relative L2 displacement of the trained weights from the starting point, per group."""

    def __init__(self, model: VT):
        st = vis_stage(model.vis)
        self.grp = {_key(w): f"vis_{s}" for w, s in st.items()}
        self.w0 = {k: p.detach().clone() for k, p in model.net.params.items() if p.requires_grad}
        self.a0 = {k: v.detach().clone() for k, v in model.adapter.state_dict().items()}

    @torch.no_grad()
    def __call__(self, model: VT) -> dict:
        num, den = {}, {}
        for k, w0 in self.w0.items():
            for g in ((self.grp[k], "vis") if k in self.grp else ("policy",)):
                num[g] = num.get(g, 0.0) + float((model.net.params[k].float() - w0.float()).pow(2).sum())
                den[g] = den.get(g, 0.0) + float(w0.float().pow(2).sum())
        sd = model.adapter.state_dict()
        shared = [k for k in self.a0 if not k.startswith(("side_in.", "cam_emb", "t_emb", "s_emb"))]
        num["adapter"] = sum(float((sd[k] - self.a0[k]).pow(2).sum()) for k in shared)
        den["adapter"] = sum(float(self.a0[k].pow(2).sum()) for k in shared)
        return {g: (num[g] / max(den[g], 1e-30)) ** 0.5 for g in num}


# ---------------------------------------------------------------- data
def t0_tok(tok: T.Tokens, rows) -> torch.Tensor:
    """The t0-slot tokens (n, 32, 512) fp16 of store rows (host stores: only that slot is read)."""
    if not tok.host:
        return tok.t[torch.as_tensor(rows, device=tok.dev)][:, -1]
    r = np.asarray(rows.cpu() if torch.is_tensor(rows) else rows)
    out = np.empty((len(r),) + tuple(tok.mms[0].shape[2:]), np.float16)
    k = np.searchsorted(tok.off, r, side="right") - 1
    for s in np.unique(k):
        m = np.flatnonzero(k == s)
        o = np.argsort(r[m])
        out[m[o]] = tok.mms[s][r[m][o] - tok.off[s], -1]
    return torch.from_numpy(out).to(tok.dev, non_blocking=True)


def derange(log: np.ndarray, rng) -> np.ndarray:
    """A permutation p with log[p[i]] != log[i] for every row: memory of another log."""
    p = rng.permutation(len(log))
    for _ in range(1000):
        bad = np.flatnonzero(log[p] == log)
        if not len(bad):
            return p
        for i in bad:
            j = int(rng.integers(len(log)))
            p[i], p[j] = p[j], p[i]
    raise RuntimeError("no derangement found")


class Lane:
    """Everything one arm needs for a step: stores, model at its starting point, losses, optimizer, row streams."""

    def __init__(self, cfg: Cfg, dev, a):
        self.cfg, self.dev, self.k = cfg, dev, ARMS[cfg.arm]
        k = self.k
        if a.vram_cap:
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_cap * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory))
        torch.manual_seed(cfg.seed)
        self.S = S = T.Store(list(cfg.data), dev, need_side=False, frames="warp", host=True)
        self.tr_rows, self.dv_rows, self.sp = T.split_rows({"names": S.tab["names"]}, cfg.split)
        self.PS = PX.PixelStore(list(cfg.data), dev, mode="all" if k["vis"] == "inplace" else "t0") if (k["vis"] or k["fut"]) else None
        assert self.PS is None or len(self.PS) == S.n
        self.SS = SD.SideStore(list(cfg.data), dev) if k.get("views", 1) > 1 else None
        assert self.SS is None or (self.SS.names == S.tab["names"]).all()
        self.model = m = VT(cfg.arm).to(dev)
        m.enc_ckpt, m.enc_compiled, m.enc_chunk = a.enc_ckpt, a.enc_compile, a.enc_chunk
        m.load_state(torch.load(RUNS / cfg.init / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"], init=True)
        from drivable_hinge import Hinge
        self.hinge = Hinge([data_dir() / cfg.hinge_labels], S.tab["names"], dev, cfg.hinge_margin, ["pacifica"])
        tstd = S.t_out[torch.as_tensor(self.tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
        self.LS = T.Losses(m.net, T.Cfg(arm="P2", lam_i=cfg.lam_i, lam_c=cfg.lam_c, lam_d=cfg.lam_d, hinge_lam=cfg.hinge_lam, hinge_margin=cfg.hinge_margin),
                           tstd, S.di, S.pi, dev, self.hinge)
        srng = np.random.default_rng([cfg.seed, 0, 13])                       # normalisation statistics: a fixed sample of train rows
        if m.head is not None or m.pred is not None:
            rs = np.sort(srng.choice(self.tr_rows, 8192, replace=False))
            x = t0_tok(S.front, rs).float()
            if m.head is not None:
                e, xh = S.ego[torch.as_tensor(self.tr_rows, device=dev)], x
                if self.SS is not None:        # W: no cache holds frozen tokens of the protocol-W side pairs; the branch at its init is the frozen encoder
                    xs = torch.cat([m.encode(*(v.flatten(0, 1) for v in self.SS.t0(rs[i:i + 128]))).reshape(-1, len(SD.CAMS), *A.H_SHAPE)
                                    for i in range(0, len(rs), 128)])
                    xh = torch.cat([x[:, None], xs.float()], 1)
                m.head.mu.copy_(torch.cat([xh.flatten(1).mean(0), e.mean(0)]))
                m.head.sd.copy_(torch.cat([xh.flatten(1).std(0), e.std(0)]).clamp_min(1e-6))
                del xh
            if m.pred is not None:
                m.pred.mu.copy_(x.mean((0, 1)))
                m.pred.sd.copy_(x.std((0, 1)).clamp_min(1e-6))
        self.disp = Disp(m)
        pol, vis, ad, new = m.groups()
        self.clip = [pol + ad] + ([vis] if vis else []) + ([new] if new else [])     # each clipped on its own: the policy's clip is that of F
        g = [{"params": pol, "lr": cfg.lr, "base": cfg.lr, "weight_decay": cfg.wd}, {"params": ad, "lr": cfg.lr_new, "base": cfg.lr_new, "weight_decay": cfg.wd}]
        g += [{"params": vis, "lr": cfg.lr_vis, "base": cfg.lr_vis, "weight_decay": cfg.wd_vis}] if vis else []
        g += [{"params": new, "lr": cfg.lr_new, "base": cfg.lr_new, "weight_decay": cfg.wd}] if new else []
        self.opt = torch.optim.AdamW(g, fused=True)
        self.scaler = torch.amp.GradScaler()
        self.rng = np.random.default_rng([cfg.seed, 0])                       # pp_train's row stream: the same rows for every arm of a seed
        self.mrng = np.random.default_rng([cfg.seed, 0, 11])                  # memory drop: its own stream
        self.n_par = {n: sum(p.numel() for p in ps) for n, ps in zip(("policy", "vision", "adapter", "head+pred"), (pol, vis, ad, new))}
        S.front.pin = a.prefetch > 1
        self.policy = torch.compile(m.policy, dynamic=False) if a.compile else m.policy      # the training step's policy forward

    def draw(self):
        c, nB = self.cfg, self.cfg.batch
        r = self.rng.choice(self.tr_rows, nB, replace=len(self.tr_rows) < nB)
        an = self.rng.random(nB) < c.d_frac
        sm = self.mrng.random((nB, 1)) >= c.mem_drop if self.k["mem"] else None
        return r, an, sm

    def fut_targets(self, r):
        """[(tokens (n, 32, 512) fp16 of the row fut_k half-seconds ahead, ok (n,))] per horizon."""
        out = []
        for kk in self.cfg.fut_k:
            j, ok = self.PS.future(np.asarray(r), kk)
            out.append((t0_tok(self.S.front, j), torch.from_numpy(ok).to(self.dev)))
        return torch.stack([t for t, _ in out], 1), torch.stack([o for _, o in out], 1)

    def fetch(self, dr):
        r, b, k = dr[0], {}, self.k
        if k["vis"] != "inplace":
            b["front"] = self.S.front[r]
        if k["vis"]:
            b["px"] = self.PS.all(r) if k["vis"] == "inplace" else self.PS.t0(r)
        if self.SS is not None:
            b["sd"] = self.SS.t0(r)
        if k["fut"]:
            b["fut"] = self.fut_targets(r)
        return dr, b

    def losses(self, b, rows, anchor, smask, grad=True):
        """-> total, parts, outputs. The step's forward (training: grad through the trainable vision weights)."""
        c, m, S, k = self.cfg, self.model, self.S, self.k
        ego_in = S.ego[rows]
        ego = ego_in * (~anchor)[:, None].float()                             # present = 0 on anchor rows -> the bias is exactly 0
        mem = None
        if k["vis"] == "inplace":
            front = m.slots(*b["px"], grad=grad)
        else:
            front = b["front"]
            if k["mem"]:
                mem = (m.branch(b["px"], b.get("sd"), grad=grad) if k["vis"] else front[:, -1]).float()
        out = (self.policy if grad else m.policy)(front, ego, S.tc[rows], mem, smask)
        total, Ls = self.LS(out, S, rows, anchor)
        if m.head is not None:
            Ls["head"], Ls["head_hinge"], _ = self.head_loss(m.head(mem, ego_in), rows)
            total = total + c.lam_head * (Ls["head"] + c.head_hinge_lam * Ls["head_hinge"])
        if m.pred is not None:
            Ls["fut"], Ls["fut_copy"], Ls["fut_cov"] = self.fut_loss(mem, front[:, -1], ego_in, *b["fut"])
            total = total + c.lam_fut * Ls["fut"]
        return total, Ls, out

    def head_loss(self, P, rows):
        """opb_probe's decoder loss: Huber on x, y (delta 1) + 3 x Huber on yaw (delta 0.1); footprint hinge on the labelled rows; ADE."""
        S, ok = self.S, self.S.has_fut[rows]
        if not ok.any():
            z = P.sum() * 0.0
            return z, z, z
        Y, Q = S.fut[rows][ok], P[ok]
        li = F.huber_loss(Q[..., :2], Y[..., :2], delta=1.0) + 3.0 * F.huber_loss(Q[..., 2], Y[..., 2], delta=0.1)
        hk = self.hinge.ok[rows][ok]
        lh = (torch.relu(self.cfg.head_hinge_margin - self.hinge.margins(Q[hk][..., 0], Q[hk][..., 1], Q[hk][..., 2], rows[ok][hk])).mean()
              if hk.any() else Q.sum() * 0.0)
        return li, lh, torch.hypot(Q[..., 0] - Y[..., 0], Q[..., 1] - Y[..., 1]).mean(1)

    def fut_loss(self, mem, cur, ego, tgt, ok):
        """smooth-L1 of the predicted against the standardised frozen tokens ahead, over the (row, horizon) pairs that exist; the same for
        the copy baseline `future = the frozen current token`; coverage."""
        p = self.model.pred
        if not ok.any():
            z = mem.sum() * 0.0
            return z, z, z
        y = p.z(tgt)[ok]
        return (F.smooth_l1_loss(p(mem, ego)[ok], y), F.smooth_l1_loss(p.z(cur)[:, None].expand(-1, ok.shape[1], -1, -1)[ok], y), ok.float().mean())

    def step(self, item, step, lr_scale=1.0):
        c = self.cfg
        (rows_np, an_np, sm_np), b = item
        rows, anchor = torch.as_tensor(rows_np, device=self.dev), torch.as_tensor(an_np, device=self.dev)
        smask = torch.as_tensor(sm_np, device=self.dev) if sm_np is not None else None
        for g in self.opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (step + 1) / c.warmup) * lr_scale
        total, Ls, _ = self.losses(b, rows, anchor, smask)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.opt)
        gn = [float(torch.nn.utils.clip_grad_norm_(ps, 1.0)) for ps in self.clip]
        self.scaler.step(self.opt)
        self.scaler.update()
        Ls = {k: float(v.detach()) for k, v in Ls.items()} | {"total": float(total.detach()), "gn_policy": gn[0]}
        if self.k["vis"]:
            Ls["gn_vis"] = gn[1]
        return Ls | {"skipped": float(not np.isfinite(gn).all())}                # a non-finite fp16 gradient: GradScaler skips the step

    @torch.no_grad()
    def dev_eval(self, bs=128) -> dict:
        """dev rows: ADE to the log with the memory on / masked (W: also with only the side views masked), drift to shipped with the inputs
        on / off, the branch head's loss and ADE, B's future loss with its copy baseline."""
        m, S, k, dev = self.model.eval(), self.S, self.k, self.dev
        pi = torch.as_tensor(S.pi, device=dev)
        acc = {}
        add = lambda n, v: acc.setdefault(n, []).append(v.reshape(-1).float())  # noqa: E731
        for i in range(0, len(self.dv_rows), 32 if k["vis"] == "inplace" else bs):
            rn = self.dv_rows[i:i + (32 if k["vis"] == "inplace" else bs)]
            r = torch.as_tensor(rn, device=dev)
            mem = None
            if k["vis"] == "inplace":
                front = m.slots(*self.PS.all(rn))
            else:
                front = S.front[rn]
                if k["mem"]:
                    mem = (m.branch(self.PS.t0(rn), self.SS.t0(rn) if self.SS else None) if k["vis"] else front[:, -1]).float()
            tx, ty, _ = T.rear(S.t_plan[r], S.cam_x[r], self.LS.W)
            ok = S.has_fut[r]
            for name in ("on", "off") + (("masked",) if k["mem"] else ()) + (("sideoff",) if self.SS else ()):
                mask = mem_mask(len(r), dev, {"masked": "off", "sideoff": "sideoff"}.get(name, "on"))
                p = m.policy(front, S.ego[r], S.tc[r], mem, mask, inputs_on=name != "off").float()[:, pi].view(-1, 33, 15)
                x, y, _ = T.rear(p, S.cam_x[r], self.LS.W)
                if name in ("on", "off"):
                    add(f"drift_{name}", torch.hypot(x - tx, y - ty).mean(1))
                if name != "off":
                    add("ade" if name == "on" else f"ade_{name}", torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1)[ok])
            if m.head is not None:
                li, lh, ade = self.head_loss(m.head(mem, S.ego[r]), r)
                add("head", li), add("head_hinge", lh), add("head_ade", ade)
            if m.pred is not None:
                lf, lc, cov = self.fut_loss(mem, front[:, -1], S.ego[r], *self.fut_targets(rn))
                add("fut", lf), add("fut_copy", lc), add("fut_cov", cov)
        m.train()
        return {n: float(torch.cat(v).mean()) for n, v in acc.items()}


def _atomic_save(obj, f: _pl.Path):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, f)


def snap_tag(tag: str, step: int) -> str:
    return f"{tag}-k{step // 1000:02d}"


def claim(tag: str, cfgd: dict, a) -> _pl.Path:
    """The tag's run dir, claimed for this run. A tag (or a snapshot of it) that exists is refused, unless --resume finds this run's own
    marker with the same configuration. Exits 0 when that run already finished, 3 when it was stopped by a stop rule."""
    d, mark = RUNS / tag, RUNS / tag / "vt_run.json"
    snaps = sorted(RUNS.glob(f"{tag}-k[0-9]*"))
    if a.scratch and tag.startswith("smoke-vt"):
        for x in [d, *snaps]:
            shutil.rmtree(x, ignore_errors=True)
        snaps = []
    if d.exists() or snaps:
        own = mark.exists() and json.loads(mark.read_text()) == cfgd
        if not (a.resume and own):
            raise SystemExit(f"{tag}: {d} or a snapshot dir exists ({[x.name for x in snaps]}); not this run's own (--resume: {a.resume}, marker match: {own})")
        if (d / "STOP").exists():
            print(f"{tag}: stopped earlier by a stop rule: {(d / 'STOP').read_text()}", flush=True)
            raise SystemExit(3)
        if (d / "ckpt-final.pt").exists():
            print(f"{tag}: already finished", flush=True)
            raise SystemExit(0)
    else:
        d.mkdir(parents=True)
        mark.write_text(json.dumps(cfgd))
    return d


# ---------------------------------------------------------------- train
def cmd_train(a):
    from jevdrive.run import Run
    dev = torch.device("cuda")
    k = ARMS[a.arm]
    cfg = Cfg(arm=a.arm, seed=a.seed, steps=a.steps, eval_every=a.eval_every, snap_every=a.snap_every, data=tuple(a.data), split=a.split,
              d_frac=0.25 if k["anchor"] else 0.0, lam_d=30.0 if k["anchor"] else 0.0, init=a.init or f"SH30-F-s{a.seed}")
    tag = a.tag or f"VT-{a.arm}-s{a.seed}"
    cfgd = json.loads(json.dumps(asdict(cfg)))
    d = claim(tag, cfgd, a)
    with Run("vis_train", f"train-{tag}", seed=cfg.seed, config=cfgd | {"tag": tag, "speed": dict(prefetch=a.prefetch, enc_ckpt=a.enc_ckpt, compile=a.compile, enc_compile=a.enc_compile) | ({"enc_chunk": a.enc_chunk} if a.enc_chunk else {})}) as run:
        ln = Lane(cfg, dev, a)
        m = ln.model
        for x in ln.sp:
            run.use_split(x)
        step0, ade0, bad, hist_ev = 0, None, 0, []
        rs = d / "resume.pt"
        if a.resume and rs.exists():
            st = torch.load(rs, map_location="cpu", weights_only=False)
            m.load_state(st["model"])
            ln.opt.load_state_dict(st["opt"])
            ln.scaler.load_state_dict(st["scaler"])
            step0, ade0, bad, hist_ev = st["step"], st["ade0"], st["bad"], st["evals"]
            for _ in range(step0):                                            # replay the row streams to the snapshot step
                ln.draw()
            run.info(f"{tag}: resumed at step {step0} from {rs}")
        run.info(f"{tag} (arm {cfg.arm}, from {cfg.init}): train {len(ln.tr_rows)} dev {len(ln.dv_rows)} rows; trainable " +
                 ", ".join(f"{n} {v / 1e6:.2f}M" for n, v in ln.n_par.items()) + f"; hinge labels cover {ln.hinge.coverage:.4f}; "
                 f"speed: prefetch {a.prefetch} x {a.fetch_workers} threads, fast_encode {hasattr(PX, 'fast_encode')}, enc_ckpt {a.enc_ckpt}, enc_chunk {a.enc_chunk}, compiled policy {a.compile} / encoder {a.enc_compile}")

        def evaluate(step):
            nonlocal ade0, bad
            ev = ln.dev_eval() | {f"disp_{g}": v for g, v in ln.disp(m).items()}
            run.scalars({(f"disp/{n[5:]}" if n.startswith("disp_") else f"dev/{n}"): v for n, v in ev.items()}, step)
            run.info(f"dev @ {step}: " + ", ".join(f"{n} {v:.4f}" for n, v in ev.items()))
            run.summary.update({f"dev_{n}": v for n, v in ev.items()})
            hist_ev.append({"step": step} | ev)
            (d / "evals.json").write_text(json.dumps(hist_ev))
            if ade0 is None:                                                  # the starting point = SH30 itself (memory arms: memory masked)
                ade0 = ev.get("ade_masked", ev["ade"])
            # armed after the warmup: on the memory arms the memory-on ADE of step 0 is ~0.5 m over the start by construction (fresh side_in,
            # prereg amendment 1 point 5), which is not a regression (it tripped the 6-step preflight smoke: amendment 4)
            bad = bad + 1 if (step >= cfg.warmup and ev["ade"] > ade0 + 0.3) else 0
            if bad >= 2:                                                      # registered stop rule: this arm stops, the others go on
                save(step, final=False, snapshot=False)
                (d / "STOP").write_text(f"dev ADE {ev['ade']:.3f} m > start {ade0:.3f} + 0.3 m at two consecutive evals (step {step})\n")
                raise RuntimeError(f"stop rule: dev ADE {ev['ade']:.3f} vs start {ade0:.3f} at step {step}")

        def save(step, final, snapshot=True):
            ck = {"model": m.state(), "cfg": cfgd, "step": step, "tag": tag}
            if final:
                _atomic_save(ck, d / "ckpt-final.pt")
                rs.unlink(missing_ok=True)
                return
            _atomic_save(ck | {"opt": ln.opt.state_dict(), "scaler": ln.scaler.state_dict(), "ade0": ade0, "bad": bad, "evals": hist_ev}, rs)
            if snapshot:
                _atomic_save(ck, RUNS / snap_tag(tag, step) / "ckpt-final.pt")

        if step0 == 0:
            evaluate(0)
        pre = T.Prefetch(ln.draw, ln.fetch, cfg.steps - step0, a.prefetch, a.fetch_workers)
        t0, tw, hist = time.time(), time.time(), []
        for step in range(step0, cfg.steps):
            hist.append(ln.step(pre.get(), step))
            n = step + 1
            if n % 25 == 0 or n == cfg.steps:
                mean = {x: float(np.nanmean([h[x] for h in hist if x in h])) for x in hist[-1]}
                its = len(hist) / (time.time() - tw)
                hist, tw = [], time.time()
                run.scalars({f"loss/{x}": v for x, v in mean.items()} | {"throughput/steps_per_s": its, "throughput/samples_per_s": its * cfg.batch,
                                                                         "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, n)
                if n % 100 == 0 or n == cfg.steps:
                    run.info(f"step {n}: " + ", ".join(f"{x} {v:.4f}" for x, v in mean.items()) +
                             f"; {its:.2f} it/s ({(n - step0) / (time.time() - t0):.2f} since start), {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {n}/{cfg.steps}, {its:.2f} it/s, eta {(cfg.steps - n) / max(its, 1e-6) / 3600:.2f} h")
            if n % cfg.eval_every == 0 or n == cfg.steps:
                evaluate(n)
                tw = time.time()
            if n % cfg.snap_every == 0 and n < cfg.steps:
                save(n, final=False)
                tw = time.time()
        save(cfg.steps, final=True)
        run.summary.update(steps=cfg.steps, train_s=time.time() - t0, it_s=(cfg.steps - step0) / (time.time() - t0), ckpt=str(d / "ckpt-final.pt"),
                           gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


# ---------------------------------------------------------------- inference: plans, tokens, the identity gate
class Infer:
    """A VT model on one cached data dir (lb_navtest, lb_navhard, a navtrain shard): cached slot tokens + the pixel cache where the arm reads pixels."""

    def __init__(self, model: VT, data: str, dev, host=False):
        self.m, self.dev, self.k = model, dev, model.k
        self.S = T.Store([data], dev, need_side=False, frames="warp", host=host)
        self.PS = PX.PixelStore([data], dev, mode="all" if self.k["vis"] == "inplace" else "t0") if self.k["vis"] else None
        assert self.PS is None or len(self.PS) == self.S.n
        self.SS = SD.SideStore([data], dev) if model.nv > 1 else None
        assert self.SS is None or (self.SS.names == self.S.tab["names"]).all()

    @torch.no_grad()
    def mem(self, rn):
        return (self.m.branch(self.PS.t0(rn), self.SS.t0(rn) if self.SS else None) if self.k["vis"] else t0_tok(self.S.front, rn)).float()

    @torch.no_grad()
    def out(self, rn, mem="on", perm=None):
        """Raw outputs (n, n_out) fp32 of rows rn. mem: on | off (masked) | shuf (the memory of row perm[rn]) | sideoff (W: side views masked)."""
        S, r = self.S, torch.as_tensor(rn, device=self.dev)
        front = self.m.slots(*self.PS.all(rn)) if self.k["vis"] == "inplace" else S.front[r]
        mm = self.mem(perm[rn] if mem == "shuf" else rn) if self.k["mem"] else None
        return self.m.policy(front, S.ego[r], S.tc[r], mm, mem_mask(len(r), self.dev, mem) if self.k["mem"] else None).float()


def plan_arrays(I: Infer, n: int, mem="on", perm=None, bs=0, tq=None):
    sl = I.m.net.slices
    pi, ps = np.arange(sl["plan"].start, sl["plan"].start + 495), np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
    mu, sd = np.zeros((n, 33, 15), np.float32), np.zeros((n, 33, 15), np.float32)
    bs = bs or (32 if I.k["vis"] == "inplace" else 128)
    for i in (tq or (lambda x, **kw: x))(range(0, n, bs), desc="plans"):
        rn = np.arange(i, min(i + bs, n))
        o = I.out(rn, mem, perm).cpu().numpy()
        mu[rn], sd[rn] = o[:, pi].reshape(-1, 33, 15), np.exp(np.minimum(o[:, ps], 11)).reshape(-1, 33, 15)
    return mu, sd


def cmd_plans(a):
    import op_lb as OL
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    mt = OL.meta(a.data)
    with Run("vis_train", f"plans-{a.tag}-{a.data}-{a.mem}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{mt['split']}"))
        m, ck = load_tag(a.tag, dev)
        assert a.mem == "on" or m.k["mem"], f"{a.tag} (arm {m.arm}) has no memory channel to mask or shuffle"
        assert a.mem != "sideoff" or m.nv > 1, f"{a.tag} (arm {m.arm}) has no side views"
        I = Infer(m, a.data, dev)
        assert I.S.tab["names"].tolist() == mt["names"], "pp_prep cache rows differ from op_lb meta"
        perm = derange(I.S.tab["log"], np.random.default_rng(0)) if a.mem == "shuf" else None
        t0 = time.time()
        mu, sd = plan_arrays(I, I.S.n, a.mem, perm, tq=run.tqdm)
        assert np.isfinite(mu).all()
        out = _pl.Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.{os.getpid()}.npz")
        np.savez(tmp, names=np.array(mt["names"]), plan_pos=mu[:, :, 0:3], plan_vel=mu[:, :, 3:6], plan_yaw=mu[:, :, 11], plan_mu=mu, plan_std=sd, steps=31,
                 info=json.dumps({"model": f"vis_train {a.tag} (arm {m.arm}, step {ck.get('step')})", "frames": "warp", "mem": a.mem,
                                  "source": "experiments/vis_train/scripts/vt.py plans"}))
        os.replace(tmp, out)
        run.summary.update(n=I.S.n, tokens_per_s=I.S.n / (time.time() - t0), out=str(out), arm=m.arm, mem=a.mem)


def cmd_tokens(a):
    from jevdrive.run import Run
    dev = torch.device("cuda")
    with Run("vis_train", f"tokens-{a.tag}", config=vars(a)) as run:
        m, _ = load_tag(a.tag, dev)
        assert m.k["vis"] or m.k["mem"], f"{a.tag} (arm {m.arm}) has neither a branch nor its own encoder"
        if m.k["vis"] == "inplace":                                           # C: the encoder's own t0 token, read as a branch reads it
            m.k = m.k | {"vis": "branch", "mem": True}
        for data in a.datas:
            I = Infer(m, data, dev, host=True)
            od = OUT / "tokens" / a.tag
            od.mkdir(parents=True, exist_ok=True)
            o = np.lib.format.open_memmap(od / f".{data}.npy", "w+", np.float16, (I.S.n, *I.mem(np.arange(1)).shape[1:]))   # W: (N, 3, 32, 512)
            for i in run.tqdm(range(0, I.S.n, 256), desc=data):
                rn = np.arange(i, min(i + 256, I.S.n))
                o[rn] = I.mem(rn).half().cpu().numpy()
            o.flush()
            del o
            os.replace(od / f".{data}.npy", od / f"{data}.npy")
            np.save(od / f"{data}.names.npy", I.S.tab["names"])
            run.info(f"{data}: {I.S.n} rows -> {od / (data + '.npy')}")
        run.summary.update(out=str(OUT / "tokens" / a.tag), datas=a.datas)


def cmd_ident(a):
    """Identity gate (prereg check 2), under the run's speed settings: the arm built from SH30-F-s<seed> must reproduce SH30's cached-token
    plans when what it adds is off (memory masked; C: its encoder on the pixel cache, after --lr0-steps real training steps at lr 0)."""
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    k = ARMS[a.arm]
    cfg = Cfg(arm=a.arm, seed=a.seed, steps=max(a.lr0_steps, 1), data=tuple(a.data), split=a.split, d_frac=0.25 if k["anchor"] else 0.0,
              lam_d=30.0 if k["anchor"] else 0.0, init=f"SH30-F-s{a.seed}")
    with Run("vis_train", f"ident-{a.arm}-s{a.seed}", seed=a.seed, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        ln = Lane(cfg, dev, a)
        for x in ln.sp:
            run.use_split(x)
        m = ln.model
        res = {"arm": a.arm, "seed": a.seed, "lr0_steps": a.lr0_steps, "n": a.n, "tol_m": TOL_M, "points": NPT}
        if a.lr0_steps:
            pre = T.Prefetch(ln.draw, ln.fetch, a.lr0_steps, a.prefetch, a.fetch_workers)
            ls = [ln.step(pre.get(), s, lr_scale=0.0) for s in range(a.lr0_steps)]
            res["lr0_loss"] = ls[-1]
            res["lr0_disp"] = ln.disp(m)
            assert max(res["lr0_disp"].values()) == 0.0, f"weights moved at lr 0: {res['lr0_disp']}"
        m.eval()
        I = Infer(m, "lb_navtest", dev)
        N = min(a.n, I.S.n)
        P = {"arm": plan_arrays(I, N, "off")[0]}
        if k["mem"]:
            P["arm_mem_on"] = plan_arrays(I, N, "on")[0]
        if m.nv > 1:
            P["arm_side_off"] = plan_arrays(I, N, "sideoff")[0]
        for name, tag in (("ref", f"SH30-F-s{a.seed}"), ("other_seed", f"SH30-F-s{1 - a.seed}")):
            pm = T.load_pmodel(tag, dev)
            pi = np.arange(pm.net.slices["plan"].start, pm.net.slices["plan"].start + 495)
            with torch.no_grad():
                P[name] = np.concatenate([pm(I.S.front[r], I.S.ego[r], I.S.tc[r], None, None).float().cpu().numpy()[:, pi].reshape(-1, 33, 15)
                                          for r in (torch.arange(i, min(i + 128, N), device=dev) for i in range(0, N, 128))])
            del pm
        dd = lambda x, y: np.abs(x[:, :NPT, :2] - y[:, :NPT, :2]).max((1, 2))  # noqa: E731
        for name in [n for n in P if n != "ref"]:
            e = dd(P[name], P["ref"])
            res[f"{name}_vs_ref"] = dict(share_over_tol=float((e > TOL_M).mean()), share_over_2ulp=float((e > ULP2_M).mean()), max_m=float(e.max()),
                                         median_m=float(np.median(e)), mean_m=float(e.mean()))
        if m.nv > 1:                                                          # W: the side views are read (masking them moves the plans)
            e = dd(P["arm_side_off"], P["arm_mem_on"])
            res["side_off_vs_mem_on"] = dict(share_over_tol=float((e > TOL_M).mean()), max_m=float(e.max()), median_m=float(np.median(e)), mean_m=float(e.mean()))
        if k["mem"]:                                                          # the masked-memory bias against SH30's own adapter (fp32, before the fp16 sum)
            ref = PA.ParityAdapter(use_ego=True, use_side=False).to(dev)
            ref.load_state_dict(torch.load(RUNS / cfg.init / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]["parity"])
            with torch.no_grad():
                e, tk = I.S.ego[:N], I.mem(np.arange(min(N, 256)))
                b0 = ref(e[:len(tk)])
                b1 = m.adapter(e[:len(tk)], side5(tk), torch.zeros(len(tk), 1, dtype=torch.bool, device=dev))
            res["masked_bias_vs_ref"] = dict(max_abs=float((b1 - b0).abs().max()), ref_rms=float(b0.pow(2).mean().sqrt()))
        if k["vis"]:                                                          # the pixel cache reproduces the cached tokens (frozen weights)
            rn = np.arange(min(512, N))
            tk = (m.slots(*I.PS.all(rn))[:, -1] if k["vis"] == "inplace" else m.encode(*I.PS.t0(rn))).float()
            ref = t0_tok(I.S.front, rn).float()
            res["px_tokens_vs_cache"] = dict(max_abs=float((tk - ref).abs().max()), mean_abs=float((tk - ref).abs().mean()), ref_rms=float(ref.pow(2).mean().sqrt()))
            if m.nv > 1:                                                      # W's F0 view against arm A's branch tokens (tk: the F0 pairs alone), and the side views' scale
                w = I.mem(rn)
                res["f0_view_vs_arm_A"] = dict(max_abs=float((w[:, 0] - tk).abs().max()), mean_abs=float((w[:, 0] - tk).abs().mean()))
                res["side_view_vs_f0"] = dict(mean_abs=float((w[:, 1:] - w[:, :1]).abs().mean()), rms=float(w[:, 1:].pow(2).mean().sqrt()))
                res["head_norm"] = dict(mu_rms=[float(v.pow(2).mean().sqrt()) for v in m.head.mu[:-PA.EGO_DIM].view(m.nv, -1)],
                                        sd_mean=[float(v.mean()) for v in m.head.sd[:-PA.EGO_DIM].view(m.nv, -1)])
        res["pass"] = bool(res["arm_vs_ref"]["share_over_tol"] < 1e-3 and res["other_seed_vs_ref"]["share_over_tol"] > 0.5)      # the registered line
        res["pass_2ulp"] = bool(res["arm_vs_ref"]["share_over_2ulp"] < 1e-3 and res["other_seed_vs_ref"]["share_over_2ulp"] > 0.5)   # prereg amendment
        if m.nv > 1:
            res["pass_2ulp"] &= res["side_off_vs_mem_on"]["share_over_tol"] > 0.5
        (OUT / "ident").mkdir(parents=True, exist_ok=True)
        (OUT / "ident" / f"{a.arm}-s{a.seed}.json").write_text(json.dumps(res, indent=1))
        run.info(json.dumps(res))
        run.summary.update(res)
        if not res["pass_2ulp"]:
            raise SystemExit(f"identity gate failed: {res['arm_vs_ref']} (control {res['other_seed_vs_ref']})")


def speed_args(p):
    p.add_argument("--prefetch", type=int, default=4, help="batches fetched ahead on worker threads (page-locked gathers)")
    p.add_argument("--fetch-workers", type=int, default=3)
    p.add_argument("--enc-ckpt", type=int, default=0, help="activation checkpointing chunk of the encoder's grad pass (0 = off)")
    p.add_argument("--enc-chunk", type=int, default=0, help="pairs per call of the encoder's fast grad pass (0 = the whole batch; W: 64 = one view per call)")
    p.add_argument("--vram-cap", type=float, default=0, help="GB: hard cap of this process's CUDA allocator (0 = none), so a job cannot outgrow its pool booking")
    p.add_argument("--compile", action="store_true", help="torch.compile of the policy forward")
    p.add_argument("--enc-compile", action="store_true", help="torch.compile of the encoder's grad pass (pixel_store.fast_encode)")
    p.add_argument("--data", nargs="+", default=list(FULL))
    p.add_argument("--split", default="navsim/op-parity-full")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("train")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--tag", default="")
    p.add_argument("--init", default="", help="starting checkpoint tag (default SH30-F-s<seed>)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--snap-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true", help="continue this tag's own interrupted run (a fresh start when there is none)")
    p.add_argument("--scratch", action="store_true", help="smoke-vt* tags only: delete the tag's dirs first")
    speed_args(p)
    p = sp.add_parser("plans")
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default="lb_navtest")
    p.add_argument("--out", required=True)
    p.add_argument("--mem", default="on", choices=["on", "off", "shuf", "sideoff"])
    p = sp.add_parser("tokens")
    p.add_argument("--tag", required=True)
    p.add_argument("--datas", nargs="+", default=["lb_navtest", "navtrain_full.s2of12", "navtrain_full.s3of12", "navtrain_full.s4of12"])
    p = sp.add_parser("ident")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--lr0-steps", type=int, default=0)
    speed_args(p)
    a = ap.parse_args()
    {"train": cmd_train, "plans": cmd_plans, "tokens": cmd_tokens, "ident": cmd_ident}[a.cmd](a)
