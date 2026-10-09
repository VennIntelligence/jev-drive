"""op_parity trainer: Cinque fine-tuned on navtrain with WA-JEPA's extra inputs (plans/2026-10-06-parity-prereg.md), from the pp_prep cache.

Model = shipped Cinque port (jevdrive.op_adapt, fp16 compute) with the vision encoder FROZEN (its hidden tokens come from the cache), the
off-policy plan pathway trainable (ONNX nodes 479-665, fp32 masters; experiments/op_adapt_l pol_weights) and lib/parity_adapter's bias on
the 9 context frames:
  P1  adapter absent (= every new input zeroed): fine-tuning alone
  P2  ego status + 4-pose history + command
  P3  P2 + CAM_L0 / CAM_R0 / CAM_B0 tokens (each camera dropped per row with p = cam_drop in training, so "side off" is in distribution)
Every arm: same rows, same row order (seeded), same targets, same steps.

Batch rows: (1 - d_frac) imitation rows (plan -> logged 8 poses x, y, yaw at 0.5 .. 4 s; every non-plan head distilled to shipped) and d_frac
anchor rows (new inputs zeroed, plan + every head distilled to shipped on the same navtrain frames: the decision-137 guard, built from
navtrain only).

Joint action arms (plans/2026-10-07-joint-action-prereg.md, --act-lab): the on-policy action pathway is trainable too and action[0] (the
lateral command openpilot steers with: desired curvature = action[0] / max(1, v)^2, + = right) is no longer distilled to shipped; it is
fitted on every row to a label x max(1, v0)^2: `plan` the curvature of the model's own (detached) plan over 0.5-1.5 s (= HUGSIM
spec_plan_smooth), `log` the logged curvature at t + 0.275 s (cubic spline through the 4 history + 8 future poses), `logwin` the logged
curvature over 0.5-1.5 s. Speed weight min(1, v0 / 3); vy / ay of the ego input zeroed on a fraction --ego-lat-drop of rows (HUGSIM feeds 0). Multi-GPU: torchrun -> DDP-style gradient all-reduce, each rank its own row stream; single GPU without torchrun.

B2D rows (experiments/op_parity/scripts/b2d_prep.py, plans/2026-10-07-b2d-p2-prereg.md): a data dir named b2d_* is a tick-indexed token store
(ticks.npy + front_idx.npy, read through IndexedMM; needs --host), its rows are split by route (--b2d-split b2d/b2dc-v2: -train / -val), sampled with
--b2d-mass of the batch (turn balancing then acts inside the B2D rows only), and carry the MKZ footprint hinge (--hinge-labels may list one label
file per source, --hinge-footprint one footprint per file).

Mixed-domain rows (plans/2026-10-08-mixed-domain-prereg.md, --wod-mass): wod_* data dirs (scripts/wod_parity.py prep) next to NAVSIM ones. NAVSIM
rows are split by token (--split), the wod_* rows by sequence (--wod-split wod/r2: -train / -dev); every batch takes --wod-mass of its rows from
the wod_* rows. Each source keeps its own frames (NAVSIM: the --frames protocol, 8 slots + a zero slot; WOD: the cached real frames) and its own
teacher; the hinge acts on the rows its label file covers (NAVSIM). --wod-slots 8 zeroes the oldest of the 9 WOD slots (teacher8.npz of
scripts/mixed_domain.py teacher8), so the number of real slots is not a domain cue. Needs --host.

  python experiments/op_parity/scripts/pp_train.py --arm P2 --steps 600 --data lb_navtrain lb_h1train [--tag pilot]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib")]
import argparse, json, os, time  # noqa: E401,E402
from dataclasses import asdict, dataclass  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import parity_adapter as PA  # noqa: E402
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402
from experiments.op_adapt_r2.lib import op_adapt_r2 as R2  # noqa: E402

AT = (0.275, 0.525)
T8 = 0.5 * np.arange(1, 9)
SIG_X, SIG_Y, SIG_PSI = 0.3 + 0.2 * T8, 0.1 + 0.1 * T8, np.radians(1.0 + 1.0 * T8)
ARMS = {"P1": dict(ego=False, side=False), "P2": dict(ego=True, side=False), "P3": dict(ego=True, side=True)}
# representation fix (plans/2026-10-07-representation-design.md, --mem): P2 + 32 front JEPA tokens as adapter memory (side channel, n_cam = n_t = 1)
# turn-oracle (plans/2026-10-08-turn-oracle-prereg.md): drivable-SDF banks in the same 32 x 512 slot (scripts/turn_oracle.py bank): sdf_gt = true
# geometry (privileged, an oracle probe only), sdf_shuf = the same rows permuted across logs (matched control), sdf_wa / sdf_v = probe read-outs
# geo-oracle (plans/2026-10-09-geo-oracle-prereg.md, scripts/geo_oracle.py tok): tokenizer outputs of the true drivable SDF (geo_s), true agent
# occupancy (geo_a), both (geo_b) and geo_b permuted across logs (geo_x); privileged, oracle probes only
MEM_KINDS = ("wa_cf", "vj21", "sdf_gt", "sdf_shuf", "sdf_wa", "sdf_v", "geo_s", "geo_a", "geo_b", "geo_x")
ARMS |= {f"P2+{k}": dict(ego=True, side=False, mem=k) for k in MEM_KINDS}
MEM_ROOT = data_dir() / "runs" / "op_parity" / "mem"          # <kind>/<data>.npy (N, 32, 512) fp16 in tab order (scripts/rep.py mem)
MEM_DROP = 0.25                                                 # rows whose memory is masked in training ("memory off" in distribution)
# geo-e2e (plans/2026-10-09-geo-e2e-prereg.md, scripts/geo_e2e.py, --mem-e2e): the memory tokens come from a tokenizer trained jointly with the
# adapter; arm "P2+ge_<tag>", its navtest bank is written to mem/ge_<tag>/ after training. Privileged rasters, oracle probes only.
# path-req (plans/2026-10-09-path-req-prereg.md, scripts/path_req.py, --mem-e2e q<kind>): the same channel fed with degraded fields of the LOGGED
# FUTURE path (label leak: oracle probes only); --mem-lr sets the tokenizer's own learning rate.


def arm_kw(arm: str) -> dict:
    return ARMS.get(arm) or (dict(ego=True, side=False, mem=arm[3:]) if arm.startswith("P2+ge_") else dict(ego=False, side=False))


ACT_COL = 2062                                    # raw output column of action[0] mu (lateral; openpilot sign, + = right turn)
ACT_T, ACT_WIN = 0.275, (0.5, 1.5)                # lateral action time; the spec_plan_smooth window (jevdrive/openpilot/model.py)
EGO_LAT = [5, 7]                                  # vy / 10, ay / 3 in parity_adapter.ego_features


@dataclass
class Cfg:
    arm: str
    seed: int = 0
    steps: int = 600
    batch: int = 64
    d_frac: float = 0.25
    cam_drop: float = 0.15
    lam_i: float = 1.0
    lam_c: float = 3.0
    lam_d: float = 30.0
    lr: float = 3e-5
    lr_new: float = 3e-4
    wd: float = 0.01
    warmup: int = 100
    eval_every: int = 200
    data: tuple = ("lb_navtrain", "lb_h1train")
    host: bool = False                    # token arrays gathered per batch from the page cache (full navtrain) instead of held on the GPU
    frames: str = "gimm"                  # front protocol (prereg addendum 1): gimm (G) | warp (W) | keys (N)
    split: str = "navsim/op-parity-pilot"
    hinge_lam: float = 0.0                # footprint drivable-area SDF hinge on the plan (lib/drivable_hinge.py), imitation rows only; 0 = off
    hinge_margin: float = 0.3
    hinge_labels: tuple = ("runs/op_probe/labels/navtrain_all.npz",)   # one label file per source (first file that labels a row wins)
    hinge_footprint: tuple = ("pacifica",)                                # footprint of each label file: pacifica (NAVSIM) | mkz (B2D)
    hinge_replay: bool = False            # hinge on the devkit LQR replay of the plan (lib/replay_hinge.py, plans/2026-10-07-replay-hinge-prereg.md)
    hinge_front_turn_margin: float = 0.0  # replay hinge: front-corner margin on rows with logged |heading change| > 20 deg (RM); 0 = off
    agent_lam: float = 0.0                # agent-box hinge on the plan (lib/agent_hinge.py, plans/2026-10-07-agent-hinge-prereg.md), imitation rows; 0 = off
    agent_margin: float = 0.5
    agent_side_margin: float = -1.0       # margin for objects outside the ego's lateral corridor; < 0 = agent_margin
    agent_labels: str = "runs/op_parity/agent_labels/navtrain_all.npz"
    b2d_split: str = "b2d/b2dc-v2"                                        # route split of the b2d_* data dirs: <ref>-train / <ref>-val
    anchor_b2d: bool = True               # anchor rows may fall on b2d_* rows (False: anchors only on the other sources, B2D rows are imitation only)
    b2d_mass: float = 0.0                                                 # share of the imitation batch rows drawn from the b2d_* rows (0 = natural mix)
    turn_bal: str = ""                    # turn-balanced sampling: target mass per |heading change| bin (<5, 5-20, 20-45, >45 deg), "a,b,c,d"; "" = off
    anchor_off_turn: bool = False         # no anchor rows on tokens with logged |heading change| > 20 deg
    late_lat_w: float = 1.0               # loss weight on the y and yaw terms of the poses at >= 2 s (imitation rows), 1 = off
    act_lab: str = ""                     # action[0] label: "" distilled to shipped (P2 recipe) | plan | log | logwin (module docstring)
    act_lam: float = 3.0
    ego_lat_drop: float = 0.0             # fraction of rows with vy / ay of the ego input zeroed (own rng stream; row order unchanged)
    stop_gate: float = 0.0                # wod-launch (plans/2026-10-08-wod-launch-prereg.md addendum): adapter off (present = 0) on rows fed a speed below this (m/s); 0 = off
    mem: str = ""                         # front-token memory (wa_cf | vj21; arm P2+<mem>), dropped per row with MEM_DROP (own rng stream)
    mem_e2e: str = ""                     # geo-e2e: memory tokens from a jointly trained tokenizer over b | x | p rasters (geo_e2e.py); mem becomes ge_<tag>
    mem_init: str = ""                    # geo-e2e: tokenizer state dict to start from ("" = random init)
    mem_lr: float = 0.0                   # geo-e2e / path-req: learning rate of the tokenizer group (0 = lr_new)
    stop_gate_free: bool = False          # mixed-domain addendum 2: no anchor rows on the gated rows (every gated row with a log is an imitation row)
    wod_split: str = "wod/r2"             # mixed-domain: sequence split of the wod_* data dirs when --split is a NAVSIM token split (<ref>-train / -dev)
    wod_mass: float = 0.0                 # mixed-domain: exact share of every batch drawn from the wod_* rows (0 = natural mix)
    wod_slots: int = 0                    # mixed-domain: real policy slots kept on wod_* rows (8 = oldest slot zeroed, as NAVSIM rows; 0 = all cached)


def proot(*p) -> _pl.Path:
    d = data_dir() / "runs" / "op_parity" / _pl.Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- model
def act_weights(model="cinque") -> list[str]:
    """Float initializers of the on-policy action pathway (its temporal summarizer + action hydra), disjoint from L.pol_weights."""
    import onnx
    g = onnx.load(str(A.MODELS_DIR / A.FILES[model]), load_external_data=False).graph
    w = sorted(t.name for t in g.initializer if t.name.startswith("model.on_policy.") and t.data_type in (1, 10, 11, 16) and len(t.dims) >= 1)
    assert w and not set(w) & set(L.pol_weights(model)), "on-policy weights missing or shared with the plan pathway"
    return w


class PModel(nn.Module):
    def __init__(self, arm: str, dtype=torch.float16, pol: bool = True, act: bool = False):
        super().__init__()
        self.arm = arm
        self.gate, self.bias_sub = 0.0, None        # serving side only (op_parity stop-gate-xboard), off by default: bias = 0 where the fed speed < gate m/s; bias_sub (32, 512) is subtracted
        tr = (L.pol_weights() if pol else []) + (act_weights() if act else [])
        self.net = A.load("cinque", dtype, trainable=tr)
        k = arm_kw(arm)
        self.mem = k.get("mem")
        if self.mem:
            self.adapter = PA.ParityAdapter(use_ego=True, use_side=True, n_cam=1, n_t=1)
        else:
            self.adapter = PA.ParityAdapter(use_ego=k["ego"], use_side=k["side"]) if (k["ego"] or k["side"]) else None

    def forward(self, front, ego, tc, side=None, side_mask=None, inputs_on=True, nv=None):
        """front (B, n, 32, 512) cached hidden tokens of the n newest policy slots (n = 8: the 0.2 s protocols; 4: N, one slot per 2 Hz key)
        -> outputs (B, n_out). The 9 - n older slots are zero and invalid. inputs_on False / no adapter: the bias is not added.
        nv (B,) int, mixed-domain batches only: the number of real (newest) slots of each row; the older ones are invalid."""
        B, n = front.shape[:2]
        H = torch.cat([front.new_zeros(B, A.CONTEXT - n, *front.shape[2:]), front], 1).to(self.net.dtype)
        valid = torch.zeros(B, A.CONTEXT, dtype=torch.bool, device=H.device)
        valid[:, A.CONTEXT - n:] = True
        if nv is not None:
            valid = valid & (torch.arange(A.CONTEXT, device=H.device)[None] >= (A.CONTEXT - nv)[:, None])
        if self.mem and side is not None and side.dim() == 3:
            side = side[:, None, None]                                  # memory (B, 32, 512) -> side channel (B, 1 cam, 1 time, 32, 512)
        if self.adapter is not None and inputs_on:
            if self.gate > 0 or self.bias_sub is not None:
                b = self.adapter(ego, side if self.adapter.use_side else None, side_mask)
                if self.bias_sub is not None:
                    b = b - self.bias_sub.to(b)
                if self.gate > 0:
                    b = b * (ego[:, 4:5] * 10.0 >= self.gate).to(b.dtype)[:, :, None]
                H = H + b[:, None].to(H.dtype)
            else:
                H = self.adapter.apply(H, ego, side if self.adapter.use_side else None, side_mask)
        H = H * valid[:, :, None, None].to(H.dtype)
        o = self.net.run_batched(A.policy_feeds(self.net, H, AT, tc.to(self.net.dtype)), ["outputs"])
        return o["outputs"].reshape(B, -1)

    def groups(self):
        base = [p for p in self.net.params.values() if p.requires_grad]
        return base, (list(self.adapter.parameters()) if self.adapter is not None else [])

    def state(self) -> dict:
        return {"net": {k: p.detach().cpu() for k, p in self.net.params.items() if p.requires_grad}, "adapter": None,
                "parity": self.adapter.state_dict() if self.adapter is not None else None, "arm": self.arm}

    def load_state(self, st):
        for k, v in st["net"].items():
            self.net.params[k].data.copy_(v)
        if self.adapter is not None and st.get("parity") is not None:
            self.adapter.load_state_dict(st["parity"])


def load_pmodel(tag: str, dev) -> PModel:
    """'P0' -> shipped Cinque port (no adapter); 'P*-init' -> an arm at initialisation; else a run tag under $R/runs/<tag>/ckpt-final.pt."""
    if tag == "P0":
        return PModel("P0", pol=False).to(dev).eval()
    if tag.endswith("-init"):
        torch.manual_seed(0)
        return PModel(tag[:-5]).to(dev).eval()
    ck = torch.load(proot("runs", tag) / "ckpt-final.pt", map_location="cpu", weights_only=False)
    m = PModel(ck["model"]["arm"]).to(dev).eval()
    m.load_state(ck["model"])
    return m


def rear(plan: torch.Tensor, cam_x: torch.Tensor, W: torch.Tensor):
    """plan (B, 33, 15) camera frame (x fwd, y right, ch 11 yaw) -> rear-axle x, y (left +), yaw (left +) at the grid of W (K, 33)."""
    c = cam_x[:, None]
    x, y, psi = (plan[..., k] @ W.T for k in (0, 1, 11))
    psi = -psi
    return x + c - c * torch.cos(psi), -y - c * torch.sin(psi), psi


# ---------------------------------------------------------------- data on the GPU (or gathered per batch from the page cache)
class IndexedMM:
    """A b2d_prep cache as a (N, 8, 32, 512) fp16 array: row r = ticks[front_idx[r]] (the memory-mapped per-tick token store; no copy)."""

    def __init__(self, d):
        self.ticks = np.load(d / "ticks.npy", mmap_mode="r")
        self.idx = np.load(d / "front_idx.npy")
        self.shape = (len(self.idx), self.idx.shape[1]) + self.ticks.shape[1:]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, rows):
        r = np.arange(len(self.idx))[rows] if isinstance(rows, slice) else np.asarray(rows)
        return self.ticks[self.idx[r]]


def open_front(f):
    """np.load(f, mmap) of a front.npy, or the IndexedMM of a b2d_prep dir that has no front.npy."""
    f = _pl.Path(f)
    return np.load(f, mmap_mode="r") if f.exists() else IndexedMM(f.parent)


class Tokens:
    """Row access to token arrays of several cache dirs. On the GPU (one tensor), or host mode: the .npy files stay memory-mapped and every
    batch gathers its rows (the OS page cache is shared by all concurrent runs, so N runs cost the files once in RAM, not N copies)."""

    def __init__(self, files, dev, host=False):
        self.dev, self.host = dev, host
        mms = [open_front(f) if _pl.Path(f).name == "front.npy" else np.load(f, mmap_mode="r") for f in files]
        assert host or not any(isinstance(m, IndexedMM) for m in mms), "b2d_* token stores need --host (410k rows x 8 slots do not fit a card)"
        self.nslot = [m.shape[1] for m in mms]                                  # sources may differ in slot count (NAVSIM 8, WOD 9): host mode only
        self.ns, self.mixed = max(self.nslot), len(set(self.nslot)) > 1
        assert host or not self.mixed, "sources with different slot counts need --host"
        if host:
            self.mms, self.off = mms, np.cumsum([0] + [len(m) for m in mms])
        else:
            n = sum(len(m) for m in mms)
            self.t = torch.empty((n,) + mms[0].shape[1:], dtype=torch.float16, device=dev)
            i = 0
            for m in mms:                                                       # shard by shard: no host-side concatenation
                for j in range(0, len(m), 4096):
                    x = torch.from_numpy(np.ascontiguousarray(m[j:j + 4096]))
                    self.t[i:i + len(x)] = x.to(dev)
                    i += len(x)
        self.device = dev

    def __getitem__(self, rows):
        if not self.host:
            return self.t[rows]
        r = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        out = (np.zeros if self.mixed else np.empty)((len(r), self.ns) + self.mms[0].shape[2:], np.float16)
        k = np.searchsorted(self.off, r, side="right") - 1
        for s in np.unique(k):
            m = k == s
            loc = r[m] - self.off[s]
            o = np.argsort(loc)                                                 # sorted reads, then back to the batch order
            g = self.mms[s][loc[o]]
            tmp = np.empty_like(g)
            tmp[o] = g
            if self.mixed:
                out[m, self.ns - g.shape[1]:] = tmp                             # right-aligned: the missing older slots stay zero
            else:
                out[m] = tmp
        return torch.from_numpy(out).to(self.dev, non_blocking=True)


class Store:
    """The pp_prep caches of several data dirs, concatenated and moved to the device once (fp16 tokens)."""

    def __init__(self, datas, dev, need_side=True, rows=None, frames="gimm", host=False, mem=None, wod_slots=0):
        """frames: the front protocol (cache/<data>@<frames>/front.npy; gimm = cache/<data>/front.npy). tab, side and the teacher always come from
        cache/<data>/ (teacher = shipped Cinque on the G protocol, its in-distribution input: every protocol is anchored to the same targets)."""
        cr = data_dir() / "runs" / "op_parity" / "cache"
        own = ("b2d_", "wod_")                                                     # b2d_*: native 0.2 s frames = W protocol; wod_*: its real frames
        fdir = (lambda d: d) if frames == "gimm" else (lambda d: d if d.startswith(own) else f"{d}@{frames}")
        tabs = [dict(np.load(cr / d / "tab.npz")) for d in datas]
        self.tab = {k: np.concatenate([t[k] for t in tabs]) for k in tabs[0] if all(k in t for t in tabs)}   # mixed sources: the shared columns
        n = len(self.tab["names"])
        self.is_b2d = np.concatenate([np.full(len(t["names"]), d.startswith("b2d_")) for d, t in zip(datas, tabs)])
        self.is_wod = np.concatenate([np.full(len(t["names"]), d.startswith("wod_")) for d, t in zip(datas, tabs)])
        self.rows = np.arange(n) if rows is None else rows
        sel = self.rows
        assert rows is None, "row subsets are selected by the caller (split_rows)"
        t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        self.front = Tokens([cr / fdir(d) / "front.npy" for d in datas], dev, host)
        self.side = Tokens([cr / d / "side.npy" for d in datas], dev, host) if need_side else None
        tdir = lambda d: cr / fdir(d) if (cr / fdir(d) / "teacher.npz").exists() else cr / d  # noqa: E731  W full run: teacher on its own frames
        tfile = lambda d: tdir(d) / ("teacher8.npz" if wod_slots == 8 and d.startswith("wod_") else "teacher.npz")  # noqa: E731
        ns = [min(k, wod_slots) if wod_slots and d.startswith("wod_") else k for d, k in zip(datas, self.front.nslot)]
        self.nv = (t(np.concatenate([np.full(len(z["names"]), k) for z, k in zip(tabs, ns)])[sel])      # per-row real slots (PModel.forward nv)
                   if (self.front.mixed or wod_slots) else None)                                          # None = uniform (every other run)
        if all(tfile(d).exists() for d in datas):
            tz = [dict(np.load(tfile(d))) for d in datas]
            self.t_out = t(np.concatenate([z["out"] for z in tz])[sel])
            self.t_plan = t(np.concatenate([z["plan"] for z in tz])[sel])
            self.di, self.pi = tz[0]["di"], tz[0]["pi"]
        else:                                                                     # eval-only subsets (lb_hq_navtestX)
            self.t_out = self.t_plan = self.di = self.pi = None
        tb = {k: v[sel] for k, v in self.tab.items()}
        self.tb = tb
        self.ego = t(tb["ego"])
        self.fut = t(np.nan_to_num(tb["fut"]).astype(np.float32))
        self.has_fut = t(~np.isnan(tb["fut"][:, 0, 0]))
        self.cam_x = t(tb["cam"][:, 0].astype(np.float32))
        self.tc = t(np.where(tb["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32))
        self.v0 = t(tb["speed"].astype(np.float32))
        self.n = len(sel)
        self.mem = Tokens([MEM_ROOT / mem / f"{d}.npy" for d in datas], dev, host) if mem else None   # front-token memory (--mem)

    def act_labels(self, kind: str):
        k, ok = log_curv(self.tb, kind)
        self.klab, self.klab_ok = (torch.from_numpy(x).to(self.ego.device) for x in (k, ok))


def split_rows(tab, split_ref, b2d_split=None, wod_split=None) -> tuple:
    """Train / dev rows: NAVSIM rows by token (<split_ref>-train / -dev), b2d_* rows (tab["is_b2d"]) by route (tab["log"]; <b2d_split>-train / -val).
    A split whose unit is `sequence` (WOD, e.g. wod/r2) or `log` (NAVSIM cross-fit folds, sh30_crossfit.py) selects rows by tab["log"]
    (the WOD sequence / the NAVSIM log of the row; scripts/wod_parity.py)."""
    from jevdrive.data import splits
    tr, dv = splits.load(f"{split_ref}-train"), splits.load(f"{split_ref}-dev")
    by_log = tr.unit in ("sequence", "log")
    toks = tab["log"] if by_log else tab["names"]
    trm, dvm, sp = tr.mask(toks), dv.mask(toks), [tr, dv]
    if b2d_split and tab.get("is_b2d") is not None and tab["is_b2d"].any():
        bt, bv = splits.load(f"{b2d_split}-train"), splits.load(f"{b2d_split}-val")
        trm |= tab["is_b2d"] & bt.mask(tab["log"])
        dvm |= tab["is_b2d"] & bv.mask(tab["log"])
        sp += [bt, bv]
    if wod_split and not by_log and tab.get("is_wod") is not None and tab["is_wod"].any():   # mixed-domain: wod_* rows by sequence
        wt, wv = splits.load(f"{wod_split}-train"), splits.load(f"{wod_split}-dev")
        trm |= tab["is_wod"] & wt.mask(tab["log"])
        dvm |= tab["is_wod"] & wv.mask(tab["log"])
        sp += [wt, wv]
    return np.flatnonzero(trm), np.flatnonzero(dvm), tuple(sp)


# ---------------------------------------------------------------- losses
class Losses:
    def __init__(self, net, cfg: Cfg, tstd: torch.Tensor, di, pi, dev, hinge=None, agent=None):
        self.cfg, self.hinge, self.agent = cfg, hinge, agent
        self.W = torch.as_tensor(R2.t_weights(T8), device=dev)
        self.s = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (SIG_X, SIG_Y, SIG_PSI)]
        self.di = torch.as_tensor(di, device=dev)
        self.pi = torch.as_tensor(pi, device=dev)
        self.plan_cols = torch.as_tensor(np.isin(di, pi), device=dev)
        self.tstd = tstd
        self.dmask = torch.ones(len(di), device=dev)
        if cfg.act_lab:
            j = int(np.flatnonzero(np.asarray(di) == ACT_COL)[0])
            self.dmask[j] = 0.0                                                 # action[0] leaves the distillation
            self.asd = tstd[j]
            self.Wk = torch.as_tensor(R2.t_weights(np.asarray(ACT_WIN)), device=dev)

    def dist(self, plan, cam_x, tx, ty, tpsi, wl=None):
        x, y, psi = rear(plan, cam_x, self.W)
        h = lambda z: F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
        wl = 1.0 if wl is None else wl
        return (h((x - tx) / self.s[0]) + wl * (h((y - ty) / self.s[1]) + h((psi - tpsi) / self.s[2]))).mean(1)

    def __call__(self, out, S: Store, rows, anchor):
        c, out = self.cfg, out.float()
        plan = out[:, self.pi].view(-1, 33, 15)
        imit = ~anchor & S.has_fut[rows]
        Ls = {}
        if imit.any():
            f = S.fut[rows][imit]
            wl = None
            if c.late_lat_w != 1.0:
                wl = torch.where(torch.as_tensor(T8 >= 2.0, device=f.device), c.late_lat_w, 1.0).float()
            Ls["imit"] = self.dist(plan[imit], S.cam_x[rows][imit], f[..., 0], f[..., 1], f[..., 2], wl).mean()
            if self.hinge is not None:
                Ls["hinge"] = self.hinge(*rear(plan[imit], S.cam_x[rows][imit], self.W), rows[imit])
            if self.agent is not None:
                Ls["agent"] = self.agent(*rear(plan[imit], S.cam_x[rows][imit], self.W), rows[imit])
        if anchor.any():
            tx, ty, tpsi = rear(S.t_plan[rows][anchor], S.cam_x[rows][anchor], self.W)
            Ls["cons"] = self.dist(plan[anchor], S.cam_x[rows][anchor], tx, ty, tpsi).mean()
        e = ((out[:, self.di] - S.t_out[rows]) / self.tstd).pow(2) * self.dmask
        num = e[:, ~self.plan_cols].sum(1) + (~imit).float() * e[:, self.plan_cols].sum(1)
        Ls["distill"] = (num / e.shape[1]).mean()
        total = c.lam_i * Ls.get("imit", 0.0) + c.lam_c * Ls.get("cons", 0.0) + c.lam_d * Ls["distill"]
        if "hinge" in Ls:
            total = total + c.hinge_lam * Ls["hinge"]
        if "agent" in Ls:
            total = total + c.agent_lam * Ls["agent"]
        if c.act_lab:
            if c.act_lab == "plan":
                k, ok = plan_curv(plan.detach(), self.Wk), torch.ones(len(rows), dtype=torch.bool, device=out.device)
            else:
                k, ok = S.klab[rows], S.klab_ok[rows]
            v0 = S.v0[rows]
            w = (v0 / 3).clamp(0, 1) * ok
            z = (out[:, ACT_COL] - k * v0.clamp_min(1).pow(2)) / self.asd
            Ls["act"] = (w * F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)).sum() / w.sum().clamp_min(1.0)
            total = total + c.act_lam * Ls["act"]
        return total, Ls


def plan_curv(plan: torch.Tensor, Wk: torch.Tensor) -> torch.Tensor:
    """(B, 33, 15) plans -> mean curvature over the window of Wk (2, 33): heading change / arc length (floored at 1 m x window), as
    jevdrive.openpilot.model.curvature_window, in the plan's own sign (= the action's)."""
    s = torch.cat([plan.new_zeros(plan.shape[0], 1), torch.linalg.norm(plan[:, 1:, :2] - plan[:, :-1, :2], dim=-1).cumsum(1)], 1)
    sq, pq = s @ Wk.T, plan[..., 11] @ Wk.T
    return (pq[:, 1] - pq[:, 0]) / (sq[:, 1] - sq[:, 0]).clamp_min(ACT_WIN[1] - ACT_WIN[0])


def log_curv(tb: dict, kind: str) -> tuple:
    """Logged curvature labels in the action sign (+ = right) from the rear-axle poses (x fwd, y left, yaw left): `log` d psi / ds at
    t = ACT_T of a cubic spline through the 4 history poses (-1.5 .. 0 s) and the 8 future ones (0.5 .. 4 s); `logwin` the heading change
    over arc length between the logged poses at 0.5 and 1.5 s (floored at 1 m). -> (k (n,), ok (n,))."""
    from scipy.interpolate import CubicSpline
    fut = tb["fut"].astype(np.float64)
    ok = ~np.isnan(fut[:, :, :3]).any((1, 2))
    P = np.concatenate([tb["pose"].astype(np.float64), np.nan_to_num(fut)], 1)
    tt = np.r_[-1.5, -1.0, -0.5, 0.0, 0.5 * np.arange(1, 9)]
    psi = np.unwrap(P[..., 2], axis=1)
    s = np.concatenate([np.zeros((len(P), 1)), np.cumsum(np.linalg.norm(np.diff(P[..., :2], axis=1), axis=-1), 1)], 1)
    if kind == "log":
        k = CubicSpline(tt, psi, axis=1)(ACT_T, 1) / np.maximum(CubicSpline(tt, s, axis=1)(ACT_T, 1), 0.5)
    elif kind == "logwin":
        k = (psi[:, 6] - psi[:, 4]) / np.maximum(s[:, 6] - s[:, 4], ACT_WIN[1] - ACT_WIN[0])
    else:
        raise ValueError(kind)
    return (-k).astype(np.float32), ok & np.isfinite(k)


@torch.no_grad()
def dev_eval(model: PModel, S: Store, dev_rows: np.ndarray, W, bs=128) -> dict:
    """ADE of the 8 poses to the log (inputs on) and drift to shipped (inputs on / off), dev rows."""
    pi = torch.as_tensor(S.pi, device=S.front.device)
    acc = {"ade": [], "drift_on": [], "drift_off": []}
    for i in range(0, len(dev_rows), bs):
        r = torch.as_tensor(dev_rows[i:i + bs], device=S.front.device)
        side = S.side[r] if S.side is not None else (S.mem[r] if getattr(S, "mem", None) is not None else None)
        tx, ty, _ = rear(S.t_plan[r], S.cam_x[r], W)
        for on in (True, False):
            p = model(S.front[r], S.ego[r], S.tc[r], side, None, inputs_on=on, nv=None if S.nv is None else S.nv[r]).float()[:, pi].view(-1, 33, 15)
            x, y, _ = rear(p, S.cam_x[r], W)
            d = torch.hypot(x - tx, y - ty).mean(1)
            acc["drift_on" if on else "drift_off"].append(d)
            if on:
                ok = S.has_fut[r]
                acc["ade"].append(torch.hypot(x - S.fut[r][..., 0], y - S.fut[r][..., 1]).mean(1)[ok])
    return {k: float(torch.cat(v).mean()) for k, v in acc.items()}


def main(a):
    from jevdrive.run import Run
    ddp = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda")
    cfg = Cfg(arm=a.arm, seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data), split=a.split, frames=a.frames, host=a.host,
              warmup=a.warmup, eval_every=a.eval_every,
              hinge_lam=a.hinge_lam, hinge_margin=a.hinge_margin, hinge_labels=tuple(a.hinge_labels), hinge_footprint=tuple(a.hinge_footprint),
              hinge_replay=a.hinge_replay, hinge_front_turn_margin=a.hinge_front_turn_margin,
              b2d_split=a.b2d_split, b2d_mass=a.b2d_mass, anchor_b2d=not a.no_anchor_b2d,
              turn_bal=a.turn_bal, anchor_off_turn=a.anchor_off_turn, late_lat_w=a.late_lat_w,
              act_lab=a.act_lab, act_lam=a.act_lam, ego_lat_drop=a.ego_lat_drop, mem=a.mem, stop_gate=a.stop_gate,
              wod_split=a.wod_split, wod_mass=a.wod_mass, wod_slots=a.wod_slots, stop_gate_free=a.stop_gate_free,
              agent_lam=a.agent_lam, agent_margin=a.agent_margin, agent_side_margin=a.agent_side_margin, agent_labels=a.agent_labels,
              mem_e2e=a.mem_e2e, mem_init=a.mem_init, mem_lr=a.mem_lr)
    tag = a.tag or f"{a.arm}-s{a.seed}"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, rank])                   # the same row stream for every arm of one seed
    tz = [np.load(data_dir() / "runs" / "op_parity" / "cache" / d / "tab.npz") for d in cfg.data]
    tabs = dict(names=np.concatenate([z["names"] for z in tz]), log=np.concatenate([z["log"] for z in tz]),
                is_b2d=np.concatenate([np.full(len(z["names"]), d.startswith("b2d_")) for d, z in zip(cfg.data, tz)]),
                is_wod=np.concatenate([np.full(len(z["names"]), d.startswith("wod_")) for d, z in zip(cfg.data, tz)]))
    tr_rows, dv_rows, sp = split_rows(tabs, cfg.split, cfg.b2d_split, cfg.wod_split)
    if cfg.mem_e2e:
        assert not cfg.mem and a.tag and world == 1 and cfg.hinge_lam > 0 and not cfg.hinge_replay, "--mem-e2e: P2 + hinge, a --tag, one GPU, no --mem"
        cfg.mem = f"ge_{tag}"
    if cfg.mem:
        assert cfg.arm == "P2", "--mem extends arm P2"
        cfg.arm = f"P2+{cfg.mem}"
    S = Store(cfg.data, dev, need_side=arm_kw(cfg.arm)["side"], frames=cfg.frames, host=cfg.host, mem=None if cfg.mem_e2e else (cfg.mem or None),
              wod_slots=cfg.wod_slots)
    model = PModel(cfg.arm, act=bool(cfg.act_lab)).to(dev)
    if cfg.stop_gate > 0:                                           # the whole ego row is zeroed (present = 0 -> bias exactly 0) in training and dev eval
        S.ego = S.ego * (S.ego[:, 4:5] * 10.0 >= cfg.stop_gate).float()
    if cfg.act_lab in ("log", "logwin"):
        S.act_labels(cfg.act_lab)
    lrng = np.random.default_rng([cfg.seed, rank, 7])                       # ego_lat_drop: own stream, the row stream is unchanged
    mrng = np.random.default_rng([cfg.seed, rank, 11])                      # memory drop: own stream, the row stream is unchanged
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    hinge = None
    if cfg.hinge_lam > 0:
        if cfg.hinge_replay:
            from replay_hinge import ReplayHinge
            hinge = ReplayHinge([data_dir() / f for f in cfg.hinge_labels], S.tab, dev, cfg.hinge_margin, list(cfg.hinge_footprint),
                                cfg.hinge_front_turn_margin)
        else:
            from drivable_hinge import Hinge
            hinge = Hinge([data_dir() / f for f in cfg.hinge_labels], S.tab["names"], dev, cfg.hinge_margin, list(cfg.hinge_footprint))
        print(f"hinge lambda {cfg.hinge_lam}, margin {cfg.hinge_margin}: labels cover {hinge.coverage:.4f} of {S.n} rows", flush=True)
    agent = None
    if cfg.agent_lam > 0:
        from agent_hinge import AgentHinge
        agent = AgentHinge(data_dir() / cfg.agent_labels, S.tab["names"], dev, cfg.agent_margin,
                           None if cfg.agent_side_margin < 0 else cfg.agent_side_margin)
        print(f"agent hinge lambda {cfg.agent_lam}, margin {cfg.agent_margin} / side {agent.side_margin}: labels cover {agent.coverage:.4f} of {S.n} rows", flush=True)
    LS = Losses(model.net, cfg, tstd, S.di, S.pi, dev, hinge, agent)
    om = None
    if cfg.mem_e2e:                                                 # built after the model: the adapter's init draws are those of the frozen-bank arms
        assert cfg.mem_e2e in ("b", "x", "p") or cfg.mem_e2e.startswith("q"), f"unknown --mem-e2e kind {cfg.mem_e2e!r}"
        if cfg.mem_e2e.startswith("q"):                             # path-req: degraded fields of the logged future path (scripts/path_req.py)
            import path_req as GE
        else:
            import geo_e2e as GE
        om = S.mem = GE.attach(cfg, S, hinge, dev)                  # S.mem[rows] -> tokens (dev_eval reads it like a bank)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}] + ([{"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}] if new else []) +
                            ([{"params": om.params, "lr": cfg.mem_lr or cfg.lr_new, "base": cfg.mem_lr or cfg.lr_new}] if om else []), weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = proot("runs", tag)
    ctx = Run("op_parity", f"train-{tag}", seed=cfg.seed, config=asdict(cfg)) if rank == 0 else None
    run = ctx.__enter__() if ctx else None
    try:
        if run:
            for x in sp:
                run.use_split(x)
            run.info(f"{tag}: train {len(tr_rows)} dev {len(dv_rows)} rows, base {sum(p.numel() for p in base) / 1e6:.1f}M, "
                     f"adapter {sum(p.numel() for p in new) / 1e6:.2f}M, world {world}")
        t0, hist = time.time(), []
        nB = cfg.batch
        use_side = arm_kw(cfg.arm)["side"]
        use_mem = bool(cfg.mem)

        turn = np.zeros(S.n, bool)
        pw = None
        if cfg.turn_bal or cfg.anchor_off_turn:
            fut = np.nan_to_num(S.tb["fut"][:, :, 2].astype(np.float64))           # logged yaw of the 8 future poses (rear-axle frame); no log -> 0
            dy = np.abs(np.degrees(np.unwrap(fut, axis=1)[:, -1]))
            turn = dy > 20
        if cfg.b2d_mass > 0:                                                    # share of the draws from the b2d_* rows; turn balancing inside them
            b = S.is_b2d[tr_rows]
            assert b.any() and (~b).any(), "--b2d-mass needs b2d_* and other rows in the train split"
            w = np.where(b, cfg.b2d_mass / b.sum(), (1 - cfg.b2d_mass) / (~b).sum())
            if cfg.turn_bal:
                bi = np.digitize(dy[tr_rows][b], [5, 20, 45])
                nat = np.bincount(bi, minlength=4) / len(bi)
                tgt = np.asarray([float(x) for x in cfg.turn_bal.split(",")])
                wb = (tgt / np.maximum(nat, 1e-9))[bi]
                w[b] = cfg.b2d_mass * wb / wb.sum()
                if run:
                    run.info(f"B2D turn bins (<5, 5-20, 20-45, >45 deg): natural {np.round(nat, 3).tolist()}, target {tgt.tolist()}")
            pw = w / w.sum()
            if run:
                run.info(f"b2d mass {cfg.b2d_mass}: {int(b.sum())} B2D / {int((~b).sum())} other train rows")
        elif cfg.turn_bal:
            bi = np.digitize(dy[tr_rows], [5, 20, 45])
            nat = np.bincount(bi, minlength=4) / len(tr_rows)
            tgt = np.asarray([float(x) for x in cfg.turn_bal.split(",")])
            w = tgt / np.maximum(nat, 1e-9)
            pw = w[bi] / w[bi].sum()
            if run:
                run.info(f"turn-balanced sampling: natural mass {np.round(nat, 3).tolist()}, target {tgt.tolist()}, per-token weights {np.round(w / w[0], 3).tolist()} "
                         f"(relative to the < 5 deg bin); anchor off on turning tokens: {cfg.anchor_off_turn} ({turn[tr_rows].mean():.3f} of train rows)")

        gated = S.tb["ego"][:, 4] * 10.0 < cfg.stop_gate if (cfg.stop_gate > 0 and cfg.stop_gate_free) else None
        if gated is not None and run:
            run.info(f"stop gate {cfg.stop_gate} m/s, no anchors on gated rows: {int(gated[tr_rows].sum())} of {len(tr_rows)} train rows gated")
        if cfg.wod_mass > 0:                                                    # mixed-domain: an exact share of every batch from the wod_* rows
            assert pw is None, "--wod-mass does not combine with --b2d-mass / --turn-bal"
            w_tr, o_tr = tr_rows[S.is_wod[tr_rows]], tr_rows[~S.is_wod[tr_rows]]
            kw = int(round(nB * cfg.wod_mass))
            assert len(w_tr) and len(o_tr) and 0 < kw < nB, "--wod-mass needs wod_* and other rows in the train split"
            if run:
                run.info(f"wod mass {cfg.wod_mass}: {kw} WOD + {nB - kw} other rows per batch; {len(w_tr)} WOD / {len(o_tr)} other train rows; "
                         f"real slots per row: {torch.unique(S.nv).tolist() if S.nv is not None else 'uniform'}")

        def draw():                                                             # the rng order of the GPU-store loop (same row stream)
            if cfg.wod_mass > 0:
                r = np.concatenate([rng.choice(o_tr, nB - kw, replace=len(o_tr) < nB - kw), rng.choice(w_tr, kw, replace=len(w_tr) < kw)])
            else:
                r = rng.choice(tr_rows, nB, replace=len(tr_rows) < nB, p=pw)
            an = rng.random(nB) < cfg.d_frac
            if cfg.anchor_off_turn:
                an &= ~turn[r]
            if not cfg.anchor_b2d:
                an &= ~S.is_b2d[r]
            if gated is not None:
                an &= ~gated[r]
            sm = rng.random((nB, len(PA.SIDE_CAMS))) >= cfg.cam_drop if use_side else None
            if use_mem:
                sm = mrng.random((nB, 1)) >= MEM_DROP                           # True = memory present
            return r, an, sm

        def fetch(dr):
            r = dr[0]
            return dr, S.front[r], (S.side[r] if use_side else (S.mem[r] if (use_mem and om is None) else None))
        from concurrent.futures import ThreadPoolExecutor
        pre = ThreadPoolExecutor(2)
        nxt = pre.submit(fetch, draw())
        for step in range(cfg.steps):
            (rows_np, an_np, sm_np), front_b, side = nxt.result()
            if step + 1 < cfg.steps:
                nxt = pre.submit(fetch, draw())
            rows = torch.as_tensor(rows_np, device=dev)
            anchor = torch.as_tensor(an_np, device=dev)
            smask = torch.as_tensor(sm_np, device=dev) if (use_side or use_mem) else None
            frac = step / cfg.steps
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
            ego = S.ego[rows] * (~anchor)[:, None].float()                           # present = 0 on anchor rows -> the bias is exactly 0
            if cfg.ego_lat_drop > 0:
                dm = torch.as_tensor(lrng.random(nB) < cfg.ego_lat_drop, device=dev)
                ego[:, EGO_LAT] = ego[:, EGO_LAT] * (~dm)[:, None].float()
            if om is not None:
                side = om[rows]                                                      # tokens of this step's tokenizer: the losses reach its weights
            out = model(front_b, ego, S.tc[rows], side, smask, nv=None if S.nv is None else S.nv[rows])
            total, Ls = LS(out, S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if world > 1:
                for p in base + new:
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad)
                        p.grad /= world
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(base + new, 1.0)
            if om is not None:                                                       # clipped on its own: the policy's clip is that of the other arms
                tok_gn = float(torch.nn.utils.clip_grad_norm_(om.params, 1.0))
            scaler.step(opt)
            scaler.update()
            hist.append({k: float(v) for k, v in Ls.items()})
            if run and ((step + 1) % 25 == 0 or step + 1 == cfg.steps):
                m = {k: float(np.mean([h[k] for h in hist if k in h])) for k in hist[-1]}
                hist = []
                run.scalars({f"loss/{k}": v for k, v in m.items()}, step + 1)
                el = time.time() - t0
                run.scalars({"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if om is not None:
                    run.scalars({"geotok/grad_norm": tok_gn}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                             f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if run and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
                ev = dev_eval(model.eval(), S, dv_rows, LS.W)
                dvb = dv_rows[S.is_b2d[dv_rows]]
                if len(dvb) and len(dvb) < len(dv_rows):
                    ev |= {"b2d_" + k: v for k, v in dev_eval(model, S, dvb, LS.W).items()}
                dvw = dv_rows[S.is_wod[dv_rows]]
                if len(dvw) and len(dvw) < len(dv_rows):                    # mixed-domain: each domain's dev rows
                    ev |= {"wod_" + k: v for k, v in dev_eval(model, S, dvw, LS.W).items()}
                    ev |= {"nav_" + k: v for k, v in dev_eval(model, S, dv_rows[~S.is_wod[dv_rows]], LS.W).items()}
                model.train()
                run.scalars({f"dev/{k}": v for k, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
                run.summary.update({f"dev_{k}": v for k, v in ev.items()})
        if run:
            torch.save({"model": model.state(), "cfg": asdict(cfg)}, d / "ckpt-final.pt")
            if model.adapter is not None:
                torch.save(model.adapter.state_dict(), d / "adapter.pt")
            run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"))
            if om is not None:
                GE.finish(om, model, S, dv_rows, LS.W, rear, hinge, run, tag, d)
                run.summary.update(gpu_peak_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
    except BaseException as e:
        if ctx:
            ctx.__exit__(type(e), e, e.__traceback__)
            ctx = None
        raise
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
        if ddp:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data", nargs="+", default=["lb_navtrain", "lb_h1train"])
    ap.add_argument("--split", default="navsim/op-parity-pilot")
    ap.add_argument("--tag", default="")
    ap.add_argument("--frames", default="gimm", choices=["gimm", "warp", "keys"])
    ap.add_argument("--host", action="store_true", help="gather token rows per batch from the memory-mapped caches (full navtrain)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--hinge-lam", type=float, default=0.0, help="weight of the footprint drivable-area SDF hinge on the plan (0 = off)")
    ap.add_argument("--hinge-margin", type=float, default=0.3)
    ap.add_argument("--hinge-labels", nargs="+", default=list(Cfg.hinge_labels), help="hinge label file(s) under $DATA_DIR (the first that labels a row wins)")
    ap.add_argument("--hinge-replay", action="store_true", help="hinge on the devkit LQR replay of the plan (lib/replay_hinge.py)")
    ap.add_argument("--hinge-front-turn-margin", type=float, default=0.0, help="replay hinge: front-corner margin on turning rows (0 = off)")
    ap.add_argument("--hinge-footprint", nargs="+", default=list(Cfg.hinge_footprint), help="footprint of each --hinge-labels file: pacifica | mkz")
    ap.add_argument("--agent-lam", type=float, default=0.0, help="weight of the agent-box hinge on the plan (lib/agent_hinge.py; 0 = off)")
    ap.add_argument("--agent-margin", type=float, default=0.5)
    ap.add_argument("--agent-side-margin", type=float, default=-1.0, help="margin outside the ego's lateral corridor (< 0: = --agent-margin)")
    ap.add_argument("--agent-labels", default=Cfg.agent_labels, help="agent label file under $DATA_DIR")
    ap.add_argument("--no-anchor-b2d", action="store_true", help="no anchor rows on b2d_* rows (imitation only there)")
    ap.add_argument("--b2d-split", default=Cfg.b2d_split)
    ap.add_argument("--b2d-mass", type=float, default=0.0, help="share of imitation draws from the b2d_* rows (turn balancing then acts inside them)")
    ap.add_argument("--turn-bal", default="", help="target sampling mass per |heading change| bin <5,5-20,20-45,>45 deg, e.g. 0.35,0.15,0.25,0.25")
    ap.add_argument("--anchor-off-turn", action="store_true", help="no anchor rows on tokens with logged |heading change| > 20 deg")
    ap.add_argument("--late-lat-w", type=float, default=1.0, help="weight on y / yaw imitation terms of poses at >= 2 s")
    ap.add_argument("--act-lab", default="", choices=["", "plan", "log", "logwin"], help="train action[0] on this label (joint action arms)")
    ap.add_argument("--act-lam", type=float, default=3.0)
    ap.add_argument("--ego-lat-drop", type=float, default=0.0, help="fraction of rows with vy / ay of the ego input zeroed")
    ap.add_argument("--stop-gate", type=float, default=0.0, help="adapter off on rows whose fed speed (ego vx) is below this, m/s (wod-launch); 0 = off")
    ap.add_argument("--stop-gate-free", action="store_true", help="with --stop-gate: no anchor rows on the gated rows")
    ap.add_argument("--wod-split", default=Cfg.wod_split, help="mixed-domain: sequence split of the wod_* dirs when --split is a NAVSIM split")
    ap.add_argument("--wod-mass", type=float, default=0.0, help="mixed-domain: exact share of every batch drawn from the wod_* rows (0 = natural mix)")
    ap.add_argument("--wod-slots", type=int, default=0, choices=[0, 8], help="mixed-domain: 8 = oldest WOD slot zeroed (teacher8.npz); 0 = all 9")
    ap.add_argument("--mem", default="", choices=["", *MEM_KINDS], help="32-token memory for arm P2 (representation fix / turn-oracle; runs/op_parity/mem)")
    ap.add_argument("--mem-e2e", default="", help="geo-e2e: jointly trained tokenizer over true SDF + agents (b), the same shuffled across logs (x), "
                    "the logged-path field (p); path-req: q<kind>, a degraded field of the logged future (scripts/path_req.py). Privileged, oracle probes only")
    ap.add_argument("--mem-init", default="", help="geo-e2e: tokenizer state dict to start from (geo_oracle.py tok --weights)")
    ap.add_argument("--mem-lr", type=float, default=0.0, help="geo-e2e / path-req: learning rate of the tokenizer group (0 = the adapter's lr_new)")
    main(ap.parse_args())
