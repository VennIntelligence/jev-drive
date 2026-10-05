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
navtrain only). Multi-GPU: torchrun -> DDP-style gradient all-reduce, each rank its own row stream; single GPU without torchrun.

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
    split: str = "navsim/op-parity-pilot"


def proot(*p) -> _pl.Path:
    d = data_dir() / "runs" / "op_parity" / _pl.Path(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- model
class PModel(nn.Module):
    def __init__(self, arm: str, dtype=torch.float16, pol: bool = True):
        super().__init__()
        self.arm = arm
        tr = L.pol_weights() if pol else []
        self.net = A.load("cinque", dtype, trainable=tr)
        k = ARMS.get(arm, dict(ego=False, side=False))
        self.adapter = PA.ParityAdapter(use_ego=k["ego"], use_side=k["side"]) if (k["ego"] or k["side"]) else None

    def forward(self, front, ego, tc, side=None, side_mask=None, inputs_on=True):
        """front (B, 8, 32, 512) cached hidden tokens -> outputs (B, n). inputs_on False / no adapter: the bias is not added (P1, anchor rows)."""
        B = front.shape[0]
        H = torch.cat([torch.zeros_like(front[:, :1]), front], 1).to(self.net.dtype)
        valid = torch.ones(B, A.CONTEXT, dtype=torch.bool, device=H.device)
        valid[:, 0] = False
        if self.adapter is not None and inputs_on:
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


# ---------------------------------------------------------------- data on the GPU
class Store:
    """The pp_prep caches of several data dirs, concatenated and moved to the device once (fp16 tokens)."""

    def __init__(self, datas, dev, need_side=True, rows=None):
        cr = data_dir() / "runs" / "op_parity" / "cache"
        tabs = [dict(np.load(cr / d / "tab.npz")) for d in datas]
        self.tab = {k: np.concatenate([t[k] for t in tabs]) for k in tabs[0]}
        n = len(self.tab["names"])
        self.rows = np.arange(n) if rows is None else rows
        sel = self.rows
        cat = lambda f: np.concatenate([np.load(cr / d / f, mmap_mode="r") for d in datas]) if len(datas) > 1 else np.load(cr / datas[0] / f, mmap_mode="r")  # noqa: E731
        t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        self.front = t(cat("front.npy")[sel])
        self.side = t(cat("side.npy")[sel]) if need_side else None
        tz = [dict(np.load(cr / d / "teacher.npz")) for d in datas]
        self.t_out = t(np.concatenate([z["out"] for z in tz])[sel])
        self.t_plan = t(np.concatenate([z["plan"] for z in tz])[sel])
        self.di, self.pi = tz[0]["di"], tz[0]["pi"]
        tb = {k: v[sel] for k, v in self.tab.items()}
        self.tb = tb
        self.ego = t(tb["ego"])
        self.fut = t(np.nan_to_num(tb["fut"]).astype(np.float32))
        self.has_fut = t(~np.isnan(tb["fut"][:, 0, 0]))
        self.cam_x = t(tb["cam"][:, 0].astype(np.float32))
        self.tc = t(np.where(tb["lht"][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32))
        self.n = len(sel)


def split_rows(tab, split_ref) -> tuple:
    from jevdrive.data import splits
    tr, dv = splits.load(f"{split_ref}-train"), splits.load(f"{split_ref}-dev")
    toks = tab["names"]
    return np.flatnonzero(tr.mask(toks)), np.flatnonzero(dv.mask(toks)), (tr, dv)


# ---------------------------------------------------------------- losses
class Losses:
    def __init__(self, net, cfg: Cfg, tstd: torch.Tensor, di, pi, dev):
        self.cfg = cfg
        self.W = torch.as_tensor(R2.t_weights(T8), device=dev)
        self.s = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in (SIG_X, SIG_Y, SIG_PSI)]
        self.di = torch.as_tensor(di, device=dev)
        self.pi = torch.as_tensor(pi, device=dev)
        self.plan_cols = torch.as_tensor(np.isin(di, pi), device=dev)
        self.tstd = tstd

    def dist(self, plan, cam_x, tx, ty, tpsi):
        x, y, psi = rear(plan, cam_x, self.W)
        h = lambda z: F.huber_loss(z, torch.zeros_like(z), reduction="none", delta=1.0)  # noqa: E731
        return (h((x - tx) / self.s[0]) + h((y - ty) / self.s[1]) + h((psi - tpsi) / self.s[2])).mean(1)

    def __call__(self, out, S: Store, rows, anchor):
        c, out = self.cfg, out.float()
        plan = out[:, self.pi].view(-1, 33, 15)
        imit = ~anchor & S.has_fut[rows]
        Ls = {}
        if imit.any():
            f = S.fut[rows][imit]
            Ls["imit"] = self.dist(plan[imit], S.cam_x[rows][imit], f[..., 0], f[..., 1], f[..., 2]).mean()
        if anchor.any():
            tx, ty, tpsi = rear(S.t_plan[rows][anchor], S.cam_x[rows][anchor], self.W)
            Ls["cons"] = self.dist(plan[anchor], S.cam_x[rows][anchor], tx, ty, tpsi).mean()
        e = ((out[:, self.di] - S.t_out[rows]) / self.tstd).pow(2)
        num = e[:, ~self.plan_cols].sum(1) + (~imit).float() * e[:, self.plan_cols].sum(1)
        Ls["distill"] = (num / e.shape[1]).mean()
        total = c.lam_i * Ls.get("imit", 0.0) + c.lam_c * Ls.get("cons", 0.0) + c.lam_d * Ls["distill"]
        return total, Ls


@torch.no_grad()
def dev_eval(model: PModel, S: Store, dev_rows: np.ndarray, W, bs=512) -> dict:
    """ADE of the 8 poses to the log (inputs on) and drift to shipped (inputs on / off), dev rows."""
    pi = torch.as_tensor(S.pi, device=S.front.device)
    acc = {"ade": [], "drift_on": [], "drift_off": []}
    for i in range(0, len(dev_rows), bs):
        r = torch.as_tensor(dev_rows[i:i + bs], device=S.front.device)
        side = S.side[r] if S.side is not None else None
        tx, ty, _ = rear(S.t_plan[r], S.cam_x[r], W)
        for on in (True, False):
            p = model(S.front[r], S.ego[r], S.tc[r], side, None, inputs_on=on).float()[:, pi].view(-1, 33, 15)
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
    cfg = Cfg(arm=a.arm, seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data), split=a.split)
    tag = a.tag or f"{a.arm}-s{a.seed}"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, rank])                   # the same row stream for every arm of one seed
    tabs = np.concatenate([np.load(data_dir() / "runs" / "op_parity" / "cache" / d / "tab.npz")["names"] for d in cfg.data])
    tr_rows, dv_rows, sp = split_rows({"names": tabs}, cfg.split)
    S = Store(cfg.data, dev, need_side=ARMS[cfg.arm]["side"])
    model = PModel(cfg.arm).to(dev)
    base, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    LS = Losses(model.net, cfg, tstd, S.di, S.pi, dev)
    opt = torch.optim.AdamW([{"params": base, "lr": cfg.lr, "base": cfg.lr}] + ([{"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}] if new else []),
                            weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    d = proot("runs", tag)
    ctx = Run("op_parity", f"train-{tag}", seed=cfg.seed, config=asdict(cfg)) if rank == 0 else None
    run = ctx.__enter__() if ctx else None
    try:
        if run:
            run.use_split(sp[0]), run.use_split(sp[1])
            run.info(f"{tag}: train {len(tr_rows)} dev {len(dv_rows)} rows, base {sum(p.numel() for p in base) / 1e6:.1f}M, "
                     f"adapter {sum(p.numel() for p in new) / 1e6:.2f}M, world {world}")
        t0, hist = time.time(), []
        nB = cfg.batch
        for step in range(cfg.steps):
            rows = torch.as_tensor(rng.choice(tr_rows, nB, replace=len(tr_rows) < nB), device=dev)
            anchor = torch.as_tensor(rng.random(nB) < cfg.d_frac, device=dev)
            side, smask = None, None
            if ARMS[cfg.arm]["side"]:
                side = S.side[rows]
                smask = torch.as_tensor(rng.random((nB, len(PA.SIDE_CAMS))) >= cfg.cam_drop, device=dev)
            frac = step / cfg.steps
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
            ego = S.ego[rows] * (~anchor)[:, None].float()                           # present = 0 on anchor rows -> the bias is exactly 0
            out = model(S.front[rows], ego, S.tc[rows], side, smask)
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
            scaler.step(opt)
            scaler.update()
            hist.append({k: float(v) for k, v in Ls.items()})
            if run and ((step + 1) % 25 == 0 or step + 1 == cfg.steps):
                m = {k: float(np.mean([h[k] for h in hist if k in h])) for k in hist[-1]}
                hist = []
                run.scalars({f"loss/{k}": v for k, v in m.items()}, step + 1)
                el = time.time() - t0
                run.scalars({"throughput/steps_per_s": (step + 1) / el, "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) +
                             f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if run and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
                ev = dev_eval(model.eval(), S, dv_rows, LS.W)
                model.train()
                run.scalars({f"dev/{k}": v for k, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
                run.summary.update({f"dev_{k}": v for k, v in ev.items()})
        if run:
            torch.save({"model": model.state(), "cfg": asdict(cfg)}, d / "ckpt-final.pt")
            if model.adapter is not None:
                torch.save(model.adapter.state_dict(), d / "adapter.pt")
            run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"))
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
    main(ap.parse_args())
