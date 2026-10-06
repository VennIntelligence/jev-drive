"""op_parity vision unfreeze (plans/2026-10-06-unfreeze-prereg.md): how much of P2's 3.55 EPDMS gap to WA-JEPA closes when Cinque's frozen
front encoder may adapt. Recipe = the full run's P2 (protocol W, ego / pose / command adapter, anchor rows distilled to shipped on the same
navtrain frames); only the vision part differs per variant:

  F    vision frozen (= P2-frozen): input the cached view_39 tokens (cache/<shard>@warp/front.npy)
  U1   stage 4 (3 ConvNeXt blocks, 2048 ch at 4 x 8) + head (LN, fc 2048 -> 512) trainable, 110 M weights, lr_vis 1e-5: input the cached
       frozen stage 1-3 + downsample output `conv2d_36` (cache/<shard>@warp/x4.npy, (8, 2048, 4, 8) fp16 per token, 1 MB)
  U1L  LoRA rank 16 on the six stage-4 MLP matmuls (3.0 M weights, lr 3e-4), same input as U1
  U2   the whole vision encoder trainable (349 M, lr_vis 5e-6): pixels rendered online on CPU (4 CAM_F0 keys + 6 warped lattice frames,
       pp_prep's exact W path) and encoded every step; gradient through the t0 image pair, the 7 older pairs run the same (current) encoder
       without autograd (op_adapt C')
  V    vision frozen, frames from a virtual camera 1.40 m above the road (decision 104; scripts/op_lb.py _vcam_job: keys re-rendered for
       the lower camera, warp on that road plane): cache/<shard>@vh140/{front.npy, teacher.npz}; geometry fixed instead of the encoder adapting

  prep   --data navtrain_full.s0of12 --what x4|vh140          caches above (CPU render pool + one GPU)
  train  --var U1 --seed 0 --data <shards> --steps --batch     world size from torchrun or PP_WORLD / PP_RANK / PP_RDV (one pool job per rank,
                                                                 file rendezvous); the global batch is fixed, each rank draws batch / world rows
  plans  --frames warp|vh140 --models P0 F-pilot-s0 ...         navtest plans from pixels (every variant through its own encoder), op_lb format
                                                                 -> runs/op_lb/lb_navtest/plans/<frames>@cinque_PP<tag>.npz; scored by pp_eval.py score
Checkpoints keep pp_train's format (net = every trained initializer, LoRA merged; parity = adapter; arm P2), so pp_eval / pp_hugsim load them.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, os, time  # noqa: E401,E402
from collections import deque  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402

import numpy as np  # noqa: E402

import parity_adapter as PA  # noqa: E402
from jevdrive import cache  # noqa: E402
from jevdrive.common import data_dir, n_cpus  # noqa: E402

X4 = "conv2d_36"                              # stage-4 downsample output (2048, 4, 8): the U1 / U1L cache point
X4_SHAPE = (2048, 4, 8)
VH = 1.40                                     # V: virtual camera height above the road (m)
VARS = {"F": dict(vis="frozen", inp="h", frames="warp", lr_vis=0.0),
        "U1": dict(vis="s4", inp="x4", frames="warp", lr_vis=1e-5),
        "U1L": dict(vis="lora", inp="x4", frames="warp", lr_vis=3e-4, rank=16),
        "U2": dict(vis="all", inp="px", frames="warp", lr_vis=5e-6),
        "V": dict(vis="frozen", inp="h", frames="vh140", lr_vis=0.0)}
CR = data_dir() / "runs" / "op_parity" / "cache"


# ---------------------------------------------------------------- frames (CPU workers): pp_prep's W path, or the virtual camera
def _slots(kf, sf):
    """keys (4, ...), lattice frames (6, ...) -> prev, cur (8, 2, 6, 128, 256): the 8 valid policy slots exactly as pp_prep (steps 2..30)."""
    import op_lb as OL
    import pp_prep as P
    _, src = OL._steps(0.0, False)
    img = lambda s: (kf[src[s][1]] if src[s][0] == "k" else sf[src[s][1]]) if s >= 0 else np.zeros(P.FRAME, np.uint8)  # noqa: E731
    return np.stack([img(s - 4) for s in P.STEPS]), np.stack([img(s) for s in P.STEPS])


def job_w_full(args):
    """navtrain_full token (index entry, pose, vel, cam, syn_t) -> prev, cur under protocol W (pp_prep._full_job)."""
    import pp_prep as P
    return _slots(*P._full_job(args))


def job_w_keys(args):
    """lb_navtest token (keys from the op_lb frames cache, pose, vel, cam, syn_t) -> prev, cur under protocol W (pp_prep._warp_job)."""
    import pp_prep as P
    return _slots(args[0], P._warp_job(args))


def job_vcam(args):
    """index entry -> prev, cur from the virtual camera at VH m (scripts/op_lb.py _vcam_job: keys + warp on that road plane)."""
    import op_lb as OL
    kf, sf, _ = OL._vcam_job((args[0], VH, 0.0))
    return _slots(kf, sf)


class Frames:
    """Row -> (prev, cur) for one data set and frame protocol, rendered by a CPU process pool, prefetched in submission order."""

    def __init__(self, datas, frames, workers):
        import op_lb as OL
        import pp_prep as P
        from jevdrive import navsim_zs as Z
        self.items, self.fn = [], None
        for d in datas:
            if d.startswith("navtrain_full"):
                mt, _, ents = P.full_meta(d)
                if frames == "warp":
                    self.fn = job_w_full
                    self.items += [(ents[i], mt["pose"][i], mt["vel"][i], mt["cam"][i], np.asarray(mt["syn_t"])) for i in range(len(ents))]
                else:
                    self.fn = job_vcam
                    self.items += [(e,) for e in ents]
            else:                                                     # an op_lb test dir (lb_navtest, lb_navhard)
                mt = OL.meta(d)
                if frames == "warp":
                    keys = OL.Keys(d)
                    self.fn, self.keys = job_w_keys, keys
                    self.items += [(i, mt["pose"][i], mt["vel"][i], mt["cam"][i], np.asarray(mt["syn_t"])) for i in range(len(mt["names"]))]
                else:
                    idx = Z.load_index(mt["split"])
                    self.fn = job_vcam
                    self.items += [(idx[k],) for k in mt["index"]]
                    del idx
        self.ex = ProcessPoolExecutor(workers)

    def _arg(self, i):
        it = self.items[i]
        return (np.asarray(self.keys[int(it[0])]),) + it[1:] if self.fn is job_w_keys else it

    def submit(self, rows):
        return [self.ex.submit(self.fn, self._arg(int(i))) for i in rows]

    @staticmethod
    def gather(futs):
        pc = [f.result() for f in futs]
        return np.stack([p for p, _ in pc]), np.stack([c for _, c in pc])

    def close(self):
        self.ex.shutdown(cancel_futures=True)


# ---------------------------------------------------------------- model
def vision_names(kind: str) -> list[str]:
    from jevdrive import op_adapt as A
    if kind == "s4":
        return [w for w in A.stage4_weights() if "downsample" not in w]
    if kind == "all":
        import onnx
        g = onnx.load(str(A.MODELS_DIR / A.FILES["cinque"]), load_external_data=False).graph
        return A.node_weights(A.MODELS_DIR / A.FILES["cinque"], g.node[0].output[0], "view_39")
    return []


def _torch_model():
    import torch
    import torch.nn as nn
    from jevdrive import op_adapt as A
    from jevdrive.op_torch import _key
    from experiments.op_adapt_l.lib import op_adapt_l as L
    import pp_train as T

    class UModel(nn.Module):
        """P2 (ego / pose / command adapter) with the vision part of variant `var`; inputs per VARS[var]['inp']."""

        def __init__(self, var: str, dtype=torch.float16):
            super().__init__()
            k = VARS[var]
            self.var, self.arm, self.inp = var, "P2", k["inp"]
            self.vis = vision_names(k["vis"])
            lora = {w: k["rank"] for w in A.stage4_matmuls()} if k["vis"] == "lora" else None
            self.net = A.load("cinque", dtype, trainable=L.pol_weights() + self.vis, lora=lora)
            self.adapter = PA.ParityAdapter(use_ego=True, use_side=False)
            self.vkeys = {_key(w) for w in self.vis}

        def encode(self, prev, cur, chunk=128, ckpt=0):
            """(n, 2, 6, 128, 256) uint8 pairs -> (n, 32, 512), in chunks; ckpt > 0: activation checkpointing per chunk of that size
            (only each chunk's output is kept; the backward recomputes it)."""
            from torch.utils.checkpoint import checkpoint
            f = lambda p, c: self.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE)  # noqa: E731
            k = ckpt or chunk
            return torch.cat([checkpoint(f, prev[i:i + k], cur[i:i + k], use_reentrant=False) if ckpt else f(prev[i:i + k], cur[i:i + k])
                              for i in range(0, len(cur), k)])

        def tokens(self, x):
            """The variant's input -> (B, 8, 32, 512) hidden tokens of the 8 valid slots."""
            if self.inp == "h":
                return x
            if self.inp == "x4":
                B = x.shape[0]
                h = self.net.run_batched({X4: x.reshape(B * 8, 1, *X4_SHAPE).to(self.net.dtype)}, ["view_39"])["view_39"]
                return h.reshape(B, 8, *A.H_SHAPE)
            prev, cur = x                                                         # px: (B, 8, 2, 6, 128, 256) uint8
            B = cur.shape[0]
            with torch.no_grad():
                hp = self.encode(prev[:, :7].reshape(-1, *prev.shape[2:]), cur[:, :7].reshape(-1, *cur.shape[2:])).reshape(B, 7, *A.H_SHAPE)
            return torch.cat([hp, self.encode(prev[:, 7], cur[:, 7], ckpt=8)[:, None]], 1)

        def forward(self, x, ego, tc, inputs_on=True):
            front = self.tokens(x)
            B, n = front.shape[:2]
            H = torch.cat([front.new_zeros(B, A.CONTEXT - n, *front.shape[2:]), front], 1).to(self.net.dtype)
            valid = torch.zeros(B, A.CONTEXT, dtype=torch.bool, device=H.device)
            valid[:, A.CONTEXT - n:] = True
            if inputs_on:
                H = self.adapter.apply(H, ego)
            H = H * valid[:, :, None, None].to(H.dtype)
            return self.net.run_batched(A.policy_feeds(self.net, H, T.AT, tc.to(self.net.dtype)), ["outputs"])["outputs"].reshape(B, -1)

        def groups(self):
            pol = [p for k, p in self.net.params.items() if p.requires_grad and k not in self.vkeys]
            vis = [p for k, p in self.net.params.items() if p.requires_grad and k in self.vkeys] + list(self.net.lora.parameters())
            return pol, vis, list(self.adapter.parameters())

        @torch.no_grad()
        def state(self) -> dict:
            """pp_train.PModel format: every trained initializer (LoRA merged into its weight), the adapter, arm P2."""
            net = {k: p.detach().float().cpu() for k, p in self.net.params.items() if p.requires_grad}
            for w, s in self.net.lora_scale.items():
                k = _key(w)
                net[k] = (self.net.params[k].float() + self.net.lora[k + "__A"] @ self.net.lora[k + "__B"] * s).cpu()
            return {"net": net, "adapter": None, "parity": self.adapter.state_dict(), "arm": "P2", "variant": self.var}

    return UModel


# ---------------------------------------------------------------- prep: x4 (U1 / U1L) and vh140 (V) caches
def cmd_prep(a):
    import torch
    import pp_prep as P
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    from jevdrive.data import splits
    mt, _, _ = P.full_meta(a.data)
    names = mt["names"][: a.limit] if a.limit else mt["names"]
    N = len(names)
    tag = a.data + (f"-first{a.limit}" if a.limit else "")
    out = CR / f"{tag}@{'warp' if a.what == 'x4' else a.what}"
    out.mkdir(parents=True, exist_ok=True)
    with Run("op_parity", f"uprep-{a.what}-{tag}", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        dev = torch.device("cuda")
        net = A.load("cinque", torch.float16).to(dev).eval()
        fr = Frames([a.data], "warp" if a.what == "x4" else "vh140", a.workers or max(1, n_cpus() - 2))
        want = X4 if a.what == "x4" else "view_39"
        shape = X4_SHAPE if a.what == "x4" else A.H_SHAPE
        k = cache.key(params=dict(data=a.data, n=N, what=a.what, vh=VH, names=cache.key(params=dict(n=names))), code=[_slots, job_vcam])

        def make():
            o = np.zeros((N, 8) + shape, np.float16)
            chunks = [np.arange(i, min(i + 16, N)) for i in range(0, N, 16)]
            q, t0 = deque(), time.time()
            for c in chunks[:6]:
                q.append((c, fr.submit(c)))
            nxt = 6
            for _ in run.tqdm(range(len(chunks)), desc=a.what):
                rows, futs = q.popleft()
                if nxt < len(chunks):
                    q.append((chunks[nxt], fr.submit(chunks[nxt]))); nxt += 1   # noqa: E702
                prev, cur = Frames.gather(futs)
                p, c = (torch.from_numpy(x.reshape(-1, *x.shape[2:])).to(dev) for x in (prev, cur))
                with torch.no_grad():
                    h = net.run_batched(A.vision_feeds(p, c), [want])[want]
                o[rows] = h.reshape(len(rows), 8, *shape).cpu().numpy().astype(np.float16)
            run.summary["tokens_per_s"] = N / (time.time() - t0)
            return o
        x = cache.cached(out / ("x4.npy" if a.what == "x4" else "front.npy"), k, make, force=a.force)
        fr.close()
        ref = CR / f"{tag}@warp" / "front.npy"
        if a.what == "x4" and ref.exists():                      # equivalence: frozen stage 4 on x4 = the W front cache
            r = np.load(ref, mmap_mode="r")
            with torch.no_grad():
                xs = torch.from_numpy(np.ascontiguousarray(x[:64])).to(dev)
                h = net.run_batched({X4: xs.reshape(-1, 1, *X4_SHAPE)}, ["view_39"])["view_39"].reshape(len(xs), 8, *A.H_SHAPE).float().cpu().numpy()
            d = np.abs(h - r[:64].astype(np.float32))
            run.summary |= {"x4_vs_front_max_abs": float(d.max()), "x4_vs_front_mean_abs": float(d.mean()), "front_rms": float(np.sqrt((r[:64].astype(np.float32) ** 2).mean()))}
            run.info(f"x4 -> view_39 vs W front cache: max {d.max():.4f}, mean {d.mean():.5f}")
        if a.what == "vh140":                                    # anchor teacher = shipped Cinque on the V frames
            tab = np.load(CR / a.data / "tab.npz")

            def make_teacher():
                di, pi = A.distill_index(net.slices), A.plan_index(net.slices)
                to, tp = np.zeros((N, len(di)), np.float32), np.zeros((N, 33, 15), np.float32)
                tc_all = np.where(tab["lht"][:N, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
                with torch.no_grad():
                    for i in range(0, N, 256):
                        r = slice(i, min(i + 256, N))
                        H = torch.from_numpy(np.ascontiguousarray(x[r])).to(dev)
                        H = torch.cat([torch.zeros_like(H[:, :1]), H], 1)
                        valid = torch.ones(H.shape[:2], dtype=torch.bool, device=dev)
                        valid[:, 0] = False
                        o = A._policy(net, H, (0.275, 0.525), torch.from_numpy(tc_all[r]).to(dev), valid)["outputs"].float()
                        to[r], tp[r] = o[:, di].cpu().numpy(), o[:, pi].cpu().numpy().reshape(-1, 33, 15)
                return dict(out=to, plan=tp, di=di, pi=pi)
            cache.cached(out / "teacher.npz", cache.key(params=dict(k=k), inputs=[out / "front.npy"]), make_teacher, force=a.force)


# ---------------------------------------------------------------- train
@dataclass
class UCfg:
    var: str
    seed: int = 0
    steps: int = 3000
    batch: int = 64                       # global batch (all ranks)
    d_frac: float = 0.25
    lam_i: float = 1.0
    lam_c: float = 3.0
    lam_d: float = 30.0
    lr: float = 3e-5
    lr_new: float = 3e-4
    lr_vis: float = 0.0
    wd: float = 0.01
    warmup: int = 100
    eval_every: int = 1000
    data: tuple = ("navtrain_full.s0of12", "navtrain_full.s1of12")
    split: str = "navsim/op-parity-full"
    workers: int = 0                      # U2 render workers per rank (0: n_cpus() - 2)


def _dist():
    """(rank, world, device): torchrun, or one pool job per rank (PP_WORLD / PP_RANK, file rendezvous PP_RDV), else a single GPU."""
    import datetime
    import torch
    import torch.distributed as dist
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return dist.get_rank(), dist.get_world_size()
    world = int(os.environ.get("PP_WORLD", "1"))
    if world > 1:
        dist.init_process_group("nccl", init_method=f"file://{os.environ['PP_RDV']}", rank=int(os.environ["PP_RANK"]), world_size=world,
                                timeout=datetime.timedelta(hours=8))
        return int(os.environ["PP_RANK"]), world
    return 0, 1


def cmd_train(a):
    import torch
    import torch.distributed as dist
    import pp_train as T
    from jevdrive.run import Run
    rank, world = _dist()
    dev = torch.device("cuda")
    k = VARS[a.var]
    cfg = UCfg(var=a.var, seed=a.seed, steps=a.steps, batch=a.batch, data=tuple(a.data), warmup=a.warmup, eval_every=a.eval_every,
               lr_vis=a.lr_vis if a.lr_vis is not None else k["lr_vis"], workers=a.workers)
    tag = a.tag or f"{a.var}-s{a.seed}"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, rank])
    nb = cfg.batch // world
    assert nb * world == cfg.batch, "global batch must divide by the world size"
    S = T.Store(list(cfg.data), dev, need_side=False, frames=k["frames"], host=True)
    if k["inp"] == "x4":
        S.front = T.Tokens([CR / f"{d}@warp" / "x4.npy" for d in cfg.data], dev, host=True)
    tr_rows, dv_rows, sp = T.split_rows({"names": S.tab["names"]}, cfg.split)
    fr = Frames(list(cfg.data), "warp", cfg.workers or max(1, n_cpus() - 2)) if k["inp"] == "px" else None
    model = _torch_model()(cfg.var).to(dev)
    pol, vis, new = model.groups()
    tstd = S.t_out[torch.as_tensor(tr_rows, device=dev)].float().std(0).clamp_min(1e-3)
    LS = T.Losses(model.net, T.Cfg(arm="P2", lam_i=cfg.lam_i, lam_c=cfg.lam_c, lam_d=cfg.lam_d), tstd, S.di, S.pi, dev)
    groups = [{"params": pol, "lr": cfg.lr, "base": cfg.lr}, {"params": new, "lr": cfg.lr_new, "base": cfg.lr_new}]
    if vis:
        groups.append({"params": vis, "lr": cfg.lr_vis, "base": cfg.lr_vis})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler()
    allp = pol + vis + new
    d = T.proot("runs", tag)
    ctx = Run("op_parity", f"utrain-{tag}", seed=cfg.seed, config=asdict(cfg) | {"world": world}) if rank == 0 else None
    run = ctx.__enter__() if ctx else None

    def draw():
        return rng.choice(tr_rows, nb, replace=len(tr_rows) < nb), rng.random(nb) < cfg.d_frac

    def submit(dr):                                              # input of a drawn batch: a future (px) or a host gather
        return (dr, fr.submit(dr[0])) if fr else (dr, S.front[dr[0]])

    def inp(x):
        if fr is None:
            return x
        p, c = Frames.gather(x)
        return tuple(torch.from_numpy(v).to(dev, non_blocking=True) for v in (p, c))

    def dev_eval():
        import pp_train as T_
        pi = torch.as_tensor(S.pi, device=dev)
        acc = {"ade": [], "drift_on": [], "drift_off": []}
        model.eval()
        with torch.no_grad():
            for i in range(0, len(dv_rows), 32):
                r = dv_rows[i:i + 32]
                x = inp(fr.submit(r)) if fr else S.front[r]
                rt = torch.as_tensor(r, device=dev)
                tx, ty, _ = T_.rear(S.t_plan[rt], S.cam_x[rt], LS.W)
                for on in (True, False):
                    p = model(x, S.ego[rt], S.tc[rt], inputs_on=on).float()[:, pi].view(-1, 33, 15)
                    px_, py_, _ = T_.rear(p, S.cam_x[rt], LS.W)
                    acc["drift_on" if on else "drift_off"].append(torch.hypot(px_ - tx, py_ - ty).mean(1))
                    if on:
                        ok = S.has_fut[rt]
                        acc["ade"].append(torch.hypot(px_ - S.fut[rt][..., 0], py_ - S.fut[rt][..., 1]).mean(1)[ok])
        model.train()
        return {k_: float(torch.cat(v).mean()) for k_, v in acc.items()}

    try:
        if run:
            run.use_split(sp[0]), run.use_split(sp[1])
            run.info(f"{tag} ({cfg.var}): train {len(tr_rows)} dev {len(dv_rows)} rows, plan {sum(p.numel() for p in pol) / 1e6:.1f}M, "
                     f"vision {sum(p.numel() for p in vis) / 1e6:.1f}M, adapter {sum(p.numel() for p in new) / 1e6:.2f}M, world {world}, batch {nb}/rank")
        from concurrent.futures import ThreadPoolExecutor
        pre = ThreadPoolExecutor(2) if fr is None else None
        q = deque()
        ahead = 3 if fr else 1
        for _ in range(ahead):
            dr = draw()
            q.append(pre.submit(submit, dr) if pre else submit(dr))
        t0, hist = time.time(), []
        for step in range(cfg.steps):
            item = q.popleft()
            (rows_np, an_np), x = item.result() if pre else item
            if step + ahead < cfg.steps:
                dr = draw()
                q.append(pre.submit(submit, dr) if pre else submit(dr))
            x = inp(x)
            rows = torch.as_tensor(rows_np, device=dev)
            anchor = torch.as_tensor(an_np, device=dev)
            frac = step / cfg.steps
            for g in opt.param_groups:
                g["lr"] = g["base"] * min(1.0, (step + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * frac))
            ego = S.ego[rows] * (~anchor)[:, None].float()
            out = model(x, ego, S.tc[rows])
            total, Ls = LS(out, S, rows, anchor)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {k_: float(v) for k_, v in Ls.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if world > 1:                                                   # one coalesced all-reduce of every gradient
                gs = [p.grad if p.grad is not None else torch.zeros_like(p) for p in allp]
                flat = torch.cat([g.reshape(-1).float() for g in gs])
                dist.all_reduce(flat)
                flat /= world
                o = 0
                for p, g in zip(allp, gs):
                    p.grad = flat[o:o + g.numel()].view_as(p).to(p.dtype)
                    o += g.numel()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            scaler.step(opt)
            scaler.update()
            hist.append({k_: float(v) for k_, v in Ls.items()})
            if run and ((step + 1) % 25 == 0 or step + 1 == cfg.steps):
                m = {k_: float(np.mean([h[k_] for h in hist if k_ in h])) for k_ in hist[-1]}
                hist = []
                el = time.time() - t0
                run.scalars({f"loss/{k_}": v for k_, v in m.items()} | {"throughput/steps_per_s": (step + 1) / el,
                                                                         "gpu/peak_gb": torch.cuda.max_memory_reserved() / 2 ** 30}, step + 1)
                if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
                    run.info(f"step {step + 1}: " + ", ".join(f"{k_} {v:.4f}" for k_, v in m.items()) +
                             f"; {(step + 1) / el:.2f} it/s, {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} GB")
                    run.status(f"step {step + 1}/{cfg.steps}")
            if run and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
                ev = dev_eval()
                run.scalars({f"dev/{k_}": v for k_, v in ev.items()}, step + 1)
                run.info(f"dev @ {step + 1}: " + ", ".join(f"{k_} {v:.3f}" for k_, v in ev.items()))
                run.summary.update({f"dev_{k_}": v for k_, v in ev.items()})
        if run:
            torch.save({"model": model.state(), "cfg": asdict(cfg) | {"world": world}}, d / "ckpt-final.pt")
            run.summary.update(steps=cfg.steps, train_s=time.time() - t0, ckpt=str(d / "ckpt-final.pt"))
    except BaseException as e:
        if ctx:
            ctx.__exit__(type(e), e, e.__traceback__)
            ctx = None
        raise
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
        if fr:
            fr.close()
        if world > 1:
            dist.destroy_process_group()


# ---------------------------------------------------------------- navtest plans from pixels
def cmd_plans(a):
    import torch
    import op_lb as OL
    import pp_train as T
    from jevdrive import op_adapt as A
    from jevdrive.run import Run
    from jevdrive.data import splits
    dev = torch.device("cuda")
    mt = OL.meta(a.data)
    names = mt["names"][: a.limit] if a.limit else mt["names"]
    N = len(names)
    tab = np.load(CR / a.data / "tab.npz")
    assert tab["names"][:N].tolist() == names
    ego = torch.from_numpy(tab["ego"][:N]).to(dev)
    tc = torch.from_numpy(np.where(tab["lht"][:N, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)).to(dev)
    with Run("op_parity", f"uplans-{a.frames}-{a.data}", config=vars(a)) as run:
        run.use_split(splits.load(f"navsim/{mt['split']}"))
        from jevdrive.op_torch import _key
        from experiments.op_adapt_l.lib import op_adapt_l as L
        models = {m: T.load_pmodel(m, dev) for m in a.models}
        base = A.load("cinque", torch.float16).to(dev).eval()
        polk = {_key(w) for w in L.pol_weights()}
        ck = {}
        for m in a.models:                                       # which models carry their own (trained) vision weights
            if m == "P0" or m.endswith("-init"):
                ck[m] = False
            else:
                st = torch.load(T.proot("runs", m) / "ckpt-final.pt", map_location="cpu", weights_only=False)["model"]["net"]
                ck[m] = any(k_ not in polk for k_ in st)
        run.info("own vision weights: " + ", ".join(f"{m} {v}" for m, v in ck.items()))
        sl = base.slices
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        ps = np.arange(sl["plan"].start + 495, sl["plan"].start + 990)
        mu = {m: np.zeros((N, 33, 15), np.float32) for m in a.models}
        sd = {m: np.zeros((N, 33, 15), np.float32) for m in a.models}
        fr = Frames([a.data], a.frames, a.workers or max(1, n_cpus() - 2))
        chunks = [np.arange(i, min(i + 32, N)) for i in range(0, N, 32)]
        q = deque((c, fr.submit(c)) for c in chunks[:8])
        nxt, t0 = 8, time.time()
        with torch.no_grad():
            for _ in run.tqdm(range(len(chunks)), desc="plans"):
                rows, futs = q.popleft()
                if nxt < len(chunks):
                    q.append((chunks[nxt], fr.submit(chunks[nxt]))); nxt += 1   # noqa: E702
                prev, cur = Frames.gather(futs)
                p, c = (torch.from_numpy(x.reshape(-1, *x.shape[2:])).to(dev) for x in (prev, cur))
                b = len(rows)
                h0 = base.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(b, 8, *A.H_SHAPE)
                r = torch.as_tensor(rows, device=dev)
                for m, x in models.items():
                    h = x.net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(b, 8, *A.H_SHAPE) if ck[m] else h0
                    o = x(h, ego[r], tc[r], None, None).float().cpu().numpy()
                    mu[m][rows] = o[:, pi].reshape(-1, 33, 15)
                    sd[m][rows] = np.exp(np.minimum(o[:, ps], 11)).reshape(-1, 33, 15)
        fr.close()
        run.summary["tokens_per_s"] = N / (time.time() - t0)
        pdir = OL.root(a.data, "plans")
        eq = {}
        for m in a.models:
            stem = f"{a.frames}@cinque_PP{m}" + (f"-first{a.limit}" if a.limit else "")
            ref = pdir / f"{a.frames}@cinque_PP{m}.npz"                 # the cached-token plans of the same model (pp_eval), if any
            if ref.exists() and a.frames == "warp":
                z = np.load(ref)
                d = np.linalg.norm(mu[m][:, :, :2] - z["plan_mu"][:N, :, :2], axis=-1)
                eq[m] = {"vs_cached_plan_max_m": float(d.max()), "vs_cached_plan_mean_m": float(d.mean())}
                run.info(f"{m} online vs cached plans: {eq[m]}")
            if a.limit or (ref.exists() and m in ("P0", "P2-F-s0") and a.frames == "warp"):   # checks only: keep the cached-path files
                continue
            np.savez(pdir / f"{stem}.npz", names=np.array(names), plan_pos=mu[m][:, :, 0:3], plan_vel=mu[m][:, :, 3:6], plan_yaw=mu[m][:, :, 11],
                     plan_mu=mu[m], plan_std=sd[m], steps=31,
                     info=json.dumps({"model": f"op_parity unfreeze {m}", "frames": a.frames, "source": "experiments/op_parity/scripts/pp_unfreeze.py"}))
        run.summary["equivalence"] = eq
        o = data_dir() / "runs" / "op_parity" / "unfreeze"
        o.mkdir(parents=True, exist_ok=True)
        (o / f"equivalence_{a.frames}_{a.data}{'-first%d' % a.limit if a.limit else ''}.json").write_text(json.dumps(eq, indent=1))


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--what", required=True, choices=["x4", "vh140"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    cli_args(p)
    p = sp.add_parser("train")
    p.add_argument("--var", required=True, choices=list(VARS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data", nargs="+", default=["navtrain_full.s0of12", "navtrain_full.s1of12"])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lr-vis", type=float, default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tag", default="")
    p = sp.add_parser("plans")
    p.add_argument("--data", default="lb_navtest")
    p.add_argument("--frames", default="warp", choices=["warp", "vh140"])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "plans": cmd_plans}[a.cmd](a)
