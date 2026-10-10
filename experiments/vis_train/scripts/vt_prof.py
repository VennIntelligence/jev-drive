"""vis_train throughput profile: where the time of an encoder-training step goes, before (pp_unfreeze U2 as shipped) and after.

  enc   --variants u2 plain ... --batch 64     the encoder alone on one batch of image pairs: forward / backward ms, peak VRAM, and the
                                               numeric distance of every variant to `plain` (tokens, gradients)
  step  --shape A|C --mode old|fast            the whole training step with real data (pixel cache + cached tokens): time split
                                               read / upload / encoder forward / policy forward / backward / optimizer, it/s, VRAM

Shapes (plans/2026-10-10-vis-train-prereg.md): A = cached tokens of the 8 slots + a trainable encoder copy on the t0 pair whose tokens
enter the policy through the memory channel; C = in-place unfreeze, all 8 slots through the current encoder, gradient through the t0
pair. Results -> $DATA_DIR/runs/vis_train/prof/<tag>.json.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import pixel_store as PX  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

OUT = data_dir() / "runs" / "vis_train" / "prof"
SH = [f"navtrain_full.s{i}of12" for i in range(12)]


def mk_enc(dtype, train=True):
    from jevdrive import op_adapt as A
    import pp_unfreeze as U
    return A.load("cinque", dtype, trainable=U.vision_names("all") if train else ())


def enc_u2(net, prev, cur, ckpt=8):
    """pp_unfreeze.UModel.encode as shipped for the t0 pair (activation checkpointing per 8 pairs)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from jevdrive import op_adapt as A
    f = lambda p, c: net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(c), *A.H_SHAPE)  # noqa: E731
    return torch.cat([checkpoint(f, prev[i:i + ckpt], cur[i:i + ckpt], use_reentrant=False) for i in range(0, len(cur), ckpt)])


def pixels(n, dev, seed=0):
    """n real t0 pairs from the pixel cache if it is there (else random bytes: the timing does not depend on the content)."""
    import torch
    f = [d for d in SH if (PX.px_root() / d / "frames.npy").exists()]
    if f:
        PS = PX.PixelStore(f[:1], dev)
        return PS.t0(np.sort(np.random.default_rng(seed).choice(len(PS), n, replace=False)))
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randint(0, 256, (n,) + PX.FRAME, dtype=torch.uint8, generator=g).to(dev) for _ in range(2))


def sync():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def cmd_enc(a):
    import torch
    dev = torch.device("cuda")
    res, ref = {}, None
    with Run("vis_train", f"prof-enc-{a.tag}", config=vars(a)) as run:
        prev, cur = pixels(a.batch, dev)
        for v in a.variants:
            opts = set(v.split("+"))
            torch.backends.cudnn.benchmark = "bench" in opts
            dt = torch.bfloat16 if "bf16" in opts else torch.float16
            net = mk_enc(dt).to(dev)
            if "cl" in opts:
                for p in net.params.values():
                    if p.dim() == 4:
                        p.data = p.data.contiguous(memory_format=torch.channels_last)
            if "u2" in opts:
                f = lambda p, c: enc_u2(net, p, c)  # noqa: E731
            else:
                kw = dict(compiled="compile" in opts)
                f = lambda p, c: PX.fast_encode(net, p, c, grad=True, **kw)  # noqa: E731
            ps = [p for p in net.params.values() if p.requires_grad]
            tf, tb = [], []
            torch.cuda.reset_peak_memory_stats()
            t00 = time.time()
            for i in range(a.warm + a.iters):
                t0 = sync()
                h = f(prev, cur)
                t1 = sync()
                loss = h.float().pow(2).mean()
                for p in ps:
                    p.grad = None
                loss.backward()
                t2 = sync()
                if i == 0:
                    first = time.time() - t00
                if i >= a.warm:
                    tf.append(t1 - t0), tb.append(t2 - t1)
            g = torch.cat([p.grad.flatten().float() for p in (ps[0], ps[len(ps) // 2], ps[-1])])
            hh = h.detach().float()
            if ref is None:
                ref = (hh, g)
            r = {"fwd_ms": 1e3 * float(np.median(tf)), "bwd_ms": 1e3 * float(np.median(tb)), "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                 "first_s": first, "pairs_per_s": a.batch / float(np.median(tf) + np.median(tb)),
                 "tok_mean_abs_d": float((hh - ref[0]).abs().mean()), "tok_max_abs_d": float((hh - ref[0]).abs().max()), "tok_rms": float(ref[0].pow(2).mean().sqrt()),
                 "grad_cos": float(torch.nn.functional.cosine_similarity(g, ref[1], 0)), "grad_finite": bool(torch.isfinite(g).all())}
            res[v] = r
            run.info(f"{v}: " + ", ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}" for k, x in r.items()))
            if a.profile and v == a.variants[0]:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    h = f(prev, cur)
                    h.float().pow(2).mean().backward()
                    sync()
                run.info("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=40))
                run.info("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=40))
            del net, h, loss, ps
            torch.cuda.empty_cache()
        run.summary |= res
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"enc-{a.tag}.json").write_text(json.dumps(res, indent=1))


def cmd_step(a):
    """One training step of shape A or C on real rows, timed by segment (every segment ends in a cuda synchronize)."""
    import torch
    import pp_train as T
    import pp_unfreeze as U
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = a.bench
    datas = [d for d in (a.data or SH) if (PX.px_root() / d / "frames.npy").exists()]
    with Run("vis_train", f"prof-step-{a.tag}", config=vars(a)) as run:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        S = T.Store(datas, dev, need_side=False, frames="warp", host=True)
        S.front.pin = True
        PS = PX.PixelStore(datas, dev, mode="t0" if a.shape == "A" else "all", threads=a.threads)
        assert len(PS) == S.n
        if a.shape == "A":
            model = T.PModel("P2+ge_vt").to(dev)
            enc = mk_enc(torch.float16).to(dev)
            pol, new = model.groups()
            vis = [p for p in enc.params.values() if p.requires_grad]
        else:
            model = U._torch_model()("U2").to(dev)
            enc = model.net
            pol, vis, new = model.groups()
        tstd = S.t_out[:20000].float().std(0).clamp_min(1e-3)
        LS = T.Losses(model.net, T.Cfg(arm="P2", lam_i=1.0, lam_c=3.0, lam_d=30.0), tstd, S.di, S.pi, dev)
        opt = torch.optim.AdamW([{"params": pol, "lr": 1e-5}, {"params": new, "lr": 1e-4}, {"params": vis, "lr": 1e-5}], weight_decay=0.01, fused=a.fused)
        scaler = torch.amp.GradScaler()
        allp = pol + vis + new
        d_frac = 0.25 if a.shape == "A" else 0.0
        io = {"read": [], "up": []}

        def draw():
            return rng.choice(S.n, a.batch, replace=False), rng.random(a.batch) < d_frac, rng.random((a.batch, 1)) >= 0.25

        def fetch(dr):
            t0 = time.perf_counter()
            if a.shape == "A":
                hb, front = PS.host(dr[0], PX.T0, 2), S.front[dr[0]]
            else:
                hb, front = PS.host(dr[0]), None
            t1 = time.perf_counter()
            g = hb.to(dev, non_blocking=True)
            torch.cuda.current_stream().synchronize() if a.sync_io else None
            io["read"].append(t1 - t0), io["up"].append(time.perf_counter() - t1)
            return dr, front, g

        def encode(p, c, grad):
            if a.enc == "u2":
                return enc_u2(enc, p, c) if grad else model.encode(p, c) if a.shape == "C" else None
            return PX.fast_encode(enc, p, c, grad=grad, compiled=a.compiled, chunk=0 if grad else a.chunk)

        fwd = torch.compile(model, dynamic=False) if a.pol_compile and a.shape == "A" else model
        pre = T.Prefetch(draw, fetch, a.warm + a.steps, depth=a.depth, workers=a.pw)
        seg = {k: [] for k in ("wait", "enc", "policy", "backward", "optim", "log")}
        torch.cuda.reset_peak_memory_stats()
        import resource
        t_all = cpu0 = None
        for step in range(a.warm + a.steps):
            if step == a.warm:
                t_all, cpu0 = sync(), sum(resource.getrusage(w).ru_utime + resource.getrusage(w).ru_stime for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
                io = {"read": [], "up": []}
            t0 = sync()
            (rows_np, an_np, sm_np), front, g = pre.get()
            rows, anchor = torch.as_tensor(rows_np, device=dev), torch.as_tensor(an_np, device=dev)
            t1 = sync()
            ego = S.ego[rows] * (~anchor)[:, None].float()
            if a.shape == "A":
                h = encode(g[:, 0], g[:, 1], True)
                t2 = sync()
                out = fwd(front, ego, S.tc[rows], side=h, side_mask=torch.as_tensor(sm_np, device=dev))
            else:
                with torch.no_grad():
                    hp = encode(torch.cat([torch.zeros_like(g[:, :1]), g[:, :6]], 1).flatten(0, 1), g[:, :7].flatten(0, 1), False)
                ht = encode(g[:, 6], g[:, 7], True)
                front = torch.cat([hp.reshape(a.batch, 7, *A.H_SHAPE), ht[:, None]], 1)
                t2 = sync()
                H = torch.cat([front.new_zeros(a.batch, 1, *A.H_SHAPE), front], 1).to(model.net.dtype)
                valid = torch.ones(a.batch, A.CONTEXT, dtype=torch.bool, device=dev)
                valid[:, 0] = False
                H = model.adapter.apply(H, ego) * valid[:, :, None, None].to(H.dtype)
                out = model.net.run_batched(A.policy_feeds(model.net, H, T.AT, S.tc[rows].to(model.net.dtype)), ["outputs"])["outputs"].reshape(a.batch, -1)
            total, Ls = LS(out, S, rows, anchor)
            t3 = sync()
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            t4 = sync()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            scaler.step(opt)
            scaler.update()
            t5 = sync()
            _ = {k: float(v) for k, v in Ls.items()}
            t6 = sync()
            assert torch.isfinite(total), "non-finite loss"
            if step >= a.warm:
                for k, v in zip(seg, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)):
                    seg[k].append(v)
            if step == 0:
                run.info(f"first step {time.perf_counter() - t0:.1f} s (compile / cudnn search included)")
        el = sync() - t_all
        cpu = sum(resource.getrusage(w).ru_utime + resource.getrusage(w).ru_stime for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) - cpu0
        import subprocess
        r = {"shape": a.shape, "enc": a.enc, "batch": a.batch, "it_s": a.steps / el, "samples_s": a.steps * a.batch / el,
             "ms": {k: 1e3 * float(np.mean(v)) for k, v in seg.items()}, "read_ms_thread": 1e3 * float(np.mean(io["read"])),
             "upload_ms_thread": 1e3 * float(np.mean(io["up"])), "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
             "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30, "cpu_cores_used": cpu / el, "rows": int(S.n), "datas": len(datas),
             "loss": float(total)}
        run.info(json.dumps(r))
        run.summary |= r
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"step-{a.tag}.json").write_text(json.dumps(r | {"args": vars(a)}, indent=1))


def cmd_equiv(a):
    """Plans of one model on fixed rows through every pixel path against its cached-token path (front.npy): plan points <= 4 s, metres."""
    import torch
    import pp_prep as P
    import pp_train as T
    from jevdrive import op_adapt as A
    dev = torch.device("cuda")
    with Run("vis_train", f"prof-equiv-{a.tag}", config=vars(a)) as run:
        S = T.Store([a.data], dev, need_side=False, frames="warp", host=True)
        PS = PX.PixelStore([a.data], dev, mode="all")
        rows = np.arange(min(a.rows, len(PS)))
        m = T.load_pmodel(a.model, dev)
        pi = torch.as_tensor(A.plan_index(m.net.slices), device=dev)
        keep = torch.as_tensor(np.flatnonzero(A.T_IDXS <= 4.0), device=dev)
        B = a.batch

        def plans(tokens, model=m):
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), B):
                    r = torch.as_tensor(rows[i:i + B], device=dev)
                    o = model(tokens(rows[i:i + B]), S.ego[r], S.tc[r]).float()
                    out.append(o[:, pi].view(-1, 33, 15)[:, keep, :2])
            return torch.cat(out)

        def px(kind):
            def f(r):
                g = PS.frames(r)
                prev = torch.cat([torch.zeros_like(g[:, :1]), g[:, :-1]], 1)
                if kind == "shipped":                                               # pp_prep's passes: all 8 slots, 128 pairs per pass
                    h = torch.from_numpy(P.enc_dev(m.net, prev.flatten(0, 1), g.flatten(0, 1))).to(dev)
                    return h.reshape(len(r), 8, *A.H_SHAPE)
                hp = PX.fast_encode(m.net, prev[:, :7].flatten(0, 1), g[:, :7].flatten(0, 1), grad=False)
                ht = PX.fast_encode(m.net, prev[:, 7], g[:, 7], grad=False, chunk=len(r), compiled=kind == "compiled")
                return torch.cat([hp.reshape(len(r), 7, *A.H_SHAPE), ht[:, None]], 1)
            return f
        ref = plans(lambda r: S.front[r])
        res = {}
        paths = {"px_shipped": px("shipped"), "px_fast": px("fast")} | ({"px_fast_compiled": px("compiled")} if a.compiled else {})
        cmp = {k: plans(f) for k, f in paths.items()}
        if a.wrong:
            cmp["wrong_model_cached"] = plans(lambda r: S.front[r], T.load_pmodel(a.wrong, dev))
        for k, p in cmp.items():
            d = torch.linalg.norm(p - ref, dim=-1)                                  # (n, points)
            res[k] = {"mean_m": float(d.mean()), "median_row_max_m": float(d.amax(1).median()), "max_m": float(d.max()),
                      "rows_over_0.03m": float((d.amax(1) > 0.03).float().mean())}
            run.info(f"{a.model} on {a.data} first {len(rows)} rows, {k} vs cached tokens: {res[k]}")
        run.summary |= res
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"equiv-{a.tag}.json").write_text(json.dumps(res | {"args": vars(a)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("enc")
    p.add_argument("--variants", nargs="+", default=["plain", "u2"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--tag", required=True)
    cli_args(p)
    p = sp.add_parser("step")
    p.add_argument("--shape", required=True, choices=["A", "C"])
    p.add_argument("--enc", default="fast", choices=["fast", "u2"], help="u2: the shipped encode (checkpoint per 8 pairs, chunks of 128 without autograd)")
    p.add_argument("--compiled", action="store_true", help="torch.compile of the encoder pass")
    p.add_argument("--pol-compile", action="store_true", help="torch.compile of the policy forward (shape A)")
    p.add_argument("--fused", action="store_true", help="fused AdamW")
    p.add_argument("--bench", action="store_true", help="cudnn.benchmark")
    p.add_argument("--sync-io", action="store_true", help="synchronise the upload inside the fetch thread (to time it)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chunk", type=int, default=128, help="pairs per pass of the slots without autograd")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--pw", type=int, default=2, help="prefetch threads")
    p.add_argument("--threads", type=int, default=0, help="reader threads of the pixel store")
    p.add_argument("--data", nargs="+", default=None)
    p.add_argument("--tag", required=True)
    cli_args(p)
    p = sp.add_parser("equiv")
    p.add_argument("--data", default="lb_navtest")
    p.add_argument("--model", default="SH30-F-s0")
    p.add_argument("--wrong", default="SH30-F-s1", help="wrong-model control on the cached tokens ('' = none)")
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--compiled", action="store_true")
    p.add_argument("--tag", required=True)
    cli_args(p)
    a = ap.parse_args()
    {"enc": cmd_enc, "step": cmd_step, "equiv": cmd_equiv}[a.cmd](a)
