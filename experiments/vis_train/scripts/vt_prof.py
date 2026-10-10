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
    a = ap.parse_args()
    {"enc": cmd_enc}[a.cmd](a)
