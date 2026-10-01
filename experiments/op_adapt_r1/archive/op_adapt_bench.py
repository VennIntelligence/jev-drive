"""op-adapt step 2: training throughput of the Cinque port on this box's card (op-train venv).

Real input sizes with the full 9-frame temporal context, bf16 compute, AdamW step included:
  B       stage 4 unfrozen, stage-3 output cached (input (B, 9, 1024, 8, 16))
  B-lora  stage 4 frozen + LoRA r=16 on its MLP weights
  C       every weight trainable, the whole vision stack on all 9 context frames with autograd (9 image pairs)
  C-past  every weight trainable, only the current frame with autograd, the 8 past frames under no_grad
  infer   the frozen full model, forward only (feature extraction / evaluation)
samples/s, peak VRAM, and GPU-h per 100k training samples, per batch size (until OOM).

  CUDA_VISIBLE_DEVICES=2 python experiments/op_adapt_r1/archive/op_adapt_bench.py
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.runlog import RunLog  # noqa: E402

AT = (0.275, 0.525)


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}


def build(mode, dt):
    if mode == "B":
        net = A.load("cinque", dt, trainable=A.stage4_weights())
    elif mode == "B-lora":
        net = A.load("cinque", dt, lora={w: 16 for w in A.stage4_matmuls()})
    elif mode in ("C", "C-past"):
        import onnx
        g = onnx.load(str(A.MODELS_DIR / A.FILES["cinque"])).graph
        net = A.load("cinque", dt, trainable=[t.name for t in g.initializer if t.data_type in (1, 10)])
    else:
        net = A.load("cinque", dt)
    return net.cuda()


def batch(mode, B, dev):
    if mode.startswith("B"):
        return (torch.randn(B, A.CONTEXT, 1024, 8, 16, device=dev, dtype=torch.float16),)
    img = lambda: torch.randint(0, 255, (B, A.CONTEXT, 2, 6, 128, 256), dtype=torch.uint8, device=dev)  # noqa: E731
    return img(), img()


def step_fn(mode, net, heads, opt):
    def f(*x):
        if mode.startswith("B"):
            o = A.stage4_policy(net, x[0], AT)
        elif mode == "infer":
            with torch.no_grad():
                return A.full_policy(net, x[0], x[1], AT)
        else:
            o = A.full_policy(net, x[0], x[1], AT, grad_past=(mode == "C"))
        a, b = heads(o["select_4"], o["tokens"])
        loss = o["outputs"][:, :2066].float().pow(2).mean() + a.pow(2).mean() + b.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["B", "B-lora", "C-past", "C", "infer"])
    ap.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--max-gb", type=float, default=0, help="stop growing the batch past this peak (shared card)")
    ap.add_argument("--dtypes", nargs="+", default=["fp16", "tf32", "bf16"])
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    log = RunLog("op_adapt", "bench")
    dev = torch.device("cuda")
    rows = []
    for mode, dn in [(m, d) for m in a.modes for d in a.dtypes]:
        net = build(mode, DTYPES[dn])
        heads = A.AuxHeads().to(dev)
        params = [p for p in net.parameters() if p.requires_grad] + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=1e-5, fused=True)
        ntr = sum(p.numel() for p in net.parameters() if p.requires_grad)
        f = step_fn(mode, net, heads, opt)
        for B in a.batches:
            try:
                x = batch(mode, B, dev)
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    f(*x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    f(*x)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / a.iters
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                r = dict(mode=mode, dtype=dn, batch=B, trainable_M=ntr / 1e6, samples_per_s=B / dt, peak_gb=peak,
                         gpu_h_per_100k=1e5 / (B / dt) / 3600)
                rows.append(r)
                log.info(str(r))
                log.event("bench", **r)
                del x
                if a.max_gb and peak > a.max_gb * 0.6:
                    break
            except torch.OutOfMemoryError:
                log.info(f"{mode} {dn} batch {B}: OOM")
                break
            finally:
                torch.cuda.empty_cache()
        del net, heads, opt, f
        torch.cuda.empty_cache()
    import pandas as pd
    pd.DataFrame(rows).to_csv(log.dir / "bench.csv", index=False)
    log.event("end", n=len(rows))


if __name__ == "__main__":
    main()
