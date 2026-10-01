"""op-adapt L: step-time micro-benchmark of the trainer's compute (fwd + bwd + AdamW) against the batch size, on real cached rows.
  CUDA_VISIBLE_DEVICES=0 python experiments/op_adapt_l/scripts/op_adapt_l_bench.py --arm sel_polia --batches 64 128 256"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.op_adapt_l.lib import op_adapt_l as L  # noqa: E402
from experiments.op_adapt_l.lib import op_adapt_l_arms as ARMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="sel_polia")
ap.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256])
ap.add_argument("--iters", type=int, default=12)
a = ap.parse_args()
dev = torch.device("cuda")
cfg = ARMS.get(a.arm)
use_h = not cfg.s4
D = L.Data(("wod", "nus"), hstore=use_h)
m = L.LModel(cfg).to(dev).train()
base, new = m.trainable()
opt = torch.optim.AdamW([{"params": base, "lr": 1e-5}] + ([{"params": new, "lr": 1e-4}] if new else []))
scaler = torch.amp.GradScaler()
rows = D.rows("wod", "train")
rng = np.random.default_rng(0)
for B in a.batches:
    r = rng.choice(rows, B, replace=False)
    b = L.to_dev(L.assemble(D, [("wod", r, 0)], use_h), dev)
    ts = []
    for i in range(a.iters + 3):
        torch.cuda.synchronize()
        t0 = time.time()
        o = m.forward_h(b["H"], b["valid"], b["tc"], b["intent"]) if use_h else m(b["trunk"], b["valid"], b["tc"], b["intent"])
        loss = o["outputs"].float().pow(2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    t = float(np.median(ts[3:]))
    print(f"{a.arm} B={B}: {t * 1000:.0f} ms/step, {B / t:.0f} seq/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
