#!/usr/bin/env python3
"""STOPLINE-PROBE PAI features (Tokyo, CPU, inside the driver image jev-alpasim:* which ships jevdrive/ and cinque.ort.onnx): the frozen
Cinque `view_39` tokens (32 x 512) of the t0 slot of PAI_DUMP frames, computed as pai_core.plan does (vision_feeds(prev = slot 6, cur = slot 7)),
fp32 on the CPU (the served model runs the same graph in fp16 on the card).

  slp_feats.py --list files.txt --out feats.npz [--onnx /app/data/models/openpilot/cinque.ort.onnx] [--threads 8] [--batch 16]
  -> feats.npz: files (n,), X (n, 16384) float16
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app/jev-drive")
from jevdrive import op_adapt as A  # noqa: E402
from jevdrive.op_torch import OnnxTorch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default="/app/data/models/openpilot/cinque.ort.onnx")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    files = [f for f in open(a.list).read().split() if f]
    net = OnnxTorch(a.onnx, dtype=torch.float32).eval()
    X = np.zeros((len(files), 32 * 512), np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(files), a.batch):
            cur = np.stack([np.load(f)["cur"] for f in files[i:i + a.batch]])          # (b, 8, 2, 6, 128, 256)
            p, c = torch.from_numpy(cur[:, 6]), torch.from_numpy(cur[:, 7])
            H = net.run_batched(A.vision_feeds(p, c), ["view_39"])["view_39"].reshape(len(cur), -1)
            X[i:i + len(cur)] = H.float().numpy().astype(np.float16)
            print(f"{i + len(cur)}/{len(files)} {time.time() - t0:.0f} s", flush=True)
    np.savez(a.out, files=np.array(files), X=X)


if __name__ == "__main__":
    main()
