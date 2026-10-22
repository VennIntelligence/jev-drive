"""Knows-vs-does features of WA-JEPA on navtest (plans/2026-10-23-p1-knowdo-prereg.md): the planner-input tensor pooled
without training, plus the planner's non-image inputs. Encoder only (the planner is skipped via usage_probe.Captured).

  X: per camera, the later tubelet's 16 x 32 grid of context_scene average-pooled to 2 x 4 cells, each cell projected on the
     top-64 navtrain PCA directions (level (a)'s pca.npz): 4 x 8 x 64 = 2048, then the global token mean (512)
  B: ego_status (command one-hot, vx, vy, ax, ay) + history_trajectory (4 x 3) = 20

  feats --shard I N -> $P1_OUT/feats/sIofN.npz ; merge -> $P1_OUT/feats_navtest.npz (tokens, X, B)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_wajepa as W  # noqa: E402
from p1_wajepa import UP  # noqa: E402

NPC = 64


def pool(ctx: torch.Tensor, views: int, mu: torch.Tensor, U: torch.Tensor) -> np.ndarray:
    """ctx [1, V * 1024, 512] (per view: 2 tubelets x 16 x 32, row-major) -> (V * 8 * NPC + 512,)."""
    x = ctx.float()[0].reshape(views, 2, 16, 32, -1)[:, 1]                     # later tubelet [V, 16, 32, D]
    cells = x.reshape(views, 2, 8, 4, 8, x.shape[-1]).mean((2, 4))           # [V, 2, 4, D]
    z = (cells - mu) @ U[:, :NPC]                                             # [V, 2, 4, NPC]
    return torch.cat([z.reshape(-1), ctx.float()[0].mean(0)]).cpu().numpy().astype(np.float32)


def cmd_feats(a):
    from jevdrive import par
    A = W.agent()
    m = A.model
    L, toks = W.loader("navtest", A.get_sensor_config())
    mine = par.shards(toks, a.shard[1], a.shard[0])
    pca = UP.PCA.load(W.OUT / "pca.npz")
    mu = torch.as_tensor(pca.mean, dtype=torch.float32, device="cuda")
    U = torch.as_tensor(pca.comps, dtype=torch.float32, device="cuda")
    got = {}

    def grab(ctx):
        got["x"] = pool(ctx, m.num_cameras, mu, U)
        raise UP.Captured
    T, X, B = [], [], []
    t0 = time.time()
    with UP.Probe(m, W.SITE) as p, torch.no_grad():
        p.apply(grab)
        for i, (t, f) in enumerate(W.batches(A, L, mine, W.workers()), 1):
            try:
                m.predict_trajectory(f)
            except UP.Captured:
                pass
            T.append(t)
            X.append(got.pop("x"))
            B.append(np.r_[f["ego_status"][0].float().cpu().numpy(), f["history_trajectory"][0].float().cpu().numpy().ravel()])
            if i % 500 == 0:
                print(f"[feats] {i}/{len(mine)} {(time.time() - t0) / i:.2f} s/token", flush=True)
    d = W.OUT / "feats"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"s{a.shard[0]}of{a.shard[1]}.npz", tokens=np.array(T), X=np.stack(X), B=np.stack(B).astype(np.float32))


def cmd_merge(a):
    zs = [np.load(f) for f in sorted((W.OUT / "feats").glob("s*of*.npz"))]
    T = np.concatenate([z["tokens"] for z in zs])
    assert len(set(T)) == len(T), "duplicate tokens"
    np.savez(W.OUT / "feats_navtest.npz", tokens=T, X=np.concatenate([z["X"] for z in zs]), B=np.concatenate([z["B"] for z in zs]))
    print(f"{len(T)} tokens -> {W.OUT / 'feats_navtest.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "merge"])
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    a = ap.parse_args()
    os.chdir(W.WJ)
    from jevdrive.run import Run
    from jevdrive.data import splits
    with Run("wm_surplus", f"p1kd-{a.cmd}" + (f"-s{a.shard[0]}of{a.shard[1]}" if a.cmd == "feats" else ""), config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
