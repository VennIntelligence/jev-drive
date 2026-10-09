"""BODY1 plan forward (plans/2026-10-10-body1-prereg.md, 2.2): the student's own plan and the shipped plan on every state, from cached tokens.

For every (state family, shard) cache dir: P2H10-F-s0 / -s1 (pp_train.PModel on the cached frozen vision tokens, fp16, the path of
jevdrive.bench.navsim.parity_plans) -> 8 rear-axle poses at 0.5 .. 4 s in the state's own frame (pp_train.rear on the plan mean, the grid the
adapter is trained on); the shipped plan = the dir's teacher.npz through the same conversion.
  -> $DATA_DIR/runs/body1/plans/<cache dir>[-first<N>].npz: names, own (M, n, 8, 3), ship (n, 8, 3), models

  CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/op-train/bin/python experiments/body1/scripts/bd1_plans.py --fams log ot1 yr1 [--shards 0] [--limit 200]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "lib"))
import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import b1 as B  # noqa: E402

MODELS = ("P2H10-F-s0", "P2H10-F-s1")


def forward(d, models, W, dev, limit, batch):
    import torch
    import pp_train as T
    S = T.Store([d], dev, need_side=False, frames="warp")
    n = min(S.n, limit) if limit else S.n
    pi = torch.as_tensor(S.pi, device=dev)
    own = np.zeros((len(models), n, 8, 3), np.float32)
    with torch.no_grad():
        ship = torch.stack(T.rear(S.t_plan[:n].float(), S.cam_x[:n], W), -1).cpu().numpy()
        for m, model in enumerate(models.values()):
            for i in range(0, n, batch):
                r = torch.arange(i, min(i + batch, n), device=dev)
                p = model(S.front[r], S.ego[r], S.tc[r], None, None).float()[:, pi].view(-1, 33, 15)
                own[m, i:i + len(r)] = torch.stack(T.rear(p, S.cam_x[r], W), -1).cpu().numpy()
    return dict(names=S.tab["names"][:n], own=own, ship=ship.astype(np.float32), models=np.array(list(models)))


def main(a):
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import cache
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    units = [(f, k) for f in a.fams for k in (a.shards if a.shards is not None else range(B.NSH))]
    with Run("body1", "plans" + (f"-first{a.limit}" if a.limit else ""), config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        models = {t: T.load_pmodel(t, dev) for t in a.models}
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        out = B.root() / "plans"
        out.mkdir(parents=True, exist_ok=True)
        t0, n = time.time(), 0
        for f, k in run.tqdm(units, desc="dirs"):
            d = B.cdir(f, k)
            key = cache.key(params=dict(models=list(a.models), limit=a.limit), inputs=[B.cache_root() / f"{d}@warp" / "front.npy", B.cache_root() / d / "tab.npz"]
                            + [T.proot("runs", t) / "ckpt-final.pt" for t in a.models], code=[forward])
            z = cache.cached(out / (d + (f"-first{a.limit}" if a.limit else "") + ".npz"), key, lambda: forward(d, models, W, dev, a.limit, a.batch), force=a.force)
            n += len(z["names"])
            if (f, k) == units[0]:
                ade = float(np.hypot(*(z["own"][0] - z["ship"])[..., :2].transpose(2, 0, 1)).mean())
                run.info(f"{d}: {len(z['names'])} rows; own-vs-shipped mean pose distance {ade:.3f} m; own 4 s x mean {z['own'][0][:, -1, 0].mean():.2f} m")
        run.summary.update(dirs=len(units), rows=n, compute_s=time.time() - t0, vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1"])
    ap.add_argument("--shards", type=int, nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    cli_args(ap)
    main(ap.parse_args())
