"""Dry read of pp_train's B2D path (no optimizer step, no checkpoint): the Store over navtrain_full shard 0 + b2d_v2 (host mode), the route / token split, a
--b2d-mass draw, the mixed hinge (Pacifica labels for NAVSIM rows, MKZ labels for B2D rows) and one forward + loss on a mixed batch. One pool job.

  python experiments/op_parity/scripts/b2d_trainer_check.py [--b2d-mass 0.5]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pp_train as T  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402


def main(a):
    from jevdrive.run import Run
    from drivable_hinge import Hinge
    dev = torch.device("cuda")
    datas = ("navtrain_full.s0of12", "b2d_v2")
    with Run("op_parity", "b2d-trainer-check", config=vars(a)) as run:
        S = T.Store(datas, dev, need_side=False, frames="warp", host=True)
        tz = [np.load(data_dir() / "runs/op_parity/cache" / d / "tab.npz") for d in datas]
        tabs = dict(names=S.tab["names"], log=S.tab["log"], is_b2d=S.is_b2d)
        tr, dv, sp = T.split_rows(tabs, "navsim/op-parity-full", "b2d/b2dc-v2")
        run.info(f"rows {S.n} (b2d {int(S.is_b2d.sum())}); train {len(tr)} (b2d {int(S.is_b2d[tr].sum())}), dev {len(dv)} (b2d {int(S.is_b2d[dv].sum())})")
        assert not (set(tr) & set(dv))
        assert set(S.tab["log"][tr][S.is_b2d[tr]]).isdisjoint(set(S.tab["log"][dv][S.is_b2d[dv]])), "route leak"
        h = Hinge([data_dir() / "runs/op_probe/labels/navtrain_all.npz", data_dir() / "runs/op_parity/cache/b2d_v2/hinge_labels.npz"], S.tab["names"], dev, 0.3,
                  ["pacifica", "mkz"])
        run.info(f"hinge coverage {h.coverage:.4f}; sources: NAVSIM {int((h.src[h.ok] == 0).sum())}, B2D {int((h.src[h.ok] == 1).sum())}")
        model = T.PModel("P2").to(dev)
        cfg = T.Cfg(arm="P2", hinge_lam=10.0)
        tstd = S.t_out[torch.as_tensor(tr, device=dev)].float().std(0).clamp_min(1e-3)
        LS = T.Losses(model.net, cfg, tstd, S.di, S.pi, dev, h)
        rng = np.random.default_rng(0)
        b = S.is_b2d[tr]
        w = np.where(b, a.b2d_mass / b.sum(), (1 - a.b2d_mass) / (~b).sum())
        rows = rng.choice(tr, 128, p=w / w.sum())
        anchor = torch.as_tensor(rng.random(128) < 0.25, device=dev)
        r = torch.as_tensor(rows, device=dev)
        t0 = time.time()
        for _ in range(5):
            front = S.front[r]
        fetch = (time.time() - t0) / 5
        ego = S.ego[r] * (~anchor)[:, None].float()
        model.train()
        out = model(front, ego, S.tc[r], None, None)
        total, Ls = LS(out, S, r, anchor)
        total.backward()                                           # gradient only (no optimizer): proves the graph and the hinge are differentiable
        g = float(sum(p.grad.float().norm() ** 2 for p in model.groups()[0] if p.grad is not None) ** 0.5)
        bm = torch.as_tensor(S.is_b2d[rows], device=dev)
        res = dict(batch_b2d_share=float(bm.float().mean()), fetch_s_per_128_rows=fetch, loss=float(total), parts={k: float(v) for k, v in Ls.items()},
                   grad_norm=g, finite=bool(torch.isfinite(total)), vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
        run.info(json.dumps(res))
        run.summary.update(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2d-mass", type=float, default=0.5)
    from jevdrive.run import cli_args
    cli_args(ap)
    main(ap.parse_args())
