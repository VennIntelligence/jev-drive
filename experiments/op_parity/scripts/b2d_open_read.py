"""Open-loop read of op_parity arms on the held-out B2D val routes (b2d/b2dc-v2-val) from the b2d_prep cache: stage 0 of plans/2026-10-07-b2d-p2-prereg.md.

For every model tag (P0 = shipped, P2-F-s0, B1-F-s0, ... any pp_train checkpoint) the plan is computed from the cached tokens, the cache's ego features and route command
(`--cache b2d_v2` = 30 m lookahead as collected, `b2d_v2L20` = 20 m) and compared with the logged PDM-Lite future. Rows: all val rows (16 257 at the cache build).

  ADE            mean over the 8 poses (m), rows where the log moves > 2 m in 4 s
  turn_sign      sign of the plan's 4 s heading change = the log's, rows with |logged heading change| > 20 deg and motion
  len_ratio      median of plan path length / logged path length over the 4 s, moving rows
  junction stop  rows with a LEFT / RIGHT junction option within 30 m ahead: `false_stop` = the log drives >= 4 m in 4 s and the plan < 50% of it;
                 `false_go` = the log stands (< 0.5 m in 4 s) and the plan drives > 2 m. Decision 151's blocker is the plan near-stopping in junctions.

  python experiments/op_parity/scripts/b2d_open_read.py --models P0 P2-F-s0 P2H10-F-s0 [--cache b2d_v2L20] [--out DIR]
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

MOUNT_X = 1.59


def path_len(p):
    return np.linalg.norm(np.diff(np.concatenate([np.zeros((len(p), 1, 2)), p[:, :, :2]], 1), axis=1), axis=-1).sum(1)


def main(a):
    import torch
    import pp_train as T
    from jevdrive import op_adapt as A
    from jevdrive.data import splits
    from jevdrive.run import Run
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    d = data_dir() / "runs" / "op_parity" / "cache" / a.cache
    dev = torch.device("cuda")
    with Run("op_parity", f"b2d-open-read-{a.cache}", config=vars(a)) as run:
        val = splits.load("b2d/b2dc-v2-val")
        run.use_split(val)
        ticks = np.load(d / "ticks.npy", mmap_mode="r")
        fidx = np.load(d / "front_idx.npy")
        tab, ex = dict(np.load(d / "tab.npz")), np.load(d / "extra.npz")
        rows = np.flatnonzero(val.mask(ex["route"]))
        fut = tab["fut"][rows]
        L = path_len(fut)
        yaw4 = np.degrees(fut[:, -1, 2])
        mv = L > 2.0
        big = mv & (np.abs(yaw4) > 20)
        jn = (ex["turn_next"][rows] > 0) & (ex["turn_dist"][rows] <= 30)
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        res = dict(cache=a.cache, rows=len(rows), moving=int(mv.sum()), turn_rows=int(big.sum()), junction_go=int((jn & (L >= 4)).sum()),
                   junction_stand=int((jn & (L < 0.5)).sum()), models={})
        for m in a.models:
            model = T.load_pmodel(m, dev)
            pi = torch.as_tensor(A.plan_index(model.net.slices), device=dev)
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 128):
                    r = rows[i:i + 128]
                    o = model(torch.from_numpy(ticks[fidx[r]]).to(dev), torch.from_numpy(tab["ego"][r]).to(dev),
                              torch.tensor([[1.0, 0.0]], device=dev).expand(len(r), 2)).float()[:, pi].view(-1, 33, 15)
                    x, y, psi = T.rear(o, torch.full((len(r),), MOUNT_X, device=dev), W)
                    out.append(torch.stack([x, y, psi], -1).cpu().numpy())
            p = np.concatenate(out)
            ade = np.linalg.norm(p[..., :2] - fut[..., :2], axis=-1).mean(1)
            lp = path_len(p)
            go, st = jn & (L >= 4), jn & (L < 0.5)
            res["models"][m] = dict(
                ade_moving=float(ade[mv].mean()), ade_all=float(ade.mean()),
                turn_sign=float((np.sign(np.degrees(p[big, -1, 2])) == np.sign(yaw4[big])).mean()),
                len_ratio_med=float(np.median(lp[mv] / L[mv])),
                junction_false_stop=float((lp[go] < 0.5 * L[go]).mean()), junction_false_go=float((lp[st] > 2.0).mean()),
                junction_len_ratio_med=float(np.median(lp[go] / L[go])))
            run.info(f"{m}: " + json.dumps(res["models"][m]))
            del model
            torch.cuda.empty_cache()
        out_dir = _pl.Path(a.out) if a.out else data_dir() / "runs" / "op_parity" / "b2d_open_read"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"open_read_{a.cache}.json").write_text(json.dumps(res, indent=1))
        run.summary.update(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["P0", "P2-F-s0", "P2H10-F-s0"])
    ap.add_argument("--cache", default="b2d_v2")
    ap.add_argument("--out", default="")
    from jevdrive.run import cli_args
    cli_args(ap)
    main(ap.parse_args())
