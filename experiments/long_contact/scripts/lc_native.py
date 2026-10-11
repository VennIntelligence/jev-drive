#!/usr/bin/env python3
"""LONG-PROBE rung 3 inputs (GPU, one card): openpilot's native outputs that the driver computes anyway, from the SAME forward that makes the
student's own plan. For every (state family, shard) cache dir and each student P2H10-F-s0 / -s1 (body1's own-plan models, bd1_plans.py path):
  nat (n, 2, 206) float32  raw output slices lead (144: 3 hypotheses x 6 times x (x, y, v, a) means, then their log std), lead_prob (3 logits),
                           meta (55), action (4)
  hid (n, 2, 1024) float16 the policy's temporal summary `select_4` (512) and pooled vision `mean` (512) (jevdrive.op_adapt.POLICY_OUT taps)
  err (n, 2) float32       max |pose difference| (m / rad) of this forward's rear-axle plan against body1/plans (identity gate)
-> $DATA_DIR/runs/long_contact/native/<cache dir>.npz
  $DATA_DIR/envs/op-train/bin/python experiments/long_contact/scripts/lc_native.py [--fams log ot1 yr1] [--shards ...] [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1] / "lib")]
import lc as LC  # noqa: E402
import b1 as B  # noqa: E402

MODELS = ("P2H10-F-s0", "P2H10-F-s1")
SLICES = ("lead", "lead_prob", "meta", "action")
TOL = 0.03                                       # m: the fp16 step of decision 192 / nc_slow G-plan


def forward(d, models, W, dev, limit, batch):
    import torch
    import pp_train as T
    from jevdrive import op_adapt as A
    S = T.Store([d], dev, need_side=False, frames="warp")
    n = min(S.n, limit) if limit else S.n
    pi = torch.as_tensor(S.pi, device=dev)
    ref = dict(np.load(B.root() / "plans" / f"{d}.npz"))
    assert (ref["names"][:n] == S.tab["names"][:n]).all() and list(ref["models"]) == list(models)
    nat = np.zeros((n, len(models), 206), np.float32)
    hid = np.zeros((n, len(models), 1024), np.float16)
    err = np.zeros((n, len(models)), np.float32)
    for m, model in enumerate(models.values()):
        stash, orig = {}, model.net.run_batched

        def tap(feeds, want, **kw):
            o = orig(feeds, A.POLICY_OUT, **kw)
            stash.update(o)
            return {k: o[k] for k in want}
        model.net.run_batched = tap
        sl = model.net.slices
        cols = torch.as_tensor(np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in SLICES]), device=dev)
        try:
            with torch.no_grad():
                for i in range(0, n, batch):
                    r = torch.arange(i, min(i + batch, n), device=dev)
                    o = model(S.front[r], S.ego[r], S.tc[r], None, None).float()
                    p = o[:, pi].view(-1, 33, 15)
                    own = torch.stack(T.rear(p, S.cam_x[r], W), -1).cpu().numpy()
                    e = np.abs(own - ref["own"][m, i:i + len(r)])
                    err[i:i + len(r), m] = e.max((1, 2))
                    nat[i:i + len(r), m] = o[:, cols].cpu().numpy()
                    hid[i:i + len(r), m] = torch.cat([stash["select_4"].reshape(len(r), -1), stash["mean"].reshape(len(r), -1)], 1).float().cpu().numpy()
        finally:
            model.net.run_batched = orig
    return dict(names=S.tab["names"][:n], nat=nat, hid=hid, err=err)


def main(a):
    import torch
    import pp_train as T
    from experiments.op_adapt_r2.lib import op_adapt_r2 as R2
    from jevdrive import cache
    from jevdrive.data import splits
    from jevdrive.run import Run
    dev = torch.device("cuda")
    units = [(f, k) for f in a.fams for k in (a.shards if a.shards is not None else range(B.NSH))]
    with Run("long_contact", "native" + (f"-first{a.limit}" if a.limit else ""), config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtrain"))
        models = {t: T.load_pmodel(t, dev) for t in MODELS}
        W = torch.as_tensor(R2.t_weights(T.T8), device=dev)
        out = LC.root() / ("native" + (f"-first{a.limit}" if a.limit else ""))
        out.mkdir(parents=True, exist_ok=True)
        t0, n, worst = time.time(), 0, []
        for f, k in run.tqdm(units, desc="dirs"):
            d = B.cdir(f, k)
            key = cache.key(params=dict(models=list(MODELS), limit=a.limit, slices=list(SLICES)),
                            inputs=[B.cache_root() / f"{d}@warp" / "front.npy", B.cache_root() / d / "tab.npz", B.root() / "plans" / f"{d}.npz"]
                            + [T.proot("runs", t) / "ckpt-final.pt" for t in MODELS], code=[forward])
            z = cache.cached(out / f"{d}.npz", key, lambda: forward(d, models, W, dev, a.limit, a.batch), force=a.force)
            n += len(z["names"])
            worst.append(dict(dir=d, max_err=float(z["err"].max()), frac_over=float((z["err"] > TOL).mean())))
            run.info("%s: %d rows, plan identity max %.4f, > %.2f on %.5f of rows", d, len(z["names"]), worst[-1]["max_err"], TOL, worst[-1]["frac_over"])
        frac = float(np.mean([w["frac_over"] for w in worst]))
        run.summary.update(dirs=len(units), rows=n, compute_s=time.time() - t0, max_err=max(w["max_err"] for w in worst), frac_over_tol=frac,
                           vram_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
        assert frac < 1e-3, f"the forward does not reproduce the stored own plans: {worst}"


if __name__ == "__main__":
    from jevdrive.run import cli_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fams", nargs="+", default=["log", "ot1", "yr1"])
    ap.add_argument("--shards", type=int, nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    cli_args(ap)
    main(ap.parse_args())
