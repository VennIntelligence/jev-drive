"""op_parity self-consist (plans/2026-10-08-self-consist-prereg.md): inference-only pass that also stores the road-edge and lead heads.

  python sc_infer.py --spec SH30-F-s0@warp --data lb_navtest [--limit 64] [--ref <stored plans npz>]

Same loop, batch and fp16 path as jevdrive.bench.navsim.parity_plans (plans are bit-identical to the stored ones, checked against --ref);
additionally writes road_edges (N, 2, 33, 2) mean and std ([y, z] at X_IDXS, [left, right]) and lead_prob / lead_x / lead_v.
Output: $DATA_DIR/runs/op_parity/self_consist/infer/<spec>__<data>.npz
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.run import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--data", required=True, help="pp_prep cache name: lb_navtest | navtrain_full.s0of12")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ref", default="", help="stored plans npz of the same model; plan_pos must agree (rows with speed >= 0.5 m/s)")
    a = ap.parse_args()
    import torch
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import data_dir, resolve
    out = data_dir() / "runs/op_parity/self_consist/infer" / f"{a.spec.replace('@', '-').replace(':', '_')}__{a.data}{'-first%d' % a.limit if a.limit else ''}.npz"
    with Run("self_consist", f"infer-{a.spec}-{a.data}", seed=0, config=vars(a)) as run:
        N._pp_path()
        import pp_train as T
        m = resolve(a.spec, check=True)
        dev = torch.device("cuda")
        S = T.Store([a.data], dev, need_side=(N.cache_dir(a.data, "gimm") / "side.npy").exists(), frames=m.frames)
        names = S.tab["names"]
        model = T.load_pmodel(m.name, dev) if not m.ckpt else N._load_ckpt(T, m.ckpt, dev)
        assert getattr(model, "mem", None) is None and not m.opt, "plain models only"
        sl = model.net.slices
        n = min(S.n, a.limit) if a.limit else S.n
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        mu = np.zeros((n, 33, 15), np.float32)
        re_mu, re_sd = np.zeros((n, 2, 33, 2), np.float32), np.zeros((n, 2, 33, 2), np.float32)
        lp, lx, lv = (np.full(n, np.nan, np.float32) for _ in range(3))
        rs = sl["road_edges"]
        half = (rs.stop - rs.start) // 2
        assert half == 132, (rs, half)
        with torch.no_grad():
            for i in range(0, n, a.batch):
                r = torch.arange(i, min(i + a.batch, n), device=dev)
                side = S.side[r] if S.side is not None else None
                o = model(S.front[r], S.ego[r], S.tc[r], side, None).float().cpu().numpy()
                j = slice(i, i + len(r))
                mu[j] = o[:, pi].reshape(-1, 33, 15)
                e = o[:, rs]
                re_mu[j] = e[:, :half].reshape(-1, 2, 33, 2)
                re_sd[j] = np.exp(np.minimum(e[:, half:], 11)).reshape(-1, 2, 33, 2)
                ld = o[:, sl["lead"]]
                ld = ld[:, : ld.shape[1] // 2].reshape(-1, 3, 6, 4)
                lp[j] = 1 / (1 + np.exp(-np.clip(o[:, sl["lead_prob"]][:, 0], -11, None)))
                lx[j], lv[j] = ld[:, 0, 0, 0], ld[:, 0, 0, 2]
                run.info(f"{i + len(r)}/{n}")
        res = dict(names=names[:n], plan_pos=mu[:, :, 0:3], plan_yaw=mu[:, :, 11], road_edges=re_mu, road_edges_std=re_sd,
                   lead_prob=lp, lead_x=lx, lead_v=lv)
        if a.ref:
            z = np.load(a.ref)
            idx = {t: k for k, t in enumerate(z["names"].tolist())}
            rows = np.array([idx[t] for t in names[:n]])
            ok = S.tb["speed"][:n] >= 0.5
            d = np.abs(z["plan_pos"][rows][ok] - mu[:, :, 0:3][ok]).max()
            dl = np.nanmax(np.abs(z["lead_x"][rows][ok] - lx[ok]))
            run.summary.update(ref_max_plan_diff_m=float(d), ref_max_lead_x_diff=float(dl), ref_rows=int(ok.sum()))
            run.info(f"vs stored plans: max |dplan_pos| {d:.4g} m, max |dlead_x| {dl:.4g} over {int(ok.sum())} rows")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), **res)
        out.with_suffix(".tmp.npz").rename(out)
        run.summary.update(out=str(out), n=int(n))
        if a.ref:
            assert d < 0.05, f"re-run plan differs from the stored plan by {d} m"


if __name__ == "__main__":
    main()
