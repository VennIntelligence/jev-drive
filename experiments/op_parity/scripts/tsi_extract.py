"""op_parity turn selector input (plans/2026-10-08-turn-selector-input-prereg.md): SH30's own policy hidden state on lb_navtest.

  python tsi_extract.py --spec SH30-F-s0@warp --ref <stored plans npz>

Same loop, batch and fp16 path as sc_infer.py (plans bit-identical to the stored ones, checked against --ref); additionally taps the policy's
`select_4` and `mean` (512 each) by wrapping the ONNX runner. Output: $DATA_DIR/runs/op_parity/turn_selinput/hidden/<spec>__lb_navtest.npz
(names, select_4 (N, 512) fp16, mean (N, 512) fp16, plan_pos). GPU job: submit through the pool (jevdrive.cl submit).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse  # noqa: E401,E402

import numpy as np  # noqa: E402

from jevdrive.run import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--data", default="lb_navtest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ref", default="", help="stored plans npz of the same model (sc_infer output); plan_pos must agree on rows with speed >= 0.5 m/s")
    a = ap.parse_args()
    import torch
    from jevdrive import op_adapt as A
    from jevdrive.bench import navsim as N
    from jevdrive.bench.models import data_dir, resolve
    out = data_dir() / "runs/op_parity/turn_selinput/hidden" / f"{a.spec.replace('@', '-').replace(':', '_')}__{a.data}{'-first%d' % a.limit if a.limit else ''}.npz"
    with Run("op_parity", f"turn_selinput/extract-{a.spec}", seed=0, config=vars(a)) as run:
        N._pp_path()
        import pp_train as T
        m = resolve(a.spec, check=True)
        dev = torch.device("cuda")
        S = T.Store([a.data], dev, need_side=(N.cache_dir(a.data, "gimm") / "side.npy").exists(), frames=m.frames)
        names = S.tab["names"]
        model = T.load_pmodel(m.name, dev) if not m.ckpt else N._load_ckpt(T, m.ckpt, dev)
        assert getattr(model, "mem", None) is None and not m.opt, "plain models only"
        stash, orig = {}, model.net.run_batched

        def tap(feeds, want, **kw):
            o = orig(feeds, A.POLICY_OUT, **kw)
            stash.update(o)
            return {k: o[k] for k in want}
        model.net.run_batched = tap
        sl = model.net.slices
        n = min(S.n, a.limit) if a.limit else S.n
        pi = np.arange(sl["plan"].start, sl["plan"].start + 495)
        mu = np.zeros((n, 33, 15), np.float32)
        h4, hm = np.zeros((n, 512), np.float16), np.zeros((n, 512), np.float16)
        with torch.no_grad():
            for i in range(0, n, a.batch):
                r = torch.arange(i, min(i + a.batch, n), device=dev)
                side = S.side[r] if S.side is not None else None
                o = model(S.front[r], S.ego[r], S.tc[r], side, None).float().cpu().numpy()
                j = slice(i, i + len(r))
                mu[j] = o[:, pi].reshape(-1, 33, 15)
                h4[j] = stash["select_4"].reshape(len(r), -1).float().cpu().numpy()
                hm[j] = stash["mean"].reshape(len(r), -1).float().cpu().numpy()
                run.info(f"{i + len(r)}/{n}")
        res = dict(names=names[:n], select_4=h4, mean=hm, plan_pos=mu[:, :, 0:3])
        if a.ref:
            z = np.load(a.ref)
            idx = {t: k for k, t in enumerate(z["names"].tolist())}
            rows = np.array([idx[t] for t in names[:n]])
            ok = S.tb["speed"][:n] >= 0.5
            d = np.abs(z["plan_pos"][rows][ok] - mu[:, :, 0:3][ok]).max()
            run.summary.update(ref_max_plan_diff_m=float(d), ref_rows=int(ok.sum()))
            run.info(f"vs stored plans: max |dplan_pos| {d:.4g} m over {int(ok.sum())} rows")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out.with_suffix(".tmp.npz"), **res)
        out.with_suffix(".tmp.npz").rename(out)
        run.summary.update(out=str(out), n=int(n))
        if a.ref:
            assert d < 0.05, f"re-run plan differs from the stored plan by {d} m (gate G3)"


if __name__ == "__main__":
    main()
