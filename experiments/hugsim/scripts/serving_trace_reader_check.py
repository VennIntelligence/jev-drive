#!/usr/bin/env python
"""Second-reader check of experiments/body1/results/served_plan_length.md item 7 (the offline frame-source table), and two controls it
did not run (experiments/hugsim/results/serving_trace.md). CPU fp32 forwards of one parity checkpoint on cached front tokens.

  table     the checkpoint on the same navtest tokens under three lattice sources: warp (lb_navtest@warp, trained and scored), real 10 Hz
            (lb_hq_navtestX@real), GIMM (lb_navtest): arc / log, ADE vs log, real / warp by speed          (the note's table)
  pipeline  the real tokens come from another data dir (lb_hq_navtestX) than the warp tokens (lb_navtest). Control: the 2 Hz `keys` protocol
            exists for both dirs and uses no lattice frame at all, so hq@keys against navtest@keys isolates the data dir
  fill      a stream server (HUGSIM) fills all 9 policy slots with frame pairs; the training rows have 8 slots, the oldest zeroed and the next
            one made from a (zero image, frame) pair. Stand-in for a filled queue on the cached rows: the token of slot -1.2 s copied into the
            zero slot (fill9), and also over the (zero, frame) slot -1.4 s (fill9b). Not the true older frames: an estimate of the sensitivity

  $DATA_DIR/envs/op-train/bin/python experiments/hugsim/scripts/serving_trace_reader_check.py --n 600 --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_R / "experiments/op_parity/scripts")]
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from jevdrive.common import data_dir, n_cpus  # noqa: E402
from jevdrive.data import splits  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402


def arc(p):
    p = np.asarray(p, float)[..., :2]
    p = np.concatenate([np.zeros(p.shape[:-2] + (1, 2)), p], -2)
    return np.linalg.norm(np.diff(p, axis=-2), axis=-1).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="P2H10-F-s0")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", required=True)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("hugsim_serving_trace", "reader", seed=a.seed, config=vars(a), resume=a.resume) as run:
        import torch
        import pp_train as T
        from jevdrive import navsim_zs as Z, op_interp as I
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        c = data_dir() / "runs/op_parity/cache"
        tx, tn = np.load(c / "lb_hq_navtestX/tab.npz"), np.load(c / "lb_navtest/tab.npz")
        F = {k: np.load(c / p / "front.npy", mmap_mode="r") for k, p in (("real", "lb_hq_navtestX@real"), ("warp", "lb_navtest@warp"), ("gimm", "lb_navtest"),
                                                                        ("keys_hq", "lb_hq_navtestX@keys"), ("keys_nt", "lb_navtest@keys"))}
        rown = {t: i for i, t in enumerate(tn["names"].tolist())}
        ix = np.array([k for k, t in enumerate(tx["names"].tolist()) if t in rown])
        sel = np.sort(ix[np.random.default_rng(0).permutation(len(ix))[:a.n]])        # the note's selection (script g, seed 0)
        rn = np.array([rown[t] for t in tx["names"][sel].tolist()])
        assert nt.mask(np.asarray(tn["names"])[rn]).all()
        torch.set_num_threads(min(12, n_cpus()))
        m = T.load_pmodel(a.tag, torch.device("cpu"))
        m.float()
        m.net.dtype = torch.float32
        s = m.net.slices["plan"].start

        def plans(front, tab, rows, edit=None, desc=""):
            res = []
            with torch.no_grad():
                for b in run.tqdm(range(0, len(rows), 64), desc=desc):
                    r = rows[b:b + 64]
                    tc = np.where(tab["lht"][r][:, None], [[0.0, 1.0]], [[1.0, 0.0]]).astype(np.float32)
                    f = torch.from_numpy(np.asarray(front[r], np.float32))
                    o = m(edit(f) if edit else f, torch.from_numpy(tab["ego"][r]), torch.from_numpy(tc)).float()
                    mu = o[:, s:s + 33 * 15].reshape(-1, 33, 15).numpy()
                    res += [I.to_rear(mu[j, :, 0:3], mu[j, :, 11], I.T_IDXS, tab["cam"][q, :2], Z.T_OUT, "lever") for j, q in enumerate(r)]
            return np.array(res)

        fill9 = lambda f: torch.cat([f[:, 1:2], f], 1)                               # noqa: E731  slot -1.6 s <- the token of slot -1.2 s
        fill9b = lambda f: torch.cat([f[:, 1:2], f[:, 1:2], f[:, 1:]], 1)            # noqa: E731  ... and slot -1.4 s (zero-image pair) too
        P = {"warp": plans(F["warp"], tn, rn, desc="warp"), "real": plans(F["real"], tx, sel, desc="real"), "gimm": plans(F["gimm"], tn, rn, desc="gimm"),
             "keys_hq": plans(F["keys_hq"], tx, sel, desc="keys hq"), "keys_nt": plans(F["keys_nt"], tn, rn, desc="keys navtest"),
             "warp_fill9": plans(F["warp"], tn, rn, fill9, "warp fill9"), "warp_fill9b": plans(F["warp"], tn, rn, fill9b, "warp fill9b"),
             "real_fill9b": plans(F["real"], tx, sel, fill9b, "real fill9b")}
        fut = tn["fut"][rn]
        ok = ~np.isnan(fut).any((1, 2))
        A = {k: arc(v)[ok] for k, v in P.items()}
        A["log"] = arc(fut)[ok]
        ade = lambda X, Y: float(np.linalg.norm(X[ok][..., :2] - Y[ok][..., :2], axis=-1).mean())   # noqa: E731
        res = {"tag": a.tag, "n": int(len(sel)), "n_ok": int(ok.sum()), "ego_tab_max_abs_diff": float(np.abs(tx["ego"][sel] - tn["ego"][rn]).max()),
               "arc_over_log": {k: float(A[k].sum() / A["log"].sum()) for k in P}, "ade_vs_log_m": {k: ade(P[k], fut) for k in P},
               "real_over_warp": float(A["real"].sum() / A["warp"].sum()), "gimm_over_warp": float(A["gimm"].sum() / A["warp"].sum()),
               "keys_hq_over_keys_nt": float(A["keys_hq"].sum() / A["keys_nt"].sum()), "keys_hq_vs_keys_nt_mean_dist_m": ade(P["keys_hq"], P["keys_nt"]),
               "real_vs_warp_mean_dist_m": ade(P["real"], P["warp"]),
               "fill9_over_warp": float(A["warp_fill9"].sum() / A["warp"].sum()), "fill9b_over_warp": float(A["warp_fill9b"].sum() / A["warp"].sum()),
               "fill9_vs_warp_mean_dist_m": ade(P["warp_fill9"], P["warp"]), "fill9b_vs_warp_mean_dist_m": ade(P["warp_fill9b"], P["warp"]),
               "real_fill9b_over_warp": float(A["real_fill9b"].sum() / A["warp"].sum())}
        bench = data_dir() / f"runs/bench/ol/lb_navtest/preds/{a.tag}-warp__base.npz"
        if bench.exists():
            res["warp_vs_bench_preds_max_m"] = float(np.abs(P["warp"] - np.load(bench)["poses"][rn]).max())
        v0 = tn["vel"][rn, -1, 0][ok]
        res["by_speed"] = []
        for lo, hi in ((-1, 1), (1, 3), (3, 6), (6, 10), (10, 99)):
            q = (v0 >= lo) & (v0 < hi)
            res["by_speed"].append(dict(lo=lo, hi=hi, n=int(q.sum()), **{f"{k}_over_warp": float(A[k][q].sum() / A["warp"][q].sum())
                                                                        for k in ("real", "gimm", "warp_fill9", "warp_fill9b", "real_fill9b")},
                                        keys_hq_over_keys_nt=float(A["keys_hq"][q].sum() / A["keys_nt"][q].sum())))
        (out / "reader_check.json").write_text(json.dumps(res, indent=1))
        run.info("%s", json.dumps(res))
        run.summary["out"] = str(out)


if __name__ == "__main__":
    main()
