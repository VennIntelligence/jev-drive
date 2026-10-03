"""Decision-92 history-rotation probe at small yaw rates (experiments/hugsim/plans/2026-10-04-launch-lean-prereg.md, part 1).

Port forward (h_train.plans_of) on the probe pools pnav / pwod / pcarla, stop / low / mid bins, at most --per-bin samples per
bin (fixed seed); variants normal + rot at +-RATES deg/s (op_adapt_h.hist_rot, t0 frame unchanged). Per model and sample:
G(w) = (psi3(+w) - psi3(-w)) / 2 (deg, plan heading at 3 s), gain g(w) = G / (1.5 w).

    CUDA_VISIBLE_DEVICES=2 $DATA_DIR/envs/op-train/bin/python experiments/op_adapt_h/scripts/h_rate_probe.py run \
        --models O pilot-s0 it_dw3-s0                          -> $H/rate_probe/plans_<dom>.npz
    ... h_rate_probe.py table                                  -> $H/rate_probe/{g_rate.csv, ratio.csv}
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("", "scripts", "experiments/op_adapt_r2/lib",
                                                                              "experiments/op_adapt_h/scripts")]
import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import h_train as T  # noqa: E402
from experiments.op_adapt_h.lib import op_adapt_h as H  # noqa: E402
from jevdrive import stats  # noqa: E402

RATES = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
VARS = [("normal", None)] + [("rot", s * r) for r in RATES for s in (1.0, -1.0)]
BINS = ("stop", "low", "mid")
OUT = H.hroot("rate_probe")


def rows_of(S, per_bin, seed=0):
    rng = np.random.default_rng([seed, T.PROBE.index(S.dom)])
    out = []
    for b in BINS:
        r = np.flatnonzero(S.t["bin"] == b)
        out.append(np.sort(r[rng.permutation(len(r))[:per_bin]]))
    return np.concatenate(out)


def cmd_run(a):
    dev = torch.device("cuda")
    models = T.load_models(a.models, dev)
    with ProcessPoolExecutor(a.workers) as ex:
        for d in T.PROBE:
            p = OUT / f"plans_{d}.npz"
            have = dict(np.load(p)) if p.exists() else {}
            todo = {k: m for k, m in models.items() if f"plan_{k}" not in have}
            if not todo:
                continue
            S = H.Samples(d)
            rows = rows_of(S, a.per_bin)
            jobs = [(d, int(i), k, v) for k, v in VARS for i in rows]
            t0 = time.time()
            P = T.plans_of(todo, jobs, dev, ex, bank=f"rate_{d}_{a.per_bin}")
            have |= {f"plan_{k}": P[k].reshape(len(VARS), len(rows), 33, 15) for k in todo}
            have["rows"] = rows
            np.savez(p, **have)
            print(f"rate probe {d}: {len(rows)} x {len(VARS)} x {len(todo)} models in {time.time() - t0:.0f} s", flush=True)


def cmd_table(a):
    rec, rat = [], []
    for d in T.PROBE:
        z = np.load(OUT / f"plans_{d}.npz")
        S = H.Samples(d)
        rows = z["rows"]
        b, cl = S.t["bin"][rows], S.t["cluster"][rows]
        G = {}
        for k in a.models:
            pl = z[f"plan_{k}"]
            psi = np.stack([H.psi3(pl[j]) for j in range(len(VARS))])
            for r in RATES:
                jp, jm = VARS.index(("rot", r)), VARS.index(("rot", -r))
                G[(k, r)] = (psi[jp] - psi[jm]) / 2
            G[(k, 0)] = psi[0]
        for bn in BINS + ("low+stop",):
            sel = np.isin(b, ("stop", "low")) if bn == "low+stop" else b == bn
            for k in a.models:
                for r in RATES:
                    g = G[(k, r)][sel]
                    ci = stats.paired(g, np.zeros_like(g), groups=cl[sel])
                    rec.append(dict(domain=d, bin=bn, model=k, rate=r, n=int(sel.sum()), G=float(g.mean()), lo=ci["lo"], hi=ci["hi"],
                                    gain=float(g.mean() / (1.5 * r))))
                if k == "O":
                    continue
                for r in RATES:                                  # ratio of mean G (adapted / shipped), cluster bootstrap
                    ga, go, c = G[(k, r)][sel], G[("O", r)][sel], cl[sel]
                    u = np.unique(c)
                    idx = [np.flatnonzero(c == x) for x in u]
                    rng = np.random.default_rng(0)
                    bs = []
                    for _ in range(2000):
                        q = np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])
                        bs.append(ga[q].mean() / go[q].mean())
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    rat.append(dict(domain=d, bin=bn, model=k, rate=r, ratio=float(ga.mean() / go.mean()), lo=float(lo), hi=float(hi)))
    pd.DataFrame(rec).to_csv(OUT / "g_rate.csv", index=False)
    pd.DataFrame(rat).to_csv(OUT / "ratio.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(pd.DataFrame(rec).pivot_table(index=["domain", "bin", "model"], columns="rate", values="G").round(2))
        print(pd.DataFrame(rat).pivot_table(index=["domain", "bin", "model"], columns="rate", values="ratio").round(2))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--models", nargs="+", default=["O", "pilot-s0", "it_dw3-s0"])
    p.add_argument("--per-bin", type=int, default=250)
    p.add_argument("--workers", type=int, default=24)
    p = sp.add_parser("table")
    p.add_argument("--models", nargs="+", default=["O", "pilot-s0", "it_dw3-s0"])
    a = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
