"""turn-prior (plans/2026-10-08-turn-prior-prereg.md, results/turn_prior.md): what the decision-173 floor head (ego + command + plan descriptor -> one of the F20
candidates) does on the turn-intent frames of WOD-E2E val, whether it adds on top of WLG and of the rater-preference plans, and whether it is a fixed rule.
Candidate sets, re-timing, RFS scoring and the ridge head are s2_gohold / s2_thinhead / s2_thinhead_heads; all CPU.

  prep   (jevdrive env)  F20 candidates + candidate RFS of every plan set (WP2, WLG, pref top / f20) -> $DATA_DIR/runs/op_parity/turn_prior/sets.npz
  desc   (jevdrive env)  Step 1: the floor head's picks on WP2 by v0 stratum x side, and the overlap with WLG   -> results/turn_prior/desc_*.csv
  stack  (jevdrive env)  Step 2: the floor head refitted on WLG's candidates and on top of the preference plans -> results/turn_prior/stack_*.csv
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import s2_thinhead as T  # noqa: E402
import s2_thinhead_heads as H  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

G = T.G
OUT, FIG = _R / "experiments/op_parity/results/turn_prior", _R / "experiments/op_parity/figs/turn_prior"
RUNS = data_dir() / "runs/op_parity/turn_prior"
KEEP, S, FOLDS, REPS, INNER = H.KEEP, 2, H.FOLDS, H.REPS, H.INNER
SETS = ("wp2", "wlg", "top", "f20")
V_STRATA = (("v0<0.5", 0.0, 0.5), ("0.5<=v0<3", 0.5, 3.0), ("v0>=3", 3.0, 1e9))


# ---------------------------------------------------------------- data
def ctx():
    import wod_launch_report as R
    C = R.Ctx()
    z = np.load(T.RUNS / "data.npz")
    assert (z["va_names"] == C.names[: C.n]).all()
    return C, z


def plans_of(C, name):
    if name == "wp2":
        return [C.preds(t) for t in T.ARMS]
    z = np.load(_R / "experiments/op_parity/results/wod_pref/plans.npz")
    assert (z["names"] == C.names[: C.n]).all()
    return [z[f"{name}_s{s}"].astype(np.float64) for s in range(S)]


def build(C, P, cb):
    """Plans (2 x (n, 20, 2)) -> candidates F (2, n, 20, 20, 2), candidate RFS J (2, 20, n), plan descriptor (2, n, 15)."""
    F = np.stack([G.factored(p, C.v0, 3, cb) for p in P])
    J = np.stack([np.stack([C.rfs(np.ascontiguousarray(f[:, m])) for m in range(20)]) for f in F])
    assert np.abs(J[:, KEEP] - np.stack([C.rfs(p) for p in P])).max() < 1e-9
    return F.astype(np.float32), J, np.stack([T.plan_desc(p, C.v0, cb) for p in P]).astype(np.float32)


def cmd_prep(a):
    from jevdrive.run import Run
    with Run("op_parity", "turn-prior-prep", seed=0, config=vars(a)) as run:
        C, z = ctx()
        cb = T.codebook()
        out = {}
        for k in SETS:
            t0 = time.time()
            F, J, pd_ = build(C, plans_of(C, k), cb)
            out |= {f"F_{k}": F, f"J_{k}": J, f"pd_{k}": pd_}
            run.info("%s: RFS %.3f, F20 oracle d %+.3f (%.0f s)", k, C.cm(J[:, KEEP].mean(0)), C.cm(J.max(1).mean(0) - J[:, KEEP].mean(0)), time.time() - t0)
        assert np.abs(out["J_wp2"] - z["va_J"]).max() < 1e-6 and np.abs(out["pd_wp2"] - z["va_plan"]).max() < 1e-4
        out["J_shipped"] = C.rfs(C.preds("shipped"))
        RUNS.mkdir(parents=True, exist_ok=True)
        np.savez(RUNS / "sets.npz", **out)


def load_sets(C, z):
    d = dict(np.load(RUNS / "sets.npz"))
    H._D.update({"va_ego": z["va_ego"], "va_seq": z["va_seq"]})
    import pandas as pd
    H._D["scode"] = pd.factorize(pd.Series(z["va_seq"].astype(str)))[0]
    return d


# ---------------------------------------------------------------- the head, generalised: fit on one candidate set, apply to another
def X_of(pd_, ego):
    return np.concatenate([np.broadcast_to(ego, (S, *ego.shape)), pd_], 2).astype(np.float64)       # (S, n, 35): ego then plan, as H.X_val(["ego", "plan"])


def pick_of(pred):
    p = pred.copy()
    p[:, :, KEEP] = 0.0
    return np.stack([T.first(p[s].T) for s in range(S)])                                             # (S, m)


def realised(J, pk, idx):
    return np.mean([J[s][pk[s], idx] - J[s][KEEP, idx] for s in range(S)], 0)


def fit_apply(Xf, Jf, tr, inner, Xa=None):
    """Ridge of the candidates' RFS gain (fit set Jf, frames tr; lam by the inner folds on tr) -> a predictor for inputs (S, m, d)."""
    Y = (Jf - Jf[:, KEEP:KEEP + 1]).transpose(0, 2, 1)
    best, lam = -np.inf, None
    acc = np.zeros(len(H.LAMS))
    d = Xf.shape[2]
    for g in range(INNER):
        a_, b_ = tr[inner[tr] != g], tr[inner[tr] == g]
        P = H.ridge_fit(Xf[:, a_].reshape(-1, d), Y[:, a_].reshape(-1, 20))
        for li, l in enumerate(H.LAMS):
            pk = pick_of(P(Xf[:, b_].reshape(-1, d), l).reshape(S, len(b_), 20))
            acc[li] += realised(Jf, pk, b_).sum()
    for li in reversed(range(len(H.LAMS))):                                                          # ties to the larger lam, as H.select_fit
        if acc[li] > best + 1e-9:
            best, lam = acc[li], H.LAMS[li]
    P = H.ridge_fit(Xf[:, tr].reshape(-1, d), Y[:, tr].reshape(-1, 20))
    return lambda X: P(X.reshape(-1, d), lam).reshape(S, -1, 20), lam


def oof(Xf, Jf, Xa, Ja, fold, rep, inner_salt=1000):
    """Out-of-fold: for each fold j fit on the other frames (set f), pick on the frames of fold j (set a). -> per-frame gain over keep (n,), picks (S, n), lams."""
    n = Xf.shape[1]
    gain, picks, lams = np.zeros(n), np.zeros((S, n), int), []
    for j in range(fold.max() + 1):
        te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
        pred, lam = fit_apply(Xf, Jf, tr, H.folds(rep * 10 + j, INNER, inner_salt))
        pk = pick_of(pred(Xa[:, te]).reshape(S, len(te), 20))
        gain[te], picks[:, te] = realised(Ja, pk, te), pk
        lams.append(lam)
    return gain, picks, lams


def ref_unit(a):
    """The decision-173 unit (E arm, WP2), for the reproduction gate."""
    return H.unit(a)[1]


# ---------------------------------------------------------------- stats helpers
def pair(C, d, rows=None):
    c = C.ci(d, rows)
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def strata(C):
    n, v0, it = C.n, C.v0, C.intent
    st = {"all": np.ones(n, bool), "turn-intent": it >= 2, "left": it == 2, "right": it == 3, "straight": it < 2}
    for nm, lo, hi in V_STRATA:
        m = (v0 >= lo) & (v0 < hi)
        st[nm] = m
        st[f"turn {nm}"] = m & (it >= 2)
        st[f"straight {nm}"] = m & (it < 2)
    return st


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "desc", "stack", "rule", "figs"):
        p = sp.add_parser(c)
        p.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
