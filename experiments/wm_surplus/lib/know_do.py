"""Knows vs does: do a driving model's frozen planner-input features encode what the plan should do, and does the plan do it?

Model-agnostic (numpy / sklearn / pandas; run in the repo venv). A subject supplies one npz per bench split:
  tokens (n,)            NAVSIM tokens
  X      (n, d)          pooled planner-input features (any fixed, training-free pooling of the probed tensor)
  B      (n, b)          the planner's non-image inputs (command, ego state, history): the blind baseline
and its plans as a bench pred file (tokens, poses (n, 8, 3)). Ground truth = the logged 4 s future, classed with the bench's
manoeuvre strata (`jevdrive.bench.tables.motion`), for both GT and plan. `knows_vs_does(...)` returns the per-class table.

Probes: multinomial logistic regression (standardised, L2, C from an inner log split), out-of-fold over whole-log folds, so a
probe never sees the logs it is scored on. Regression side: ridge on GT dyaw and lateral offset at 4 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CS = (0.01, 0.1, 1.0)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def log_folds(logs, k: int = 5) -> np.ndarray:
    """Fold id per token: sorted unique log names dealt i % k."""
    u = sorted(set(logs))
    f = {lg: i % k for i, lg in enumerate(u)}
    return np.array([f[lg] for lg in logs])


def _inner_pick(make, X, y, folds, grid, score):
    """Pick a hyper-parameter on an inner 3-fold log split of the training part."""
    best, bv = grid[0], -np.inf
    inner = folds % 3
    for g in grid:
        s = []
        for j in range(3):
            tr, va = inner != j, inner == j
            if len(np.unique(y[tr])) < 2 or va.sum() == 0:
                continue
            m = make(g).fit(X[tr], y[tr])
            s.append(score(m, X[va], y[va]))
        if s and np.mean(s) > bv:
            best, bv = g, float(np.mean(s))
    return best


def _clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _reg(a):
    return make_pipeline(StandardScaler(), Ridge(alpha=a))


def _acc(m, Xv, yv):
    return float((m.predict(Xv) == yv).mean())


def _negmse(m, Xv, yv):
    return -float(((m.predict(Xv) - yv) ** 2).mean())


def _outer(make, grid, score, X, y, folds, f):
    tr, te = folds != f, folds == f
    g = _inner_pick(make, X[tr], y[tr], folds[tr], grid, score)
    return f, g, make(g).fit(X[tr], y[tr]).predict(X[te])


def _oof(make, grid, score, X, y, folds, dtype):
    """Out-of-fold predictions; the outer folds run in parallel processes."""
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=len(np.unique(folds)))(delayed(_outer)(make, grid, score, X, y, folds, f) for f in np.unique(folds))
    out = np.empty(len(y), dtype=dtype)
    for f, _, p in res:
        out[folds == f] = p
    return out


def oof_classify(X, y, folds) -> np.ndarray:
    return _oof(_clf, CS, _acc, X, y, folds, object)


def oof_regress(X, y, folds) -> np.ndarray:
    return _oof(_reg, ALPHAS, _negmse, X, y, folds, float)


def r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def verdict(kd, kb) -> str:
    """Pre-registered lines (plans/2026-10-23-p1-knowdo-prereg.md). kd = knows - does, kb = knows - blind (paired dicts)."""
    if kd["mean"] >= 0.10 and kd["lo"] > 0:
        return "sees but does not do"
    if kb["mean"] < 0.05 and kb["hi"] < 0.10 and kb["mean_a"] < 0.6:
        return "does not see (features add nothing)"
    if abs(kd["mean"]) < 0.10:
        return "does what it sees"
    return "unclear"


def knows_vs_does(X, B, gt_cls, plan_cls, logs, gt_dyaw=None, gt_lat=None, plan_dyaw=None, plan_lat=None,
                  min_n: int = 50, k: int = 5) -> dict:
    """Per-class knows / blind / does recall with log-bootstrap CIs, plus balanced accuracies and the regression side."""
    from jevdrive import stats
    gt_cls, plan_cls, logs = np.asarray(gt_cls, object), np.asarray(plan_cls, object), np.asarray(logs)
    folds = log_folds(logs, k)
    XB = np.hstack([X, B])
    pred = {"blind": oof_classify(B, gt_cls, folds), "feat": oof_classify(X, gt_cls, folds), "feat+blind": oof_classify(XB, gt_cls, folds)}
    rows = []
    classes = [c for c, n in pd.Series(gt_cls).value_counts().items() if n >= min_n]
    for c in classes:
        m = gt_cls == c
        hit = {n: (p[m] == c).astype(float) for n, p in pred.items()}
        does = (plan_cls[m] == c).astype(float)
        kd = stats.paired(hit["feat+blind"], does, groups=logs[m])
        kb = stats.paired(hit["feat+blind"], hit["blind"], groups=logs[m])
        rows.append(dict(cls=c, n=int(m.sum()), knows=hit["feat+blind"].mean(), feat_only=hit["feat"].mean(), blind=hit["blind"].mean(),
                         does=does.mean(), knows_minus_does=kd["mean"], kd_lo=kd["lo"], kd_hi=kd["hi"],
                         knows_minus_blind=kb["mean"], kb_lo=kb["lo"], kb_hi=kb["hi"], verdict=verdict(kd, kb)))
    T = pd.DataFrame(rows)
    bal = {n: float(np.mean([(p[gt_cls == c] == c).mean() for c in classes])) for n, p in pred.items()}
    bal["plan"] = float(np.mean([(plan_cls[gt_cls == c] == c).mean() for c in classes]))
    reg = {}
    for name, gt, pl in (("dyaw_deg", gt_dyaw, plan_dyaw), ("lat_4s_m", gt_lat, plan_lat)):
        if gt is None:
            continue
        gt = np.asarray(gt, float)
        ok = np.isfinite(gt)
        reg[name] = {n: r2(gt[ok], oof_regress(Z[ok], gt[ok], folds[ok])) for n, Z in (("blind", B), ("feat+blind", XB))}
        if pl is not None:
            reg[name]["plan"] = r2(gt[ok], np.asarray(pl, float)[ok])
    return dict(table=T, balanced_accuracy=bal, regression_r2=reg, predictions=pred, folds=folds)
