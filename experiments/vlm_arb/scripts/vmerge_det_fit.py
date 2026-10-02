"""Trigger heads of the merged arm `vmerge` (plan 2026-10-03-vmerge.md): logistic regression on frozen Cinque's native
output vector (2066 values, decision 89) for "a traffic light / a stop sign governs a junction within 40 m ahead".

Trained on P4 only (152 Bench2Drive training routes, none of the 19 evaluation routes); route-grouped 5-fold out-of-fold
scores on P4 fix the threshold (recall 0.90); the closed-loop frames (CL, other rig timing, includes the evaluation routes)
are only scored as a transfer check. Output: $DATA_DIR/runs/vlm_arb_vmerge/det_head.npz (mu, sd, W [2, 2066], b [2],
thr [2], names) and results/vmerge_det.json.

  .venv/bin/python experiments/vlm_arb/scripts/vmerge_det_fit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stoppos_data import DATA, PAST_JUNC, have_features, load_feats, load_frames  # noqa: E402

OUT = DATA / "runs/vlm_arb_vmerge"
RES = HERE.parent / "results"
NAMES = ("light", "sign")
C, RECALL = 0.1, 0.90


def targets(f):
    near = f.d_junc.between(PAST_JUNC, 40.0)
    return np.stack([(near & (f.has_light == 1)).to_numpy(int), (near & (f.has_sign == 1)).to_numpy(int)], 1)


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return mu, sd, m.coef_[0], m.intercept_[0]


def score(p, X):
    mu, sd, w, b = p
    return 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))


def main():
    from sklearn.metrics import roc_auc_score
    f = load_frames()
    X = load_feats(f, "native")                    # aligned to the full frame table (streams are indexed whole)
    ok = (f.has_label & have_features(f)).to_numpy() & np.isfinite(X).all(1)
    X = X.astype(np.float64)
    f, X = f[ok].reset_index(drop=True), X[ok]
    Y = targets(f)
    p4, cl = (f.src == "p4").to_numpy(), (f.src == "cl").to_numpy()
    routes = f.route[p4].unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(routes, rng.permutation(len(routes)) % 5))
    fk = f.route.map(fold).to_numpy()
    rep, W, B, thr, mus, sds = {}, [], [], [], [], []
    for j, n in enumerate(NAMES):
        oof = np.full(len(f), np.nan)
        for k in range(5):
            tr, te = p4 & (fk != k), p4 & (fk == k)
            oof[te] = score(fit(X[tr], Y[tr, j]), X[te])
        pos = np.sort(oof[p4 & (Y[:, j] == 1)])
        t = float(pos[int((1 - RECALL) * len(pos))])
        full = fit(X[p4], Y[p4, j])
        s_cl = score(full, X[cl])
        rep[n] = dict(p4_oof_auc=round(roc_auc_score(Y[p4, j], oof[p4]), 4), thr=round(t, 5),
                      p4_oof_recall=round(float((oof[p4 & (Y[:, j] == 1)] >= t).mean()), 4),
                      p4_oof_fpr=round(float((oof[p4 & (Y[:, j] == 0)] >= t).mean()), 4),
                      cl_auc=round(roc_auc_score(Y[cl, j], s_cl), 4) if Y[cl, j].any() else None,
                      cl_recall=round(float((s_cl[Y[cl, j] == 1] >= t).mean()), 4) if Y[cl, j].any() else None,
                      cl_fpr=round(float((s_cl[Y[cl, j] == 0] >= t).mean()), 4),
                      n_p4=int(p4.sum()), n_p4_pos=int(Y[p4, j].sum()), n_cl=int(cl.sum()), n_cl_pos=int(Y[cl, j].sum()))
        mus.append(full[0]), sds.append(full[1]), W.append(full[2]), B.append(full[3]), thr.append(t)
        print(n, rep[n], flush=True)
    assert np.allclose(mus[0], mus[1]) and np.allclose(sds[0], sds[1])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "det_head.npz", mu=mus[0].astype(np.float32), sd=sds[0].astype(np.float32), W=np.stack(W).astype(np.float32),
             b=np.asarray(B, np.float32), thr=np.asarray(thr, np.float32), names=np.asarray(NAMES))
    (RES / "vmerge_det.json").write_text(json.dumps(dict(C=C, recall_target=RECALL, tap="native", train="p4", heads=rep), indent=1) + "\n")


if __name__ == "__main__":
    main()
