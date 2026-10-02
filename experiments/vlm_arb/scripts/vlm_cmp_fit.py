"""Linear heads on the cached hidden states of the vlm_cmp study: at which layer is each label linearly readable?

  vlm_cmp_fit.py fit --run RUN --model 4b --res r1153 [--dir-too]

Same head as vlm_thin_fit.py (multinomial logistic regression on standardised features, class-balanced, L2, LBFGS full batch,
lambda in {1e-3, 1e-2, 1e-1, 1, 10} chosen by validation balanced accuracy, ties to the larger lambda), same registered route
split (b2d/vlm-thin-{train,val,test}). Supplementary: 7-fold route-grouped CV over all 21 routes (fold = rank of the route id
modulo 7; route 17280 moved to fold 3 so that the two stop-sign routes 9196 and 17280 sit in different folds), lambda of
the primary split.

Features: `ans_<q>` = hidden state at the answer position of question q after layer N (raw residual, N = last layer is the
normed one); `pool` = mean over the image tokens of each camera at layer N (both cameras concatenated, independent of the
question). Labels (rows with label -1 are neither trained on nor scored):
  light  0 ego red / yellow within -5..50 m, 1 ego green within -5..50 m, 2 no ego light (a light of another approach may show)
  sign   1 stop sign within 25 m, 0 none within 80 m (25-80 m ignored)
  block  0 clear, 1 moving_lead, 2 static_block  (the log's Q_block truth)
  lead   1 a lead vehicle in the lane that is stopped (< 0.5 m/s), 0 one that moves; rows without a lead ignored
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_cmp_frames import SPLIT  # noqa: E402
from vlm_thin_fit import fit_lin, probs, standardise  # noqa: E402

LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_FOLDS = 7
FOLD_OVERRIDE = {"17280": 3}
LABEL_FEATS = {                                            # (feature source, labels read from it)
    "ans_light": ["light"], "ans_sign": ["sign"], "ans_block": ["block", "lead"], "pool": ["light", "sign", "block", "lead"],
    "ans_dir": ["light", "sign", "block", "lead"]}
NCLS = dict(light=3, sign=2, block=3, lead=2)


def labels(df):
    y = {}
    y["light"] = np.where(df.ego_red, 0, np.where(df.ego_green, 1, np.where(df.tl == -1, 2, -1)))
    y["sign"] = np.where(df.sign_near, 1, np.where(df.stop_dist.isna(), 0, -1))
    y["block"] = df.block.map({"clear": 0, "moving_lead": 1, "static_block": 2}).fillna(-1).astype(int).to_numpy()
    y["lead"] = np.where(df.lead_state == "stopped", 1, np.where(df.lead_state == "moving", 0, -1))
    return y


def load_feats(run, model, res, df):
    d = Path(run) / "features" / model / res
    parts = [np.load(p, allow_pickle=False) for p in sorted(d.glob("shard*-chunk*.npz"))]
    keys = [k for k in parts[0].files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    order = pd.Series(range(len(out["ids"])), index=out["ids"]).loc[df.id].to_numpy()
    return {k: v[order] for k, v in out.items()}


def bacc(pred, y, ncls):
    """Balanced accuracy over the classes present in y, and the per-class recalls (nan when absent)."""
    rec = [float((pred[y == c] == c).mean()) if (y == c).any() else np.nan for c in range(ncls)]
    return float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan, rec


def feature(F, src, N, layers):
    if src == "pool":
        return F["pool"][:, ([0] + layers).index(N)].reshape(len(F["pool"]), -1).astype(np.float32)
    q = src.split("_")[1]
    return F["h_" + q][:, layers.index(N)].astype(np.float32)


def fit(a):
    import torch
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    df = pd.read_csv(Path(a.run) / "frames.csv", dtype={"route": str, "attempt": str}, keep_default_na=False, na_values=[""])
    F = load_feats(a.run, a.model, a.res, df)
    nl = F["h_light"].shape[1]
    layers = list(range(2, 2 * nl + 1, 2))
    Y = labels(df)
    part = df.part.to_numpy()
    routes = sorted(df.route.unique(), key=int)
    fold = df.route.map({r: FOLD_OVERRIDE.get(r, i % N_FOLDS) for i, r in enumerate(routes)}).to_numpy()
    rows, preds = [], {}
    srcs = [s for s in LABEL_FEATS if s != "ans_dir" or a.dir_too]
    for src in srcs:
        Ns = ([0] + layers) if src == "pool" else layers
        for N in Ns:
            X = feature(F, src, N, layers)
            for lab in LABEL_FEATS[src]:
                y, ncls = Y[lab], NCLS[lab]
                tr, va, te = (part == "train") & (y >= 0), (part == "val") & (y >= 0), (part == "test") & (y >= 0)
                row = dict(model=a.model, res=a.res, src=src, layer=N, label=lab, n_train=int(tr.sum()), n_val=int(va.sum()),
                           n_test=int(te.sum()), train_classes=[int((y[tr] == c).sum()) for c in range(ncls)],
                           test_classes=[int((y[te] == c).sum()) for c in range(ncls)])
                ok_tr = len(set(y[tr])) == ncls
                key = "%s|%d|%s" % (src, N, lab)
                preds["tr|" + key] = np.full(len(df), -1, np.int8)
                if not ok_tr:                                       # a class missing from the train routes: no head on the registered split
                    row.update(lam=None, val_bacc=None, test_bacc=None, test_rec=None)
                    lam = 1.0
                else:
                    Xs, mu, sd = standardise(torch, X[tr], X, dev)
                    ytt = torch.tensor(y, device=dev)
                    trt = torch.tensor(tr, device=dev)
                    best = None
                    for lam in LAMS:
                        m = fit_lin(torch, Xs[trt], ytt[trt], lam)
                        p = probs(torch, m, Xs).argmax(-1).cpu().numpy()
                        vb = bacc(p[va], y[va], ncls)[0] if va.any() and len(set(y[va])) > 1 else np.nan
                        sc = -1 if not np.isfinite(vb) else vb
                        if best is None or sc > best[0] + 1e-9 or (abs(sc - best[0]) <= 1e-9 and lam > best[1]):
                            best = (sc, lam, p, vb)
                    lam = best[1]
                    preds["tr|" + key] = best[2].astype(np.int8)
                    tb, trec = bacc(best[2][te], y[te], ncls) if te.any() else (np.nan, [np.nan] * ncls)
                    row.update(lam=lam, val_bacc=best[3], test_bacc=tb, test_rec=trec)
                    if lab == "light":
                        p = best[2]
                        r_ = lambda m_, c_: float((p[m_] == c_).mean()) if m_.any() else np.nan   # noqa: E731
                        mr, mg, mn = te & (y == 0), te & (y == 1), te & (y == 2) & (df.any_light.to_numpy() == 0)
                        row["test_S"] = r_(mr, 0) - r_(mr, 1) + r_(mg, 1) - r_(mn, 0)
                # grouped CV over all routes
                cv = np.full(len(df), -1)
                for k in range(N_FOLDS):
                    trk, tek = (fold != k) & (y >= 0), fold == k
                    if len(set(y[trk])) < ncls:
                        continue
                    Xk, _, _ = standardise(torch, X[trk], X, dev)
                    tt = torch.tensor(trk, device=dev)
                    mk = fit_lin(torch, Xk[tt], torch.tensor(y, device=dev)[tt], lam)
                    cv[tek] = probs(torch, mk, Xk).argmax(-1).cpu().numpy()[tek]
                preds["cv|" + key] = cv.astype(np.int8)
                scored = (y >= 0) & (cv >= 0)
                cb, crec = bacc(cv[scored], y[scored], ncls) if scored.any() else (np.nan, [np.nan] * ncls)
                row.update(cv_bacc=cb, cv_rec=crec, cv_n=int(scored.sum()))
                if lab == "light" and scored.any():
                    mr, mg, mn = scored & (y == 0), scored & (y == 1), scored & (y == 2) & (df.any_light.to_numpy() == 0)
                    r_ = lambda m_, c_: float((cv[m_] == c_).mean()) if m_.any() else np.nan   # noqa: E731
                    row["cv_S"] = r_(mr, 0) - r_(mr, 1) + r_(mg, 1) - r_(mn, 0)
                rows.append(row)
        print("%s %s %s done" % (a.model, a.res, src), flush=True)
    out = Path(a.run) / "fit"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / ("%s_%s_preds.npz" % (a.model, a.res)), ids=df.id.to_numpy(), **preds)
    (out / ("%s_%s.json" % (a.model, a.res))).write_text(json.dumps(rows, indent=0, default=lambda o: None if o is None else float(o)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--dir-too", action="store_true")
    a = ap.parse_args()
    fit(a)
