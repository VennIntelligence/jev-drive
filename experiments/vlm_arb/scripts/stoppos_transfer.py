"""Domain transfer of the probes (plan 2026-10-03-stoppos-probe.md, Split): fit on one set (P4 or CL), test on the other.

  python stoppos_transfer.py --run RUN_DIR --out DIR [--taps temporal,vision,native]

Ridge for the junction / stop-line distance, logistic for `stop40`; hyper-parameters from a 25% route-grouped inner split of the training set;
the same macro-MAE-by-bin and AUC readouts as the cross-validated tables (all test routes of the other set).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_data as D  # noqa: E402
import stoppos_heads as H  # noqa: E402
import stoppos_probe as P  # noqa: E402


def main(a):
    out = Path(a.out)
    frames = D.load_frames()
    routes = D.routes_table()
    have = D.have_features(frames)
    rows, crow = [], []
    for tap in a.taps.split(","):
        X = D.load_feats(frames, tap)
        for src_tr, src_te in (("p4", "cl"), ("cl", "p4")):
            tr_rows = have & (frames.src == src_tr).to_numpy()
            te_rows = have & (frames.src == src_te).to_numpy()
            Z = P.make_features(tap, X, tr_rows, 0, print)
            cal_ids = D.inner_cal(routes, set(frames.id[tr_rows]), seed=0)
            is_cal = frames.id.isin(cal_ids).to_numpy()
            for name, mcol, ycol, kind, truth in (("junc", "m_junc", "y_junc", "reg", "d_junc"), ("stop", "m_stop", "y_stop", "reg", "d_stop"),
                                                  ("stop40", "m_cls", "y_stop40", 2, None)):
                M = frames[mcol].to_numpy() & have
                y = np.nan_to_num(frames[ycol].to_numpy(np.float64), nan=0.0)
                tr, sub, cal = np.flatnonzero(M & tr_rows), np.flatnonzero(M & tr_rows & ~is_cal), np.flatnonzero(M & tr_rows & is_cal)
                if len(tr) < 100 or len(sub) < 100 or len(cal) < 30:
                    continue
                w_all, w_sub, w_cal = (D.attempt_weights(frames, M & tr_rows), D.attempt_weights(frames, M & tr_rows & ~is_cal),
                                       D.attempt_weights(frames, M & tr_rows & is_cal))
                if kind == "reg":
                    a_i = P.pick_ridge(Z, y, w_sub, sub, cal, w_cal)[0]
                    pred = H.ridge_predict(Z, H.ridge_fit(Z[tr], y[tr], w_all[tr])[a_i])
                    d = frames[M & te_rows & (frames[truth] >= 0).to_numpy()]
                    d = d.assign(err=pred[d.index] - d[truth])
                    rows += D.bin_rows(d, truth, "err", dict(task=name, tap=tap, train=src_tr, test=src_te))
                else:
                    if len(np.unique(y[cal])) < 2 or len(np.unique(y[sub])) < 2:
                        continue
                    c_i = P.pick_logit(Z, y, w_sub, sub, cal, w_cal, 2)[0]
                    Wb = H.logistic_fit(Z[tr], y[tr], w_all[tr], 2, H.LOGIT_C[c_i])
                    lg = H.logits(Z, Wb)
                    s = (lg[:, 1] - lg[:, 0]).cpu().numpy()
                    d = frames[M & te_rows]
                    crow.append(dict(task=name, tap=tap, train=src_tr, test=src_te, frames=len(d), routes=d.id.nunique(),
                                     auc=D.auc(d[ycol].to_numpy().astype(bool), s[d.index], D.group_weights(d))))
        del X
    pd.DataFrame(rows).to_csv(out / "transfer_reg.csv", index=False)
    pd.DataFrame(crow).to_csv(out / "transfer_clf.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", default="temporal,vision,native")
    main(ap.parse_args())
