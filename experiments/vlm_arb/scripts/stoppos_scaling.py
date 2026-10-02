"""How much data: accuracy against the number of training routes (plan 2026-10-03-stoppos-probe.md, Heads and readouts 6).

  python stoppos_scaling.py --run RUN_DIR --tap temporal --out experiments/vlm_arb/results/stoppos

Per outer fold: n routes drawn at random from the training routes (n = 10, 20, 40, 80, all; 20 draws for n <= 40, 5 for 80), ridge for the
stop-line distance and logistic for `stop40` with the hyper-parameters the fold chose on its full training set, tested on the fold's routes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stoppos_data as D  # noqa: E402
import stoppos_heads as H  # noqa: E402
import stoppos_probe as P  # noqa: E402

SIZES = (10, 20, 40, 80, 0)
DRAWS = {10: 20, 20: 20, 40: 20, 80: 5, 0: 1}


def main(a):
    out = Path(a.out)
    frames = D.load_frames()
    routes = D.routes_table()
    have = D.have_features(frames)
    fold = D.fold_of(frames).astype(float)
    hp = json.load(open(Path(a.run) / "hparams.json"))
    X = D.load_feats(frames, a.tap)
    rows = []
    t0 = time.time()
    usable = set(frames[frames.m_stop].id)          # routes that carry a light approach with a known stop line
    for k in range(D.K):
        tr_rows = have & (fold != k) & ~np.isnan(fold)
        te_rows = have & (fold == k)
        Z = P.make_features(a.tap, X, tr_rows, k, print)
        alpha = next(h["alpha"] for h in hp if h["tag"] == f"{a.tap}/ridge/stop" and h["fold"] == k)
        C = next(h["C"] for h in hp if h["tag"] == f"{a.tap}/logit/stop40" and h["fold"] == k)
        a_i = H.RIDGE_ALPHAS.index(alpha)
        tr_ids = np.array(sorted(set(frames.id[tr_rows])))
        rng = np.random.default_rng(100 + k)
        Mst, Mc = frames.m_stop.to_numpy() & have, frames.m_cls.to_numpy() & have
        y_s = np.nan_to_num(frames.y_stop.to_numpy(np.float64), nan=0.0)
        y_c = frames.y_stop40.to_numpy()
        test_s = frames[te_rows & Mst & (frames.d_stop >= 0).to_numpy()]
        test_c = frames[te_rows & Mc]
        for n in SIZES:
            for draw in range(DRAWS[n]):
                ids = tr_ids if n == 0 else rng.choice(tr_ids, n, replace=False)
                sel = frames.id.isin(set(ids)).to_numpy() & tr_rows
                n_light = len(set(ids) & usable)
                r = dict(fold=k, n=n, n_routes=len(ids), draw=draw, light_routes=n_light)
                ts = np.flatnonzero(Mst & sel)
                if len(ts) >= 50 and n_light >= 2:
                    w = D.attempt_weights(frames, Mst & sel)
                    coef = H.ridge_fit(Z[ts], y_s[ts], w[ts], alphas=(alpha,))[0]
                    pred = H.ridge_predict(Z[test_s.index.to_numpy()], coef)
                    d = test_s.assign(err=pred - test_s.d_stop)
                    for lo, hi, nm in ((0, 10, "mae_0_10"), (10, 40, "mae_10_40")):
                        m = d[(d.d_stop >= lo) & (d.d_stop < hi)]
                        r[nm] = float(D.route_means(m.assign(ae=m.err.abs()), "ae").mean()) if len(m) else np.nan
                tc = np.flatnonzero(Mc & sel)
                if len(tc) >= 100 and len(np.unique(y_c[tc])) == 2:
                    w = D.attempt_weights(frames, Mc & sel)
                    Wb = H.logistic_fit(Z[tc], y_c[tc], w[tc], 2, C)
                    lg = H.logits(Z[test_c.index.to_numpy()], Wb)
                    s = (lg[:, 1] - lg[:, 0]).cpu().numpy()
                    r["auc_stop40"] = D.auc(test_c.y_stop40.to_numpy().astype(bool), s, D.group_weights(test_c))
                rows.append(r)
        print(f"fold {k} done {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"scaling_{a.tap}_draws.csv", index=False)
    g = df.groupby("n").agg(draws=("fold", "size"), n_routes=("n_routes", "mean"), light_routes=("light_routes", "mean"), mae_0_10=("mae_0_10", "mean"), mae_0_10_sd=("mae_0_10", "std"),
                            mae_10_40=("mae_10_40", "mean"), mae_10_40_sd=("mae_10_40", "std"), auc_stop40=("auc_stop40", "mean"),
                            auc_stop40_sd=("auc_stop40", "std")).reset_index()
    # n = 0 stands for all training routes of the fold (about 138 routes)
    g.to_csv(out / f"scaling_{a.tap}.csv", index=False)
    print(g.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tap", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
