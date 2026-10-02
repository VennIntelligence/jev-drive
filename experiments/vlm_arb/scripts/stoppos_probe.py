"""Route-grouped cross-validated probes, heads and baselines of the stop-position probe (plan 2026-10-03-stoppos-probe.md).

  python stoppos_probe.py --run RUN_DIR [--taps temporal,vision,hidden,native] [--folds 0,1,2,3,4]

Writes RUN_DIR/oof.npz: out-of-fold predictions, one array per `<tap or base>/<model>/<name>` (NaN where the frame is not a
test frame of the fold that was run), and RUN_DIR/hparams.json. Resumable per (tap, fold) through RUN_DIR/oof/<tap>-f<k>.npz.
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

CLOCK_FEATS = ("v", "tdrive", "dist")


def clock_matrix(f: pd.DataFrame) -> np.ndarray:
    a = f[list(CLOCK_FEATS)].to_numpy(np.float64)
    a = np.nan_to_num(a, nan=0.0)
    return np.c_[a, a ** 2, a[:, [0]] * a[:, [1]]].astype(np.float32)


def make_features(tap, X, rows, k, log):
    """Standardised (and for `hidden` PCA-whitened) features of every frame; the transforms are fitted on `rows` (train frames)."""
    std = H.Standardizer().fit(X[rows])
    if tap != "hidden":
        return std(np.nan_to_num(X, nan=0.0))
    t0 = time.time()
    g = np.random.default_rng(k)
    sub = g.choice(np.flatnonzero(rows), size=min(20000, int(rows.sum())), replace=False)
    pca = H.Pca(512, seed=k).fit(std(X[np.sort(sub)]))
    Z = []
    for i in range(0, len(X), 10000):
        Z.append(pca(std(np.nan_to_num(X[i:i + 10000].astype(np.float32), nan=0.0))))
    log(f"  hidden PCA fold {k}: {time.time() - t0:.0f} s")
    return torch_cat(Z)


def torch_cat(Z):
    import torch
    return torch.cat(Z)


def pick_ridge(Z, y, w, tr_idx, cal_idx, wcal):
    fits = H.ridge_fit(Z[tr_idx], y[tr_idx], w[tr_idx])
    errs = [H.mae_w(H.ridge_predict(Z[cal_idx], c), y[cal_idx], wcal[cal_idx]) for c in fits]
    return int(np.argmin(errs)), errs


def pick_logit(Z, y, w, tr_idx, cal_idx, wcal, ncls):
    nll = []
    for C in H.LOGIT_C:
        Wb = H.logistic_fit(Z[tr_idx], y[tr_idx], w[tr_idx], ncls, C)
        nll.append(H.nll_w(H.logits(Z[cal_idx], Wb), y[cal_idx], wcal[cal_idx]))
    return int(np.argmin(nll)), nll


def run_fold(frames, X, tap, k, fold, have, routes, log, hp, with_clock=True, mlp=True):
    """All heads of one (tap, fold); returns {name: array over the frames of the test fold (len = test rows)}."""
    tr_rows = have & (fold != k) & ~np.isnan(fold)
    te_rows = have & (fold == k)
    train_ids = set(frames.id[tr_rows])
    cal_ids = D.inner_cal(routes, train_ids, seed=k)
    is_cal = frames.id.isin(cal_ids).to_numpy()
    Z = make_features(tap, X, tr_rows, k, log)
    clk = None
    if with_clock:
        cm = clock_matrix(frames)
        cs = H.Standardizer().fit(cm[tr_rows])
        clk = cs(cm)
        Zc = H.torch.cat([Z, clk], 1)
    te = np.flatnonzero(te_rows)
    out = {}
    tasks = {"junc": ("m_junc", "y_junc", "reg"), "stop": ("m_stop", "y_stop", "reg"), "stop40": ("m_cls", "y_stop40", 2),
             "light": ("m_ls", "y_light", 2), "sign": ("m_ls", "y_sign", 2), "cls4": ("m_cls", "y_cls4", 4)}
    for name, (mcol, ycol, kind) in tasks.items():
        M = frames[mcol].to_numpy() & have
        y = frames[ycol].to_numpy(np.float64)
        y = np.nan_to_num(y, nan=0.0)
        tr = np.flatnonzero(M & tr_rows)
        sub = np.flatnonzero(M & tr_rows & ~is_cal)
        cal = np.flatnonzero(M & is_cal & tr_rows)
        w_all = D.attempt_weights(frames, M & tr_rows)
        w_sub = D.attempt_weights(frames, M & tr_rows & ~is_cal)
        w_cal = D.attempt_weights(frames, M & is_cal & tr_rows)
        enough = len(cal) >= 30 and len(sub) >= 100
        if kind != "reg":
            enough = enough and len(np.unique(y[cal])) >= 2 and len(np.unique(y[sub])) >= 2
        if len(tr) < 100 or (kind != "reg" and len(np.unique(y[tr])) < 2):
            log(f"  {tap} fold {k} task {name}: too little training data ({len(tr)} frames), skipped")
            continue
        for variant, ZZ in (("", Z),) + ((("clk", Zc),) if (clk is not None and name in ("junc", "stop", "stop40")) else ()):
            tag = f"{tap}/{'ridge' if kind == 'reg' else 'logit'}{variant}/{name}"
            if kind == "reg":
                a_i = pick_ridge(ZZ, y, w_sub, sub, cal, w_cal)[0] if enough else 2
                fits = H.ridge_fit(ZZ[tr], y[tr], w_all[tr])
                out[tag] = H.ridge_predict(ZZ[te], fits[a_i])
                hp.append(dict(tap=tap, fold=k, tag=tag, alpha=H.RIDGE_ALPHAS[a_i], n_train=len(tr), n_cal=len(cal)))
            else:
                c_i = pick_logit(ZZ, y, w_sub, sub, cal, w_cal, kind)[0] if enough else 1
                Wb = H.logistic_fit(ZZ[tr], y[tr], w_all[tr], kind, H.LOGIT_C[c_i])
                lg = H.logits(ZZ[te], Wb)
                if kind == 2:
                    out[tag] = (lg[:, 1] - lg[:, 0]).cpu().numpy()          # log-odds of the positive class
                else:
                    Wi = H.logistic_fit(ZZ[sub], y[sub], w_sub[sub], 4, H.LOGIT_C[c_i]) if enough else Wb
                    Tt = H.fit_temperature(H.logits(ZZ[cal], Wi), y[cal], w_cal[cal]) if enough else 1.0
                    for c in range(4):
                        out[f"{tag}_{c}"] = lg[:, c].cpu().numpy()
                    out[f"{tag}_T"] = np.full(len(te), Tt)
                hp.append(dict(tap=tap, fold=k, tag=tag, C=H.LOGIT_C[c_i], n_train=len(tr), n_cal=len(cal)))
        if name == "stop" and enough:
            # linear quantile heads and the conformalised correction for the nominal 80% interval
            lin_i = H.quantile_fit(Z[sub], y[sub], w_sub[sub], seed=k)
            qc = H.quantile_predict(Z[cal], lin_i)
            c = H.cqr_constant(qc[:, 0], qc[:, 2], y[cal], w_cal[cal])
            lin = H.quantile_fit(Z[tr], y[tr], w_all[tr], seed=k)
            q = H.quantile_predict(Z[te], lin)
            out[f"{tap}/q/lo_raw"], out[f"{tap}/q/med"], out[f"{tap}/q/hi_raw"] = q[:, 0], q[:, 1], q[:, 2]
            out[f"{tap}/q/lo"], out[f"{tap}/q/hi"] = q[:, 0] - c, q[:, 2] + c
            hp.append(dict(tap=tap, fold=k, tag=f"{tap}/q", cqr_c=c))
        if mlp and enough and name in ("junc", "stop", "stop40"):
            mk, task = ("reg", "reg") if kind == "reg" else ("cls", "cls")
            yy = y if kind == "reg" else y.astype(int)
            m_i, ep = H.mlp_fit(Z[sub], yy[sub], w_sub[sub], task, Z[cal], yy[cal], w_cal[cal], seed=k)
            m_f, _ = H.mlp_fit(Z[tr], yy[tr], w_all[tr], task, epochs=ep, seed=k)
            p = H.mlp_predict(m_f, Z[te], task)
            out[f"{tap}/mlp/{name}"] = p if kind == "reg" else np.log(np.clip(p[:, 1], 1e-6, 1)) - np.log(np.clip(p[:, 0], 1e-6, 1))
            hp.append(dict(tap=tap, fold=k, tag=f"{tap}/mlp/{name}", epochs=ep))
    return te, out


def baselines(frames, have, fold, routes, log, hp):
    """Route-command baseline and ego speed / time baselines, per fold (no tap)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    out = {}
    N = len(frames)
    cm = clock_matrix(frames)
    for k in range(D.K):
        tr_rows = have & (fold != k) & ~np.isnan(fold)
        te = np.flatnonzero(have & (fold == k))
        train_ids = set(frames.id[tr_rows])
        cal_ids = D.inner_cal(routes, train_ids, seed=k)
        is_cal = frames.id.isin(cal_ids).to_numpy()
        # route command baseline: offset = median over training routes of the route's median (command point - stop line)
        fr = frames[tr_rows & frames.m_stop.to_numpy() & frames.d_cmd.notna().to_numpy()]
        off = (fr.d_cmd - fr.d_stop).groupby(fr.id).median()
        delta = float(off.median()) if len(off) else 3.3
        hp.append(dict(tap="base", fold=k, tag="route", delta=delta, routes=len(off)))
        dc = frames.d_cmd_c.to_numpy()
        for nm, val in (("junc", np.clip(dc, 0, 40)), ("stop", np.clip(dc - delta, -2, 40))):
            out.setdefault(f"base/route/{nm}", np.full(N, np.nan, np.float32))[te] = val[te]
        out.setdefault("base/route/stop40", np.full(N, np.nan, np.float32))[te] = (-dc)[te]
        out.setdefault("base/route/cls4_d", np.full(N, np.nan, np.float32))[te] = (dc - delta)[te]
        cs = H.Standardizer().fit(cm[tr_rows])
        Zc = cs(cm)
        for name, (mcol, ycol, kind) in {"junc": ("m_junc", "y_junc", "reg"), "stop": ("m_stop", "y_stop", "reg"), "stop40": ("m_cls", "y_stop40", 2),
                                         "light": ("m_ls", "y_light", 2), "sign": ("m_ls", "y_sign", 2), "cls4": ("m_cls", "y_cls4", 4)}.items():
            M = frames[mcol].to_numpy() & have
            y = np.nan_to_num(frames[ycol].to_numpy(np.float64), nan=0.0)
            tr = np.flatnonzero(M & tr_rows)
            sub = np.flatnonzero(M & tr_rows & ~is_cal)
            cal = np.flatnonzero(M & is_cal & tr_rows)
            w_all, w_sub, w_cal = (D.attempt_weights(frames, M & tr_rows), D.attempt_weights(frames, M & tr_rows & ~is_cal),
                                   D.attempt_weights(frames, M & is_cal & tr_rows))
            if len(tr) < 100 or (kind != "reg" and len(np.unique(y[tr])) < 2):
                continue
            enough = len(cal) >= 30 and len(sub) >= 100 and (kind == "reg" or len(np.unique(y[cal])) >= 2)
            if kind == "reg":
                a_i = pick_ridge(Zc, y, w_sub, sub, cal, w_cal)[0] if enough else 2
                fits = H.ridge_fit(Zc[tr], y[tr], w_all[tr])
                out.setdefault(f"base/clockridge/{name}", np.full(N, np.nan, np.float32))[te] = H.ridge_predict(Zc[te], fits[a_i])
                g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
                g.fit(cm[tr], y[tr], sample_weight=w_all[tr])
                out.setdefault(f"base/clockgbm/{name}", np.full(N, np.nan, np.float32))[te] = g.predict(cm[te])
            elif kind == 2:
                c_i = pick_logit(Zc, y, w_sub, sub, cal, w_cal, 2)[0] if enough else 1
                Wb = H.logistic_fit(Zc[tr], y[tr], w_all[tr], 2, H.LOGIT_C[c_i])
                lg = H.logits(Zc[te], Wb)
                out.setdefault(f"base/clocklogit/{name}", np.full(N, np.nan, np.float32))[te] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
                g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
                g.fit(cm[tr], y[tr].astype(int), sample_weight=w_all[tr])
                p = np.clip(g.predict_proba(cm[te])[:, 1], 1e-6, 1 - 1e-6)
                out.setdefault(f"base/clockgbm/{name}", np.full(N, np.nan, np.float32))[te] = np.log(p) - np.log(1 - p)
            else:
                g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0)
                g.fit(cm[tr], y[tr].astype(int), sample_weight=w_all[tr])
                P = g.predict_proba(cm[te])
                for c in range(4):
                    a = out.setdefault(f"base/clockgbm/cls4_{c}", np.full(N, np.nan, np.float32))
                    a[te] = np.log(np.clip(P[:, c], 1e-6, 1)) if c < P.shape[1] else -14
        log(f"baselines fold {k} done")
    return out


def main(a):
    run_dir = Path(a.run)
    (run_dir / "oof").mkdir(exist_ok=True)
    logf = open(run_dir / "probe.log", "a")

    def log(m):
        s = time.strftime("%H:%M:%S ") + m
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    frames = D.load_frames()
    routes = D.routes_table()
    have = D.have_features(frames)
    fold = D.fold_of(frames).astype(float)
    log(f"{len(frames)} frames, {int(have.sum())} with features, routes {frames.id.nunique()}")
    taps = a.taps.split(",")
    folds = [int(x) for x in a.folds.split(",")]
    hp = []
    N = len(frames)
    for tap in taps:
        X = None
        for k in folds:
            f = run_dir / "oof" / f"{tap}-f{k}.npz"
            if f.exists():
                continue
            if X is None:
                X = D.load_feats(frames, tap)
            t0 = time.time()
            hp_k = []
            te, out = run_fold(frames, X, tap, k, fold, have, routes, log, hp_k, mlp=not a.no_mlp)
            json.dump(hp_k, open(run_dir / "oof" / f"{tap}-f{k}.json", "w"))
            np.savez(f.with_suffix(".tmp.npz"), te=te, **{n.replace("/", "|"): v for n, v in out.items()})
            f.with_suffix(".tmp.npz").replace(f)
            (run_dir / "STATUS").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" stoppos_probe: {tap} fold {k} done ({time.time() - t0:.0f} s)\n")
            log(f"{tap} fold {k}: {time.time() - t0:.0f} s, {len(out)} outputs")
        del X
    bf = run_dir / "oof" / "base.npz"
    if not bf.exists():
        hp_b = []
        out = baselines(frames, have, fold, routes, log, hp_b)
        json.dump(hp_b, open(run_dir / "oof" / "base.json", "w"))
        np.savez(bf, **{n.replace("/", "|"): v for n, v in out.items()})
    hp = [h for f in sorted((run_dir / "oof").glob("*.json")) for h in json.load(open(f))]
    json.dump(hp, open(run_dir / "hparams.json", "w"), indent=1)
    # assemble
    arrs = {}
    for f in sorted((run_dir / "oof").glob("*-f*.npz")):
        z = np.load(f)
        te = z["te"]
        for n in z.files:
            if n == "te":
                continue
            a_ = arrs.setdefault(n.replace("|", "/"), np.full(N, np.nan, np.float32))
            a_[te] = z[n]
    with np.load(bf) as z:
        for n in z.files:
            arrs[n.replace("|", "/")] = z[n]
    np.savez(run_dir / "oof.npz", **{n.replace("/", "|"): v for n, v in arrs.items()})
    log(f"assembled {len(arrs)} arrays")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--taps", default=",".join(D.TAPS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-mlp", action="store_true")
    main(ap.parse_args())
