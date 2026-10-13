#!/usr/bin/env python3
"""STOPLINE-PROBE: is the light (red vs non-red) of the ego lane readable from the frozen openpilot (Cinque) `view_39` tokens of one frame?
Linear / small-MLP probes, log- (nuPlan) or scene- (PAI) grouped 5-fold CV, AUC with a cluster-bootstrap CI, against a position-only
baseline (distance to the line, speed, acceleration). CPU only.

  nuplan  (box, .venv)  slp_probe.py nuplan [--threads 8] [--seed 0]
          rows: navtrain tokens whose hindsight route crosses a traffic-light stop line (map type 2) 0-40 m ahead of the front bumper with
          the line's lane connector matched and its light known at t0 (LONG-PROBE ctx.npz, lc_ctx.py: lane connector -> the log's
          traffic_lights (lane_connector_id, is_red)); y = red at t0. Tokens: front.npy slot 7 (the t0 frame, 32 x 512) of the op_parity
          navtrain_full@warp caches (the frozen tokens BODY1 S0 / LONG-PROBE read). lead = a vehicle box centre within 2 m of the route,
          0-15 m ahead of the front bumper at t0 (body1 agent labels).
  pai     (box)         slp_probe.py pai --frames frames.csv --feats feats.npz --out pai.json [--transfer]   (slp_labels.py + slp_feats.py
          outputs): red / stop_sign frames against green ones; --transfer = the nuPlan probe (tok_lr fitted on all nuPlan rows) applied unchanged

Models per fold (fit on the train folds only): base = gradient-boosted trees on (d, v, a); tok = PCA-256 of the 16 384 token values +
logistic regression (C 0.05) and an MLP (256, early stopping); tok+base = the same on PCA ++ (d, v, a). Out-of-fold scores are pooled, AUC per
stratum, 95 % percentile CI over 2 000 resamples of the groups (seed 0); the gain over base is paired (same resamples).
Output: <run dir>/results.json, oof.npz; copied to experiments/stopline_probe/results/<name>.json by the caller.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/body1/lib")]

FRONT, D_MAX, LEAD_M, LEAD_LAT = 4.049, 40.0, 15.0, 2.0
N_BOOT, NPC = 2000, 256


def strata(d, v, lead):
    return {"all": np.ones(len(d), bool), "moving (v >= 2 m/s)": v >= 2.0, "moving, no lead 15 m": (v >= 2.0) & ~lead,
            "stopped (v < 0.5)": v < 0.5, "d 0-15 m": d <= 15, "d 15-40 m": d > 15}


def auc_ci(y, s, g, base=None, seed=0):
    """AUC of scores s (and of base, and s - base) with a cluster bootstrap over groups g."""
    from sklearn.metrics import roc_auc_score
    out = dict(n=int(len(y)), pos=int(y.sum()), groups=int(len(np.unique(g))))
    if y.min() == y.max():
        return out
    out["auc"] = float(roc_auc_score(y, s))
    if base is not None:
        out["auc_base"] = float(roc_auc_score(y, base))
    codes, uniq = pd.factorize(g)
    idx_by = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_by[j] for j in rng.integers(0, len(uniq), len(uniq))])
        if y[ii].min() == y[ii].max():
            continue
        A.append(roc_auc_score(y[ii], s[ii]))
        if base is not None:
            B.append(roc_auc_score(y[ii], base[ii]))
    A = np.array(A)
    out["lo"], out["hi"] = map(float, np.quantile(A, [0.025, 0.975]))
    if base is not None:
        B, dlt = np.array(B), A - np.array(B)
        out["base_lo"], out["base_hi"] = map(float, np.quantile(B, [0.025, 0.975]))
        out["gain"], (out["gain_lo"], out["gain_hi"]) = out["auc"] - out["auc_base"], map(float, np.quantile(dlt, [0.025, 0.975]))
    return out


def cv(X, K, y, g, seed, log):
    """X (n, 16384) token values, K (n, k) position features -> out-of-fold scores {model: (n,)}."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    names = ("base", "tok_lr", "tok_mlp", "tokbase_lr", "tokbase_mlp")
    S = {k: np.full(len(y), np.nan) for k in names}
    ug = np.unique(g)
    perm = dict(zip(ug, np.random.default_rng(seed).permutation(len(ug))))
    folds = GroupKFold(5).split(X, y, np.array([perm[x] for x in g]))
    for f, (tr, te) in enumerate(folds):
        S["base"][te] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(K[tr], y[tr]).predict_proba(K[te])[:, 1]
        sc = StandardScaler().fit(X[tr])
        pca = PCA(NPC, svd_solver="randomized", random_state=seed).fit(sc.transform(X[tr]))
        Z = pca.transform(sc.transform(X)).astype(np.float32)
        Z /= Z[tr].std(0, keepdims=True) + 1e-6
        Kz = StandardScaler().fit(K[tr]).transform(K).astype(np.float32)
        for tag, F in (("tok", Z), ("tokbase", np.c_[Z, Kz])):
            S[f"{tag}_lr"][te] = LogisticRegression(C=0.05, max_iter=2000).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
            S[f"{tag}_mlp"][te] = MLPClassifier((256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=seed).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        log(f"fold {f}: train {len(tr)} test {len(te)}, PCA var {pca.explained_variance_ratio_.sum():.3f}")
    return S


def report(S, y, g, d, v, lead, log):
    res = {}
    for sn, m in strata(d, v, lead).items():
        if m.sum() < 30 or y[m].min() == y[m].max():
            continue
        res[sn] = {k: auc_ci(y[m], S[k][m], g[m], base=None if k == "base" else S["base"][m]) for k in S}
        r = res[sn]
        log(f"{sn:24s} n {r['base']['n']:6d} red {r['base']['pos']:6d} groups {r['base']['groups']:4d} | base {r['base']['auc']:.3f} "
            + " ".join(f"{k} {r[k]['auc']:.3f} [{r[k]['lo']:.3f}, {r[k]['hi']:.3f}] (+{r[k]['gain']:+.3f})" for k in S if k != "base"))
    return res


def load_nuplan(log, elig=False):
    import b1 as B
    D = B.data_dir()
    z = np.load(D / "runs/long_contact/ctx/ctx.npz")
    ty, dd, conn, st0, route = z["ln_type"], z["ln_d"] - FRONT, z["ln_conn"], z["ln_st"][:, :, 0], z["route"]
    tl = ty == 2
    q = np.where(tl.any(1), tl.argmax(1), -1)
    i = np.flatnonzero(q >= 0)
    dq, cq, sq = dd[i, q[i]], conn[i, q[i]], st0[i, q[i]]
    keep = (dq > 0) & (dq <= D_MAX) & cq & (sq >= 0)
    if elig:                                                       # LONG-PROBE eligibility: the logged driver does not cross this line on red
        keep &= z["ln_elig"][i, q[i]]
    rows, d, y = i[keep], dq[keep], (sq[keep] == 1).astype(int)
    L = B.labels()
    assert (L["tokens"] == z["tokens"]).all()
    off = B.shard_offsets()
    tabs = [B.tab("log", k) for k in range(B.NSH)]
    v = np.concatenate([t["speed"] for t in tabs])[rows].astype(float)
    a = np.concatenate([t["acc"][:, -1, 0] for t in tabs])[rows].astype(float)
    box, val, cls = np.asarray(L["box"][rows, 0]), np.asarray(L["valid"][rows, 0]), np.asarray(L["cls"][rows])
    lead = np.zeros(len(rows), bool)
    for j, r in enumerate(rows):                                   # vehicle centre near the route, 0-15 m ahead of the front bumper
        m = val[j] & (cls[j] == 0)
        if not m.any():
            continue
        P, R = box[j, m, :2], route[r]
        e = np.hypot(*(P[:, None] - R[None]).transpose(2, 0, 1))
        k = e.argmin(1)
        ahead = (k - FRONT > 0) & (k - FRONT - box[j, m, 3] / 2 <= LEAD_M) & (e[np.arange(len(k)), k] < LEAD_LAT)
        lead[j] = ahead.any()
    X = np.empty((len(rows), 32 * 512), np.float32)
    for k in range(B.NSH):
        sel = np.flatnonzero((rows >= off[k]) & (rows < off[k + 1]))
        if len(sel):
            f = np.load(B.cache_root() / f"{B.cdir('log', k)}@warp" / "front.npy", mmap_mode="r")
            rl = rows[sel] - off[k]
            o = np.argsort(rl)
            X[sel[o]] = np.asarray(f[rl[o], 7], np.float32).reshape(len(sel), -1)
        log(f"shard {k}: {len(sel)} rows")
    return dict(X=X, K=np.c_[d, v, a], y=y, g=L["log"][rows], d=d, v=v, lead=lead, tokens=z["tokens"][rows])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("nuplan")
    n.add_argument("--elig", action="store_true", help="only lines the logged driver does not cross on red (ln_elig)")
    p = sp.add_parser("pai")
    p.add_argument("--frames", required=True)
    p.add_argument("--feats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--transfer", action="store_true", help="also fit the probe on all nuPlan rows (box) and score the PAI frames")
    for x in (n, p):
        x.add_argument("--threads", type=int, default=8)
        x.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(a.threads)
    global np, pd
    import numpy as np
    import pandas as pd
    if a.cmd == "nuplan":
        from jevdrive.data import splits
        from jevdrive.run import Run
        with Run("stopline_probe", "nuplan-elig" if a.elig else "nuplan", seed=a.seed, config=vars(a)) as run:
            run.use_split(splits.load("navsim/navtrain"))
            T = load_nuplan(run.info, a.elig)
            run.info(f"rows {len(T['y'])}, red {T['y'].sum()}, logs {len(np.unique(T['g']))}, lead {T['lead'].mean():.3f}")
            S = cv(T["X"], T["K"], T["y"], T["g"], a.seed, run.info)
            res = report(S, T["y"], T["g"], T["d"], T["v"], T["lead"], run.info)
            np.savez(run.path("oof.npz"), y=T["y"], g=T["g"], d=T["d"], v=T["v"], lead=T["lead"], tokens=T["tokens"], **S)
            Path(run.path("results.json")).write_text(json.dumps(dict(set="nuplan navtrain" + (" eligible" if a.elig else ""), rows=int(len(T["y"])),
                                                                      strata=res), indent=1))
            run.summary["auc_tok_lr_all"] = res["all"]["tok_lr"]["auc"]
    else:
        pai(a)


def pai(a):
    """PAI frames: per class (red / stop_sign against green) (i) the probe trained on the nuPlan rows, applied unchanged (transfer), and
    (ii) a within-PAI scene-grouped CV (descriptive: few positive scenes). Baseline = (d, v) trees, CV within PAI."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fr = pd.read_csv(a.frames)
    Z = np.load(a.feats)
    idx = {f: i for i, f in enumerate(Z["files"])}
    fr = fr[fr.file.isin(idx)].reset_index(drop=True)
    X = Z["X"][[idx[f] for f in fr.file]].astype(np.float32)
    out = dict(set="pai train160 PAI_DUMP frames", rows=int(len(fr)), classes={}, counts=fr.groupby("cls").agg(
        frames=("file", "size"), scenes=("scene", "nunique")).to_dict("index"))
    T = None
    if a.transfer:
        T = load_nuplan(print)
        sc = StandardScaler().fit(T["X"])
        pca = PCA(NPC, svd_solver="randomized", random_state=a.seed).fit(sc.transform(T["X"]))
        Zn = pca.transform(sc.transform(T["X"]))
        sd = Zn.std(0, keepdims=True) + 1e-6
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(Zn / sd, T["y"])
        fr["transfer"] = lr.predict_proba(pca.transform(sc.transform(X)) / sd)[:, 1]
    for cls in ("red", "stop_sign"):
        m = fr.cls.isin([cls, "green"]).values
        f, Xc = fr[m].reset_index(drop=True), X[m]
        y, g = (f.cls == cls).values.astype(int), f.scene.values
        r = dict(frames=int(len(f)), pos_frames=int(y.sum()), pos_scenes=int(f[y == 1].scene.nunique()))
        S = cv(Xc, np.c_[f.d_line, f.v], y, g, a.seed, print) if r["pos_scenes"] >= 5 else None
        if S is not None:
            r["cv"] = {k: auc_ci(y, S[k], g, base=None if k == "base" else S["base"]) for k in S}
        if T is not None:
            r["transfer"] = auc_ci(y, f.transfer.values, g)
            r["transfer_moving"] = auc_ci(y[f.v.values >= 2], f.transfer.values[f.v.values >= 2], g[f.v.values >= 2])
            r["transfer_mean_red_prob"] = {c: float(f.transfer[f.cls == c].mean()) for c in (cls, "green")}
        out["classes"][cls] = r
        print(cls, json.dumps(r)[:600])
    if T is not None:
        fr[["file", "scene", "cls", "d_line", "v", "transfer"]].to_csv(Path(a.out).with_suffix(".csv"), index=False)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
