#!/usr/bin/env python3
"""PROXY: can a reweighting of the 1491 public AlpaSim nuPlan scenes reproduce the private leaderboard ordering and gaps of policies whose
private numbers are known (anchors)? Pre-registration: experiments/alpasim_proxy/plans/2026-10-12-proxy-prereg.md.

  proxy_fit.py <per_scene.csv> [<per_scene.csv> ...] --out DIR [--anchors A1,A2,...]      (Mac, CPU; seconds)

Inputs: per-scene tables with columns driver, scene, score, dist_to_gt_trajectory, dist_traveled_m, offroad_or_collision_at_fault
(aud1_drift_collect.py / proxy_collect.py output); scene metadata from experiments/alpasim/results/aud1/per_scene.csv and
results/scene_city.csv. Metrics as in AUD1 (src/eval @ 0bb4c4b): mean scene score (failed scenes 0, in the denominator), GT distance =
mean over clips with metrics of the truncated per-clip max, at-fault distance km = sum driven / sum at-fault flags / 1000; with scene
weights w they become weighted means / a weighted ratio.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parents[3]
# private leaderboard numbers (results/leaderboard_nuplan_20261012.json): score, at-fault km, GT m
PRIVATE = {"APY10m10-AB-s0": (0.640, 0.132, 2.288), "YR10m10-F-s0": (0.800, 0.529, 1.733), "LTF-sample": (0.738, 0.270, 2.266),
           "DD-sample": (0.772, 0.267, 2.158), "G0014": (0.848, 0.228, 0.718)}
SOFT = {"WA-JEPA": (0.762, 0.626, 1.847)}   # board entry is not our configuration: reported, never fitted
MET = ("score", "km", "gt")


def meta():
    m = pd.read_csv(REPO / "experiments/alpasim/results/aud1/per_scene.csv")
    m = m[["scene", "log", "vehicle", "route0_x", "route0_y", "cmd_ap2", "v0", "a0x", "APY10m10-AB-s0|gt_dist_traveled_m"]].rename(
        columns={"APY10m10-AB-s0|gt_dist_traveled_m": "log_m"})
    c = pd.read_csv(REPO / "experiments/alpasim_proxy/results/scene_city.csv", names=["scene", "city"])
    m = m.merge(c, on="scene", how="left")
    cut = lambda x, b, lab: pd.cut(x, b, labels=lab, right=False).astype(object).fillna("none").astype(str)  # noqa: E731
    f = pd.DataFrame({"scene": m.scene})
    f["city"] = m.city.map({"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"})
    f["lht"] = np.where(m.city.eq("sg-one-north"), "LHT", "RHT")
    f["v0"] = cut(m.v0, [-1, 1, 5, 10, 1e9], ["<1", "1-5", "5-10", ">10"])
    f["stat"] = np.where((m.log_m < 5) & (m.v0 < 1), "nearstat", "moving")
    f["cmd"] = m.cmd_ap2.map({0.0: "L", 1.0: "S", 2.0: "R"}).fillna("none")
    f["ylat"] = cut(m.route0_y.abs(), [0, 2, 10, 1e9], ["<2", "2-10", ">10"])
    f["xfirst"] = cut(m.route0_x, [-1e9, 20, 35, 1e9], ["<20", "20-35", ">=35"])
    f["logm"] = cut(m.log_m, [-1, 5, 20, 40, 60, 1e9], ["<5", "5-20", "20-40", "40-60", ">60"])
    f["acc"] = cut(m.a0x, [-1e9, -1, 1, 1e9], ["brake", "-1..1", "accel"])
    return f.set_index("scene")


FACTORS = ("city", "lht", "v0", "stat", "cmd", "ylat", "xfirst", "logm", "acc")


def arrays(S, drivers, scenes):
    """driver -> dict of per-scene arrays aligned to `scenes`."""
    A = {}
    for d in drivers:
        x = S[S.driver == d].drop_duplicates("scene").set_index("scene").reindex(scenes)
        ok = x.dist_traveled_m.notna().to_numpy()
        A[d] = dict(score=x.score.fillna(0).to_numpy(), ok=ok, gt=np.where(ok, x.dist_to_gt_trajectory.fillna(0), 0),
                    dist=np.where(ok, x.dist_traveled_m.fillna(0), 0), af=np.where(ok, x.offroad_or_collision_at_fault.fillna(0), 0))
    return A


def metrics(a, w):
    wo = w * a["ok"]
    return np.array([(w * a["score"]).sum() / w.sum(), (wo * a["dist"]).sum() / max((wo * a["af"]).sum(), 1e-9) / 1000,
                     (wo * a["gt"]).sum() / wo.sum()])


def tr(v):  # per-metric transform: score -> log failure rate, km and GT -> log
    v = np.asarray(v, float)
    return np.stack([np.log(np.clip(1 - v[..., 0], 1e-6, None)), np.log(np.clip(v[..., 1], 1e-6, None)), np.log(v[..., 2])], -1)


def loss(Mp, Mq):
    """Mp proxy, Mq private, (n_anchor, 3): squared error of anchor-centred transformed metrics."""
    a, b = tr(Mp), tr(Mq)
    return float((((a - a.mean(0)) - (b - b.mean(0))) ** 2).sum())


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum()


def fit_strata(A, anchors, codes, n_lev, kl=0.0, base=None):
    """Weights per level (softmax), scene weight = level weight / level count. Returns scene weights, level weights, loss."""
    cnt = np.bincount(codes, minlength=n_lev).astype(float)
    base = cnt / cnt.sum() if base is None else base
    Mq = np.array([PRIVATE[d] for d in anchors])

    def wfun(th):
        p = np.exp(th - th.max())
        p /= p.sum()
        return p[codes] / np.maximum(cnt[codes], 1), p

    def f(th):
        w, p = wfun(th)
        pen = kl * float((p * np.log(np.clip(p, 1e-12, None) / base)).sum()) if kl else 0.0
        return loss(np.array([metrics(A[d], w) for d in anchors]), Mq) + pen

    best = None
    rng = np.random.default_rng(0)
    for s in range(12):   # multistart; the loss is non-convex in the level weights
        th0 = np.log(base + 1e-9) + (rng.normal(0, 1.5, n_lev) if s else 0)
        r = minimize(f, th0, method="Powell", options=dict(maxiter=20000, xtol=1e-4, ftol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    w, p = wfun(best.x)
    return w * len(w) / w.sum(), p, best.fun


def fit_tilt(A, anchors, X, l2):
    """Max-entropy tilting w = exp(X b) with an L2 penalty on b (upper bound of what reweighting can do)."""
    Mq = np.array([PRIVATE[d] for d in anchors])

    def f(b):
        z = X @ b
        w = np.exp(z - z.max())
        return loss(np.array([metrics(A[d], w) for d in anchors]), Mq) + l2 * float(b @ b)

    r = minimize(f, np.zeros(X.shape[1]), method="L-BFGS-B")
    z = X @ r.x
    w = np.exp(z - z.max())
    return w * len(w) / w.sum(), r.fun


def validity(A, anchors, w):
    M = {d: metrics(A[d], w) for d in A}
    out = dict(ess=ess(w))
    if {"APY10m10-AB-s0", "YR10m10-F-s0"} <= set(M):
        a, y = M["APY10m10-AB-s0"], M["YR10m10-F-s0"]
        out |= dict(d_score=y[0] - a[0], d_gt=y[2] - a[2], km_ratio=y[1] / a[1])
        out["V1"] = bool(y[0] > a[0] and y[2] < a[2] and y[1] > a[1])
        out["V2"] = bool(0.08 <= y[0] - a[0] <= 0.32 and -1.12 <= y[2] - a[2] <= -0.28 and 2 <= y[1] / a[1] <= 8)
    if len(anchors) >= 3:
        P = np.array([PRIVATE[d] for d in anchors])
        Q = np.array([M[d] for d in anchors])
        out["tau_score"] = kendalltau(P[:, 0], Q[:, 0]).statistic
        disc = []
        for j in (1, 2):   # discordant pairs, private ties (< 5 %) excluded
            n = 0
            for i, k in itertools.combinations(range(len(anchors)), 2):
                if abs(P[i, j] / P[k, j] - 1) < 0.05:
                    continue
                n += (P[i, j] - P[k, j]) * (Q[i, j] - Q[k, j]) < 0
            disc.append(int(n))
        out["disc_km"], out["disc_gt"] = disc
        out["V3"] = bool(out["tau_score"] >= 0.6 and max(disc) <= 1)
    out["V4_ess"] = bool(out["ess"] >= 300)
    return out, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", default="APY10m10-AB-s0,YR10m10-F-s0")
    ap.add_argument("--frozen", help="weights.npz of an earlier fit: evaluate those proxies unchanged on the current anchors (V5)")
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    S = pd.concat([pd.read_csv(t) for t in a.tables])
    F = meta()
    scenes = F.index.to_numpy()
    drivers = sorted(S.driver.unique())
    A = arrays(S, drivers, scenes)
    anchors = [d for d in a.anchors.split(",") if d in A]
    rep = dict(anchors=anchors, drivers=drivers)

    # G0: unweighted numbers reproduce AUD1-drift
    one = np.ones(len(scenes))
    P0 = {d: metrics(A[d], one) for d in drivers}
    g0 = np.allclose(P0["APY10m10-AB-s0"], [0.935, 0.863, 1.130], atol=6e-4) and np.allclose(P0["YR10m10-F-s0"], [0.925, 0.789, 0.941], atol=6e-4)
    rep["G0"] = bool(g0)
    print("G0", g0, {d: np.round(v, 4).tolist() for d, v in P0.items()})
    if not g0:
        raise SystemExit("G0 failed: unweighted metrics do not reproduce AUD1")

    # the 12 route-fold scenes fail before the driver runs, for every driver; the organisers state the private set has none: weight 0
    fold = (F.cmd == "none").to_numpy()
    rep["fold_scenes_dropped"] = int(fold.sum())
    F, scenes = F[~fold], scenes[~fold]
    A = arrays(S, drivers, scenes)
    one = np.ones(len(scenes))
    rows = []
    if a.frozen:   # V5: proxies fitted earlier, evaluated out of sample on every anchor now present
        Z = np.load(a.frozen, allow_pickle=True)   # scene ids are stored as an object array
        pos = pd.Series(np.arange(len(Z["scenes"])), index=Z["scenes"]).reindex(scenes).to_numpy()
        assert not np.isnan(pos).any(), "frozen weights miss scenes"
        fr = []
        for k in Z.files:
            if k == "scenes":
                continue
            w = Z[k][pos.astype(int)]
            v, M = validity(A, anchors, w)
            fr.append(dict(proxy=k, **v, **{f"{d}|{m}": M[d][i] for d in anchors for i, m in enumerate(MET)}))
        pd.DataFrame(fr).to_csv(o / "frozen_eval.csv", index=False)

    def add(name, w, fit_loss, extra=None):
        v, M = validity(A, anchors, w)
        r = dict(proxy=name, loss=fit_loss, **v, **(extra or {}))
        for d in drivers:
            r |= {f"{d}|{m}": M[d][i] for i, m in enumerate(MET)}
        rows.append(r)
        return r

    Mq = np.array([PRIVATE[d] for d in anchors])
    add("P0 unweighted 1479", one, loss(np.array([metrics(A[d], one) for d in anchors]), Mq))

    # strata table: per factor level, every driver's three metrics (descriptive)
    st = []
    for fac in FACTORS:
        for lev, idx in F.groupby(fac).indices.items():
            w = np.zeros(len(scenes))
            w[idx] = 1
            r = dict(factor=fac, level=lev, n=len(idx))
            for d in drivers:
                mm = metrics(A[d], w)
                r |= {f"{d}|{m}": mm[i] for i, m in enumerate(MET)}
            st.append(r)
    pd.DataFrame(st).to_csv(o / "strata.csv", index=False)

    # P1 single factor, P2 factor pairs with a KL penalty
    W = {}
    for fac in FACTORS:
        codes, levs = pd.factorize(F[fac])
        w, p, fl = fit_strata(A, anchors, codes, len(levs))
        W[f"P1 {fac}"] = w
        add(f"P1 {fac}", w, fl, dict(levels=json.dumps({str(l): round(float(x), 3) for l, x in zip(levs, p)})))
    for f1, f2 in itertools.combinations(FACTORS, 2):
        key = F[f1] + "|" + F[f2]
        codes, levs = pd.factorize(key)
        if len(levs) > 20:
            continue
        for kl in (0.01, 0.1):
            w, p, fl = fit_strata(A, anchors, codes, len(levs), kl=kl)
            W[f"P2 {f1}*{f2} kl{kl}"] = w
            add(f"P2 {f1}*{f2} kl{kl}", w, fl, dict(levels=json.dumps({str(l): round(float(x), 3) for l, x in zip(levs, p)})))
    # P3 max-entropy tilt on every factor's one-hot
    X = pd.get_dummies(F[list(FACTORS)].astype(str)).to_numpy(float)
    for l2 in (0.001, 0.01, 0.1):
        w, fl = fit_tilt(A, anchors, X, l2)
        W[f"P3 tilt l2{l2}"] = w
        add(f"P3 tilt l2{l2}", w, fl)
    R = pd.DataFrame(rows).sort_values("loss")
    R.to_csv(o / "fits.csv", index=False)

    # leave-one-anchor-out for the best few proxies (>= 4 anchors)
    if len(anchors) >= 4:
        loo = []
        for name in R.proxy.head(10):
            if not name.startswith(("P1", "P2")):
                continue
            hits = 0
            for held in anchors:
                tr_a = [d for d in anchors if d != held]
                parts = name.split()
                facs = parts[1].split("*")
                key = F[facs[0]] if len(facs) == 1 else F[facs[0]] + "|" + F[facs[1]]
                codes, levs = pd.factorize(key)
                kl = float(parts[2][2:]) if len(parts) > 2 else 0.0
                w, _, _ = fit_strata(A, tr_a, codes, len(levs), kl=kl)
                Q = {d: metrics(A[d], w)[0] for d in anchors}
                pos_q = sorted(anchors, key=lambda d: Q[d]).index(held)
                pos_p = sorted(anchors, key=lambda d: PRIVATE[d][0]).index(held)
                hits += pos_q == pos_p
            loo.append(dict(proxy=name, loo_hits=hits, n=len(anchors)))
        pd.DataFrame(loo).to_csv(o / "loo.csv", index=False)
        rep["loo"] = loo

    # P4: which public statistic orders the anchors like the private score
    P4 = []
    for stat, fn in (("score", lambda d: P0[d][0]), ("km", lambda d: P0[d][1]), ("-gt", lambda d: -P0[d][2]),
                     ("-zeros", lambda d: -float((A[d]["score"] == 0).sum())),
                     ("-exit4m", lambda d: -float(((A[d]["gt"] >= 4) & A[d]["ok"]).sum() / A[d]["ok"].sum()))):
        if len(anchors) >= 3:
            P4.append(dict(stat=stat, tau_vs_private_score=kendalltau([PRIVATE[d][0] for d in anchors], [fn(d) for d in anchors]).statistic,
                           **{d: fn(d) for d in anchors}))
    pd.DataFrame(P4).to_csv(o / "p4_metric_transfer.csv", index=False)
    rep["P0"] = {d: dict(zip(MET, np.round(v, 4).tolist())) for d, v in P0.items()}
    rep["private"] = {d: dict(zip(MET, PRIVATE[d])) for d in anchors} | {f"{d} (soft)": dict(zip(MET, v)) for d, v in SOFT.items()}
    rep["best"] = R.head(15).drop(columns=[c for c in R.columns if "|" in c]).to_dict("records")
    (o / "summary.json").write_text(json.dumps(rep, indent=1, default=float))
    np.savez_compressed(o / "weights.npz", scenes=scenes, **{k.replace(" ", "_"): v for k, v in W.items()})
    cols = ["proxy", "loss", "ess", "d_score", "d_gt", "km_ratio", "V1", "V2"] + [c for c in ("tau_score", "disc_km", "disc_gt", "V3") if c in R]
    print(R[cols].head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
