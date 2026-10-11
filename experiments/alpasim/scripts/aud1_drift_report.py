#!/usr/bin/env python3
"""AUD1-drift read-out (Mac, numpy + pandas + matplotlib) from aud1_drift_collect.py's tables and aud1's per_scene.csv (strata metadata).

  aud1_drift_report.py <collect dir> <aud1 dir> <out dir>   -> <out dir>/*.csv, *.json, drift_*.png / .pdf, markdown tables on stdout
CIs: log-clustered bootstrap (44 logs; 10 000 draws for run-level metrics, 4 000 for slopes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAND = ["SH30-F-s0", "AP2-AB-s0", "P2H10-F-s0", "P2H10-F-s1", "YR10m10-F-s0", "YR10m10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1", "WA-JEPA"]
RANKED = ["SH30-F-s0", "AP2-AB-s0", "P2H10-F-s0", "P2H10-F-s1", "YR10m10-F-s0", "YR10m10-F-s1", "APY10m10-AB-s0", "APY10m10-AB-s1"]
KS = np.arange(1, 11)
KFIT = np.arange(4, 11)
rng = np.random.default_rng(0)


def md(df, fl=3):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(":--" if df[c].dtype == object else "--:" for c in cols) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(f"{v:.{fl}f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(L)


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def slope(y):
    """OLS slope over KFIT of y (..., len(KFIT))."""
    x = KFIT - KFIT.mean()
    return (y - y.mean(-1, keepdims=True)) @ x / (x @ x)


def main():
    i, a1, o = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    o.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(i / "per_scene_k.parquet")
    S = pd.read_csv(i / "per_scene.csv")
    meta = pd.read_csv(a1 / "per_scene.csv")[["scene", "log", "vehicle", "route0_x", "v0", "APY10m10-AB-s0|gt_dist_traveled_m"]].rename(
        columns={"APY10m10-AB-s0|gt_dist_traveled_m": "log_path_m"})
    meta["day"] = meta.scene.str[:10]
    S = S.merge(meta, on="scene", how="left")
    P = P.merge(meta[["scene", "log"]], on="scene", how="left").merge(S[["driver", "scene", "score"]], on=["driver", "scene"], how="left")
    # heading of the logged path is undefined where the logged car hardly moves: no heading error on logs shorter than 5 m
    P = P.merge(meta[["scene", "log_path_m"]], on="scene", how="left")
    P.loc[~(P.log_path_m >= 5), ["head_err", "head_err_time"]] = np.nan
    P["abs_head"] = P.head_err.abs()
    P = P.sort_values(["driver", "scene", "k"])
    P["cum4"] = P.groupby(["driver", "scene"]).dist_gt.cummax() >= 4
    logs = np.array(sorted(meta.log.unique()))
    L = len(logs)
    out = {}

    # ---------- 1. drift by decision index ----------
    curves, slopes = [], []
    BS = 4000
    draws = rng.integers(0, L, (BS, L))
    W = np.stack([np.bincount(d, minlength=L) for d in draws])            # (BS, L) multiplicity of each log
    for sub in ("all", "no_zero"):
        sl_store = {}
        for drv in CAND:
            x = P[(P.driver == drv) & P.k.between(1, 10) & P.dist_gt.notna()]
            if sub == "no_zero":
                x = x[x.score > 0]
            for k, g in x.groupby("k"):
                curves.append(dict(subset=sub, driver=drv, k=k, n=len(g), dist_mean=g.dist_gt.mean(), dist_p90=g.dist_gt.quantile(.9),
                                   head_abs_mean=g.abs_head.mean(), head_abs_p90=g.abs_head.quantile(.9), head_sd=g.head_err.std(),
                                   lat_signed_mean=g.lat_signed.mean(), frac_ge4_by_k=g.cum4.mean(), head_time_abs_mean=g.head_err_time.abs().mean()))
            # per-log sums for the bootstrap: (L, len(KFIT))
            xi = x[x.k.isin(KFIT)]
            li = pd.Categorical(xi.log, categories=logs).codes
            ki = xi.k.to_numpy() - KFIT[0]
            def lsum(v):
                M = np.zeros((L, len(KFIT)))
                np.add.at(M, (li, ki), v)
                return M
            n_, d_, h_ = lsum(np.ones(len(xi))), lsum(xi.dist_gt.to_numpy()), lsum(xi.abs_head.fillna(0).to_numpy())
            nh = lsum(xi.abs_head.notna().to_numpy().astype(float))
            pt = slope(d_.sum(0) / n_.sum(0)), slope(h_.sum(0) / nh.sum(0))
            bd, bh = slope((W @ d_) / (W @ n_)), slope((W @ h_) / (W @ nh))
            sl_store[drv] = (bd, bh)
            slopes.append(dict(subset=sub, driver=drv, dist_slope_m_per_dec=pt[0], dist_slope_CI=ci(bd), head_slope_deg_per_dec=pt[1],
                               head_slope_CI=ci(bh)))
        for r in slopes:
            if r["subset"] == sub and r["driver"] != "APY10m10-AB-s0":
                r["dist_slope_minus_APY_s0_CI"] = ci(sl_store[r["driver"]][0] - sl_store["APY10m10-AB-s0"][0])
                r["head_slope_minus_APY_s0_CI"] = ci(sl_store[r["driver"]][1] - sl_store["APY10m10-AB-s0"][1])
    C = pd.DataFrame(curves)
    SL = pd.DataFrame(slopes)
    C.to_csv(o / "drift_by_decision.csv", index=False)
    SL.to_csv(o / "drift_slopes.csv", index=False)

    # ---------- 2. leaderboard metrics ----------
    def lsums(x):
        ok = x.dist_traveled_m.notna()
        li = pd.Categorical(x.log, categories=logs).codes
        f = lambda v: np.bincount(li, weights=v, minlength=L)  # noqa: E731
        return np.c_[f(x.score.to_numpy()), f(np.ones(len(x))), f(np.where(ok, x.dist_traveled_m.fillna(0), 0)),
                     f(np.where(ok, x.offroad_or_collision_at_fault.fillna(0), 0)), f(np.where(ok, x.dist_to_gt_trajectory.fillna(0), 0)),
                     f(ok.to_numpy().astype(float))]

    def rl(M):   # M (..., L, 6) per-log sums -> score, at-fault km, GT m, incidents per km
        return rs(M.sum(-2))

    def rs(s):   # s (..., 6) pooled sums
        return np.stack([s[..., 0] / s[..., 1], s[..., 2] / np.maximum(s[..., 3], 1e-9) / 1000, s[..., 4] / s[..., 5], s[..., 3] / (s[..., 2] / 1000)], -1)
    BR = 10000
    W2 = np.stack([np.bincount(d, minlength=L) for d in rng.integers(0, L, (BR, L))]).astype(float)
    rows, LS = [], {}
    for drv in CAND:
        x = S[S.driver == drv]
        M = lsums(x)
        LS[drv] = M
        pt = rl(M)
        b = rl(np.einsum("bl,lj->blj", W2, M))
        ok = x.dist_traveled_m.notna()
        rows.append({"driver": drv, "scenes": len(x), "score": pt[0], "score CI": ci(b[:, 0]), "GT dist m": pt[2], "GT CI": ci(b[:, 2]),
                     "at-fault dist km": pt[1], "km CI": ci(b[:, 1]), "at-fault per km": pt[3], "per-km CI": ci(b[:, 3]),
                     "at-fault events": int(x.loc[ok, "offroad_or_collision_at_fault"].sum()), "zeros": int((x.score == 0).sum()),
                     "collision zeros": int(((x.score == 0) & (x.collision_at_fault > 0)).sum()),
                     "offroad zeros": int(((x.score == 0) & (x.offroad > 0)).sum()),
                     "corridor zeros": int(((x.score == 0) & (x.left_corridor_laterally > 0)).sum()),
                     "share GT >= 4 m": float((x.loc[ok, "dist_to_gt_trajectory"] >= 4).mean()), "km driven": x.loc[ok, "dist_traveled_m"].sum() / 1000})
    T2 = pd.DataFrame(rows)
    T2.to_csv(o / "runlevel.csv", index=False)
    # paired against the submitted checkpoint (same bootstrap draws for both drivers)
    rows, bb = [], rl(np.einsum("bl,lj->blj", W2, LS["APY10m10-AB-s0"]))
    for drv in CAND:
        if drv == "APY10m10-AB-s0":
            continue
        d = rl(LS[drv]) - rl(LS["APY10m10-AB-s0"])
        b = rl(np.einsum("bl,lj->blj", W2, LS[drv])) - bb
        rows.append({"driver": drv, "score - APY s0": d[0], "CI": ci(b[:, 0]), "GT m - APY s0": d[2], "GT CI": ci(b[:, 2]),
                     "at-fault per km - APY s0": d[3], "per-km CI": ci(b[:, 3])})
    T2p = pd.DataFrame(rows)
    T2p.to_csv(o / "paired_vs_apy_s0.csv", index=False)

    # ---------- 3. transfer proxies ----------
    rows, grp = [], []
    for drv in CAND:
        x = S[S.driver == drv]
        ok = x.dist_traveled_m.notna()
        r = {"driver": drv}
        for by in ("vehicle", "day", "log"):
            g = x.groupby(by).agg(score=("score", "mean"), n=("score", "size"))
            g["gtm"] = x[ok].groupby(by).dist_to_gt_trajectory.mean()
            for lev, v in g.iterrows():
                grp.append(dict(driver=drv, by=by, level=lev, n=int(v.n), score=v.score, gt_m=v.gtm))
            r |= {f"{by} score sd": g.score.std(), f"{by} score worst-best": g.score.min() - g.score.max(), f"{by} GT sd": g.gtm.std(),
                  f"{by} GT worst-best": g.gtm.max() - g.gtm.min(), f"{by} worst score": g.score.min()}
        M = LS[drv]
        lolo = rs(M.sum(0) - M)    # (L, 4): run-level with each log left out
        r |= {"LOLO score range": lolo[:, 0].max() - lolo[:, 0].min(), "LOLO score sd": lolo[:, 0].std(ddof=1),
              "LOLO GT range": lolo[:, 2].max() - lolo[:, 2].min(), "LOLO at-fault km range": lolo[:, 1].max() - lolo[:, 1].min()}
        rows.append(r)
    T3 = pd.DataFrame(rows)
    T3.to_csv(o / "transfer_spread.csv", index=False)
    pd.DataFrame(grp).to_csv(o / "by_group.csv", index=False)

    # ---------- 4. hard strata ----------
    S["near_stationary"] = (S.log_path_m < 5) & (S.v0 < 1)
    S["route_beside"] = S.route0_x.between(0, 20, inclusive="left")
    rows = []
    for st in ("near_stationary", "route_beside"):
        for drv in CAND:
            x = S[(S.driver == drv) & S[st]]
            ok = x.dist_traveled_m.notna()
            rows.append(dict(stratum=st, driver=drv, n=len(x), score=x.score.mean(), zeros=int((x.score == 0).sum()),
                             collision_zeros=int(((x.score == 0) & (x.collision_at_fault > 0)).sum()),
                             at_fault_events=int(x.loc[ok, "offroad_or_collision_at_fault"].sum()), gt_m=x.loc[ok, "dist_to_gt_trajectory"].mean(),
                             driven_m=x.loc[ok, "dist_traveled_m"].mean(), log_m=x.loc[ok, "gt_dist_traveled_m"].mean()))
    T4 = pd.DataFrame(rows)
    T4.to_csv(o / "hard_strata.csv", index=False)

    # ---------- 5. summary ----------
    sa = SL[SL.subset == "all"].set_index("driver")
    sn = SL[SL.subset == "no_zero"].set_index("driver")
    t2, t3 = T2.set_index("driver"), T3.set_index("driver")
    hz = T4.groupby("driver").zeros.sum()
    Su = pd.DataFrame({"drift slope m/dec (all)": sa.dist_slope_m_per_dec, "drift slope m/dec (no-zero)": sn.dist_slope_m_per_dec,
                       "heading slope deg/dec (no-zero)": sn.head_slope_deg_per_dec, "GT dist m": t2["GT dist m"],
                       "at-fault per km": t2["at-fault per km"], "LOLO score range": t3["LOLO score range"], "vehicle score sd": t3["vehicle score sd"],
                       "hard-strata zeros": hz, "public score": t2.score}).loc[CAND]
    keys = ["drift slope m/dec (all)", "drift slope m/dec (no-zero)", "heading slope deg/dec (no-zero)", "GT dist m", "at-fault per km",
            "LOLO score range", "vehicle score sd", "hard-strata zeros"]
    R = Su.loc[RANKED, keys].rank(method="min")
    R["public score"] = Su.loc[RANKED, "public score"].rank(ascending=False, method="min")
    Su["mean rank (excl. public score)"] = R[keys].mean(axis=1)
    Su["public-score rank"] = R["public score"]
    Su.reset_index().rename(columns={"index": "driver"}).to_csv(o / "summary.csv", index=False)
    out = dict(n_logs=L)
    (o / "summary.json").write_text(json.dumps(out))

    for name, T in (("1 curves (k=1,4,7,10)", C[C.k.isin([1, 4, 7, 10])]), ("1 slopes", SL), ("2 runlevel", T2), ("2 paired", T2p), ("3 spread", T3),
                    ("4 strata", T4), ("5 summary", Su.reset_index().rename(columns={"index": "driver"}))):
        print(f"\n## {name}\n" + md(T))
    plots(C, o)


def plots(C, o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research"))
    import plot_style as PS
    PS.apply()
    col = {"SH30-F": PS.PALETTE["sky_blue"], "AP2-AB": PS.PALETTE["green"], "P2H10-F": PS.PALETTE["orange"], "YR10m10-F": PS.PALETTE["purple"],
           "APY10m10-AB": PS.PALETTE["blue"], "WA-JEPA": "#555555"}
    spec = [("dist_mean", "distance from logged path, mean (m)", "dist_mean"), ("dist_p90", "distance from logged path, p90 (m)", "dist_p90"),
            ("head_abs_mean", "|heading error| vs logged path, mean (deg)", "head_mean"), ("head_abs_p90", "|heading error| vs logged path, p90 (deg)", "head_p90"),
            ("frac_ge4_by_k", "share of scenes already >= 4 m off the path", "frac_ge4")]
    for c, lab, stem in spec:
        fig, ax = plt.subplots(1, 2, figsize=(PS.DOUBLE_COLUMN_IN, 2.7), sharey=True)
        for i, sub in enumerate(("all", "no_zero")):
            for drv in CAND:
                g = C[(C.subset == sub) & (C.driver == drv)]
                fam = drv.rsplit("-s", 1)[0] if drv != "WA-JEPA" else drv
                ax[i].plot(g.k, g[c], color=col[fam], ls="--" if drv.endswith("-s1") else "-", lw=1.4 if drv == "APY10m10-AB-s0" else 1.0,
                           marker="o", ms=2.5, label=drv)
            ax[i].set_xlabel("decision index k (state after k decisions, 0.5 s each)")
            ax[i].set_xticks(range(1, 11))
            PS.panel(ax[i], "all scored scenes" if sub == "all" else "scenes with score > 0 (this driver)")
        ax[0].set_ylabel(lab)
        ax[1].legend(fontsize=6.5, loc="upper left")
        fig.tight_layout()
        PS.save(fig, o / f"drift_{stem}")
        plt.close(fig)


if __name__ == "__main__":
    main()
