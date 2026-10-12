"""FAMILY-Q: per-driver slopes of scene score / GT distance / k = 10 drift on per-asset quality measures, the family difference
(mean of YR, APY minus mean of LTF, DD, garage) with a 44-log cluster bootstrap, tail strata, and the 95th-percentile extrapolation
(plans/2026-10-12-quality-prereg.md).

  quality_report.py --out DIR [--q5 q5.csv]      (Mac, project venv; reads tmp/aud1_drift, tmp/family/q, the proxy anchors)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[3]
FAM, SAMP = ["YR10m10-F-s0", "APY10m10-AB-s0"], ["LTF-sample", "DD-sample", "G0014"]
DRV = FAM + SAMP + ["WA-JEPA"]
QS = ["pose_t_m", "pose_r_deg", "fyfx_dev", "k1_dev", "bg_bytes", "pitch_mean", "dz_end"]
B = 2000


def load(q5):
    st = pd.read_csv(R / "tmp/aud1_drift/per_scene.csv")
    an = [pd.read_csv(R / f"experiments/alpasim_proxy/results/anchors/{k}.csv") for k in ("ltf", "dd", "g0014")]
    S = pd.concat([st] + an)
    S = S[S.driver.isin(DRV)].dropna(subset=["dist_to_gt_trajectory"])
    K = pd.concat([pd.read_parquet(R / "tmp/aud1_drift/per_scene_k.parquet")] + [pd.read_parquet(R / f"tmp/family/q/k_{k}.parquet") for k in ("ltf", "dd", "g")])
    k10 = K[K.k == 10][["driver", "scene", "dist_gt"]].rename(columns={"dist_gt": "d10"})
    S = S.merge(k10, on=["driver", "scene"], how="left")
    Q = pd.read_csv(R / "experiments/alpasim_family/results/asset_quality_public.csv").merge(
        pd.read_csv(R / "experiments/alpasim_family/results/asset_grade_public.csv")[["scene", "pitch_mean", "dz_end"]], on="scene")
    if q5:
        Q = Q.merge(pd.read_csv(q5), on="scene", how="left")
    S = S.merge(Q, on="scene")
    S["log"] = S.scene.str.rsplit("-", n=1).str[0]
    return S, Q


def slopes(S, q, y, logs, idx):
    out = {}
    for d in DRV:
        x = S[(S.driver == d)]
        x = x.set_index("log").loc[[logs[i] for i in idx]] if idx is not None else x
        x = x.dropna(subset=[q, y])
        z = (x[q] - MU[q]) / SD[q]
        out[d] = np.polyfit(z, x[y], 1)[0] if len(x) > 10 and z.std() > 0 else np.nan
    out["family_diff"] = np.mean([out[d] for d in FAM]) - np.mean([out[d] for d in SAMP])
    return out


def resample(df, pick):
    g = {k: v for k, v in df.groupby("log")}
    parts = [g[l] for l in pick if l in g]
    return pd.concat(parts) if parts else df.iloc[:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--q5")
    ap.add_argument("--skip-slopes", action="store_true")
    a = ap.parse_args()
    S, Q = load(a.q5)
    qs = QS + ([c for c in ("q5_l1", "q5_ncc") if c in S] if a.q5 else [])
    global MU, SD
    MU, SD = Q[qs].mean(), Q[qs].std()
    logs = sorted(S.log.unique())
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(logs), len(logs)) for _ in range(B)]
    rows = []
    for q in ([] if a.skip_slopes else qs):
        for y in ("score", "dist_to_gt_trajectory", "d10"):
            pt = slopes(S, q, y, logs, None)
            bs = pd.DataFrame([slopes(S, q, y, logs, i) for i in draws])
            for k, v in pt.items():
                lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
                slo, shi = np.nanpercentile(bs[k], [0.3125, 99.6875])     # Bonferroni over 8 tests
                rows.append(dict(measure=q, outcome=y, driver=k, slope_per_sd=pt[k], lo=lo, hi=hi, lo_bonf=slo, hi_bonf=shi))
            print(q, y, "family diff", round(pt["family_diff"], 4), np.round(np.nanpercentile(bs["family_diff"], [2.5, 97.5]), 4), flush=True)
    T = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        T.to_csv(out / "slopes.csv", index=False)
    # tails: worst 10 % / 5 % of each measure (largest value; smallest for pitch_mean, bg_bytes, q5_ncc); log bootstrap on per-log sums
    low = ("pitch_mean", "bg_bytes", "q5_ncc")
    D = np.array(draws[:1000])
    L = {l: i for i, l in enumerate(logs)}

    def logsum(x, col):
        s = np.zeros(len(logs)); n = np.zeros(len(logs))
        np.add.at(s, x.log.map(L).values, x[col].values); np.add.at(n, x.log.map(L).values, 1)
        return s, n
    tails = []
    for q in qs:
        v = Q.set_index("scene")[q].dropna()
        for p in (0.10, 0.05):
            worst = set(v[v <= v.quantile(p)].index) if q in low else set(v[v >= v.quantile(1 - p)].index)
            for d in DRV:
                x = S[S.driver == d]
                t, r = x[x.scene.isin(worst)], x[~x.scene.isin(worst)]
                (ts, tn), (rs, rn) = logsum(t, "score"), logsum(r, "score")
                diff = ts[D].sum(1) / np.maximum(tn[D].sum(1), 1) - rs[D].sum(1) / rn[D].sum(1)
                tails.append(dict(measure=q, tail=p, driver=d, n=len(t), score=t.score.mean(), rest=r.score.mean(), d=t.score.mean() - r.score.mean(),
                                  d_lo=np.percentile(diff, 2.5), d_hi=np.percentile(diff, 97.5), gt=t.dist_to_gt_trajectory.mean(),
                                  gt_rest=r.dist_to_gt_trajectory.mean()))
    pd.DataFrame(tails).to_csv(out / "tails.csv", index=False)
    # extrapolation of the score to the 95th percentile of each measure (5th for the low-is-worse ones); log bootstrap on per-log moments
    ex = []
    for q in qs:
        v = Q[q].dropna()
        z95 = (v.quantile(0.05 if q in low else 0.95) - MU[q]) / SD[q]
        for d in DRV:
            x = S[S.driver == d].dropna(subset=[q]).copy()
            x["z"] = (x[q] - MU[q]) / SD[q]
            x["zz"], x["zy"] = x.z ** 2, x.z * x.score
            M = {c: logsum(x, c)[0] for c in ("z", "zz", "zy", "score")}
            n = logsum(x, "z")[1]
            def fit(idx):
                N, sz, szz, szy, sy = (np.sum(a[idx], -1) for a in (n, M["z"], M["zz"], M["zy"], M["score"]))
                b = (N * szy - sz * sy) / (N * szz - sz ** 2)
                return (sy - b * sz) / N + b * z95
            pr = fit(D)
            ex.append(dict(measure=q, at=v.quantile(0.05 if q in low else 0.95), driver=d, mean=x.score.mean(), pred=float(fit(np.arange(len(logs)))),
                           lo=np.percentile(pr, 2.5), hi=np.percentile(pr, 97.5)))
    pd.DataFrame(ex).to_csv(out / "extrapolate.csv", index=False)


if __name__ == "__main__":
    main()
