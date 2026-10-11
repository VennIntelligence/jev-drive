#!/usr/bin/env python3
"""PROXY P4: which PUBLIC statistic orders the anchors like their PRIVATE leaderboard numbers (metric transfer instead of scene
reweighting). Per statistic: the value per driver with a log-clustered 95 % CI (44 logs, 4000 draws), Kendall tau against each private
metric over the anchors, and the same statistic for every stored candidate (what a selection by that statistic would pick).

  proxy_transfer.py <per_scene.csv> [...] --anchors A,B,... --out DIR [--drift <per_scene_k.parquet> ...]      (Mac, CPU)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_fit as P  # noqa: E402

STATS = ("score", "-gt", "km", "-exit4m", "-zeros", "-atfault_per_clip", "-drift_slope")


def stat_values(a, w):
    """a: arrays of one driver (proxy_fit.arrays); w: scene weights (bootstrap multiplicities). -> dict of statistics."""
    wo = w * a["ok"]
    sc, km, gt = P.metrics(a, w)
    return {"score": sc, "-gt": -gt, "km": km, "-exit4m": -(wo * (a["gt"] >= 4)).sum() / wo.sum(),
            "-zeros": -(w * (a["score"] == 0)).sum() / w.sum(), "-atfault_per_clip": -(wo * a["af"]).sum() / wo.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drift", nargs="*", default=[])
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    S = pd.concat([pd.read_csv(t) for t in a.tables])
    F = P.meta()
    keep = (F.cmd != "none").to_numpy()   # the 12 route-fold scenes, as in proxy_fit
    scenes = F.index.to_numpy()[keep]
    logs = pd.read_csv(P.REPO / "experiments/alpasim/results/aud1/per_scene.csv").set_index("scene").log.reindex(scenes).to_numpy()
    drivers = sorted(S.driver.unique())
    A = P.arrays(S, drivers, scenes)
    anchors = a.anchors.split(",")
    # drift slope k = 4..10 (AUD1-drift definition), point value only
    slope = {}
    if a.drift:
        K = pd.concat([pd.read_parquet(f, columns=["driver", "scene", "k", "dist_gt"]) for f in a.drift])
        K = K[K.k.between(4, 10) & K.scene.isin(scenes)]
        for d, g in K.groupby("driver"):
            m = g.groupby("k").dist_gt.mean()
            slope[d] = -np.polyfit(m.index, m.values, 1)[0]
    L, codes = np.unique(logs, return_inverse=True)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(L), (4000, len(L)))
    rows = []
    for d in drivers:
        pt = stat_values(A[d], np.ones(len(scenes)))
        bs = [stat_values(A[d], np.bincount(dr, minlength=len(L))[codes].astype(float)) for dr in draws[:1000]]
        r = dict(driver=d, anchor=d in anchors, **{f"private_{m}": v for m, v in zip(P.MET, P.PRIVATE.get(d, (np.nan,) * 3))})
        for s in STATS:
            if s == "-drift_slope":
                r[s] = slope.get(d, np.nan)
                continue
            v = np.array([b[s] for b in bs])
            r[s], r[s + "_lo"], r[s + "_hi"] = pt[s], np.percentile(v, 2.5), np.percentile(v, 97.5)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(o / "transfer_values.csv", index=False)
    X = T[T.anchor].set_index("driver").loc[anchors]
    tau = []
    for s in STATS:
        if X[s].isna().any():
            continue
        tau.append(dict(stat=s, **{f"tau_vs_private_{m}": kendalltau(X[f"private_{m}"] * (-1 if m == "gt" else 1), X[s]).statistic for m in P.MET}))
    TT = pd.DataFrame(tau)
    TT.to_csv(o / "transfer_tau.csv", index=False)
    print(TT.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(T[["driver", "anchor", "private_score", "private_gt", "score", "-gt", "-gt_lo", "-gt_hi", "-exit4m", "km", "-drift_slope"]]
          .sort_values("-gt", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
