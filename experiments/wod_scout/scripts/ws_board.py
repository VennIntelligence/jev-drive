"""WODSCOUT Q1 / A2: what the public WOD-E2E leaderboard says about the test-only Spotlight cluster (plans/2026-10-10-wodscout-prereg.md).

  .venv/bin/python experiments/wod_scout/scripts/ws_board.py            (Mac or box, CPU, seconds)

Input: exports/waymo-e2e-2026-10-08/leaderboard.json (155 rows, the user's export; untracked) -> the per-cluster test scores of every row are
written to results/q1_spotlight/board_clusters.csv, which the script reads when the export is absent. Sample: the best row of each account with
RFS >= 7.5. Reads: Spearman of Spotlight with each other cluster across accounts (bootstrap over accounts), the partial Spearman given the mean of
the other nine, leave-one-out linear prediction of Spotlight from the ten, and the proxy criteria (i) - (iii). No test label is involved: these are
the cluster means the official page shows for every entry.
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research")]
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

OUT, FIG = _R / "experiments/wod_scout/results/q1_spotlight", _R / "experiments/wod_scout/figs"
COLS = {"construction_score": "Construction", "intersection_score": "Interections", "pedestrian_score": "Pedestrian", "cyclist_score": "Cyclist",
        "multi_lane_maneuver_score": "Multi-Lane Maneuvers", "single_lane_maneuver_score": "Single-Lane Maneuvers", "cut_in_score": "Cut_ins",
        "foreign_object_debris_score": "Foreign Object Debris", "special_vehicle_score": "Special Vehicles", "others_score": "Others", "spotlight_score": "Spotlight"}
WLG = {"Construction": 8.725, "Single-Lane Maneuvers": 8.377, "Cut_ins": 8.353, "Special Vehicles": 8.289, "Foreign Object Debris": 8.265, "Multi-Lane Maneuvers": 8.079,
       "Interections": 8.073, "Cyclist": 8.007, "Pedestrian": 7.957, "Others": 7.616, "Spotlight": 7.342}                # decision 180, not on the public board
B, MIN_RFS = 4000, 7.5


def sp(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def psp(x, y, z):
    """Partial Spearman of x and y given z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = _R / "exports/waymo-e2e-2026-10-08/leaderboard.json"
    if src.exists():
        d = json.loads(src.read_text())
        df = pd.DataFrame([{"method": r["method_name"], "account": r["obfuscated_email"], "params": r.get("num_model_parameters", ""), "RFS": r["average_score"],
                            "ADE@5s": r["ade_at_five_sec"], **{v: r[k] for k, v in COLS.items()}} for r in d]).drop_duplicates()
        df.to_csv(OUT / "board_clusters.csv", index=False)
    df = pd.read_csv(OUT / "board_clusters.csv")
    best = df.sort_values("RFS", ascending=False).groupby("account", as_index=False).first()
    s = best[best.RFS >= MIN_RFS].reset_index(drop=True)
    cl = [c for c in COLS.values() if c != "Spotlight"]
    X, y = s[cl].to_numpy(), s["Spotlight"].to_numpy()
    n = len(s)
    rng = np.random.default_rng(0)
    draws = rng.integers(n, size=(B, n))
    val = pd.read_csv(_R / "experiments/wod_scout/results/q3_strata/val_vs_test.csv").set_index("cluster")
    rows = []
    for j, c in enumerate(cl + ["mean of the 10"]):
        x = X[:, j] if j < len(cl) else X.mean(1)
        rest = np.delete(X, j, 1).mean(1) if j < len(cl) else None
        r = sp(x, y)
        bs = np.array([sp(x[k], y[k]) for k in draws])
        row = {"cluster": c, "Spearman with Spotlight": r, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "sd across accounts": float(x.std(ddof=1)),
               "mean": float(x.mean())}
        if rest is not None:
            p = psp(x, y, rest)
            bp = np.array([psp(x[k], y[k], rest[k]) for k in draws])
            row |= {"partial Spearman | mean of the other 9": p, "p_lo": np.nanpercentile(bp, 2.5), "p_hi": np.nanpercentile(bp, 97.5)}
        key = c if c in val.index else "mean of the 10"
        inside = bool(val.loc[key, "test inside val CI"])
        c1 = bool(r >= 0.70 and row["lo"] >= 0.50)
        c2 = bool(rest is not None and row["p_lo"] > 0)
        row |= {"(i) Spearman >= 0.70, lo >= 0.50": c1, "(ii) partial lo > 0": c2, "(iii) WLG test inside val CI": inside,
                "verdict": "usable proxy" if c1 and c2 and inside else "level proxy only" if c1 else "no"}
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "board_proxy.csv", index=False)
    (OUT / "board_proxy.md").write_text(T.to_markdown(index=False, floatfmt=".3f") + f"\n\nn = {n} accounts (best row per account, RFS >= {MIN_RFS}); bootstrap over accounts, B = {B}, seed 0.\n")
    # leave-one-out linear prediction of Spotlight from the ten clusters, and from their mean alone
    def loo(F):
        pr = np.empty(n)
        for i in range(n):
            m = np.arange(n) != i
            A = np.c_[np.ones(m.sum()), F[m]]
            pr[i] = np.r_[1, F[i]] @ np.linalg.lstsq(A, y[m], rcond=None)[0]
        return pr, 1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    p10, r10 = loo(X)
    p1, r1 = loo(X.mean(1, keepdims=True))
    A = np.c_[np.ones(n), X.mean(1)]
    co = np.linalg.lstsq(A, y, rcond=None)[0]
    resid_sd = float((y - A @ co).std(ddof=2))
    wl_mean = float(np.mean([WLG[c] for c in cl]))
    wl_pred = float(co[0] + co[1] * wl_mean)
    s["spot_resid"] = y - A @ co
    top = s.sort_values("RFS", ascending=False).head(15)
    small = s[s.params.astype(str).str.contains("M|k|K", regex=True)]
    big = s[s.params.astype(str).str.contains("B")]
    meta = dict(n_rows=int(len(df)), n_accounts=int(best.account.nunique()), n_sample=int(n), loo_r2_ten_clusters=float(r10), loo_r2_mean_only=float(r1),
                slope_spot_on_mean=float(co[1]), intercept=float(co[0]), resid_sd=resid_sd, wlg_mean10=wl_mean, wlg_spot=WLG["Spotlight"], wlg_spot_pred=wl_pred,
                wlg_spot_resid=WLG["Spotlight"] - wl_pred, wlg_resid_z=(WLG["Spotlight"] - wl_pred) / resid_sd,
                wlg_spot_rank_in_sample=int((y > WLG["Spotlight"]).sum() + 1), spot_mean=float(y.mean()), spot_sd=float(y.std(ddof=1)),
                spot_minus_mean10=float((y - X.mean(1)).mean()), spot_minus_mean10_sd=float((y - X.mean(1)).std(ddof=1)),
                big_n=int(len(big)), small_n=int(len(small)),
                big_minus_small={c: float(big[c].mean() - small[c].mean()) for c in cl + ["Spotlight"]} if len(big) and len(small) else {},
                top_resid=top[["method", "RFS", "Spotlight", "spot_resid"]].round(3).to_dict("records"))
    (OUT / "board_meta.json").write_text(json.dumps(meta, indent=1))
    print(T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(json.dumps({k: v for k, v in meta.items() if k != "top_resid"}, indent=1))
    print(top[["method", "params", "RFS", "Spotlight", "spot_resid"]].to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.8))
    xm = X.mean(1)
    ax.scatter(xm, y, s=9, color=ps.BASELINE, label=f"accounts (n = {n})", zorder=2)
    xs = np.linspace(xm.min(), xm.max(), 50)
    ax.plot(xs, co[0] + co[1] * xs, color="#444444", lw=0.8, zorder=1)
    ax.scatter([wl_mean], [WLG["Spotlight"]], s=42, marker="*", color=ps.PALETTE["blue"], label="WLG", zorder=3)
    ax.set_xlabel("mean of the 10 other test clusters (RFS)")
    ax.set_ylabel("Spotlight (RFS)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, FIG / "q1_spotlight_vs_rest")
    (FIG / "q1_spotlight_vs_rest.pdf").unlink()


if __name__ == "__main__":
    main()
