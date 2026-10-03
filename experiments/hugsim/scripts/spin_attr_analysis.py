"""HUGSIM spin attribution, Q1 / Q3 (Mac, CPU).  Plan: experiments/hugsim/plans/2026-10-04-spin-attribution-prereg.md
    python spin_attr_analysis.py <traces.json> <out_dir>
Writes features.csv (one row per scenario x agent), q1_auc.csv, q1_loo.json, q1_blocked.json, q3.csv / q3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1] / "results"
K = 40          # window cap (steps of 0.25 s)
LAUNCH = 5      # launch window: steps 0-4, before any spinner's divergence start (min 5)
OBST = 25.0     # m, "obstacle ahead" range


def phi1(step):
    """Direction of the plan point at 1 s (deg, + right as the simulator) and its length (m)."""
    p = step.get("plan")
    if not p or len(p) < 2:
        return np.nan, 0.0
    x, y = p[1]
    return float(np.degrees(np.arctan2(x, max(y, 1e-6)))), float(np.hypot(x, y))


def feats(t, start):
    S = t["steps"]
    v = np.array([s["v"] for s in S])
    n = len(v)
    W = int(min(K, n, start if np.isfinite(start) else K))      # pre-divergence window
    W = max(W, 1)
    vw = v[:W]
    f = {}
    f["win"] = W
    f["frac_v3"] = float((vw < 3).mean())
    f["frac_v1"] = float((vw < 1).mean())
    # launches per second in the window (upward crossings of 1.5 m/s after a dip below 0.7 m/s), stopped fraction after the launch window
    low, nl = False, 0
    for x in vw[1:]:
        if low and x >= 1.5:
            low, nl = False, nl + 1
        elif not low and x < 0.7:
            low = True
    f["launch_rate"] = nl / (0.25 * W)
    f["frac_stop"] = float((vw[LAUNCH:] < 0.3).mean()) if W > LAUNCH + 2 else np.nan
    f["v_l"] = float(v[min(LAUNCH, n) - 1])
    f["acc_l"] = float((v[min(LAUNCH, n) - 1] - v[0]) / (0.25 * max(min(LAUNCH, n) - 1, 1)))
    ph = np.array([phi1(s)[0] for s in S])
    f["phi1_l_abs"] = float(np.nanmean(np.abs(ph[1:LAUNCH])))
    f["phi1_l"] = float(np.nanmean(ph[1:LAUNCH]))
    f["lp_l"] = float(np.nanmean([s["lp"] if s["lp"] is not None else np.nan for s in S[:LAUNCH]]))
    f["eng_l"] = float(np.nanmean([s["eng"] if s["eng"] is not None else np.nan for s in S[:LAUNCH]]))
    la0 = t["lead_actor"][0]
    f["actor_min"] = float(la0) if la0 is not None else 99.0      # initial state only (no window leakage)
    f["actor_ahead"] = int(f["actor_min"] < OBST)
    for k in ("blocked_pts", "blocked_pts_15"):
        f[k] = t.get(k, np.nan)
    f["blocked_near"] = t["blocked_near"] if t.get("blocked_near") is not None else 99.0
    f["road_w"] = (t.get("gl_p90", np.nan) or np.nan) + (t.get("gr_p90", np.nan) or np.nan)
    f["road_w_med"] = (t.get("gl_med", np.nan)) + (t.get("gr_med", np.nan))
    f["road_asym"] = (t.get("gl_p90", np.nan) - t.get("gr_p90", np.nan))
    for D in (10, 20, 40):
        f[f"route_abs{D}"] = abs(t.get(f"route_dyaw{D}", np.nan))
    return f


def auc_ci(x, y, B=2000, seed=0):
    """AUC of x for label y (1 = spin), Mann-Whitney, stratified bootstrap 95% CI."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    p, q = x[y == 1], x[y == 0]
    if len(p) < 3 or len(q) < 3:
        return np.nan, np.nan, np.nan, len(p), len(q)

    def auc(p, q):
        d = p[:, None] - q[None, :]
        return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

    rng = np.random.default_rng(seed)
    bs = [auc(rng.choice(p, len(p)), rng.choice(q, len(q))) for _ in range(B)]
    return auc(p, q), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(p), len(q)


def build(T, agent, sc, ep, lean):
    rows = []
    for t in T:
        if t["tag"] != f"{agent}-fixed" or t.get("run_err"):
            continue
        e = ep[(ep.scenario == t["scenario"]) & (ep.agent == agent) & (ep.controller == "fixed")].iloc[0]
        spin = bool(e.spin)
        f = feats(t, float(e.start) if spin else np.inf)
        f.update(scenario=t["scenario"], agent=agent, dataset=t["dataset"], difficulty=t["difficulty"], spin=int(spin),
                 start=float(e.start) if spin else np.nan, hd=float(t["hdscore"]), steps=len(t["steps"]), end=t["end"])
        s = sc[sc.scenario == t["scenario"]].iloc[0]
        f["n_ahead"], f["ahead_min_b"] = int(s.n_ahead), (float(s.ahead_min_b) if pd.notna(s.ahead_min_b) else 99.0)
        f["n_actors"] = int(s.n_actors)
        f["hd_map"] = int(bool(s.hd_map))
        f["heading_change_abs"] = abs(float(s.heading_change_deg))
        f["max_abs_heading"] = float(s.max_abs_heading_deg)
        f["turn_nonstraight"] = int(s.turn != "straight")
        f["start_yaw_abs"] = abs(eval(s.start_euler)[1])
        f["start_ab_b"] = eval(s.start_ab)[1]
        l = lean[lean.scenario == t["scenario"]]
        f["lean_abs"] = float(abs(l.base.iloc[0])) if len(l) else np.nan
        f["lean"] = float(l.base.iloc[0]) if len(l) else np.nan
        rows.append(f)
    return pd.DataFrame(rows)


FEATS = ["lean_abs", "phi1_l_abs", "frac_v3", "frac_v1", "launch_rate", "frac_stop", "v_l", "acc_l",
         "actor_min", "actor_ahead", "n_ahead", "ahead_min_b", "lp_l", "eng_l", "blocked_pts", "blocked_pts_15", "blocked_near",
         "road_w", "road_w_med", "road_asym", "route_abs10", "route_abs20", "route_abs40", "heading_change_abs", "max_abs_heading",
         "turn_nonstraight", "start_yaw_abs", "n_actors", "hd_map", "nuscenes", "waymo", "kitti360", "pandaset", "diff_easy", "diff_medium", "diff_hard", "diff_extreme"]


def add_dummies(df):
    for d in ("nuscenes", "waymo", "kitti360", "pandaset"):
        df[d] = (df.dataset == d).astype(int)
    for d in ("easy", "medium", "hard", "extreme"):
        df[f"diff_{d}"] = (df.difficulty == d).astype(int)
    return df


def loo(df, feats_, topk=3, B=500):
    """Leave-one-out: in each fold pick the topk features by |AUC - 0.5| on the training part, fit a standardised L2 logistic model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = df[feats_].astype(float)
    X = X.fillna(X.median())
    y = df.spin.values
    pred = np.zeros(len(df))
    chosen = []
    for i in range(len(df)):
        tr = np.arange(len(df)) != i
        sc_ = []
        for c in feats_:
            xc = X[c].values[tr]
            if np.std(xc) == 0:
                sc_.append(0)
                continue
            sc_.append(abs(roc_auc_score(y[tr], xc) - 0.5))
        top = [feats_[j] for j in np.argsort(sc_)[::-1][:topk]]
        chosen.append(top)
        mu, sd = X[top].values[tr].mean(0), X[top].values[tr].std(0) + 1e-9
        m = LogisticRegression(C=0.5, class_weight="balanced").fit((X[top].values[tr] - mu) / sd, y[tr])
        pred[i] = m.decision_function(((X[top].values[i] - mu) / sd)[None])[0]
    a, lo, hi, *_ = auc_ci(pred, y, B=B)
    from collections import Counter
    return dict(loo_auc=a, ci=[lo, hi], chosen=Counter(c for top in chosen for c in top).most_common(8)), pred


def main():
    T = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(R / "spin/spin_episodes.csv")
    sc = pd.read_csv(R / "hugsim-exam/scenarios.csv")
    lean = pd.read_csv(R / "lean/lean_per_scenario.csv")
    res = {}
    dfs = {}
    for agent in ("cinque", "lebowski"):
        df = add_dummies(build(T, agent, sc, ep, lean))
        dfs[agent] = df
        df.to_csv(out / f"features_{agent}.csv", index=False)
        rows = []
        for c in FEATS:
            if c not in df or df[c].nunique(dropna=True) < 2:
                continue
            a, lo, hi, npos, nneg = auc_ci(df[c], df.spin)
            rows.append(dict(agent=agent, feature=c, auc=a, lo=lo, hi=hi, sep=int(lo > 0.5 or hi < 0.5), med_spin=df[df.spin == 1][c].median(),
                             med_non=df[df.spin == 0][c].median(), n_spin=npos, n_non=nneg))
        A = pd.DataFrame(rows).sort_values("auc", key=lambda s: -(s - 0.5).abs())
        A.to_csv(out / f"q1_auc_{agent}.csv", index=False)
        print(agent, "n", len(df), "spin", int(df.spin.sum()), "features with CI excluding 0.5:", int(A.sep.sum()), "of", len(A))
        print(A.head(14).round(3).to_string())
        fs = [c for c in FEATS if c in df and df[c].nunique() > 1 and c not in ("n_actors",)]
        r, _ = loo(df, fs)
        # best single feature LOO AUC reference = its own AUC (no fitting): report max |AUC-0.5| feature
        res[agent] = dict(n=len(df), n_spin=int(df.spin.sum()), n_sep=int(A.sep.sum()), n_feat=len(A), loo=r)
        print(agent, "LOO", r)
    json.dump(res, open(out / "q1_loo.json", "w"), indent=1, default=str)
    # blocked hypothesis
    for agent, df in dfs.items():
        for name, thr in (("A_actor25", None), ("B_pts30", 30), ("B_pts100", 100), ("B_pts10", 10)):
            if thr is None:
                blk = df.actor_ahead == 1
            else:
                blk = (df.actor_ahead == 1) | (df.blocked_pts >= thr)
            slow = df.frac_v3 >= 0.5   # window fraction, leaky for short windows (see report)
            for lab, m in (("blocked", blk), ("blocked&slow", blk & slow)):
                tab = [[int((m & (df.spin == 1)).sum()), int((~m & (df.spin == 1)).sum())], [int((m & (df.spin == 0)).sum()), int((~m & (df.spin == 0)).sum())]]
                orr, p = stats.fisher_exact(tab, alternative="greater")
                print(agent, name, lab, tab, "OR", round(orr, 2), "p", round(p, 3))
    return dfs


if __name__ == "__main__":
    main()
