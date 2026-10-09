"""C1 / C2 arbitration feasibility, offline (decision 202): can a decision-time signal pick the better of SH30 and AP2 per scene.

Both drivers receive identical inputs at decisions 0 and 1 (the first 0.5 s is log replay), so a selector that runs both models there and
commits to one driver for the rest of the scene reproduces that driver's logged rollout exactly: its score is known without a new run.
Signals use only what a driver is sent (its own two plans, the other driver's two plans, ego speed, command, the route waypoint); the
scores enter as labels. Rules are fit on training folds and scored on held-out folds, folds split by log (scene-clustered).

  c1_arb.py [--out experiments/alpasim/results/c1/arb.json]      (Mac, repo .venv, pickles of c1_extract.py in tmp/c1/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import c1_lib as L  # noqa: E402
from jevdrive import stats  # noqa: E402

ORACLE = None


def plan_feats(r) -> dict:
    P = np.r_[[[0.0, 0.0, 0.0]], np.array(r["poses"])]
    return dict(arc=float(np.hypot(*np.diff(P[:, :2], axis=0).T).sum()), y2=P[4, 1], y4=P[8, 1], yaw4=float(np.degrees(P[8, 2])), P=P[1:, :2])


def table() -> pd.DataFrame:
    R = L.runs()
    rows = []
    for s in sorted(R["SH30"]):
        a, b = R["SH30"][s], R["AP2"][s]
        assert np.allclose(a["rec"][1]["anchor"], b["rec"][1]["anchor"]) and np.allclose(a["rec"][0]["anchor"], b["rec"][0]["anchor"])
        r = dict(scene=s, log=s.rsplit("-", 1)[0], s_sh=a["summary"]["score"], s_ap=b["summary"]["score"])
        e = a["rec"][1]["ego"]
        r.update(v=10 * float(np.hypot(e[4], e[5])), ax=3 * e[6], cmd=a["rec"][1]["cmd"], cmd_ap=b["rec"][1]["cmd"], ry=abs(a["rec"][1]["route0"][1]))
        for k in (0, 1):
            fa, fb = plan_feats(a["rec"][k]), plan_feats(b["rec"][k])
            r.update({f"sh_arc{k}": fa["arc"], f"ap_arc{k}": fb["arc"], f"sh_y4{k}": fa["y4"], f"ap_y4{k}": fb["y4"],
                      f"dlon{k}": fb["arc"] - fa["arc"], f"dlat{k}": abs(fb["y4"] - fa["y4"]), f"dmax{k}": float(np.hypot(*(fb["P"] - fa["P"]).T).max()),
                      f"dyaw{k}": abs(fb["yaw4"] - fa["yaw4"])})
        r["rel"] = r["dlon1"] / max(0.5 * (r["sh_arc1"] + r["ap_arc1"]), 2.0)
        r["sh_jit"], r["ap_jit"] = abs(r["sh_arc1"] - r["sh_arc0"]), abs(r["ap_arc1"] - r["ap_arc0"])
        r["ap_lat"], r["sh_lat"] = abs(r["ap_y41"]), abs(r["sh_y41"])
        rows.append(r)
    return pd.DataFrame(rows)


# rule families: name -> (feature column, direction); the rule is "commit to SH30 when dir * feature > dir * threshold, else AP2"
FAM = {"standstill: ego speed below t": ("v", -1), "AP2 plans longer than SH30 by more than t m": ("dlon1", 1),
       "AP2 plans shorter than SH30 by more than t m": ("dlon1", -1), "relative length gap above t": ("rel", 1),
       "lateral gap at 4 s above t m": ("dlat1", 1), "largest point gap above t m": ("dmax1", 1), "heading gap at 4 s above t deg": ("dyaw1", 1),
       "AP2 lateral offset at 4 s above t m": ("ap_lat", 1), "AP2 plan change between decisions above t m": ("ap_jit", 1),
       "route waypoint offset above t m": ("ry", 1)}


def fit_threshold(x, d, sgn):
    """Threshold maximising the summed gain d on the scenes where sgn * x > sgn * t; 'never switch' (gain 0) is allowed."""
    o = np.argsort(-sgn * x)
    c = np.cumsum(d[o])
    j = int(np.argmax(c))
    if c[j] <= 0:
        return sgn * np.inf, 0.0
    xs = x[o]
    t = xs[j] if j + 1 >= len(xs) else 0.5 * (xs[j] + xs[j + 1])
    return float(t), float(c[j])


def folds(logs, k, seed):
    u = np.unique(logs)
    rng = np.random.default_rng(seed)
    f = dict(zip(rng.permutation(u), np.arange(len(u)) % k))
    return np.array([f[x] for x in logs])


def cv_rule(D, col, sgn, k=5, reps=20):
    """Out-of-fold per-scene gain of a one-threshold rule over the best single driver (AP2), averaged over repeated log-grouped splits."""
    x, d, logs = D[col].to_numpy(float), (D.s_sh - D.s_ap).to_numpy(float), D.log.to_numpy()
    g = np.zeros(len(D))
    for rep in range(reps):
        f = folds(logs, k, rep)
        for i in range(k):
            tr, te = f != i, f == i
            t, _ = fit_threshold(x[tr], d[tr], sgn)
            g[te] += np.where(sgn * x[te] > sgn * t, d[te], 0.0) / reps
    return g


def cv_model(D, cols, k=5, reps=5):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, d, logs = D[cols].to_numpy(float), (D.s_sh - D.s_ap).to_numpy(float), D.log.to_numpy()
    out = {}
    for name, mk in (("ridge on all signals", lambda: make_pipeline(StandardScaler(), Ridge(10.0))),
                     ("gradient boosting on all signals", lambda: GradientBoostingRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0))):
        g = np.zeros(len(D))
        for rep in range(reps):
            f = folds(logs, k, rep)
            for i in range(k):
                tr, te = f != i, f == i
                m = mk().fit(X[tr], d[tr])
                t, _ = fit_threshold(m.predict(X[tr]), d[tr], 1)          # the switch threshold on the predicted difference, fit on train
                g[te] += np.where(m.predict(X[te]) > t, d[te], 0.0) / reps
        out[name] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(L.ROOT / "experiments/alpasim/results/c1/arb.json"))
    a = ap.parse_args()
    D = table()
    d = (D.s_sh - D.s_ap).to_numpy()
    oracle = float(np.maximum(d, 0).mean())
    res = dict(n=len(D), logs=int(D.log.nunique()), sh=float(D.s_sh.mean()), ap=float(D.s_ap.mean()), oracle_gain=oracle,
               sh_better=int((d > 0).sum()), ap_better=int((d < 0).sum()), tie=int((d == 0).sum()),
               sum_pos=float(d[d > 0].sum()), sum_neg=float(d[d < 0].sum()), rules=[])
    for name, (col, sgn) in FAM.items():
        t, tot = fit_threshold(D[col].to_numpy(float), d, sgn)
        sw = sgn * D[col].to_numpy(float) > sgn * t
        g = cv_rule(D, col, sgn)
        b = stats.bootstrap(g, groups=D.log.to_numpy())
        res["rules"].append(dict(rule=name, signal=col, t_in=t, switched=int(sw.sum()), in_gain=tot / len(D), cv_gain=b["mean"], lo=b["lo"], hi=b["hi"],
                                 recovered=b["mean"] / oracle, hits=int((sw & (d > 0)).sum()), misses=int((sw & (d < 0)).sum())))
    cols = ["v", "ax", "ry", "dlon0", "dlon1", "rel", "dlat0", "dlat1", "dmax0", "dmax1", "dyaw1", "sh_arc1", "ap_arc1", "ap_lat", "sh_lat", "ap_jit", "sh_jit"]
    for name, g in cv_model(D, cols).items():
        b = stats.bootstrap(g, groups=D.log.to_numpy())
        res["rules"].append(dict(rule=name, signal="model", t_in=None, switched=None, in_gain=None, cv_gain=b["mean"], lo=b["lo"], hi=b["hi"],
                                 recovered=b["mean"] / oracle, hits=None, misses=None))
    # where the hindsight gain sits: by start speed and by AP2's zero flag
    R = L.runs()
    D["ap_zero"] = [",".join(L.SHORT[f] for f in L.zero_flags(R["AP2"][s])) for s in D.scene]
    D["sh_zero"] = [",".join(L.SHORT[f] for f in L.zero_flags(R["SH30"][s])) for s in D.scene]
    D["vband"] = pd.cut(D.v, [-1, 1, 3, 6, 10, 99], labels=["< 1", "1-3", "3-6", "6-10", ">= 10"])
    res["by_speed"] = [dict(band=str(b), n=len(g), sh=float(g.s_sh.mean()), ap=float(g.s_ap.mean()), oracle=float(np.maximum(g.s_sh, g.s_ap).mean()),
                            sh_zero=int((g.s_sh == 0).sum()), ap_zero=int((g.s_ap == 0).sum()), sh_slow=int(((g.s_sh > 0) & (g.s_sh < 1)).sum()),
                            ap_slow=int(((g.s_ap > 0) & (g.s_ap < 1)).sum())) for b, g in D.groupby("vband", observed=True)]
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    D.drop(columns=["vband"]).to_csv(Path(a.out).with_name("arb_scenes.csv"), index=False, float_format="%.4f")
    print(f"n {res['n']} logs {res['logs']} SH30 {res['sh']:.4f} AP2 {res['ap']:.4f} oracle +{oracle:.4f}; SH30 better {res['sh_better']} (sum {res['sum_pos']:.2f}), "
          f"AP2 better {res['ap_better']} (sum {res['sum_neg']:.2f})")
    for r in res["rules"]:
        print(f"{r['rule']:48s} t {r['t_in']!s:>8.8} sw {r['switched']!s:>4} in {0 if r['in_gain'] is None else r['in_gain']:+.4f} "
              f"cv {r['cv_gain']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] rec {100 * r['recovered']:.0f}% hits {r['hits']} misses {r['misses']}")
    for r in res["by_speed"]:
        print(r)


if __name__ == "__main__":
    main()
