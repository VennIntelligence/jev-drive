"""op_parity gap analysis, tables: where P2 (and shipped P0) lose to WA-JEPA, sub-metric by sub-metric (navtest EPDMS v2, navhard two-stage).
No model runs: reads the devkit's per-token CSVs / harness_tokens.csv already on the box. CPU only.

Attribution (state it on the page): the v2 EPDMS of one token is
    score = NC * DAC * DDC * TLC * (5 EP + 5 TTC + 2 LK + 2 HC + 2 EC) / (14 + 2 [EC present])      (EC missing -> dropped; checked to 1e-15)
For the gap WA-JEPA - arm, every sub-score is a player and v(S) is the token's score when the terms in S take WA-JEPA's values and the
others the arm's own. The exact Shapley value of term i (average over all 512 coalitions) splits the per-token gap additively, so the
per-term losses sum to the mean-score gap exactly. Seeds: Shapley per seed, then averaged. CI: cluster bootstrap over logs (navtest) or
scene-mapping groups (navhard), B 10 000, paired.

  python experiments/op_parity/scripts/pp_gap_tables.py        (box; writes $OUT/gap_tables.json, gap_navtest_shap.npz)
"""
import json
import os
import sys as _sys
import pathlib as _pl

_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "scripts"), str(_pl.Path(__file__).parent)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive.common import data_dir  # noqa: E402

TERMS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
COLS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance", "ego_progress",
        "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
WAJEPA_CSV = "runs/top10_t2/navsim/wajepa/20260926-122804/v2/2026.09.26.16.32.53.csv"
OUT = _pl.Path(os.environ.get("GAP_OUT", data_dir() / "runs" / "op_parity" / "gap"))
B = 10_000


def score_of(X):
    """X (..., 9) sub-scores in TERMS order (EC may be NaN) -> EPDMS of the token."""
    ec = X[..., 8]
    num = 5 * X[..., 4] + 5 * X[..., 5] + 2 * X[..., 6] + 2 * X[..., 7] + 2 * np.nan_to_num(ec)
    return np.prod(X[..., :4], -1) * num / (14 + 2 * np.isfinite(ec))


def shapley(A, Bv):
    """Exact Shapley split of score(Bv) - score(A) over the 9 terms. A, Bv (n, 9) -> (n, 9)."""
    from math import factorial
    n, k = A.shape
    sub = np.arange(1 << k)
    bits = ((sub[:, None] >> np.arange(k)) & 1).astype(bool)                       # (512, 9)
    v = np.empty((n, 1 << k))
    for s in sub:
        v[:, s] = score_of(np.where(bits[s], Bv, A))
    pop = bits.sum(1)
    w = np.array([factorial(p) * factorial(k - p - 1) / factorial(k) if p < k else 0 for p in pop])
    phi = np.zeros((n, k))
    for i in range(k):
        s0 = sub[~bits[:, i]]
        phi[:, i] = (w[s0] * (v[:, s0 | (1 << i)] - v[:, s0])).sum(1)
    return phi


def boot_ratio(num, den, grp, seed=0):
    """Cluster bootstrap of sum(num) / sum(den) per cluster draw; num (n, m), den (n,) -> (m, 3) mean, lo, hi (mean = full-sample ratio)."""
    codes, uniq = pd.factorize(grp)
    nu = len(uniq)
    S = np.stack([np.bincount(codes, num[:, j], nu) for j in range(num.shape[1])], 1)
    D = np.bincount(codes, den, nu)
    idx = np.random.default_rng(seed).integers(nu, size=(B, nu))
    r = S[idx].sum(1) / D[idx].sum(1)[:, None]
    full = num.sum(0) / den.sum()
    return np.stack([full, *np.quantile(r, [0.025, 0.975], axis=0)], 1)


def arm_stats(Xs, Xw, w, grp):
    """Xs: list of per-seed arrays (n, 9) of one arm; Xw: WA-JEPA (n, 9); w: token weights (n,); grp: cluster labels.
    Returns dict term -> {arm mean, wa mean, diff [lo, hi], loss [lo, hi]} on a 0-100 scale, plus score row."""
    n = len(w)
    phi = np.mean([shapley(X, Xw) for X in Xs], 0)                                   # gap WA - arm, additive per token
    sc = np.mean([score_of(X) for X in Xs], 0)
    scw = score_of(Xw)
    ok = np.isfinite(Xw)
    out = {}
    Xm = np.mean(Xs, 0)                                                              # nan if any seed has nan: EC presence is shared, checked
    ind = np.isfinite(Xm) & ok
    # mean difference per term (arm - WA) over tokens where both exist, as sum / weight so the CI is on the same units
    diff = (np.nan_to_num(Xm) - np.nan_to_num(Xw)) * ind
    mw = np.stack([ind[:, j] * w for j in range(9)], 1)
    for j, t in enumerate(TERMS):
        d = boot_ratio(diff[:, [j]], mw[:, j], grp)[0]
        a = (np.nan_to_num(Xm[:, j]) * mw[:, j]).sum() / mw[:, j].sum()
        b = (np.nan_to_num(Xw[:, j]) * mw[:, j]).sum() / mw[:, j].sum()
        out[t] = dict(arm=100 * a, wa=100 * b, diff=100 * d[0], diff_lo=100 * d[1], diff_hi=100 * d[2])
    L = boot_ratio(phi * w[:, None], w, grp)
    for j, t in enumerate(TERMS):
        out[t].update(loss=100 * L[j, 0], loss_lo=100 * L[j, 1], loss_hi=100 * L[j, 2])
    G = boot_ratio(np.stack([(scw - sc) * w, sc * w, scw * w], 1), w, grp)
    out["_score"] = dict(arm=100 * G[1, 0], wa=100 * G[2, 0], gap=100 * G[0, 0], gap_lo=100 * G[0, 1], gap_hi=100 * G[0, 2],
                         shap_sum=100 * float((phi * w[:, None]).sum() / w.sum()))
    # transition counts on binary-ish terms (weights ignored): arm fails & WA passes, and the reverse; fail = < 1
    for j, t in enumerate(TERMS):
        am = np.nanmean([X[:, j] for X in Xs], 0)
        out[t]["arm_fail_wa_pass"] = int(((am < 1 - 1e-9) & (Xw[:, j] >= 1 - 1e-9)).sum())
        out[t]["wa_fail_arm_pass"] = int(((Xw[:, j] < 1 - 1e-9) & (am >= 1 - 1e-9)).sum())
        out[t]["arm_fail_rate"] = 100 * float((am < 1 - 1e-9).mean())
        out[t]["wa_fail_rate"] = 100 * float((Xw[:, j] < 1 - 1e-9).mean())
    return out, phi


def read_csv(path):
    df = pd.read_csv(path)
    t = df[~df.token.astype(str).str.startswith("average") & ~df.token.astype(str).str.startswith("extended_pdm_score")]
    return t.set_index("token")


def navtest():
    from jevdrive import navsim_zs as Z
    ev = data_dir() / "runs" / "navsim" / "eval"

    def csv(name):
        return read_csv(sorted((ev / f"v2_navtest_opi_lb_navtest_warp-cinque_PP{name}__base").glob("*/*.csv"))[-1])
    tabs = {"P0": [csv("P0")], "P2": [csv("P2-F-s0"), csv("P2-F-s1")]}
    wa = read_csv(data_dir() / WAJEPA_CSV)
    toks = sorted(set(wa.index).intersection(*[set(t.index) for v in tabs.values() for t in v]))
    log = {e["token"]: e["log_name"] for e in Z.load_index("navtest", slim=True)}
    grp = np.array([log[t] for t in toks])
    arr = lambda t: t.loc[toks, COLS].to_numpy(float)  # noqa: E731
    Xw = arr(wa)
    w = np.ones(len(toks))
    res = {"n": len(toks), "units": int(len(set(grp))), "arms": {}}
    shap = {}
    for arm, v in tabs.items():
        Xs = [arr(t) for t in v]
        assert all(np.array_equal(np.isfinite(X[:, 8]), np.isfinite(Xw[:, 8])) for X in Xs), "EC presence differs"
        res["arms"][arm], phi = arm_stats(Xs, Xw, w, grp)
        res["arms"][arm]["_recomputed_max_err"] = float(max(np.abs(score_of(X) - t.loc[toks, "score"].to_numpy()).max() for X, t in zip(Xs, v)))
        shap[arm] = phi
    np.savez(OUT / "gap_navtest_shap.npz", tokens=np.array(toks), log=grp, shap_P2=shap["P2"], shap_P0=shap["P0"], Xw=Xw,
             X2=np.stack([arr(t) for t in tabs["P2"]]), X0=arr(tabs["P0"][0]))
    return res


def navhard(root, p2, p0, label):
    from jevdrive.bench.compat import navhard_dir
    ar = data_dir() / "runs" / "op_parity"
    wa = pd.read_csv(ar / "navhard" / "harness" / "wajepa" / "harness_tokens.csv").set_index("token")
    def tokens(model):
        spec = model + ("@gimm" if root == "navhard_gimm" else "@warp")
        return pd.read_csv(navhard_dir(spec, ar / root / "harness" / model) / "harness_tokens.csv").set_index("token")
    tabs = {"P0": [tokens(p0)], "P2": [tokens(p) for p in p2]}
    res = {"label": label, "arms": {}}
    for st in (1, 2):
        toks = wa.index[wa.stage == st]
        grp = wa.loc[toks, "group"].to_numpy()
        w = np.ones(len(toks))                                                       # stage-2 weights are arm-specific (stage-1 outcome): unweighted here, official in _official
        Xw = wa.loc[toks, COLS].to_numpy(float)
        res[f"n_stage{st}"] = int(len(toks))
        for arm, v in tabs.items():
            Xs = [t.loc[toks, COLS].to_numpy(float) for t in v]
            s, _ = arm_stats(Xs, Xw, w, grp)
            s["_official"] = float(np.mean([100 * (t.loc[toks, "score"] * t.loc[toks, "weight"]).sum() / t.loc[toks, "weight"].sum() for t in v]))
            s["_official_wa"] = float(100 * (wa.loc[toks, "score"] * wa.loc[toks, "weight"]).sum() / wa.loc[toks, "weight"].sum())
            res["arms"].setdefault(arm, {})[f"stage{st}"] = s
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"navtest": navtest(),
           "navhard_G": navhard("navhard_gimm", ["P2-F-s0", "P2-F-s1"], "P0", "G"),
           "navhard_W": navhard("navhard", ["P2-F-s0", "P2-F-s1"], "P0", "W")}
    (OUT / "gap_tables.json").write_text(json.dumps(out, indent=1))
    for k in ("navtest",):
        for arm, s in out[k]["arms"].items():
            print(k, arm, {t: round(s[t]["loss"], 2) for t in TERMS}, {a: round(b, 2) for a, b in s["_score"].items()})
    for k in ("navhard_G", "navhard_W"):
        for arm, d in out[k]["arms"].items():
            for st, s in d.items():
                print(k, arm, st, {t: round(s[t]["loss"], 2) for t in TERMS}, {a: round(b, 2) for a, b in s["_score"].items()}, round(s["_official"], 2))
