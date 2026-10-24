#!/usr/bin/env python3
"""FITNESS-NOISE (results/fitness_noise/README.md): how many practice scenes a PBT fitness needs before it ranks learners by signal, not
noise. CPU only, no reads. Proxy for the noise structure: the stored PAI 281 ACC + GAP Harmonizer-OFF per-scene reads (pai_fix facts).

  python fitness_noise.py [--facts DIR] [--out results/fitness_noise/fitness_noise.json] [--B 4000]

DIR defaults to $DATA_DIR/runs/pai_fix/facts (<tag>-281ag.csv). Sections of the JSON:
  reads     per-read mean on the common scene set
  anova     two-way (seed x scene) decomposition within recipes: population seed sd vs scene x seed interaction
  ckpt      checkpoint-level ranking (truth = full-set mean diff): P(correct) by K, |D| bin, design (uniform / stratified / Neyman),
            fitness lam (new-break penalty on NTV0-solved scenes)
  null      same-recipe seed pairs: P(|slice diff| >= 0.03) by K
  recipe    recipe-level ranking (truth = all-seed recipe mean diff, one random seed per learner)
  inform    informative slices: scenes where design reads disagree; temporal split (design = older reads, test = newer pairs)
  ema       persistence of per-scene diffs across snapshot pairs (rho) and fixed vs rotating slice averaging over rounds
  model     normal-model table: P(correct) and K for 80 % at true gaps 0.02 / 0.03 / 0.05, pairwise and P-way truncation
"""
from __future__ import annotations

import argparse
import glob
import itertools as it
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

KS = [40, 60, 80, 100, 140, 200]
LAMS = [0.0, 0.5, 1.0]
SOLVED = 0.98
BINS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.035), (0.035, 0.06), (0.06, 1.0)]
OLD = {"NTV0", "R2C", "R2J", "R3MV", "R5A", "R5LC", "R6FIX", "R6REC"}  # design reads for the temporal split (before R6DAG)
PHI = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
Z80 = 0.8416


def load(d):
    S, meta = {}, None
    for f in sorted(glob.glob(f"{d}/*-281ag.csv")):
        x = pd.read_csv(f).set_index("scene")
        fam, _, seed = Path(f).name.replace("-281ag.csv", "").split("-")
        S[f"{fam}-{seed}"] = x.score
        if fam == "NTV0":
            meta = x[["fclass", "stratum"]]
    common = sorted(set.intersection(*[set(s.index) for s in S.values()]))
    tags = list(S)
    M = np.stack([S[t].loc[common].to_numpy(float) for t in tags])
    return tags, M, meta.loc[common]


def fam(t):
    return t.split("-")[0]


def fitness(M, ntv, lam):
    return M - lam * ((ntv >= SOLVED)[None] & (M < SOLVED))


# ------------------------------------------------------------------ slice designs: weight matrices W [B, N], estimate = d @ W.T
def w_uniform(N, K, B, rng, pool=None):
    pool = np.arange(N) if pool is None else pool
    W = np.zeros((B, N))
    idx = pool[rng.integers(0, len(pool), (B, K))]
    np.add.at(W, (np.repeat(np.arange(B), K), idx.ravel()), 1.0 / K)
    return W


def w_strat(strata, K, B, rng, alloc):
    """alloc: dict stratum -> n_h (sum K); weights W_h / n_h so the estimate is the population-weighted stratum mean."""
    N = len(strata)
    W = np.zeros((B, N))
    for h, n_h in alloc.items():
        pool = np.flatnonzero(strata == h)
        if n_h == 0:
            continue
        idx = pool[rng.integers(0, len(pool), (B, n_h))]
        np.add.at(W, (np.repeat(np.arange(B), n_h), idx.ravel()), len(pool) / N / n_h)
    return W


def allocate(strata, K, sd=None):
    hs, cnt = np.unique(strata, return_counts=True)
    w = cnt * (np.ones(len(hs)) if sd is None else np.maximum(np.array([sd[h] for h in hs]), 1e-3))
    n = np.maximum(1, np.floor(K * w / w.sum()).astype(int))
    while n.sum() < K:
        n[np.argmax(K * w / w.sum() - n)] += 1
    while n.sum() > K:
        n[np.argmax(n)] -= 1
    return dict(zip(hs, n))


def p_correct(Dd, W, truth):
    """Dd [P, N] per-scene fitness diffs, W [B, N]; P(sign(estimate) == sign(truth)), ties count 1/2."""
    est = Dd @ W.T
    s = np.sign(truth)[:, None]
    return ((np.sign(est) == s) + 0.5 * (est == 0)).mean(1)


def binned(D, P):
    out = {}
    for lo, hi in BINS:
        m = (np.abs(D) >= lo) & (np.abs(D) < hi)
        if m.any():
            out[f"{lo:.3f}-{hi:.3f}"] = {"n_pairs": int(m.sum()), "P": round(float(P[m].mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default=f"{os.environ.get('DATA_DIR', '/root/autodl-tmp/ujs')}/runs/pai_fix/facts")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results/fitness_noise/fitness_noise.json"))
    ap.add_argument("--B", type=int, default=4000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tags, M, meta = load(a.facts)
    R, N = M.shape
    ntv = M[tags.index("NTV0-s0")]
    fams = sorted({fam(t) for t in tags})
    out = {"n_scenes": N, "n_reads": R, "reads": {t: round(float(M[i].mean()), 4) for i, t in enumerate(tags)}}

    # ---------------------------------------------------------- ANOVA within recipes (seed x scene, no replication)
    ss_a = ss_e = df_a = df_e = 0.0
    per = {}
    for f in fams:
        Y = M[[i for i, t in enumerate(tags) if fam(t) == f]]
        m = len(Y)
        if m < 2:
            continue
        r = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()
        sa, se = N * ((Y.mean(1) - Y.mean()) ** 2).sum(), (r ** 2).sum()
        ss_a, ss_e, df_a, df_e = ss_a + sa, ss_e + se, df_a + m - 1, df_e + (m - 1) * (N - 1)
        per[f] = {"seeds": m, "sd_seed_means": round(float(Y.mean(1).std(ddof=1)), 4),
                  "sd_interaction_per_scene": round(float(math.sqrt(se / ((m - 1) * (N - 1)))), 4)}
    ms_a, ms_e = ss_a / df_a, ss_e / df_e
    var_a = max(0.0, (ms_a - ms_e) / N)
    out["anova"] = {"per_recipe": per, "pooled_sd_seed_means_obs": round(math.sqrt(ms_a / N), 4),
                    "sd_interaction_per_scene": round(math.sqrt(ms_e), 4),
                    "sd_seed_population": round(math.sqrt(var_a), 4),
                    "share_of_seed_mean_var_from_scene_sampling": round(ms_e / ms_a, 3),
                    "df_seed": df_a}
    sig_a = math.sqrt(var_a)

    # ---------------------------------------------------------- checkpoint level
    pairs = list(it.combinations(range(R), 2))
    same = np.array([fam(tags[i]) == fam(tags[j]) for i, j in pairs])
    D0 = np.array([M[i].mean() - M[j].mean() for i, j in pairs])
    sd0 = np.array([(M[i] - M[j]).std(ddof=1) for i, j in pairs])
    out["pair_sd"] = {"median_sd_d_all": round(float(np.median(sd0)), 3),
                      "median_sd_d_same_recipe": round(float(np.median(sd0[same])), 3),
                      "median_sd_d_absD_lt_0.06": round(float(np.median(sd0[np.abs(D0) < 0.06])), 3),
                      "median_n_scenes_differ": int(np.median([((M[i] - M[j]) != 0).sum() for i, j in pairs]))}
    solved = np.where(ntv >= SOLVED, "ntv_solved", "ntv_failed")
    fcl = meta.fclass.to_numpy()
    fcl4 = np.select([fcl == "ok", np.isin(fcl, ["rear-hit", "slow"]), fcl == "overshoot"], ["ok", "partial", "overshoot"], "zero_other")
    strata = {"ntv_solved": solved, "ntv_fclass4": fcl4, "stratum": np.where(meta.stratum.to_numpy() == "static", "straight",
                                                                             meta.stratum.to_numpy())}
    ck = {}
    for lam in LAMS:
        F = fitness(M, ntv, lam)
        Dd = np.stack([F[i] - F[j] for i, j in pairs])
        ck_l = {}
        for K in KS:
            designs = {"uniform": w_uniform(N, K, a.B, rng)}
            for sn, st in strata.items():
                designs[f"strat_{sn}"] = w_strat(st, K, a.B, rng, allocate(st, K))
                sdh = {h: float(np.sqrt((Dd[np.abs(D0) < 0.06][:, st == h] ** 2).mean())) for h in np.unique(st)}
                designs[f"neyman_{sn}"] = w_strat(st, K, a.B, rng, allocate(st, K, sdh))
            row = {}
            for dn, W in designs.items():
                P = p_correct(Dd, W, D0)  # truth = full-set mean score (lam = 0), what the board rewards
                row[dn] = {"vs_score": binned(D0, P)}
                if lam and dn == "uniform":
                    row[dn]["vs_own_fitness"] = binned(D0, p_correct(Dd, W, Dd.mean(1)))
                if dn == "uniform":
                    est = Dd @ W.T
                    row[dn]["sd_est_median"] = round(float(np.median(est.std(1))), 4)
            ck_l[K] = row
        ck[f"lam{lam}"] = ck_l
    out["ckpt"] = ck
    # without replacement from the 281 (finite population) for the uniform lam 0 design, for reference
    Dd0 = np.stack([M[i] - M[j] for i, j in pairs])
    wor = {}
    for K in KS:
        W = np.zeros((a.B, N))
        for b in range(a.B):
            W[b, rng.choice(N, K, replace=False)] = 1.0 / K
        wor[K] = binned(D0, p_correct(Dd0, W, D0))
    out["ckpt_without_replacement"] = wor

    # ---------------------------------------------------------- null: same-recipe seed pairs
    nl = {}
    sp = [k for k, s in enumerate(same) if s]
    for K in KS:
        est = Dd0[sp] @ w_uniform(N, K, a.B, rng).T
        nl[K] = {"P_abs_ge_0.03": round(float((np.abs(est) >= 0.03).mean()), 3),
                 "P_abs_ge_0.02": round(float((np.abs(est) >= 0.02).mean()), 3),
                 "sd_est": round(float(est.std(1).mean()), 4)}
    nl["full_set"] = {f"{tags[pairs[k][0]]} vs {tags[pairs[k][1]]}": round(float(D0[k]), 4) for k in sp}
    out["null"] = nl

    # ---------------------------------------------------------- recipe level
    fidx = {f: [i for i, t in enumerate(tags) if fam(t) == f] for f in fams}
    fmean = {f: M[fidx[f]].mean() for f in fams}
    rp = []
    for f, g in it.combinations(fams, 2):
        T = fmean[f] - fmean[g]
        sub = [(i, j) for i in fidx[f] for j in fidx[g]]
        rp.append((f, g, T, np.stack([M[i] - M[j] for i, j in sub])))
    rec = {}
    for K in KS + [N]:
        W = w_uniform(N, K, a.B, rng) if K != N else np.full((1, N), 1.0 / N)
        res = []
        for f, g, T, Ds in rp:
            res.append((T, float(p_correct(Ds, W, np.full(len(Ds), T)).mean()), len(Ds)))
        T = np.array([r[0] for r in res])
        P = np.array([r[1] for r in res])
        multi = np.array([len(fidx[f]) > 1 and len(fidx[g]) > 1 for f, g, *_ in rp])
        rec["full_set" if K == N else K] = {"all": binned(T, P), "both_multi_seed": binned(T[multi], P[multi])}
    rec["recipe_means"] = {f: round(float(v), 4) for f, v in fmean.items()}
    out["recipe"] = rec

    # ---------------------------------------------------------- informative slices, temporal split
    old = [i for i, t in enumerate(tags) if fam(t) in OLD]
    new = [i for i, t in enumerate(tags) if fam(t) not in OLD]
    test_pairs = [(i, j) for i, j in it.combinations(new, 2)] + [(i, tags.index("NTV0-s0")) for i in new]
    Dt = np.stack([M[i] - M[j] for i, j in test_pairs])
    Tt = Dt.mean(1)
    var_old = M[old].var(0)
    rng_old = M[old].max(0) - M[old].min(0)
    pool = np.flatnonzero(rng_old >= 0.02)
    var_all = M.var(0)
    inf = {"n_disagree_scenes_old": int(len(pool)), "n_disagree_scenes_all": int(((M.max(0) - M.min(0)) >= 0.02).sum()),
           "test_pairs": len(test_pairs),
           "share_abs_diff_mass_outside_old_pool": round(float(np.abs(Dt[:, np.setdiff1d(np.arange(N), pool)]).sum() / np.abs(Dt).sum()), 3),
           "new_breaks_outside_old_pool": int(sum(((ntv >= SOLVED) & (M[i] < SOLVED))[np.setdiff1d(np.arange(N), pool)].sum() for i in new)),
           "new_breaks_total": int(sum(((ntv >= SOLVED) & (M[i] < SOLVED)).sum() for i in new))}
    rows = {}
    for K in KS:
        r = {}
        r["uniform"] = binned(Tt, p_correct(Dt, w_uniform(N, K, a.B, rng), Tt))
        r["random_from_old_pool"] = binned(Tt, p_correct(Dt, w_uniform(N, K, a.B, rng, pool), Tt))
        for nm, v in (("topvar_old", var_old), ("topvar_all_insample", var_all)):
            W = np.zeros((1, N))
            W[0, np.argsort(-v, kind="stable")[:K]] = 1.0 / K
            P = p_correct(Dt, W, Tt)
            r[nm] = {**binned(Tt, P), "all_pairs_absD_ge_0.02": round(float(P[np.abs(Tt) >= 0.02].mean()), 3)}
        r["uniform"]["all_pairs_absD_ge_0.02"] = round(float(p_correct(Dt, w_uniform(N, K, a.B, rng), Tt)[np.abs(Tt) >= 0.02].mean()), 3)
        r["random_from_old_pool"]["all_pairs_absD_ge_0.02"] = round(float(
            p_correct(Dt, w_uniform(N, K, a.B, rng, pool), Tt)[np.abs(Tt) >= 0.02].mean()), 3)
        # mixed: 70 % from the old pool, 30 % uniform (catches new breaks on scenes nobody failed before)
        k1 = int(round(0.7 * K))
        W = w_uniform(N, k1, a.B, rng, pool) * k1 / K + w_uniform(N, K - k1, a.B, rng) * (K - k1) / K
        P = p_correct(Dt, W, Tt)
        r["mix70_pool_30_uniform"] = {**binned(Tt, P), "all_pairs_absD_ge_0.02": round(float(P[np.abs(Tt) >= 0.02].mean()), 3)}
        rows[K] = r
    inf["by_K"] = rows
    # honest version: random scene halves; slice from half A (top variance of the old reads, or uniform), truth = new pairs' mean on half B
    nsplit, sp_res = 300, {K: {"uniform": [], "topvar_old": [], "mix70": []} for K in KS if K <= N // 2}
    TB_all = []
    for _ in range(nsplit):
        perm = rng.permutation(N)
        A, Bh = perm[: N // 2], perm[N // 2:]
        TB = Dt[:, Bh].mean(1)
        m = np.abs(TB) >= 0.02
        TB_all.append(int(m.sum()))
        for K in sp_res:
            WU = np.zeros((200, N))
            idx = A[rng.integers(0, len(A), (200, K))]
            np.add.at(WU, (np.repeat(np.arange(200), K), idx.ravel()), 1.0 / K)
            sp_res[K]["uniform"].append(p_correct(Dt[m], WU, TB[m]).mean())
            top = A[np.argsort(-var_old[A], kind="stable")[:K]]
            WT = np.zeros((1, N))
            WT[0, top] = 1.0 / K
            sp_res[K]["topvar_old"].append(p_correct(Dt[m], WT, TB[m]).mean())
            k1 = int(round(0.7 * K))
            top1 = A[np.argsort(-var_old[A], kind="stable")[:k1]]
            rest = np.setdiff1d(A, top1)
            WM = np.zeros((200, N))
            WM[:, top1] = 1.0 / K
            idx = rest[rng.integers(0, len(rest), (200, K - k1))]
            np.add.at(WM, (np.repeat(np.arange(200), K - k1), idx.ravel()), 1.0 / K)
            sp_res[K]["mix70"].append(p_correct(Dt[m], WM, TB[m]).mean())
    inf["split_half"] = {"n_splits": nsplit, "mean_test_pairs_absD_ge_0.02_on_B": round(float(np.mean(TB_all)), 1),
                         "P_correct_vs_halfB_truth": {K: {k: round(float(np.mean(v)), 3) for k, v in r.items()} for K, r in sp_res.items()}}
    out["inform"] = inf

    # ---------------------------------------------------------- persistence across snapshots, fixed vs rotating slice
    multi = [f for f in fams if len(fidx[f]) > 1]
    rhos = []
    for f, g in it.combinations(multi, 2):
        for (i1, i2), (j1, j2) in it.product(it.combinations(fidx[f], 2), it.combinations(fidx[g], 2)):
            d1, d2 = M[i1] - M[j1], M[i2] - M[j2]
            rhos.append(np.corrcoef(d1, d2)[0, 1])
            d1, d2 = M[i1] - M[j2], M[i2] - M[j1]
            rhos.append(np.corrcoef(d1, d2)[0, 1])
    rho = float(np.median(rhos))
    # empirical: lineage f vs g over m rounds = matched seeds; fixed slice reused vs fresh slice per round; truth = recipe mean diff
    emp = {}
    for K in [60, 100]:
        acc = {"single": [], "avg_fixed": [], "avg_rotating": []}
        Ts = []
        for f, g in it.combinations(multi, 2):
            m = min(len(fidx[f]), len(fidx[g]))
            T = fmean[f] - fmean[g]
            Ds = np.stack([M[fidx[f][k]] - M[fidx[g][k]] for k in range(m)])
            Wf = w_uniform(N, K, a.B, rng)
            ests_rot = np.stack([Ds[k] @ w_uniform(N, K, a.B, rng).T for k in range(m)])
            ests_fix = Ds @ Wf.T
            acc["single"].append(float((np.sign(ests_fix[0]) == np.sign(T)).mean()))
            acc["avg_fixed"].append(float((np.sign(ests_fix.mean(0)) == np.sign(T)).mean()))
            acc["avg_rotating"].append(float((np.sign(ests_rot.mean(0)) == np.sign(T)).mean()))
            Ts.append(T)
        Ts = np.array(Ts)
        emp[K] = {k: binned(Ts, np.array(v)) for k, v in acc.items()}
    out["ema"] = {"rho_median": round(rho, 3), "rho_iqr": [round(float(np.percentile(rhos, 25)), 3), round(float(np.percentile(rhos, 75)), 3)],
                  "n_rho": len(rhos), "empirical_matched_seed_rounds": emp}

    # ---------------------------------------------------------- normal model
    sds = {"close_pairs": float(np.median(sd0[np.abs(D0) < 0.06])), "same_recipe": float(np.median(sd0[same]))}
    mdl = {"sd_d": {k: round(v, 3) for k, v in sds.items()}, "sigma_seed_population": round(sig_a, 4), "rho": round(rho, 3)}
    tab = {}
    for (sdn, sd_close), gap in it.product(sds.items(), (0.02, 0.03, 0.05)):
        r = {"K80_checkpoint": int(math.ceil((Z80 * sd_close / gap) ** 2)),
             "K80_recipe_one_seed_each": (int(math.ceil((Z80 * sd_close) ** 2 / (gap ** 2 - 2 * var_a * Z80 ** 2)))
                                          if gap ** 2 > 2 * var_a * Z80 ** 2 else None),
             "P_recipe_K_inf": round(PHI(gap / math.sqrt(2 * var_a)) if var_a else 1.0, 3)}
        for K in KS:
            se = sd_close / math.sqrt(K)
            r[K] = {"P_checkpoint": round(PHI(gap / se), 3), "P_recipe": round(PHI(gap / math.sqrt(se ** 2 + 2 * var_a)), 3)}
            # averaging R rounds (EMA ~ R_eff): fixed slice keeps the persistent part rho of the slice error, rotating slice does not
            for Rr in (3,):
                fixed = se ** 2 * (rho + (1 - rho) / Rr)
                r[K][f"P_checkpoint_avg{Rr}_fixed"] = round(PHI(gap / math.sqrt(fixed)), 3)
                r[K][f"P_checkpoint_avg{Rr}_rotating"] = round(PHI(gap / math.sqrt(se ** 2 / Rr)), 3)
            # P-way truncation: 4 learners at 0, gap, 2 gap, 3 gap; P(top pick is the truly best), independent per-learner noise se / sqrt 2
            z = rng.normal(0, se / math.sqrt(2), (20000, 4)) + gap * np.arange(4)
            r[K]["P4_top_is_best"] = round(float((z.argmax(1) == 3).mean()), 3)
            r[K]["P4_bottom_is_worst"] = round(float((z.argmin(1) == 0).mean()), 3)
        tab[f"{sdn}_gap{gap}"] = r
    mdl["table"] = tab
    out["model"] = mdl

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":"), default=str))
    print(json.dumps({k: out[k] for k in ("n_scenes", "anova", "pair_sd", "model")}, indent=1, default=str))


if __name__ == "__main__":
    main()
