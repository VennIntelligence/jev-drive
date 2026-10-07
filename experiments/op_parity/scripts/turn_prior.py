"""turn-prior (plans/2026-10-08-turn-prior-prereg.md, results/turn_prior.md): what the decision-173 floor head (ego + command + plan descriptor -> one of the F20
candidates) does on the turn-intent frames of WOD-E2E val, whether it adds on top of WLG and of the rater-preference plans, and whether it is a fixed rule.
Candidate sets, re-timing, RFS scoring and the ridge head are s2_gohold / s2_thinhead / s2_thinhead_heads; all CPU.

  prep   (jevdrive env)  F20 candidates + candidate RFS of every plan set (WP2, WLG, pref top / f20) -> $DATA_DIR/runs/op_parity/turn_prior/sets.npz
  desc   (jevdrive env)  Step 1: the floor head's picks on WP2 by v0 stratum x side, and the overlap with WLG   -> results/turn_prior/desc_*.csv
  stack  (jevdrive env)  Step 2: the floor head refitted on WLG's candidates and on top of the preference plans -> results/turn_prior/stack_*.csv
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "lib"), str(_R / "scripts"), str(_R / "experiments/op_adapt_r2/lib"), str(_pl.Path(__file__).parent)]
import argparse, json, time  # noqa: E401,E402

import numpy as np  # noqa: E402

import s2_thinhead as T  # noqa: E402
import s2_thinhead_heads as H  # noqa: E402
from jevdrive.common import data_dir  # noqa: E402

G = T.G
OUT, FIG = _R / "experiments/op_parity/results/turn_prior", _R / "experiments/op_parity/figs/turn_prior"
RUNS = data_dir() / "runs/op_parity/turn_prior"
KEEP, S, FOLDS, REPS, INNER = H.KEEP, 2, H.FOLDS, H.REPS, H.INNER
SETS = ("wp2", "wlg", "top", "f20")
V_STRATA = (("v0<0.5", 0.0, 0.5), ("0.5<=v0<3", 0.5, 3.0), ("v0>=3", 3.0, 1e9))


# ---------------------------------------------------------------- data
def ctx():
    import wod_launch_report as R
    C = R.Ctx()
    z = np.load(T.RUNS / "data.npz")
    assert (z["va_names"] == C.names[: C.n]).all()
    return C, z


def plans_of(C, name):
    if name == "wp2":
        return [C.preds(t) for t in T.ARMS]
    z = np.load(_R / "experiments/op_parity/results/wod_pref/plans.npz")
    assert (z["names"] == C.names[: C.n]).all()
    return [z[f"{name}_s{s}"].astype(np.float64) for s in range(S)]


def build(C, P, cb):
    """Plans (2 x (n, 20, 2)) -> candidates F (2, n, 20, 20, 2), candidate RFS J (2, 20, n), plan descriptor (2, n, 15)."""
    F = np.stack([G.factored(p, C.v0, 3, cb) for p in P])
    J = np.stack([np.stack([C.rfs(np.ascontiguousarray(f[:, m])) for m in range(20)]) for f in F])
    assert np.abs(J[:, KEEP] - np.stack([C.rfs(p) for p in P])).max() < 1e-9
    return F.astype(np.float32), J, np.stack([T.plan_desc(p, C.v0, cb) for p in P])


def plan_scaler(raw, std):
    """(mu, sd) of the s2-thinhead plan-stream standardisation (WOD train rows), recovered from WP2's raw and stored values: std = (raw - mu) / sd."""
    x, y = raw.reshape(-1, raw.shape[-1]), std.reshape(-1, std.shape[-1])
    sd = np.array([np.polyfit(y[:, j], x[:, j], 1)[0] for j in range(x.shape[1])])
    mu = np.array([np.polyfit(y[:, j], x[:, j], 1)[1] for j in range(x.shape[1])])
    assert np.abs((x - mu) / sd - y).max() < 1e-3
    return mu, sd


def cmd_prep(a):
    from jevdrive.run import Run
    with Run("op_parity", "turn-prior-prep", seed=0, config=vars(a)) as run:
        C, z = ctx()
        cb = T.codebook()
        out = {}
        for k in SETS:
            t0 = time.time()
            F, J, pd_ = build(C, plans_of(C, k), cb)
            out |= {f"F_{k}": F, f"J_{k}": J, f"pd_{k}": pd_}
            run.info("%s: RFS %.3f, F20 oracle d %+.3f (%.0f s)", k, C.cm(J[:, KEEP].mean(0)), C.cm(J.max(1).mean(0) - J[:, KEEP].mean(0)), time.time() - t0)
        assert np.abs(out["J_wp2"] - z["va_J"]).max() < 1e-6
        mu, sd = plan_scaler(out["pd_wp2"], z["va_plan"])
        for k in SETS:
            out[f"pd_{k}"] = ((out[f"pd_{k}"] - mu) / sd).astype(np.float32)
        out["plan_mu_sd"] = np.stack([mu, sd])
        out["J_shipped"] = C.rfs(C.preds("shipped"))
        RUNS.mkdir(parents=True, exist_ok=True)
        np.savez(RUNS / "sets.npz", **out)


def load_sets(C, z):
    d = dict(np.load(RUNS / "sets.npz"))
    H._D.update({"va_ego": z["va_ego"], "va_seq": z["va_seq"]})
    import pandas as pd
    H._D["scode"] = pd.factorize(pd.Series(z["va_seq"].astype(str)))[0]
    return d


# ---------------------------------------------------------------- the head, generalised: fit on one candidate set, apply to another
def X_of(pd_, ego):
    return np.concatenate([np.broadcast_to(ego, (S, *ego.shape)), pd_], 2).astype(np.float64)       # (S, n, 35): ego then plan, as H.X_val(["ego", "plan"])


def pick_of(pred):
    p = pred.copy()
    p[:, :, KEEP] = 0.0
    return np.stack([T.first(p[s].T) for s in range(S)])                                             # (S, m)


def realised(J, pk, idx):
    return np.mean([J[s][pk[s], idx] - J[s][KEEP, idx] for s in range(S)], 0)


def fit_apply(Xf, Jf, tr, inner, Xa=None):
    """Ridge of the candidates' RFS gain (fit set Jf, frames tr; lam by the inner folds on tr) -> a predictor for inputs (S, m, d)."""
    Y = (Jf - Jf[:, KEEP:KEEP + 1]).transpose(0, 2, 1)
    best, lam = -np.inf, None
    acc = np.zeros(len(H.LAMS))
    d = Xf.shape[2]
    for g in range(INNER):
        a_, b_ = tr[inner[tr] != g], tr[inner[tr] == g]
        P = H.ridge_fit(Xf[:, a_].reshape(-1, d), Y[:, a_].reshape(-1, 20))
        for li, l in enumerate(H.LAMS):
            pk = pick_of(P(Xf[:, b_].reshape(-1, d), l).reshape(S, len(b_), 20))
            acc[li] += realised(Jf, pk, b_).sum()
    for li in reversed(range(len(H.LAMS))):                                                          # ties to the larger lam, as H.select_fit
        if acc[li] > best + 1e-9:
            best, lam = acc[li], H.LAMS[li]
    P = H.ridge_fit(Xf[:, tr].reshape(-1, d), Y[:, tr].reshape(-1, 20))
    return lambda X: P(X.reshape(-1, d), lam).reshape(S, -1, 20), lam


def oof(Xf, Jf, Xa, Ja, fold, rep, inner_salt=1000):
    """Out-of-fold: for each fold j fit on the other frames (set f), pick on the frames of fold j (set a). -> per-frame gain over keep (n,), picks (S, n), lams."""
    n = Xf.shape[1]
    gain, picks, lams = np.zeros(n), np.zeros((S, n), int), []
    for j in range(fold.max() + 1):
        te, tr = np.flatnonzero(fold == j), np.flatnonzero(fold != j)
        pred, lam = fit_apply(Xf, Jf, tr, H.folds(rep * 10 + j, INNER, inner_salt))
        pk = pick_of(pred(Xa[:, te]).reshape(S, len(te), 20))
        gain[te], picks[:, te] = realised(Ja, pk, te), pk
        lams.append(lam)
    return gain, picks, lams


def ref_unit(a):
    """The decision-173 unit (E arm, WP2), for the reproduction gate."""
    return H.unit(a)[1]


# ---------------------------------------------------------------- stats helpers
def pair(C, d, rows=None):
    c = C.ci(d, rows)
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def strata(C):
    n, v0, it = C.n, C.v0, C.intent
    st = {"all": np.ones(n, bool), "turn-intent": it >= 2, "left": it == 2, "right": it == 3, "straight": it < 2}
    for nm, lo, hi in V_STRATA:
        m = (v0 >= lo) & (v0 < hi)
        st[nm] = m
        st[f"turn {nm}"] = m & (it >= 2)
        st[f"straight {nm}"] = m & (it < 2)
    return st


# ---------------------------------------------------------------- Step 1
def cand_geo(F):
    """F (S, n, 20, 20, 2) -> 5 s arc length (S, n, 20) and 5 s lateral position (S, n, 20) of every candidate."""
    s5 = np.stack([G.arc(f.reshape(-1, 20, 2).astype(np.float64))[:, -1].reshape(f.shape[:2]) for f in F])
    return s5, F[..., -1, 1].astype(np.float64)


def summarise(C, d, rows, picks, geo, side):
    """One row of the pick table for the frames `rows`: picks (R, S, n) over the repeats, geo = (s5, y5)."""
    s5, y5 = geo
    ii = np.flatnonzero(rows)
    pk = picks[:, :, ii]                                                    # (R, S, m)
    sx = lambda a: np.take_along_axis(np.broadcast_to(a[None, :, ii], (len(picks), *a[:, ii].shape)), pk[..., None], 3)[..., 0]   # noqa: E731
    d5, dk5 = s5[:, ii, KEEP][None], sx(s5)
    ratio = np.where(d5 > 0.5, dk5 / np.maximum(d5, 1e-9), np.nan)
    inward = (side[ii][None, None] * (sx(y5) - y5[:, ii, KEEP][None]))
    changed = pk != KEEP
    r = {"n": len(ii), "picks keep/follow": float((pk == KEEP).mean())}
    for j, nm in enumerate(G.SPEEDS):
        r[f"speed {nm}"] = float((pk % 4 == j).mean())
    for j, nm in enumerate("ABCDE"):
        r[f"path {nm}"] = float((pk // 4 == j).mean())
    r |= {"plan 5 s displacement (m)": float(d5.mean()), "pick 5 s displacement (m)": float(dk5.mean()), "median pick / plan displacement (plan > 0.5 m)": float(np.nanmedian(ratio)),
          "mean pick / plan displacement (plan > 0.5 m)": float(np.nanmean(ratio)), "inward shift of the pick at 5 s (m, + = towards the turn)": float(inward.mean()),
          "inward shift among changed-path picks": float(inward[changed & (pk // 4 != 2)].mean()) if (changed & (pk // 4 != 2)).any() else np.nan}
    return r


def cmd_desc(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("op_parity", "turn-prior-desc", seed=0, config=vars(a)) as run:
        C, z = ctx()
        d_ = load_sets(C, z)
        OUT.mkdir(parents=True, exist_ok=True)
        st, n = strata(C), C.n
        ego = z["va_ego"]
        res = {}
        for k in ("wp2", "wlg"):
            X, J = X_of(d_[f"pd_{k}"], ego), d_[f"J_{k}"]
            runs = [oof(X, J, X, J, H.folds(r), r) for r in range(REPS)]
            res[k] = dict(gain=np.mean([r[0] for r in runs], 0), picks=np.stack([r[1] for r in runs]), lams=[r[2] for r in runs])
            run.info("%s: floor head OOF %s over keep (10 reps)", k, pair(C, res[k]["gain"]))
        ref = np.load(T.RUNS / "heads.npz")["E|b|L"]
        run.info("reproduction of decision 173 E|b|L: ours %s, stored %s, max frame |diff| %.4f", pair(C, res["wp2"]["gain"]), pair(C, ref), np.abs(res["wp2"]["gain"] - ref).max())
        base = {k: d_[f"J_{k}"][:, KEEP].mean(0) for k in ("wp2", "wlg")}
        ship = d_["J_shipped"]
        side = np.where(C.intent == 2, 1.0, np.where(C.intent == 3, -1.0, 0.0))
        geo = {k: cand_geo(d_[f"F_{k}"]) for k in ("wp2", "wlg")}
        rows = []
        for sn, m in st.items():
            for k in ("wp2", "wlg"):
                g = res[k]["gain"]
                c = C.ci(g, m)
                r = {"plans": k.upper(), "stratum": sn, "n": int(m.sum()), "RFS plan": C.cm(base[k], m), "floor head OOF d": c[0], "lo": c[1], "hi": c[2],
                     "oracle F20 d": C.cm(d_[f"J_{k}"].max(1).mean(0) - base[k], m)}
                if k == "wp2":
                    r["WLG - WP2"] = pair(C, base["wlg"] - base["wp2"], m)
                    r["shipped - WP2"] = pair(C, ship - base["wp2"], m)
                rows.append(r | summarise(C, g, m, res[k]["picks"], geo[k], side))
        df = pd.DataFrame(rows)
        stats.write_table(rows, OUT / "desc_picks")
        # the oracle's choice on the same groups (privileged) for reference
        orows = []
        for sn in ("turn-intent", "left", "right", "turn v0<0.5", "turn 0.5<=v0<3", "turn v0>=3"):
            m = st[sn]
            for k in ("wp2", "wlg"):
                J = d_[f"J_{k}"]
                pk = np.stack([T.first(J[s]) for s in range(S)])[None]
                orows.append({"plans": k.upper(), "stratum": sn} | summarise(C, None, m, pk, geo[k], side))
        stats.write_table(orows, OUT / "desc_oracle")
        # per-frame table
        sh = lambda k, f: np.array([np.bincount(res[k]["picks"][:, :, i].ravel() % 4 if f == "speed" else res[k]["picks"][:, :, i].ravel() // 4, minlength=5).argmax() for i in range(n)])  # noqa: E731
        pd.DataFrame({"name": C.names[:n], "intent": C.intent, "v0": C.v0, "rfs_WP2": base["wp2"], "rfs_WLG": base["wlg"], "rfs_shipped": ship, "d_floor_wp2": res["wp2"]["gain"],
                      "d_floor_wlg": res["wlg"]["gain"], "mode_speed_wp2": sh("wp2", "speed"), "mode_path_wp2": sh("wp2", "path"),
                      "plan_s5_wp2": geo["wp2"][0][0, :, KEEP]}).to_csv(OUT / "desc_frames.csv", index=False, float_format="%.4f")
        drows = []
        for k in ("wp2", "wlg"):
            J, pk = d_[f"J_{k}"], res[k]["picks"]
            jj = np.arange(n)
            part = {"speed part (pick's speed, keep path)": lambda s, r: J[s][KEEP + pk[r, s] % 4, jj], "path part (pick's path, follow)": lambda s, r: J[s][(pk[r, s] // 4) * 4, jj],
                    "the pick": lambda s, r: J[s][pk[r, s], jj]}
            for nm, f in part.items():
                g = np.mean([f(s, r) - J[s][KEEP] for s in range(S) for r in range(REPS)], 0)
                for sn in ("all", "turn-intent", "turn v0<0.5", "turn 0.5<=v0<3", "turn v0>=3", "straight"):
                    c = C.ci(g, st[sn])
                    drows.append({"plans": k.upper(), "part": nm, "stratum": sn, "n": int(st[sn].sum()), "d": c[0], "lo": c[1], "hi": c[2]})
        stats.write_table(drows, OUT / "desc_decomp")
        pd.set_option("display.width", 250, "display.max_columns", 99)
        run.info("\n%s", pd.DataFrame(drows).to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("\n%s", df[df.stratum.str.startswith(("turn", "left", "right", "all", "straight"))].T.to_string(float_format=lambda v: f"{v:+.3f}"))
        run.info("\noracle:\n%s", pd.DataFrame(orows).T.to_string(float_format=lambda v: f"{v:+.3f}"))


# ---------------------------------------------------------------- Step 2
_W = {}                                                         # arrays shared with forked workers
N_PERM, PERM_REPS = 200, 3


def perm_unit(a):
    """One permuted-input control: the rows of the (fit and apply) inputs moved between frames (same permutation for both seeds), PERM_REPS fold repeats."""
    arm, i = a
    Xf, Jf, Xa, Ja, fold = (_W[arm][k] for k in ("Xf", "Jf", "Xa", "Ja", "fold"))
    pi = np.random.default_rng(5000 + i).permutation(Xf.shape[1])
    g = [oof(Xf[:, pi], Jf, Xa[:, pi], Ja, fold(r), r)[0] for r in range(PERM_REPS)]
    return arm, i, np.mean(g, 0)


def cmd_stack(a):
    import multiprocessing as mp
    import pandas as pd
    from jevdrive import par, stats
    from jevdrive.run import Run
    with Run("op_parity", "turn-prior-stack", seed=0, config=vars(a) | dict(n_perm=N_PERM, perm_reps=PERM_REPS)) as run:
        C, z = ctx()
        d_ = load_sets(C, z)
        OUT.mkdir(parents=True, exist_ok=True)
        n, ego, st = C.n, z["va_ego"], strata(C)
        fr = pd.read_csv(_R / "experiments/op_parity/results/wod_pref/frames.csv")
        assert (fr["name"].to_numpy() == C.names[:n]).all()
        pfold, cl = fr["fold"].to_numpy(), fr["cluster"].to_numpy().astype(str)
        for c in sorted(set(cl)):
            st[f"cluster {c}"] = cl == c
        X = {k: X_of(d_[f"pd_{k}"], ego) for k in SETS}
        J = {k: d_[f"J_{k}"] for k in SETS}
        base = {k: J[k][:, KEEP].mean(0) for k in SETS} | {"shipped": d_["J_shipped"]}
        tf = lambda r: H.folds(r)                                                    # noqa: E731  decision-173 folds (random per repeat)
        pf = lambda r: pfold                                                         # noqa: E731  wod_pref outer folds (fixed; the repeat only moves the inner folds)
        # arms: name -> (fit set, apply set, fold function, base plan set)
        spec = {"WLG+floor (173 folds)": ("wlg", "wlg", tf, "wlg"), "WLG+floor (pref folds)": ("wlg", "wlg", pf, "wlg"),
                "pref-top+floor (nested)": ("wlg", "top", pf, "top"), "pref-f20+floor (nested)": ("wlg", "f20", pf, "f20"),
                "pref-top+floor (head on pref plans, not nested)": ("top", "top", pf, "top"), "pref-f20+floor (head on pref plans, not nested)": ("f20", "f20", pf, "f20")}
        gain, lams, picks = {}, {}, {}
        for nm, (fs, as_, fo, _) in spec.items():
            runs = [oof(X[fs], J[fs], X[as_], J[as_], fo(r), r) for r in range(REPS)]
            gain[nm], picks[nm] = np.mean([r[0] for r in runs], 0), np.stack([r[1] for r in runs])
            lams[nm] = [x for r in runs for x in r[2]]
            run.info("%s: %s over its base plan", nm, pair(C, gain[nm]))
        # permutation controls (the three arms that are read as 'delivers' candidates)
        perm_arms = {"WLG+floor (173 folds)": ("wlg", "wlg", tf), "pref-top+floor (nested)": ("wlg", "top", pf), "pref-f20+floor (nested)": ("wlg", "f20", pf)}
        for nm, (fs, as_, fo) in perm_arms.items():
            _W[nm] = dict(Xf=X[fs], Jf=J[fs], Xa=X[as_], Ja=J[as_], fold=fo)
        t0 = time.time()
        res = par.pmap(perm_unit, [(nm, i) for nm in perm_arms for i in range(N_PERM)], workers=min(a.workers, 64), run=run, desc="permutations", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        run.info("permutations: %d units in %.0f s", len(res.values), time.time() - t0)
        perm = {nm: np.stack([g for arm, i, g in res.values if arm == nm]) for nm in perm_arms}                    # (N_PERM, n)
        # ---- tables
        floor_of = {nm: spec[nm][3] for nm in spec}
        arms = {"shipped": base["shipped"], "WP2": base["wp2"], "WLG": base["wlg"], "pref-top (oof)": base["top"], "pref-f20 (oof)": base["f20"]}
        arms |= {nm: base[spec[nm][3]] + g for nm, g in gain.items()}
        gated = {"WLG+floor (173 folds)": "WLG+floor, turn-gated (post hoc)", "pref-top+floor (nested)": "pref-top+floor, turn-gated (post hoc)",
                 "pref-f20+floor (nested)": "pref-f20+floor, turn-gated (post hoc)"}
        for nm, gn in gated.items():                                                  # the head's pick is used only where the command says left / right
            arms[gn] = base[spec[nm][3]] + gain[nm] * (C.intent >= 2)
            spec[gn], floor_of[gn] = spec[nm], spec[nm][3]
        main, srows, prows = [], [], []
        for nm, v in arms.items():
            r = {"arm": nm, "RFS": C.cm(v)}
            for ref, rv in (("WLG", arms["WLG"]), ("shipped", arms["shipped"])):
                c = C.ci(v - rv)
                r |= {f"d vs {ref}": c[0], f"{ref} lo": c[1], f"{ref} hi": c[2]}
            if nm in spec:
                bs = arms[{"wlg": "WLG", "top": "pref-top (oof)", "f20": "pref-f20 (oof)"}[floor_of[nm]]]
                c = C.ci(v - bs)
                r |= {"d vs its base": c[0], "base lo": c[1], "base hi": c[2]}
            if nm in perm_arms:
                pm = np.array([C.cm(g) for g in perm[nm]])
                act = C.cm(gain[nm])
                r |= {"perm null mean": float(pm.mean()), "perm null lo": float(np.percentile(pm, 2.5)), "perm null hi": float(np.percentile(pm, 97.5)),
                      "perm p": float((1 + (pm >= act).sum()) / (1 + len(pm)))}
            main.append(r)
            for sn, m in st.items():
                c = C.ci(v - arms["WLG"], m)
                cs = C.ci(v - arms["shipped"], m)
                srows.append({"arm": nm, "stratum": sn, "n": int(m.sum()), "RFS": C.cm(v, m), "d vs WLG": c[0], "lo": c[1], "hi": c[2], "d vs shipped": cs[0], "s lo": cs[1], "s hi": cs[2],
                                 "d vs WLG (frame mean)": float((v - arms["WLG"])[m].mean())})
        stats.write_table(main, OUT / "stack_arms")
        stats.write_table(srows, OUT / "stack_strata")
        pd.DataFrame({"name": C.names[:n], "fold": pfold, "v0": C.v0, "intent": C.intent} | {f"rfs {k}": v for k, v in arms.items()}).to_csv(OUT / "stack_frames.csv", index=False,
                                                                                                                                          float_format="%.4f")
        np.savez(RUNS / "stack.npz", **{f"gain|{k}": v for k, v in gain.items()}, **{f"perm|{k}": v for k, v in perm.items()}, **{f"picks|{k}": v for k, v in picks.items()})
        pd.DataFrame([{"arm": k, "lams": " ".join(f"{x:g}" for x in v)} for k, v in lams.items()]).to_csv(OUT / "stack_lams.csv", index=False)
        pd.set_option("display.width", 250, "display.max_columns", 99)
        run.info("\n%s", pd.DataFrame(main).to_string(float_format=lambda v: f"{v:+.3f}"))
        sd = pd.DataFrame(srows)
        run.info("\n%s", sd[sd.stratum.isin(["all", "turn-intent", "straight", "turn v0<0.5", "turn 0.5<=v0<3", "turn v0>=3"])].to_string(float_format=lambda v: f"{v:+.3f}"))


# ---------------------------------------------------------------- Step 3: the rule family R(alpha, nudge, v-range)
ALPHAS, NUDGES, VRANGES = (1.0, 0.85, 0.7, 0.5), (0.0, 1.2), ((0.5, 3.0), (0.5, 1e9), (0.0, 1e9))
CONFIGS = [(a_, d_, v_) for v_ in VRANGES for d_ in NUDGES for a_ in ALPHAS]
CONFIGS = sorted(CONFIGS, key=lambda c: (c[1] != 0 or c[0] != 1.0, c[1] != 0, 1 - c[0], VRANGES.index(c[2])))          # identity first, then the smaller change
IDENT = CONFIGS[0]
assert IDENT[0] == 1.0 and IDENT[1] == 0.0


def apply_rule(plan, v0, intent, cfg):
    """Plan (n, 20, 2) -> plan with the rule applied on the frames it is eligible for (turn-intent, v0 inside the range)."""
    al, dy, (lo, hi) = cfg
    out = plan.copy()
    for side, code in ((1.0, 2), (-1.0, 3)):
        m = np.flatnonzero((intent == code) & (v0 >= lo) & (v0 < hi))
        if len(m) == 0 or (al == 1.0 and dy == 0):
            continue
        p = plan[m]
        out[m] = G.offset_path(p, v0[m], al * G.arc(p), -side * dy, 10.0, 2.0) if dy else G.along(p, al * G.arc(p))
    return out


def rule_scores(C, P):
    """RFS of every config on every frame: (len(CONFIGS), S, n)."""
    return np.stack([np.stack([C.rfs(apply_rule(p, C.v0, C.intent, c)) for p in P]) for c in CONFIGS])


def log_alpha(C):
    """3a-log: median of (logged 5 s displacement / plan 5 s displacement) on the r2-dev turn-intent rows with 0.5 <= v0 < 3 (WP2 token-path plans; out of sample)."""
    from jevdrive import waymo as W
    d = np.load(data_dir() / "runs/op_parity/wod/launch/tok_dev.npz")
    v0 = W.init_speed(d["past"]).astype(np.float64)
    m = (d["intent"] >= 2) & (v0 >= 0.5) & (v0 < 3.0)
    rr = {}
    for t in ("WP2-full-s0", "WP2-full-s1"):
        s_log, s_plan = G.arc(d["fut"][m][..., :2].astype(np.float64))[:, -1], G.arc(d[f"plan_{t}"][m].astype(np.float64))[:, -1]
        rr[t] = float(np.median(s_log / np.maximum(s_plan, 0.5)))
    return float(np.mean(list(rr.values()))), int(m.sum()), rr


def cmd_rule(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.run import Run
    with Run("op_parity", "turn-prior-rule", seed=0, config=vars(a) | dict(configs=[list(map(str, c)) for c in CONFIGS])) as run:
        C, z = ctx()
        d_ = load_sets(C, z)
        n, st = C.n, strata(C)
        fr = pd.read_csv(_R / "experiments/op_parity/results/wod_pref/frames.csv")
        pfold, cl = fr["fold"].to_numpy(), fr["cluster"].to_numpy().astype(str)
        for c in sorted(set(cl)):
            st[f"cluster {c}"] = cl == c
        P = {k: plans_of(C, k) for k in ("wlg", "top", "f20")}
        R = {k: rule_scores(C, P[k]).mean(1) for k in P}                                        # (configs, n): seed-mean RFS
        base = {k: R[k][0] for k in P}
        assert all(np.abs(base[k] - d_[f"J_{k}"][:, KEEP].mean(0)).max() < 1e-9 for k in P)
        ship = d_["J_shipped"]
        turn = C.intent >= 2
        al, n_log, rr = log_alpha(C)
        run.info("alpha from the r2-dev logs: %.3f (%d turn rows, per seed %s)", al, n_log, rr)
        # ---- in-sample grid (descriptive, fitted and scored on the same frames)
        grid = []
        for ci_, c in enumerate(CONFIGS):
            g = R["wlg"][ci_] - base["wlg"]
            cs, ct = C.ci(g), C.ci(g, turn)
            grid.append({"alpha": c[0], "nudge m": c[1], "v-range": f"[{c[2][0]:g}, {c[2][1]:g})", "d vs WLG (all, in-sample)": cs[0], "lo": cs[1], "hi": cs[2], "d on turn-intent": ct[0], "turn lo": ct[1],
                         "turn hi": ct[2]})
        stats.write_table(grid, OUT / "rule_grid_insample")
        # ---- 3a-oof: parameters from the 4 other pref folds' turn-intent frames (WLG plans), applied to the held-out fold
        chosen, params = np.zeros(n, int), []
        for j in range(K := pfold.max() + 1):
            tr = (pfold != j) & turn
            gains = (R["wlg"][:, tr] - base["wlg"][tr]).mean(1)
            best = int(np.flatnonzero(gains >= gains.max() - 1e-9)[0])                           # CONFIGS is ordered identity first, then the smaller change
            chosen[pfold == j] = best
            params.append({"fold": j, "alpha": CONFIGS[best][0], "nudge m": CONFIGS[best][1], "v-range": f"[{CONFIGS[best][2][0]:g}, {CONFIGS[best][2][1]:g})",
                           "train turn frames": int(tr.sum()), "train gain": float(gains[best]), "n eligible train": int(((pfold != j) & turn & (C.v0 >= CONFIGS[best][2][0]) & (C.v0 < CONFIGS[best][2][1])).sum())})
        stats.write_table(params, OUT / "rule_params")
        ii = np.arange(n)
        cfg_log = (al, 0.0, (0.5, 3.0))
        Rlog = {k: np.mean([C.rfs(apply_rule(p, C.v0, C.intent, cfg_log)) for p in P[k]], 0) for k in P}
        arms = {"shipped": ship, "WLG": base["wlg"], "pref-top (oof)": base["top"], "pref-f20 (oof)": base["f20"]}
        bmap = {}
        for k, lab in (("wlg", "WLG"), ("top", "pref-top"), ("f20", "pref-f20")):
            arms[f"{lab}+rule (oof params)"] = R[k][chosen, ii]
            arms[f"{lab}+rule (log alpha %.2f)" % al] = Rlog[k]
            bmap[f"{lab}+rule (oof params)"] = bmap[f"{lab}+rule (log alpha %.2f)" % al] = {"wlg": "WLG", "top": "pref-top (oof)", "f20": "pref-f20 (oof)"}[k]
        main, srows = [], []
        for nm, v in arms.items():
            r = {"arm": nm, "RFS": C.cm(v)}
            for ref, rv in (("WLG", arms["WLG"]), ("shipped", arms["shipped"])):
                c = C.ci(v - rv)
                r |= {f"d vs {ref}": c[0], f"{ref} lo": c[1], f"{ref} hi": c[2]}
            if nm in bmap:
                c = C.ci(v - arms[bmap[nm]])
                r |= {"d vs its base": c[0], "base lo": c[1], "base hi": c[2]}
            main.append(r)
            for sn, m in st.items():
                c, cs = C.ci(v - arms["WLG"], m), C.ci(v - arms["shipped"], m)
                srows.append({"arm": nm, "stratum": sn, "n": int(m.sum()), "RFS": C.cm(v, m), "d vs WLG": c[0], "lo": c[1], "hi": c[2], "d vs shipped": cs[0], "s lo": cs[1], "s hi": cs[2],
                                 "d vs WLG (frame mean)": float((v - arms["WLG"])[m].mean())})
        stats.write_table(main, OUT / "rule_arms")
        stats.write_table(srows, OUT / "rule_strata")
        pd.DataFrame({"name": C.names[:n], "fold": pfold, "chosen": chosen} | {f"rfs {k}": v for k, v in arms.items()}).to_csv(OUT / "rule_frames.csv", index=False, float_format="%.4f")
        (OUT / "rule_info.json").write_text(json.dumps({"alpha_log": al, "alpha_log_rows": n_log, "alpha_log_per_seed": rr, "configs": [[c[0], c[1], list(c[2])] for c in CONFIGS]}, indent=1))
        pd.set_option("display.width", 250, "display.max_columns", 99)
        run.info("\ngrid (in-sample):\n%s\nparams:\n%s\n%s", pd.DataFrame(grid).to_string(float_format=lambda v: f"{v:+.3f}"), pd.DataFrame(params).to_string(),
                 pd.DataFrame(main).to_string(float_format=lambda v: f"{v:+.3f}"))
        sd = pd.DataFrame(srows)
        run.info("\n%s", sd[sd.stratum.isin(["all", "turn-intent", "straight", "turn v0<0.5", "turn 0.5<=v0<3", "turn v0>=3"]) & sd.arm.str.contains("rule")].to_string(float_format=lambda v: f"{v:+.3f}"))


def cmd_figs(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    blue, orange, grey = "#0072B2", "#D55E00", "#888888"
    # ---- figure 1: the floor head's gain on the turn-intent frames by v0 stratum, on WP2 and on WLG plans, split into its speed and path parts
    dc = pd.read_csv(OUT / "desc_decomp.csv")
    strata_ = ["turn-intent", "turn v0<0.5", "turn 0.5<=v0<3", "turn v0>=3", "straight"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, part in zip(axs, ("the pick", "speed part (pick's speed, keep path)", "path part (pick's path, follow)")):
        for off, (pl, col) in zip((-0.18, 0.18), (("WP2", grey), ("WLG", blue))):
            d = dc[(dc.plans == pl) & (dc.part == part)].set_index("stratum").loc[strata_]
            y = np.arange(len(strata_)) + off
            ax.errorbar(d["d"], y, xerr=[d["d"] - d["lo"], d["hi"] - d["d"]], fmt="o", color=col, ms=4, capsize=2, label=f"on {pl} plans")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title({"the pick": "whole pick", "speed part (pick's speed, keep path)": "speed part only", "path part (pick's path, follow)": "path part only"}[part], fontsize=9)
        ax.set_yticks(range(len(strata_)), [f"{s} (n={int(dc[dc.stratum == s].n.iloc[0])})" for s in strata_])
        ax.set_xlabel("out-of-fold gain over the plan (RFS)")
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "floor_head_parts.png")
    # ---- figure 2: forest plot of every arm against WLG
    A = pd.concat([pd.read_csv(OUT / "stack_arms.csv"), pd.read_csv(OUT / "rule_arms.csv")]).drop_duplicates("arm")
    order = ["shipped", "WP2", "pref-top (oof)", "pref-f20 (oof)", "WLG+floor (173 folds)", "WLG+floor, turn-gated (post hoc)", "pref-top+floor (nested)", "pref-f20+floor (nested)",
             "pref-top+floor, turn-gated (post hoc)", "pref-f20+floor, turn-gated (post hoc)", "WLG+rule (oof params)", "pref-top+rule (oof params)", "pref-f20+rule (oof params)",
             [x for x in A.arm if "log alpha" in x and x.startswith("WLG")][0]]
    A = A.set_index("arm").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))
    col = [grey if o in ("shipped", "WP2") else orange if "post hoc" in o else blue for o in order]
    for yi, (o, r), c in zip(y, A.iterrows(), col):
        ax.errorbar(r["d vs WLG"], yi, xerr=[[r["d vs WLG"] - r["WLG lo"]], [r["WLG hi"] - r["d vs WLG"]]], fmt="o", color=c, ms=4, capsize=2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("RFS difference to WLG (cluster mean, paired bootstrap over sequences)")
    fig.tight_layout()
    fig.savefig(FIG / "arms_vs_wlg.png")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prep", "desc", "stack", "rule", "figs"):
        p = sp.add_parser(c)
        p.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
