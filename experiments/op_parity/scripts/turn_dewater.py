"""op_parity turn-ceiling de-water (plans/2026-10-08-turn-ceiling-dewater-prereg.md): how much of the F19 best-of-K ceiling (+12.20, decision 178)
is an artefact of the 4 s horizon and of EP farming, and is the per-token pick learnable. Reads only the per-token candidate scores of turn_ceiling
(score_all.csv); no new simulator scoring, no training of a base model, CPU only.

  ceiling    de-watered ceilings: families (no slow-down, lateral only) x score conventions (raw / epfix / epcap / pc), same buckets and bootstrap
             as turn_ceiling.py report; Shapley split and failure rates of the main rows -> <out>/, figs
  select     cross-fitted selectors (folds by log): ridge / ridge + margin / boosted trees on ego, plan, seed disagreement, shipped-teacher and frozen
             vision streams; constant, permutation, in-sample and learning-curve controls -> $OUT/select.pkl
  selreport  tables and figures of `select` -> <out>/, figs
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import sys as _sys, pathlib as _pl  # noqa: E401,E402
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research"), str(_pl.Path(__file__).parent)]
import argparse, json, pickle  # noqa: E401,E402

import numpy as np  # noqa: E402

import turn_ceiling as TC  # noqa: E402

D = TC.D
OUT = D / "runs/op_parity/turn_dewater"
RES = _R / "experiments/op_parity/results/turn_ceiling_dewater"
FIG = _R / "experiments/op_parity/figs/turn_ceiling_dewater"
SCORE = TC.OUT / "score_all.csv"
LINE = 4.0                                              # pre-registered: P19 = F19 x pc on > 20 deg, EPDMS x 100
WOD_RECOVERY = 0.14                                     # decision 173
MAPT = [1, 2, 3, 6]                                     # DAC, DDC, TLC, LK in bench TERMS order: terms a slow-down may not repair under `pc`
EP = 4
CONVS = ("raw", "epfix", "epcap", "pc")
MAIN = (("F19", "raw"), ("F19ns", "raw"), ("L9", "epcap"), ("F19", "pc"))          # the selector's family x convention; the last is the trusted one
LAMS = (3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e6)
MS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16)         # margins of the L+m head, score units (0 = plain argmax)
KS, FOLDS, REPS, INNER, S = (8, 32, 128), 5, 10, 4, 2
N_PERM, PERM_REPS, CURVE, GK = 100, 2, (0.25, 0.5, 0.75), 32
ARMS = {"E0": ["ego"], "E": ["ego", "plan"], "E+D": ["ego", "plan", "dis"], "T+E": ["T", "ego", "plan"], "V+E": ["V", "ego", "plan"],
        "V+T+E": ["V", "T", "ego", "plan"]}
VIS = ("T", "V")
_D = {}                                                 # arrays shared with forked workers


# ---------------------------------------------------------------- scores, conventions, families
def load(score=SCORE):
    """-> candidates, tokens, dyaw, logs, X (seeds, C, N, 9) sub-scores in bench TERMS order with EC = NaN."""
    import pandas as pd
    from jevdrive.bench.navsim import SUBS
    C, (tok, dyaw) = TC.candidates(), TC.bucket_tokens()
    df = pd.read_csv(score)
    X = np.full((len(TC.SEEDS), len(C), len(tok), 9), np.nan)
    ti = {t: i for i, t in enumerate(tok)}
    for kname, g in df.groupby("key"):
        s, c = TC.SEEDS.index(int(kname[1])), int(kname.split("_c")[1])
        X[s, c, g.token.map(ti).to_numpy(), :8] = g[[SUBS[k] for k in TC.SUB8]].to_numpy(float)
    assert np.isfinite(X[..., :8]).all(), "score CSV does not cover every (seed, candidate, token)"
    return C, tok, dyaw, TC.archived(TC.MODEL.format(TC.SEEDS[0]), tok)[1], X


def conv(X, C, name):
    """Sub-scores under a score convention. epfix: EP of the identity; epcap: EP gains not credited; pc: epcap, and a candidate with speed < 1
    cannot have a better map term (DAC / DDC / TLC / LK) than the same (offset, gain) at speed 1, whose path it follows a prefix of."""
    Y = X.copy()
    if name == "epfix":
        Y[..., EP] = X[:, :1, :, EP]
    if name in ("epcap", "pc"):
        Y[..., EP] = np.minimum(X[..., EP], X[:, :1, :, EP])
    if name == "pc":
        at = {c[1:]: i for i, c in enumerate(C)}
        for i, (_, o, k, v) in enumerate(C):
            if v < 1:
                j = at[(o, k, 1.0)]
                for t in MAPT:
                    Y[:, i, :, t] = np.minimum(X[:, i, :, t], X[:, j, :, t])
    return Y


def fams(C):
    F = TC.families(C)
    v = np.array([c[3] for c in C])
    at = {c[1:]: i for i, c in enumerate(C)}
    for f in ("F19", "F27", "F33"):
        F[f + "ns"] = [i for i in F[f] if v[i] >= 1]
    F["L9"], F["L13"] = [i for i in F["F27"] if v[i] == 1], [i for i in F["F33"] if v[i] == 1]
    F["Vup"], F["Vdn"] = [0, at[(0.0, 1.0, 1.2)]], [0, at[(0.0, 1.0, 0.8)]]
    assert all(x[0] == 0 for x in F.values())
    return F


def buckets(dyaw):
    return {"> 20 deg": np.ones(len(dyaw), bool), "20-45 deg": np.abs(dyaw) < 45, "> 45 deg": np.abs(dyaw) >= 45, "left > 20 deg": dyaw > 0,
            "right > 20 deg": dyaw < 0}


def oracle(Sc, idx):
    """Scores (seeds, C, N) under one convention -> picks (seeds, N) within the family idx (ties to the earliest candidate), best (N,) seed mean."""
    return np.asarray(idx)[Sc[:, idx].argmax(1)], Sc[:, idx].max(1).mean(0)


def at_pick(A, p):
    """A (seeds, C, N, ...) at the picks p (seeds, N)."""
    return np.stack([A[s, p[s], np.arange(A.shape[2])] for s in range(len(p))])


# ---------------------------------------------------------------- 1. de-watered ceilings
def cmd_ceiling(a):
    from jevdrive import stats
    from jevdrive.bench import tables as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_dewater/ceiling", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        out, fig = _pl.Path(a.out), _pl.Path(a.figs)
        out.mkdir(parents=True, exist_ok=True)
        C, tok, dyaw, log, X = load(a.score)
        assert nt.mask(tok).all()
        F, B = fams(C), buckets(dyaw)
        Yc = {cv: conv(X, C, cv) for cv in CONVS}
        Sc = {cv: TC.noec(y) for cv, y in Yc.items()}
        idn = Sc["raw"][:, 0].mean(0)
        assert all(np.array_equal(Sc[cv][:, 0], Sc["raw"][:, 0]) for cv in CONVS), "the identity must score the same under every convention"
        order = ["F19", "F19ns", "L9", "F27", "F27ns", "L13", "F33", "F33ns", "O3", "K3", "V3", "Vup", "Vdn"]
        G, P, BEST, rows, rows_raw, rows_pk = {}, {}, {}, [], [], []
        ax = np.array([c[1:] for c in C])
        for f in order:
            for cv in CONVS:
                if cv == "pc" and not any(ax[i, 2] < 1 for i in F[f]):
                    continue                                                    # pc = epcap when the family has no slow-down
                p, best = oracle(Sc[cv], F[f])
                rawv = at_pick(Sc["raw"], p).mean(0)
                P[f, cv], BEST[f, cv] = p, best
                r, rr = dict(family=f, K=len(F[f]), convention=cv), dict(family=f, K=len(F[f]), convention=cv)
                for b, m in B.items():
                    G[f, cv, b] = TC.ci(best[m], idn[m], log[m])
                    r[b] = TC.cell(G[f, cv, b])
                    rr[b] = TC.cell(TC.ci(rawv[m], idn[m], log[m]))
                rows.append(r), rows_raw.append(rr)
                mv = ax[p]                                                      # (seeds, N, 3)
                rows_pk.append(dict(family=f, convention=cv, **{"identity %": 100 * (p == 0).mean(), "speed < 1 %": 100 * (mv[..., 2] < 1).mean(),
                                    "speed > 1 %": 100 * (mv[..., 2] > 1).mean(), "offset moved %": 100 * (mv[..., 0] != 0).mean(),
                                    "curvature moved %": 100 * (mv[..., 1] != 1).mean()}))
        note = "EPDMS x 100 without EC, seed mean of the per-seed oracle, paired log-cluster bootstrap (B 10000). Conventions: raw = as scored; " \
               "epfix = EP of the identity; epcap = EP gains not credited; pc = epcap + a slow-down cannot repair DAC / DDC / TLC / LK"
        stats.write_table(rows, out / "ceiling", note="best-of-K under the convention minus SH30. " + note)
        stats.write_table(rows_raw, out / "ceiling_rawvalued", note="picked under the convention, valued by the raw score, minus SH30. " + note)
        stats.write_table(rows_pk, out / "picks_axis", floatfmt=".1f", note="% of (token, seed) picks of the oracle under the convention, > 20 deg")
        # reference without any candidate: the better of the two seeds' own plans per token
        r = dict(reference="best of the 2 seeds' identity plans - seed mean")
        seedmax = Sc["raw"][:, 0].max(0)
        for b, m in B.items():
            G["seedmax", b] = TC.ci(seedmax[m], idn[m], log[m])
            r[b] = TC.cell(G["seedmax", b])
        stats.write_table([r], out / "seed_oracle", note="scale reference: what swapping in another plan of the same distribution is worth")
        # contrasts
        cons = [("slow-down candidates: F19 raw - F19ns raw", ("F19", "raw"), ("F19ns", "raw")), ("EP: F19 raw - F19 epfix", ("F19", "raw"), ("F19", "epfix")),
                ("EP gains: F19 raw - F19 epcap", ("F19", "raw"), ("F19", "epcap")), ("slow-down map repairs: F19 epcap - F19 pc", ("F19", "epcap"), ("F19", "pc")),
                ("all water: F19 raw - F19 pc", ("F19", "raw"), ("F19", "pc")), ("speed axis on NC / TTC: F19 pc - L9 epcap", ("F19", "pc"), ("L9", "epcap")),
                ("speed-up: F19ns raw - L9 raw", ("F19ns", "raw"), ("L9", "raw")), ("EP gains inside lateral: L9 raw - L9 epcap", ("L9", "raw"), ("L9", "epcap")),
                ("speed-up without EP: F19ns epcap - L9 epcap", ("F19ns", "epcap"), ("L9", "epcap")), ("outer lateral points: L13 epcap - L9 epcap", ("L13", "epcap"), ("L9", "epcap"))]
        rows, CON = [], {}
        for name, k1, k2 in cons:
            r = dict(contrast=name)
            for b, m in B.items():
                CON[name, b] = TC.ci(BEST[k1][m], BEST[k2][m], log[m])
                r[b] = TC.cell(CON[name, b])
            rows.append(r)
        stats.write_table(rows, out / "contrasts", note="differences of ceilings, EPDMS x 100; paired log-cluster bootstrap")
        # sub-score split and failure rates of the main rows
        GRP = {"DAC": ["DAC"], "NC": ["NC"], "TTC": ["TTC"], "EP": ["EP"], "LK": ["LK"], "rest": ["DDC", "TLC", "HC", "EC"]}
        gcol = {g: [T.TERMS.index(t) for t in ts] for g, ts in GRP.items()}
        rows, rates = [], []
        XW = TC.archived(TC.REF, tok)[0]
        XWn = np.array(XW, float)
        XWn[:, 8] = np.nan
        arms = {"SH30": X[:, 0], "WA-JEPA": np.stack([XWn] * len(X))}
        for f, cv in MAIN:
            Yp = at_pick(Yc[cv], P[f, cv])
            arms[f"oracle {f} x {cv}"] = Yp
            ph = np.mean([T.shapley(x, y) for x, y in zip(X[:, 0], Yp)], 0) * 100
            for b in ("> 20 deg", "20-45 deg", "> 45 deg"):
                m = B[b]
                tot = stats.bootstrap(ph[m].sum(1), groups=log[m])
                r = dict(row=f"oracle {f} x {cv} - SH30", bucket=b, total=TC.cell(tot))
                for g, cols in gcol.items():
                    r[g] = TC.cell(stats.bootstrap(ph[m][:, cols].sum(1), groups=log[m]))
                rows.append(r)
        stats.write_table(rows, out / "subscores", note="exact Shapley split of the oracle gain under the row's convention (x 100, 8 terms, no EC); rest = DDC + TLC + HC")
        for b in ("> 20 deg", "20-45 deg", "> 45 deg"):
            m = B[b]
            for name, x in arms.items():
                rates.append(dict(bucket=b, arm=name, **{f"{t} fail %": 100 * (x[:, m, T.TERMS.index(t)] < 1).mean() for t in ("NC", "DAC", "DDC", "TLC", "TTC", "LK")},
                                  **{"EP": 100 * x[:, m, EP].mean(), "EPDMS (no EC)": 100 * TC.noec(x)[:, m].mean()}))
        stats.write_table(rates, out / "rates", floatfmt=".2f", note="failure = sub-score < 1 (% of token-seeds) under the row's convention; EP mean x 100")
        b0 = "> 20 deg"
        p19, l9 = G["F19", "pc", b0], G["L9", "epcap", b0]
        wa = TC.ci(TC.noec(XW), idn, log)
        vd = dict(line=LINE, trusted="F19 x pc", P19=p19, L9_epcap=l9, F19_raw=G["F19", "raw", b0], F19ns_raw=G["F19ns", "raw", b0], L9_raw=G["L9", "raw", b0],
                  F19_epfix=G["F19", "epfix", b0], F19ns_epcap=G["F19ns", "epcap", b0], seed_oracle=G["seedmax", b0], wa_minus_sh30=wa,
                  ends_line=bool(p19["mean"] < LINE), lateral_alone_enough=bool(l9["mean"] >= LINE), bracket_gap=p19["mean"] - l9["mean"],
                  navtest_equivalent=p19["mean"] * len(tok) / 12146, n_tokens=len(tok), n_logs=len(set(log)), K={f: len(F[f]) for f in order})
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        run.summary.update(P19=p19["mean"], P19_lo=p19["lo"], P19_hi=p19["hi"], L9_epcap=l9["mean"], F19_raw=G["F19", "raw", b0]["mean"], ends_line=vd["ends_line"])
        run.info("verdict: %s", json.dumps(vd, default=float))
        fig_ceiling(fig, G, wa["mean"])


def fig_ceiling(FIGD, G, wa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGD.mkdir(parents=True, exist_ok=True)
    rows = [("F19\nraw\n(decision 178)", "F19", "raw"), ("F19\nEP fixed", "F19", "epfix"), ("no slow-down\nraw", "F19ns", "raw"), ("no slow-down\nno EP gain", "F19ns", "epcap"),
            ("lateral only\nraw", "L9", "raw"), ("lateral only\nno EP gain", "L9", "epcap"), ("F19\npath-consistent\n(trusted)", "F19", "pc")]
    bs = ("20-45 deg", "> 45 deg", "> 20 deg")
    col = dict(zip(bs, (ps.PALETTE["sky_blue"], ps.PALETTE["vermillion"], ps.PALETTE["black"])))
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 2.8), constrained_layout=True)
    w = 0.26
    for j, b in enumerate(bs):
        v = np.array([[G[f, cv, b]["mean"], G[f, cv, b]["mean"] - G[f, cv, b]["lo"], G[f, cv, b]["hi"] - G[f, cv, b]["mean"]] for _, f, cv in rows])
        ax.bar(np.arange(len(rows)) + (j - 1) * w, v[:, 0], w, yerr=v[:, 1:].T, color=col[b], label=b, error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(LINE, color=ps.BASELINE, ls="--", lw=0.8)
    ax.text(len(rows) - 0.5, LINE, "pre-registered line", ha="right", va="bottom", fontsize=7, color=ps.BASELINE)
    ax.axhline(wa, color=ps.PALETTE["green"], ls=":", lw=0.8)
    ax.text(len(rows) - 0.5, wa, "WA-JEPA - SH30, > 20 deg", ha="right", va="bottom", fontsize=7, color=ps.PALETTE["green"])
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_ylabel("best-of-K - SH30, EPDMS x 100 (no EC)")
    ax.legend(loc="upper right", ncols=3), ps.bars(ax)
    fig.savefig(FIGD / "ceiling_dewater.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 2. selectors
def plan_desc(P):
    """(N, 8, 3) rear-axle poses at 0.5 .. 4 s -> (N, 15): arc length at 1 / 2 / 3 / 4 s, lateral position and heading at 2 / 4 s, first and last
    segment speed, max |curvature| and its signed value, max lateral acceleration, |y(4 s)|, |heading(4 s)|."""
    Q = np.concatenate([np.zeros((len(P), 1, 3)), np.asarray(P, np.float64)], 1)
    Q[..., 2] = np.unwrap(Q[..., 2], axis=1)
    seg = np.hypot(*np.moveaxis(np.diff(Q[..., :2], axis=1), -1, 0))
    s = np.cumsum(seg, 1)
    kap = np.diff(Q[..., 2], axis=1) / np.maximum(seg, 0.25)
    j = np.abs(kap).argmax(1)[:, None]
    return np.concatenate([s[:, [1, 3, 5, 7]], Q[:, [4, 8], 1], Q[:, [4, 8], 2], seg[:, [0, -1]] / 0.5, np.abs(np.take_along_axis(kap, j, 1)),
                           np.take_along_axis(kap, j, 1), (np.abs(kap) * (seg / 0.5) ** 2).max(1, keepdims=True), np.abs(Q[:, [8], 1]), np.abs(Q[:, [8], 2])], 1)


def pca_white(A, fit, use, k=max(KS)):
    """Standardise and whiten on the rows `fit` (navtest tokens outside the evaluation set, no labels), project the rows `use` -> (len(use), k)."""
    mu, sd = A[fit].mean(0), A[fit].std(0) + 1e-6
    _, sv, Vt = np.linalg.svd((A[fit] - mu) / sd, full_matrices=False)
    return (((A[use] - mu) / sd) @ (Vt[:k].T / (sv[:k] / np.sqrt(len(fit))))).astype(np.float64)


def features(tok):
    """Test-time inputs of the turn tokens: {stream: (S, n, d)}. Nothing here reads a score, the logged future or the turn label."""
    cdir = D / "runs/op_parity/cache"
    tab = np.load(cdir / "lb_navtest/tab.npz")
    row = {t: i for i, t in enumerate(tab["names"].tolist())}
    use = np.array([row[t] for t in tok])
    fit = np.setdiff1d(np.arange(len(row)), use)
    Z = np.load(TC.OUT / "poses.npz")
    assert np.array_equal(Z["tokens"], tok)
    P = [Z[TC.key(s, 0)].astype(np.float64) for s in TC.SEEDS]
    d = P[0] - P[1]
    pd_ = [plan_desc(p) for p in P]
    dis = np.stack([np.hypot(d[:, -1, 0], d[:, -1, 1]), np.abs(pd_[0][:, 3] - pd_[1][:, 3]), np.abs(d[:, -1, 1]), np.abs(np.angle(np.exp(1j * d[:, -1, 2])))], 1)
    front = np.load(cdir / "lb_navtest@warp/front.npy", mmap_mode="r")
    V = np.asarray(front[:, -1], np.float32).mean(1)                              # newest context frame, mean of its 32 tokens
    Tt = np.load(cdir / "lb_navtest/teacher.npz")["out"].astype(np.float32)
    both = lambda x: np.stack([x, x])                                             # noqa: E731
    return dict(ego=both(tab["ego"][use].astype(np.float64)), plan=np.stack(pd_), dis=both(dis), V=both(pca_white(V, fit, use)),
                T=both(pca_white(Tt, fit, use))), dict(n_fit=len(fit), n_use=len(use))


def X_of(streams, k):
    return np.concatenate([_D[s][..., :k] if s in VIS else _D[s] for s in streams], -1)


def ridge_fit(X, Y):
    mu, sd, ym = X.mean(0), X.std(0), Y.mean(0)
    sd = np.where(sd < 1e-6, 1.0, sd)                                             # a column that is constant in the training fold stays unscaled
    Xc = (X - mu) / sd
    w, V = np.linalg.eigh(Xc.T @ Xc)
    A = V.T @ (Xc.T @ (Y - ym))
    return lambda Xq, lam: ((Xq - mu) / sd) @ (V @ (A / (np.maximum(w, 0)[:, None] + lam))) + ym


def picks_of(pred, m=0.0):
    """Predicted gains (.., K) -> the selected candidate (..): argmax with the identity (column 0) at 0; keep the identity unless the best > m."""
    p = pred.copy()
    p[..., 0] = 0.0
    pk = p.argmax(-1)
    pk[p.max(-1) <= m] = 0
    return pk


def take(Y, pk):
    return np.take_along_axis(Y, pk[..., None], -1)[..., 0]


def folds(rep, k=FOLDS, salt=0):
    lc = _D["lcode"]
    return (np.random.default_rng(salt + rep).permutation(lc.max() + 1) % k)[lc]


def select_fit(Xk, Y, tr, inner):
    """Inner CV over the tokens tr (folds by log): (k, lam) of the plain head and (k, lam, m) of the margin head by the realised gain (ties to
    the smaller k, larger lam, smaller m), then the fits on tr -> ((k, lam), (k, lam, m), {k: predictor})."""
    K = Y.shape[-1]
    acc = {k: np.zeros((len(LAMS), len(MS))) for k in Xk}
    for k, X in Xk.items():
        d = X.shape[-1]
        for g in range(INNER):
            a_, b_ = tr[inner[tr] != g], tr[inner[tr] == g]
            P = ridge_fit(X[:, a_].reshape(-1, d), Y[:, a_].reshape(-1, K))
            for li, lam in enumerate(LAMS):
                pred = P(X[:, b_].reshape(-1, d), lam).reshape(S, len(b_), K)
                for mi, m in enumerate(MS):
                    acc[k][li, mi] += take(Y[:, b_], picks_of(pred, m)).sum()
    bl, bm, al, am = -np.inf, -np.inf, None, None
    for k in Xk:
        for li in reversed(range(len(LAMS))):
            if acc[k][li, 0] > bl + 1e-9:
                bl, al = acc[k][li, 0], (k, LAMS[li])
            for mi in range(len(MS)):
                if acc[k][li, mi] > bm + 1e-9:
                    bm, am = acc[k][li, mi], (k, LAMS[li], MS[mi])
    return al, am, {k: ridge_fit(Xk[k][:, tr].reshape(-1, Xk[k].shape[-1]), Y[:, tr].reshape(-1, K)) for k in {al[0], am[0]}}


def unit(a):
    """One selector unit -> (tag, dict of per-token arrays). kinds: oof (one repeat), ins, const, perm (PERM_REPS repeats, inputs permuted over
    tokens), curve (a fraction of the training logs), hgb (boosted trees, one repeat). Gains are seed means under the family's convention."""
    kind, arm, fk, rep, opt = a
    Y, Yr, lc = _D["Y", fk], _D["Yraw", fk], _D["lcode"]
    n, K = Y.shape[1], Y.shape[2]
    tag = (kind, arm, fk, rep, opt)
    if kind == "const":
        d, f = np.zeros(n), folds(rep)
        for j in range(FOLDS):
            d[f == j] = Y[:, f == j, Y[:, f != j].mean((0, 1)).argmax()].mean(0)
        return tag, dict(d=d)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, cf = X_of(ARMS[arm], GK), _D["cand", fk][1:]                           # (S, n, d), (K - 1, 3)
        d, dr, pk, f = np.zeros(n), np.zeros(n), np.zeros((S, n), int), folds(rep)

        def rows(idx):
            x = np.repeat(X[:, idx].reshape(-1, X.shape[-1]), K - 1, 0)
            return np.concatenate([x, np.tile(cf, (S * len(idx), 1))], 1)
        for j in range(FOLDS):
            te, tr = np.flatnonzero(f == j), np.flatnonzero(f != j)
            g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
                                              early_stopping=False, random_state=0).fit(rows(tr), Y[:, tr, 1:].reshape(-1))
            p = picks_of(np.concatenate([np.zeros((S, len(te), 1)), g.predict(rows(te)).reshape(S, len(te), K - 1)], -1))
            d[te], dr[te], pk[:, te] = take(Y[:, te], p).mean(0), take(Yr[:, te], p).mean(0), p
        return tag, dict(d=d, draw=dr, picks=pk)
    Xk = {k: X_of(ARMS[arm], k) for k in (KS if any(s in VIS for s in ARMS[arm]) else KS[:1])}
    if kind == "perm":
        pi = np.random.default_rng(5000 + rep).permutation(n)
        Xk = {k: x[:, pi] for k, x in Xk.items()}
    if kind == "ins":
        al, _, P = select_fit(Xk, Y, np.arange(n), folds(0, INNER, 900))
        X = Xk[al[0]]
        return tag, dict(d=take(Y, picks_of(P[al[0]](X.reshape(-1, X.shape[-1]), al[1]).reshape(S, n, K))).mean(0), hp=[al])
    d, dr, dm, pk, hp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((S, n), int), []
    reps = range(PERM_REPS) if kind == "perm" else (rep,)
    for r in reps:
        f = folds(r)
        for j in range(FOLDS):
            te, tr = np.flatnonzero(f == j), np.flatnonzero(f != j)
            if kind == "curve":
                lg = np.unique(lc[tr])
                tr = tr[np.isin(lc[tr], np.random.default_rng(7000 + r * 10 + j).permutation(lg)[: max(INNER, int(round(opt * len(lg))))])]
            al, am, P = select_fit(Xk, Y, tr, folds(r * 10 + j, INNER, 1000))
            Xl, Xm = Xk[al[0]], Xk[am[0]]
            p = picks_of(P[al[0]](Xl[:, te].reshape(-1, Xl.shape[-1]), al[1]).reshape(S, len(te), K))
            pm = picks_of(P[am[0]](Xm[:, te].reshape(-1, Xm.shape[-1]), am[1]).reshape(S, len(te), K), am[2])
            d[te] += take(Y[:, te], p).mean(0) / len(reps)
            dr[te] += take(Yr[:, te], p).mean(0) / len(reps)
            dm[te] += take(Y[:, te], pm).mean(0) / len(reps)
            pk[:, te] = p
            hp.append((al, am))
    return tag, dict(d=d, draw=dr, dm=dm, picks=pk, hp=hp)


def fkey(f, cv):
    return f"{f} x {cv}"


def cmd_select(a):
    import multiprocessing as mp
    import pandas as pd
    from jevdrive import par
    from jevdrive.common import n_cpus
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_dewater/select", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        C, tok, dyaw, log, X = load(a.score)
        assert nt.mask(tok).all()
        F = fams(C)
        feats, finfo = features(tok)
        assert all(np.isfinite(v).all() for v in feats.values())
        _D.update(feats)
        _D["lcode"] = pd.factorize(log)[0]
        Sraw, ceil = TC.noec(X), {}
        for f, cv in MAIN:
            Sc = TC.noec(conv(X, C, cv))
            fk = fkey(f, cv)
            _D["Y", fk] = np.ascontiguousarray((Sc[:, F[f]] - Sc[:, :1]).transpose(0, 2, 1))          # (S, n, K) gain over the identity
            _D["Yraw", fk] = np.ascontiguousarray((Sraw[:, F[f]] - Sraw[:, :1]).transpose(0, 2, 1))
            _D["cand", fk] = np.array([C[i][1:] for i in F[f]], float)
            ceil[fk] = _D["Y", fk].max(-1).mean(0)
        fks = [fkey(*x) for x in MAIN]
        units = [("oof", arm, fk, r, 1.0) for arm in ARMS for fk in fks for r in range(REPS)]
        units += [("ins", arm, fk, 0, 1.0) for arm in ARMS for fk in fks] + [("const", "", fk, r, 1.0) for fk in fks for r in range(REPS)]
        units += [("perm", "E", fk, r, 1.0) for fk in (fks[0], fks[-1]) for r in range(N_PERM)]
        units += [("curve", "E", fk, r, fr) for fk in fks for fr in CURVE for r in range(REPS)]
        units += [("hgb", arm, fk, r, 1.0) for arm in ("E", "V+E") for fk in fks for r in range(REPS)]
        if a.smoke:
            units = [u for u in units if u[3] == 0 and u[2] == fks[-1]]
        workers = max(1, min(n_cpus() // 2, a.workers))
        run.info("%d units on %d workers; streams %s; PCA fitted on %d non-turn navtest tokens", len(units), workers,
                 {k: list(v.shape) for k, v in feats.items()}, finfo["n_fit"])
        res = par.pmap(unit, units, workers=workers, run=run, desc="selector units", mp_context=mp.get_context("fork"))
        res.raise_if_failed()
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / ("select_smoke.pkl" if a.smoke else "select.pkl"), "wb") as fh:
            pickle.dump(dict(res=dict(res.values), tok=tok, dyaw=dyaw, log=log, ceil=ceil, fks=fks, finfo=finfo,
                             cand={fk: _D["cand", fk] for fk in fks}), fh)
        run.summary.update(n_units=len(units), workers=workers)


def cmd_selreport(a):
    from jevdrive import stats
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_dewater/selreport", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        out, fig = _pl.Path(a.out), _pl.Path(a.figs)
        out.mkdir(parents=True, exist_ok=True)
        with open(OUT / "select.pkl", "rb") as fh:
            Z = pickle.load(fh)
        R, log, dyaw, ceil, fks = Z["res"], Z["log"], Z["dyaw"], Z["ceil"], Z["fks"]
        B = {k: v for k, v in buckets(dyaw).items() if "left" not in k and "right" not in k}
        zero = np.zeros(len(log))

        def mean_of(kind, arm, fk, key="d", opt=1.0, reps=range(REPS)):
            return np.mean([R[kind, arm, fk, r, opt][key] for r in reps], 0)

        def g(x, m):
            return TC.ci(x[m], zero[m], log[m])
        rows, GN, PT, verdict = [], {}, {}, {}
        for fk in fks:
            cl = g(ceil[fk], B["> 20 deg"])
            null = np.array([R["perm", "E", fk, r, 1.0]["d"].mean() for r in range(N_PERM)]) * 100 if ("perm", "E", fk, 0, 1.0) in R else None
            arms = [("L", arm, mean_of("oof", arm, fk), mean_of("oof", arm, fk, "draw"), R["ins", arm, fk, 0, 1.0]["d"]) for arm in ARMS]
            arms += [("L+m", arm, mean_of("oof", arm, fk, "dm"), None, None) for arm in ARMS]
            arms += [("G", arm, mean_of("hgb", arm, fk), mean_of("hgb", arm, fk, "draw"), None) for arm in ("E", "V+E")]
            arms += [("const", "best fixed candidate in fold", mean_of("const", "", fk), None, None)]
            for head, arm, d, dr, ins in arms:
                PT[fk, head, arm] = d
                r = {"family x convention": fk, "head": head, "arm": arm}
                for b, m in B.items():
                    GN[fk, head, arm, b] = g(d, m)
                    r[b] = TC.cell(GN[fk, head, arm, b])
                q = TC.ratio_ci(d, ceil[fk], log)
                r["recovery of the matching ceiling"] = f"{100 * q['mean']:.1f}% [{100 * q['lo']:.1f}, {100 * q['hi']:.1f}]"
                r["ceiling"] = f"{cl['mean']:+.2f}"
                r["valued raw, > 20 deg"] = TC.cell(g(dr, B["> 20 deg"])) if dr is not None else ""
                r["in-sample"] = f"{100 * ins.mean():+.2f}" if ins is not None else ""
                if head in ("L", "G"):
                    pk = np.stack([R["oof" if head == "L" else "hgb", arm, fk, rr, 1.0]["picks"] for rr in range(REPS)])
                    r["moved %"] = f"{100 * (pk != 0).mean():.1f}"
                else:
                    r["moved %"] = ""
                r["permutation p"] = ""
                if head == "L" and arm == "E" and null is not None:
                    act = 100 * mean_of("oof", arm, fk, reps=range(PERM_REPS)).mean()
                    pv = (1 + (null >= act).sum()) / (1 + len(null))
                    r["permutation p"] = f"{pv:.3f} (null {null.mean():+.2f}, 95% {np.quantile(null, 0.025):+.2f} .. {np.quantile(null, 0.975):+.2f}; actual {act:+.2f})"
                    gn = GN[fk, head, arm, "> 20 deg"]
                    verdict[fk] = dict(gain=gn, recovery=q, ceiling=cl, perm_p=float(pv), null_mean=float(null.mean()),
                                       reading="learnable" if gn["lo"] > 0 and pv < 0.05 and q["mean"] >= WOD_RECOVERY else
                                       "weakly learnable" if gn["lo"] > 0 else "not learnable at this label scale with these inputs")
                rows.append(r)
        stats.write_table(rows, out / "selector", note="out-of-fold gain over SH30 of the selected candidate under the family's convention, EPDMS x 100, seed mean, "
                          f"mean over {REPS} repeats of {FOLDS} folds by log; paired log-cluster bootstrap. L = ridge, L+m = ridge with a margin, G = boosted trees")
        # arm contrasts and the moved-vs-oracle reading
        rows = []
        for fk in fks:
            for name, k1, k2 in [("E - E0 (plan descriptor)", ("L", "E"), ("L", "E0")), ("E+D - E (seed disagreement)", ("L", "E+D"), ("L", "E")),
                                 ("T+E - E (shipped teacher outputs)", ("L", "T+E"), ("L", "E")), ("V+E - E (frozen vision)", ("L", "V+E"), ("L", "E")),
                                 ("V+T+E - E", ("L", "V+T+E"), ("L", "E")), ("G - L on E (trees)", ("G", "E"), ("L", "E")), ("G V+E - G E", ("G", "V+E"), ("G", "E")),
                                 ("L+m - L on E (margin)", ("L+m", "E"), ("L", "E")), ("L on E - best fixed candidate", ("L", "E"), ("const", "best fixed candidate in fold"))]:
                r = {"family x convention": fk, "contrast": name}
                for b, m in B.items():
                    r[b] = TC.cell(TC.ci(PT[(fk, *k1)][m], PT[(fk, *k2)][m], log[m]))
                rows.append(r)
        stats.write_table(rows, out / "selector_contrasts", note="paired differences of out-of-fold gains (same folds), EPDMS x 100")
        rows, CV = [], {}
        for fk in fks:
            for fr in (*CURVE, 1.0):
                d = mean_of("curve", "E", fk, opt=fr) if fr < 1 else mean_of("oof", "E", fk)
                CV[fk, fr] = g(d, B["> 20 deg"])
                rows.append({"family x convention": fk, "share of the training logs": fr, "> 20 deg": TC.cell(CV[fk, fr]),
                             "20-45 deg": TC.cell(g(d, B["20-45 deg"])), "> 45 deg": TC.cell(g(d, B["> 45 deg"]))})
        stats.write_table(rows, out / "learning_curve", floatfmt=".2f", note="ridge head on E, out of fold, trained on a random share of the training logs")
        rows = []
        for fk in fks:
            cand = Z["cand"][fk]
            for head, kind in (("L", "oof"), ("G", "hgb")):
                pk = np.stack([R[kind, "E", fk, rr, 1.0]["picks"] for rr in range(REPS)])              # (reps, S, n) family-local index
                mv = cand[pk]
                rows.append({"family x convention": fk, "head": head, "identity %": 100 * (pk == 0).mean(), "speed < 1 %": 100 * (mv[..., 2] < 1).mean(),
                             "speed > 1 %": 100 * (mv[..., 2] > 1).mean(), "offset moved %": 100 * (mv[..., 0] != 0).mean(),
                             "curvature moved %": 100 * (mv[..., 1] != 1).mean(), "distinct candidates used": int(len(np.unique(pk)))})
        stats.write_table(rows, out / "selector_picks", floatfmt=".1f", note="what the E-arm selectors pick out of fold, % of (repeat, seed, token)")
        hp = {fk: [list(map(float, x[0])) for r in range(REPS) for x in R["oof", "E", fk, r, 1.0]["hp"]] for fk in fks}
        (out / "selector_verdict.json").write_text(json.dumps(dict(verdict=verdict, wod_recovery=WOD_RECOVERY, finfo=Z["finfo"],
                                                                   lam_median={fk: float(np.median([h[1] for h in v])) for fk, v in hp.items()}), indent=1, default=float))
        run.summary.update({f"gain {fk}": v["gain"]["mean"] for fk, v in verdict.items()})
        run.info("selector verdict: %s", json.dumps(verdict, default=float))
        fig_select(fig, GN, CV, ceil, fks, log)


def fig_select(FIGD, GN, CV, ceil, fks, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIGD.mkdir(parents=True, exist_ok=True)
    b = "> 20 deg"
    arms = [("L", x) for x in ARMS] + [("G", "E"), ("G", "V+E"), ("const", "best fixed candidate in fold")]
    labs = [f"{x}" for x in ARMS] + ["trees\nE", "trees\nV+E", "best\nfixed"]
    cols = [ps.PALETTE["blue"], ps.PALETTE["sky_blue"], ps.PALETTE["vermillion"], ps.PALETTE["orange"]]
    fig, axs = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
    w = 0.8 / len(fks)
    for j, fk in enumerate(fks):
        v = np.array([[GN[fk, h, a, b]["mean"], GN[fk, h, a, b]["mean"] - GN[fk, h, a, b]["lo"], GN[fk, h, a, b]["hi"] - GN[fk, h, a, b]["mean"]] for h, a in arms])
        axs[0].bar(np.arange(len(arms)) + (j - (len(fks) - 1) / 2) * w, v[:, 0], w, yerr=v[:, 1:].T, color=cols[j],
                   label=f"{fk} (ceiling {100 * ceil[fk].mean():+.1f})", error_kw=dict(lw=0.6, capsize=1.2))
    axs[0].set_xticks(range(len(arms)), labs, fontsize=7)
    axs[0].set_ylabel("selector - SH30, out of fold, EPDMS x 100")
    axs[0].legend(fontsize=6), ps.bars(axs[0]), ps.zero_line(axs[0])
    xs = (*CURVE, 1.0)
    for j, fk in enumerate(fks):
        v = np.array([[CV[fk, x]["mean"], CV[fk, x]["lo"], CV[fk, x]["hi"]] for x in xs])
        axs[1].plot(xs, v[:, 0], color=cols[j], marker="o", ms=2.5)
        axs[1].fill_between(xs, v[:, 1], v[:, 2], color=cols[j], alpha=0.12, lw=0)
    axs[1].set_xticks(xs, [f"{int(100 * x)}%" for x in xs])
    axs[1].set_xlabel("share of the training logs (ridge, E)"), ps.zero_line(axs[1])
    fig.savefig(FIGD / "selector.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ceiling", "select", "selreport"):
        p = sp.add_parser(name)
        p.add_argument("--score", default=str(SCORE))
        p.add_argument("--out", default=str(RES))
        p.add_argument("--figs", default=str(FIG))
        p.add_argument("--workers", type=int, default=36)
        p.add_argument("--smoke", action="store_true", help="select: one repeat on the trusted family only, written to select_smoke.pkl (not reported)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
