"""op_parity turn-ceiling (plans/2026-10-08-turn-ceiling-prereg.md): privileged best-of-K ceiling of a small trajectory family around SH30's
own plan on the navtest > 20 deg tokens (3 154). Three degrees of freedom: lateral offset, curvature gain, speed scaling. CPU only, no training.

  family    (.venv)     candidate poses of SH30-F-s0 / s1 on the > 20 deg tokens -> poses.npz (keys s<seed>_c<idx>), family.json, token lists
                        (t24 / t300 / all)
  gate      (.venv)     stage-0 gate: reads ONLY the identity keys of a score-poses CSV and compares them with SH30's archived per-token
                        sub-scores; --stage t300 also measures the cost and fixes the stage-1 key set by the pre-registered cut ladder
  report    (op-train)  ceilings per bucket and family, degree-of-freedom split, small-K readings, oracle picks, sub-score (Shapley) split
                        of the SH30 vs WA-JEPA gap and of the oracle gain -> results/turn_ceiling/, figs/turn_ceiling/
  navtrain  (.venv)     read-only: navtrain tokens with |dyaw| > 20 deg, which metric caches exist
Scoring itself is `python -m jevdrive.bench score-poses` (turn_ceiling_chain.sh). The parallel loop of this lane is score-poses' own, so
jevdrive.par is not used here; the poses file goes through jevdrive.cache (its bytes are part of the score-poses run identity).
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R), str(_R / "research")]
import argparse, itertools, json, os, zlib  # noqa: E401,E402

import numpy as np  # noqa: E402

D = _pl.Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/turn_ceiling"
RES = _R / "experiments/op_parity/results/turn_ceiling"
FIG = _R / "experiments/op_parity/figs/turn_ceiling"
SEEDS = (0, 1)
MODEL = "SH30-F-s{}"
REF = "WA-JEPA"
ID = (0.0, 1.0, 1.0)                                    # (lateral offset m, left +; curvature gain; speed scale)
INNER = ((-0.5, 0.5), (0.85, 1.15), (0.8, 1.2))
OUTER = ((-1.0, 1.0), (0.7, 1.4), (0.6, 1.4))
S_RAMP = 6.0                                            # m of travel over which the lateral offset is blended in (smoothstep)
KAPPA_MAX = 0.3                                         # 1 / m, cap of the curvature used to extend a path beyond its 4 s end
LINE = 4.0                                              # pre-registered stop line: F19 ceiling on > 20 deg, EPDMS x 100
EP_TOL = 1e-6
WALL_OK_H, WALL_MAX_H = 1.5, 3.0
NB_RATIO = 4000


# ---------------------------------------------------------------- the family
def candidates():
    """[(name, offset, gain, speed)]: c00 identity, c01-c06 one inner axis moved, c07-c18 two, c19-c26 three (the inner 3 x 3 x 3
    product ordered by the number of moved axes), c27-c32 the outer single-axis points."""
    lv = [(i, *x) for i, x in zip(ID, INNER)]
    prod = sorted(itertools.product(*lv), key=lambda c: sum(x != i for x, i in zip(c, ID)))
    outer = [tuple(x if j == ax else ID[j] for j in range(3)) for ax in range(3) for x in OUTER[ax]]
    return [(f"c{i:02d}", *c) for i, c in enumerate(prod + outer)]


def families(C=None):
    """name -> candidate indices (always with the identity)."""
    C = C or candidates()
    moved = np.array([[x != i for x, i in zip(c[1:], ID)] for c in C])
    n, inner = moved.sum(1), np.arange(len(C)) < 27
    ax = {k: np.flatnonzero((n <= 1) & (moved[:, j] | (n == 0))) for j, k in enumerate("OKV")}
    F = {f"{k}3": [i for i in v if inner[i]] for k, v in ax.items()}
    F |= {f"{k}5": list(v) for k, v in ax.items()}
    F |= {"F7": list(np.flatnonzero(inner & (n <= 1))), "F13": list(np.flatnonzero(n <= 1)), "F19": list(np.flatnonzero(inner & (n <= 2))),
          "F27": list(np.flatnonzero(inner)), "F33": list(range(len(C)))}
    return {k: [int(i) for i in v] for k, v in F.items()}


def _seg(Q):
    d = np.diff(Q[..., :2], axis=1)
    return np.hypot(d[..., 0], d[..., 1]), np.arctan2(d[..., 1], d[..., 0])


def curv(Q, k):
    """Curvature gain: every heading (pose yaw and chord direction, relative to the t0 heading) x k, segment lengths kept, so the
    speed profile is unchanged. The chord's slip against the mean pose yaw of its ends is kept as is."""
    seg, phi = _seg(Q)
    yb = (Q[:, :-1, 2] + Q[:, 1:, 2]) / 2
    ph = k * yb + np.angle(np.exp(1j * (phi - yb)))
    out = np.zeros_like(Q)
    out[:, 1:, :2] = np.cumsum(seg[..., None] * np.stack([np.cos(ph), np.sin(ph)], -1), 1)
    out[..., 2] = k * Q[..., 2]
    return out


def offset(Q, o):
    """Lateral offset o (m, left +) along the pose normal, blended in by a smoothstep over the first S_RAMP m of travel (no offset
    at standstill); the yaw follows the tangent of the shifted path."""
    seg, _ = _seg(Q)
    u = np.clip(np.concatenate([np.zeros((len(Q), 1)), np.cumsum(seg, 1)], 1) / S_RAMP, 0, 1)
    r, dr = u * u * (3 - 2 * u), 6 * u * (1 - u) / S_RAMP
    out = Q.copy()
    out[..., 0] -= o * r * np.sin(Q[..., 2])
    out[..., 1] += o * r * np.cos(Q[..., 2])
    out[..., 2] += np.arctan(o * dr)
    return out


def speed(Q, v):
    """Speed scaling: the same path, travelled v x as far at every time (arc length x v). Beyond the 4 s end the path continues
    as an arc with the curvature of its last segment (|kappa| <= KAPPA_MAX)."""
    out = Q.copy()
    for n, q in enumerate(Q):
        seg = np.hypot(*np.diff(q[:, :2], axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 1e-3:
            continue
        tgt, sm = v * s[1:], s + 1e-9 * np.arange(9)
        for j in range(3):
            out[n, 1:, j] = np.interp(tgt, sm, q[:, j])
        over = tgt > s[-1]
        if over.any():
            u, th0 = tgt[over] - s[-1], q[-1, 2]
            kap = float(np.clip((q[-1, 2] - q[-2, 2]) / max(seg[-1], 0.5), -KAPPA_MAX, KAPPA_MAX))
            th = th0 + kap * u
            if abs(kap) < 1e-6:
                x, y = q[-1, 0] + u * np.cos(th0), q[-1, 1] + u * np.sin(th0)
            else:
                x, y = q[-1, 0] + (np.sin(th) - np.sin(th0)) / kap, q[-1, 1] - (np.cos(th) - np.cos(th0)) / kap
            out[n, 1:][over] = np.stack([x, y, th], -1)
    return out


def transform(P, o=0.0, k=1.0, v=1.0):
    """P (N, 8, 3) rear-axle poses at 0.5 .. 4 s -> the candidate (N, 8, 3). Order: curvature gain, lateral offset, speed scaling;
    an axis at its identity value is skipped, and the identity candidate is the input array itself."""
    if (o, k, v) == ID:
        return P
    Q = np.concatenate([np.zeros((len(P), 1, 3)), np.asarray(P, np.float64)], 1)
    Q[..., 2] = np.unwrap(Q[..., 2], axis=1)
    if k != 1.0:
        Q = curv(Q, k)
    if o != 0.0:
        Q = offset(Q, o)
    if v != 1.0:
        Q = speed(Q, v)
    Q[..., 2] = np.angle(np.exp(1j * Q[..., 2]))
    return Q[:, 1:].astype(P.dtype)


def key(seed, i):
    return f"s{seed}_c{i:02d}"


def bucket_tokens():
    """(tokens, dyaw deg) of the navtest tokens with logged 4 s |heading change| >= 20 deg (bench strata bins 20-45 and > 45), sorted."""
    from jevdrive.bench import tables as T
    S = T.navtest_strata()
    S = S[S.turn.isin(["20-45", ">45"])].sort_index()
    return S.index.to_numpy(str), S.dyaw.to_numpy(float)


def cmd_family(a):
    from jevdrive import cache
    from jevdrive.bench import compat
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_ceiling/family", seed=0, config=vars(a)) as run:
        nt = splits.load("navsim/navtest")
        run.use_split(nt)
        tok, dyaw = bucket_tokens()
        assert nt.mask(tok).all(), "bucket tokens outside navsim/navtest"
        C = candidates()
        preds = [compat.pred_file(MODEL.format(s)) for s in SEEDS]

        def build():
            out = {"tokens": tok}
            for s, p in zip(SEEDS, preds):
                z = np.load(p)
                row = {t: i for i, t in enumerate(z["tokens"].tolist())}
                P = z["poses"][[row[t] for t in tok]]
                for i, (_, o, k, v) in enumerate(C):
                    out[key(s, i)] = transform(P, o, k, v)
            return out
        k = cache.key(dict(cands=C, ramp=S_RAMP, kmax=KAPPA_MAX, tokens=zlib.crc32("".join(tok).encode())), inputs=preds,
                      code=[transform, curv, offset, speed], version=nt.id)
        Z = cache.cached(OUT / "poses.npz", k, build, force=a.force)
        for s in SEEDS:                                                       # the identity is the archived prediction, bit for bit
            z = np.load(compat.pred_file(MODEL.format(s)))
            row = {t: i for i, t in enumerate(z["tokens"].tolist())}
            assert np.array_equal(Z[key(s, 0)], z["poses"][[row[t] for t in tok]])
        big = np.abs(dyaw) >= 45
        pick = np.concatenate([run.rng.permutation(np.flatnonzero(m))[:a.n_smoke // 2] for m in (~big, big)])
        pick = run.rng.permutation(pick)
        for name, t in (("t24", tok[pick[:24]]), ("t300", tok[pick]), ("all", tok)):
            (OUT / f"tokens_{name}.txt").write_text("\n".join(t) + "\n")
        (OUT / "keys_F33.txt").write_text(" ".join(key(s, i) for s in SEEDS for i in range(len(C))) + "\n")
        (OUT / "family.json").write_text(json.dumps(dict(candidates=[dict(name=n, offset=o, gain=k, speed=v) for n, o, k, v in C],
                                                         families=families(C), s_ramp=S_RAMP, kappa_max=KAPPA_MAX, seeds=SEEDS,
                                                         n_tokens=len(tok), n_20_45=int((~big).sum()), n_45=int(big.sum())), indent=1))
        disp = {n: float(np.abs(Z[key(0, i)][:, -1, :2].astype(float) - Z[key(0, 0)][:, -1, :2]).max()) for i, (n, *_) in enumerate(C)}
        run.summary.update(n_tokens=len(tok), n_candidates=len(C), max_end_shift_m=max(disp.values()))
        run.info("%d tokens x %d candidates x %d seeds -> %s", len(tok), len(C), len(SEEDS), OUT / "poses.npz")


# ---------------------------------------------------------------- stage-0 gate
SUB8 = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]


def archived(spec, tok):
    """(X (n, 9) sub-scores in bench TERMS order, logs) of a stored navtest run on `tok`."""
    from jevdrive.bench import tables as T
    u, src = T.load("navtest", spec)
    assert u is not None, f"no navtest units for {spec}"
    return u.loc[tok, T.TERMS].to_numpy(float), u.loc[tok, "log"].astype(str).to_numpy(), src


def noec(X):
    """Per-token EPDMS without extended comfort (score-poses' `score`) from (.., 9) sub-scores."""
    from jevdrive.bench import tables as T
    Y = np.array(X, float)
    Y[..., 8] = np.nan
    return T.score_of(Y)


def cmd_gate(a):
    import pandas as pd
    from jevdrive.bench import poses as BP
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    with Run("op_parity", f"turn_ceiling/gate-{a.stage}", config=vars(a)) as run:
        df = pd.read_csv(a.score)
        num = df[[SUBS[k] for k in SUB8] + ["score"]].to_numpy(float)
        sane = bool(np.isfinite(num).all() and (num >= 0).all() and (num <= 1 + 1e-9).all())    # ranges only, no value is reported
        tok = BP.read_tokens(OUT / f"tokens_{a.stage}.txt")
        ids = df[df.key.isin([key(s, 0) for s in SEEDS])]                     # from here on only the identity rows are read
        del df, num
        res = dict(stage=a.stage, n_tokens=len(tok), sane_ranges=sane, seeds={})
        ok = sane
        for s in SEEDS:
            g = ids[ids.key == key(s, 0)].set_index("token").loc[tok]
            X, _, src = archived(MODEL.format(s), tok)
            r = dict(source=src)
            for j, kname in enumerate(SUB8):
                dlt = np.abs(g[SUBS[kname]].to_numpy(float) - X[:, j])
                r[kname] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > (EP_TOL if kname == "EP" else 0)).sum()))
            dlt = np.abs(g.score.to_numpy(float) - noec(X))
            r["score_noec"] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > EP_TOL).sum()))
            r["pass"] = all(v["n_diff"] == 0 for v in r.values() if isinstance(v, dict))
            ok &= r["pass"]
            res["seeds"][s] = r
        res["pass"] = bool(ok)
        if a.stage == "t300":
            from jevdrive.bench import runner as BR
            rd = BR.bench_root("poses") / BP.run_key(str(OUT / "poses.npz"), (OUT / "keys_F33.txt").read_text().split(), tok)
            sm = json.loads((rd / "summary.json").read_text())
            c, nk, n_all = sm["cost"], len(sm["keys"]), len(BP.read_tokens(OUT / "tokens_all.txt"))
            cores = int(BP.pool_budget() // BP.CORES_PER_JOB) * BP.CORES_PER_JOB
            fam, nc = families(), len(candidates())
            ladder = [("F33, 2 seeds", [key(s, i) for s in SEEDS for i in range(nc)]),
                      ("F27, 2 seeds", [key(s, i) for s in SEEDS for i in fam["F27"]]),
                      ("F33, seed 0", [key(0, i) for i in range(nc)]), ("F27, seed 0", [key(0, i) for i in fam["F27"]])]
            opts = []
            for name, ks in ladder:
                cs = c["load_s_per_token"] + c["union_s_per_token"] + c["diag_s_per_token"] * len(ks) / nk + c["pdm_s_per_token"] * (len(ks) + 1) / (nk + 1)
                opts.append(dict(option=name, n_keys=len(ks), core_s_per_token=cs, core_h=cs * n_all / 3600, wall_h=cs * n_all / cores / 3600 + 0.05))
            pickd = next((i for i, o in enumerate(opts) if o["wall_h"] <= WALL_OK_H), None)
            if pickd is None:
                pickd = next((i for i in reversed(range(len(opts))) if opts[i]["wall_h"] <= WALL_MAX_H), None)
            res["cost"] = dict(measured=c, n_keys=nk, core_s_per_token=sum(c[f"{x}_s_per_token"] for x in ("load", "pdm", "diag", "union")),
                               core_s_per_key=sum(c[f"{x}_s_per_token"] for x in ("load", "pdm", "diag", "union")) / nk,
                               wall_s_first_to_last=sm["wall_s_first_to_last"], run_dir=str(rd), pool_cores=cores, options=opts,
                               chosen=None if pickd is None else opts[pickd]["option"])
            if pickd is None:
                ok = False
                res["pass"] = False
                res["stop"] = f"no option of the cut ladder fits {WALL_MAX_H} h"
            else:
                (OUT / "stage1_keys.txt").write_text(" ".join(ladder[pickd][1]) + "\n")
        (OUT / f"gate-{a.stage}.json").write_text(json.dumps(res, indent=1, default=float))
        run.summary.update(gate=res["pass"], stage=a.stage)
        run.info("gate %s: %s", a.stage, json.dumps(res, default=float))
        if not ok:
            raise SystemExit(f"stage-0 gate {a.stage} FAILED: see {OUT / f'gate-{a.stage}.json'}")


# ---------------------------------------------------------------- report
def ci(x, y, g, scale=100.0):
    from jevdrive import stats
    return stats.paired(np.asarray(x) * scale, np.asarray(y) * scale, groups=g)


def cell(r, f="+.2f"):
    return f"{r['mean']:{f}} [{r['lo']:{f}}, {r['hi']:{f}}]"


def ratio_ci(num, den, g, n_boot=NB_RATIO):
    """Cluster bootstrap over logs of sum(num) / sum(den)."""
    import pandas as pd
    codes, uniq = pd.factorize(g)
    a, b = np.bincount(codes, num, len(uniq)), np.bincount(codes, den, len(uniq))
    idx = np.random.default_rng(0).integers(len(uniq), size=(n_boot, len(uniq)))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = a[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.nanquantile(q, [0.025, 0.975])
    return dict(mean=float(num.sum() / den.sum()) if den.sum() else np.nan, lo=float(lo), hi=float(hi))


def greedy(S, m, pool, kmax):
    """Forward selection on tokens `m`: start from the identity, add the candidate that raises mean(max over the set) most.
    S (seeds, C, N). Returns the ordered candidate list."""
    cur, chosen = S[:, 0][:, m].copy(), [0]
    rest = [c for c in pool if c != 0]
    while len(chosen) < kmax and rest:
        gain = [np.maximum(cur, S[:, c][:, m]).mean() for c in rest]
        c = rest.pop(int(np.argmax(gain)))
        cur = np.maximum(cur, S[:, c][:, m])
        chosen.append(c)
    return chosen


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench import tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_ceiling/report", seed=0, config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        out = _pl.Path(a.out) if a.out else RES
        out.mkdir(parents=True, exist_ok=True)
        C, tok0 = candidates(), bucket_tokens()
        tok, dyaw = tok0
        keys = (OUT / "stage1_keys.txt").read_text().split()
        seeds = sorted({int(k[1]) for k in keys})
        cidx = sorted({int(k.split("_c")[1]) for k in keys})
        df = pd.read_csv(a.score)
        assert set(df.key) == set(keys) and set(df.token) == set(tok), "score CSV does not match stage1_keys / the token set"
        nS, nC, N = len(seeds), len(C), len(tok)
        X = np.full((nS, nC, N, 9), np.nan)                                       # sub-scores, bench TERMS order, EC = NaN
        raw = np.full((nS, nC, N), np.nan)
        ti = {t: i for i, t in enumerate(tok)}
        for kname, g in df.groupby("key"):
            s, c = seeds.index(int(kname[1])), int(kname.split("_c")[1])
            r = g.token.map(ti).to_numpy()
            X[s, c, r, :8] = g[[SUBS[k] for k in SUB8]].to_numpy(float)
            raw[s, c, r] = g.raw_out.astype(float).to_numpy()
        S = noec(X)                                                              # (seeds, C, N)
        A = [archived(MODEL.format(s), tok) for s in seeds]                      # archived SH30 with EC
        log = A[0][1]
        XA = np.stack([x for x, _, _ in A])
        XW, _, wsrc = archived(REF, tok)
        chk = {s: float(np.abs(S[i, 0] - noec(XA[i])).max()) for i, s in enumerate(seeds)}
        fam = {k: [c for c in v if c in cidx] for k, v in families(C).items()}
        fam = {k: v for k, v in fam.items() if len(v) == len(families(C)[k])}
        B = {"> 20 deg": np.ones(N, bool), "20-45 deg": np.abs(dyaw) < 45, "> 45 deg": np.abs(dyaw) >= 45, "left > 20 deg": dyaw > 0,
             "right > 20 deg": dyaw < 0, "left > 45 deg": dyaw >= 45, "right > 45 deg": dyaw <= -45}
        idn, wa, wa_ec = S[:, 0].mean(0), noec(XW), T.score_of(XW)
        sh_ec = np.mean([T.score_of(x) for x in XA], 0)

        def best(f):
            return S[:, fam[f]].max(1).mean(0)

        def pick(f):                                                             # (seeds, N) candidate index; ties -> the earliest candidate
            return np.asarray(fam[f])[S[:, fam[f]].argmax(1)]
        # 1. levels and ceilings
        L, rows = [], []
        for b, m in B.items():
            r = dict(bucket=b, n=int(m.sum()), SH30=100 * idn[m].mean(), **{"WA-JEPA": 100 * wa[m].mean()},
                     **{"WA - SH30": cell(ci(wa[m], idn[m], log[m])), "SH30 (with EC, archived)": 100 * sh_ec[m].mean(),
                        "WA-JEPA (with EC, archived)": 100 * wa_ec[m].mean(), "WA - SH30 (with EC)": cell(ci(wa_ec[m], sh_ec[m], log[m])),
                        "headroom to 100": 100 * (1 - idn[m]).mean()})
            rows.append(r)
        lev = stats.write_table(rows, out / "levels", floatfmt=".2f", note="EPDMS x 100; columns without a note are EPDMS without extended comfort "
                                "(score-poses), SH30 = identity candidate, seed mean; CIs: paired log-cluster bootstrap")
        order = [f for f in ("O3", "K3", "V3", "F7", "F19", "F27", "O5", "K5", "V5", "F13", "F33") if f in fam]
        rows, G = [], {}
        for f in order:
            r = dict(family=f, K=len(fam[f]))
            for b, m in B.items():
                G[f, b] = ci(best(f)[m], idn[m], log[m])
                r[b] = cell(G[f, b])
            rows.append(r)
        stats.write_table(rows, out / "ceiling", note="best-of-K minus the identity (SH30), EPDMS x 100 without EC, seed mean of the per-seed oracle; "
                          "paired log-cluster bootstrap, B 10000")
        # 2. degree-of-freedom split
        rows = []
        cons = [("joint excess: F27 - F7 (needs >= 2 axes at once)", "F27", "F7"), ("three-way excess: F27 - F19", "F27", "F19"),
                ("pairs excess: F19 - F7", "F19", "F7"), ("offset - curvature (O3 - K3)", "O3", "K3"), ("offset - speed (O3 - V3)", "O3", "V3"),
                ("curvature - speed (K3 - V3)", "K3", "V3"), ("outer points: F33 - F27", "F33", "F27")]
        dof = {}
        for name, f1, f2 in cons:
            if f1 not in fam or f2 not in fam:
                continue
            r = dict(contrast=name)
            for b, m in B.items():
                dof[name, b] = ci(best(f1)[m], best(f2)[m], log[m])
                r[b] = cell(dof[name, b])
            rows.append(r)
        stats.write_table(rows, out / "dof", note="differences of best-of-K ceilings, EPDMS x 100 without EC; paired log-cluster bootstrap")
        # 3. oracle picks
        rows = []
        for f in [x for x in ("F19", "F27", "F33") if x in fam]:
            p = pick(f)
            for c in fam[f]:
                r = dict(family=f, cand=C[c][0], offset=C[c][1], gain=C[c][2], speed=C[c][3])
                for b, m in B.items():
                    r[b] = 100 * (p[:, m] == c).mean()
                rows.append(r)
        picks = stats.write_table(rows, out / "picks", floatfmt=".1f", note="% of (token, seed) on which the oracle picks the candidate; ties go to the "
                                  "earliest candidate (identity first, then fewer moved axes)")
        sgn = np.sign(dyaw)
        rows, marg = [], {}
        fm = "F27" if "F27" in fam else "F19"
        p = pick(fm)
        P = np.array([c[1:] for c in C])[p]                                       # (seeds, N, 3)
        lat = np.sign(P[..., 0]) * sgn                                            # + = toward the inside of the logged turn
        ev = {"offset toward inside": lat > 0, "offset none": lat == 0, "offset toward outside": lat < 0,
              "curvature gain < 1": P[..., 1] < 1, "curvature gain = 1": P[..., 1] == 1, "curvature gain > 1": P[..., 1] > 1,
              "speed < 1": P[..., 2] < 1, "speed = 1": P[..., 2] == 1, "speed > 1": P[..., 2] > 1, "identity": p == 0}
        gainf = S[:, fam[fm]].max(1) - S[:, 0]                                    # (seeds, N)
        for name, e in ev.items():
            r = dict(event=name)
            for b, m in B.items():
                marg[name, b] = 100 * e[:, m].mean()
                r[b] = marg[name, b]
                r[f"{b}: share of gain %"] = 100 * (gainf[:, m] * e[:, m]).sum() / max(gainf[:, m].sum(), 1e-12)
            rows.append(r)
        stats.write_table(rows, out / "picks_marginal", floatfmt=".1f", note=f"oracle on {fm}: % of (token, seed) picks, and the share of the oracle "
                          "gain that those picks carry; inside / outside by the sign of the logged heading change")
        # 4. small K: greedy sets, in sample and cross-fitted over logs; the best single fixed candidate
        fold = np.array([zlib.crc32(x.encode()) % 2 for x in log])
        pool, KS = fam["F33"] if "F33" in fam else fam["F27"], (2, 3, 4, 6, 8, 12, 20)
        gi = greedy(S, B["> 20 deg"], pool, max(KS))
        gx = [greedy(S, fold != h, pool, max(KS)) for h in (0, 1)]
        rows, SK = [], {}
        for K in KS:
            ins = S[:, gi[:K]].max(1).mean(0)
            xf = np.where(fold == 0, S[:, gx[0][:K]].max(1).mean(0), S[:, gx[1][:K]].max(1).mean(0))
            r = dict(K=K, added=C[gi[K - 1]][0], **{"added (offset, gain, speed)": str(C[gi[K - 1]][1:])})
            for b, m in B.items():
                SK["in", K, b], SK["x", K, b] = ci(ins[m], idn[m], log[m]), ci(xf[m], idn[m], log[m])
                r[f"{b}: in-sample"], r[f"{b}: cross-fit"] = cell(SK["in", K, b]), cell(SK["x", K, b])
            rows.append(r)
        stats.write_table(rows, out / "small_k", note="greedy forward selection of a K-set (identity included) on the > 20 deg tokens, oracle within the "
                          "set; cross-fit = set chosen on the other half of the logs (crc32 of the log name, 2 folds)")
        rows, single = [], {}
        sm = S.mean(0)                                                            # (C, N) seed mean per fixed candidate
        cf = [int(np.asarray(pool)[sm[pool][:, fold != h].mean(1).argmax()]) for h in (0, 1)]
        xs = np.where(fold == 0, sm[cf[0]], sm[cf[1]])
        r = dict(candidate=f"cross-fit best fixed ({C[cf[0]][0]} / {C[cf[1]][0]})")
        for b, m in B.items():
            single["x", b] = ci(xs[m], idn[m], log[m])
            r[b] = cell(single["x", b])
        rows.append(r)
        for c in pool[1:]:
            r = dict(candidate=f"{C[c][0]} {C[c][1:]}")
            for b, m in B.items():
                single[c, b] = ci(sm[c][m], idn[m], log[m])
                r[b] = cell(single[c, b])
            rows.append(r)
        stats.write_table(rows, out / "fixed", note="one fixed transform applied to every token (no oracle), minus the identity; EPDMS x 100 without EC")
        # 5. sub-score split of the gaps (Shapley), and what the oracle recovers
        GRP = {"DAC": ["DAC"], "NC": ["NC"], "TTC": ["TTC"], "EP": ["EP"], "LK": ["LK"], "rest": ["DDC", "TLC", "HC", "EC"]}
        gcol = {g: [T.TERMS.index(t) for t in ts] for g, ts in GRP.items()}

        def nan_ec(x):
            y = np.array(x, float)
            y[..., 8] = np.nan
            return y

        def phi_of(Aa, Bb):                                                       # per-token Shapley of score(B) - score(A), seed mean, x 100
            return np.mean([T.shapley(x, y) for x, y in zip(Aa, Bb)], 0) * 100
        ph = {"WA - SH30 (with EC, archived)": phi_of(XA, [XW] * nS), "WA - SH30": phi_of(nan_ec(X[:, 0]), [nan_ec(XW)] * nS)}
        for f in [x for x in ("F19", "F27", "F33") if x in fam]:
            pf = pick(f)
            Xo = np.stack([X[s, pf[s], np.arange(N)] for s in range(nS)])
            ph[f"oracle {f} - SH30"] = phi_of(nan_ec(X[:, 0]), nan_ec(Xo))
        rows, rec = [], {}
        for b in ("> 20 deg", "20-45 deg", "> 45 deg"):
            m = B[b]
            for name, p9 in ph.items():
                tot = stats.bootstrap(p9[m].sum(1), groups=log[m])
                r = dict(bucket=b, gap=name, total=f"{tot['mean']:+.2f} [{tot['lo']:+.2f}, {tot['hi']:+.2f}]")
                for g, cols in gcol.items():
                    v = stats.bootstrap(p9[m][:, cols].sum(1), groups=log[m])
                    r[g] = f"{v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]"
                rows.append(r)
            for f in [x for x in ("F19", "F27", "F33") if x in fam]:
                r = dict(bucket=b, gap=f"recovered by oracle {f} (share of WA - SH30)")
                q = ratio_ci(ph[f"oracle {f} - SH30"][m].sum(1), ph["WA - SH30"][m].sum(1), log[m])
                r["total"] = f"{q['mean']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}]"
                for g, cols in gcol.items():
                    q = ratio_ci(ph[f"oracle {f} - SH30"][m][:, cols].sum(1), ph["WA - SH30"][m][:, cols].sum(1), log[m])
                    rec[f, b, g] = q
                    r[g] = f"{q['mean']:.2f} [{q['lo']:.2f}, {q['hi']:.2f}]"
                rows.append(r)
        stats.write_table(rows, out / "subscores", note="exact Shapley split of the per-token EPDMS gap (x 100); rows without the EC note are without "
                          "extended comfort (8 terms); rest = DDC + TLC + HC (+ EC); recovered = ratio of sums, cluster bootstrap over logs (B 4000)")
        rows = []
        for b in ("> 20 deg", "20-45 deg", "> 45 deg"):
            m = B[b]
            arms = {"SH30": X[:, 0], "WA-JEPA": np.stack([XW] * nS)}
            for f in [x for x in ("F19", "F27") if x in fam]:
                pf = pick(f)
                arms[f"oracle {f}"] = np.stack([X[s, pf[s], np.arange(N)] for s in range(nS)])
            for name, x in arms.items():
                rows.append(dict(bucket=b, arm=name, **{f"{t} fail %": 100 * (x[:, m, T.TERMS.index(t)] < 1).mean() for t in ("NC", "DAC", "DDC", "TLC", "TTC", "LK")},
                                 **{"EP": 100 * x[:, m, 4].mean(), "EPDMS (no EC)": 100 * noec(x)[:, m].mean()}))
        stats.write_table(rows, out / "rates", floatfmt=".2f", note="failure = sub-score < 1 (% of token-seeds); EP mean x 100")
        # verdict against the pre-registered line and branch rule
        g19 = G["F19", "> 20 deg"]
        vd = dict(line=LINE, family="F19", bucket="> 20 deg", gain=g19, ends_line=bool(g19["mean"] < LINE),
                  navtest_equivalent=g19["mean"] * N / 12146, single_axis={f: G[f, "> 20 deg"] for f in ("O3", "K3", "V3")},
                  identity_check_max_abs=chk, seeds=seeds, n_keys=len(keys), sources=dict(wa=wsrc, sh=[s for _, _, s in A]))
        if not vd["ends_line"]:
            ok, ov, kv = dof["offset - curvature (O3 - K3)", "> 20 deg"], dof["offset - speed (O3 - V3)", "> 20 deg"], dof["curvature - speed (K3 - V3)", "> 20 deg"]
            if ok["lo"] > 0 and ov["mean"] > 0:
                br = "a: lateral offset"
            elif ok["hi"] < 0 and kv["mean"] > 0:
                br = "b: curvature gain"
            elif ov["hi"] < 0 and kv["hi"] < 0:
                br = "d: speed profile (neither lateral axis leads)"
            else:
                br = "mixed: no single axis leads"
            vd["branch"] = br
        else:
            vd["branch"] = "c: the small family does not contain much better trajectories"
        (out / "verdict.json").write_text(json.dumps(vd, indent=1, default=float))
        pd.DataFrame(dict(token=tok, log=log, dyaw=dyaw, sh30=idn, wa=wa, **{f"best_{f}": best(f) for f in order},
                          **{f"pick_{f}_s{s}": pick(f)[i] for f in ("F19", "F27", "F33") if f in fam for i, s in enumerate(seeds)})).to_csv(out / "tokens.csv", index=False)
        run.summary.update(gain_F19=g19["mean"], lo=g19["lo"], hi=g19["hi"], ends_line=vd["ends_line"], branch=vd["branch"])
        run.info("verdict: %s", json.dumps(vd, default=float))
        figures(out / "figs" if a.out else FIG, B, G, SK, marg, order, KS, lev, fm)


def figures(FIG, B, G, SK, marg, order, KS, lev, fm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as ps
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    bs = ("20-45 deg", "> 45 deg", "> 20 deg")
    col = dict(zip(bs, (ps.PALETTE["sky_blue"], ps.PALETTE["vermillion"], ps.PALETTE["black"])))
    gap = {row["bucket"]: float(row["WA - SH30"].split(" ")[0]) for _, row in lev.iterrows()}
    # 1. ceiling per family
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COLUMN_IN, 2.6), constrained_layout=True)
    w = 0.26
    for j, b in enumerate(bs):
        v = np.array([[G[f, b]["mean"], G[f, b]["mean"] - G[f, b]["lo"], G[f, b]["hi"] - G[f, b]["mean"]] for f in order])
        ax.bar(np.arange(len(order)) + (j - 1) * w, v[:, 0], w, yerr=v[:, 1:].T, color=col[b], label=f"{b} (WA-JEPA gap {gap[b]:+.1f})",
               error_kw=dict(lw=0.6, capsize=1.5))
    ax.axhline(LINE, color=ps.BASELINE, ls="--", lw=0.8)
    ax.text(len(order) - 0.5, LINE, "pre-registered line (F19, > 20 deg)", ha="right", va="bottom", fontsize=7, color=ps.BASELINE)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("best-of-K - SH30, EPDMS x 100 (no EC)")
    ax.legend(loc="upper left"), ps.bars(ax)
    fig.savefig(FIG / "ceiling_by_family.png", dpi=300)
    plt.close(fig)
    # 2. small K
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.5), constrained_layout=True)
    for b in ("20-45 deg", "> 45 deg", "> 20 deg"):
        for kind, ls in (("in", "-"), ("x", "--")):
            v = np.array([[SK[kind, K, b]["mean"], SK[kind, K, b]["lo"], SK[kind, K, b]["hi"]] for K in KS])
            ax.plot(KS, v[:, 0], ls=ls, color=col[b], marker="o", ms=2.5, label=f"{b}, {'in-sample' if kind == 'in' else 'cross-fit'} set")
            if kind == "x":
                ax.fill_between(KS, v[:, 1], v[:, 2], color=col[b], alpha=0.12, lw=0)
    ax.axhline(LINE, color=ps.BASELINE, ls="--", lw=0.8)
    ax.set_xscale("log"), ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("K (greedy set, identity included)"), ax.set_ylabel("best-of-K - SH30, EPDMS x 100")
    ax.legend(fontsize=6)
    fig.savefig(FIG / "small_k.png", dpi=300)
    plt.close(fig)
    # 3. what the oracle picks
    fig, axs = plt.subplots(1, 3, figsize=(ps.DOUBLE_COLUMN_IN, 2.3), constrained_layout=True, sharey=True)
    grp = (("lateral offset", ("offset toward inside", "offset none", "offset toward outside"), ("inside", "none", "outside")),
           ("curvature gain", ("curvature gain < 1", "curvature gain = 1", "curvature gain > 1"), ("< 1", "1", "> 1")),
           ("speed scale", ("speed < 1", "speed = 1", "speed > 1"), ("< 1", "1", "> 1")))
    for ax, (ttl, evs, labs) in zip(axs, grp):
        for j, b in enumerate(("20-45 deg", "> 45 deg")):
            ax.bar(np.arange(3) + (j - 0.5) * 0.38, [marg[e, b] for e in evs], 0.38, color=col[b], label=b)
        ax.set_xticks(range(3), labs), ax.set_xlabel(ttl), ps.bars(ax)
    axs[0].set_ylabel(f"oracle picks on {fm}, % of token-seeds"), axs[0].legend()
    fig.savefig(FIG / "oracle_picks.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- navtrain (read-only)
def cmd_navtrain(a):
    import glob
    from jevdrive.bench import tables as T
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "turn_ceiling/navtrain", config=vars(a)) as run:
        tr = splits.load("navsim/navtrain")
        run.use_split(tr)
        names, dy = [], []
        for d in sorted(glob.glob(str(D / "runs/op_parity/cache/navtrain_full.s*of12/tab.npz"))):
            z = np.load(d)
            names += z["names"].tolist()
            dy += [T.motion(np.asarray(f, float), float(v))[0] for f, v in zip(z["fut"], z["speed"])]
        names, dy = np.array(names), np.abs(np.array(dy))
        mc = {}
        for d in sorted(glob.glob(str(D / "runs/navsim/metric_cache/*"))):
            have = {_pl.Path(p).parent.name for p in glob.glob(d + "/*/*/*/metric_cache.pkl")}
            mc[_pl.Path(d).name] = dict(n=len(have), navtrain_cached=int(np.isin(names, list(have)).sum()),
                                        navtrain_turn20=int(np.isin(names[dy >= 20], list(have)).sum()))
        res = dict(split=tr.id, split_n=len(tr), cached_tokens=len(names), unique=len(set(names.tolist())), in_split=int(tr.mask(names).sum()),
                   nan_future=int(np.isnan(dy).sum()), turn_20=int((dy >= 20).sum()), turn_20_45=int(((dy >= 20) & (dy < 45)).sum()),
                   turn_45=int((dy >= 45).sum()), metric_caches=mc)
        (OUT / "navtrain.json").write_text(json.dumps(res, indent=1))
        run.summary.update({k: v for k, v in res.items() if k != "metric_caches"})
        run.info("%s", json.dumps(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("family")
    p.add_argument("--n-smoke", type=int, default=300)
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("gate")
    p.add_argument("--stage", choices=["t24", "t300"], required=True)
    p.add_argument("--score", required=True)
    p = sp.add_parser("report")
    p.add_argument("--score", default=str(OUT / "score_all.csv"))
    p.add_argument("--out", default="")
    sp.add_parser("navtrain")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
