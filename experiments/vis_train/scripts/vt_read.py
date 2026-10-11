"""vis_train (VT) reads: the registered read-outs of every arm at every snapshot, from the `jevdrive.bench` outputs (prereg "读数" / "判定线",
amendments 1 (arm W) and 2). Idempotent: every command can be rerun at any time on whatever snapshots exist; per-snapshot work is cached.

  box      (box, envs/navsim2, one pool CPU job)  replay + build:
           replay  per model with a finished navtest read and no valid cache: fd_navsim's instrumented devkit replay (decision 153's scorer
                   hook, unchanged) of its DAC / NC / TTC failing tokens -> $OUT/tok/<model>.parquet, one row per navtest token: the bench
                   sub-scores, plan 4 s arc length, heading gain, side of the first footprint departure, NC class (nc_tax.classify, d196)
           build   paired differences to every control with the by-log cluster bootstrap of jevdrive.stats (B 10 000), boards of the final
                   checkpoints, weight displacement, verdicts -> $OUT/out/{reads.parquet, reads.md, disp.csv, probe.csv, trend.png, trend.pdf}
  sync     (Mac)   submit `box` to the pool over ssh, wait, copy $OUT/out/* to experiments/vis_train/results/
  first    (any)   the short first-snapshot table from results/reads.parquet
  probe    (box, .venv)  decision 160's Stage-0 probe on the branch tokens of --tags: three chained pool jobs (vt.py tokens on a card,
                   rep.py's thin decoder on a card, opb_score.py on the > 20 deg navtest tokens) -> $OUT/probe/<name>/score_t20.csv
  check    (box)   the vectorised bootstrap against jevdrive.stats.paired on real cells

Definitions (all reused, none re-derived here):
  buckets   jevdrive.bench.tables.navtest_strata: |logged 4 s heading change| <5 / 5-20 / 20-45 / >45 deg, and >20 = the last two.
  offroad   DAC < 1. cut_in = offroad and the first footprint corner outside along the LQR replay is on the inside of the turn
            (lqr_side == sign of the logged heading change); under = offroad, not cut_in, heading gain at 4 s < 0.9 (fd_navsim.plan_kin):
            corr_report.py's `inside` / `cannot` (decision 240). Rates are per token of the bucket, in percentage points.
  nc_*      NC < 1 by the class of nc_tax.classify (A1 stopped vehicle ahead, A2 moving lead, B cut-in, C crossing, D side contact, E other);
            ttc_only = TTC < 1 with NC = 1.
  arc       plan 4 s arc length (nc_tax.arc): ratio of sums to the logged path's (arc_ratio_log) and to the control's (arc_ratio_ctrl).
  control   same seed; the control's snapshot at the same step, else its nearest registered snapshot (ties: the earlier), column ctrl_step.
  seed mean per-token mean of the two seeds on both sides (as `bench report` does), then the same bootstrap.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "experiments/op_parity/scripts"), str(REPO / "experiments/op_probe/scripts"), str(REPO / "research")]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/vis_train/reads"
RES = REPO / "experiments/vis_train/results"
RUNS = D / "runs/op_parity/runs"
TAB = D / "runs/op_parity/cache/lb_navtest/tab.npz"
FOV = D / "runs/corridor/fov"
SSH = ["ssh", "-o", "ControlPath=none", "autodl"]
VERSION = "vt_read-1"                                   # bump to invalidate the per-model token caches

STEPS = {"F": 60, "F0": 60, "A0": 50, "A": 50, "B": 50, "C": 40, "W": 50}       # thousand steps (plans/state.md)
EVERY = {"F": 5, "F0": 10, "A0": 10, "A": 5, "B": 5, "C": 5, "W": 5}            # registered snapshot spacing
SEEDS = {"F": (0, 1), "F0": (0, 1), "A0": (0, 1), "A": (0, 1), "B": (0, 1), "C": (0, 1), "W": (0, 1)}   # F0 / C seed 1: prereg amendment 6; a seed with fewer snapshots only enters the seed mean at the steps both have
CTRL = {"F": ("SH30",), "F0": ("F",), "A0": ("F",), "A": ("A0", "F"), "B": ("A0", "A", "F"), "C": ("F0", "F"), "W": ("A", "F")}
PRIMARY = [("A", "A0"), ("B", "A0"), ("B", "A"), ("C", "F0"), ("W", "A")]         # the turn line's comparisons (prereg + amendment 1)
OPTS = {"A": ("noside", "mshuf"), "B": ("noside", "mshuf"), "W": ("noside", "mshuf", "sideoff")}
# prereg amendment 7. A2 / B2: arm A / B continued under the new tags VT-<X>2-s<seed> for 30 k steps, read as later snapshots of the same arm (steps
# counted from SH30; the controls were not continued and are read at their nearest snapshot). R0 / RB / RW0 / RW: arm R, the policy trained from the
# shipped weights with the SH30 recipe (10 k steps, one checkpoint) reading a frozen token bank through the memory channel (VTR-0 / VTR-B / VTR-W0 / VTR-W; amendment 8).
START = {"A2": 50, "B2": 50}                            # thousand steps before the tag's own step 0
RTAG = {"R0": "VTR-0", "RB": "VTR-B", "RW0": "VTR-W0", "RW": "VTR-W"}   # amendment 8: VTR-A dropped, the W arms added
RSRC = {"RB": "B", "RW": "W"}                           # the first-wave arm whose final branch fills the bank
STEPS |= {"A2": 80, "B2": 80} | dict.fromkeys(RTAG, 10)
EVERY |= {"A2": 5, "B2": 5} | dict.fromkeys(RTAG, 10)
SEEDS |= dict.fromkeys([*START, *RTAG], (0, 1))
CTRL |= {"A2": ("A0", "A", "F"), "B2": ("A0", "B", "F"), "R0": ("SH30",), "RB": ("R0", "SH30"), "RW0": ("R0", "SH30"), "RW": ("R0", "RW0", "SH30")}
PRIMARY += [("A2", "A0"), ("B2", "A0"), ("RB", "R0"), ("RW0", "R0"), ("RW", "R0"), ("RW", "RW0")]
OPTS |= dict.fromkeys([*START, *RTAG], ("noside", "mshuf"))
SC = ["EPDMS", "NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
RATES = ["offroad", "cut_in", "under", "nc_fail", "ttc_only", "nc_A", "nc_A1", "nc_A2", "nc_B", "nc_C", "nc_D", "nc_E"]
METRICS = SC + RATES
TURNS = ["<5", "5-20", "20-45", ">45"]
BUCKETS = ["all", *TURNS, ">20"]
FOVB = [">45 edge out of FoV", ">45 edge in FoV", ">45 SH30 cut-in, point out of FoV", ">45 SH30 cut-in, point in FoV"]
HUG = "spec_plan_smooth"
V_REF, WA_REF = 10.59, 6.88                             # decision 160: > 20 deg DAC failure of the thin decoder on V / on WA-Cf


def reg(arm):
    return list(range(START.get(arm, 0) + EVERY[arm], STEPS[arm] + 1, EVERY[arm]))


def name(arm, seed, step):
    if arm == "SH30" or step == 0:
        return f"SH30-F-s{seed}"
    if arm in RTAG:
        return f"{RTAG[arm]}-s{seed}"
    return f"VT-{arm}-s{seed}" + ("" if step == STEPS[arm] else f"-k{step - START.get(arm, 0):02d}")


def ctrl_step(c, step):
    return 0 if c == "SH30" else min(reg(c), key=lambda s: (abs(s - step), s))


def specs():
    """Every registered navtest read: (arm label, seed, step in thousands, bench model spec)."""
    out = [("SH30", s, 0, f"SH30-F-s{s}") for s in (0, 1)]
    for arm in STEPS:
        for s in SEEDS[arm]:
            out += [(arm, s, k, name(arm, s, k)) for k in reg(arm)]
            out += [(f"{arm}:{o}", s, STEPS[arm], f"{name(arm, s, STEPS[arm])}:{o}") for o in OPTS.get(arm, ())]
    return out


def tokf(spec):
    return OUT / "tok" / (spec.replace(":", "_") + ".parquet")


# ---------------------------------------------------------------- replay (envs/navsim2): the per-model token table
def replay(run):
    import multiprocessing as mp
    import pandas as pd
    import fd_navsim as FD
    import nc_tax as NT
    from jevdrive import bench, cache
    from jevdrive.bench import compat
    from jevdrive.common import n_cpus
    tab = np.load(TAB)
    names, fut = tab["names"].astype(str), tab["fut"].astype(np.float64)
    todo = {}
    for _, _, _, spec in specs():
        try:
            uf, pf = bench.run_dir(spec, "navtest") / "units.csv", compat.pred_file(spec, "navtest")
        except Exception:                                # a model this checkout's bench does not know yet (arm W before its routing lands)
            continue
        if uf.exists() and pf.exists():
            k = cache.key(params=dict(spec=spec), inputs=[uf, pf], version=VERSION)
            if not cache.valid(tokf(spec), k):
                todo[spec] = (uf, pf, k)
    run.info(f"replay: {len(todo)} models without a valid token table: {' '.join(todo)}")
    if not todo:
        return
    plans, want, U, PL = {}, {}, {}, {}
    for spec, (uf, pf, _) in todo.items():
        u = pd.read_csv(uf).set_index("token").reindex(names)
        assert u.score.notna().all(), f"{spec}: navtest units miss tokens"
        z = np.load(pf)
        pos = {t: i for i, t in enumerate(z["tokens"].astype(str))}
        P = z["poses"][[pos[t] for t in names]].astype(np.float64)
        bad = np.where(((u.DAC < 1) | (u.NC < 1) | (u.TTC < 1)).to_numpy())[0]
        plans[spec] = {names[i]: P[i] for i in bad}
        for i in bad:
            want.setdefault(names[i], []).append(spec)
        U[spec], PL[spec] = u, P
    (OUT / "tok").mkdir(parents=True, exist_ok=True)
    kf = OUT / f"keys-{os.getpid()}.pkl"
    pickle.dump({"plans": plans, "want": want}, open(kf, "wb"))
    procs = max(2, min(n_cpus(), len(os.sched_getaffinity(0))))
    t0, rows, tk = time.time(), [], sorted(want)
    with mp.get_context("fork").Pool(procs, initializer=FD._init, initargs=("navtest", str(kf))) as pool:
        for i, r in enumerate(pool.imap_unordered(FD.work, tk, chunksize=2)):
            rows.extend({k: v for k, v in x.items() if k != "_states"} for x in r["rows"])
            if (i + 1) % 500 == 0:
                run.status(f"replay {i + 1}/{len(tk)} tokens of {len(todo)} models, {time.time() - t0:.0f} s, {procs} workers")
    kf.unlink()
    R = pd.DataFrame(rows)
    for spec, (uf, pf, k) in todo.items():
        u, P = U[spec], PL[spec]
        r = R[R.key == spec].set_index("token").reindex(names)
        nc, bad = (u.NC < 1).to_numpy(), r.key.notna().to_numpy()
        ev = nc | ((u.TTC < 1).to_numpy() & ~nc)
        diff = float(np.abs(r.loc[bad, FD.SUBS].to_numpy(float) - u.loc[bad, NT.SUB8].to_numpy(float)).max()) if bad.any() else 0.0
        cls = np.full(len(names), "", object)
        for i in np.where(ev)[0]:
            cls[i] = NT.classify(r.iloc[i].to_dict())[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = FD.plan_kin(P, fut)["gain"]
        q = r.reindex(columns=["lqr_out", "lqr_side", "lqr_depth", "raw_out"]).astype(float)
        df = u[["log", "score", *SC[1:]]].assign(arc=NT.arc(P), gain=gain, nc_cls=cls, replay_diff=diff, **{c: q[c].to_numpy() for c in q}).reset_index()
        cache.cached(tokf(spec), k, lambda: df, force=True)
        run.info(f"{spec}: {int(bad.sum())} failing tokens replayed, max |replayed - stored sub-score| {diff:.2e}")
    run.summary.update(replayed=list(todo), replay_tokens=len(tk), replay_s=round(time.time() - t0, 1), workers=procs)


# ---------------------------------------------------------------- statistics
class CB:
    """jevdrive.stats' cluster bootstrap by log for many columns at once: the same resample (default_rng(0).integers(n_u, size=(B, n_u)) over the
    logs present, in order of appearance), the same statistic (ratio of sums), written as draw counts W (B, n_u) so that one matrix product
    serves every column. `check` compares it with jevdrive.stats.paired."""

    def __init__(self, logs, B=None):
        from jevdrive import stats
        self.logs, self.B, self.seed, self.W = np.asarray(logs), B or stats.N_BOOT, stats.SEED, {}

    def w(self, n):
        if n not in self.W:
            idx = np.random.default_rng(self.seed).integers(n, size=(self.B, n))
            W = np.zeros((self.B, n))
            np.add.at(W, (np.arange(self.B)[:, None], idx), 1.0)
            self.W[n] = W
        return self.W[n]

    def sums(self, m, *X):
        """Draw counts and the per-log sums of each X (n,) / (n, M) and of the token count, over the tokens of mask m."""
        import pandas as pd
        codes, uniq = pd.factorize(self.logs[m])
        out = []
        for x in X + (np.ones(len(self.logs)),):
            x = np.asarray(x, float)[m].reshape(len(codes), -1)
            S = np.zeros((len(uniq), x.shape[1]))
            np.add.at(S, codes, x)
            out.append(S)
        return self.w(len(uniq)), out

    def mean(self, X, m):
        """(mean, lo, hi, n) per column of X (n, M) over mask m; tokens where a column is not finite are dropped for that column."""
        X = np.asarray(X, float)
        fin = np.isfinite(X) & m[:, None]
        mu, lo, hi, n = (np.full(X.shape[1], np.nan) for _ in range(4))
        pat = {}
        for j in range(X.shape[1]):
            pat.setdefault(fin[:, j].tobytes(), []).append(j)
        for cols in pat.values():
            ok = fin[:, cols[0]]
            n[cols] = ok.sum()
            if not ok.any():
                continue
            W, (S, c) = self.sums(ok, X[:, cols])
            b = (W @ S) / (W @ c)
            mu[cols], (lo[cols], hi[cols]) = X[ok][:, cols].mean(0), np.quantile(b, [0.025, 0.975], axis=0)
        return mu, lo, hi, n

    def ratio(self, num, den, m, num2=None):
        """sum(num) / sum(den) over mask m (minus sum(num2) / sum(den) when given) with its CI."""
        W, S = self.sums(m, num, den, *(() if num2 is None else (num2,)))
        a, d = S[0], S[1]
        v, b = a.sum() / d.sum(), (W @ a) / (W @ d)
        if num2 is not None:
            v, b = v - S[2].sum() / d.sum(), b - (W @ S[2]) / (W @ d)
        lo, hi = np.quantile(b[:, 0], [0.025, 0.975])
        return float(v), float(lo), float(hi)


def feats(df, sgn):
    """(n, len(METRICS)) per-token values in points / percentage points, rows in the order of df."""
    dac, nc = (df.DAC < 1).to_numpy(), (df.NC < 1).to_numpy()
    inside = dac & (df.lqr_side.to_numpy(float) == sgn)
    with np.errstate(invalid="ignore"):
        under = dac & ~inside & (df.gain.to_numpy(float) < 0.9)
    cls = df.nc_cls.fillna("").astype(str)
    cols = [df.score.to_numpy(float)] + [df[k].to_numpy(float) for k in SC[1:]] + [dac, inside, under, nc, (df.TTC < 1).to_numpy() & ~nc] + \
           [nc & cls.str.startswith(p).to_numpy() for p in ("A", "A1", "A2", "B", "C", "D", "E")]
    return 100.0 * np.stack([np.asarray(c, float) for c in cols], 1)


def masks(names, st):
    """Bucket masks over the tab order: the six turn buckets for everyone, per seed the fixed > 45 deg FoV strata (plans/state.md, arm W)."""
    import pandas as pd
    turn = st.turn.reindex(names).to_numpy()
    M = {"all": np.ones(len(names), bool)} | {b: turn == b for b in TURNS}
    M[">20"] = M["20-45"] | M[">45"]
    F = {0: {}, 1: {}, "mean": {}}
    if (FOV / "tok.parquet").exists():
        t = pd.read_parquet(FOV / "tok.parquet").set_index("token").reindex(names)
        e = t.edge_in_w_fov_t0.to_numpy(float)
        has = M[">45"] & (t.edge_in_n.to_numpy(float) > 0)
        un = pd.read_parquet(FOV / "unit.parquet")
        for s in F:
            F[s] = {FOVB[0]: has & (e == 0), FOVB[1]: has & (e > 0)}
            if s != "mean":
                q = un[(un.unit == f"sh{s}_pp") & (un.lqr_out.astype(float) > 0) & (un.lqr_side.astype(float) == un.sgn.astype(float))]
                vis = q.set_index("token").lqr_pt_w_fov_t0.astype(float).reindex(names).to_numpy()
                F[s] |= {FOVB[2]: M[">45"] & (vis == 0), FOVB[3]: M[">45"] & (vis > 0)}
    return M, F


def board_units(bench, spec):
    from jevdrive.bench import tables as T
    try:
        u, _ = T.load(bench, spec, **({"preset": HUG} if bench == "hugsim" else {}))
    except Exception:
        return None
    return u


# ---------------------------------------------------------------- build
def build(run):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench import tables as T
    tab = np.load(TAB)
    names, fut = tab["names"].astype(str), tab["fut"].astype(np.float64)
    sgn, ref_arc = np.sign(fut[:, -1, 2]), __import__("nc_tax").arc(fut)
    M, FV = masks(names, T.navtest_strata())
    L, X, ARC, rdiff = {}, {}, {}, {}
    for arm, seed, step, spec in specs():
        if tokf(spec).exists():
            df = pd.read_parquet(tokf(spec)).set_index("token").reindex(names)
            L[(arm, seed, step)], X[(arm, seed, step)], ARC[(arm, seed, step)] = spec, feats(df, sgn), df.arc.to_numpy(float)
            rdiff[spec] = float(df.replay_diff.iloc[0])
            logs = df["log"].astype(str).to_numpy()
    assert L, "no navtest read is cached yet"
    cb, rows = CB(logs), []

    def emit(meta, mk, xa, aa, xc=None, ac=None):
        for b, m in mk.items():
            if not m.any():
                continue
            va = cb.mean(xa, m)
            vc = cb.mean(xc, m)[0] if xc is not None else None
            d = cb.mean(xa - xc, m) if xc is not None else None
            for j, k in enumerate(METRICS):
                rows.append(meta | dict(bucket=b, metric=k, n=int(va[3][j]), value=va[0][j]) |
                            (dict(ci_lo=va[1][j], ci_hi=va[2][j]) if d is None else dict(ctrl_value=vc[j], diff=d[0][j], ci_lo=d[1][j], ci_hi=d[2][j])))
            v, lo, hi = cb.ratio(aa, ref_arc, m)
            if d is None:
                rows.append(meta | dict(bucket=b, metric="arc_ratio_log", n=int(m.sum()), value=v, ci_lo=lo, ci_hi=hi))
                continue
            dv, lo, hi = cb.ratio(aa, ref_arc, m, ac)
            rows.append(meta | dict(bucket=b, metric="arc_ratio_log", n=int(m.sum()), value=v, ctrl_value=v - dv, diff=dv, ci_lo=lo, ci_hi=hi))
            v, lo, hi = cb.ratio(aa, ac, m)
            rows.append(meta | dict(bucket=b, metric="arc_ratio_ctrl", n=int(m.sum()), value=v, ctrl_value=1.0, diff=v - 1, ci_lo=lo - 1, ci_hi=hi - 1))

    def side(arm, step, seed):
        """Per-token values and arc of (arm, step) for one seed or the mean of both; None when a member is missing."""
        ks = [(arm, s, step) for s in ((0, 1) if seed == "mean" else (seed,))]
        if not all(k in X for k in ks) or (seed == "mean" and len(SEEDS.get(arm.split(":")[0], (0, 1))) < 2):
            return None
        return np.mean([X[k] for k in ks], 0), np.mean([ARC[k] for k in ks], 0)

    for arm, step in sorted({(a, k) for a, _, k in L}):
        base = arm.split(":")[0]
        ctrls = [(base, step)] if ":" in arm else [(c, ctrl_step(c, step)) for c in CTRL.get(arm, ())]
        for seed in (0, 1, "mean"):
            a = side(arm, step, seed)
            if a is None:
                continue
            meta = dict(board="navtest", arm=arm, seed=str(seed), step=step, model=L[(arm, 0 if seed == "mean" else seed, step)] if seed != "mean" else "seed mean")
            mk = M | FV[seed]
            emit(meta | dict(control="", ctrl_step=-1), mk, *a)
            for c, cs in ctrls:
                b = side(c, cs, seed)
                if b is not None:
                    emit(meta | dict(control=c, ctrl_step=cs), mk, *a, *b)
        run.status(f"build: navtest {arm} k{step:02d}")

    # ---- flips: tokens the control fails and the arm passes ("fixed") / the control passes and the arm fails ("broken"), per failure class
    FLC = [(k, b) for k in ("offroad", "cut_in", "under") for b in (">45", ">20")] + \
          [(k, "all") for k in ("nc_fail", "nc_A", "nc_A1", "nc_A2", "nc_B", "nc_C", "nc_D", "nc_E", "ttc_fail", "offroad")]
    col = {k: j for j, k in enumerate(METRICS)}

    def fails(key, k):
        x = X[key]
        return x[:, col["TTC"]] < 100 if k == "ttc_fail" else x[:, col[k]] > 50
    fl = []
    for arm in STEPS:
        for c in CTRL[arm]:
            per = {}
            for s in SEEDS[arm]:
                st = sorted(t for (a, sd, t) in X if a == arm and sd == s and (c, s, ctrl_step(c, t)) in X)
                if st:
                    per[s] = st
            if not per:
                continue
            for s in [*per, "sum"]:
                ss = list(per) if s == "sum" else [s]
                common = set.intersection(*[set(per[q]) for q in ss])
                for pick, step in (("latest common", max(common) if common else None), ("final", STEPS[arm] if common and STEPS[arm] in common else None)):
                    if step is None:
                        continue
                    for k, b in FLC:
                        fx = br = na = nc_ = 0
                        for q in ss:
                            a_, c_ = fails((arm, q, step), k)[M[b]], fails((c, q, ctrl_step(c, step)), k)[M[b]]
                            fx, br, na, nc_ = fx + int((c_ & ~a_).sum()), br + int((~c_ & a_).sum()), na + int(a_.sum()), nc_ + int(c_.sum())
                        fl.append(dict(arm=arm, control=c, seed=str(s), pick=pick, step=step, ctrl_step=ctrl_step(c, step), cls=k, bucket=b, fixed=fx, broken=br,
                                       net=fx - br, ctrl_fail=nc_, arm_fail=na))
    FL = pd.DataFrame(fl, columns=["arm", "control", "seed", "pick", "step", "ctrl_step", "cls", "bucket", "fixed", "broken", "net", "ctrl_fail", "arm_fail"])

    # ---- boards of the final checkpoints: navhard two-stage (by group, clustered by the stage-1 log), HUGSIM 64 (per scenario)
    BU = {}
    for bench, cols in (("navhard", ["combined", "stage1", "stage2"]), ("hugsim", ["hdscore"])):
        for arm in ["SH30", *STEPS]:
            for s in SEEDS.get(arm, (0, 1)):
                u = board_units(bench, f"VTCP2-s{s}" if (bench, arm) == ("hugsim", "C") else name(arm, s, STEPS.get(arm, 0)))   # C is served as a P2 tag (vt_native.py export)
                if u is not None:
                    BU[(bench, arm, s)] = u
        for (bn, arm, s), u in list(BU.items()):
            if bn != bench:
                continue
            step = STEPS.get(arm, 0)
            meta = dict(board=bench, arm=arm, seed=str(s), step=step, model=name(arm, s, step), bucket="all")
            for k in cols:
                r = stats.bootstrap(u[k].to_numpy(float), u["log"].astype(str).to_numpy() if bench == "navhard" else None)
                rows.append(meta | dict(control="", ctrl_step=-1, metric=k, n=r["n"], value=r["mean"], ci_lo=r["lo"], ci_hi=r["hi"]))
            for c in [*CTRL.get(arm, ()), *(["SH30"] if arm not in ("SH30", "F") else [])]:
                v = BU.get((bench, c, s))
                if v is None:
                    continue
                ix = u.index.intersection(v.index)
                for k in cols:
                    r = stats.paired(u.loc[ix, k].to_numpy(float), v.loc[ix, k].to_numpy(float), u.loc[ix, "log"].astype(str).to_numpy() if bench == "navhard" else None)
                    rows.append(meta | dict(control=c, ctrl_step=STEPS.get(c, 0), metric=k, n=r["n"], value=r["mean_a"], ctrl_value=r["mean_b"], diff=r["mean"], ci_lo=r["lo"], ci_hi=r["hi"]))
    R = pd.DataFrame(rows)
    for c in ("ctrl_value", "diff"):
        if c not in R:
            R[c] = np.nan
    R = R[["board", "arm", "seed", "step", "model", "control", "ctrl_step", "bucket", "metric", "n", "value", "ctrl_value", "diff", "ci_lo", "ci_hi"]]

    # ---- weight displacement and the dev ADE (vt.py train's evals.json, every eval step)
    ev = []
    for arm in STEPS:
        for s in SEEDS[arm]:
            f = RUNS / name(arm, s, STEPS[arm]) / "evals.json"
            if f.exists():                                # a continuation counts its steps from the first run's start
                ev += [dict(arm=arm, seed=s) | e | dict(step=e["step"] + 1000 * START.get(arm, 0)) for e in json.loads(f.read_text())]
    E = pd.DataFrame(ev)

    # ---- Stage-0 probe (decision 160): > 20 deg DAC failure of the thin decoder on each token source, paired with V of the same decode run
    pr = []
    for f in sorted((OUT / "probe").glob("*/score_t20.csv")):
        sc = pd.read_csv(f)
        fail = 100.0 * (1 - sc.pivot_table(index="token", columns="key", values="drivable_area_compliance")).reindex(names[M[">20"]])
        lg = logs[M[">20"]]
        for k in fail.columns:
            r = stats.bootstrap(fail[k].to_numpy(float), lg)
            p = stats.paired(fail[k].to_numpy(float), fail["V"].to_numpy(float), lg) if "V" in fail and k != "V" else {}
            pr.append(dict(run=f.parent.name, source=k, n=r["n"], fail_pct=r["mean"], lo=r["lo"], hi=r["hi"],
                           minus_V=p.get("mean", np.nan), d_lo=p.get("lo", np.nan), d_hi=p.get("hi", np.nan)))
    PR = pd.DataFrame(pr, columns=["run", "source", "n", "fail_pct", "lo", "hi", "minus_V", "d_lo", "d_hi"])

    od = OUT / "out"
    od.mkdir(parents=True, exist_ok=True)
    R.to_parquet(od / "reads.parquet", index=False)
    FL.to_csv(od / "flips.csv", index=False)
    E.to_csv(od / "disp.csv", index=False)
    PR.to_csv(od / "probe.csv", index=False)
    figure(R, od / "trend")
    (od / "reads.md").write_text(report(R, E, PR, L, rdiff))
    run.summary.update(models=len(L), rows=len(R), out=str(od))


# ---------------------------------------------------------------- figure
def figure(R, stem):
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style as PSY
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    PSY.apply()
    col = PSY.PALETTE
    C = {"F": PSY.BASELINE, "F0": col["black"], "A0": col["sky_blue"], "A": col["blue"], "B": col["vermillion"], "C": col["green"], "W": col["purple"]}
    C |= {"A2": C["A"], "B2": C["B"]}                   # the continuation: the same line, drawn on from the arm's final point (arm R is not a step series)
    S = R[(R.board == "navtest") & (R.control == "")].set_index(["arm", "seed", "bucket", "metric", "step"]).value
    fig, axs = plt.subplots(1, 3, figsize=(PSY.DOUBLE_COLUMN_IN, 2.5))
    for ax, (b, k, yl) in zip(axs, ((">20", "offroad", "off-road rate, > 20 deg (%)"), (">45", "offroad", "off-road rate, > 45 deg (%)"), ("all", "EPDMS", "navtest EPDMS"))):
        for arm, c in C.items():
            ls = "--" if arm in ("F", "F0", "A0") else "-"
            for seed in ("0", "1", "mean"):
                if (arm, seed, b, k) not in S.index:
                    continue
                y = S[(arm, seed, b, k)].sort_index()
                k0 = (arm[0] if arm in START else "SH30", seed, b, k, START.get(arm, 0))
                y0 = [S[k0]] if k0 in S.index else []
                x = ([k0[-1]] if y0 else []) + list(y.index)
                thin = seed != "mean" and (arm, "mean", b, k) in S.index
                ax.plot(x, y0 + list(y.to_numpy()), ls, color=c, lw=0.5 if thin else 1.2, alpha=0.55 if thin else 1.0, marker="o", ms=1.2 if thin else 2.2,
                        label=arm if arm not in START and not thin and (seed == "mean" or (arm, "mean", b, k) not in S.index) else None)
        ax.set_xlabel("training steps after SH30 (thousands)")
        ax.set_ylabel(yl)
    h, lab = axs[0].get_legend_handles_labels()
    u = dict(zip(lab, h))
    axs[2].legend(u.values(), u.keys(), fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout(pad=0.4)
    PSY.save(fig, stem)
    plt.close(fig)


# ---------------------------------------------------------------- report (reads.md) and verdicts
def md(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(":--" if i == 0 else "--:" for i in range(len(head))) + "|"] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def ci(r, f="+.2f"):
    return "n/a" if r is None or not np.isfinite(r["diff"]) else f"{r['diff']:{f}} [{r['ci_lo']:{f}}, {r['ci_hi']:{f}}]"


class Q:
    """Row lookup on the long table."""

    def __init__(self, R):
        self.R = R
        self.ix = {k: i for i, k in enumerate(zip(R.board, R.arm, R.seed, R.step, R.control, R.bucket, R.metric))}

    def __call__(self, arm, seed, step, control, bucket, metric, board="navtest"):
        i = self.ix.get((board, arm, str(seed), step, control, bucket, metric))
        return None if i is None else self.R.iloc[i]

    def steps(self, arm, seed, control=""):
        R = self.R
        return sorted(R[(R.board == "navtest") & (R.arm == arm) & (R.seed == str(seed)) & (R.control == control)].step.unique())

    def seeds(self, arm, control):
        """Seed labels with rows for this comparison, the seed mean first."""
        R = self.R
        s = set(R[(R.board == "navtest") & (R.arm == arm) & (R.control == control)].seed)
        return [x for x in ("mean", "0", "1") if x in s]


def disp_at(E, arm, seed, step, col="disp_vis"):
    """Displacement (or any eval column) at a snapshot step; the seed mean when seed == 'mean'. None when not logged."""
    if not len(E) or col not in E:
        return None
    e = E[(E.arm == arm) & (E.step == 1000 * step) & (True if seed == "mean" else E.seed == int(seed))][col].dropna()
    return float(e.mean()) if len(e) and (seed != "mean" or len(e) == len(SEEDS[arm])) else None


def tri(*xs):
    """Three-valued and: False if any is False, None if any is unknown, else True."""
    return False if any(x is False for x in xs) else None if any(x is None for x in xs) else True


def verdict(q, E, PR, arm, c, seed):
    """The registered turn line (prereg "判定线"; amendment 1 for W), the guardrail and their inputs for one comparison and seed label."""
    st = q.steps(arm, seed, c)
    if not st:
        return None
    last, w = st[-1], arm == "W"
    b, k, thr_c, thr_f = (">45", "cut_in", -1.0, 0.5) if w else (">20", "offroad", -0.5, 0.3)
    r = q(arm, seed, last, c, b, k)
    ser = [q(arm, seed, s, c, b, k)["diff"] for s in st]
    if arm in START:                                    # a continuation: the series goes on from the arm's own snapshots against the same control
        ser = [q(arm[0], seed, s, c, b, k)["diff"] for s in q.steps(arm[0], seed, c)] + ser
    one = arm in RTAG                                   # arm R: one checkpoint, the trend clause does not apply (amendment 7)
    down = True if one else None if len(ser) < 6 else bool(np.mean(ser[-3:]) < np.mean(ser[:3]))
    dv = disp_at(E, RSRC[arm], seed, STEPS[RSRC[arm]]) if arm in RSRC else disp_at(E, arm, seed, last)      # arm R: its bank's source checkpoint
    drop = {}
    for o in OPTS.get(arm, ()):
        x = q(f"{arm}:{o}", seed, STEPS[arm], arm, "all", "EPDMS")
        drop[o] = None if x is None else -float(x["diff"])
    src = [name(RSRC.get(arm, arm), s, STEPS[RSRC.get(arm, arm)]) for s in ((0, 1) if seed == "mean" else (int(seed),))]
    p = PR[PR.source.isin(src)] if len(PR) else PR
    moved = None if not len(p) else bool(((p.d_lo > 0) | (p.d_hi < 0)).any())
    d, lo, hi = float(r["diff"]), float(r["ci_lo"]), float(r["ci_hi"])
    has0 = lo <= 0 <= hi
    mem = arm in OPTS
    read = None if not mem else None if drop.get("noside") is None else drop["noside"] >= 0.2
    if w:
        so = drop.get("sideoff")
        closing = d <= thr_c and hi < 0
        flat = tri(abs(d) < thr_f and has0, None if so is None else so < 0.2)
        unmeasured = False
    else:
        closing = tri(d <= thr_c and hi < 0, down)
        flat = tri(abs(d) < thr_f and has0, None if dv is None else dv >= 0.01, read if mem else True)
        unread = tri(None if read is None else not read, None if moved is None else not moved) if mem else False
        unmeasured = True if (dv is not None and dv < 0.005) else unread if (dv is not None or unread is None) else None
    names_ = (("seeing the curb helps", closing), ("not used", flat)) if w else (("closing", closing), ("flat", flat), ("not measured", unmeasured))
    hit = [n for n, x in names_ if x is True]
    pend = [n for n, x in names_ if x is None]
    line = hit[0] if hit else f"pending ({', '.join(pend)} undecidable: inputs missing)" if pend else "undetermined"
    g = [q(arm, seed, last, c, bb, "EPDMS") for bb in ("all", "<5")]
    guard = all(float(x["ci_lo"]) >= -0.3 for x in g)
    return dict(arm=arm, control=c, seed=seed, step=last, final=last == STEPS[arm], quantity=f"{k} {b}", d=d, lo=lo, hi=hi, row=r, n_snap=len(ser), trend_down=None if one else down, one=one,
                first3=float(np.mean(ser[:3])), last3=float(np.mean(ser[-3:])), disp=dv, drop=drop, probe_moved=moved, line=line, guard=guard, g=g)


def cells(q, arm, c, seed, step):
    """Amendment 2's no-regression cells: sub-metric x bucket on navtest, then the boards. -> [(label, row)]"""
    out = [(f"{k} {b}", q(arm, seed, step, c, b, k)) for b in ["all", *TURNS] for k in SC]
    for board, ks in (("navhard", ["combined", "stage1", "stage2"]), ("hugsim", ["hdscore"])):
        for k in ks:
            for s in ((0, 1) if seed == "mean" else (int(seed),)):
                out.append((f"{board} {k} s{s}", q(arm, s, STEPS[arm], c, "all", k, board)))
    return out


def report(R, E, PR, L, rdiff):
    q, P = Q(R), []
    A = P.append
    have = {}
    for (arm, seed, step) in L:
        have.setdefault((arm, seed), []).append(step)
    A("# VT reads (generated by `scripts/vt_read.py`; do not edit)\n")
    A(f"Generated {time.strftime('%F %T %Z')} from the `jevdrive.bench` navtest reads present on the box. Units: scores in points (x 100), rates in percentage "
      "points of the bucket's tokens, `diff [lo, hi]` = arm minus control, same seed (or the per-token mean of both seeds), by-log cluster bootstrap, "
      "B 10 000 (`jevdrive.stats`). A control without a snapshot at the arm's step is read at its nearest registered snapshot (column `ctrl`). "
      "Definitions: header of `scripts/vt_read.py`. Every number is in `reads.parquet` (long format); this file shows the selections below. "
      "These are trends at a small dose; the tables carry no interpretation.\n")
    A("## Snapshots present\n")
    rws = []
    for arm in ["SH30", *STEPS]:
        for s in SEEDS.get(arm, (0, 1)):
            got = sorted(have.get((arm, s), []))
            if arm in START and not got:                  # the continuation that was not chosen (or not started yet)
                continue
            regs = [0] if arm == "SH30" else reg(arm)
            opts = [o for o in OPTS.get(arm, ()) if (f"{arm}:{o}", s, STEPS[arm]) in L]
            rws.append([f"{arm}-s{s}" + (f" (`{name(arm, s, STEPS[arm])}`)" if arm in START or arm in RTAG else ""), " ".join(f"k{k:02d}" for k in got) or "none", " ".join(f"k{k:02d}" for k in regs if k not in got) or "none",
                        " ".join(opts) or ("none" if arm in OPTS else "n/a")])
    A(md(["arm", "navtest reads present (final = last registered step)", "registered, not read yet", "test-time options read"], rws))
    bad = {k: v for k, v in rdiff.items() if v > 1e-6}
    A(f"Replay identity: the instrumented replay of the failing tokens reproduces the stored 8 sub-scores on {len(rdiff) - len(bad)} / {len(rdiff)} models "
      f"(max |difference| {max(rdiff.values()):.1e})" + (f"; **differs on {bad}**" if bad else "") + ".\n")

    A("## Arm R and the continuation (prereg amendment 7)\n")
    A("`R0` / `RB` / `RW0` / `RW` = `VTR-0` / `VTR-B` / `VTR-W0` / `VTR-W-s<seed>`: the policy trained from the shipped weights with the SH30 recipe (10 000 steps x 128, cosine), reading through the "
      "memory channel a frozen token bank: Cinque's own t0 tokens (R0, the control), the final branch of `VT-B-s<seed>` (RB), Cinque's frozen encoder on the three views of arm W (RW0), or the trained branch of `VT-W-s<seed>` on those views (RW). Their one checkpoint is "
      "listed as k10; the registered comparisons are RB - R0, RW0 - R0, RW - R0, RW - RW0, each also against SH30 (R0 - SH30 is a reference row). `A2` / `B2` = arm A / B continued for 30 000 "
      "steps under the tags `VT-A2` / `VT-B2-s<seed>`, read as steps k55 .. k80 of the same arm; its controls were not continued (`ctrl` shows the snapshot used). Both appear in "
      "every table below under these labels: trend, full decomposition, NC classes and arc ratios, boards (navhard), test-time options, verdicts, no-regression, flips, moved lists.\n")
    gf = OUT.parent / "chain" / "R" / "gate.json"
    if gf.exists():
        g = json.loads(gf.read_text())
        gt, fb = g.get("gate", {}), g.get("fallback", {})
        A(f"Gate (`scripts/vt_r.py gate`, {g.get('time', '?')}): " + ("not evaluated" if gt.get("passed") is None else
          f"masked-memory drop of arm A at its final checkpoint {gt['drop']:+.2f} EPDMS on the seed mean (seeds {', '.join(f'{v:+.2f}' for v in gt['per_seed'].values())}; "
          f"line >= {g['drop_line']}): **{'passed, R runs' if gt['passed'] else 'failed, R is not run' if not (gf.parent / 'GATE_OVERRIDE').exists() else 'failed; overridden by the user (prereg amendment 8), R is run'}**") + ". Continuation: " +
          ("not chosen" if not fb.get("pick") else f"> 45 deg off-road rate against A0 over the last three snapshots, seed mean: A {fb['value']['A']:+.2f} pp, B {fb['value']['B']:+.2f} pp "
           f"-> **arm {fb['pick']}**") + ".\n")
    else:
        A("Gate: not evaluated yet (`$DATA_DIR/runs/vis_train/chain/R/gate.json`).\n")
    bk = [json.loads(f.read_text()) for f in sorted((D / "runs/op_parity/mem").glob("vtr_*/bank.json"))]
    if bk:
        A(md(["bank", "source checkpoint", "its step", "navtest: mean |token - frozen Cinque token| (RMS of the frozen token)"],
             [[b["kind"], b["src"], b["step"], "n/a" if "navtest_vs_cinque" not in b else f"{b['navtest_vs_cinque']['mean_abs']:.4f} ({b['navtest_vs_cinque']['ref_rms']:.3f})"] for b in bk]))
    A("## Figure\n\n![trend](trend.png)\n")
    A("Off-road (DAC failure) rate on > 20 deg and > 45 deg tokens and the full navtest EPDMS against training steps; step 0 is `SH30-F-s<seed>`. One colour per arm, "
      "controls (F, F0, A0) dashed, thin lines are single seeds and the thick line their mean. What to look at: whether a trained-vision arm (A, B, C, W; solid) "
      "separates from its dashed control as steps grow, in the left two panels downward and in the right panel not downward. PDF: `trend.pdf`.\n")

    A("## First snapshot (`vt_read.py first`)\n")
    A(first_table(R, E))

    A("## Trend per comparison\n")
    A("One row per snapshot and seed label; `ctrl` is the control's snapshot actually used.\n")
    for arm in STEPS:
        for c in CTRL[arm]:
            if not q.seeds(arm, c):
                continue
            A(f"### {arm} - {c}" + ("" if (arm, c) in PRIMARY or (arm, c) == ("F0", "F") or c == "F" else " (reference, not a registered comparison)") + "\n")
            rws = []
            for seed in q.seeds(arm, c):
                for s in q.steps(arm, seed, c):
                    g = lambda b, k: ci(q(arm, seed, s, c, b, k))  # noqa: E731
                    cs = int(q(arm, seed, s, c, "all", "EPDMS")["ctrl_step"])
                    dv = disp_at(E, arm, seed, s)
                    rws.append([seed, f"k{s:02d}", f"k{cs:02d}" + ("" if cs == s or c == "SH30" else " (nearest)"), g("all", "EPDMS"), g("<5", "EPDMS"), g(">20", "offroad"),
                                g(">45", "offroad"), g(">45", "cut_in"), g(">45", "under"), g("all", "NC"), g("all", "EP"), "n/a" if dv is None else f"{100 * dv:.2f}%"])
            A(md(["seed", "step", "ctrl", "EPDMS all", "EPDMS < 5", "off-road > 20", "off-road > 45", "cut-inside > 45", "under-turn > 45", "NC all", "EP all", "vision disp."], rws))

    A("## Full decomposition at the latest snapshot\n")
    A("Arm minus control per sub-metric and turn bucket (every snapshot is in `reads.parquet`). Seed mean where both seeds exist, else the seed shown.\n")
    for arm in STEPS:
        for c in CTRL[arm]:
            sd = q.seeds(arm, c)
            if not sd:
                continue
            s = q.steps(arm, sd[0], c)[-1]
            A(f"### {arm} - {c}, k{s:02d}, seed {sd[0]}\n")
            A(md(["metric", *BUCKETS], [[k] + [ci(q(arm, sd[0], s, c, b, k)) for b in BUCKETS] for k in SC + RATES[:3]]))

    A("## Longitudinal: NC failures by class (decision 196) and plan arc length\n")
    A("Counts are tokens of the full navtest (seed mean where both seeds exist) at the latest snapshot; `A` = stationary or slow lead in the own lane (A1 stopped, A2 moving). "
      "`arc / log` = plan 4 s arc length over the logged one (ratio of sums); `arc / ctrl` = over the first control's.\n")
    rws = []
    for arm in ["SH30", *STEPS]:
        sd = [x for x in ("mean", "0", "1") if q.steps(arm, x)]
        if not sd:
            continue
        s, seed = q.steps(arm, sd[0])[-1], sd[0]
        v = lambda k, b="all": q(arm, seed, s, "", b, k)  # noqa: E731
        n = v("nc_fail")["n"]
        c = CTRL.get(arm, ("",))[0]
        ar, ac = v("arc_ratio_log"), q(arm, seed, s, c, "all", "arc_ratio_ctrl") if c else None
        rws.append([arm, seed, f"k{s:02d}"] + [f"{v(k)['value'] * n / 100:.1f}" for k in RATES[3:]] +
                   [ci(q(arm, seed, s, c, "all", "nc_A")) if c else "n/a", f"{ar['value']:.3f} [{ar['ci_lo']:.3f}, {ar['ci_hi']:.3f}]",
                    "n/a" if ac is None else f"{ac['value']:.3f} [{1 + ac['ci_lo']:.3f}, {1 + ac['ci_hi']:.3f}] vs {c}"])
    A(md(["arm", "seed", "step", "NC fail", "TTC-only", "A", "A1", "A2", "B", "C", "D", "E", "A rate - first control (pp)", "arc / log", "arc / ctrl"], rws))

    A("## Weight displacement and dev ADE (`evals.json`, at the snapshot steps)\n")
    A("Relative L2 displacement from the starting point in percent (vision total and stages, policy, adapter), dev ADE in metres; full eval history: `disp.csv`.\n")
    if len(E):
        dc = [c for c in ("disp_vis", "disp_vis_stem", "disp_vis_s0", "disp_vis_s1", "disp_vis_s2", "disp_vis_s3", "disp_vis_head", "disp_policy", "disp_adapter") if c in E]
        ac = [c for c in ("ade", "ade_masked", "head_ade") if c in E]
        e = E[[r.step % (1000 * EVERY[r.arm]) == 0 for r in E.itertuples()]]
        f = lambda x, sc, p: "" if not np.isfinite(x) else f"{sc * x:.{p}f}"  # noqa: E731
        A(md(["arm", "seed", "step"] + [c[5:] for c in dc] + ac,
             [[r.arm, r.seed, f"k{r.step // 1000:02d}"] + [f(r[c], 100, 3) for c in dc] + [f(r[c], 1, 3) for c in ac] for _, r in e.iterrows()]))
    else:
        A("No `evals.json` found.\n")

    A("## Stage-0 probe on the branch tokens (decision 160)\n")
    A(f"Thin decoder (`rep.py decode`, unchanged) on [tokens, ego], DAC failure rate on the 3 154 navtest tokens > 20 deg; references from decision 160: V {V_REF}%, WA-Cf {WA_REF}%. "
      "`V` below is the same decoder refitted on Cinque's frozen tokens in the same run (reproduction and the paired reference).\n")
    A(md(["run", "source", "fail %", "minus V (pp)"], [[r.run, r.source, f"{r.fail_pct:.2f} [{r.lo:.2f}, {r.hi:.2f}]", "" if not np.isfinite(r.minus_V) else f"{r.minus_V:+.2f} [{r.d_lo:+.2f}, {r.d_hi:+.2f}]"]
                                                       for r in PR.itertuples()]) if len(PR) else "Not run yet (`vt_read.py probe --tags ...`).\n")

    A("## Final checkpoints: boards and test-time options\n")
    B = R[R.board != "navtest"]
    if len(B):
        rws = [[r.board, f"{r.arm}-s{r.seed}", r.metric, f"{r.value:.3f}" if r.board == "hugsim" else f"{r.value:.2f}", r.control or "",
                "" if not r.control else f"k{int(r.ctrl_step):02d}", "" if not r.control else ci(r._asdict(), "+.3f" if r.board == "hugsim" else "+.2f")] for r in B.itertuples()]
        A(md(["board", "model (final)", "metric", "value", "control", "ctrl step", "diff"], [[x[0], x[1], x[2], x[3], x[4], x[5], x[6]] for x in rws]))
    else:
        A("No navhard or HUGSIM read of a final checkpoint exists yet.\n")
    rws = []
    for arm in OPTS:
        for o in OPTS[arm]:
            for seed in q.seeds(f"{arm}:{o}", arm):
                g = lambda b, k: ci(q(f"{arm}:{o}", seed, STEPS[arm], arm, b, k))  # noqa: E731
                rws.append([f"{arm}:{o}", seed, g("all", "EPDMS"), g(">20", "EPDMS"), g(">20", "offroad"), g(">45", "cut_in")])
    A("Test-time options minus the unmodified final checkpoint (`:noside` masks the memory, `:mshuf` feeds another log's memory, `:sideoff` masks W's side views); "
      "a negative EPDMS difference = the channel is read:\n")
    A(md(["model", "seed", "EPDMS all", "EPDMS > 20", "off-road > 20", "cut-inside > 45"], rws) if rws else "None read yet.\n")

    A("## W: > 45 deg cut-inside by whether the inner curb was inside the normal field of view at t0\n")
    A("Strata (fixed token sets, `runs/corridor/fov/`): `edge out / in` = share of the inner-curb samples inside the wide frame's horizontal view at t0 is 0 / above 0; "
      "`SH30 cut-in, point out / in` = the tokens where `SH30-F-s<seed>` cut inside, by whether its departure point was inside the view (per seed only).\n")
    rws = []
    for arm, c in (("W", "A"), ("A", "A0"), ("B", "A0"), ("RW", "R0"), ("RW0", "R0"), ("RW", "RW0")):
        for seed in q.seeds(arm, c):
            st = q.steps(arm, seed, c)
            for b in FOVB:
                r = q(arm, seed, st[-1], c, b, "cut_in")
                if r is not None:
                    rws.append([f"{arm} - {c}", seed, f"k{st[-1]:02d}", b, int(r["n"]), f"{r['value']:.2f}", f"{r['ctrl_value']:.2f}", ci(r)])
    A(md(["comparison", "seed", "step", "stratum", "tokens", "arm cut-inside %", "control %", "diff"], rws) if rws else "No snapshot of these arms yet.\n")

    A("## Registered verdicts (computed, not judged)\n")
    A("Turn line (prereg): on the > 20 deg off-road rate, arm minus control at the latest snapshot. **closing** = diff <= -0.5 pp, CI upper bound < 0 and the mean of the last three "
      "snapshots below the first three; **flat** = |diff| < 0.3 pp, CI contains 0, vision displacement >= 1% (A / B: masked-memory drop >= 0.2 EPDMS); **not measured** = displacement "
      "< 0.5%, or (A / B) masked-memory drop < 0.2 and the Stage-0 probe unmoved (its paired CI against V contains 0); otherwise undetermined. W (amendment 1): on the > 45 deg "
      "cut-inside rate, **seeing the curb helps** = diff <= -1.0 pp and CI upper bound < 0; **not used** = |diff| < 0.5 pp, CI contains 0 and the masked-side-view drop < 0.2. "
      "Arm R (amendment 7): the same line without the trend clause (one checkpoint); displacement and Stage-0 probe are those of the bank's source checkpoint, the masked-memory "
      "drop is the R arm's own. A2 / B2: the line of A / B, the snapshot series continued. "
      "A line whose inputs are missing is `pending`; a verdict before the final checkpoint is provisional. Guardrail: CI lower bound of the full and the < 5 deg EPDMS difference >= -0.3.\n")
    rws, opp = [], []
    for arm, c in PRIMARY:
        V = {s: verdict(q, E, PR, arm, c, s) for s in q.seeds(arm, c)}
        for s, v in V.items():
            dr = ", ".join(f"{o} {'n/a' if x is None else format(x, '+.2f')}" for o, x in v["drop"].items()) or "n/a"
            rws.append([f"{arm} - {c}", s, f"k{v['step']:02d}" + ("" if v["final"] else " (provisional)"), v["quantity"], ci(v["row"]),
                        "n/a (one checkpoint)" if v["one"] else "n/a (< 6 snapshots)" if v["trend_down"] is None else f"{'down' if v['trend_down'] else 'not down'} ({v['first3']:+.2f} -> {v['last3']:+.2f})",
                        "n/a" if v["disp"] is None else f"{100 * v['disp']:.2f}%", dr, {None: "not run", True: "moved", False: "unmoved"}[v["probe_moved"]], f"**{v['line']}**",
                        ("pass" if v["guard"] else "**fail: gain counts as traded**") + f" ({ci(v['g'][0])}; {ci(v['g'][1])})"])
        if "0" in V and "1" in V and V["0"]["d"] * V["1"]["d"] < 0:
            opp.append(f"{arm} - {c} ({V['0']['quantity']}: s0 {V['0']['d']:+.2f}, s1 {V['1']['d']:+.2f})")
    A(md(["comparison", "seed", "snapshot", "quantity", "diff", "trend, first 3 -> last 3", "vision disp.", "option drop (EPDMS)", "Stage-0 probe", "line", "guardrail (EPDMS all; < 5)"], rws)
      if rws else "No snapshot of a trained-vision arm yet.\n")
    A("Seeds pointing in opposite directions on the line's quantity (the conclusion drops one grade): " + ("; ".join(opp) if opp else "none") + ".\n")

    A("### No-regression rule (amendment 2) and what moved\n")
    A("Cells: EPDMS and the nine sub-metrics x (all + four turn buckets) on navtest at the latest snapshot, plus navhard (combined, stage 1, stage 2) and HUGSIM where read. "
      "`down` = CI entirely below 0, `up` = entirely above 0. An arm is a candidate against a control only with no `down` cell; boards not read yet are listed as missing. "
      "With 50 navtest cells at 95%, about 1.25 cells fall on each side by chance when nothing differs.\n")
    rws = []
    for arm in STEPS:
        for c in CTRL[arm]:
            sd = q.seeds(arm, c)
            if not sd:
                continue
            seed, s = sd[0], q.steps(arm, sd[0], c)[-1]
            cl = cells(q, arm, c, seed, s)
            got = [(n, r) for n, r in cl if r is not None and np.isfinite(r["diff"])]
            up, dn = [f"{n} {ci(r)}" for n, r in got if r["ci_lo"] > 0], [f"{n} {ci(r)}" for n, r in got if r["ci_hi"] < 0]
            split = []
            for n, r in got:                                                    # the two seeds each exclude 0 on opposite sides
                if seed == "mean" and r["board"] == "navtest":
                    b, k = n.split(" ")[1], n.split(" ")[0]
                    a0, a1 = q(arm, 0, s, c, b, k), q(arm, 1, s, c, b, k)
                    if a0 is not None and a1 is not None and ((a0["ci_lo"] > 0 and a1["ci_hi"] < 0) or (a0["ci_hi"] < 0 and a1["ci_lo"] > 0)):
                        split.append(n)
            miss = sorted({n.split(" ")[0] for n, r in cl if r is None and not n[0].isupper()})
            rws.append([f"{arm} - {c}", seed, f"k{s:02d}" + ("" if s == STEPS[arm] else " (provisional)"), "; ".join(up) or "none", "; ".join(dn) or "none", len(got) - len(up) - len(dn),
                        ("candidate" if not dn else "**not a candidate**") + (f" (missing: {', '.join(miss)})" if miss else ""), "; ".join(split) or "none"])
    A(md(["comparison", "seed", "snapshot", "up", "down", "not moved (cells)", "no-regression", "seeds significant in opposite directions"], rws))

    A("### Flips: saved N, hurt M (token counts)\n")
    A("Per comparison, failure class and bucket: **fixed** = tokens the control fails and the arm passes, **broken** = tokens the control passes and the arm fails, "
      "net = fixed - broken, `ctrl fail` / `arm fail` = the failing token counts. Failure = the rate columns' definitions (off-road DAC < 1; cut-inside / under-turn on "
      "off-road tokens; NC < 1 by the class of decision 196, A1 stopped vehicle ahead, A2 moving lead; TTC < 1). `latest common` = the latest snapshot every listed "
      "seed has (control at its nearest registered snapshot), `final` = the registered last step. `sum` adds the seeds' counts. Same seed pairing as everywhere else; "
      "file `flips.csv`.\n")
    fp = OUT / "out" / "flips.csv"
    if fp.exists():
        import pandas as pd
        FLD = pd.read_csv(fp, dtype={"seed": str})
        for (arm, c), g in FLD.groupby(["arm", "control"], sort=False):
            A(f"**{arm} - {c}**\n")
            rw = []
            for (k, b), h in g.groupby(["cls", "bucket"], sort=False):
                cells_ = []
                for pick in ("latest common", "final"):
                    x = h[h.pick == pick]
                    cells_.append("; ".join(f"s{r.seed} k{r.step:02d}: +{r.fixed} / -{r.broken} = {r.net:+d}" if r.seed != "sum" else f"**sum k{r.step:02d}: +{r.fixed} / -{r.broken} = {r.net:+d}** ({r.ctrl_fail} -> {r.arm_fail})"
                                            for r in x.itertuples()) or "-")
                rw.append([k, b, *cells_])
            A(md(["class", "bucket", "latest common (fixed / broken = net)", "final"], rw))
    else:
        A("Not built yet.\n")
    A("### Moved up / moved down / did not move (every metric x bucket x board)\n")
    A("Per comparison at the latest snapshot: every cell of the long table, i.e. the scores, the rates (off-road, cut-inside, under-turn, NC classes, TTC-only), the two arc "
      "ratios, each over all + the four turn buckets + > 20 deg, then the boards read. **up** = CI entirely above 0, **down** = CI entirely below 0, **did not move** = CI contains 0 "
      "(cells grouped by metric, buckets in brackets). For rates and arc ratios up is not better: read the sign against the metric.\n")
    for arm in STEPS:
        for c in CTRL[arm]:
            sd = q.seeds(arm, c)
            if not sd:
                continue
            seed, s = sd[0], q.steps(arm, sd[0], c)[-1]
            cl = cells(q, arm, c, seed, s)
            for k in [*RATES, "arc_ratio_log", "arc_ratio_ctrl"]:
                cl += [(f"{k} {b}", q(arm, seed, s, c, b, k)) for b in BUCKETS]
            cl += [(f"{k} >20", q(arm, seed, s, c, ">20", k)) for k in SC]
            got = [(n, r) for n, r in cl if r is not None and np.isfinite(r["diff"])]
            grp = {"up": {}, "down": {}, "did not move": {}}
            for n, r in got:
                k, b = n.split(" ", 1)
                g = "up" if r["ci_lo"] > 0 else "down" if r["ci_hi"] < 0 else "did not move"
                grp[g].setdefault(k, []).append(b if g == "did not move" else f"{b} {ci(r)}")
            A(f"**{arm} - {c}**, seed {seed}, k{s:02d}" + ("" if s == STEPS[arm] else " (provisional)") + f", {len(got)} cells\n")
            for g, d in grp.items():
                A(f"- {g} ({sum(len(v) for v in d.values())}): " + ("; ".join(f"{k} [{', '.join(v)}]" for k, v in d.items()) or "none"))
            A("")
    A("## Prereg read 5: arm C, straight-road ADE on comma1M native cameras against the shipped model (decision 137's forgetting read)\n")
    nt = OUT.parent / "native" / "table.csv"
    if nt.exists():
        import pandas as pd
        T = pd.read_csv(nt)
        A("30 straight 10 s windows of `experiments/op_fov` (native rig, road 910 / wide 455 px), ADE5 = mean plan-to-logged-path distance over the first 5 s, "
          "`experiments/op_fov/scripts/fov_report.window_metrics` unchanged; by-segment cluster bootstrap. Ego-adapter bias from comma motion "
          "(`scripts/vt_native.py` header). `value` = metres, `ratio` = tag / shipped per window (decision 137: x2.6 for the fine-tuned arms there), "
          "`diff` = C minus the F0 snapshot nearest in steps.\n")
        A(md(["metric", "tag", "vs", "kind", "windows", "mean [lo, hi]"],
             [[r.metric, r.tag, r.ref if isinstance(r.ref, str) else "-", r.kind, r.nwin, f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]"]
              for _, r in T.sort_values(["metric", "kind", "tag"]).iterrows()]))
    else:
        A("Not run yet: `.venv/bin/python experiments/vis_train/scripts/vt_native.py submit` on the box (cached per tag; rerun after each new C snapshot).\n")
    A("## Not produced\n")
    A("- HUGSIM for A0 / A / B / W: `jevdrive.bench` has no serving path for the branch encoder (amendment 2 point 3). Arm C goes through the parity path as the tag "
      "`VTCP2-s0` (`vt_native.py export`); F / F0 are queued in their chains; their boards appear in the table above when read.\n"
      "- HUGSIM / WOD for arm R and for A2 / B2: memory arms, no serving path (amendment 7 point 11); their `hugsim` cells are listed as missing in the no-regression table.\n"
      "- navhard reference `SH30-F-s{0,1}` on protocol W frames appears in the board table once `bench run --model SH30-F-s0 SH30-F-s1 --bench navhard` has been run.\n")
    return "\n".join(P)


def first_table(R, E):
    """Per arm against each control at the first snapshot both have at the same step (else the arm's first, control at its nearest)."""
    q, rws = Q(R), []
    for arm in STEPS:
        for c in CTRL[arm]:
            for seed in q.seeds(arm, c):
                st = q.steps(arm, seed, c)
                ex = [s for s in st if int(q(arm, seed, s, c, "all", "EPDMS")["ctrl_step"]) in (s, 0)]
                s = (ex or st)[0]
                cs = int(q(arm, seed, s, c, "all", "EPDMS")["ctrl_step"])
                g = lambda b, k: ci(q(arm, seed, s, c, b, k))  # noqa: E731
                dv = disp_at(E, arm, seed, s)
                rws.append([f"{arm} - {c}", seed, f"k{s:02d}", f"k{cs:02d}", g(">45", "offroad"), g(">45", "cut_in"), g(">45", "under"), g(">20", "offroad"), g("all", "EPDMS"),
                            "n/a" if dv is None else f"{100 * dv:.2f}%"])
    return md(["comparison", "seed", "step", "ctrl step", "off-road > 45", "cut-inside > 45", "under-turn > 45", "off-road > 20", "EPDMS all", "vision disp."], rws) if rws else "No comparison yet.\n"


# ---------------------------------------------------------------- commands
def cmd_box(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("vis_train", "reads", config=vars(a)) as run:
        run.use_split(splits.load("navsim/navtest"))
        if not a.no_replay:
            replay(run)
        build(run)


def cmd_check(a):
    """CB against jevdrive.stats.paired on real cells of two cached models."""
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench import tables as T
    tab = np.load(TAB)
    names, sgn = tab["names"].astype(str), np.sign(tab["fut"][:, -1, 2].astype(float))
    M, _ = masks(names, T.navtest_strata())
    fs = sorted((OUT / "tok").glob("*.parquet"))[:2]
    d = [pd.read_parquet(f).set_index("token").reindex(names) for f in fs]
    xa, xb = (feats(x, sgn) for x in d)
    cb, worst = CB(d[0]["log"].astype(str).to_numpy()), 0.0
    for b in BUCKETS:
        mu, lo, hi, n = cb.mean(xa - xb, M[b])
        for j, k in enumerate(METRICS):
            r = stats.paired(xa[M[b], j], xb[M[b], j], cb.logs[M[b]])
            worst = max(worst, *(abs(u - v) for u, v in ((mu[j], r["mean"]), (lo[j], r["lo"]), (hi[j], r["hi"]))), abs(n[j] - r["n"]))
    print(f"{fs[0].stem} - {fs[1].stem}: {len(BUCKETS) * len(METRICS)} cells, max |CB - jevdrive.stats.paired| over mean / lo / hi / n = {worst:.2e}")
    assert worst < 1e-9


def cmd_first(a):
    import pandas as pd
    src = Path(a.src)
    print(first_table(pd.read_parquet(src / "reads.parquet"), pd.read_csv(src / "disp.csv") if (src / "disp.csv").stat().st_size > 2 else pd.DataFrame()))


def cmd_sync(a):
    """Mac: run `box` as one pool CPU job on the box, wait for it, copy the small outputs into results/."""
    ld = f"{OUT}/pool/{time.strftime('%Y%m%d-%H%M%S')}"
    sub = (f"cd ~/data/jev-drive && .venv/bin/python -m jevdrive.cl submit --owner vis_train --name vt-read --vram 0.5 --cpu {a.cpu} --ram 32 --log-dir {ld} -- "
           f"{D}/envs/navsim2/bin/python experiments/vis_train/scripts/vt_read.py box" + (" --no-replay" if a.no_replay else ""))
    if not a.copy_only:
        print(subprocess.run([*SSH, sub], check=True, capture_output=True, text=True).stdout.strip(), ld)
        wait = f"until [ -f {ld}/DONE ] || [ -f {ld}/ERROR ]; do sleep 15; done; [ -f {ld}/DONE ] || {{ tail -30 {ld}/log.txt; exit 1; }}"
        subprocess.run([*SSH, wait], check=True)
    RES.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", "-o", "ControlPath=none", *(f"autodl:{OUT}/out/{f}" for f in ("reads.parquet", "reads.md", "disp.csv", "probe.csv", "flips.csv", "trend.png", "trend.pdf")), str(RES)], check=True)
    print(f"copied to {RES}")


def cmd_probe(a):
    """Box (.venv): the Stage-0 probe chain of --tags as pool jobs. Rerunning with the same --name resumes (finished stages are skipped)."""
    od, py, cl = OUT / "probe" / a.name, D / "envs/op-train/bin/python", [sys.executable, "-m", "jevdrive.cl", "submit", "--owner", "vis_train"]
    od.mkdir(parents=True, exist_ok=True)
    S = "experiments/vis_train/scripts"
    datas = ["lb_navtest", "navtrain_full.s2of12", "navtrain_full.s3of12", "navtrain_full.s4of12"]
    assert a.gated or all((RUNS / t / "ckpt-final.pt").exists() for t in a.tags), "a tag has no checkpoint (--gated queues the chain on the checkpoints)"
    lowp = ["--priority", str(a.priority)]

    jf = od / "jobs.json"
    jobs = json.loads(jf.read_text()) if jf.exists() else {}
    live = {ln.split()[0] for ln in subprocess.run([*cl[:3], "queue"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
            if len(ln.split()) > 1 and ln.split()[1] in ("queued", "running", "inbox")}

    def sub(nm, done, after, *args):
        if done:
            return None
        if jobs.get(nm) in live:
            return jobs[nm]
        o = subprocess.run([*cl, "--name", nm, "--log-dir", str(od / "pool" / nm), *(["--after", ",".join(after)] if after else []), *args], check=True, capture_output=True, text=True, cwd=REPO)
        jobs[nm] = o.stdout.strip().split()[-1]
        jf.write_text(json.dumps(jobs))
        print(nm, jobs[nm])
        return jobs[nm]
    tj = [sub(f"vt-probe-tok-{t}", all((D / "runs/vis_train/tokens" / t / f"{d}.npy").exists() for d in datas), [], "--vram", "16", "--cpu", "4", "--ram", "24", *lowp, "--when-exists", str(RUNS / t / "ckpt-final.pt"), "--",
              str(py), f"{S}/vt.py", "tokens", "--tag", t) for t in a.tags]
    dj = sub(f"vt-probe-dec-{a.name}", (od / "decoder_poses.npz").exists(), [j for j in tj if j], "--vram", "12", "--cpu", "4", "--ram", "48", *lowp, "--",
             str(py), f"{S}/vt_read.py", "probe-decode", "--name", a.name, "--tags", *a.tags)
    sub(f"vt-probe-score-{a.name}", (od / "score_t20.csv").exists(), [dj] if dj else [], "--vram", "0.5", "--cpu", str(a.cpu), "--ram", "48", *lowp, "--",
        str(D / "envs/navsim2/bin/python"), "experiments/op_probe/scripts/opb_score.py", "--poses", str(od / "decoder_poses.npz"), "--keys", "V", *a.tags,
        "--tokens", str(od / "tokens_t20.txt"), "--out", str(od / "score_t20.csv"))


def cmd_probe_decode(a):
    """Pool job (envs/op-train, one card): rep.py's Stage-0 decoder, unchanged, on V and on the dumped branch tokens (`vt.py tokens`)."""
    import rep
    tk, base = D / "runs/vis_train/tokens", rep._load_tok
    rep._load_tok = lambda kind, data: (np.load(tk / kind / f"{data}.names.npy"), np.load(tk / kind / f"{data}.npy", mmap_mode="r")) if kind in a.tags else base(kind, data)
    rep.fit_pca = lambda *x, **k: {"mean": np.zeros(1), "comp": np.zeros(1), "explained": float("nan")}        # VJ21's PCA: not used by these sources
    rep.OUT, rep.ARMS = OUT / "probe" / a.name, {"V": ("V",)} | {t: (t,) for t in a.tags}
    rep.cmd_decode(argparse.Namespace(arms=list(rep.ARMS), junction_arms=[], steps=4000, batch=512, lam_hinge=10.0, hinge_margin=0.3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("box")
    p.add_argument("--no-replay", action="store_true", help="rebuild the tables from the cached token tables only")
    p = sp.add_parser("sync")
    p.add_argument("--cpu", type=int, default=16)
    p.add_argument("--no-replay", action="store_true")
    p.add_argument("--copy-only", action="store_true", help="copy the last build without running anything")
    p = sp.add_parser("first")
    p.add_argument("--src", default=str(RES))
    sp.add_parser("check")
    p = sp.add_parser("probe")
    p.add_argument("--tags", nargs="+", required=True, help="checkpoint tags with a branch or their own encoder (VT-A / B / C / W-s<seed>[-k<NN>])")
    p.add_argument("--name", required=True, help="probe run name ($OUT/probe/<name>)")
    p.add_argument("--cpu", type=int, default=24)
    p.add_argument("--gated", action="store_true", help="queue the chain before the checkpoints exist (the token jobs wait for them)")
    p.add_argument("--priority", type=int, default=2)
    p = sp.add_parser("probe-decode")
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    {"box": cmd_box, "sync": cmd_sync, "first": cmd_first, "check": cmd_check, "probe": cmd_probe, "probe-decode": cmd_probe_decode}[a.cmd](a)
