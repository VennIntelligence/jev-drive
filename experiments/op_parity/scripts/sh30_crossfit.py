#!/usr/bin/env python
"""op_parity sh30-crossfit: held-out (cross-fitted) SH30 plans and simulator labels for the navtrain turn tokens.

Pre-registration: plans/2026-10-08-sh30-crossfit-prereg.md. K = 5 log-disjoint folds of navtrain; fold model j = the SH30 recipe (decision 170)
trained on op-parity-full-train minus the logs of fold j; every navtrain turn token (|dyaw| >= 20 deg, 28 323) is planned by the model that never
saw its log. Restartable commands, each skips what is finished:
  folds               register the fold splits (navsim/op-parity-cf5f{j}-{train,dev}, unit log) and check them
  plans   --fold j    pool run: plans + nav-export of fold model j on the 12 navtrain_full shards (stages as nt_labels.py)
  assemble            held-out plan of every turn token -> $OUT/labels/poses.npz (key h) + tokens.txt + fold_of_token.csv
  score               score-poses --traffic non_reactive --mcache v2_navtrain on key h -> labels/score.csv
  gate                fold models on navtest: the recipe gate of the pre-registration
  diag                held-out vs in-sample vs navtest failure rates, telescoping decomposition, log-clustered CIs, figure
  cands               33-candidate family (turn_ceiling.candidates, unchanged) around the held-out plans -> labels/poses_f33.npz
  cscore --stage S    score stage t300 / all of the family (family chosen by the measured cost, see cands)
  ceiling             privileged ceiling table on navtrain (decision 178's table, one plan per token)
Outputs: $DATA_DIR/runs/op_parity/sh30_crossfit/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevdrive.common import data_dir

K = 5
NSH = 12
REPO = Path(__file__).resolve().parents[3]
SPLIT = "navsim/op-parity-cf5f{j}"
SHARDS = [f"navtrain_full.s{i}of{NSH}" for i in range(NSH)]
SH30 = ("SH30-F-s0", "SH30-F-s1")
FIG = REPO / "experiments/op_parity/figs/sh30_crossfit"
RES = REPO / "experiments/op_parity/results/sh30_crossfit"
NB = 10_000


def model(j: int) -> str:
    return f"CF{K}f{j}-F-s0"


def O() -> Path:
    return data_dir() / "runs/op_parity/sh30_crossfit"


def L() -> Path:
    return O() / "labels"


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def tabs():
    """(names, logs, shard) of the 103 288 navtrain_full rows, in shard order."""
    n, l, s = [], [], []
    for i, d in enumerate(SHARDS):
        z = np.load(data_dir() / f"runs/op_parity/cache/{d}/tab.npz")
        n += z["names"].tolist()
        l += z["log"].tolist()
        s += [i] * len(z["names"])
    return np.array(n), np.array(l), np.array(s)


def fold_of_log(log: str) -> int:
    return int(hashlib.sha256(f"cf{K}|{log}".encode()).hexdigest(), 16) % K


# ---------------------------------------------------------------- folds
def cmd_folds(a):
    from jevdrive.data import splits
    names, logs, _ = tabs()
    tr0, dv0 = splits.load("navsim/op-parity-full-train"), splits.load("navsim/op-parity-full-dev")
    dev_logs = set(logs[dv0.mask(names)])
    assert not dev_logs & set(logs[tr0.mask(names)]), "full-train and full-dev share a log"
    ul = sorted(set(logs.tolist()))
    fold = {l: fold_of_log(l) for l in ul}
    tok = __import__("nt_labels").turn_tokens()
    isturn = np.isin(names, tok)
    origin = (f"navsim navtrain logs ({len(ul)}); fold(log) = int(sha256('cf{K}|' + log), 16) % {K}; dev = the logs of fold j (every token of them is held out of "
              f"fold model j); train = all other logs except the {len(dev_logs)} dev logs of navsim/op-parity-full (sha256(log) % 50 == 0)")
    rows = []
    for j in range(K):
        dv = [l for l in ul if fold[l] == j]
        tr = [l for l in ul if fold[l] != j and l not in dev_logs]
        sd = splits.define("navsim", f"op-parity-cf{K}f{j}-dev", dv, "log", origin, used_by=["experiments/op_parity"], status="frozen",
                           notes=f"sh30-crossfit fold {j}: held-out logs of fold model {model(j)}")
        st = splits.define("navsim", f"op-parity-cf{K}f{j}-train", tr, "log", origin, used_by=["experiments/op_parity"], status="frozen",
                           notes=f"sh30-crossfit fold {j}: training logs of fold model {model(j)}")
        splits.check_disjoint(sd, st)
        mt, md = st.mask(logs), sd.mask(logs)
        assert np.array_equal(mt, tr0.mask(names) & ~md), f"fold {j}: train tokens != full-train minus the fold's tokens"
        rows.append(dict(fold=j, logs_dev=len(dv), logs_train=len(tr), tokens_train=int(mt.sum()), tokens_dev=int(md.sum()), turn_dev=int((md & isturn).sum()),
                         train_ids=st.id, dev_ids=sd.id))
    assert sum(r["tokens_dev"] for r in rows) == len(names) and sum(r["turn_dev"] for r in rows) == len(tok) == 28323
    import pandas as pd
    df = pd.DataFrame(rows)
    (O()).mkdir(parents=True, exist_ok=True)
    df.to_csv(O() / "folds.csv", index=False)
    print(df.to_string(index=False))


# ---------------------------------------------------------------- plans (pool run of one fold; stages as nt_labels.py)
def cmd_plans_shard(a):
    import nt_labels as NL
    NL.MODELS = (a.model,)
    (NL.cmd_plans if a.what == "plans" else NL.cmd_export)(argparse.Namespace(shard=a.shard))


def cmd_plans(a):
    from jevdrive.bench import runner as R
    import nt_labels as NL
    m, me = model(a.fold), str(Path(__file__).resolve())
    NL.MODELS = (m,)
    rd = O() / "plans" / m
    S = []
    for i in range(NSH):
        if not NL.N_pred(i, m).exists():
            S.append(R.Stage(f"pl{i}", [R.py("op-train"), me, "plans-shard", "--what", "plans", "--model", m, "--shard", str(i)],
                             done=str(NL.ol(i, "plans", f"{NL.stem(m)}.npz")), vram=16, cpu=6, ram=24, tries=2))
            S.append(R.Stage(f"ex{i}", [R.py("jev"), me, "plans-shard", "--what", "export", "--model", m, "--shard", str(i)],
                             done=str(NL.N_pred(i, m)), vram=0.5, cpu=2, ram=8, after=[f"pl{i}"], tries=2))
    if S:
        S.append(R.Stage("fin", ["touch", str(rd / "DONE")], done=str(rd / "DONE"), vram=0.5, cpu=1, ram=1, after=[s.name for s in S]))
        R.submit(rd, f"cf-plans-{a.fold}", S, owner="op_parity", priority=-1.0)
        if not R.wait([rd], poll_s=60, quiet=True) and not all(NL.N_pred(i, m).exists() for i in range(NSH)):
            raise SystemExit(f"plan / export stages of {m} failed; re-run the same command to resume")


# ---------------------------------------------------------------- assemble + score
def cmd_assemble(a):
    import nt_labels as NL
    from jevdrive.data import splits
    from jevdrive.run import Run
    out = L() / "poses.npz"
    if out.exists() and not a.force:
        print("poses.npz exists")
        return
    toks = __import__("nt_labels").turn_tokens()
    want = {t: j for j, t in enumerate(toks)}
    with Run("op_parity", "sh30_crossfit/assemble", config=vars(a)) as run:
        dev = [splits.load(SPLIT.format(j=j) + "-dev") for j in range(K)]
        for d in dev:
            run.use_split(d)
        P = np.full((len(toks), 8, 3), np.nan, np.float32)
        fut = np.full((len(toks), 8, 3), np.nan, np.float32)
        fold = np.full(len(toks), -1)
        logs_of = [""] * len(toks)
        for i in range(NSH):
            tab = np.load(data_dir() / f"runs/op_parity/cache/{SHARDS[i]}/tab.npz")
            nm, lg = tab["names"].tolist(), tab["log"]
            fo = np.full(len(nm), -1)
            for j in range(K):
                fo[dev[j].mask(lg)] = j
            assert (fo >= 0).all()
            for j in range(K):
                z = np.load(NL.N_pred(i, model(j)))
                assert z["tokens"].tolist() == nm, f"shard {i} fold {j}: export tokens != token cache rows"
                for r, t in enumerate(nm):
                    k = want.get(t)
                    if k is not None and fo[r] == j:
                        P[k], fut[k], fold[k], logs_of[k] = z["poses"][r], tab["fut"][r], j, lg[r]
        assert (fold >= 0).all() and not np.isnan(P[:, 0, 0]).any(), "turn tokens without a held-out plan"
        ade = float(np.median(np.linalg.norm(P[:, :, :2] - fut[:, :, :2], axis=-1).mean(1)))
        L().mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".poses.{os.getpid()}.npz")
        np.savez(tmp, tokens=np.array(toks), h=P)
        os.replace(tmp, out)
        atomic(L() / "tokens.txt", "\n".join(toks) + "\n")
        import pandas as pd
        pd.DataFrame(dict(token=toks, fold=fold, log=logs_of)).to_csv(L() / "fold_of_token.csv", index=False)
        run.summary.update(n=len(toks), median_ade_to_log_m=ade, per_fold=np.bincount(fold, minlength=K).tolist())
        run.info("assembled %d held-out plans, median ADE to the logged future %.3f m, per fold %s", len(toks), ade, np.bincount(fold, minlength=K).tolist())


def score_poses(poses: Path, out: Path, keys: list, toks: list, wait=True):
    from jevdrive.bench import poses as BP, runner as R
    if out.exists():
        print(out, "exists")
        return
    d = BP.submit(poses, out, keys, toks, owner="op_parity", traffic="non_reactive", mcache="v2_navtrain", priority=-1.0)
    if wait and not R.wait([d], poll_s=60, quiet=True):
        raise SystemExit(f"score-poses {d} failed; re-run the same command to resume")


def cmd_score(a):
    from jevdrive.bench import poses as BP
    toks = BP.read_tokens(L() / "tokens.txt")
    score_poses(L() / "poses.npz", L() / "score.csv", ["h"], toks)


# ---------------------------------------------------------------- navtest side (fold models vs SH30)
def navtest_units(spec: str):
    from jevdrive.bench import tables as T
    u, _ = T.load("navtest", spec)
    assert u is not None, f"no navtest units for {spec}"
    return u


def cmd_gate(a):
    """Fold models on navtest: EPDMS (with EC, bench) vs SH30; pass rule of the pre-registration."""
    from jevdrive import stats
    from jevdrive.bench import tables as T
    from jevdrive.run import Run
    with Run("op_parity", "sh30_crossfit/gate", config=vars(a)) as run:
        U = {m: navtest_units(m) for m in [*(model(j) for j in range(K)), *SH30, "P2H10-F-s0", "P2H10-F-s1"]}
        tok = U[SH30[0]].index
        epd = {m: 100 * u.loc[tok, "score"].to_numpy(float) for m, u in U.items()}
        lg = U[SH30[0]].loc[tok, "log"].astype(str).to_numpy()
        sh, p2 = (epd[SH30[0]] + epd[SH30[1]]) / 2, (epd["P2H10-F-s0"] + epd["P2H10-F-s1"]) / 2
        rows = [dict(model=m, epdms=float(e.mean()), vs_sh30=stats.fmt(stats.paired(e, sh, groups=lg), ".2f")) for m, e in epd.items()]
        fm = np.array([epd[model(j)].mean() for j in range(K)])
        mean_f = float(fm.mean())
        ok = bool(mean_f >= 89.05 and (fm >= 88.67).all())
        res = dict(fold_epdms=fm.tolist(), mean_fold=mean_f, sh30_mean=float(sh.mean()), p2h10_mean=float(p2.mean()), gate_pass=ok)
        (O() / "gate.json").write_text(json.dumps(res, indent=1))
        import pandas as pd
        pd.DataFrame(rows).to_csv(O() / "gate_navtest.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        print(json.dumps(res))
        run.summary.update(res)
    raise SystemExit(0 if ok else 2)


# ---------------------------------------------------------------- diagnostic
def _pct(x, d=2):
    return f"{100 * x:.{d}f}"


def cmd_diag(a):
    import pandas as pd
    from jevdrive.bench import tables as T
    from jevdrive.run import Run
    import turn_ceiling as TC
    import nt_labels as NL
    with Run("op_parity", "sh30_crossfit/diag", seed=0, config=vars(a)) as run:
        # navtrain side: B = SH30 in-sample (nt_cache, mean of 2 seeds), A = held-out fold plan
        lab = pd.read_csv(NL.L() / "labels_turn.csv.gz", index_col="token")
        sc = pd.read_csv(L() / "score.csv").set_index("token")
        fo = pd.read_csv(L() / "fold_of_token.csv").set_index("token")
        toks = lab.index.to_numpy(str)
        assert set(toks) == set(sc.index) and len(sc) == len(toks), "held-out score.csv != nt_cache turn tokens"
        sc, fo = sc.loc[toks], fo.loc[toks]
        dy = np.abs(lab.dyaw.to_numpy(float))
        nt = dict(
            dac_B=((lab.s0_id_DAC.to_numpy() == 0).astype(float) + (lab.s1_id_DAC.to_numpy() == 0)) / 2,
            nc_B=((lab.s0_id_NC.to_numpy() == 0).astype(float) + (lab.s1_id_NC.to_numpy() == 0)) / 2,
            epd_B=100 * (lab.s0_id_score.to_numpy() + lab.s1_id_score.to_numpy()) / 2,
            dac_A=(sc.drivable_area_compliance.to_numpy() == 0).astype(float), nc_A=(sc.no_at_fault_collisions.to_numpy() == 0).astype(float),
            epd_A=100 * sc.score.to_numpy(float))
        nt_log, nt_fold, nt_big = fo["log"].to_numpy(str), fo["fold"].to_numpy(int), dy >= 45
        # navtest side: C = SH30 (mean of 2 seeds), D = fold models (mean of 5 per token)
        tok_t, dy_t = TC.bucket_tokens()
        iD, iN = T.TERMS.index("DAC"), T.TERMS.index("NC")

        def side(specs):
            X = [navtest_units(m).loc[tok_t, T.TERMS].to_numpy(float) for m in specs]
            return (np.mean([(x[:, iD] == 0) for x in X], 0), np.mean([(x[:, iN] == 0) for x in X], 0),
                    np.mean([100 * TC.noec(x) for x in X], 0)), X
        (dac_C, nc_C, epd_C), _ = side(SH30)
        (dac_D, nc_D, epd_D), XD = side([model(j) for j in range(K)])
        tt_log = navtest_units(SH30[0]).loc[tok_t, "log"].astype(str).to_numpy()
        tt_big = np.abs(dy_t) >= 45
        fold_D = [(np.mean(x[:, iD] == 0), np.mean(100 * TC.noec(x))) for x in XD]

        import pandas as pd
        cu = pd.factorize(nt_log)
        tu = pd.factorize(tt_log)
        rng = np.random.default_rng(0)
        ia = rng.integers(len(cu[1]), size=(NB, len(cu[1])))
        ib = rng.integers(len(tu[1]), size=(NB, len(tu[1])))

        def csum(v, m, codes, n):
            return np.bincount(codes, np.where(m, v, 0.0), n), np.bincount(codes, m.astype(float), n)

        def boot(vN_B, vN_A, vT_C, vT_D, mN, mT):
            """B, A on the navtrain subset mN (cluster = log), C, D on the navtest subset mT; ratio-of-sums bootstrap, independent resamples of the two datasets."""
            out = {}
            sB, cB = csum(vN_B, mN, cu[0], len(cu[1]))
            sA, _ = csum(vN_A, mN, cu[0], len(cu[1]))
            sC, cC = csum(vT_C, mT, tu[0], len(tu[1]))
            sD, _ = csum(vT_D, mT, tu[0], len(tu[1]))
            with np.errstate(invalid="ignore", divide="ignore"):
                b = {"B": sB[ia].sum(1) / cB[ia].sum(1), "A": sA[ia].sum(1) / cB[ia].sum(1),
                     "C": sC[ib].sum(1) / cC[ib].sum(1), "D": sD[ib].sum(1) / cC[ib].sum(1)}
            pt = {"B": sB.sum() / cB.sum(), "A": sA.sum() / cB.sum(), "C": sC.sum() / cC.sum(), "D": sD.sum() / cC.sum()}
            comp = {"mem": ("A", "B", 1), "dom": ("D", "A", 1), "mod": ("C", "D", 1), "gap": ("C", "B", 1)}
            for k, (x, y, _) in comp.items():
                d = b[x] - b[y]
                out[k] = (pt[x] - pt[y], *np.nanquantile(d, [0.025, 0.975]))
            for k in "BADC":
                out[k] = (pt[k], *np.nanquantile(b[k], [0.025, 0.975]))
            out["share_mem"] = (pt["A"] - pt["B"]) / max(pt["A"] - pt["B"] + pt["D"] - pt["A"], 1e-12) if (pt["D"] - pt["B"]) != 0 else np.nan
            return out

        buckets = [("turn >= 20", np.ones(len(dy), bool), np.ones(len(dy_t), bool)),
                   ("20-45", ~nt_big, ~tt_big), (">= 45", nt_big, tt_big)]
        met = [("DAC fail %", "dac", 100, nt["dac_B"], nt["dac_A"], dac_C, dac_D), ("NC fail %", "nc", 100, nt["nc_B"], nt["nc_A"], nc_C, nc_D),
               ("no-EC EPDMS", "epd", 1, nt["epd_B"], nt["epd_A"], epd_C, epd_D)]
        rows, R = [], {}
        for bn, mN, mT in buckets:
            for mn, mk, sc_, vB, vA, vC, vD in met:
                r = boot(vB, vA, vC, vD, mN, mT)
                R[(bn, mk)] = r
                row = dict(bucket=bn, metric=mn, n_navtrain=int(mN.sum()), n_navtest=int(mT.sum()))
                for k in ("B", "A", "D", "C"):
                    row[k] = f"{sc_ * r[k][0]:.2f} [{sc_ * r[k][1]:.2f}, {sc_ * r[k][2]:.2f}]"
                for k in ("mem", "dom", "mod", "gap"):
                    row[k] = f"{sc_ * r[k][0]:+.2f} [{sc_ * r[k][1]:+.2f}, {sc_ * r[k][2]:+.2f}]"
                rows.append(row)
        T1 = pd.DataFrame(rows)
        T1.to_csv(O() / "diag_table.csv", index=False)

        # verdict: pre-registered rule on pooled DAC failure
        r = R[("turn >= 20", "dac")]
        mem, dom, mod, gap = r["mem"], r["dom"], r["mod"], r["gap"]
        mem_pos, dom_pos = mem[1] > 0, dom[1] > 0
        share = mem[0] / (mem[0] + dom[0]) if (mem[0] + dom[0]) > 0 else np.nan
        if mem_pos and dom_pos:
            verdict = "in-sample" if share >= 0.67 else ("navtrain easier" if share <= 0.33 else "both")
        elif mem_pos:
            verdict = "in-sample"
        elif dom_pos:
            verdict = "navtrain easier"
        else:
            verdict = "unresolved"
        reproduces = bool(dom[1] <= 0 <= dom[2] or abs(dom[0]) < 0.005)
        verdict_d = dict(verdict=verdict, share_mem=share, mem=mem, dom=dom, mod=mod, gap=gap, heldout_reproduces_navtest_level=reproduces,
                         weak_model_flag=bool(abs(mod[0]) > 0.25 * abs(gap[0])))
        (O() / "diag_verdict.json").write_text(json.dumps(verdict_d, indent=1, default=float))

        # per fold: held-out navtrain turns of fold j (model j) vs model j on navtest turns
        pf = []
        for j in range(K):
            m = nt_fold == j
            pf.append(dict(fold=j, n_turn=int(m.sum()), DAC_fail_heldout=_pct(nt["dac_A"][m].mean()), DAC_fail_insample_SH30=_pct(nt["dac_B"][m].mean()),
                           DAC_fail_model_on_navtest=_pct(fold_D[j][0]), EPDMS_heldout=f"{nt['epd_A'][m].mean():.2f}", EPDMS_insample_SH30=f"{nt['epd_B'][m].mean():.2f}",
                           EPDMS_model_on_navtest=f"{fold_D[j][1]:.2f}"))
        T2 = pd.DataFrame(pf)
        T2.to_csv(O() / "diag_perfold.csv", index=False)
        run.summary.update(verdict=verdict, reproduces=reproduces)
        print(T1.to_string(index=False))
        print(T2.to_string(index=False))
        print(json.dumps(verdict_d, indent=1, default=float))
        figure_diag(R, buckets)


def figure_diag(R, buckets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    lab = {"B": "SH30 in-sample\n(navtrain)", "A": "fold model held-out\n(navtrain)", "D": "fold models\n(navtest)", "C": "SH30\n(navtest)"}
    col = {"B": "#9ecae1", "A": "#3182bd", "D": "#fdae6b", "C": "#e6550d"}
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for a_, (mk, ttl, sc_) in zip(ax, [("dac", "DAC failure rate (%)", 100), ("nc", "NC failure rate (%)", 100), ("epd", "no-EC EPDMS", 1)]):
        for bi, (bn, _, _) in enumerate(buckets):
            for ki, k in enumerate("BADC"):
                v, lo, hi = (sc_ * x for x in R[(bn, mk)][k])
                a_.bar(bi + (ki - 1.5) * 0.2, v, 0.19, yerr=[[v - lo], [hi - v]], color=col[k], label=lab[k] if bi == 0 else None, capsize=2)
        a_.set_xticks(range(len(buckets)))
        a_.set_xticklabels([b[0] + " deg" for b in buckets])
        a_.set_title(ttl)
        if mk == "epd":
            a_.set_ylim(75, 95)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "diag.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- step 4: family around the held-out plans
def cmd_cands(a):
    import turn_ceiling as TC
    from jevdrive import cache
    z = np.load(L() / "poses.npz")
    toks, P = z["tokens"], z["h"]
    C = TC.candidates()
    out = L() / "poses_f33.npz"

    def build():
        d = {"tokens": toks}
        for i, (_, o, k, v) in enumerate(C):
            d[f"c{i:02d}"] = TC.transform(P, o, k, v)
        return d
    k = cache.key(dict(cands=C, ramp=TC.S_RAMP, kmax=TC.KAPPA_MAX), inputs=[L() / "poses.npz"], code=[TC.transform, TC.curv, TC.offset, TC.speed], version="sh30-crossfit")
    Z = cache.cached(out, k, build, force=a.force)
    assert np.array_equal(Z["c00"], P), "identity candidate != held-out plan"
    import nt_labels as NL
    import pandas as pd
    dy = np.abs(pd.read_csv(NL.L() / "labels_turn.csv.gz", index_col="token").loc[toks, "dyaw"].to_numpy(float))
    rng = np.random.default_rng(0)
    big = dy >= 45
    pick = rng.permutation(np.concatenate([rng.permutation(np.flatnonzero(m))[:150] for m in (~big, big)]))
    atomic(L() / "tokens_t300.txt", "\n".join(toks[pick]) + "\n")
    atomic(L() / "tokens_all.txt", "\n".join(toks) + "\n")
    print("family written;", len(C), "candidates;", len(toks), "tokens")


def fam_keys(fam: str) -> list:
    import turn_ceiling as TC
    F = TC.families()[fam]
    return [f"c{i:02d}" for i in F]


def cmd_cscore(a):
    from jevdrive.bench import poses as BP
    keys = fam_keys(a.family)
    toks = BP.read_tokens(L() / f"tokens_{a.stage}.txt")
    score_poses(L() / "poses_f33.npz", L() / f"cscore_{a.stage}_{a.family}.csv", keys, toks)


def cmd_cgate(a):
    """Stage-0 gate of step 4: sane ranges, c00 rows equal to the held-out `h` rows on the 300 tokens, measured cost, family chosen by the pre-registered ladder."""
    import pandas as pd
    from jevdrive.bench import poses as BP, runner as BR
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    import turn_ceiling as TC
    with Run("op_parity", "sh30_crossfit/cgate", config=vars(a)) as run:
        df = pd.read_csv(L() / "cscore_t300_F33.csv")
        cols = [SUBS[k] for k in TC.SUB8] + ["score"]
        num = df[cols].to_numpy(float)
        sane = bool(np.isfinite(num).all() and (num >= 0).all() and (num <= 1 + 1e-9).all())
        tok = BP.read_tokens(L() / "tokens_t300.txt")
        ref = pd.read_csv(L() / "score.csv")
        ref = ref[ref.key == "h"].set_index("token").loc[tok]
        g = df[df.key == "c00"].set_index("token").loc[tok]
        res = dict(n_tokens=len(tok), sane_ranges=sane)
        for c in cols:
            d = np.abs(g[c].to_numpy(float) - ref[c].to_numpy(float))
            res[c] = dict(max_abs=float(d.max()), n_diff=int((d > (TC.EP_TOL if c in (SUBS["EP"], "score") else 0)).sum()))
        ok = sane and all(v["n_diff"] == 0 for v in res.values() if isinstance(v, dict))
        keys = [f"c{i:02d}" for i in range(33)]
        rd = BR.bench_root("poses") / BP.run_key(str(L() / "poses_f33.npz"), keys, tok, "non_reactive", "v2_navtrain")
        sm = json.loads((rd / "summary.json").read_text())
        c, nk, n_all = sm["cost"], len(sm["keys"]), len((L() / "tokens.txt").read_text().split())
        opts = []
        for fam in ("F33", "F27", "F19"):
            nkf = len(fam_keys(fam))
            cs = c["load_s_per_token"] + c["union_s_per_token"] + c["diag_s_per_token"] * nkf / nk + c["pdm_s_per_token"] * (nkf + 1) / (nk + 1)
            opts.append(dict(family=fam, n_keys=nkf, core_s_per_token=cs, core_h=cs * n_all / 3600))
        rem = a.budget_core_h - a.spent_core_h
        pick = next((o for o in opts if o["core_h"] <= rem), None) or (opts[-1] if opts[-1]["core_h"] <= 1.5 * rem else None)
        res.update(cost=c, options=opts, budget_core_h=a.budget_core_h, spent_core_h_estimate=a.spent_core_h, remaining=rem,
                   chosen=None if pick is None else pick["family"], run_dir=str(rd))
        res["pass"] = bool(ok and pick is not None)
        (L() / "cgate.json").write_text(json.dumps(res, indent=1, default=float))
        if pick is not None:
            atomic(L() / "family_chosen.txt", pick["family"] + "\n")
        run.summary.update(gate=res["pass"], chosen=res["chosen"])
        print(json.dumps({k: v for k, v in res.items() if k != "cost"}, indent=1, default=float))
        if not res["pass"]:
            raise SystemExit("step-4 stage-0 gate FAILED (identity mismatch or no family fits the budget): see " + str(L() / "cgate.json"))


# ---------------------------------------------------------------- ceiling table
def cmd_ceiling(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    import turn_ceiling as TC
    import nt_labels as NL
    with Run("op_parity", "sh30_crossfit/ceiling", seed=0, config=vars(a)) as run:
        df = pd.read_csv(L() / f"cscore_all_{a.family}.csv")
        keys = sorted(df.key.unique())
        toks = (L() / "tokens.txt").read_text().split()
        assert len(df) == len(keys) * len(toks), "score CSV incomplete"
        ti = {t: i for i, t in enumerate(toks)}
        S = np.full((len(keys), len(toks)), np.nan)
        D = np.full((len(keys), len(toks)), np.nan)
        N = np.full((len(keys), len(toks)), np.nan)
        for kk, g in df.groupby("key"):
            r, c = g.token.map(ti).to_numpy(), keys.index(kk)
            S[c, r] = g.score.to_numpy(float)
            D[c, r] = g.drivable_area_compliance.to_numpy(float)
            N[c, r] = g.no_at_fault_collisions.to_numpy(float)
        assert np.isfinite(S).all()
        cid = [int(k[1:]) for k in keys]
        lab = pd.read_csv(NL.L() / "labels_turn.csv.gz", index_col="token").loc[toks]
        dy = lab.dyaw.to_numpy(float)
        fo = pd.read_csv(L() / "fold_of_token.csv").set_index("token").loc[toks]
        lg = fo["log"].to_numpy(str)
        # side of the turn from the logged heading change sign
        big = np.abs(dy) >= 45
        buckets = {"turn >= 20": np.ones(len(dy), bool), "20-45": ~big, ">= 45": big, "left": dy > 0, "right": dy < 0}
        fams = TC.families()
        idx = {c: i for i, c in enumerate(cid)}
        fam_present = {f: [idx[c] for c in v] for f, v in fams.items() if all(c in idx for c in v)}
        base = S[idx[0]]
        rows = []
        for f, ii in fam_present.items():
            best = S[ii].max(0)
            for bn, m in buckets.items():
                r = stats.bootstrap(100 * (best - base)[m], groups=lg[m])
                rows.append(dict(family=f, K=len(ii), bucket=bn, n=int(m.sum()), gain=stats.fmt(r, ".2f"), mean=r["mean"], lo=r["lo"], hi=r["hi"]))
        T1 = pd.DataFrame(rows)
        T1.to_csv(O() / "ceiling_table.csv", index=False)
        # degrees of freedom on >= 20 deg
        m = buckets["turn >= 20"]
        g = lambda f: 100 * (S[fam_present[f]].max(0) - base)[m] if f in fam_present else None
        dof = []
        for nm, v in [("offset only O3", g("O3")), ("curvature only K3", g("K3")), ("speed only V3", g("V3")), ("any single axis F7", g("F7")),
                      ("F19", g("F19")), ("F27", g("F27"))]:
            if v is not None:
                dof.append(dict(row=nm, gain=stats.fmt(stats.bootstrap(v, groups=lg[m]), ".2f")))
        if g("F27") is not None and g("F7") is not None:
            dof.append(dict(row="needs two or more axes (F27 - F7)", gain=stats.fmt(stats.bootstrap(g("F27") - g("F7"), groups=lg[m]), ".2f")))
        for x, y in (("O3", "K3"), ("O3", "V3"), ("K3", "V3")):
            if g(x) is not None:
                dof.append(dict(row=f"{x} - {y}", gain=stats.fmt(stats.bootstrap(g(x) - g(y), groups=lg[m]), ".2f")))
        T2 = pd.DataFrame(dof)
        T2.to_csv(O() / "ceiling_dof.csv", index=False)
        # failure rates identity vs oracle F19 (selection by score)
        fr = []
        f = "F19" if "F19" in fam_present else max(fam_present, key=lambda k: len(fam_present[k]))
        ii = fam_present[f]
        for bn, m in buckets.items():
            best = np.argmax(S[ii][:, m], 0)
            Dm, Nm = D[ii][:, m], N[ii][:, m]
            ar = np.arange(m.sum())
            fr.append(dict(bucket=bn, DAC_fail_identity=_pct(np.mean(D[idx[0]][m] == 0)), DAC_fail_oracle=_pct(np.mean(Dm[best, ar] == 0)),
                           NC_fail_identity=_pct(np.mean(N[idx[0]][m] == 0)), NC_fail_oracle=_pct(np.mean(Nm[best, ar] == 0)),
                           EPDMS_identity=f"{100 * base[m].mean():.2f}", EPDMS_oracle=f"{100 * S[ii][:, m].max(0).mean():.2f}", family=f))
        T3 = pd.DataFrame(fr)
        T3.to_csv(O() / "ceiling_failrates.csv", index=False)
        # best of {identity, speed x 0.8, speed x 0.6} vs the in-sample nt_cache figure (+4.89 / +5.00 on the same tokens)
        C = TC.candidates()
        cv = {round(c[3], 2): i for i, c in enumerate(C) if c[1] == 0.0 and c[2] == 1.0}
        extra = []
        if 0.8 in cv and 0.6 in cv and cv[0.8] in idx and cv[0.6] in idx:
            b3 = np.maximum(base, np.maximum(S[idx[cv[0.8]]], S[idx[cv[0.6]]]))
            r = stats.bootstrap(100 * (b3 - base), groups=lg)
            extra.append(dict(row="best of {identity, speed x 0.8, speed x 0.6}, held-out plans, turn >= 20", gain=stats.fmt(r, ".2f")))
        T4 = pd.DataFrame(extra)
        print(T1[T1.family.isin(["F19", "F27", "F33"]) & (T1.bucket == "turn >= 20")].to_string(index=False))
        print(T2.to_string(index=False))
        print(T3.to_string(index=False))
        print(T4.to_string(index=False))
        T4.to_csv(O() / "ceiling_speed3.csv", index=False)
        figure_ceiling(T1)


def figure_ceiling(T1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fams = [f for f in ["O3", "K3", "V3", "F7", "F19", "F27", "F33"] if f in set(T1.family)]
    bk = ["turn >= 20", "20-45", ">= 45"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for bi, b in enumerate(bk):
        for fi, f in enumerate(fams):
            r = T1[(T1.family == f) & (T1.bucket == b)].iloc[0]
            ax.bar(fi + (bi - 1) * 0.27, r["mean"], 0.26, yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color=["#3182bd", "#6baed6", "#e6550d"][bi],
                   label=b + " deg" if fi == 0 else None, capsize=2)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylabel("privileged best-of-K gain, no-EC EPDMS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ceiling.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("folds")
    p = sp.add_parser("plans")
    p.add_argument("--fold", type=int, required=True)
    p = sp.add_parser("plans-shard")
    p.add_argument("--what", choices=["plans", "export"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--shard", type=int, required=True)
    sp.add_parser("assemble").add_argument("--force", action="store_true")
    sp.add_parser("score")
    sp.add_parser("gate")
    sp.add_parser("diag")
    sp.add_parser("cands").add_argument("--force", action="store_true")
    p = sp.add_parser("cscore")
    p.add_argument("--stage", choices=["t300", "all"], required=True)
    p.add_argument("--family", choices=["F19", "F27", "F33"], default="F33")
    p = sp.add_parser("cgate")
    p.add_argument("--spent-core-h", type=float, required=True)
    p.add_argument("--budget-core-h", type=float, default=40.0)
    sp.add_parser("ceiling").add_argument("--family", choices=["F19", "F27", "F33"], default="F33")
    a = ap.parse_args()
    {"folds": cmd_folds, "plans": cmd_plans, "plans-shard": cmd_plans_shard, "assemble": cmd_assemble, "score": cmd_score, "gate": cmd_gate,
     "diag": cmd_diag, "cands": cmd_cands, "cscore": cmd_cscore, "cgate": cmd_cgate, "ceiling": cmd_ceiling}[a.cmd](a)


if __name__ == "__main__":
    main()
