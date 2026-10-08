"""op_parity nc-taxonomy (plans/2026-10-09-nc-taxonomy-prereg.md, overnight N1): what SH30's NC and TTC failures are, which of them
WA-JEPA passes, and how many a same-path longitudinal rescaling of the plan would recover. CPU only, no model runs.

  select  (.venv)          failing tokens (NC < 1; TTC < 1 with NC = 1) of SH30-F-s0 / s1 from the bench archives + their exported plans
                           -> $OUT/keys_<bench>.pkl (the input of fd_navsim's instrumented replay)
  replay  (envs/navsim2)   fd_navsim's scorer hook (decision 153's event extraction, unchanged) on those tokens: first at-fault collision
                           and first TTC event (object, nuPlan collision type, relative pose and speed) -> $OUT/replay_<bench>.parquet
  family  (.venv)          the speed-scaled plans (turn_ceiling.speed: same path, arc length x a) of both seeds on every navtest token
                           -> $OUT/poses.npz (keys s<seed>_a<100 a>), tokens_fail.txt, tokens_all.txt
  gate    (.venv)          identity gate of a score-poses CSV (a = 1.0 rows == the bench archive) + measured cost
  report  (.venv)          tables -> $OUT/report/{*.csv, tables.md, summary.json} (committed under results/nc_taxonomy/)

Scoring of the family is `python -m jevdrive.bench score-poses --traffic non_reactive` (nc_tax_chain.sh). Rules (fixed in the pre-registration):
  sets      NC failure = NC < 1; TTC-only = TTC < 1 and NC = 1.
  fine      fd_navsim.ctype_class on the first at-fault collision (NC set) or the first TTC event (TTC-only set).
  side      object centroid at the event beside the ego body: dx < 4.05 m (rear-axle frame, not beyond the front bumper) and |dy| >= 1.0 m.
  class     first match of: static object -> E; VRU -> E; vehicle and (fine sideswipe, or fine stopped and side) -> D side contact;
            fine stopped -> A1; fine lead -> A2; cut-in -> B; crossing (30-150 deg) -> C; oncoming / unresolved -> E.
  buckets   bench strata: turn (|logged 4 s heading change| <5 / 5-20 / 20-45 / >45 deg; navhard: the PDM-Closed reference path),
            speed (t0 ego speed <2 / 2-5 / 5-10 / >=10 m/s).
  oracle    main family a in {0.7 .. 1.1}, extended adds {0.5, 0.6}; recovered = some a != 1 passes the set's criterion
            (NC set: NC = 1; TTC-only: TTC = 1 and NC = 1); clean = that a also scores above a = 1.0 (no-EC EPDMS);
            a* = the recovering main-family scale with the highest no-EC EPDMS (ties: closest to 1.0, then the slower one).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
OUT = D / "runs/op_parity/nc_tax"
RES = OUT / "report"                                     # copied to experiments/op_parity/results/nc_taxonomy/ on the Mac
SEEDS = (0, 1)
MODEL = {"navtest": "SH30-F-s{}", "navhard": "SH30-F-s{}@gimm"}
SPLIT = {"navtest": "navsim/navtest", "navhard": "navsim/navhard_two_stage"}
MAIN = (0.7, 0.8, 0.9, 1.0, 1.1)
EXT = (0.5, 0.6) + MAIN
DOSE = {"0.9-1.1": (0.9, 1.1), "0.8-1.1": (0.8, 0.9, 1.1), "0.7-1.1 (main)": (0.7, 0.8, 0.9, 1.1), "0.5-1.1 (extended)": (0.5, 0.6, 0.7, 0.8, 0.9, 1.1),
        "slower only (0.7-0.9)": (0.7, 0.8, 0.9), "faster only (1.1)": (1.1,)}
SUB8 = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC"]
EP_TOL = 1e-6
SIDE_DX, SIDE_DY = 4.05, 1.0
BIG = {"A": "A stopped / slow vehicle ahead", "B": "B cut-in", "C": "C crossing at junction", "D": "D side contact", "E": "E other"}
CLS = ["A1 stopped vehicle ahead", "A2 lead vehicle (moving)", "B cut-in", "C crossing / turn conflict", "D side contact",
       "E static object", "E VRU", "E oncoming", "E unresolved"]
TURNS, SPEEDS = ["<5", "5-20", "20-45", ">45"], ["<2", "2-5", "5-10", ">=10"]


def akey(seed, a):
    return f"s{seed}_a{round(100 * a):03d}"


def archive(bench, who):
    """Stored per-token sub-scores (index token; NC .. EC, score, log) of a model: bench navtest units, or the navhard harness stage-1 tokens."""
    import pandas as pd
    from jevdrive.bench import compat, tables as T
    from jevdrive.bench.navsim import SUBS
    if bench == "navtest":
        u, src = T.load("navtest", who)
        assert u is not None, f"no navtest units for {who}"
        return u, src
    from jevdrive.bench.models import resolve
    d = T._navhard_legacy(resolve(who)) if who == "WA-JEPA" else compat.navhard_dir(who)
    h = pd.read_csv(d / "harness_tokens.csv")
    h = h[h.stage == 1].drop_duplicates("token").set_index("token").rename(columns={v: k for k, v in SUBS.items()})
    lg = T.tokens_meta("navhard")
    h["log"] = [lg.get(t, "?") for t in h.index]
    return h, f"stored {d}"


def fail_sets(u):
    nc = u.NC < 1
    return nc, (u.TTC < 1) & ~nc


def preds(bench, seed):
    from jevdrive.bench import compat
    z = np.load(compat.pred_file(MODEL[bench].format(seed), bench))
    return z["tokens"].astype(str), z["poses"]


# ---------------------------------------------------------------- select / replay
def cmd_select(a):
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", f"nc_tax/select-{a.bench}", config=vars(a)) as run:
        run.use_split(splits.load(SPLIT[a.bench]))
        OUT.mkdir(parents=True, exist_ok=True)
        plans, want, n = {}, {}, {}
        for s in SEEDS:
            u, src = archive(a.bench, MODEL[a.bench].format(s))
            nc, ttc = fail_sets(u)
            tok, P = preds(a.bench, s)
            pos = {t: i for i, t in enumerate(tok)}
            bad = sorted(u.index[nc | ttc])
            plans[f"s{s}"] = {t: np.asarray(P[pos[t]], np.float64) for t in bad}
            for t in bad:
                want.setdefault(t, []).append(f"s{s}")
            n[f"s{s}"] = dict(source=src, tokens=len(u), nc_fail=int(nc.sum()), ttc_only=int(ttc.sum()))
        pickle.dump({"plans": plans, "want": want}, open(OUT / f"keys_{a.bench}.pkl", "wb"))
        run.summary.update(bench=a.bench, replay_tokens=len(want), **n)
        run.info(json.dumps(run.summary, indent=1))


def cmd_replay(a):
    import multiprocessing as mp
    import pandas as pd
    import fd_navsim as FD
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    kf = OUT / f"keys_{a.bench}.pkl"
    todo = sorted(pickle.load(open(kf, "rb"))["want"])
    procs = a.procs or max(4, n_cpus() // 3)
    with Run("op_parity", f"nc_tax/replay-{a.bench}", config=vars(a) | {"n_tokens": len(todo), "procs": procs}) as run:
        t0, rows = time.time(), []
        with mp.get_context("fork").Pool(procs, initializer=FD._init, initargs=(a.bench, kf)) as pool:
            for i, r in enumerate(pool.imap_unordered(FD.work, todo, chunksize=2)):
                rows.extend(r["rows"])
                if (i + 1) % 100 == 0:
                    run.status(f"{i + 1}/{len(todo)} tokens, {time.time() - t0:.0f} s")
        S = np.stack([r.pop("_states") for r in rows])
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / f"replay_{a.bench}.parquet")
        np.savez_compressed(OUT / f"replay_{a.bench}_states.npz", key=df.key.to_numpy(str), token=df.token.to_numpy(str), states=S)
        run.summary.update(bench=a.bench, tokens=len(todo), rows=len(df), wall_s=time.time() - t0)


# ---------------------------------------------------------------- family / gate
def cmd_family(a):
    import turn_ceiling as TC
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "nc_tax/family", config=vars(a)) as run:
        nt = splits.load(SPLIT["navtest"])
        run.use_split(nt)
        arrs, bad, tok0 = {}, set(), None
        for s in SEEDS:
            tok, P = preds("navtest", s)
            tok0 = tok if tok0 is None else tok0
            assert (tok == tok0).all() and nt.mask(tok).all()
            for sc in EXT:
                arrs[akey(s, sc)] = TC.transform(P, v=sc)
            assert arrs[akey(s, 1.0)] is P                                     # identity = the archived array itself
            u, _ = archive("navtest", MODEL["navtest"].format(s))
            nc, ttc = fail_sets(u)
            bad |= set(u.index[nc | ttc])
        np.savez(OUT / "poses.npz", tokens=tok0, **arrs)
        (OUT / "tokens_fail.txt").write_text("\n".join(sorted(bad)) + "\n")
        (OUT / "tokens_all.txt").write_text("\n".join(sorted(tok0)) + "\n")
        (OUT / "keys.txt").write_text(" ".join(arrs) + "\n")
        run.summary.update(keys=len(arrs), tokens=len(tok0), fail_tokens=len(bad))


def cmd_gate(a):
    import pandas as pd
    from jevdrive.bench import poses as BP, tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.run import Run
    with Run("op_parity", f"nc_tax/gate-{a.stage}", config=vars(a)) as run:
        tok = BP.read_tokens(OUT / f"tokens_{a.stage}.txt")
        df = pd.read_csv(a.score)
        ids = df[df.key.isin([akey(s, 1.0) for s in SEEDS])]                   # only the identity rows are read here
        res, ok = dict(stage=a.stage, n_tokens=len(tok), rows=len(df), seeds={}), True
        del df
        for s in SEEDS:
            g = ids[ids.key == akey(s, 1.0)].set_index("token").loc[tok]
            u, src = archive("navtest", MODEL["navtest"].format(s))
            r = dict(source=src)
            for k in SUB8:
                dlt = np.abs(g[SUBS[k]].to_numpy(float) - u.loc[tok, k].to_numpy(float))
                r[k] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > (EP_TOL if k == "EP" else 0)).sum()))
            X = u.loc[tok, T.TERMS].to_numpy(float)
            X[:, 8] = np.nan
            dlt = np.abs(g.score.to_numpy(float) - T.score_of(X))
            r["score_noec"] = dict(max_abs=float(dlt.max()), n_diff=int((dlt > EP_TOL).sum()))
            r["pass"] = all(v["n_diff"] == 0 for v in r.values() if isinstance(v, dict))
            ok &= r["pass"]
            res["seeds"][s] = r
        res["pass"] = bool(ok)
        (OUT / f"gate_{a.stage}.json").write_text(json.dumps(res, indent=1))
        run.summary.update(res)
        run.info(json.dumps(res, indent=1))
        if not ok:
            raise SystemExit("identity gate failed")


# ---------------------------------------------------------------- report
def classify(r):
    """(fine class of decision 153, side flag, class label) of one replayed failing (seed, token) row."""
    import fd_navsim as FD
    def ok(v):  # noqa: E306
        return v is not None and not (isinstance(v, float) and np.isnan(v))
    fine = FD.ctype_class(r)
    pre = "col" if ok(r.get("col_t")) else "ttc" if ok(r.get("ttc_t")) else None
    dx, dy = (r.get(f"{pre}_dx"), r.get(f"{pre}_dy")) if pre else (None, None)
    side = bool(ok(dx) and ok(dy) and dx < SIDE_DX and abs(dy) >= SIDE_DY)
    ev_v = r.get(f"{pre}_ego_v") if pre else np.nan
    if fine in ("static object", "VRU"):
        cls = "E " + fine
    elif fine.startswith("sideswipe") or (fine == "stopped vehicle ahead" and side):
        cls = "D side contact"
    else:
        cls = {"stopped vehicle ahead": "A1 stopped vehicle ahead", "lead vehicle": "A2 lead vehicle (moving)", "cut-in": "B cut-in",
               "crossing / turn conflict": "C crossing / turn conflict", "oncoming": "E oncoming", "unresolved": "E unresolved"}[fine]
    return fine, side, cls, pre, ev_v


def arc(P):
    Q = np.concatenate([np.zeros((len(P), 1, 2)), np.asarray(P, float)[:, :, :2]], 1)
    return np.linalg.norm(np.diff(Q, axis=1), axis=2).sum(1)


def geometry(bench):
    """token -> turn / speed bucket, v0, log, logged (navtest) or PDM-reference (navhard) 4 s arc length."""
    import pandas as pd
    from jevdrive.bench import tables as T
    if bench == "navtest":
        g = T.navtest_strata()[["dyaw", "v0", "turn", "speed"]].copy()
        tab = np.load(D / "runs/op_parity/cache/lb_navtest/tab.npz")
        g["ref_arc"] = pd.Series(arc(tab["fut"]), index=tab["names"].astype(str)).reindex(g.index)
        return g
    z = np.load(D / "runs/op_parity/four_dirs/refpath_navhard.npz")
    fut = z["path"][:, 5::5].astype(float)
    g = pd.DataFrame(dict(dyaw=np.degrees(fut[:, -1, 2]), v0=z["v0"].astype(float), ref_arc=arc(fut)), index=z["tokens"].astype(str))
    g["turn"] = pd.cut(g.dyaw.abs(), T.TURN_BINS, right=False, labels=T.TURN_LABELS).astype(str)
    g["speed"] = pd.cut(g.v0, [0, 2, 5, 10, 99], right=False, labels=SPEEDS).astype(str)
    return g


def fail_rows(bench):
    """One row per failing (seed, token): set, class, buckets, WA-JEPA pass, replay check; plus the archives."""
    import pandas as pd
    rp = pd.read_parquet(OUT / f"replay_{bench}.parquet").set_index(["key", "token"])
    geo = geometry(bench)
    wa, wa_src = archive(bench, "WA-JEPA")
    from jevdrive.bench.navsim import SUBS
    rows, U, chk = [], {}, {}
    for s in SEEDS:
        u, _ = archive(bench, MODEL[bench].format(s))
        U[s] = u
        nc, ttc = fail_sets(u)
        tok, P = preds(bench, s)
        plan_arc = pd.Series(arc(P), index=tok)
        dmax = 0.0
        for t in u.index[nc | ttc]:
            r = rp.loc[(f"s{s}", t)]
            d = max(abs(float(r[SUBS[k]]) - float(u.at[t, k])) for k in SUB8)
            dmax = max(dmax, d)
            fine, side, cls, pre, ev_v = classify(r.to_dict())
            st = "NC" if nc[t] else "TTC-only"
            w = wa.loc[t] if t in wa.index else None
            wa_pass = np.nan if w is None else float(w.NC == 1 if st == "NC" else (w.TTC == 1 and w.NC == 1))
            g = geo.loc[t]
            rows.append(dict(bench=bench, seed=s, token=t, log=u.at[t, "log"], set=st, fine=fine, side=side, cls=cls, big=cls[0], event=pre,
                             turn=g.turn, speed=g.speed, v0=float(g.v0), dyaw=float(g.dyaw), ev_v=float(ev_v) if ev_v is not None else np.nan,
                             NC=float(u.at[t, "NC"]), TTC=float(u.at[t, "TTC"]), DAC=float(u.at[t, "DAC"]),
                             speed_ratio=float(plan_arc[t] / max(g.ref_arc, 0.5)), wa_pass=wa_pass, replay_diff=d))
        chk[f"s{s}"] = dmax
    F = pd.DataFrame(rows)
    F["ev_lt3"] = (F.ev_v < 3).astype(float).where(F.ev_v.notna())
    F["fast"] = (F.speed_ratio > 1.1).astype(float)
    return F, U, wa, geo, chk


def oracle(F, sc):
    """Adds the recovery columns to the navtest failing rows; sc = score-poses CSV indexed (seed, token, a)."""
    from jevdrive.bench.navsim import SUBS
    nc_c, ttc_c, ep_c = SUBS["NC"], SUBS["TTC"], SUBS["EP"]
    out = []
    for r in F.itertuples():
        g = sc.loc[(r.seed, r.token)]
        base = g.loc[1.0]
        crit = (g[nc_c] == 1) if r.set == "NC" else ((g[ttc_c] == 1) & (g[nc_c] == 1))
        rec = {a for a in g.index if a != 1.0 and crit[a]}
        clean = {a for a in rec if g.at[a, "score"] > base.score + 1e-9}
        o = {f"rec {k}": float(bool(rec & set(v))) for k, v in DOSE.items()}
        o |= {f"clean {k}": float(bool(clean & set(v))) for k, v in DOSE.items()}
        for nm, pool_ in (("", rec & set(MAIN)), ("c", clean & set(MAIN)), ("cx", clean)):
            if pool_:
                a = sorted(pool_, key=lambda x: (-g.at[x, "score"], abs(x - 1.0), x))[0]
                o |= {f"a{nm}": a, f"dEP{nm}": float(g.at[a, ep_c] - base[ep_c]), f"dscore{nm}": float(g.at[a, "score"] - base.score),
                      f"dNC{nm}": float(g.at[a, nc_c] - base[nc_c]), f"dTTC{nm}": float(g.at[a, ttc_c] - base[ttc_c])}
        o["min_dev"] = min((abs(a - 1.0) for a in rec), default=np.nan)
        out.append(o)
    import pandas as pd
    return pd.concat([F.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def counts(F, by, extra=()):
    """Per `by` cell: n per seed and the seed mean, plus pooled means of the `extra` 0/1 columns (in %)."""
    import pandas as pd
    n = F.groupby(by + ["seed"]).size().unstack("seed", fill_value=0).reindex(columns=list(SEEDS), fill_value=0)
    o = pd.DataFrame({"n_s0": n[0], "n_s1": n[1], "n": n.mean(1)})
    for c in extra:
        o[c] = 100 * F.groupby(by)[c].mean()
    return o.reset_index()


def ci(stats, x, g):
    q = stats.bootstrap(np.asarray(x, float), np.asarray(g))
    return f"{100 * q['mean']:.0f} [{100 * q['lo']:.0f}, {100 * q['hi']:.0f}]"


def md(df, digits=1):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.{digits}f}")
        return str(v)
    cols = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def cross(F, col, order):
    """class x bucket table of seed-mean counts."""
    import pandas as pd
    t = (F.groupby(["cls", col]).size() / len(SEEDS)).unstack(col, fill_value=0.0).reindex(index=[c for c in CLS if c in set(F.cls)], columns=order, fill_value=0.0)
    t["all"] = t.sum(1)
    t.loc["all"] = t.sum(0)
    return t.reset_index().rename(columns={"cls": "class"})


def cmd_report(a):
    import pandas as pd
    from jevdrive import stats
    from jevdrive.bench import tables as T
    from jevdrive.bench.navsim import SUBS
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("op_parity", "nc_tax/report", config=vars(a)) as run:
        run.use_split(splits.load(SPLIT["navtest"]))
        run.use_split(splits.load(SPLIT["navhard"]))
        RES.mkdir(parents=True, exist_ok=True)
        L, summ = [], {}
        F, U, wa, geo, chk = fail_rows("navtest")
        summ["replay_max_abs_diff_vs_bench"] = {"navtest": chk}
        badrep = F.replay_diff > 1e-6
        summ["navtest_rows_dropped_replay_mismatch"] = int(badrep.sum())
        F = F[~badrep].reset_index(drop=True)
        sc = pd.read_csv(a.score_fail)
        sc["seed"], sc["a"] = sc.key.str[1].astype(int), sc.key.str[4:].astype(int) / 100
        F = oracle(F, sc.set_index(["seed", "token", "a"]).sort_index())
        F.to_csv(RES / "navtest_fail_tokens.csv", index=False)
        N = len(geo)
        X = ["wa_pass", "ev_lt3", "fast", "rec 0.7-1.1 (main)", "clean 0.7-1.1 (main)", "rec 0.5-1.1 (extended)"]
        ren = {"wa_pass": "WA-JEPA passes %", "ev_lt3": "event at ego < 3 m/s %", "fast": "plan > 1.1 x logged arc %", "rec 0.7-1.1 (main)": "recovered, main %",
               "clean 0.7-1.1 (main)": "clean recovered, main %", "rec 0.5-1.1 (extended)": "recovered, extended %"}

        # 1. sets and classes
        L.append("## navtest: failure sets\n")
        t = counts(F, ["set"], X).rename(columns=ren)
        t["rate % of 12 146"] = 100 * t.n / N
        L.append(md(t))
        for st in ("NC", "TTC-only"):
            G = F[F.set == st]
            for by, nm, tag in ((["big"], "class", "class"), (["cls"], "sub-class", "subclass"), (["fine"], "decision-153 fine class", "fine")):
                t = counts(G, by, X).rename(columns=ren)
                t.insert(4, "share %", 100 * t.n / t.n.sum())
                if by == ["big"]:
                    t["big"] = t.big.map(BIG)
                    t.insert(5, "share % [95% CI]", [ci(stats, G.big == b[0], G.log) for b in t.big])
                    t["WA-JEPA passes % [95% CI]"] = [ci(stats, G.wa_pass[G.big == b[0]], G.log[G.big == b[0]]) for b in t.big]
                    t["SH30-specific (fails, WA passes): n"] = [G.wa_pass[G.big == b[0]].sum() / len(SEEDS) for b in t.big]
                    t["SH30-specific share of set % [95% CI]"] = [ci(stats, (G.big == b[0]) & (G.wa_pass == 1), G.log) for b in t.big]
                    t["recovered, main % [95% CI]"] = [ci(stats, G["rec 0.7-1.1 (main)"][G.big == b[0]], G.log[G.big == b[0]]) for b in t.big]
                    summ[f"navtest_{st}_by_class"] = t.to_dict("records")
                L.append(f"\n### {st} failures by {nm}\n\n" + md(t))
                t.to_csv(RES / f"navtest_{st}_{tag}.csv", index=False)
            L.append(f"\n### {st} failures: class x turn bucket (seed-mean counts)\n\n" + md(cross(G, "turn", TURNS)))
            L.append(f"\n### {st} failures: class x t0 ego speed band, m/s (seed-mean counts)\n\n" + md(cross(G, "speed", SPEEDS)))
            t = counts(G, ["turn"], X).rename(columns=ren)
            nb = geo.turn.value_counts()
            t["rate % of bucket"] = [100 * n / nb[b] for n, b in zip(t.n, t.turn)]
            L.append(f"\n### {st} failures by turn bucket\n\n" + md(t.set_index("turn").reindex(TURNS).dropna(subset=["n"]).reset_index(), 2))
            t = counts(G, ["speed"], X).rename(columns=ren)
            nb = geo.speed.value_counts()
            t["rate % of band"] = [100 * n / nb[b] for n, b in zip(t.n, t.speed)]
            L.append(f"\n### {st} failures by t0 ego speed band\n\n" + md(t.set_index("speed").reindex(SPEEDS).dropna(subset=["n"]).reset_index(), 2))
        g3 = counts(F, ["set", "big", "cls", "turn", "speed"], ["wa_pass", "rec 0.7-1.1 (main)"])
        g3.to_csv(RES / "navtest_class_turn_speed.csv", index=False)
        L.append("\n### NC failures: class x turn x speed cells with seed-mean n >= 3\n\n"
                 + md(g3[(g3.set == "NC") & (g3.n >= 3)].drop(columns=["set", "big"]).sort_values("n", ascending=False).rename(columns=ren)))

        # WA-JEPA's own failures (stored per-token scores only; no replay, so no type)
        wn, wt = fail_sets(wa.loc[geo.index.intersection(wa.index)])
        w = pd.DataFrame({"WA-JEPA NC failures": geo.turn[wn[wn].index].value_counts(), "WA-JEPA TTC-only failures": geo.turn[wt[wt].index].value_counts()}).reindex(TURNS).fillna(0)
        w.loc["all"] = w.sum(0)
        L.append("\n### WA-JEPA's own failures by turn bucket (stored per-token scores)\n\n" + md(w.reset_index().rename(columns={"index": "turn"})))
        summ["wa_own"] = dict(nc=int(wn.sum()), ttc_only=int(wt.sum()))

        # 2. oracle
        L.append("\n## navtest: same-path longitudinal scaling oracle (non-reactive, no-EC EPDMS)\n")
        rows = []
        for st in ("NC", "TTC-only"):
            G = F[F.set == st]
            for k in DOSE:
                rows.append(dict(set=st, family=k, n=len(G) / len(SEEDS), **{"recovered % [95% CI]": ci(stats, G[f"rec {k}"], G.log),
                                                                              "clean recovered % [95% CI]": ci(stats, G[f"clean {k}"], G.log)}))
        t = pd.DataFrame(rows)
        t.to_csv(RES / "oracle_recovery.csv", index=False)
        summ["oracle_recovery"] = rows
        L.append("### Recovery rate by family\n\n" + md(t))
        rows = []
        for st in ("NC", "TTC-only"):
            for nm, sfx in (("recovered (main)", ""), ("clean recovered (main)", "c")):
                G = F[(F.set == st) & F[f"a{sfx}"].notna()]
                if not len(G):
                    continue
                r = dict(set=st, tokens=nm, n=len(G) / len(SEEDS), **{"EP(a*) - EP(1.0), x100": 100 * G[f"dEP{sfx}"].mean(),
                                                                       "no-EC EPDMS(a*) - (1.0), x100": 100 * G[f"dscore{sfx}"].mean()})
                r |= {f"a*={x}": int((G[f"a{sfx}"] == x).sum()) / len(SEEDS) for x in MAIN if x != 1.0}
                rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "oracle_ep_cost_tokens.csv", index=False)
        summ["oracle_ep_cost_tokens"] = rows
        L.append("\n### EP cost on the recovered tokens (a* = best-scoring recovering scale)\n\n" + md(t, 2))
        t = counts(F[F.set == "NC"], ["cls"], [f"rec {k}" for k in DOSE] + ["clean 0.7-1.1 (main)"])
        L.append("\n### NC recovery by sub-class and family (%)\n\n" + md(t))
        t.to_csv(RES / "oracle_nc_by_class.csv", index=False)
        t = counts(F[F.set == "TTC-only"], ["cls"], [f"rec {k}" for k in DOSE] + ["clean 0.7-1.1 (main)"])
        L.append("\n### TTC-only recovery by sub-class and family (%)\n\n" + md(t))
        t.to_csv(RES / "oracle_ttc_by_class.csv", index=False)
        t = counts(F[F.set == "NC"], ["turn"], ["rec 0.7-1.1 (main)", "clean 0.7-1.1 (main)", "rec 0.5-1.1 (extended)"])
        L.append("\n### NC recovery by turn bucket (%)\n\n" + md(t.set_index("turn").reindex(TURNS).dropna(subset=["n"]).reset_index()))
        # board-level upper bound: replace only the failing tokens that have a clean recovering scale
        rows = []
        for fam, sfx in (("main 0.7-1.1", "c"), ("extended 0.5-1.1", "cx")):
            r = dict(family=fam)
            for k, c in (("NC", "dNC"), ("TTC", "dTTC"), ("EP", "dEP"), ("no-EC EPDMS", "dscore")):
                per = [F[(F.seed == s)][f"{c}{sfx}"].sum() / N * 100 for s in SEEDS]
                r[f"{k} (s0 / s1)"] = f"{np.mean(per):+.3f} ({per[0]:+.3f} / {per[1]:+.3f})"
            r["tokens replaced (seed mean)"] = F[f"a{sfx}"].notna().sum() / len(SEEDS)
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(RES / "oracle_board_upper.csv", index=False)
        summ["oracle_board_upper"] = rows
        L.append("\n### Board-level upper bound: a* on SH30's failing tokens only, all other tokens unchanged (points of the 12 146-token mean)\n\n" + md(t))

        # 3. uniform scaling of every token (stage 1)
        if a.score_all and Path(a.score_all).exists():
            sa = pd.read_csv(a.score_all)
            sa["seed"], sa["a"] = sa.key.str[1].astype(int), sa.key.str[4:].astype(int) / 100
            rows = []
            for sc_ in EXT:
                r = dict(a=sc_)
                per = {k: [] for k in ("NC", "TTC", "DAC", "EP", "no-EC EPDMS", "NC fail", "new NC fail", "fixed NC fail", "TTC-only fail")}
                for s in SEEDS:
                    b = sa[(sa.seed == s) & (sa.a == 1.0)].set_index("token")
                    x = sa[(sa.seed == s) & (sa.a == sc_)].set_index("token").loc[b.index]
                    for k in ("NC", "TTC", "DAC", "EP"):
                        per[k].append(100 * (x[SUBS[k]].mean() - b[SUBS[k]].mean()))
                    per["no-EC EPDMS"].append(100 * (x.score.mean() - b.score.mean()))
                    fb, fx = b[SUBS["NC"]] < 1, x[SUBS["NC"]] < 1
                    per["NC fail"].append(int(fx.sum()))
                    per["new NC fail"].append(int((fx & ~fb).sum()))
                    per["fixed NC fail"].append(int((fb & ~fx).sum()))
                    per["TTC-only fail"].append(int(((x[SUBS["TTC"]] < 1) & ~fx).sum()))
                for k, v in per.items():
                    r[k if "fail" in k else f"d{k}"] = float(np.mean(v))
                rows.append(r)
            t = pd.DataFrame(rows)
            t.to_csv(RES / "uniform_scaling.csv", index=False)
            summ["uniform_scaling"] = rows
            L.append("\n### Uniform scaling of every token (change against a = 1.0 in points; counts are seed means)\n\n" + md(t, 2))
            summ["uniform_base"] = {f"s{s}": {k: 100 * float(sa[(sa.seed == s) & (sa.a == 1.0)][SUBS[k]].mean()) for k in ("NC", "TTC", "DAC", "EP")} for s in SEEDS}

        # 4. navhard stage 1: counts only
        H, UH, wah, geoh, chkh = fail_rows("navhard")
        summ["replay_max_abs_diff_vs_bench"]["navhard"] = chkh
        H.to_csv(RES / "navhard_s1_fail_tokens.csv", index=False)
        L.append(f"\n## navhard stage 1 ({len(UH[0])} tokens, G frames): counts only\n")
        t = counts(H, ["set"], ["wa_pass", "ev_lt3", "fast"]).rename(columns=ren)
        L.append(md(t))
        for st in ("NC", "TTC-only"):
            G = H[H.set == st]
            if not len(G):
                continue
            t = counts(G, ["cls"], ["wa_pass"]).rename(columns=ren)
            t.to_csv(RES / f"navhard_s1_{st}_class.csv", index=False)
            L.append(f"\n### navhard stage 1, {st} failures by sub-class\n\n" + md(t))
            L.append(f"\n### navhard stage 1, {st}: class x turn bucket (seed-mean counts)\n\n" + md(cross(G, "turn", TURNS)))
            L.append(f"\n### navhard stage 1, {st}: class x t0 ego speed band (seed-mean counts)\n\n" + md(cross(G, "speed", SPEEDS)))
        wn, wt = fail_sets(wah)
        summ["navhard_s1"] = dict(tokens=len(UH[0]), wa_nc=int(wn.sum()), wa_ttc_only=int(wt.sum()),
                                  sh30={f"s{s}": dict(nc=int(fail_sets(UH[s])[0].sum()), ttc_only=int(fail_sets(UH[s])[1].sum())) for s in SEEDS})
        L.append(f"\nWA-JEPA on the same {len(wah)} stage-1 tokens: {int(wn.sum())} NC failures, {int(wt.sum())} TTC-only failures (stored per-token scores).")

        (RES / "tables.md").write_text("\n".join(L) + "\n")
        (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
        run.summary.update(navtest_rows=len(F), navhard_rows=len(H), replay_check=summ["replay_max_abs_diff_vs_bench"])
        run.info(f"wrote {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for c in ("select", "replay"):
        p = sp.add_parser(c)
        p.add_argument("--bench", required=True, choices=["navtest", "navhard"])
        if c == "replay":
            p.add_argument("--procs", type=int, default=0, help="worker processes (default: a third of the cgroup quota)")
    sp.add_parser("family")
    p = sp.add_parser("gate")
    p.add_argument("--stage", required=True, choices=["fail", "all"])
    p.add_argument("--score", required=True)
    p = sp.add_parser("report")
    p.add_argument("--score-fail", required=True)
    p.add_argument("--score-all", default="")
    a = ap.parse_args()
    {"select": cmd_select, "replay": cmd_replay, "family": cmd_family, "gate": cmd_gate, "report": cmd_report}[a.cmd](a)
