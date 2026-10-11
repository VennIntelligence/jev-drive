"""long_oracle report (plans/2026-10-11-long-oracle-prereg.md, readings 3-5): uniform speed scaling of the whole board, its gain split by
what changed, and the non-privileged conditioned operating point (scale per t0 speed band, log cross-fit), for each arm and seed.
The taxonomy and the privileged oracle (readings 1-2) are nc_tax.py's report per arm ($DATA_DIR/runs/long_oracle/<arm>/report).

Inputs per arm: <arm>/score_all.csv (score-poses non_reactive, keys s<seed>_a<100 a>), <arm>/report/navtest_fail_tokens.csv.
Output: $DATA_DIR/runs/long_oracle/report/{tables.md, *.csv, summary.json} (copied to experiments/long_oracle/results/long_oracle/).
Rules (fixed in the pre-registration):
  uniform     every token at scale a against a = 1.0; paired, cluster bootstrap over logs (jevdrive.stats.paired, B 10 000).
  split       (i) tokens whose NC or TTC changed, (ii) DAC changed and NC / TTC did not, (iii) the rest; points of the 12 146-token mean.
  conditioned fold = sha1(log) parity; per cell pick the scale with the highest training-fold mean no-EC EPDMS (ties: closest to 1.0,
              then slower); out-of-fold over both folds. Primary cells = t0 speed band (4); secondary = band x plan 4 s turn bucket (16).
              Scale set = 0.7 .. 1.15 (primary) and 0.5 .. 1.15 (secondary family).
  room        both seeds of P2H10S-F meet (U) some a != 1 with CI lo > 0 or (C) the primary conditioned point with CI lo > 0, and, where a < 1
              is used, the NC / TTC-changed tokens carry at least half of the gain.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO)]
D = Path(os.environ.get("DATA_DIR", "/root/autodl-tmp/ujs"))
ROOT = D / "runs/long_oracle"
RES = ROOT / "report"
ARMS = ("P2H10S-F", "P2H10-F")
PRIMARY = "P2H10S-F"
SEEDS = (0, 1)
GRID = (0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
WIDE = (0.5, 0.6) + GRID
SPEEDS, TURNS = ["<2", "2-5", "5-10", ">=10"], ["<5", "5-20", "20-45", ">45"]
COLS = {"NC": "no_at_fault_collisions", "TTC": "time_to_collision_within_bound", "DAC": "drivable_area_compliance", "EP": "ego_progress"}


def fmt(r, k=100):
    return f"{k * r['mean']:+.2f} [{k * r['lo']:+.2f}, {k * r['hi']:+.2f}]"


def md(df, digits=2):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() and abs(v) >= 10 else f"{v:.{digits}f}")
        return str(v)
    cols = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def load(arm):
    """{seed: {a: DataFrame indexed by token (NC TTC DAC EP score)}}, token meta (log, speed band, plan turn bucket, fold), fail table."""
    from jevdrive.bench import compat, tables as T
    sa = pd.read_csv(ROOT / arm / "score_all.csv")
    sa["seed"], sa["a"] = sa.key.str[1].astype(int), sa.key.str[4:].astype(int) / 100
    sa = sa.rename(columns={v: k for k, v in COLS.items()})
    S = {s: {a: g.set_index("token")[list(COLS) + ["score"]] for a, g in sa[sa.seed == s].groupby("a")} for s in SEEDS}
    strata = T.navtest_strata()
    meta = {}
    for s in SEEDS:
        u, _ = T.load("navtest", f"{arm}-s{s}")
        z = np.load(compat.pred_file(f"{arm}-s{s}", "navtest"))
        pturn = pd.Series(np.abs(np.degrees(z["poses"][:, -1, 2].astype(float))), index=z["tokens"].astype(str))
        m = pd.DataFrame(dict(log=u.log, speed=strata.speed.reindex(u.index), turn=strata.turn.reindex(u.index),
                              pturn=pd.cut(pturn.reindex(u.index), [0, 5, 20, 45, 999], right=False, labels=TURNS).astype(str)))
        m["fold"] = [int(hashlib.sha1(lg.encode()).hexdigest(), 16) % 2 for lg in m.log]
        meta[s] = m
    F = pd.read_csv(ROOT / arm / "report/navtest_fail_tokens.csv")
    return S, meta, F


def split_gain(x, b):
    nct = (x.NC != b.NC) | (x.TTC != b.TTC)
    dac = ~nct & (x.DAC != b.DAC)
    d = x.score - b.score
    n = len(b)
    return {"(i) NC/TTC changed": 100 * d[nct].sum() / n, "(ii) DAC only": 100 * d[dac].sum() / n, "(iii) rest": 100 * d[~nct & ~dac].sum() / n,
            "tokens changed NC/TTC": int(nct.sum()), "tokens DAC only": int(dac.sum())}


def uniform(arm, S, meta, F):
    from jevdrive import stats
    rows, cls_rows, new_rows = [], [], []
    for s in SEEDS:
        b, m = S[s][1.0], meta[s].loc[S[s][1.0].index]
        f = F[F.seed == s].set_index("token")
        for a in WIDE:
            x = S[s][a].loc[b.index]
            r = stats.paired(x.score.to_numpy(float), b.score.to_numpy(float), groups=m.log.to_numpy())
            row = dict(arm=arm, seed=s, a=a, **{"dEPDMS no-EC [95% CI]": fmt(r)}, dEPDMS=100 * r["mean"], lo=100 * r["lo"], hi=100 * r["hi"])
            for k in ("NC", "TTC", "DAC", "EP"):
                row[f"d{k}"] = 100 * (x[k].mean() - b[k].mean())
            fb, fx = b.NC < 1, x.NC < 1
            tb, tx = (b.TTC < 1) & ~fb, (x.TTC < 1) & ~fx
            row |= {"NC fail": int(fx.sum()), "NC fixed": int((fb & ~fx).sum()), "NC new": int((fx & ~fb).sum()),
                    "TTC-only fail": int(tx.sum()), "TTC-only fixed": int((tb & ~tx & ~fx).sum()), "TTC-only new": int((tx & ~tb).sum())}
            row |= split_gain(x, b)
            rows.append(row)
            # by class: share of the a = 1.0 failures of each class that pass the set's criterion at a
            for st in ("NC", "TTC-only"):
                g = f[f.set == st]
                ok = (x.loc[g.index, "NC"] == 1) if st == "NC" else ((x.loc[g.index, "TTC"] == 1) & (x.loc[g.index, "NC"] == 1))
                for c, gg in g.groupby("cls"):
                    cls_rows.append(dict(arm=arm, seed=s, set=st, cls=c, a=a, n=len(gg), fixed=int(ok[gg.index].sum())))
            nw = (fx & ~fb) | (tx & ~tb)
            for (sp, tu), k in m[nw].groupby(["speed", "turn"]).size().items():
                new_rows.append(dict(arm=arm, seed=s, a=a, speed=sp, turn=tu, new_nc_or_ttc=int(k)))
    return pd.DataFrame(rows), pd.DataFrame(cls_rows), pd.DataFrame(new_rows)


def conditioned(arm, S, meta, cells, grid, tag):
    from jevdrive import stats
    rows, picks = [], []
    for s in SEEDS:
        b = S[s][1.0]
        m = meta[s].loc[b.index]
        sc = pd.DataFrame({a: S[s][a].score.loc[b.index] for a in grid})
        chosen = pd.Series(1.0, index=b.index)
        for fo in (0, 1):
            tr, te = m.fold != fo, m.fold == fo
            mean = sc[tr].groupby([m[c][tr] for c in cells]).mean()
            for key, v in mean.iterrows():
                key = key if isinstance(key, tuple) else (key,)
                best = sorted(grid, key=lambda a: (-round(v[a], 12), abs(a - 1.0), a))[0]
                cell = te.copy()
                for c, k in zip(cells, key):
                    cell &= m[c] == k
                chosen[cell] = best
                picks.append(dict(arm=arm, seed=s, variant=tag, fold=fo, cell=" / ".join(map(str, key)), a=best, n_test=int(cell.sum()),
                                  train_gain=100 * (v[best] - v[1.0])))
        x = b.copy()
        for a in chosen.unique():
            idx = chosen.index[chosen == a]
            x.loc[idx] = S[s][a].loc[idx, x.columns].to_numpy()
        r = stats.paired(x.score.to_numpy(float), b.score.to_numpy(float), groups=m.log.to_numpy())
        row = dict(arm=arm, seed=s, variant=tag, **{"dEPDMS no-EC [95% CI]": fmt(r)}, dEPDMS=100 * r["mean"], lo=100 * r["lo"], hi=100 * r["hi"],
                   tokens_moved=int((chosen != 1.0).sum()), slower_used=bool((chosen < 1.0).any()))
        for k in ("NC", "TTC", "DAC", "EP"):
            row[f"d{k}"] = 100 * (x[k].mean() - b[k].mean())
        row |= {"NC fail": int((x.NC < 1).sum()), "NC fail a=1": int((b.NC < 1).sum())}
        row |= split_gain(x, b)
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(picks)


def room(U, C):
    """Pre-registered room rule on the primary arm, per seed."""
    out = {}
    for s in SEEDS:
        u = U[(U.arm == PRIMARY) & (U.seed == s) & (U.a.isin([a for a in GRID if a != 1.0]))]
        def gate(r, slow):  # noqa: E306
            return r.lo > 0 and (not slow or r["(i) NC/TTC changed"] >= 0.5 * r.dEPDMS)
        uok = [float(r.a) for _, r in u.iterrows() if gate(r, r.a < 1)]
        c = C[(C.arm == PRIMARY) & (C.seed == s) & (C.variant == "speed band, 0.7-1.15")].iloc[0]
        out[f"s{s}"] = dict(U=uok, C=bool(gate(c, c.slower_used)), C_lo=float(c.lo))
    out["room"] = all(v["U"] or v["C"] for k, v in out.items())
    return out


def main():
    from jevdrive.data import splits
    from jevdrive.run import Run
    with Run("long_oracle", "report", config={}) as run:
        run.use_split(splits.load("navsim/navtest"))
        RES.mkdir(parents=True, exist_ok=True)
        Us, Cl, Nw, Cs, Pk, taxo = [], [], [], [], [], []
        for arm in ARMS:
            S, meta, F = load(arm)
            u, c, n = uniform(arm, S, meta, F)
            Us.append(u), Cl.append(c), Nw.append(n)
            for cells, grid, tag in ((["speed"], GRID, "speed band, 0.7-1.15"), (["speed"], WIDE, "speed band, 0.5-1.15"),
                                     (["speed", "pturn"], GRID, "speed band x plan turn, 0.7-1.15")):
                cc, pk = conditioned(arm, S, meta, cells, grid, tag)
                Cs.append(cc), Pk.append(pk)
            nt = pd.read_csv(ROOT / arm / "report/navtest_NC_class.csv")
            taxo.append(nt.assign(arm=arm))
            run.status(f"{arm} done")
        U, CL, NW, C, PK = (pd.concat(x, ignore_index=True) for x in (Us, Cl, Nw, Cs, Pk))
        for nm, t in (("uniform", U), ("uniform_by_class", CL), ("uniform_new_fail_cells", NW), ("conditioned", C), ("conditioned_picks", PK)):
            t.to_csv(RES / f"{nm}.csv", index=False)
        L = ["# long_oracle tables (generated by lo_report.py)\n"]
        show = ["a", "dEPDMS no-EC [95% CI]", "dNC", "dTTC", "dDAC", "dEP", "NC fail", "NC fixed", "NC new", "TTC-only fail", "TTC-only fixed",
                "TTC-only new", "(i) NC/TTC changed", "(ii) DAC only", "(iii) rest"]
        for arm in ARMS:
            for s in SEEDS:
                L.append(f"\n## Uniform scaling, {arm}-s{s} (points against a = 1.0)\n\n" + md(U[(U.arm == arm) & (U.seed == s)][show]))
        L.append("\n## Uniform scaling, seed mean of each arm\n")
        g = U.groupby(["arm", "a"])[["dEPDMS", "lo", "hi", "dNC", "dTTC", "dDAC", "dEP", "NC fail", "NC fixed", "NC new", "TTC-only fail",
                                       "(i) NC/TTC changed", "(ii) DAC only", "(iii) rest"]].mean().reset_index()
        L.append(md(g))
        g.to_csv(RES / "uniform_seed_mean.csv", index=False)
        cl = CL.groupby(["arm", "set", "cls", "a"])[["n", "fixed"]].sum().reset_index()
        cl["fixed %"] = 100 * cl.fixed / cl.n
        for arm in ARMS:
            for st in ("NC", "TTC-only"):
                t = cl[(cl.arm == arm) & (cl.set == st)].pivot_table(index="cls", columns="a", values="fixed %").reset_index()
                t.insert(1, "n (seed mean)", cl[(cl.arm == arm) & (cl.set == st) & (cl.a == 1.0)].set_index("cls").n.reindex(t.cls).to_numpy() / len(SEEDS))
                L.append(f"\n## {arm}: {st} failures of a = 1.0 that pass at uniform a, by class (% , both seeds pooled)\n\n" + md(t, 0))
        L.append("\n## Conditioned operating point (out-of-fold, points against a = 1.0)\n\n"
                 + md(C[["arm", "seed", "variant", "dEPDMS no-EC [95% CI]", "dNC", "dTTC", "dDAC", "dEP", "tokens_moved", "NC fail", "NC fail a=1",
                         "(i) NC/TTC changed", "(ii) DAC only", "(iii) rest"]]))
        pk = PK[PK.variant != "speed band x plan turn, 0.7-1.15"]
        L.append("\n## Conditioned picks (speed band variants)\n\n" + md(pk))
        tx = pd.concat(taxo, ignore_index=True)
        L.append("\n## NC classes, both arms (seed means, from nc_tax reports)\n\n" + md(tx.pivot_table(index="big", columns="arm", values="n").reset_index(), 1))
        R = room(U, C)
        L.append("\n## Pre-registered room rule (primary arm)\n\n```\n" + json.dumps(R, indent=1) + "\n```")
        (RES / "tables.md").write_text("\n".join(L) + "\n")
        (RES / "summary.json").write_text(json.dumps(dict(room=R), indent=1, default=float))
        run.summary.update(room=R["room"])
        run.info(json.dumps(R))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
