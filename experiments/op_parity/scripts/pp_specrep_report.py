"""Repeats of spec vs spec_plan_smooth on HUGSIM-64 (decision 149): pooled paired contrasts, end classes, spins, between-repeat noise.

Runs on the box (reads $DATA_DIR/runs/bench/hugsim and the stored op_parity runs):
  .venv/bin/python experiments/op_parity/scripts/pp_specrep_report.py [--out experiments/op_parity/results/hugsim_specplan/repeats]

Repeats: r0 = the stored 2026-10-06/07 run (op_parity results.csv, tags pp-spec-* / pp-specplansmooth-*), r1 / r2 = bench runs
`<arm>_<preset>-rr1/-rr2`. Pooling: HD averaged over repeats within (arm, preset, scenario), then over the 4 arms -> one value per
scenario and preset; the contrast smooth - spec is bootstrapped over scenarios (jevdrive.stats, B 10000, seed 0).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jevdrive import stats
from jevdrive.bench import tables as T
from jevdrive.bench.sets import hugsim_scenarios

ARMS = ["P2-F-s0", "P2-F-s1", "P2H10-F-s0", "P2H10-F-s1"]
PRESETS = ["spec", "spec_plan_smooth"]
REPS = {"r0": None, "r1": "rr1", "r2": "rr2"}
CLS = ["complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin"]


def load_all():
    D, src = {}, {}
    for a in ARMS:
        for p in PRESETS:
            for r, suf in REPS.items():
                u, s = T.load("hugsim", a, p) if suf is None else T.load("hugsim", f"run:{a}_{p}-{suf}")
                if u is None:
                    raise SystemExit(f"missing {a} {p} {r}")
                D[a, p, r], src[a, p, r] = u, s
    return D, src


def boot(d, sets):
    return {k: stats.bootstrap(d[d.index.isin(v)].to_numpy()) for k, v in sets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/op_parity/results/hugsim_specplan/repeats")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D, src = load_all()
    scen = sorted(set.intersection(*[set(u.index) for u in D.values()]))
    turn = {Path(s).stem for s in hugsim_scenarios("turn23")}
    sets = {"all64": set(scen), "turning23": turn & set(scen), "straight41": set(scen) - turn}
    hd = pd.DataFrame({(arm, p, r): D[arm, p, r].loc[scen, "hdscore"] for arm in ARMS for p in PRESETS for r in REPS})
    md = ["# Repeats: spec vs spec_plan_smooth on HUGSIM-64", "",
          f"{len(scen)} scenarios, turning {len(sets['turning23'])}, straight {len(sets['straight41'])}; arms {', '.join(ARMS)}. "
          "Sources: " + "; ".join(sorted({s[:60] for s in src.values()})), ""]

    def pooled(reps, p):
        return sum(hd[(arm, p, r)] for arm in ARMS for r in reps) / (len(ARMS) * len(reps))

    rows = []
    for name, reps in [("new repeats r1 + r2", ["r1", "r2"]), ("all three r0 + r1 + r2", ["r0", "r1", "r2"]),
                       ("stored r0 only", ["r0"]), ("r1 only", ["r1"]), ("r2 only", ["r2"])]:
        s, m = pooled(reps, "spec"), pooled(reps, "spec_plan_smooth")
        for k, v in sets.items():
            idx = sorted(v)
            r = stats.paired(m.loc[idx].to_numpy(), s.loc[idx].to_numpy())
            d = (m - s).loc[idx]
            rows.append(dict(pool=name, set=k, n=len(idx), spec=r["mean_b"], smooth=r["mean_a"], diff=r["mean"], lo=r["lo"], hi=r["hi"],
                             wins=int((d > 0.02).sum()), losses=int((d < -0.02).sum()), ties=int((d.abs() <= 0.02).sum())))
    C = pd.DataFrame(rows)
    C.to_csv(out / "contrasts.csv", index=False)
    md += ["## Paired contrast smooth - spec (HD-Score; scenario = unit, arms and repeats averaged, bootstrap over scenarios; wins / losses / ties at |d| 0.02)", "",
           "| pool | set | spec | smooth | smooth - spec [95% CI] | W / L / T |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pool']} | {r['set']} | {r['spec']:.3f} | {r['smooth']:.3f} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                  f"{r['wins']} / {r['losses']} / {r['ties']} |")

    # per-arm contrast (all three repeats)
    arows = []
    for arm in ARMS:
        for k, v in sets.items():
            idx = sorted(v)
            s = sum(hd[(arm, "spec", r)] for r in REPS) / 3
            m = sum(hd[(arm, "spec_plan_smooth", r)] for r in REPS) / 3
            r = stats.paired(m.loc[idx].to_numpy(), s.loc[idx].to_numpy())
            arows.append(dict(arm=arm, set=k, spec=r["mean_b"], smooth=r["mean_a"], diff=r["mean"], lo=r["lo"], hi=r["hi"]))
    A = pd.DataFrame(arows)
    A.to_csv(out / "per_arm.csv", index=False)
    md += ["", "## Per arm (3 repeats averaged)", "", "| arm | set | spec | smooth | smooth - spec |", "|---|---|---|---|---|"]
    md += [f"| {r.arm} | {r.set} | {r.spec:.3f} | {r.smooth:.3f} | {r.diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] |" for r in A.itertuples()]

    # end classes and spins: per arm-repeat counts, averaged
    crow = []
    for p in PRESETS:
        for k, v in sets.items():
            cnt = {c: [] for c in CLS}
            for arm in ARMS:
                for r in REPS:
                    u = D[arm, p, r].loc[sorted(v)]
                    for c in CLS:
                        cnt[c].append(int((u.cls == c).sum()))
            crow.append(dict(preset=p, set=k, **{c: float(np.mean(x)) for c, x in cnt.items()}))
    K = pd.DataFrame(crow)
    K.to_csv(out / "classes.csv", index=False)
    md += ["", "## End classes (mean count per arm x repeat, 12 runs per preset; cls = spin, else the end)", "",
           "| preset | set | " + " | ".join(CLS) + " |", "|---|---|" + "---|" * len(CLS)]
    md += [f"| {r.preset} | {r.set} | " + " | ".join(f"{getattr(r, c):.2f}" for c in CLS) + " |" for r in K.itertuples()]
    srow = []
    for p in PRESETS:
        for arm in ARMS:
            srow.append(dict(preset=p, arm=arm, **{r: int((D[arm, p, r].cls == "spin").sum()) for r in REPS}))
    S = pd.DataFrame(srow)
    S.to_csv(out / "spins.csv", index=False)
    md += ["", "## Spins over all 64 (per repeat r0 / r1 / r2)", "", "| preset | arm | r0 | r1 | r2 |", "|---|---|---|---|---|"]
    md += [f"| {r.preset} | {r.arm} | {r.r0} | {r.r1} | {r.r2} |" for r in S.itertuples()]

    # noise: per scenario spread over the 3 repeats of one (arm, preset)
    nrow = []
    for p in PRESETS:
        for arm in ARMS:
            x = hd[[(arm, p, r) for r in REPS]].to_numpy()
            sd = x.std(1, ddof=1)
            ends = np.stack([D[arm, p, r].loc[scen, "end"].to_numpy() for r in REPS], 1)
            same_end = np.array([len(set(e)) == 1 for e in ends])
            ident = (x.max(1) - x.min(1)) < 1e-9
            nrow.append(dict(preset=p, arm=arm, sd_mean=sd.mean(), sd_median=float(np.median(sd)), sd_max=sd.max(), range_max=(x.max(1) - x.min(1)).max(),
                             identical_hd=int(ident.sum()), same_end=int(same_end.sum()), n_range_gt_005=int(((x.max(1) - x.min(1)) > 0.05).sum()),
                             n_range_gt_02=int(((x.max(1) - x.min(1)) > 0.2).sum()),
                             hd_r0=x[:, 0].mean(), hd_r1=x[:, 1].mean(), hd_r2=x[:, 2].mean()))
    N = pd.DataFrame(nrow)
    N.to_csv(out / "noise.csv", index=False)
    md += ["", "## Between-repeat noise per scenario (3 repeats of the same arm and preset)", "",
           "| preset | arm | HD r0 / r1 / r2 | sd mean | sd median | sd max | max range | identical HD | same end | range > 0.05 | range > 0.2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.preset} | {r.arm} | {r.hd_r0:.3f} / {r.hd_r1:.3f} / {r.hd_r2:.3f} | {r.sd_mean:.3f} | {r.sd_median:.3f} | {r.sd_max:.3f} | {r.range_max:.3f} | "
           f"{r.identical_hd} / 64 | {r.same_end} / 64 | {r.n_range_gt_005} | {r.n_range_gt_02} |" for r in N.itertuples()]
    # noise of the pooled scenario value and of the contrast
    md += ["", "Per-repeat contrast (smooth - spec, 4-arm mean, one repeat at a time): " +
           "; ".join(f"{r['pool']} {r['set']} {r['diff']:+.3f}" for r in rows if r["pool"].endswith("only") and r["set"] != "straight41") + "."]
    unst = []
    for p in PRESETS:
        pm = pd.DataFrame({r: pooled([r], p) for r in REPS})
        unst.append(f"{p}: scenario-level sd of the 4-arm mean over repeats mean {pm.std(1, ddof=1).mean():.3f}, max {pm.std(1, ddof=1).max():.3f}")
    md += ["", "Noise of the 4-arm scenario value (the unit of the bootstrap): " + "; ".join(unst) + "."]
    # scenarios with the biggest swing under the contrast
    d3 = (pooled(list(REPS), "spec_plan_smooth") - pooled(list(REPS), "spec")).sort_values()
    md += ["", "Largest scenario contrasts (3 repeats, 4 arms): worst " + ", ".join(f"{i} {v:+.2f}" for i, v in d3.head(5).items()) +
           "; best " + ", ".join(f"{i} {v:+.2f}" for i, v in d3.tail(5)[::-1].items()) + "."]
    (out / "repeats.md").write_text("\n".join(md) + "\n")
    hd.columns = ["|".join(c) for c in hd.columns]
    hd.to_csv(out / "hd_by_run.csv")
    print("\n".join(md))


if __name__ == "__main__":
    main()
