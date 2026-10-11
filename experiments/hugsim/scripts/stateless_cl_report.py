#!/usr/bin/env python
"""Tables of the stateless HUGSIM closed loop (agent opt `stateless`) against the served `spec` runs of the same checkpoint. CPU only.
experiments/hugsim/plans/2026-10-11-stateless-cl-prereg.md, results/stateless_cl.md.

Runs are bench HUGSIM run dirs (units.csv + per scenario zs_steps.jsonl), given as label=dir; the first is the arm, the second the primary
reference, the rest further references (run-to-run spread). Writes into <out>:
  paired      every metric, arm - each reference and the references pairwise (jevdrive.stats.paired over scenarios), fixed / broken counts
  classes     per run: HD, RC, end classes, spins, launch stalls
  transitions end class of the primary reference -> end class of the arm
  launch      per run: time to 2 m/s, plan arc 0-3 s on decisions below 2 m/s, step-0 plan arc, plan length against speed; arm / reference ratios
  latency     per run: per-decision server / round-trip / agent time, wall per step
  identity    reference repeats against each other: scenarios with identical step-0 plan, identical plans throughout, identical end and HD
  per_scenario

  .venv/bin/python experiments/hugsim/scripts/stateless_cl_report.py --runs stateless=<dir> r3=<dir> rr1=<dir> rr2=<dir> --out <dir>
"""
import sys as _sys, pathlib as _pl  # noqa: E401
_R = _pl.Path(__file__).resolve().parents[3]
_sys.path[:0] = [str(_R)]
import argparse, json  # noqa: E401,E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jevdrive import stats  # noqa: E402
from jevdrive.run import Run, cli_args  # noqa: E402

NUM = ("hdscore", "rc", "nc", "dac", "ttc", "c", "pdms")
CLS = ("complete", "fg_coll", "bg_coll", "off_route", "stuck", "spin")
V_LAUNCH, EPS_HD = 2.0, 0.05


def arc3(plan):
    """Path length of a sent plan (8 points at 0.5 .. 4 s, origin = ego) over 0-3 s."""
    p = np.r_[[[0.0, 0.0]], np.asarray(plan, float)[:6]]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def steps(run_dir):
    """Per-decision records of one scenario (the setup line dropped)."""
    return [r for r in map(json.loads, open(Path(run_dir) / "zs_steps.jsonl")) if "step" in r]


def load(d):
    """units.csv + per scenario decision tables of one bench run."""
    u = pd.read_csv(Path(d) / "units.csv").set_index("scenario").sort_index()
    S = {}
    for sc, r in u.iterrows():
        rec = steps(r.run_dir)
        S[sc] = pd.DataFrame(dict(step=[x["step"] for x in rec], t=[float(x["t"]) for x in rec], v=[x["v"] for x in rec], arc3=[arc3(x["plan"]) for x in rec],
                                  infer_ms=[x.get("infer_ms") for x in rec], rtt_ms=[x.get("rtt_ms") for x in rec], agent_ms=[x.get("agent_ms") for x in rec],
                                  finite=[bool(np.isfinite(np.asarray(x["plan"], float)).all() and np.isfinite(x.get("kappa") or 0.0)) for x in rec]))
        S[sc].attrs["plans"] = [np.asarray(x["plan"], float) for x in rec]
        S[sc].attrs["sl"] = [x["sl"] for x in rec if "sl" in x]
    t2 = {sc: (float(s.t[s.v >= V_LAUNCH].iloc[0]) if (s.v >= V_LAUNCH).any() else np.nan) for sc, s in S.items()}
    u["t2"] = pd.Series(t2)
    return u, S


def ratio_ci(na, da, nb, db):
    """(sum na / sum da) / (sum nb / sum db) over scenarios, both arms resampled with the same scenario draw (jevdrive.stats' seed and count)."""
    na, da, nb, db = (np.asarray(x, float) for x in (na, da, nb, db))
    idx = np.random.default_rng(stats.SEED).integers(len(na), size=(stats.N_BOOT, len(na)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (na[idx].sum(1) / da[idx].sum(1)) / (nb[idx].sum(1) / db[idx].sum(1))
    lo, hi = np.nanquantile(r, [stats.ALPHA / 2, 1 - stats.ALPHA / 2])
    return dict(mean=float((na.sum() / da.sum()) / (nb.sum() / db.sum())), lo=float(lo), hi=float(hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=bench run dir; first = arm, second = primary reference")
    ap.add_argument("--out", required=True)
    cli_args(ap)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("hugsim_stateless_cl", "report", seed=a.seed, config=vars(a), resume=a.resume) as run:
        labels = [x.split("=", 1)[0] for x in a.runs]
        R = {x.split("=", 1)[0]: load(x.split("=", 1)[1]) for x in a.runs}
        arm, ref = labels[0], labels[1]
        common = sorted(set.intersection(*[set(R[k][0].index) for k in labels]))
        U = {k: R[k][0].loc[common] for k in labels}
        S = {k: R[k][1] for k in labels}
        run.info("%d scenarios common to %s", len(common), labels)

        # ---- classes
        cls = []
        for k in labels:
            u = U[k]
            cls.append(dict(run=k, n=len(u), HD=float(u.hdscore.mean()), RC=float(u.rc.mean()), **{c: int((u.cls == c).sum()) for c in CLS},
                            spin_any=int(u.spin.fillna(False).astype(bool).sum()), launch_stall=int(u.launch_stall.sum()),
                            never_2ms=int(u.t2.isna().sum()), steps=int(u.steps.sum()), non_finite_plans=int(sum((~S[k][sc].finite).sum() for sc in common))))
        stats.write_table(cls, out / "classes")

        # ---- paired metrics, fixed / broken
        pairs = [(arm, r) for r in labels[1:]] + [(x, y) for i, x in enumerate(labels[1:]) for y in labels[i + 2:]]
        paired = []
        for x, y in pairs:
            d = U[x].hdscore.values - U[y].hdscore.values
            for m in NUM:
                r = stats.paired(U[x][m].values, U[y][m].values)
                paired.append(dict(pair=f"{x} - {y}", metric=m, **r, **(dict(fixed=int((d > EPS_HD).sum()), broken=int((d < -EPS_HD).sum()), identical=int((d == 0).sum()),
                                                                           max_abs=float(np.abs(d).max()), same_end=int((U[x].cls.values == U[y].cls.values).sum())) if m == "hdscore" else {})))
        stats.write_table(paired, out / "paired")
        by = []
        for col in ("dataset", "difficulty"):
            for val in sorted(U[arm][col].astype(str).unique()):
                m = (U[arm][col].astype(str) == val).values
                by.append(dict(by=col, value=val, **stats.paired(U[arm].hdscore.values[m], U[ref].hdscore.values[m])))
        stats.write_table(by, out / "paired_strata")
        tr = pd.crosstab(U[ref].cls.rename(ref), U[arm].cls.rename(arm))
        tr.to_csv(out / "transitions.csv")

        # ---- launch and plan length
        def agg(k):
            rows = []
            for sc in common:
                s = S[k][sc]
                lo, warm = s.v < V_LAUNCH, (s.step >= 9) & (s.v > 3)
                rows.append(dict(arc_lo=s.arc3[lo].sum(), n_lo=int(lo.sum()), arc0=float(s.arc3.iloc[0]), arc_w=s.arc3[warm].sum(), len_w=3.0 * s.v[warm].sum(),
                                 t_lo=0.25 * int(lo.sum()), n=len(s)))
            return pd.DataFrame(rows, index=common)
        A = {k: agg(k) for k in labels}
        launch = []
        for k in labels:
            g, u = A[k], U[k]
            launch.append(dict(run=k, t2_median_s=float(u.t2.median()), t2_mean_reached_s=float(u.t2.mean()), never_2ms=int(u.t2.isna().sum()),
                               decisions_below_2ms=int(g.n_lo.sum()), share_below_2ms=float(g.n_lo.sum() / g.n.sum()),
                               arc3_below_2ms_m=float(g.arc_lo.sum() / max(g.n_lo.sum(), 1)), arc3_step0_m=float(g.arc0.mean()),
                               arc3_over_3v_warm=float(g.arc_w.sum() / g.len_w.sum())))
        stats.write_table(launch, out / "launch")
        lr = []
        for y in labels[1:]:
            ga, gb = A[arm], A[y]
            both = U[arm].t2.notna().values & U[y].t2.notna().values
            lr.append(dict(pair=f"{arm} vs {y}", quantity="time to 2 m/s, arm - ref (s), scenarios where both reach it", **stats.paired(U[arm].t2.values, U[y].t2.values), n_both=int(both.sum())))
            lr.append(dict(pair=f"{arm} vs {y}", quantity="time below 2 m/s per scenario, arm - ref (s)", **stats.paired(ga.t_lo.values, gb.t_lo.values)))
            lr.append(dict(pair=f"{arm} vs {y}", quantity="mean plan arc 0-3 s on decisions below 2 m/s, arm / ref", **ratio_ci(ga.arc_lo, ga.n_lo, gb.arc_lo, gb.n_lo)))
            lr.append(dict(pair=f"{arm} vs {y}", quantity="step-0 plan arc 0-3 s, arm / ref", **ratio_ci(ga.arc0, np.ones(len(ga)), gb.arc0, np.ones(len(gb)))))
            lr.append(dict(pair=f"{arm} vs {y}", quantity="plan arc 0-3 s / (3 s x v), warm v > 3 m/s, arm / ref", **ratio_ci(ga.arc_w, ga.len_w, gb.arc_w, gb.len_w)))
        stats.write_table(lr, out / "launch_pairs")

        # ---- latency
        q = lambda x: dict(median=float(np.nanmedian(x)), p95=float(np.nanquantile(x, 0.95)), p99=float(np.nanquantile(x, 0.99)), max=float(np.nanmax(x)))  # noqa: E731
        lat = []
        for k in labels:
            D = pd.concat([S[k][sc] for sc in common])
            later = D[D.step > 0]
            for name, x in (("server infer_ms, steps > 0", later.infer_ms), ("agent round trip rtt_ms, steps > 0", later.rtt_ms), ("agent total agent_ms, steps > 0", later.agent_ms),
                            ("agent total agent_ms, step 0", D[D.step == 0].agent_ms)):
                lat.append(dict(run=k, quantity=name, n=int(x.notna().sum()), **q(x.astype(float).values)))
            sl = [x for sc in common for x in S[k][sc].attrs["sl"]]
            for part in ("frames", "encode", "policy", "export", "action"):
                if sl:
                    lat.append(dict(run=k, quantity=f"server stage {part} ms", n=len(sl), **q(np.array([x["ms"][part] for x in sl], float))))
            if sl:
                lat.append(dict(run=k, quantity="server lock wait queue_ms", n=len(sl), **q(np.array([x["queue_ms"] for x in sl], float))))
            lat.append(dict(run=k, quantity="scenario wall_s / step", n=len(common), **q((U[k].wall_s / U[k].steps.clip(lower=1)).values)))
            lat.append(dict(run=k, quantity="scenario wall_s (sum = card-seconds of scenario slots)", n=len(common), **q(U[k].wall_s.values), sum=float(U[k].wall_s.sum())))
        stats.write_table(lat, out / "latency")

        # ---- identity of repeats (the off-path check: option off under the new code against the earlier repeats)
        ident = []
        for i, x in enumerate(labels[1:]):
            for y in labels[i + 2:]:
                n0 = nall = 0
                dev = []
                for sc in common:
                    px, py = S[x][sc].attrs["plans"], S[y][sc].attrs["plans"]
                    n0 += bool(np.array_equal(px[0], py[0]))
                    same = len(px) == len(py) and all(np.array_equal(p, q_) for p, q_ in zip(px, py))
                    nall += bool(same)
                    m = min(len(px), len(py))
                    dev.append(max(float(np.abs(px[j] - py[j]).max()) for j in range(m)))
                d = U[x].hdscore.values - U[y].hdscore.values
                ident.append(dict(pair=f"{x} vs {y}", scenarios=len(common), step0_plan_identical=n0, all_plans_identical=nall, same_end=int((U[x].cls.values == U[y].cls.values).sum()),
                                  same_steps=int((U[x].steps.values == U[y].steps.values).sum()), hd_identical=int((d == 0).sum()), hd_abs_diff_max=float(np.abs(d).max()),
                                  hd_abs_diff_mean=float(np.abs(d).mean()), hd_mean_diff=float(d.mean()), plan_dev_median_m=float(np.median(dev)), plan_dev_max_m=float(np.max(dev))))
        stats.write_table(ident, out / "identity")

        per = pd.DataFrame({"dataset": U[arm].dataset, "difficulty": U[arm].difficulty})
        for k in labels:
            per[f"hd_{k}"], per[f"cls_{k}"], per[f"stall_{k}"], per[f"t2_{k}"], per[f"steps_{k}"] = U[k].hdscore, U[k].cls, U[k].launch_stall, U[k].t2, U[k].steps
        per[f"dhd_{arm}_{ref}"] = U[arm].hdscore - U[ref].hdscore
        per.to_csv(out / "per_scenario.csv")
        head = next(r for r in paired if r["pair"] == f"{arm} - {ref}" and r["metric"] == "hdscore")
        (out / "summary.json").write_text(json.dumps(dict(scenarios=len(common), arm=arm, ref=ref, hd=head, classes=cls, launch=launch), indent=1, default=str))
        run.summary.update(n=len(common), dHD=head["mean"], lo=head["lo"], hi=head["hi"])
        run.info("%s - %s HD %s (fixed %d, broken %d)", arm, ref, stats.fmt(head), head["fixed"], head["broken"])


if __name__ == "__main__":
    main()
