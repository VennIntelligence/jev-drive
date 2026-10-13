#!/usr/bin/env python3
"""Plan 2 s deficit on stored AlpaSim reads (any decision count; experiments/route_follow/plans/2026-10-13-allrows-prereg.md line D): how much
shorter the served plan is than the car's own speed. rf_follow's definition without the route regeneration, so it runs on the public 1491 /
10-decision reads. CPU only, read-only. Box, AlpaSim's venv.

  rf_deficit.py --runs LABEL=<run dir>[,<run dir>..] ... [--mean NAME=L1,L2 ..] [--pairs A:B ..] --out DIR

Per decision of the last driver session of a scene (ao_e1.drive_records): deficit = arc length of the served plan from the rear axle through its
0.5 .. 2.0 s poses minus the fed speed x 2 s (m; rf_follow.plan_arc2), kept when the fed speed >= 1 m/s and k <= k_z (the first zero-score event
step of long_read.one: at-fault, or dist_to_gt >= 4 m). Per rollout: mean deficit over its kept decisions. Per label: pooled decision median
(rf_follow's number) and the mean of the rollout means. --mean NAME=L1,L2: per scene the mean over the listed labels (seed mean; scenes in all).
--pairs A:B: mean over common scenes of (A - B) rollout means, log-clustered bootstrap (jevdrive.stats.paired, log = scene id without its
last "-<hash>"). Positive = A's plan is less short. -> DIR/deficit.json, deficit.md, scenes.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rf_follow as RF  # noqa: E402  (sets the repo paths)

T0, DT = RF.T0, RF.DT


def one(job):
    label, D, scene, recs = job
    import long_read as LR
    ev = LR.one((label, D, scene))
    kz = min([r["k"] for r in ev if r["atf"] or (r["dist_gt"] or 0) >= 4.0], default=10 ** 6)
    d = []
    for r in recs:
        v = float(np.hypot(*np.asarray(r["ego"], float)[4:6]) * 10.0)
        if v >= 1.0 and (r["t0"] - T0) // DT <= kz:
            d.append(RF.plan_arc2(r["poses"]) - 2.0 * v)
    return d


def main(a):
    import pandas as pd
    from ao_e1 import drive_records
    from jevdrive import par, stats
    from jevdrive.common import n_cpus
    from jevdrive.run import Run
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Run("route_follow", "deficit", config=vars(a)) as run:
        jobs = []
        for spec in a.runs:
            label, dirs = spec.split("=", 1)
            seen = set()
            for D in dirs.split(","):
                for sc, rr in sorted(drive_records(D).items()):
                    if sc not in seen:
                        seen.add(sc)
                        jobs.append((label, D, sc, [dict(t0=r["t0"], ego=r["ego"][:6], poses=r["poses"][:4]) for r in rr]))
            run.info(f"{label}: {len(seen)} rollouts")
        res = par.pmap(one, jobs, run=run, workers=min(a.workers or n_cpus(), len(jobs)), desc="rollouts")
        if res.errors:
            run.info(f"{len(res.errors)} rollouts failed: {[(jobs[i][0], jobs[i][2], str(e)[-200:]) for i, e in list(res.errors.items())[:3]]}")
        rows = [dict(label=j[0], scene=j[2], n=len(v), mean=float(np.mean(v)) if v else np.nan) for j, v in zip(jobs, res.values) if v is not None]
        dec = {}
        for j, v in zip(jobs, res.values):
            if v:
                dec.setdefault(j[0], []).extend(v)
        S = pd.DataFrame(rows)
        S.to_csv(out / "scenes.csv", index=False)
        P = S.pivot(index="scene", columns="label", values="mean")
        groups = {lab: [lab] for lab in P.columns}
        for m in a.mean or []:
            name, ls = m.split("=", 1)
            groups[name] = ls.split(",")
            P[name] = P[groups[name]].mean(1, skipna=False)
            dec[name] = [x for lab in groups[name] for x in dec.get(lab, [])]
        R = dict(labels={}, pairs={})
        for lab in P.columns:
            x = P[lab]
            R["labels"][lab] = dict(scenes=int(x.notna().sum()), decisions=len(dec.get(lab, [])), median_dec=float(np.median(dec[lab])) if dec.get(lab) else np.nan,
                                    mean_scene=float(x.mean()))
        logs = np.array([s.rsplit("-", 1)[0] for s in P.index])
        for p in a.pairs or []:
            A, B = p.split(":")
            R["pairs"][p] = stats.paired(P[A].to_numpy(), P[B].to_numpy(), groups=logs)
        (out / "deficit.json").write_text(json.dumps(R, indent=1, default=float))
        md = ["# Plan 2 s deficit (m): plan arc to 2 s - fed speed x 2 s, decisions at >= 1 m/s up to the first zero-score event", "",
              "| label | scenes | decisions | median (decisions) | mean of rollout means |", "|:--|--:|--:|--:|--:|"]
        md += [f"| {k} | {r['scenes']} | {r['decisions']} | {r['median_dec']:+.3f} | {r['mean_scene']:+.3f} |" for k, r in R["labels"].items()]
        if R["pairs"]:
            md += ["", "| A - B (rollout means, common scenes) | scenes / logs | difference [log-clustered 95 % CI] |", "|:--|--:|:--|"]
            md += [f"| {k} | {r['n']} / {r['units']} | {stats.fmt(r, '+.3f')} |" for k, r in R["pairs"].items()]
        (out / "deficit.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mean", nargs="*")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    main(ap.parse_args())
