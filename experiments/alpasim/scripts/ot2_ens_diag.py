#!/usr/bin/env python3
"""Lane OT2 piece C: member disagreement of an ensemble run (plans/2026-10-09-ot2-ensemble-prereg.md, "what to answer on failure").

  ot2_ens_diag.py --manifest c0b/manifest.json ot2/c-ENS-OT30/manifest.json --ens ENS-OT30 --members OT30-F-s0 OT30-F-s1 --out DIR
Reads members.jsonl of the ensemble's run dirs (every member's own 8 poses on the state each decision was taken in) and the scene scores of
the ensemble and of its members' single-driver runs. A fork decision = the members' 4 s endpoints differ laterally by more than FORK m.
Plain python3 + numpy. Writes <ens>_diag.md / .json into --out.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FORK = 1.0


def scores(dirs):
    return {r["clipgt_id"]: r for d in dirs for r in json.loads((Path(d) / "aggregate/results-summary.json").read_text())["rollouts"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True), ap.add_argument("--ens", required=True)
    ap.add_argument("--members", nargs="+", required=True), ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {}
    for m in a.manifest:
        man |= json.loads(Path(m).read_text())
    E, M = scores(man[a.ens]), [scores(man[m]) for m in a.members]
    lat, dist, spd = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for d in man[a.ens]:
        for line in (Path(d) / "driver-logs/members.jsonl").open():
            r = json.loads(line)
            P = np.array(r["poses"])                               # (members, 8, 3)
            lat[r["scene"]][r["k"]] = float(np.ptp(P[:, -1, 1]))
            dist[r["scene"]][r["k"]] = float(np.linalg.norm(P[0, -1, :2] - P[-1, -1, :2]))
            spd[r["scene"]][r["k"]] = float(np.ptp(P[:, -1, 0]))
    scenes = sorted(s for s in E if s in lat and all(s in m for m in M))
    ez = np.array([E[s]["score"] == 0 for s in scenes])
    mz = np.array([[m[s]["score"] == 0 for s in scenes] for m in M])
    split, allz, nonez = mz.any(0) & ~mz.all(0), mz.all(0), ~mz.any(0)
    mx = np.array([max(lat[s].values()) for s in scenes])
    fork = mx > FORK
    alld = np.array([v for s in scenes for v in lat[s].values()])
    alls = np.array([v for s in scenes for v in spd[s].values()])
    R = dict(scenes=len(scenes), decisions=int(len(alld)), lat_median=float(np.median(alld)), lat_p90=float(np.percentile(alld, 90)), lat_p99=float(np.percentile(alld, 99)),
             fork_decisions=float((alld > FORK).mean()), long_median=float(np.median(alls)), long_p90=float(np.percentile(alls, 90)),
             fork_scenes=int(fork.sum()), ens_zero=int(ez.sum()),
             split=dict(n=int(split.sum()), ens_zero=int((ez & split).sum()), fork=int((fork & split).sum()), ens_zero_fork=int((ez & split & fork).sum())),
             all_zero=dict(n=int(allz.sum()), ens_zero=int((ez & allz).sum()), fork=int((fork & allz).sum())),
             none_zero=dict(n=int(nonez.sum()), ens_zero=int((ez & nonez).sum()), fork=int((fork & nonez).sum()), ens_zero_fork=int((ez & nonez & fork).sum())),
             zero_fork=int((ez & fork).sum()), nonzero_fork=int((~ez & fork).sum()),
             max_lat_zero_median=float(np.median(mx[ez])) if ez.any() else None, max_lat_nonzero_median=float(np.median(mx[~ez])))
    cls = lambda r: next((f for f in ("collision_at_fault", "offroad", "left_corridor_laterally") if r["score_metrics"].get(f)), "")  # noqa: E731
    L = [f"{a.ens} on {len(scenes)} scenes, {len(alld)} decisions; members {', '.join(a.members)}. Member disagreement on the shared state = spread of the members' "
         f"4 s endpoints; fork decision: lateral spread > {FORK} m.", "",
         f"- lateral spread per decision: median {R['lat_median']:.2f} m, p90 {R['lat_p90']:.2f} m, p99 {R['lat_p99']:.2f} m; fork decisions {100 * R['fork_decisions']:.1f} %; "
         f"longitudinal spread median {R['long_median']:.2f} m, p90 {R['long_p90']:.2f} m",
         f"- scenes with at least one fork decision: {R['fork_scenes']} of {len(scenes)}; among the ensemble's {R['ens_zero']} zeros {R['zero_fork']} "
         f"({100 * R['zero_fork'] / max(R['ens_zero'], 1):.0f} %), among its non-zero scenes {R['nonzero_fork']} ({100 * R['nonzero_fork'] / max(int((~ez).sum()), 1):.0f} %)",
         f"- largest lateral spread of a scene, median: ensemble zeros {R['max_lat_zero_median']} m, other scenes {R['max_lat_nonzero_median']:.2f} m", "",
         "| scenes by the members' single-driver result | scenes | ensemble zero | with a fork decision | ensemble zero with a fork decision |", "|:--|--:|--:|--:|--:|",
         f"| exactly one member zero (split) | {R['split']['n']} | {R['split']['ens_zero']} | {R['split']['fork']} | {R['split']['ens_zero_fork']} |",
         f"| all members zero | {R['all_zero']['n']} | {R['all_zero']['ens_zero']} | {R['all_zero']['fork']} | |",
         f"| no member zero | {R['none_zero']['n']} | {R['none_zero']['ens_zero']} | {R['none_zero']['fork']} | {R['none_zero']['ens_zero_fork']} |", "",
         "| ensemble zero scene | class | members' single-driver scores | largest lateral spread (m) | decision of it |", "|:--|:--|:--|--:|--:|"]
    for i, s in enumerate(scenes):
        if ez[i]:
            k = max(lat[s], key=lat[s].get)
            L.append(f"| {s} | {cls(E[s])} | {' / '.join((cls(m[s]) or format(m[s]['score'], '.2f')) for m in M)} | {mx[i]:.2f} | {k} |")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.ens}_diag.json").write_text(json.dumps(R, indent=1))
    (out / f"{a.ens}_diag.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
