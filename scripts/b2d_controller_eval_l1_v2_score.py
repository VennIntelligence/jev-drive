"""Score Task 10 v2 L1 (protocol v2): ramp and profile references, fed and scored identically.

Reuses the v1 per-case metric code (b2d_controller_eval_l1_score.one) on fixed-trace runs, checks
that each run was fed byte-identical reference files, then reports per-arm medians, paired
route-cluster bootstrap contrasts (seeds averaged within route) and interface robustness.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from b2d_controller_eval_l1_score import one

METRICS = ('primary', 'cte_rms_m', 'time_xy_rms_m', 'speed_rms_mps', 'extra_jerk_rms_mps3')
PAIRS = [('C', 'A'), ('C', 'B'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('B', 'A')]


def score_batch(root, kind, refs, cruises, label=None):
    rows = []
    manifest = json.loads((refs / f'{kind}-traces.json').read_text())
    for done in sorted(root.glob('route-*/done.json')):
        rid = done.parent.name.removeprefix('route-')
        reference = Path(manifest[rid])
        for key, case in json.loads(done.read_text()).items():
            case = Path(case)
            fed = case.parents[3] / 'inputs' / f'expert-{rid}.json'
            if fed.read_bytes() != reference.read_bytes():
                raise ValueError(f'{case}: fed reference differs from {reference}')
            row = one(case / 'validation.json', 'expert', reference, cruises.get(rid, cruises['default']))
            row.update(reference=label or kind, route=rid)
            rows.append(row)
    return rows


def route_means(rows, arm, metric):
    out = {}
    for r in rows:
        if r['arm'] == arm:
            out.setdefault(r['route'], []).append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def contrast(rows, left, right, metric, draws=10000):
    a, b = route_means(rows, left, metric), route_means(rows, right, metric)
    if set(a) != set(b):
        raise ValueError(f'unpaired routes for {left}-{right}')
    x = np.asarray([a[k] - b[k] for k in sorted(a)])
    rng = np.random.default_rng(20260924)
    boot = x[rng.integers(0, len(x), (draws, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), ci95=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                routes=len(x), left_better_routes=int(np.sum(x < 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v2', type=Path, default=Path('/data/runs/b2d/controller-eval/v2'))
    p.add_argument('--cruises', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cruises = json.loads(a.cruises.read_text())
    refs = a.v2 / 'refs'
    rows = []
    for kind in ('ramp', 'profile'):
        batch = score_batch(a.v2 / f'l1-{kind}', kind, refs, cruises)
        if len(batch) != 40 * 3 * 4:
            raise ValueError(f'{kind}: expected 480 cases, got {len(batch)}')
        rows += batch
    interface = []
    for mode in ('short_2s', 'sparse_5s', 'stop_jitter', 'stale_5hz', 'pose_plan'):
        root = a.v2 / f'l1-interface-{mode}'
        if root.exists():
            interface += score_batch(root, 'profile', refs, cruises, label=f'profile/{mode}')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'l1-v2-cases.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows + interface)
    summary = {}
    for kind in ('ramp', 'profile'):
        group = [r for r in rows if r['reference'] == kind]
        summary[kind] = dict(
            median={arm: {m: float(np.median([r[m] for r in group if r['arm'] == arm])) for m in METRICS}
                    for arm in 'ABCD'},
            status={arm: {s: sum(r['status'] == s for r in group if r['arm'] == arm)
                          for s in sorted({r['status'] for r in group})} for arm in 'ABCD'},
            contrasts={f'{l}-{r}': {m: contrast(group, l, r, m) for m in METRICS} for l, r in PAIRS})
    if interface:
        nominal = {(r['route'], r['arm']): r['primary'] for r in rows
                   if r['reference'] == 'profile' and r['seed'] == 'p01'}
        worst = {}
        for r in interface:
            key = (r['route'], r['arm'])
            worst[key] = max(worst.get(key, 0.), r['primary'] / max(nominal[key], 1e-9))
        summary['interface_robustness'] = {
            arm: dict(median_worst_over_nominal=float(np.median([v for (_, x), v in worst.items() if x == arm])),
                      routes=sum(x == arm for _, x in worst))
            for arm in 'ABCD'}
        per_mode = {}
        for mode in ('short_2s', 'sparse_5s', 'stop_jitter', 'stale_5hz', 'pose_plan'):
            group = [r for r in interface if r['reference'] == f'profile/{mode}']
            if group:
                per_mode[mode] = {arm: float(np.median([r['primary'] for r in group if r['arm'] == arm]))
                                  for arm in 'ABCD'}
        summary['interface_primary_median'] = per_mode
    verdict = {}
    for arm in 'CD':
        verdict[arm] = {base: all(summary[k]['contrasts'][f'{arm}-{base}']['primary']['ci95'][1] < 0
                                  for k in ('ramp', 'profile')) for base in 'AB'}
    summary['l1_significantly_better'] = verdict
    (a.out / 'l1-v2-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['l1_significantly_better']))


if __name__ == '__main__':
    main()
