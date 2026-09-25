"""Task 10 L1 v3 scorer: merges the v2 (A-D), P and v3 (P2, P3) batches per reference and interface.

Nominal references: ramp, profile, crawl (held-out 40 routes). Interfaces: profile reference on the
interface route subset at p01. Reports per-arm medians and status counts, paired route-cluster
contrasts, and per-interface robustness (primary / nominal profile primary on the same route).
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from b2d_controller_eval_l1_v2_score import METRICS, contrast, score_batch

INTERFACES = ('short_2s', 'sparse_5s', 'stop_jitter', 'model_noise', 'stale_5hz', 'stale_2hz', 'stale_1hz', 'pose_plan')
PAIRS = [('P7', 'P5'), ('P7', 'D'), ('P7', 'C'), ('P5', 'P4'), ('P5', 'D'), ('P5', 'C'), ('P4', 'P3'), ('P4', 'P2'), ('P4', 'D'), ('P4', 'C'), ('P3', 'P2'), ('P2', 'D'), ('P3', 'D'), ('P2', 'C'), ('P3', 'C'), ('D', 'C'), ('C', 'B'), ('P2', 'B'), ('P3', 'B')]


def gather(roots, kind, refs, cruises, label):
    rows = {}
    for root in roots:
        if root.exists():
            for r in score_batch(root, kind, refs, cruises, label=label):
                rows[(r['route'], r['seed'], r['arm'])] = r
    return list(rows.values())


def table(rows):
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        group = [r for r in rows if r['arm'] == arm]
        out[arm] = dict(cases=len(group), status=dict(Counter(r['status'] for r in group)),
                        **{m: float(np.median([r[m] for r in group])) for m in METRICS})
    return out


def pairs(rows):
    arms = {r['arm'] for r in rows}
    return {f'{a}-{b}': {m: contrast(rows, a, b, m) for m in ('primary', 'extra_jerk_rms_mps3')}
            for a, b in PAIRS if a in arms and b in arms}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v2', type=Path, default=Path('/data/runs/b2d/controller-eval/v2'))
    p.add_argument('--cruises', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cruises = json.loads(a.cruises.read_text())
    V2, refs = a.v2, a.v2 / 'refs'
    summary, cases = {'heldout': {}, 'interface': {}, 'dev': {}}, []
    for kind in ('ramp', 'profile', 'crawl'):
        rows = gather([V2 / f'l1-{kind}', V2 / f'l1-P-{kind}'] + sorted(V2.glob(f'l1-v*-{kind}')), kind, refs, cruises, kind)
        cases += rows
        summary['heldout'][kind] = dict(table=table(rows), contrasts=pairs(rows))
    nominal = {(r['route'], r['arm']): r['primary'] for r in cases if r['reference'] == 'profile' and r['seed'] == 'p01'}
    for mode in INTERFACES:
        rows = gather([V2 / f'l1-interface-{mode}', V2 / f'l1-P-interface-{mode}'] + sorted(V2.glob(f'l1-v*-if-{mode}')),
                      'profile', refs, cruises, f'profile/{mode}')
        cases += rows
        tab = table(rows)
        for arm in tab:
            ratio = [r['primary'] / max(nominal[(r['route'], arm)], 1e-9) for r in rows
                     if r['arm'] == arm and (r['route'], arm) in nominal]
            tab[arm]['median_over_nominal'] = float(np.median(ratio)) if ratio else None
        summary['interface'][mode] = tab
    dev_refs = V2 / 'dev' / 'refs'
    for name in ('ramp', 'profile', 'crawl') + ('short_2s', 'model_noise', 'stale_2hz', 'stale_1hz'):
        kind = name if name in ('ramp', 'profile', 'crawl') else 'profile'
        pattern = f'v*-{name}' if kind == name else f'v*-if-{name}'
        rows = gather(sorted((V2 / 'dev').glob(pattern)), kind, dev_refs, cruises, f'dev/{name}')
        if rows:
            summary['dev'][name] = dict(table=table(rows), contrasts=pairs(rows))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'l1-v3-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import csv
    with (a.out / 'l1-v3-cases.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0]), lineterminator='\n')
        w.writeheader(); w.writerows(cases)
    for kind, block in summary['heldout'].items():
        print(kind, {arm: (round(t['primary'], 3), round(t['extra_jerk_rms_mps3'], 2), t['status']) for arm, t in block['table'].items()})
    for mode, tab in summary['interface'].items():
        print(mode, {arm: (round(t['primary'], 3), t['status'].get('completed', 0), t['cases']) for arm, t in tab.items()})


if __name__ == '__main__':
    main()
