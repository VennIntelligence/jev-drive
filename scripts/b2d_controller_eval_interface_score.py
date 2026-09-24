"""Score the frozen L1 plan-shape stress modes against pinned expert traces."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from b2d_controller_eval_l1_score import one


ROUTES = ('24240', '17563', '24816', '25845', '24252', '26990')
SEEDS = tuple(f'p{i:02d}' for i in range(1, 6))
MODES = ('nominal', 'short_2s', 'sparse_5s', 'stop_jitter')


def collect(sources):
    found = {}
    for root in sources:
        for path in root.glob('p0[1-5]/*/[ABCD]/*/validation.json'):
            key = (path.parents[2].name, path.parents[3].name, path.parents[1].name)
            if key in found:
                raise ValueError(f'Duplicate valid interface case {key}: {path}, {found[key]}')
            found[key] = path
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal', nargs='+', required=True, type=Path)
    for mode in MODES[1:]:
        parser.add_argument('--'+mode.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--cruises', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    refs = json.loads(args.references.read_text())
    cruises = json.loads(args.cruises.read_text())
    cases = {}
    for mode in MODES:
        roots = args.nominal if mode == 'nominal' else [getattr(args, mode)]
        cases[mode] = collect(roots)
    rows = []
    for route in ROUTES:
        for seed in SEEDS:
            for arm in 'ABCD':
                key = (route, seed, arm)
                for mode in MODES:
                    if key not in cases[mode]:
                        raise ValueError(f'Missing {mode}/{key}')
                    if route not in refs:
                        raise ValueError(f'Missing expert reference for {route}')
                    score = one(cases[mode][key], 'expert', Path(refs[route]),
                                cruises.get(route, cruises['default']))
                    rows.append(dict(mode=mode, **score))
    if len(rows) != 6*5*4*4:
        raise ValueError(f'Expected 480 stress scores, got {len(rows)}')
    by_key = {(r['route'], r['seed'], r['arm'], r['mode']): r for r in rows}
    summary = {}
    for arm in 'ABCD':
        ratios = []
        for route in ROUTES:
            for seed in SEEDS:
                nominal = by_key[(route, seed, arm, 'nominal')]['primary']
                ratios.append(max(by_key[(route, seed, arm, mode)]['primary']
                                  for mode in MODES) / nominal)
        summary[arm] = dict(cases=len(ratios), median_worst_ratio=float(np.median(ratios)),
                            max_worst_ratio=float(np.max(ratios)),
                            mode_median_primary={mode:float(np.median([
                                r['primary'] for r in rows if r['arm']==arm and r['mode']==mode]))
                                for mode in MODES},
                            mode_collisions={mode:sum(r['collision_count'] for r in rows
                                                       if r['arm']==arm and r['mode']==mode)
                                             for mode in MODES})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    args.out.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Wrote {len(rows)} interface scores and summary')


if __name__ == '__main__':
    main()
