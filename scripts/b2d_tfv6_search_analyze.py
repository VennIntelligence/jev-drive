"""Score Task 8 search cases against the unchanged W2 A/B cases."""
from pathlib import Path
import argparse
import json

import numpy as np

import b2d_tfv6_campaign as base


OLD = Path('/data/runs/b2d/tfv6-w2/formal')


def official(done):
    record = json.loads((Path(done['run_dir']) / 'attempts' / done['route'] / '1' /
                         'results.json').read_text())['_checkpoint']['records'][0]
    scores = record['scores']
    infractions = record['infractions']
    return {'ds': float(scores['score_composed']),
            'rc': float(scores['score_route']),
            'complete': int(scores['score_route'] >= 100),
            'sr': int(record['status'] in ('Completed', 'Perfect') and
                      all(not events for name, events in infractions.items()
                          if name != 'min_speed_infractions')),
            'official_status': record['status']}


def read_case(root, level, route, seed, arm):
    path = root / 'cases' / level / f'route-{route}' / f'seed-{seed}' / arm / 'done.json'
    return official(json.loads(path.read_text())) if path.exists() else None


def cluster_ci(rows, field, draws=10000):
    routes = sorted({row['route'] for row in rows}, key=int)
    if not routes:
        return None
    route_values = {route: np.array([r[field] for r in rows if r['route'] == route])
                    for route in routes}
    rng = np.random.default_rng(20260923)
    samples = np.empty(draws)
    for i in range(draws):
        sampled = rng.choice(routes, len(routes), replace=True)
        samples[i] = np.mean(np.concatenate([route_values[route] for route in sampled]))
    return {'mean': float(np.mean([r[field] for r in rows])),
            'ci_low': float(np.percentile(samples, 2.5)),
            'ci_high': float(np.percentile(samples, 97.5)),
            'n_pairs': len(rows), 'n_routes': len(routes)}


def analyze(out, dataset, arms):
    level = '1' if dataset == 'dev10' else '2'
    old = OLD / ('level1' if level == '1' else 'level2')
    routes = base.routes_in(base.DEV10 if level == '1' else base.HOLDOUT)
    cases = []
    for route in routes:
        for seed in range(3):
            baseline = {arm: read_case(old, level, route, seed, arm) for arm in 'AB'}
            if not all(baseline.values()):
                raise ValueError(f'Missing W2 A/B reference: {route}/{seed}')
            for arm in arms:
                candidate = read_case(out, level, route, seed, arm)
                if candidate is None:
                    continue
                cases.append({'route': route, 'seed': seed, 'arm': arm, **candidate,
                              **{f'{field}_minus_{reference}': candidate[field] - baseline[reference][field]
                                 for reference in 'AB' for field in ('ds', 'rc', 'complete', 'sr')}})
    summary = {}
    for arm in arms:
        rows = [r for r in cases if r['arm'] == arm]
        summary[arm] = {'n': len(rows), 'routes': len({r['route'] for r in rows}),
                        'mean_ds': float(np.mean([r['ds'] for r in rows])) if rows else None,
                        'mean_rc': float(np.mean([r['rc'] for r in rows])) if rows else None,
                        'complete': sum(r['complete'] for r in rows),
                        'sr': sum(r['sr'] for r in rows),
                        **{f'{field}_minus_{reference}': cluster_ci(rows, f'{field}_minus_{reference}')
                           for reference in 'AB' for field in ('ds', 'rc', 'complete', 'sr')}}
    return {'dataset': dataset, 'cases': cases, 'summary': summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('dev10', 'holdout'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arms', required=True)
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    result = analyze(args.out, args.dataset, args.arms)
    content = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(content)
    else:
        print(content)


if __name__ == '__main__':
    main()
