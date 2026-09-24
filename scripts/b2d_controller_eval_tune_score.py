"""Score a dev L1 tuning round and apply the pre-registered selection rule (tune/README.md).

Eligible: median primary <= 1.05 x D's and collision+blocked count <= D's; the eligible arm with
the lowest median extra jerk wins. Medians pool routes x references (ramp, profile) at p01.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from b2d_controller_eval_l1_v2_score import METRICS, contrast, score_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dev', type=Path, default=Path('/data/runs/b2d/controller-eval/v2/dev'))
    p.add_argument('--round', required=True)
    p.add_argument('--cruises', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cruises = json.loads(a.cruises.read_text())
    rows = []
    for kind in ('ramp', 'profile'):
        rows += score_batch(a.dev / f'{a.round}-{kind}', kind, a.dev / 'refs', cruises)
    arms = sorted({r['arm'] for r in rows})
    table = {}
    for arm in arms:
        group = [r for r in rows if r['arm'] == arm]
        status = Counter(r['status'] for r in group)
        table[arm] = dict(cases=len(group), failures=status['collision'] + status['blocked'],
                          status=dict(status),
                          **{m: float(np.median([r[m] for r in group])) for m in METRICS})
    base = table['D']
    eligible = [arm for arm in arms if arm.startswith('E') and table[arm]['primary'] <= 1.05 * base['primary']
                and table[arm]['failures'] <= base['failures']]
    winner = min(eligible, key=lambda arm: table[arm]['extra_jerk_rms_mps3']) if eligible else None
    contrasts = {arm: {m: contrast(rows, arm, 'D', m) for m in ('primary', 'extra_jerk_rms_mps3')}
                 for arm in arms if arm != 'D'}
    result = dict(round=a.round, table=table, eligible=eligible, winner=winner, contrasts_vs_D=contrasts)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    width = max(map(len, arms))
    print(f"{'arm':{width}} " + ' '.join(f'{m[:12]:>12}' for m in METRICS) + '  fail')
    for arm in arms:
        print(f'{arm:{width}} ' + ' '.join(f'{table[arm][m]:12.3f}' for m in METRICS) + f"  {table[arm]['failures']}")
    print('eligible', eligible, 'winner', winner)


if __name__ == '__main__':
    main()
