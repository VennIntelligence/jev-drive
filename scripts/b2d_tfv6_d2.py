"""Approved D2 reruns: selected route/seed pairs, CARLA recorder, scene telemetry."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

from b2d_tfv6_campaign import DEV10, HOLDOUT, case

ROOT = Path('/data/runs/b2d/tfv6-w2/d2')
BUS = Path('/data/runs/b2d/tfv6-w2/bus')
GROUPS = (('2', '27529', 0, 'BC'),
          ('2', '28154', 0, 'BC'),
          ('1', '26405', 1, 'AB'),
          ('1', '3514', 1, 'BC'))
LOCK = threading.Lock()


def milestone(step, message, **numbers):
    row = {'t': datetime.now(timezone.utc).isoformat(), 'phase': 'D2',
           'step': step, 'msg': message, 'numbers': numbers}
    with LOCK, (BUS / 'status.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


def run_group(index, item):
    level, route, seed, arm, repeat = item
    out = ROOT / f'repeat-{repeat}'
    out.mkdir(parents=True, exist_ok=True)
    xml = HOLDOUT if level == '2' else DEV10
    result = case(out, xml, level, route, seed, arm, 90 + index, 0, record_carla=True)
    return item, result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--route', help='optional one-route validation')
    parser.add_argument('--arm', help='optional one-arm validation')
    parser.add_argument('--repeat', type=int, choices=(1, 2))
    parser.add_argument('--concurrency', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    os.environ['B2D_D2_ACTORS'] = '1'
    tasks = [(level, route, seed, arm, rep)
             for level, route, seed, arms in GROUPS for rep in (1, 2) for arm in arms
             if (args.route is None or route == args.route)
             and (args.arm is None or arm == args.arm)
             and (args.repeat is None or rep == args.repeat)]
    if not tasks:
        parser.error('selection is empty')
    ROOT.mkdir(parents=True, exist_ok=True)
    milestone('rerun_start', 'D2 selected reruns started', selected=len(tasks), concurrency=args.concurrency)
    completed = 0
    # Each worker owns a fixed server-index/port block; tasks are independent.
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = iter(tasks)
        future_map = {}
        for worker in range(min(args.concurrency, len(tasks))):
            item = next(pending)
            future_map[pool.submit(run_group, worker, item)] = worker
        while future_map:
            for future in as_completed(list(future_map)):
                worker = future_map.pop(future)
                item, result = future.result()
                completed += 1
                milestone('case_done', 'D2 rerun case finished', completed=completed, selected=len(tasks),
                          level=item[0], route=item[1], seed=item[2], arm=item[3], repeat=item[4],
                          status=result['status'], official_status=result.get('official_status'),
                          wall_s=round(result['wall_s'], 1))
                next_item = next(pending, None)
                if next_item is not None:
                    future_map[pool.submit(run_group, worker, next_item)] = worker
                break
    milestone('rerun_end', 'D2 selected reruns complete', completed=completed, selected=len(tasks))


if __name__ == '__main__':
    main()
