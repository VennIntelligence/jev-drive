"""Read-only case audit for TFv6 Task 9; canonical W2/W2b runs only."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOTS = {
    'A': Path('/data/runs/b2d/tfv6-w2/formal'),
    'B': Path('/data/runs/b2d/tfv6-w2/formal'),
    'C': Path('/data/runs/b2d/tfv6-w2b'),
    'D': Path('/data/runs/b2d/tfv6-w2b'),
}


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def contiguous_max(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def one_case(path):
    done = json.loads(path.read_text())
    route, seed, arm, level = (done[k] for k in ('route', 'seed', 'arm', 'level'))
    attempt = path.parent / f"attempt-{done['attempt']}"
    records = json.loads((Path(done['run_dir']) / 'attempts' / route / '1' / 'results.json').read_text())['_checkpoint']['records']
    if len(records) != 1:
        raise ValueError(f'Invalid result: {path}')
    record = records[0]
    frames = [json.loads(line) for line in (attempt / 'frames.jsonl').open()]
    if not frames or any(row['step'] != i for i, row in enumerate(frames)):
        raise ValueError(f'Invalid frame coverage: {path}')
    infractions = json.loads((attempt / 'infractions.json').read_text())['infractions']
    collision = min((e['step'] for e in infractions if 'COLLISION' in e['event_type']), default=len(frames))
    valid = [row for row in frames if row['step'] < collision and row.get('truth') and row.get('raw_control')]
    moving = [row for row in valid if row['truth']['forward_speed_mps'] > 2]
    if not moving:
        raise ValueError(f'No driving frames: {path}')
    controls = [row['executed_control'] for row in valid]
    speeds = np.asarray([row['truth']['forward_speed_mps'] for row in valid])
    state = np.asarray([1 if c['throttle'] >= .95 and c['brake'] < .1 else
                        -1 if c['brake'] >= .95 and c['throttle'] < .1 else 0
                        for c in controls])
    flips = 0
    for i in range(1, len(state)):
        if state[i] == 0 or state[i - 1] == state[i]:
            continue
        previous = state[max(0, i - 5):i]
        flips += int(-state[i] in previous)
    moving_controls = [row['executed_control'] for row in moving]
    targets = [(row['truth']['forward_speed_mps'], row['target_speed']) for row in moving
               if row.get('target_speed') is not None]
    moving_zero_targets = [row for row in moving if row.get('target_speed') is not None and
                           row['target_speed'] < .1]
    target_jumps = sum(
        abs(right['target_speed'] - left['target_speed']) >= 5 and
        left['truth']['forward_speed_mps'] > 2 and right['truth']['forward_speed_mps'] > 2 and
        right['sim_time'] - left['sim_time'] < .1
        for left, right in zip(valid, valid[1:])
        if left.get('target_speed') is not None and right.get('target_speed') is not None)
    drive = [row for row in frames if row['step'] >= 40 and row['step'] < collision and row.get('truth')]
    still = [abs(row['truth']['forward_speed_mps']) < .1 for row in drive]
    return {
        'level': level, 'route': route, 'seed': seed, 'arm': arm,
        'ds': float(record['scores']['score_composed']),
        'rc': float(record['scores']['score_route']),
        'official_status': record['status'],
        'ticks': len(frames), 'pre_collision_ticks': len(valid), 'moving_ticks': len(moving),
        'collision_step': collision if collision < len(frames) else '',
        'moving_full_throttle_share': sum(c['throttle'] >= .95 for c in moving_controls) / len(moving),
        'moving_full_brake_share': sum(c['brake'] >= .95 for c in moving_controls) / len(moving),
        'moving_full_pedal_share': sum(c['throttle'] >= .95 or c['brake'] >= .95 for c in moving_controls) / len(moving),
        'full_pedal_flip_count': flips,
        'full_pedal_flips_per_min': flips / (len(valid) * .05 / 60),
        'speed_p95_mps': pct(speeds, 95),
        'speed_max_mps': float(np.max(speeds)),
        'target_overshoot_gt2_share': (sum(actual - target > 2 for actual, target in targets) / len(targets)
                                       if targets else None),
        'target_undershoot_gt2_share': (sum(target - actual > 2 for actual, target in targets) / len(targets)
                                        if targets else None),
        'moving_target_zero_share': len(moving_zero_targets) / len(moving),
        'moving_target_zero_full_brake_share': (sum(row['executed_control']['brake'] >= .95
                                                   for row in moving_zero_targets) / len(moving_zero_targets)
                                                if moving_zero_targets else None),
        'target_jump_gt5_count': target_jumps,
        'target_jump_gt5_per_min': target_jumps / (len(valid) * .05 / 60),
        'stopped_after_2s_share': sum(still) / len(still) if still else None,
        'longest_standstill_after_2s_s': contiguous_max(still) * .05,
        'agent_ms_p95': pct([row['agent_ms'] for row in valid if row.get('agent_ms') is not None], 95),
        'vehicle_collisions': len(record['infractions']['collisions_vehicle']),
        'emergency_yield': len(record['infractions']['yield_emergency_vehicle_infractions']),
        'outside_lane': len(record['infractions']['outside_route_lanes']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for arm, root in ROOTS.items():
        paths = sorted(root.glob(f'level[12]/cases/[12]/route-*/seed-*/{arm}/done.json'))
        if len(paths) != 48:
            raise ValueError(f'Expected 48 valid {arm} cases, got {len(paths)}')
        rows.extend(one_case(path) for path in paths)
    rows.sort(key=lambda r: (r['level'], int(r['route']), r['seed'], r['arm']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} cases to {args.out}')


if __name__ == '__main__':
    main()
