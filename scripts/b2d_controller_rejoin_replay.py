#!/usr/bin/env python
"""Offline v2 adapter replay/latency evidence; no simulator or controller mutation.

Snapshot minimal sensor-derived pose inputs, immutable route references, exact
helper/adapter sources and per-trajectory outputs. Replays are not closed-loop
performance claims. Wall latency includes uncontrolled host scheduling noise.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from b2d_controller_adapter import RouteAdapter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'p50': float(np.quantile(a, .5)), 'p95': float(np.quantile(a, .95)),
            'p99': float(np.quantile(a, .99)), 'max': float(a.max())} if len(a) else {'n': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, help='Existing G2 root; may repeat for supplemental routes')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cruise-mps', type=float, default=6.)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.out.exists():
        parser.error('output already exists; choose a fresh version')
    out = args.out.resolve(); (out / 'source').mkdir(parents=True)
    manifest = {'protocol': 'offline replay of baseline sensor-derived poses; no closed-loop or dynamic feasibility claim',
                'latency_clock': 'perf_counter_ns wall time; uncontrolled host load; all repetitions retained',
                'argv': sys.argv, 'python': sys.version, 'platform': platform.platform(),
                'numpy': np.__version__, 'cruise_mps': args.cruise_mps, 'repeats': args.repeats,
                'sources': [], 'inputs': []}
    for filename in ('b2d_controller_adapter.py', 'b2d_controller_rejoin_replay.py', 'test_b2d_controller_rejoin.py'):
        path = Path(__file__).resolve().with_name(filename)
        target = out / 'source' / filename; shutil.copyfile(str(path), str(target))
        manifest['sources'].append({'original': str(path), 'snapshot': str(target.relative_to(out)), 'sha256': digest(target)})
    cases, timing_rows = [], []
    output = (out / 'rejoin_samples.jsonl').open('w')
    try:
        for source_root in args.source:
            for control in sorted(Path(source_root).resolve().glob('*/*/control.jsonl')):
                route_id, preset = control.parent.parent.name, control.parent.name
                tag = Path(source_root).name + '/' + route_id + '/' + preset
                reference_path = control.with_name('route_reference.json')
                reference = json.loads(reference_path.read_text())
                inputs = out / 'inputs' / tag; inputs.mkdir(parents=True)
                shutil.copyfile(str(reference_path), str(inputs / 'route_reference.json'))
                compact_path = inputs / 'pose.jsonl'
                compact = []
                for line in control.read_text().splitlines():
                    row = json.loads(line)
                    compact.append({key: row[key] for key in ('frame', 'sim_time', 'pose_xy', 'pose_yaw', 'speed_mps')})
                compact_path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in compact))
                manifest['inputs'].append({'case': tag, 'original_control': str(control), 'control_sha256': digest(control),
                    'original_reference': str(reference_path), 'reference_sha256': digest(reference_path),
                    'compact_pose': str(compact_path.relative_to(out)), 'compact_pose_sha256': digest(compact_path)})
                adapter = RouteAdapter(reference['world_xy'], cruise_mps=args.cruise_mps)
                errors, peak, counts, own_timings = [], 0., {}, []
                for index, row in enumerate(compact):
                    xy = np.array(row['pose_xy'])
                    adapter.project(xy, row['pose_yaw'], row['speed_mps'], row['sim_time'])
                    if index % 4:
                        continue
                    trajectory = None
                    for repeat in range(args.repeats):
                        started = time.perf_counter_ns()
                        try:
                            trajectory = adapter.trajectory(xy, row['pose_yaw'])
                            error = None
                        except ValueError as exc:
                            error = str(exc)
                        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                        sample = {'case': tag, 'tick_index': index, 'frame': row['frame'], 'repeat': repeat,
                                  'wall_ms': elapsed_ms, 'terminal_hold': adapter.terminal_hold,
                                  'error': error or ''}
                        timing_rows.append(sample); own_timings.append(sample)
                    if trajectory is None:
                        errors.append({'frame': row['frame'], 'error': error})
                        output.write(json.dumps({'case': tag, 'frame': row['frame'], 'error': error}) + '\n')
                        continue
                    chord_speed = np.linalg.norm(np.diff(np.vstack(([0., 0.], trajectory)), axis=0), axis=1) / .25
                    peak = max(peak, float(chord_speed.max()))
                    reason = adapter.rejoin_diagnostics.get('reason')
                    counts[reason] = counts.get(reason, 0) + 1
                    output.write(json.dumps({'case': tag, 'frame': row['frame'], 'sim_time': row['sim_time'],
                        'trajectory_xy': trajectory.tolist(), 'first_chord_speed_mps': float(chord_speed[0]),
                        'max_chord_speed_mps': float(chord_speed.max()),
                        'rejoin': adapter.rejoin_diagnostics}, separators=(',', ':')) + '\n')
                cases.append({'case': tag, 'route_length_m': adapter.official_length, 'source_pose_ticks': len(compact),
                    'replans': sum(counts.values()), 'errors': errors, 'max_timed_chord_speed_mps': peak,
                    'reasons': counts, 'all_trajectory_ms': stats([sample['wall_ms'] for sample in own_timings]),
                    'active_trajectory_ms': stats([sample['wall_ms'] for sample in own_timings if not sample['terminal_hold']])})
    finally:
        output.close()
    with (out / 'latency_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'tick_index', 'frame', 'repeat', 'wall_ms', 'terminal_hold', 'error'])
        writer.writeheader(); writer.writerows(timing_rows)
    summary = {'cases': cases, 'case_count': len(cases), 'construction_errors': sum(len(case['errors']) for case in cases),
               'all_trajectory_ms': stats([sample['wall_ms'] for sample in timing_rows]),
               'active_trajectory_ms': stats([sample['wall_ms'] for sample in timing_rows if not sample['terminal_hold']]),
               'cruise_mps': args.cruise_mps,
               'max_timed_chord_speed_mps': max(case['max_timed_chord_speed_mps'] for case in cases)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest['outputs'] = [{'path': filename, 'sha256': digest(out / filename)}
                           for filename in ('summary.json', 'latency_samples.csv', 'rejoin_samples.jsonl')]
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'cases'}, indent=2))
    return 1 if summary['construction_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
