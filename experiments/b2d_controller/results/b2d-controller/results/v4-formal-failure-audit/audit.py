#!/usr/bin/env python3
"""Read completed formal attempts; preserve frame evidence, never infer collision blame."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import time


def read(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def lines(path):
    rows = []
    try:
        with path.open() as fh:
            for number, line in enumerate(fh, 1):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append((number, value))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_sample(line, row):
    keys = ('frame', 'sim_time', 'speed_mps', 'raw_speed_mps', 'target_speed_mps',
            'reference_speed_mps', 'throttle', 'steer', 'brake', 'reason', 'pose_status',
            'route_progress_m', 'truth_cross_track_m', 'route_cross_track_m',
            'cross_track_m', 'heading_error_rad', 'pose_error_m', 'trajectory_age_s',
            'pose_xy', 'pose_yaw', 'truth_xy', 'truth_yaw', 'route_terminal_hold')
    return dict(line=line, **{key: row.get(key) for key in keys})


def segments(rows, predicate):
    result, current = [], []
    for line, row in rows:
        if predicate(row):
            current.append((line, row))
        elif current:
            result.append(current); current = []
    if current:
        result.append(current)
    return [dict(first=frame_sample(*run[0]), last=frame_sample(*run[-1]), ticks=len(run),
                 first_to_last_s=run[-1][1]['sim_time'] - run[0][1]['sim_time']) for run in result]


def snapshot(root, output, report):
    route_end = {}
    for group in root.glob('*-seed*'):
        for number, event in lines(group / 'events.jsonl'):
            if event.get('kind') == 'route_end':
                route_end[group.name, str(event['route_id']), int(event['attempt'])] = dict(path=str(group / 'events.jsonl'), line=number, event=event)
    prior = sorted(output.glob('edition-*/summary.json'))
    signature = sorted(route_end)
    final_events = lines(root / 'events.jsonl')
    ended = bool(final_events and final_events[-1][1].get('kind') == 'end')
    if prior:
        old = read(prior[-1])
        if old['attempt_signature'] == [list(x) for x in signature] and old['campaign_ended'] == ended:
            return None
    next_edition = max([int(p.name.split('-')[-1]) for p in output.glob('edition-*')], default=0) + 1
    dest = output / ('edition-%03d' % next_edition); dest.mkdir()
    shutil.copyfile(str(Path(__file__)), str(dest / 'audit.py'))
    cases, contexts, sources = [], [], []
    for group, route_id, attempt in signature:
        adir = root / group / 'attempts' / route_id / str(attempt)
        official = report.official_result(adir)
        telemetry = report.telemetry(adir, official)
        captures = read(adir / 'criterion_events.json', {})
        control = lines(adir / 'control.jsonl')
        states = Counter(str((row.get('pose_status') or {}).get('reason', 'missing')) for _, row in control)
        degraded = [(line, row) for line, row in control if (row.get('pose_status') or {}).get('degraded')]
        invalid = [(line, row) for line, row in control if row.get('reason') == 'invalid_pose']
        errors = [(line, row) for line, row in control if row.get('reason') not in ('tracking', 'stop_hold', 'stationary_trajectory')]
        infractions = official.get('infractions', {})
        sr = (official.get('official_status') in ('Completed', 'Perfect')
              and all(not value for key, value in infractions.items() if key != 'min_speed_infractions')) if official.get('official_finalized') else None
        stall_runs = segments(control, lambda row: isinstance(row.get('speed_mps'), (int, float)) and abs(row['speed_mps']) < .5
                              and isinstance(row.get('target_speed_mps'), (int, float)) and row['target_speed_mps'] >= 2.)
        stall_runs = [run for run in stall_runs if run['first_to_last_s'] >= 5.]
        low_speed_runs = [run for run in segments(control, lambda row: isinstance(row.get('speed_mps'), (int, float)) and abs(row['speed_mps']) < .5) if run['first_to_last_s'] >= 5.]
        collisions = sum(len(infractions.get(key, [])) for key in ('collisions_vehicle', 'collisions_layout', 'collisions_pedestrian'))
        blocked, deviation = len(infractions.get('vehicle_blocked', [])), len(infractions.get('route_dev', []))
        # Attribution requires independent evidence; event class alone cannot establish cause.
        cause = 'unknown' if blocked or deviation else 'not_applicable'
        context = dict(group=group, route_id=route_id, attempt=attempt, attempt_path=str(adir),
                       route_end=route_end[group, route_id, attempt], official=official,
                       telemetry=telemetry, pose_status_counts=dict(states),
                       degraded_runs=segments(control, lambda row: bool((row.get('pose_status') or {}).get('degraded'))),
                       invalid_pose_samples=[frame_sample(*pair) for pair in invalid],
                       nontracking_samples=[frame_sample(*pair) for pair in errors],
                       low_speed_high_target_runs=stall_runs, low_speed_runs=low_speed_runs, critical_events=[])
        for index, event in enumerate(captures.get('events', [])):
            kind = event.get('type', '')
            if not any(token in kind for token in ('COLLISION', 'BLOCKED', 'DEVIATION', 'TIMEOUT', 'RED_LIGHT', 'STOP')):
                continue
            frame = event.get('frame')
            nearby = [(line, row) for line, row in control if isinstance(frame, (int, float)) and abs(row.get('frame', -1000) - frame) <= 20]
            context['critical_events'].append(dict(path=str(adir / 'criterion_events.json'), event_index=index,
                                                   event=event, context_window_frames=20,
                                                   control_path=str(adir / 'control.jsonl'), control=[frame_sample(*pair) for pair in nearby],
                                                   collision_responsibility='unknown' if kind.startswith('COLLISION') else 'not_applicable'))
        q = telemetry['telemetry_quality']
        item = dict(group=group, preset=group.split('-seed')[0], seed=int(group.split('-seed')[1]), route_id=route_id, attempt=attempt,
                    scenario=official.get('scenario'), harness_status=route_end[group, route_id, attempt]['event'].get('status'),
                    official_status=official.get('official_status'), official_completion_pct=official.get('completion'),
                    driving_completed=official.get('driving_completed'), subset_diagnostic_sr=sr,
                    score_composed=official.get('score_composed'), score_penalty=official.get('score_penalty'),
                    vehicle_collisions=len(infractions.get('collisions_vehicle', [])), pedestrian_collisions=len(infractions.get('collisions_pedestrian', [])),
                    layout_collisions=len(infractions.get('collisions_layout', [])), vehicle_blocked=blocked, route_dev=deviation,
                    scenario_timeouts=len(infractions.get('scenario_timeouts', [])), route_timeout=len(infractions.get('route_timeout', [])),
                    blocked_deviation_cause=cause, unfinished_cause='unknown' if official.get('driving_completed') is not True else 'not_applicable', collision_responsibility='unknown' if collisions else 'not_applicable',
                    telemetry_state=q['state'], telemetry_ticks=q['valid_rows'], pose_degraded_ticks=len(degraded),
                    pose_degraded_fraction=len(degraded) / len(control) if control else None,
                    invalid_pose_ticks=len(invalid), controller_invalid_motion_ticks=sum(row.get('reason') == 'invalid_motion' for _, row in control), invalid_control_ticks=q['invalid_controls'], stale_ticks=telemetry.get('stale_ticks'),
                    sensor_frame_mismatch=q['sensor_frame_mismatch'], missing_control_frames=q['missing_frames'],
                    low_speed_runs=len(low_speed_runs), low_speed_max_s=max([r['first_to_last_s'] for r in low_speed_runs], default=0), low_speed_high_target_runs=len(stall_runs), low_speed_high_target_max_s=max([r['first_to_last_s'] for r in stall_runs], default=0),
                    full_truth_cte_rms_m=telemetry.get('tracking', {}).get('truth_cross_track_m', {}).get('rms'),
                    precollision_truth_cte_rms_m=telemetry.get('before_collision', {}).get('truth_cross_track_m', {}).get('rms'),
                    first_collision_frame=telemetry.get('first_collision_frame'), attempt_path=str(adir))
        cases.append(item); contexts.append(context)
        for name in ('results.json', 'criterion_events.json', 'control.jsonl', 'motion.jsonl', 'trajectories.jsonl', 'route_reference.json', 'route_result.json', 'attempt.json', 'route.log'):
            source = adir / name
            if source.exists():
                sources.append(dict(path=str(source), bytes=source.stat().st_size, sha256=digest(source)))
    # This remains all-attempt data. The final G4 owner selects first-harness-finished
    # without optimizing over retries and computes per-route denominators.
    with (dest / 'all-attempt-cases.csv').open('w', newline='') as fh:
        if cases:
            writer = csv.DictWriter(fh, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)
    (dest / 'contexts.json').write_text(json.dumps(contexts, indent=2) + '\n')
    summary = dict(attempt_signature=[list(x) for x in signature], completed_attempts=len(cases), expected_route_cases=60,
                   campaign_ended=ended, audit_unix=time.time(),
                   completed_driving=sum(row['driving_completed'] is True for row in cases),
                   successful_subset_cases=sum(row['subset_diagnostic_sr'] is True for row in cases),
                   vehicle_blocked=sum(row['vehicle_blocked'] for row in cases), route_dev=sum(row['route_dev'] for row in cases),
                   pose_degraded_ticks=sum(row['pose_degraded_ticks'] for row in cases),
                   invalid_pose_ticks=sum(row['invalid_pose_ticks'] for row in cases),
                   invalid_control_ticks=sum(row['invalid_control_ticks'] for row in cases),
                   stale_ticks=sum(row['stale_ticks'] or 0 for row in cases),
                   missing_telemetry_attempts=sum(row['telemetry_state'] == 'missing' for row in cases),
                   unresolved_causes=[row for row in cases if row['blocked_deviation_cause'] == 'unknown'],
                   failures=[row for row in cases if not row['driving_completed'] or not row['subset_diagnostic_sr']])
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (dest / 'manifest.json').write_text(json.dumps(dict(inputs=sources, helper_path=str(Path(__file__).resolve()),
                                                     helper_sha256=digest(Path(__file__)), report_path=report.__file__,
                                                     report_sha256=digest(Path(report.__file__))), indent=2) + '\n')
    print(json.dumps(dict(edition=str(dest), **{k:v for k,v in summary.items() if k not in ('attempt_signature','unresolved_causes','failures')})), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=float, default=0.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.run / 'provenance/source/scripts/b2d_report.py'
    spec = importlib.util.spec_from_file_location('frozen_report', str(source))
    report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)
    until = time.monotonic() + args.watch_seconds
    while True:
        result = snapshot(args.run, args.out, report)
        if result and result['campaign_ended']:
            return
        if time.monotonic() >= until:
            return
        time.sleep(min(15., max(0., until - time.monotonic())))


if __name__ == '__main__':
    main()
