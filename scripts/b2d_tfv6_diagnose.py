"""D1 log-only diagnosis for the frozen TFv6 W2 campaign.

Inputs are canonical done.json files and their immutable formal logs. The script
never reads aborted/ and never starts CARLA. Per-case JSON parsing runs in a
process pool; tick-level calculations use NumPy arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAL = Path('/data/runs/b2d/tfv6-w2/formal')
CASES_CSV = ROOT / 'todos/2026-09-23-tfv6-controller/results/cases.csv'
ARMS = 'ABCD'
CONTROL_KEYS = ('throttle', 'steer', 'brake')
# Locked Bench2Drive scorer: statistics_manager.py:21-29, 35-37, 387-413.
FIXED_PENALTIES = {
    'collisions_pedestrian': .5,
    'collisions_vehicle': .6,
    'collisions_layout': .65,
    'red_light': .7,
    'stop_infraction': .8,
    'scenario_timeouts': .7,
    'yield_emergency_vehicle_infractions': .7,
}
ZERO_PENALTY_NAMES = ('min_speed_infractions', 'route_dev', 'vehicle_blocked', 'route_timeout')
PERCENT_RE = re.compile(r'\(([-+\d.]+)% of the completed route\)')
EVENT_NAME = {
    'COLLISION_PEDESTRIAN': 'collisions_pedestrian',
    'COLLISION_VEHICLE': 'collisions_vehicle',
    'COLLISION_STATIC': 'collisions_layout',
    'TRAFFIC_LIGHT_INFRACTION': 'red_light',
    'STOP_INFRACTION': 'stop_infraction',
    'OUTSIDE_ROUTE_LANES_INFRACTION': 'outside_route_lanes',
    'SCENARIO_TIMEOUT': 'scenario_timeouts',
    'YIELD_TO_EMERGENCY_VEHICLE': 'yield_emergency_vehicle_infractions',
    'ROUTE_DEVIATION': 'route_dev',
    'VEHICLE_BLOCKED': 'vehicle_blocked',
}
SCORE_CAUSES = ('rc', *FIXED_PENALTIES, 'outside_route_lanes', *ZERO_PENALTY_NAMES, 'rounding')


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def score_logs(rc, ds, infractions):
    """Return additive log-score terms; zero scores have no logarithm.

    Locked scorer multiplies RC by fixed factors and (1 - outside-lane
    percentage/100). The percentages in official messages are rounded, so a
    small explicit residual reconciles to the official rounded DS exactly.
    """
    if rc <= 0 or ds <= 0:
        return None
    logs = {name: 0.0 for name in SCORE_CAUSES}
    logs['rc'] = math.log(rc)
    for name, coefficient in FIXED_PENALTIES.items():
        logs[name] = len(infractions.get(name, [])) * math.log(coefficient)
    for message in infractions.get('outside_route_lanes', []):
        match = PERCENT_RE.search(message)
        if not match:
            raise ValueError(f'Cannot parse outside-lane percentage: {message}')
        factor = 1 - float(match.group(1)) / 100
        if factor <= 0:
            # A 100% lane departure is a zero-score edge case, handled above.
            return None
        logs['outside_route_lanes'] += math.log(factor)
    logs['rounding'] = math.log(ds) - sum(logs.values())
    return logs


def decompose_ds_pair(treatment, baseline):
    """Exactly allocate a paired DS difference to additive log-score terms.

    The logarithmic mean converts each log contribution to signed DS points,
    and the DS-point contributions sum to treatment DS minus baseline DS.
    """
    if treatment['logs'] is None or baseline['logs'] is None:
        return None
    delta = {cause: treatment['logs'][cause] - baseline['logs'][cause]
             for cause in SCORE_CAUSES}
    score_diff = treatment['ds'] - baseline['ds']
    log_diff = sum(delta.values())
    scale = score_diff / log_diff if abs(log_diff) > 1e-12 else treatment['ds']
    points = {cause: scale * value for cause, value in delta.items()}
    if not math.isclose(sum(points.values()), score_diff, rel_tol=1e-9, abs_tol=1e-8):
        raise AssertionError('DS decomposition is not additive')
    return points


def first_divergence(steps_a, xy_a, speed_a, steps_b, xy_b, speed_b,
                     distance_m=1.0, speed_mps=1.0):
    """First common tick exceeding either frozen positional or speed threshold."""
    common, ia, ib = np.intersect1d(steps_a, steps_b, assume_unique=True, return_indices=True)
    if not len(common):
        return None
    distance = np.linalg.norm(xy_a[ia] - xy_b[ib], axis=1)
    speed_delta = np.abs(speed_a[ia] - speed_b[ib])
    hit = np.flatnonzero((distance > distance_m) | (speed_delta > speed_mps))
    if not len(hit):
        return None
    j = hit[0]
    return dict(step=int(common[j]), index_a=int(ia[j]), index_b=int(ib[j]),
                distance_m=float(distance[j]), speed_delta_mps=float(speed_delta[j]))


def speed_bin(speed):
    bins = np.digitize(speed, [.5, 3., 8.])
    return np.asarray(['<0.5', '0.5-3', '3-8', '>=8'])[bins]


def context_bin(speed, target, first, mean8, angle):
    if np.isfinite(target) and target <= .05:
        return 'target_stop'
    if speed < .5 and np.isfinite(mean8) and mean8 > 1:
        return 'launch'
    if speed > 1 and np.isfinite(first) and first < speed - 1:
        return 'decelerate'
    if np.isfinite(angle) and abs(angle) >= 15:
        return 'turn'
    return 'cruise'


def case_path(row):
    level, route, seed, arm = (row[key] for key in ('level', 'route', 'seed', 'arm'))
    return (FORMAL / f'level{level}' / 'cases' / level / f'route-{route}' /
            f'seed-{seed}' / arm / 'done.json')


def load_case(row):
    done_path = case_path(row)
    if not done_path.exists() or 'aborted' in done_path.parts:
        raise FileNotFoundError(f'Canonical valid case missing: {done_path}')
    done = json.loads(done_path.read_text())
    run_dir = Path(done['run_dir'])
    if run_dir.resolve() != (done_path.parent / f"attempt-{done['attempt']}" / 'run').resolve():
        raise ValueError(f'Stale run_dir in {done_path}')
    route = row['route']
    record_path = run_dir / 'attempts' / route / '1' / 'results.json'
    records = json.loads(record_path.read_text())['_checkpoint']['records']
    if len(records) != 1:
        raise ValueError(f'Expected one official result: {record_path}')
    record = records[0]
    scores = record['scores']
    ds, rc = float(scores['score_composed']), float(scores['score_route'])
    logs = score_logs(rc, ds, record['infractions'])
    with (run_dir.parent / 'frames.jsonl').open() as stream:
        frames = [json.loads(line) for line in stream]
    n = len(frames)
    steps = np.fromiter((r['step'] for r in frames), dtype=np.int32, count=n)
    time = np.fromiter((r['sim_time'] for r in frames), dtype=float, count=n)
    xy = np.asarray([r['truth']['location'][:2] for r in frames], dtype=float)
    speed = np.asarray([abs(r['truth']['forward_speed_mps']) for r in frames], dtype=float)
    target = np.asarray([float(r['target_speed']) if r.get('target_speed') is not None else np.nan
                         for r in frames], dtype=float)
    waypoint = np.full((n, 8, 2), np.nan)
    route_plan = np.full((n, 2, 2), np.nan)
    control = np.full((n, 4, 3), np.nan)
    executed = np.full((n, 3), np.nan)
    reason_c = np.empty(n, dtype='U32')
    reason_d = np.empty(n, dtype='U32')
    stop_sign = np.zeros(n, dtype=bool)
    creep = np.zeros(n, dtype=bool)
    for i, frame in enumerate(frames):
        wp = frame.get('waypoint')
        if wp is not None and len(wp) == 8:
            waypoint[i] = wp
        rp = frame.get('route_prediction')
        if rp is not None and len(rp) >= 2:
            route_plan[i] = rp[:2]
        final = frame.get('final_control') or {}
        for j, arm in enumerate(ARMS):
            if final.get(arm):
                control[i, j] = [final[arm][key] for key in CONTROL_KEYS]
        actual = frame.get('executed_control') or {}
        if actual:
            executed[i] = [actual[key] for key in CONTROL_KEYS]
        reason_c[i] = (frame.get('controller_reason') or {}).get('C', '')
        reason_d[i] = (frame.get('controller_reason') or {}).get('D', '')
        heuristic = frame.get('heuristic') or {}
        stop_sign[i] = bool(heuristic.get('stop_sign_active'))
        creep[i] = bool(heuristic.get('creep_active'))
    segments = np.diff(np.concatenate((np.zeros((n, 1, 2)), waypoint), axis=1), axis=1)
    seg_speed = np.linalg.norm(segments, axis=2) / .25
    first = seg_speed[:, 0]
    mean8 = np.mean(seg_speed, axis=1)
    u = waypoint[:, 3] - waypoint[:, 0]
    v = waypoint[:, 7] - waypoint[:, 3]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = np.sum(u * v, axis=1)
    angle = np.degrees(np.arctan2(cross, dot))
    angle[(np.linalg.norm(u, axis=1) < .1) | (np.linalg.norm(v, axis=1) < .1)] = np.nan
    q0, q1 = route_plan[:, 0], route_plan[:, 1]
    line = q1 - q0
    length = np.linalg.norm(line, axis=1)
    route_offset = (line[:, 0] * -q0[:, 1] - line[:, 1] * -q0[:, 0]) / length
    route_offset[length < .05] = np.nan
    events = []
    inf_path = run_dir.parent / 'infractions.json'
    if inf_path.exists():
        for event in json.loads(inf_path.read_text()).get('infractions', []):
            name = EVENT_NAME.get(event['event_type'].split('.')[-1])
            if name:
                events.append(dict(step=int(event['step']), cause=name,
                                   meters=event.get('meters_travelled')))
    status = record['status']
    if 'deviated from the route' in status:
        events.append(dict(step=int(steps[-1]), cause='route_dev', meters=None))
    elif 'got blocked' in status:
        events.append(dict(step=int(steps[-1]), cause='vehicle_blocked', meters=None))
    events.sort(key=lambda e: e['step'])
    return dict(level=row['level'], route=route, seed=int(row['seed']), arm=row['arm'],
                ds=ds, rc=rc, score_penalty=float(scores['score_penalty']),
                logs=logs, status=status, infractions=record['infractions'],
                steps=steps, time=time, xy=xy, speed=speed, target=target,
                first=first, mean8=mean8, turn_angle=angle, route_offset=route_offset,
                control=control, executed=executed, reason_c=reason_c, reason_d=reason_d,
                stop_sign=stop_sign, creep=creep, events=events)


def _quantiles(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan)
    return dict(n=len(x), mean=float(np.mean(x)), median=float(np.median(x)),
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))


def _worker(row):
    return load_case(row)


def _context(case, i):
    speed = case['speed'][i]
    target = case['target'][i]
    first = case['first'][i]
    mean8 = case['mean8'][i]
    angle = case['turn_angle'][i]
    return dict(speed_mps=round(float(speed), 3), target_mps=round(float(target), 3),
                plan_first_mps=round(float(first), 3), plan_mean8_mps=round(float(mean8), 3),
                turn_deg=round(float(angle), 2), stop_sign=bool(case['stop_sign'][i]),
                creep=bool(case['creep'][i]), intent=context_bin(speed, target, first, mean8, angle),
                speed_bin=str(speed_bin(speed)))


def _first_scored_event(case, step):
    for event in case['events']:
        if event['step'] >= step and event['cause'] not in ZERO_PENALTY_NAMES[:1]:
            return event
    return None


def _case_lag(case):
    # Nonnegative lag, in 0.05-s ticks. A 0.5-s smoothed speed signal suppresses
    # high-frequency CARLA speed noise; only moving/active plan ticks enter.
    plan = case['first']
    actual = case['speed']
    valid = np.isfinite(plan) & np.isfinite(actual) & (plan > .5)
    if np.sum(valid) < 50:
        return np.nan
    scores = []
    for lag in range(0, 41, 2):
        if lag:
            good = valid[:-lag] & valid[lag:]
            p, a = plan[:-lag][good], actual[lag:][good]
        else:
            p, a = plan[valid], actual[valid]
        if len(p) < 50 or np.std(p) < .2 or np.std(a) < .2:
            scores.append(-np.inf)
        else:
            scores.append(float(np.corrcoef(p, a)[0, 1]))
    return float(np.argmax(scores) * .1) if np.isfinite(max(scores)) else np.nan


def diagnose(cases):
    by_key = {(c['level'], c['route'], c['seed'], c['arm']): c for c in cases}
    if len(by_key) != len(cases):
        raise ValueError('duplicate canonical case key')
    pairs, divergences, tracking, lateral, controls, drift = [], [], [], [], [], []
    control_pool = defaultdict(list)
    reset_pool = defaultdict(list)
    for c in cases:
        arm = c['arm']
        valid = np.isfinite(c['first']) & np.isfinite(c['target'])
        for context in ('all', 'launch', 'target_stop', 'decelerate', 'turn', 'cruise'):
            mask = valid.copy()
            if context != 'all':
                mask &= np.array([context_bin(s, t, f, m, a) == context for s, t, f, m, a in
                                  zip(c['speed'], c['target'], c['first'], c['mean8'], c['turn_angle'])])
            row = dict(level=c['level'], route=c['route'], seed=c['seed'], arm=arm,
                       context=context, lag_s=_case_lag(c) if context == 'all' else np.nan)
            for name, value in [('first_error_mps', c['first'] - c['speed']),
                                ('mean8_error_mps', c['mean8'] - c['speed']),
                                ('target_error_mps', c['target'] - c['speed'])]:
                q = _quantiles(value[mask])
                row.update({f'{name}_{key}': val for key, val in q.items()})
            tracking.append(row)
        for context in ('all', 'turn', 'straight'):
            mask = np.isfinite(c['route_offset'])
            if context == 'turn':
                mask &= np.abs(c['turn_angle']) >= 15
            elif context == 'straight':
                mask &= np.abs(c['turn_angle']) < 15
            offset = c['route_offset'][mask]
            lateral.append(dict(level=c['level'], route=c['route'], seed=c['seed'], arm=arm,
                                context=context, **{f'offset_m_{k}': v for k, v in _quantiles(offset).items()},
                                abs_offset_m_median=float(np.median(np.abs(offset))) if len(offset) else np.nan))
        # Full-tick same-trajectory comparison, including the executed arm label.
        speed_group = speed_bin(c['speed'])
        intent = np.array([context_bin(s, t, f, m, a) for s, t, f, m, a in
                           zip(c['speed'], c['target'], c['first'], c['mean8'], c['turn_angle'])])
        turn_group = np.where(np.abs(c['turn_angle']) >= 15, 'turn', 'straight')
        for contrast, lhs in [('C-B', 2), ('A-B', 0)]:
            if arm not in ('B', contrast[0]):
                continue  # Exactly one controller was executed on this trajectory.
            lhs_control = c['executed'] if arm == contrast[0] else c['control'][:, lhs]
            b_control = c['executed'] if arm == 'B' else c['control'][:, 1]
            delta_long = (lhs_control[:, 0] - lhs_control[:, 2]
                          - b_control[:, 0] + b_control[:, 2])
            delta_steer = lhs_control[:, 1] - b_control[:, 1]
            good = np.isfinite(delta_long) & np.isfinite(delta_steer)
            for sb in np.unique(speed_group[good]):
                for it in np.unique(intent[good & (speed_group == sb)]):
                    for turn in ('straight', 'turn'):
                        idx = good & (speed_group == sb) & (intent == it) & (turn_group == turn)
                        if np.any(idx):
                            control_pool[(contrast, arm, sb, it, turn, 'long')].append(delta_long[idx])
                            control_pool[(contrast, arm, sb, it, turn, 'steer')].append(delta_steer[idx])
            if contrast == 'C-B' and arm == 'B':
                # Non-executed C PI state has no logged integrator. Near-standstill
                # vs later moving shadow contrasts are a measurable drift proxy.
                moving = good & (c['speed'] >= .5) & (c['target'] > .5)
                reset = (c['speed'] < .2) & (c['target'] < .2)
                since = np.arange(len(reset)) - np.maximum.accumulate(np.where(reset, np.arange(len(reset)), -100000))
                for phase, selector in [('near_reset', (since >= 1) & (since <= 5)),
                                        ('far_from_reset', since >= 20)]:
                    idx = moving & selector
                    if np.any(idx):
                        reset_pool[(arm, phase, 'long')].append(delta_long[idx])
                        reset_pool[(arm, phase, 'steer')].append(delta_steer[idx])
                        for sb in np.unique(speed_group[idx]):
                            for it in np.unique(intent[idx & (speed_group == sb)]):
                                sub = idx & (speed_group == sb) & (intent == it)
                                reset_pool[(arm, phase, 'long', sb, it)].append(delta_long[sub])
                                reset_pool[(arm, phase, 'steer', sb, it)].append(delta_steer[sub])
    for key, chunks in control_pool.items():
        contrast, executed_arm, sb, intent, turn, axis = key
        controls.append(dict(contrast=contrast, executed_arm=executed_arm, speed_bin=sb,
                             intent=intent, curvature=turn, axis=axis,
                             **_quantiles(np.concatenate(chunks))))
    for key, chunks in reset_pool.items():
        arm, phase, axis, *stratum = key
        drift.append(dict(executed_arm=arm, phase=phase, axis=axis,
                          speed_bin=stratum[0] if stratum else 'all',
                          intent=stratum[1] if stratum else 'all',
                          **_quantiles(np.concatenate(chunks))))
    for c in cases:
        if c['arm'] not in 'AC':
            continue
        b = by_key.get((c['level'], c['route'], c['seed'], 'B'))
        if b is None:
            continue
        contrast = c['arm'] + '-B'
        points = decompose_ds_pair(c, b)
        row = dict(level=c['level'], route=c['route'], seed=c['seed'], contrast=contrast,
                   ds_arm=c['ds'], ds_B=b['ds'], ds_diff=c['ds'] - b['ds'],
                   rc_arm=c['rc'], rc_B=b['rc'],
                   **{f'points_{k}': points[k] if points else np.nan for k in SCORE_CAUSES},
                   zero_ds_excluded=points is None)
        pairs.append(row)
        hit = first_divergence(c['steps'], c['xy'], c['speed'], b['steps'], b['xy'], b['speed'])
        if hit is None:
            divergences.append(dict(level=c['level'], route=c['route'], seed=c['seed'],
                                    contrast=contrast, ds_diff=row['ds_diff'], diverged=False))
            continue
        i = hit['index_a']
        event_arm = _first_scored_event(c, hit['step'])
        event_b = _first_scored_event(b, hit['step'])
        event = None
        event_side = None
        if event_arm and event_b:
            event_side, event = min((('arm', event_arm), ('B', event_b)), key=lambda x: x[1]['step'])
        elif event_arm:
            event_side, event = 'arm', event_arm
        elif event_b:
            event_side, event = 'B', event_b
        divergences.append(dict(level=c['level'], route=c['route'], seed=c['seed'],
                                contrast=contrast, ds_diff=row['ds_diff'], diverged=True,
                                divergence_step=hit['step'], divergence_time_s=float(c['time'][i]),
                                distance_m=hit['distance_m'], speed_delta_mps=hit['speed_delta_mps'],
                                **_context(c, i),
                                next_event_side=event_side or '',
                                next_event_cause=event['cause'] if event else '',
                                next_event_step=event['step'] if event else '',
                                next_event_delta_s=(event['step'] - hit['step']) * .05 if event else np.nan))
    return pairs, divergences, tracking, lateral, controls, drift


def _plot(out, pairs, divergences, controls, lateral):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from research import plot_style as ps
    ps.apply()
    out.mkdir(parents=True, exist_ok=True)
    main = [p for p in pairs if p['level'] in ('1', '2')]
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.45))
    for ax, contrast in zip(axes, ('C-B', 'A-B')):
        group = [p for p in main if p['contrast'] == contrast]
        labels = ['rc', 'collisions_vehicle', 'collisions_pedestrian', 'collisions_layout', 'red_light',
                  'stop_infraction', 'outside_route_lanes', 'rounding']
        means = [np.mean([p[f'points_{x}'] for p in group]) for x in labels]
        ax.barh(range(len(labels)), means, color=[ps.PALETTE['blue'] if v >= 0 else ps.PALETTE['vermillion'] for v in means])
        ax.set_yticks(range(len(labels)), ['RC', 'vehicle', 'pedestrian', 'static', 'red light', 'stop', 'outside lane', 'rounding'])
        ax.invert_yaxis()
        ax.set_xlabel('Mean contribution to DS difference')
        ax.set_title(contrast)
        ax.axvline(0, color='black', lw=.5)
    fig.tight_layout()
    ps.save(fig, out / 'ds_causes')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(ps.DOUBLE_COLUMN_IN, 2.4))
    for ax, contrast in zip(axes, ('C-B', 'A-B')):
        count = Counter(d['intent'] for d in divergences if d['level'] in ('1','2') and d['contrast'] == contrast and d['diverged'])
        labels = ['launch', 'decelerate', 'target_stop', 'turn', 'cruise']
        ax.bar(labels, [count[x] for x in labels], color=ps.PALETTE['blue'])
        ax.tick_params(axis='x', rotation=40)
        ax.set_ylabel('Paired routes')
        ax.set_title(contrast)
    fig.tight_layout()
    ps.save(fig, out / 'divergence_context')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COLUMN_IN, 2.35))
    for arm, color in [('A','green'), ('B','blue'), ('C','vermillion')]:
        rows = [r for r in lateral if r['level'] in ('1','2') and r['arm'] == arm and r['context'] == 'all']
        ax.scatter([{'A':0,'B':1,'C':2}[arm]]*len(rows), [r['offset_m_median'] for r in rows],
                   s=10, alpha=.45, color=ps.PALETTE[color])
    ax.axhline(0, color='black', lw=.5)
    ax.set_xticks([0,1,2], ['A','B','C'])
    ax.set_ylabel('Median signed route offset (m)')
    fig.tight_layout()
    ps.save(fig, out / 'route_offset')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    try:
        from jevdrive.common import n_cpus
        workers = n_cpus()
    except (ImportError, AttributeError):
        workers = os.cpu_count() or 1
    if args.workers is not None:
        workers = args.workers
    source = read_csv(CASES_CSV)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, row) for row in source]
        cases = [future.result() for future in as_completed(futures)]
    pairs, divergences, tracking, lateral, controls, drift = diagnose(cases)
    out = ROOT / 'todos/2026-09-23-tfv6-controller/results/diagnosis'
    route_rows = []
    for level, route, contrast in sorted({(p['level'], p['route'], p['contrast']) for p in pairs
                                          if p['level'] in ('1', '2')}):
        group = [p for p in pairs if (p['level'], p['route'], p['contrast']) == (level, route, contrast)]
        route_rows.append(dict(level=level, route=route, contrast=contrast, n=len(group),
                               mean_ds_diff=float(np.mean([p['ds_diff'] for p in group])),
                               **{f'mean_points_{name}': float(np.mean([p[f'points_{name}'] for p in group]))
                                  for name in SCORE_CAUSES}))
    for name, rows in [('ds_pairs', pairs), ('divergences', divergences), ('speed_tracking', tracking),
                       ('route_causes', route_rows), ('route_offset', lateral),
                       ('control_bins', controls), ('shadow_reset_proxy', drift)]:
        write_csv(out / f'{name}.csv', rows)
    _plot(out, pairs, divergences, controls, lateral)
    print(json.dumps(dict(workers=workers, cases=len(cases), pairs=len(pairs),
                          divergences=sum(d['diverged'] for d in divergences),
                          out=str(out)), indent=2))


if __name__ == '__main__':
    main()
