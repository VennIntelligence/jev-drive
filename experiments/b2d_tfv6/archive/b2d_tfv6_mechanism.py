"""Descriptive W2 stall table; reads the frozen formal results and frame logs."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path('/data/runs/b2d/tfv6-w2/formal')


def med(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return round(median(values), 4) if values else None


def distance(a, b=(0, 0)):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def frame_values(row):
    plan = row.get('waypoint')
    rear = row.get('rear_waypoint')
    if not plan or not rear or len(plan) != 8 or len(rear) != 8:
        return None
    if not all(math.isfinite(v) for p in plan + rear for v in p):
        return None
    first = distance(plan[0]) / .25
    mean8 = sum(distance(p, q) for p, q in zip(plan, [[0, 0]] + plan[:-1])) / 2.0
    c_desired = distance(rear[0]) / .25
    b_desired = 2.0 * distance(plan[1], plan[3])
    controls = row.get('raw_control') or {}
    finals = row.get('final_control') or {}
    b_raw = controls.get('B') or {}
    b_final = finals.get('B') or {}
    return dict(first=first, mean8=mean8, c_desired=c_desired, b_desired=b_desired,
                target=row.get('target_speed'), b_raw_throttle=b_raw.get('throttle'),
                b_final_throttle=b_final.get('throttle'), b_final_brake=b_final.get('brake'))


def stretches(rows, minimum=40):
    start = None
    for i in range(len(rows) + 1):
        stalled = i < len(rows) and abs(rows[i]['truth']['forward_speed_mps']) < .1
        if stalled and start is None:
            start = i
        elif not stalled and start is not None:
            if i - start >= minimum:
                yield rows[start:i]
            start = None


def summarize(level, route, seed, arm, span):
    vals = [frame_values(row) for row in span]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    reasons = collections.Counter(row.get('controller_reason', {}).get(arm, 'missing') for row in span)
    c_reasons = collections.Counter(row.get('controller_reason', {}).get('C', 'missing') for row in span)
    h = [row.get('heuristic') or {} for row in span]
    n = len(span)
    frac = lambda k: round(100 * k / n, 1)
    reverse = sum((row.get('raw_signed_speed_mps') or 0) < -.01 for row in span)
    stop_active = sum(bool(x.get('stop_sign_active')) for x in h)
    creep_active = sum(bool(x.get('creep_active')) for x in h)
    def changed(row):
        raw = (row.get('raw_control') or {}).get(arm) or {}
        final = (row.get('final_control') or {}).get(arm) or {}
        return bool(raw and final and any(abs(raw[key] - final[key]) > .01
                                          for key in ('steer', 'throttle', 'brake')))
    post_changed = sum(changed(row) for row in span)
    stop_changed = sum(changed(row) and bool((row.get('heuristic') or {}).get('stop_sign_active'))
                       for row in span)
    creep_changed = sum(changed(row) and bool((row.get('heuristic') or {}).get('creep_active'))
                        for row in span)
    c_raw_brakes = sum((row.get('raw_control') or {}).get('C', {}).get('brake', 0) > .5
                       for row in span)
    c_final_brakes = sum((row.get('final_control') or {}).get('C', {}).get('brake', 0) > .5
                         for row in span)
    c_raw_throttle = med((row.get('raw_control') or {}).get('C', {}).get('throttle')
                         for row in span)
    c_final_throttle = med((row.get('final_control') or {}).get('C', {}).get('throttle')
                           for row in span)
    b_thr = med(v['b_final_throttle'] for v in vals)
    # Dominant observed mechanism, with all reason fractions retained for audit.
    if reasons['invalid_motion'] >= .5 * n and reverse >= .25 * n:
        classification = 'real reverse / invalid_motion'
    elif post_changed >= .25 * n and (stop_changed or creep_changed) and reasons['stop_hold'] < .25 * n:
        classification = 'stop-sign / creep interaction'
    elif reasons['stop_hold'] >= .25 * n and b_thr and b_thr > .05:
        classification = 'launch hold'
    elif reasons['invalid_motion'] >= .25 * n and reverse >= .1 * n:
        classification = 'real reverse / invalid_motion'
    elif post_changed >= .25 * n and (stop_changed or creep_changed):
        classification = 'stop-sign / creep interaction'
    else:
        classification = 'other'
    return dict(level=level, route=route, seed=seed, arm=arm,
                start_tick=span[0]['step'], end_tick=span[-1]['step'], n_ticks=n,
                duration_s=round(.05 * n, 2), first_mps=med(v['first'] for v in vals),
                mean8_mps=med(v['mean8'] for v in vals), tfv6_target_mps=med(v['target'] for v in vals),
                c_desired_mps=med(v['c_desired'] for v in vals),
                c_effective_mps=med(0 if row.get('controller_reason', {}).get('C') == 'stop_hold'
                                    else frame_values(row)['c_desired'] if frame_values(row) else None
                                    for row in span),
                c_reason_top=c_reasons.most_common(1)[0][0],
                c_stop_hold_pct=frac(c_reasons['stop_hold']),
                c_invalid_motion_pct=frac(c_reasons['invalid_motion']),
                c_raw_brake_pct=frac(c_raw_brakes), c_final_brake_pct=frac(c_final_brakes),
                c_raw_throttle=c_raw_throttle, c_final_throttle=c_final_throttle,
                active_reason_top=reasons.most_common(1)[0][0],
                active_stop_hold_pct=frac(reasons['stop_hold']),
                active_invalid_motion_pct=frac(reasons['invalid_motion']),
                b_desired_mps=med(v['b_desired'] for v in vals),
                b_raw_throttle=med(v['b_raw_throttle'] for v in vals),
                b_final_throttle=b_thr,
                b_final_brake=med(v['b_final_brake'] for v in vals),
                stop_sign_active_pct=frac(stop_active), creep_active_pct=frac(creep_active),
                postprocessor_changed_pct=frac(post_changed),
                stop_sign_changed_pct=frac(stop_changed), creep_changed_pct=frac(creep_changed),
                meaningful_reverse_pct=frac(reverse), classification=classification)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preview-routes', default='')
    args = ap.parse_args()
    with args.cases.open(newline='') as f:
        cases = list(csv.DictReader(f))
    by_route_arm = collections.defaultdict(list)
    case_index = {}
    for r in cases:
        if r['level'] not in ('1', '2'):
            continue
        case_index[(r['level'], r['route'], int(r['seed']), r['arm'])] = r
    for (level, route, seed, arm), row in case_index.items():
        if arm in 'CD' and (level, route, seed, 'B') in case_index:
            by_route_arm[(route, arm)].append(
                float(row['ds']) - float(case_index[(level, route, seed, 'B')]['ds']))
    selected = {route for (route, arm), diffs in by_route_arm.items()
                if len(diffs) == 3 and abs(sum(diffs) / 3) >= 10}
    if args.preview_routes:
        selected = set(args.preview_routes.split(','))
    output = []
    for level, route, seed, arm in sorted(case_index, key=lambda x: (x[0], int(x[1]), x[2], x[3])):
        if route not in selected or arm not in 'CD':
            continue
        done = ROOT / f'level{level}' / 'cases' / level / f'route-{route}' / f'seed-{seed}' / arm / 'done.json'
        if not done.exists():
            raise FileNotFoundError(done)
        item = json.loads(done.read_text())
        frame_path = Path(item['run_dir']).parent / 'frames.jsonl'
        with frame_path.open() as f:
            frames = [json.loads(line) for line in f]
        for span in stretches(frames):
            summary = summarize(level, route, seed, arm, span)
            if summary:
                output.append(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        if output:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print(json.dumps({'selected_routes': sorted(selected, key=int), 'stretches': len(output),
                      'by_class': dict(collections.Counter(r['classification'] for r in output))}))


if __name__ == '__main__':
    main()
