"""Offline diagnosis of the frozen TFv6 rear-axle tangent on valid W2 logs."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b2d_tfv6_diagnose import first_divergence, speed_bin
from scripts.b2d_tfv6_coordinates import REAR_OFFSET_M, rear_waypoints

BASE = Path('/data/runs/b2d/tfv6-w2')
OUT = ROOT / 'todos/2026-09-23-tfv6-controller/results/diagnosis/tangent'
FORMAL_CSV = ROOT / 'todos/2026-09-23-tfv6-controller/results/cases.csv'
PAIRS_CSV = ROOT / 'todos/2026-09-23-tfv6-controller/results/diagnosis/ds_pairs.csv'
WHEELBASE_M = 2.8604714913890885
MAX_STEER_DEG = 69.99999237060547
R_MIN_M = WHEELBASE_M / math.tan(math.radians(MAX_STEER_DEG))
BANDS = ('<0.5', '0.5-3', '3-8', '>=8')
SCORED = {'COLLISION_PEDESTRIAN', 'COLLISION_VEHICLE', 'COLLISION_STATIC',
          'TRAFFIC_LIGHT_INFRACTION', 'STOP_INFRACTION',
          'OUTSIDE_ROUTE_LANES_INFRACTION', 'SCENARIO_TIMEOUT',
          'YIELD_TO_EMERGENCY_VEHICLE', 'ROUTE_DEVIATION', 'VEHICLE_BLOCKED',
          'ROUTE_TIMEOUT'}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)


def case_inputs():
    found = []
    for row in read_csv(FORMAL_CSV):
        p = (BASE / 'formal' / ('level' + row['level']) / 'cases' / row['level'] /
             ('route-' + row['route']) / ('seed-' + row['seed']) / row['arm'] / 'done.json')
        if not p.exists():
            raise FileNotFoundError(p)
        found.append(('formal', '0', str(p)))
    for repeat in ('1', '2'):
        root = BASE / 'd2' / ('repeat-' + repeat) / 'cases'
        for p in sorted(root.glob('*/route-*/seed-*/*/done.json')):
            if 'aborted' not in p.parts:
                found.append(('d2', repeat, str(p)))
    if len(found) != 218 or len(set(found)) != 218:
        raise ValueError(f'Expected 202 formal + 16 D2 cases, found {len(found)}')
    return found


def q(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, p10=math.nan, median=math.nan, p90=math.nan, p99=math.nan, min=math.nan, max=math.nan)
    return dict(n=len(a), p10=float(np.percentile(a, 10)), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                min=float(np.min(a)), max=float(np.max(a)))


def tangent_from_points(p):
    d = np.empty_like(p)
    d[0] = p[0]
    d[1:7] = p[2:8] - p[0:6]
    d[7] = p[7] - p[6]
    t = np.empty_like(p)
    previous = np.array([1., 0.])
    for i in range(8):
        norm = np.linalg.norm(d[i])
        if norm >= .05:
            previous = d[i] / norm
        t[i] = previous
    return t


def alternatives(p):
    t = tangent_from_points(p)
    s = np.cumsum(np.linalg.norm(np.diff(np.vstack((np.zeros(2), p)), axis=0), axis=1))
    a = t.copy()
    a[s < REAR_OFFSET_M] = (1., 0.)
    angle = np.arctan2(t[:, 1], t[:, 0])
    bangle = np.clip(angle, -s / R_MIN_M, s / R_MIN_M)
    b = np.column_stack((np.cos(bangle), np.sin(bangle)))
    shift = np.array([REAR_OFFSET_M, 0.])
    return p - REAR_OFFSET_M * a + shift, p - REAR_OFFSET_M * b + shift


def _worker(item):
    source, repeat, name = item
    donepath = Path(name)
    done = json.loads(donepath.read_text())
    run = Path(done['run_dir'])
    if run.resolve() != (donepath.parent / f"attempt-{done['attempt']}" / 'run').resolve():
        raise ValueError(f'stale done path {donepath}')
    recpath = run / 'attempts' / str(done['route']) / '1' / 'results.json'
    recs = json.loads(recpath.read_text())['_checkpoint']['records']
    if len(recs) != 1:
        raise ValueError(recpath)
    rec = recs[0]
    infpath = run.parent / 'infractions.json'
    events = []
    if infpath.exists():
        for event in json.loads(infpath.read_text()).get('infractions', []):
            kind = event['event_type'].split('.')[-1]
            if kind in SCORED:
                events.append((int(event['step']), kind))
    status = rec['status']
    if 'deviated from the route' in status:
        terminal = 'ROUTE_DEVIATION'
    elif 'got blocked' in status:
        terminal = 'VEHICLE_BLOCKED'
    elif 'route timeout' in status.lower():
        terminal = 'ROUTE_TIMEOUT'
    else:
        terminal = None
    frames = [json.loads(line) for line in (run.parent / 'frames.jsonl').open()]
    n = len(frames)
    steps = np.array([f['step'] for f in frames], dtype=int)
    times = np.array([f['sim_time'] for f in frames], dtype=float)
    speed = np.array([abs(f['truth']['forward_speed_mps']) for f in frames], dtype=float)
    xy = np.array([f['truth']['location'][:2] for f in frames], dtype=float)
    if terminal:
        events.append((int(steps[-1]), terminal))
    events.sort()
    valid = np.array([f.get('waypoint') is not None and f.get('rear_waypoint') is not None for f in frames])
    inds = np.flatnonzero(valid)
    p = np.array([frames[i]['waypoint'] for i in inds], dtype=float)
    p[:, :, 1] *= -1
    r = np.array([frames[i]['rear_waypoint'] for i in inds], dtype=float)
    # Cross-check logged output against exactly the frozen implementation.
    for j in range(min(len(p), 4)):
        if not np.allclose(rear_waypoints(frames[int(inds[j])]['waypoint']), r[j], atol=1e-8):
            raise ValueError(f'rear waypoint mismatch: {donepath}')
    distance = np.linalg.norm(r[:, 0] - p[:, 0], axis=1)
    radius0 = np.linalg.norm(p[:, 0], axis=1)
    phantom = (distance > .3) & (radius0 < .3)
    pind = inds[phantom]
    ps = steps[pind]
    pg = speed_bin(speed[inds])
    angle = np.degrees(np.arctan2((p[:, :3, 1] - r[:, :3, 1]),
                                  (p[:, :3, 0] + REAR_OFFSET_M - r[:, :3, 0])))
    # Exact logged implication; angle can be undefined only if malformed r.
    r0 = r[:, 0]
    source_row = dict(source=source, repeat=repeat, level=str(done['level']),
                      route=str(done['route']), seed=int(done['seed']), arm=done['arm'])
    case = dict(**source_row, frames=n, valid_ticks=len(inds), phantom_ticks=int(phantom.sum()),
                phantom_pct=float(100 * phantom.mean()) if len(phantom) else 0,
                phantom_episodes=0, standstill_phantom_ticks=int(np.sum(phantom & (pg == '<0.5'))),
                ds=float(rec['scores']['score_composed']), rc=float(rec['scores']['score_route']),
                official_status=status)
    bands = []
    for band in BANDS:
        mask = pg == band
        bands.append(dict(**source_row, speed_band=band, valid_ticks=int(mask.sum()),
                          phantom_ticks=int(np.sum(phantom & mask))))
    rawc = np.array([float((frames[i].get('raw_control') or {}).get('C', {}).get('throttle', math.nan)) for i in pind])
    rawd = np.array([float((frames[i].get('raw_control') or {}).get('D', {}).get('throttle', math.nan)) for i in pind])
    rawb = np.array([float((frames[i].get('raw_control') or {}).get('B', {}).get('brake', math.nan)) for i in pind])
    effectiveb = np.array([float(((frames[i].get('executed_control') if done['arm'] == 'B' else
                                   (frames[i].get('final_control') or {}).get('B')) or {}).get('brake', math.nan)) for i in pind])
    consequence = dict(**source_row, phantom_ticks=len(pind),
                       C_throttle_B_raw_brake=int(np.sum((rawc > .05) & (rawb > .05))),
                       C_throttle_B_effective_brake=int(np.sum((rawc > .05) & (effectiveb > .05))),
                       D_throttle_B_raw_brake=int(np.sum((rawd > .05) & (rawb > .05))),
                       D_throttle_B_effective_brake=int(np.sum((rawd > .05) & (effectiveb > .05))),
                       C_throttle=int(np.sum(rawc > .05)), D_throttle=int(np.sum(rawd > .05)),
                       B_raw_brake=int(np.sum(rawb > .05)), B_effective_brake=int(np.sum(effectiveb > .05)),
                       rear_first_speed_median=q(np.linalg.norm(r0[phantom], axis=1)/.25)['median'],
                       direct_first_speed_median=q(radius0[phantom]/.25)['median'])
    alt_a = alt_b = 0
    for local in np.flatnonzero(radius0 < .3):
        ar, br = alternatives(p[local])
        alt_a += int(np.linalg.norm(ar[0] - p[local, 0]) > .3)
        alt_b += int(np.linalg.norm(br[0] - p[local, 0]) > .3)
    counter = dict(**source_row, eligible_ticks=int(np.sum(radius0 < .3)),
                   frozen_phantom_ticks=int(phantom.sum()), rule_a_phantom_ticks=alt_a,
                   rule_b_phantom_ticks=alt_b)
    # Contiguous episodes in simulator step, preserving gaps from invalid frames.
    episodes = []
    for s in ps:
        if episodes and s == episodes[-1][1] + 1:
            episodes[-1] = (episodes[-1][0], int(s))
        else:
            episodes.append((int(s), int(s)))
    case['phantom_episodes'] = len(episodes)
    linked = []
    for es, ee in episodes:
        for event_step, kind in events:
            if es <= event_step <= ee + 60:
                linked.append((es, ee, event_step, kind))
    outcome = dict(**source_row, ds=case['ds'], rc=case['rc'],
                   phantom_ticks=case['phantom_ticks'], phantom_episodes=len(episodes),
                   scored_events=len(events), event_within_3s=bool(linked),
                   first_linked_episode_start=linked[0][0] if linked else '',
                   first_linked_episode_end=linked[0][1] if linked else '',
                   first_linked_event_step=linked[0][2] if linked else '',
                   first_linked_event_type=linked[0][3] if linked else '',
                   linked_event_types='|'.join(sorted(set(x[3] for x in linked))),
                   official_status=status)
    vals = {}
    for selection, mask in [('all', np.ones(len(p), dtype=bool)), ('phantom', phantom)]:
        for i in range(3):
            vals[(selection, f'theta{i}_deg')] = angle[mask, i]
            vals[(selection, f'abs_theta{i}_deg')] = np.abs(angle[mask, i])
        for col, arr in [('r0_x_m', r0[:, 0]), ('r0_y_m', r0[:, 1]),
                         ('r0_norm_m', np.linalg.norm(r0, axis=1)),
                         ('rear_first_mps', np.linalg.norm(r0, axis=1)/.25),
                         ('direct_first_mps', radius0/.25),
                         ('first_speed_excess_mps', (np.linalg.norm(r0, axis=1)-radius0)/.25),
                         ('p0_norm_m', radius0)]:
            vals[(selection, col)] = arr[mask]
    samples = []
    for local in np.flatnonzero(phantom):
        i = int(inds[local])
        samples.append(dict(**source_row, step=int(steps[i]), sim_time_s=float(times[i]),
                            speed_mps=float(speed[i]), speed_band=str(pg[local]),
                            p0_x_m=float(p[local, 0, 0]), p0_y_m=float(p[local, 0, 1]),
                            r0_x_m=float(r[local, 0, 0]), r0_y_m=float(r[local, 0, 1]),
                            theta0_deg=float(angle[local, 0]),
                            rear_first_mps=float(np.linalg.norm(r0[local])/.25),
                            direct_first_mps=float(radius0[local]/.25),
                            C_throttle=float(rawc[np.searchsorted(ps, steps[i])]),
                            B_raw_brake=float(rawb[np.searchsorted(ps, steps[i])])))
    return dict(case=case, bands=bands, consequence=consequence, counter=counter,
                outcome=outcome, values=vals, samples=samples, steps=steps, xy=xy,
                speed=speed, phantom_steps=ps)


def main():
    inputs = case_inputs()
    workers = min(12, os.cpu_count() or 1)
    try:
        from jevdrive.common import n_cpus
        workers = min(12, int(n_cpus()))
    except (ImportError, AttributeError, TypeError):
        pass
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, x) for x in inputs]
        results = [f.result() for f in as_completed(futures)]
    results.sort(key=lambda x: tuple(str(x['case'][k]) for k in ('source','repeat','level','route','seed','arm')))
    for name, key in [('cases.csv','case'), ('speed_bands.csv','bands'),
                      ('consequences.csv','consequence'), ('counterfactual.csv','counter'),
                      ('outcomes.csv','outcome')]:
        rows = [y for x in results for y in (x[key] if isinstance(x[key], list) else [x[key]])]
        write_csv(name, rows)
    samples = [row for x in results for row in x['samples']]
    write_csv('phantom_samples.csv', samples)
    distributions = []
    for source in ('formal','d2'):
        selected = [x for x in results if x['case']['source'] == source]
        for group in ('all', 'phantom'):
            for metric in ('theta0_deg','theta1_deg','theta2_deg',
                           'abs_theta0_deg','abs_theta1_deg','abs_theta2_deg',
                           'r0_x_m','r0_y_m','r0_norm_m',
                           'rear_first_mps','direct_first_mps','first_speed_excess_mps','p0_norm_m'):
                values = np.concatenate([x['values'][(group, metric)] for x in selected])
                distributions.append(dict(source=source, selection=group, metric=metric, **q(values)))
    write_csv('distributions.csv', distributions)
    lookup = {(x['case']['level'],x['case']['route'],x['case']['seed'],x['case']['arm']):x
              for x in results if x['case']['source']=='formal'}
    divergences = []
    for key, x in lookup.items():
        level, route, seed, arm = key
        if level == '1r' or arm not in ('C','D'):
            continue
        b = lookup[(level,route,seed,'B')]
        d = first_divergence(x['steps'],x['xy'],x['speed'],b['steps'],b['xy'],b['speed'])
        hit = d is not None
        tick = d['step'] if hit else None
        own = bool(hit and np.any((x['phantom_steps'] >= tick-10) & (x['phantom_steps'] <= tick)))
        base = bool(hit and np.any((b['phantom_steps'] >= tick-10) & (b['phantom_steps'] <= tick)))
        divergences.append(dict(level=level,route=route,seed=seed,arm=arm,
                                divergence_step=tick if hit else '',
                                divergent=hit, phantom_own_prior_0_5s=own,
                                phantom_B_prior_0_5s=base,
                                phantom_either_prior_0_5s=own or base,
                                own_phantom_ticks=x['case']['phantom_ticks'],
                                B_phantom_ticks=b['case']['phantom_ticks']))
    write_csv('divergences.csv',divergences)
    pairs = read_csv(PAIRS_CSV)
    outlookup = {(x['case']['level'],x['case']['route'],x['case']['seed'],x['case']['arm']):x['outcome']
                 for x in results if x['case']['source']=='formal'}
    shares = []
    for row in pairs:
        if row['level'] != '2' or row['contrast'] != 'C-B':
            continue
        o = outlookup[(row['level'],row['route'],int(row['seed']),'C')]
        shares.append(dict(level='2',route=row['route'],seed=row['seed'],
                           C_phantom=bool(o['phantom_ticks']),
                           C_linked_event=o['event_within_3s'],
                           **{k:row[k] for k in row if k in ('ds_arm','ds_B','ds_diff') or k.startswith('points_')}))
    write_csv('level2_loss.csv',shares)
    plot(samples)
    print(json.dumps({'cases':len(results),'formal':sum(x['case']['source']=='formal' for x in results),
                      'd2':sum(x['case']['source']=='d2' for x in results),
                      'phantom':sum(x['case']['phantom_ticks'] for x in results),
                      'affected_C_D':sum(x['outcome']['event_within_3s'] for x in results if x['case']['source']=='formal' and x['case']['arm'] in 'CD'),
                      'r_min_m':R_MIN_M}))


def plot(samples):
    import matplotlib.pyplot as plt
    from research import plot_style as style
    style.apply()
    data = [x for x in samples if x['source'] == 'formal']
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COLUMN_IN, 2.75))
    p = np.asarray([[x['p0_x_m'], x['p0_y_m']] for x in data])
    r = np.asarray([[x['r0_x_m'], x['r0_y_m']] for x in data])
    axes[0].scatter(r[:, 0], r[:, 1], s=2, alpha=.12,
                    color=style.PALETTE['vermillion'], rasterized=True, label='rear target $r_0$')
    axes[0].scatter(p[:, 0], p[:, 1], s=2, alpha=.12,
                    color=style.PALETTE['blue'], rasterized=True, label='model point $p_0$')
    axes[0].set(xlabel='Forward x (m)', ylabel='Left y (m)', xlim=(-.4, 3.2), ylim=(-2.2, 2.2))
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend(loc='upper right', markerscale=3)
    style.panel(axes[0], '(a) First point after tangent correction')
    rear = np.asarray([x['rear_first_mps'] for x in data])
    direct = np.asarray([x['direct_first_mps'] for x in data])
    edges = np.arange(0, 16.5, .5)
    axes[1].hist(direct, bins=edges, density=True, histtype='step', lw=1.4,
                 color=style.PALETTE['blue'], label='direct $|p_0|/0.25$')
    axes[1].hist(rear, bins=edges, density=True, histtype='step', lw=1.4,
                 color=style.PALETTE['vermillion'], label='corrected $|r_0|/0.25$')
    axes[1].set(xlabel='First-segment speed (m/s)', ylabel='Density', xlim=(0, 16))
    axes[1].legend(loc='upper right')
    style.panel(axes[1], '(b) Speed implied by the first waypoint')
    fig.tight_layout(pad=.8, w_pad=1.5)
    style.save(fig, OUT / 'tangent-pathology')
    plt.close(fig)


if __name__ == '__main__':
    main()
