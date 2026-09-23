"""Lateral v2 analysis: per-case window metrics, paired perturbation statistics and the frozen gates.

Reads the raw campaign on the GPU box (one b2d_controller_validate.py output per perturbation) and
writes only small aggregates: case-window metrics, 1 m station profiles, paired statistics, gates.
Definitions follow the frozen pose-g2 analysis (truth rear axle projected on route_reference.json
world_xy by nearest segment; CTE left positive; road heading = centred 5 m chord; CARLA angles are
right positive) and add the rear-axle course. Protocol: ../lateral-v2-protocol.md.

    envs/carla/bin/python analyze.py --campaign <run root> --windows windows.json --out <new dir>
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('prod-k0', 'prod-kfix', 'slipack-k0', 'slipack-kfix', 'prod-truth', 'slipack-truth')
CANDIDATE, PRODUCTION = 'slipack-kfix', 'prod-kfix'
SECONDARY = [('slipack-k0', 'prod-k0'), ('slipack-truth', 'prod-truth')]
C_MKZ, G = 11.0, 9.81
BOOT, BOOT_SEED = 10000, 0
PRIMARY_WINDOW, PRIMARY_LIMIT = '26966', .25
# (metric, kind, limit): 'delta' = mean paired increment <= limit; 'ratio' = mean paired relative increase.
GUARDS = [('cte_rms', 'delta', .03), ('cte_p95', 'delta', .05), ('cte_max', 'delta', .10),
          ('course_p95_deg', 'delta', 1.), ('speed_ref_err_rms', 'delta', .10),
          ('mean_speed_drop', 'delta', .20), ('lat_acc_p95', 'ratio', .10), ('steer_rate_p95', 'ratio', .20)]
PRIMARY_GUARDS = [g for g in GUARDS if not g[0].startswith('cte')] + [('cte_p95', 'delta', 0.), ('post_cte_rms', 'delta', 0.)]
RATIO_FLOOR = .01
MIN_FRAMES = 20


def wrap_deg(a):
    return (np.asarray(a) + 180.) % 360. - 180.


def rear_slip_carla(speed, yaw_rate_left):
    """Sensor-only rear sideslip, body minus course, in CARLA right-positive angle (degrees)."""
    return -np.degrees(np.arctan((speed + 1.) * yaw_rate_left / (C_MKZ * G)))


class Reference:
    def __init__(self, xy):
        xy = np.asarray(xy, float)
        self.xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
        self.d = np.diff(self.xy, axis=0)
        self.len = np.linalg.norm(self.d, axis=1)
        self.s = np.r_[0., np.cumsum(self.len)]

    def at(self, s):
        return np.column_stack([np.interp(s, self.s, self.xy[:, j]) for j in (0, 1)])

    def project(self, p):
        """Station, left-positive CTE and centred-chord road heading (CARLA degrees) per point."""
        p = np.asarray(p, float)
        rel = p[:, None, :] - self.xy[None, :-1, :]
        u = np.clip(np.einsum('nmk,mk->nm', rel, self.d) / self.len ** 2, 0, 1)
        foot = self.xy[None, :-1, :] + u[..., None] * self.d[None]
        i = np.argmin(np.sum((p[:, None, :] - foot) ** 2, axis=2), axis=1)
        n = np.arange(len(p))
        s = self.s[i] + u[n, i] * self.len[i]
        delta = p - foot[n, i]
        cte = (self.d[i, 1] * delta[:, 0] - self.d[i, 0] * delta[:, 1]) / self.len[i]
        chord = self.at(s + 2.5) - self.at(s - 2.5)
        return s, cte, np.degrees(np.arctan2(chord[:, 1], chord[:, 0]))


def number(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def case_frames(directory):
    control = [json.loads(line) for line in (directory / 'control.jsonl').read_text().splitlines() if line]
    trace = {row['frame']: row for row in json.loads((directory / 'validation_trace.json').read_text())}
    reference = Reference(json.loads((directory / 'route_reference.json').read_text())['world_xy'])
    rows = [r for r in control if isinstance(r.get('truth_xy'), list)]
    xy = np.array([r['truth_xy'] for r in rows], float)
    frame = np.array([r['frame'] for r in rows])
    t = np.array([r['sim_time'] for r in rows], float)
    s, cte, road = reference.project(xy)
    yaw = np.degrees(np.array([number(r.get('truth_yaw')) for r in rows]))
    course = np.full(len(rows), np.nan)
    ok = (frame[2:] - frame[:-2] == 2)
    step = xy[2:] - xy[:-2]
    moving = ok & (np.linalg.norm(step, axis=1) > .01)
    course[1:-1][moving] = np.degrees(np.arctan2(step[moving, 1], step[moving, 0]))
    steer = np.array([number(r.get('steer')) for r in rows])
    rate = np.full(len(rows), np.nan)
    consecutive = (np.diff(frame) == 1) & (np.diff(t) > 0)
    rate[1:][consecutive] = (np.diff(steer) / np.diff(t))[consecutive]
    speed = np.array([number(r.get('speed_mps')) for r in rows])
    gyro = np.array([number(r.get('yaw_rate_rps')) for r in rows])
    lat_acc, ref_err, true_speed = (np.full(len(rows), np.nan) for _ in range(3))
    for k, f in enumerate(frame):
        row = trace.get(f)
        if row is None:
            continue
        kin = row['plant_kinematics']
        lat_acc[k] = float(np.dot(kin['acceleration_mps2'], kin['right_vector']))
        true_speed[k] = row['speed']
        ref_err[k] = row['speed'] - row['reference_speed_mps']
    body = wrap_deg(yaw - road)
    course_err = wrap_deg(course - road)
    beta_hat = rear_slip_carla(speed, gyro)
    return dict(frame=frame, s=s, cte=cte, body=body, course=course_err, beta=wrap_deg(yaw - course),
                beta_hat=beta_hat, body_minus_beta_hat=wrap_deg(body - beta_hat), steer=steer, rate=rate,
                raw_steer=np.array([number(r.get('raw_steer')) for r in rows]), lat_acc=lat_acc,
                ref_err=ref_err, speed=true_speed, length=float(reference.s[-1]))


def p95(x):
    x = np.abs(x[np.isfinite(x)])
    return float(np.percentile(x, 95)) if len(x) else np.nan


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan


def window_metrics(f, lo, hi):
    m = (f['s'] >= lo) & (f['s'] <= hi)
    covered = bool(f['s'].min() <= lo + .5 and f['s'].max() >= hi - .5)
    cte = f['cte'][m]
    return dict(frames=int(m.sum()), covered=covered and int(m.sum()) >= MIN_FRAMES,
                cte_rms=rms(cte), cte_p95=p95(cte), cte_max=float(np.max(np.abs(cte))) if len(cte) else np.nan,
                cte_mean=float(np.mean(cte)) if len(cte) else np.nan,
                course_p95_deg=p95(f['course'][m]), body_p95_deg=p95(f['body'][m]),
                beta_p95_deg=p95(f['beta'][m]), beta_hat_p95_deg=p95(f['beta_hat'][m]),
                body_minus_beta_hat_p95_deg=p95(f['body_minus_beta_hat'][m]),
                course_nan_frames=int(np.sum(~np.isfinite(f['course'][m]))),
                speed_ref_err_rms=rms(f['ref_err'][m]),
                mean_speed=float(np.nanmean(f['speed'][m])) if m.any() else np.nan,
                lat_acc_p95=p95(f['lat_acc'][m]), steer_rate_p95=p95(f['rate'][m]),
                steer_max=float(np.nanmax(np.abs(f['steer'][m]))) if m.any() else np.nan,
                saturated_frac=float(np.mean(np.abs(f['steer'][m]) >= .8 - 1e-6)) if m.any() else np.nan)


def boot_ci(values, stat):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [np.nan, np.nan]
    rng = np.random.RandomState(BOOT_SEED)
    draws = stat(values[rng.randint(0, len(values), size=(BOOT, len(values)))], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, required=True, help='campaign root: <worker>/<perturbation>/<route>/<arm>/pursuit')
    ap.add_argument('--windows', type=Path, required=True)
    ap.add_argument('--perturbations', default='p01,p02,p03,p04,p05,p06,p07,p08,p09,p10')
    ap.add_argument('--anchor', default='p00')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('refusing to overwrite %s' % args.out)
    windows = json.loads(args.windows.read_text())
    seeds = args.perturbations.split(',')
    rows, profiles, missing, g2 = [], {}, [], []
    routes = list(dict.fromkeys(w['route_id'] for w in windows))
    for p in [args.anchor] + seeds:
        for route in routes:
            for arm in ARMS:
                found = sorted(args.campaign.glob('*/%s/%s/%s/pursuit' % (p, route, arm)))
                d = found[0] if len(found) == 1 else None
                if d is None or not (d / 'validation.json').is_file() or not (d / 'control.jsonl').is_file():
                    missing.append('%s/%s/%s (%d dirs)' % (p, route, arm, len(found)))
                    continue
                v = json.loads((d / 'validation.json').read_text())
                g2.append(dict(perturbation=p, route=route, arm=arm, status=v['status'], gate_pass=v['gate_pass'],
                               failed=[k for k, x in v['gates'].items() if not x]))
                f = case_frames(d)
                runtime = np.asarray(json.loads((d / 'route_reference.json').read_text())['world_xy'], float)
                for w in (w for w in windows if w['route_id'] == route):
                    m = window_metrics(f, *w['window_m'])
                    # Held-out stations were registered on the offline dense route: it must be this route.
                    offline = np.asarray(w.get('reference_xy', runtime), float)
                    m['reference_max_dev_m'] = (float(np.max(np.linalg.norm(offline - runtime, axis=1)))
                                                if offline.shape == runtime.shape else np.inf)
                    m['covered'] = m['covered'] and m['reference_max_dev_m'] <= .05
                    m['post_cte_rms'] = (rms(f['cte'][(f['s'] >= w['post_m'][0]) & (f['s'] <= w['post_m'][1])])
                                         if 'post_m' in w else np.nan)
                    rows.append(dict(perturbation=p, window=w['name'], route=route, arm=arm, **m))
                    if p == args.anchor:
                        continue
                    edges = np.arange(math.floor(w['window_m'][0]) - 10, math.ceil(w['window_m'][1]) + 11)
                    idx = np.digitize(f['s'], edges)
                    for b in range(1, len(edges)):
                        sel = idx == b
                        if sel.any():
                            course = f['course'][sel]
                            course = float(np.mean(course[np.isfinite(course)])) if np.isfinite(course).any() else np.nan
                            profiles.setdefault((w['name'], arm, float(edges[b - 1])), []).append(
                                (float(np.mean(f['cte'][sel])), course))
    args.out.mkdir(parents=True)
    keys = list(rows[0])
    with (args.out / 'case-window-metrics.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, keys, extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / 'station-profiles.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['window', 'arm', 'station_m', 'n', 'cte_median', 'cte_q25', 'cte_q75', 'course_median_deg'])
        for (name, arm, s), vals in sorted(profiles.items()):
            c = np.array(vals)
            writer.writerow([name, arm, s, len(c), *np.round(np.percentile(c[:, 0], [50, 25, 75]), 5),
                             round(float(np.nanmedian(c[:, 1])), 4) if np.isfinite(c[:, 1]).any() else ''])

    def series(name, arm, metric):
        out = {}
        for r in rows:
            if r['window'] == name and r['arm'] == arm and r['perturbation'] in seeds:
                out[r['perturbation']] = r[metric] if r['covered'] else np.nan
        return np.array([out.get(p, np.nan) for p in seeds], float)

    metrics = [k for k in keys if k not in ('perturbation', 'window', 'route', 'arm', 'frames', 'covered',
                                           'reference_max_dev_m')]
    per_arm, paired = {}, {}
    for w in windows:
        for arm in ARMS:
            per_arm[w['name'] + '|' + arm] = {}
            for m in metrics:
                x = series(w['name'], arm, m)
                per_arm[w['name'] + '|' + arm][m] = dict(
                    median=float(np.nanmedian(x)) if np.isfinite(x).any() else None,
                    ci=boot_ci(x, np.median), n=int(np.isfinite(x).sum()))
        for cand, base in [(CANDIDATE, PRODUCTION)] + SECONDARY:
            entry = {}
            for m in metrics:
                a, b = series(w['name'], cand, m), series(w['name'], base, m)
                if m == 'mean_speed':
                    entry['mean_speed_drop'] = paired_stats(b - a, b)
                entry[m] = paired_stats(a - b, b)
                entry[m + '_ratio'] = paired_stats(np.where(np.abs(b) > 0, a / b - 1., np.nan), b)
            paired['%s|%s-vs-%s' % (w['name'], cand, base)] = entry
    gates = evaluate_gates(windows, per_arm, paired, g2, seeds)
    anchor = {r['window'] + '|' + r['arm']: r['cte_rms'] for r in rows if r['perturbation'] == args.anchor}
    summary = dict(protocol='lateral-v2-protocol.md', windows=[{k: v for k, v in w.items() if k != 'reference_xy'}
                                                                for w in windows],
                   perturbations=seeds, anchor=args.anchor, anchor_cte_rms=anchor, missing_cases=missing,
                   per_arm=per_arm, paired=paired, gates=gates, g2=g2,
                   inputs=dict(analysis_sha256=sha(__file__), windows_sha256=sha(args.windows)),
                   definitions=dict(cte='left positive, nearest segment on route_reference world_xy, m',
                                    body='truth yaw minus centred 5 m chord heading, CARLA right positive, deg',
                                    course='rear-axle truth displacement (frame i-1 to i+1) direction minus chord, deg',
                                    beta_hat='-atan((v+1) w_left / (11 g)) from logged speed/gyro, deg',
                                    steer_rate='emitted steer difference over consecutive frames / dt',
                                    ci='percentile bootstrap, 10000 resamples, seed 0'))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1, allow_nan=True))
    print(json.dumps(gates['verdict'], indent=1))


def paired_stats(d, base):
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    return dict(mean=float(np.mean(d[ok])) if ok.any() else None, ci=boot_ci(d, np.mean), n=int(ok.sum()),
                n_positive=int(np.sum(d[ok] > 0)), base_median=float(np.nanmedian(base)) if np.isfinite(base).any() else None)


def evaluate_gates(windows, per_arm, paired, g2, seeds):
    checks = []

    def add(window, name, value, limit, passed, ci=None, note=None):
        checks.append(dict(window=window, check=name, value=value, limit=limit, passed=bool(passed),
                           ci=ci, note=note))

    main = paired['%s|%s-vs-%s' % (PRIMARY_WINDOW, CANDIDATE, PRODUCTION)]
    median = per_arm[PRIMARY_WINDOW + '|' + CANDIDATE]['cte_rms']
    n_ok = median['n'] == len(seeds)
    add(PRIMARY_WINDOW, 'P1 candidate median cte_rms', median['median'], PRIMARY_LIMIT,
        n_ok and median['median'] is not None and median['median'] <= PRIMARY_LIMIT)
    improve = main['cte_rms']
    add(PRIMARY_WINDOW, 'P2 paired cte_rms improvement CI excludes 0', improve['mean'], '<0 with CI upper < 0',
        n_ok and improve['ci'][1] < 0, ci=improve['ci'])
    for w in windows:
        entry = paired['%s|%s-vs-%s' % (w['name'], CANDIDATE, PRODUCTION)]
        guards = PRIMARY_GUARDS if w['name'] == PRIMARY_WINDOW else GUARDS
        for metric, kind, limit in guards:
            if metric == 'post_cte_rms' and 'post_m' not in w:
                continue
            stat = entry[metric + '_ratio'] if kind == 'ratio' else entry[metric]
            value = stat['mean']
            complete = stat['n'] == len(seeds)
            if kind == 'ratio' and (stat['base_median'] is None or abs(stat['base_median']) <= RATIO_FLOOR):
                add(w['name'], metric, value, limit, False, stat['ci'], 'insufficient evidence: baseline <= .01')
                continue
            passed = complete and value is not None and value <= limit
            note = None if not complete else ('uncertain: CI upper above limit' if passed and stat['ci'][1] > limit else None)
            add(w['name'], metric + (' rel' if kind == 'ratio' else ''), value, limit, passed, stat['ci'],
                note if complete else 'incomplete seeds')
    # G2: the candidate may not add a G2 failure where its paired production case passes.
    table = {(r['perturbation'], r['route'], r['arm']): r['gate_pass'] for r in g2}
    routes = sorted({w['route_id'] for w in windows})
    new_failures = [(p, r) for p in seeds for r in routes
                    if table.get((p, r, PRODUCTION)) is True and table.get((p, r, CANDIDATE)) is not True]
    complete = all((p, r, a) in table for p in seeds for r in routes for a in (CANDIDATE, PRODUCTION))
    add('all', 'G2 no new candidate failures', len(new_failures), 0, complete and not new_failures,
        note='new failures: %s' % new_failures if new_failures else None)
    # Informational only (protocol: course replaces body heading as the gated tracking heading).
    info = [dict(window=w['name'], check='frozen body_p95 increment <= 1 deg (not gated)',
                 value=paired['%s|%s-vs-%s' % (w['name'], CANDIDATE, PRODUCTION)]['body_p95_deg']['mean'],
                 ci=paired['%s|%s-vs-%s' % (w['name'], CANDIDATE, PRODUCTION)]['body_p95_deg']['ci'])
            for w in windows]
    for row in info:
        row['would_pass'] = row['value'] is not None and row['value'] <= 1.
    heldout = [w['name'] for w in windows if w.get('heldout')]
    frozen_guard = [c for c in checks if c['window'] not in heldout]
    held = [c for c in checks if c['window'] in heldout]
    verdict = dict(primary=all(c['passed'] for c in checks if c['check'].startswith('P')),
                   frozen_guards=all(c['passed'] for c in frozen_guard if not c['check'].startswith('P')),
                   heldout_guards=all(c['passed'] for c in held),
                   failed=[(c['window'], c['check']) for c in checks if not c['passed']])
    verdict['candidate_passes'] = verdict['primary'] and verdict['frozen_guards'] and verdict['heldout_guards']
    return dict(checks=checks, verdict=verdict, info=info)


if __name__ == '__main__':
    main()
