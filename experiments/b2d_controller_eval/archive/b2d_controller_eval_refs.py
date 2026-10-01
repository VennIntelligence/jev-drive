"""Controller-neutral, time-parameterized L1 references fed and scored identically.

Two kinds, both built only from the route's dense global path and the nominal spawn:
  ramp     route oracle law (launch <= a_acc, cruise, end stop at a_dec) with the same curvature
           speed limit as profile (a_lat <= 2 m/s^2), jerk-limited
  profile  seeded stop-and-go profile: cruise levels change along the route, curvature
           speed limit, one or two mid-route stops with dwell, end stop
Every controller receives the reference itself as its plan (fixed-trace interface), so the
command and the scoring target are the same trajectory and no controller's adapter is favoured.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DT = .05


def resample(points, step=.5):
    points = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]
    points = points[keep]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.arange(0., arc[-1], step)
    s = np.r_[s, arc[-1]] if arc[-1] - s[-1] > 1e-6 else s
    return s, np.column_stack([np.interp(s, arc, points[:, i]) for i in (0, 1)])


def curvature(s, xy, window_m=5.):
    heading = np.unwrap(np.arctan2(*np.gradient(xy, s, axis=0)[:, ::-1].T))
    n = max(1, int(round(window_m / (s[1] - s[0]))))
    heading = np.convolve(np.pad(heading, n, mode='edge'), np.ones(2*n+1)/(2*n+1), mode='valid')
    return np.abs(np.gradient(heading, s))


def time_parameterize(s, vmax, stops, a_acc, a_dec, smooth_s):
    """Forward/backward speed limits on s, integrate to time with stop dwells, then smooth
    speed in time (two causal box passes) so acceleration and jerk stay bounded."""
    v = np.array(vmax, float)
    dwell = dict(stops)
    v[list(dwell)] = 0.
    v[0] = v[-1] = 0.
    ds = np.diff(s)
    for i in range(len(v)-1):
        v[i+1] = min(v[i+1], np.sqrt(v[i]**2 + 2*a_acc*ds[i]))
    for i in range(len(v)-2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2*a_dec*ds[i]))
    segment_t = 2*ds / np.maximum(v[:-1] + v[1:], 1e-9)
    knots_t, knots_s, clock = [], [], 0.
    for i in range(len(s)):
        clock += segment_t[i-1] if i else 0.
        knots_t.append(clock); knots_s.append(s[i])
        if i in dwell:
            clock += dwell[i]
            knots_t.append(clock); knots_s.append(s[i])
    grid = np.arange(0., knots_t[-1] + DT, DT)
    station = np.interp(grid, knots_t, knots_s)
    speed = np.r_[0., np.diff(station) / DT]
    n = max(1, int(round(smooth_s / DT)))
    for _ in range(2):
        speed = np.convolve(np.r_[speed, np.zeros(n)], np.ones(n)/n)[:len(speed)+n]
    station = np.minimum(np.r_[0., np.cumsum(speed[:-1]) * DT], s[-1])
    return np.arange(len(speed)) * DT, station, speed


def build(kind, world_xy, start_xy, cruise, seed, hold_s=5.):
    path = np.vstack((start_xy, world_xy))
    tangent = path[-1] - path[-2]
    path = np.vstack((path, path[-1] + 3*tangent/np.linalg.norm(tangent)))
    s, xy = resample(path)
    lateral_limit = np.sqrt(2. / np.maximum(curvature(s, xy), 1e-6))  # a_lat <= 2 m/s^2
    if kind == 'ramp':
        vmax, stops = np.minimum(float(cruise), lateral_limit), []
        t, station, speed = time_parameterize(s, vmax, stops, 2., 2., 1.)
    elif kind == 'profile':
        rng = np.random.default_rng(seed)
        levels = np.empty(len(s))
        edge = 0.
        while edge < s[-1]:
            span = rng.uniform(30., 60.)
            levels[(s >= edge) & (s < edge + span)] = min(10., rng.choice([.5, .75, 1., 1.25]) * cruise)
            edge += span
        levels[s >= edge - 1e-9] = levels[s < edge][-1] if np.any(s < edge) else cruise
        vmax = np.minimum(levels, lateral_limit)
        stops = []
        candidates = s[(s > 25.) & (s < s[-1] - 30.)]
        for _ in range(int(s[-1] > 80.) + int(s[-1] > 160.)):
            if len(candidates) == 0:
                break
            at = float(rng.choice(candidates))
            index = int(np.searchsorted(s, at))
            stops.append((index, float(rng.uniform(2.5, 4.))))
            candidates = candidates[np.abs(candidates - at) > 30.]
        t, station, speed = time_parameterize(s, vmax, sorted(stops), 1.5, 2.5, 1.)
    elif kind == 'crawl':
        # Queue-like creep over the first 40 m: 0.3-1.5 m/s levels, a stop every 8-20 m.
        rng = np.random.default_rng(seed + 1)
        keep = s <= 40.
        s, xy = s[keep], xy[keep]
        levels = np.empty(len(s))
        edge = 0.
        while edge < s[-1]:
            span = rng.uniform(5., 12.)
            levels[(s >= edge) & (s < edge + span)] = rng.choice([.3, .6, 1., 1.5])
            edge += span
        stops, at = [], rng.uniform(8., 20.)
        while at < s[-1] - 4.:
            stops.append((int(np.searchsorted(s, at)), float(rng.uniform(2.5, 4.))))
            at += rng.uniform(8., 20.)
        t, station, speed = time_parameterize(s, np.minimum(levels, lateral_limit[:len(s)]), stops, 1., 1.5, .5)
    else:
        raise ValueError(kind)
    hold = int(round(hold_s / DT))
    t = np.r_[t, t[-1] + DT*np.arange(1, hold+1)]
    station = np.r_[station, np.full(hold, station[-1])]
    speed = np.r_[speed, np.zeros(hold)]
    ref_xy = np.column_stack([np.interp(station, s, xy[:, i]) for i in (0, 1)])
    acceleration = np.gradient(speed, DT)
    return dict(kind=kind, seed=seed, cruise_mps=float(cruise), elapsed_s=t.round(6).tolist(),
                world_xy=ref_xy.tolist(), speed_mps=speed.tolist(),
                stops=[[float(s[i]), d] for i, d in sorted(stops)] if kind != 'ramp' else [],
                bounds=dict(max_speed=float(speed.max()), max_accel=float(acceleration.max()),
                            max_decel=float(-acceleration.min()),
                            max_abs_jerk=float(np.abs(np.gradient(acceleration, DT)).max()),
                            duration_s=float(t[-1]), length_m=float(s[-1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probes', type=Path, required=True,
                   help='b2d_controller_eval_l1_v2.py --kind probe output')
    p.add_argument('--cruises', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kinds', default='ramp,profile,crawl')
    a = p.parse_args()
    cruises = json.loads(a.cruises.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {kind: {} for kind in a.kinds.split(',')}
    for done in sorted(a.probes.glob('route-*/done.json')):
        trace = Path(json.loads(done.read_text())['p00/A']) / 'validation_trace.json'
        route = trace.parents[2].name
        rows = json.loads(trace.read_text())
        world = json.loads((trace.parent / 'route_reference.json').read_text())['world_xy']
        start = np.asarray(rows[0]['truth_xy'], float)
        cruise = cruises.get(route, cruises['default'])
        seed = int(hashlib.sha256(f'profile-{route}'.encode()).hexdigest()[:8], 16)
        for kind in manifest:
            ref = build(kind, np.asarray(world, float), start, cruise, seed)
            ref['route'] = route
            target = a.out / f'{kind}-{route}.json'
            target.write_text(json.dumps(ref, separators=(',', ':')) + '\n')
            manifest[kind][route] = str(target.resolve())
            print(kind, route, {k: round(v, 2) for k, v in ref['bounds'].items()}, ref['stops'])
    for kind, refs in manifest.items():
        (a.out / f'{kind}-traces.json').write_text(json.dumps(refs, indent=2) + '\n')


if __name__ == '__main__':
    main()
