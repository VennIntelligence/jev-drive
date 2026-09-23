#!/usr/bin/env python3
"""Read saved route geometry only; never imports CARLA, torch, or runtime agents.

python3 lateral_windows.py --out /fresh/output
Requires NumPy. Output is descriptive geometry/frame indexing, not qualification.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def geometry(xy):
    xy = np.asarray(xy, dtype=float)
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    z = np.arange(0., s[-1] + 1e-8, .5)
    def at(q):
        return np.column_stack([np.interp(q, s, xy[:,i]) for i in range(2)])
    v = at(z + 2.5) - at(z - 2.5)
    heading = np.unwrap(np.arctan2(v[:,1], v[:,0]))
    curvature = np.gradient(heading, z)
    groups = []
    for i in np.flatnonzero(abs(curvature) >= .02):
        if not groups or z[i] - z[groups[-1][-1]] > 3.:
            groups.append([i])
        else:
            groups[-1].append(i)
    segments = []
    for g in groups:
        a, b = g[0], g[-1]
        k, q = curvature[a:b+1], z[a:b+1]
        # Explicit trapezoid implementation is compatible with NumPy 1.x and 2.x.
        integral = lambda y: float(np.sum((y[1:]+y[:-1])*.5*np.diff(q)))
        right = float(np.degrees(integral(np.maximum(k, 0))))
        left = float(np.degrees(integral(np.maximum(-k, 0))))
        if right + left < 15.:
            continue
        net = float(np.degrees(heading[b] - heading[a]))
        peak = float(max(abs(k)))
        kind = 'S' if min(left, right) >= 15 else ('right' if net > 0 else 'left')
        if kind != 'S' and (abs(net) >= 60 or peak >= .08):
            kind += '-sharp'
        segments.append(dict(segment=len(segments)+1, kind=kind,
            core_start_m=float(z[a]), core_end_m=float(z[b]),
            start_m=max(0., float(z[a])-5.), end_m=min(float(s[-1]),float(z[b])+5.),
            heading_net_deg=net, heading_absolute_deg=right+left,
            left_accumulated_deg=left, right_accumulated_deg=right,
            peak_curvature_inv_m=peak))
    chord = xy[-1]-xy[0]
    offset = xy-xy[0]
    deviation = np.max(abs(chord[0]*offset[:,1]-chord[1]*offset[:,0]))/np.linalg.norm(chord)
    return xy, s, segments, dict(length_m=float(s[-1]), max_chord_deviation_m=float(deviation),
                               smoothed_heading_range_deg=float(np.degrees(np.ptp(heading))))


def project(xy, s, point):
    d = np.diff(xy, axis=0)
    u = np.clip(np.sum((np.asarray(point)-xy[:-1])*d, axis=1)/np.sum(d*d, axis=1),0,1)
    dist2 = np.sum((xy[:-1]+u[:,None]*d-point)**2, axis=1)
    i = int(np.argmin(dist2))
    return float(s[i]+u[i]*(s[i+1]-s[i]))


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--runs', type=Path, default=Path('/data/runs/b2d/controller'))
    ap.add_argument('--xml', type=Path, default=Path('/data/third_party/Bench2Drive/leaderboard/data/bench2drive220.xml'))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing an existing output directory; use a fresh version.')
    args.out.mkdir(parents=True)
    hashes = {}
    def record(p):
        p = Path(p)
        hashes[str(p.resolve())] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    record(args.xml); record(__file__)
    routes, windows, frames = [], [], []
    root = ET.parse(args.xml).getroot()
    for route in ('24211','1711','1773'):
        el = root.find("route[@id='%s']" % route)
        points = [[float(p.get('x')),float(p.get('y'))] for p in el.findall('./waypoints/position')]
        _, _, seg, stats = geometry(points)
        routes.append(dict(dataset='TCP-fixed-XML', route=route, source=str(args.xml), windows=len(seg), **stats))
    refs = sorted((args.runs/'development-v4').glob('*/pi-max/pursuit/route_reference.json'))
    refs += sorted((args.runs/'formal-v4/pursuit-seed0/attempts').glob('*/1/route_reference.json'))
    for ref in refs:
        record(ref)
        g2 = 'development-v4' in ref.parts
        route = ref.parts[-4] if g2 else ref.parts[-3]
        dataset = 'G2-v4' if g2 else 'Dev10-v4-pursuit-seed0-attempt1'
        xy, s, seg, stats = geometry(json.loads(ref.read_text())['world_xy'])
        routes.append(dict(dataset=dataset, route=route, source=str(ref), windows=len(seg), **stats))
        for window in seg:
            windows.append(dict(dataset=dataset, route=route, **window))
        logs = sorted(ref.parents[2].glob('*/*/validation_trace.json')) if g2 else [ref.parent/'control.jsonl']
        for log in logs:
            if not log.exists() or not seg:
                continue
            record(log)
            rows = json.loads(log.read_text()) if log.suffix == '.json' else [json.loads(l) for l in log.read_text().splitlines() if l]
            projected = [(r, project(xy,s,r['truth_xy'])) for r in rows if r.get('truth_xy') is not None]
            for window in seg:
                matched = [(r,p) for r,p in projected if window['start_m'] <= p <= window['end_m']]
                # Preserve every contiguous visit; never imply a frame span includes gaps.
                visits = []
                for r,p in matched:
                    if not visits or r['frame'] != visits[-1][-1][0]['frame']+1:
                        visits.append([])
                    visits[-1].append((r,p))
                if not visits:
                    frames.append(dict(dataset=dataset,route=route,segment=window['segment'],log=str(log),visit=0,count=0))
                for n, visit in enumerate(visits,1):
                    speeds = [r.get('speed',r.get('speed_mps',0.)) for r,p in visit]
                    frames.append(dict(dataset=dataset,route=route,segment=window['segment'],log=str(log),visit=n,
                        frame_start=visit[0][0]['frame'],frame_end=visit[-1][0]['frame'],count=len(visit),
                        sim_time_start=visit[0][0]['sim_time'],sim_time_end=visit[-1][0]['sim_time'],
                        moving_count_ge_2mps=sum(v >= 2. for v in speeds),
                        speed_min_mps=min(speeds),speed_max_mps=max(speeds)))
    csv_write(args.out/'routes.csv',routes)
    csv_write(args.out/'turn-windows.csv',windows)
    csv_write(args.out/'frame-windows.csv',frames)
    manifest = dict(inputs=hashes, outputs={}, protocol=dict(arc_step_m=.5, heading_chord_m=5.,
        curvature_threshold_inv_m=.02, merge_gap_m=3., min_absolute_heading_deg=15.,padding_m=5.,
        s_each_direction_min_deg=15.,sharp_net_heading_deg=60.,sharp_peak_curvature_inv_m=.08,
        moving_speed_mps=2.,frame_mapping='independent nearest projection of recorded truth rear-axle xy; diagnostic only'),
        scope='G2 all three available variants; Dev10 pursuit seed0 attempt1 only as frame index, not outcome comparison',
        limitations='Polyline smoothing depends on sparse route geometry. No turn window is not proof of no obstacle avoidance. World xy positive heading is CARLA right. Frame IDs are run-specific.')
    for p in args.out.glob('*.csv'):
        manifest['outputs'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(routes=len(routes),windows=len(windows),frame_visits=len(frames),out=str(args.out))))


if __name__ == '__main__':
    main()
