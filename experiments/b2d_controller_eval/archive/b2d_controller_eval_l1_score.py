"""Score Task 10's frozen L1 oracle/expert CARLA runs, case-first."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def arc_path(points):
    points = np.asarray(points, float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]
    points = points[keep]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, arc


def oracle_reference(path, cruise, start):
    source = json.loads(path.read_text())
    world = np.asarray(source['world_xy'], float)
    end = world[-1] - world[-2]
    world = np.vstack((start, world, world[-1] + 3*end/np.linalg.norm(end)))
    path, arc = arc_path(world)
    t, station, speed = [0.], [0.], [0.]
    for _ in range(2400):
        remaining = max(arc[-1] - station[-1], 0.)
        command = min(cruise, 2*(t[-1]+.05), np.sqrt(4*remaining))
        advance = min(remaining, command*.05)
        station.append(station[-1] + advance)
        speed.append(command if advance > 0 else 0.)
        t.append(t[-1]+.05)
        if remaining <= 1e-4:
            break
    for _ in range(100):
        station.append(arc[-1]);speed.append(0.);t.append(t[-1]+.05)
    xy = np.column_stack((np.interp(station, arc, path[:, 0]),
                          np.interp(station, arc, path[:, 1])))
    return np.asarray(t), xy, np.asarray(speed), path


def expert_reference(path):
    data = json.loads(path.read_text())
    t = np.asarray(data['elapsed_s'], float)
    xy = np.asarray(data['world_xy'], float)
    speed = np.asarray(data['speed_mps'], float)
    if len(t) < 20 or np.any(np.diff(t) <= 0) or xy.shape != (len(t), 2):
        raise ValueError(f'Invalid expert reference {path}')
    return t, xy, speed, arc_path(xy)[0]


def cte_to_path(xy, path):
    start, delta = path[:-1], np.diff(path, axis=0)
    norm2 = np.sum(delta*delta, axis=1)
    result = []
    for chunk in np.array_split(xy, max(1, int(np.ceil(len(xy)/256)))):
        diff = chunk[:, None, :] - start[None, :, :]
        fraction = np.clip(np.sum(diff*delta[None,:,:], axis=2)/np.maximum(norm2, 1e-12), 0, 1)
        closest = start[None,:,:] + fraction[:,:,None]*delta[None,:,:]
        result.extend(np.min(np.linalg.norm(chunk[:,None,:]-closest, axis=2), axis=1))
    return np.asarray(result)


def jerk(speed, t):
    # Frozen centered five-tick smoothing and centered 0.25 s differences.
    smooth = np.convolve(speed, np.ones(5)/5, mode='same')
    acceleration = (np.roll(smooth, -5)-np.roll(smooth, 5))/.5
    result = (np.roll(acceleration, -5)-np.roll(acceleration, 5))/.5
    return result[12:-12]


def one(path, ref_kind, reference, cruise, oracle_start=None):
    summary = json.loads(path.read_text())
    run = path.parent
    rows = json.loads((run/'validation_trace.json').read_text())
    controls = [json.loads(s) for s in (run/'control.jsonl').open()]
    if (summary['exception'] or summary['cleanup_errors'] or summary['telemetry_parse_errors']
            or not summary['gates']['telemetry_complete'] or len(rows) != len(controls)
            or [r['frame'] for r in rows] != [r['frame'] for r in controls]):
        raise ValueError(f'Invalid L1 case {path}')
    actual_t = np.asarray([r['elapsed_s'] for r in rows])
    actual_xy = np.asarray([r['truth_xy'] for r in rows])
    actual_speed = np.asarray([r['speed'] for r in rows])
    if ref_kind == 'expert':
        t, ref_xy, ref_speed, path_xy = expert_reference(reference)
    else:
        if oracle_start is None or np.linalg.norm(actual_xy[0] - oracle_start) > .05:
            raise ValueError(f'Paired oracle spawn differs by >5 cm: {path}')
        t, ref_xy, ref_speed, path_xy = oracle_reference(run/'route_reference.json', cruise,
                                                        oracle_start)
    ego_xy = np.column_stack((np.interp(t, actual_t, actual_xy[:,0]),
                              np.interp(t, actual_t, actual_xy[:,1])))
    ego_speed = np.interp(t, actual_t, actual_speed)
    ego_speed[t > actual_t[-1]] = 0.
    cte = cte_to_path(ego_xy, path_xy)
    xy_error = np.linalg.norm(ego_xy - ref_xy, axis=1)
    speed_error = ego_speed - ref_speed
    delta_jerk = jerk(ego_speed, t) - jerk(ref_speed, t)
    cte_rms = float(np.sqrt(np.mean(cte**2)))
    xy_rms = float(np.sqrt(np.mean(xy_error**2)))
    speed_rms = float(np.sqrt(np.mean(speed_error**2)))
    primary = float(np.sqrt((cte_rms/.5)**2 + (xy_rms/2)**2 + (speed_rms/.5)**2))
    tangent = path_xy[-1] - path_xy[-2]
    tangent = tangent / np.linalg.norm(tangent)
    overshoot = float(max(0, np.max((actual_xy-path_xy[-1])@tangent)))
    convergence = None
    for i in range(len(t)-20):
        if ref_speed[i] >= 3 and np.all(np.abs(speed_error[i:i+20]) <= .5):
            convergence = float(t[i]);break
    terminal = bool(len(rows) >= 100 and
                    np.all(actual_speed[-100:] < .1) and
                    np.all(np.linalg.norm(actual_xy[-100:] - ref_xy[-1], axis=1) < 1.))
    return dict(route=summary['route_id'], seed=summary['perturbation_id'],
                arm=summary['variant'], reference=ref_kind,
                status=summary['status'], ticks=len(rows), ref_duration_s=float(t[-1]),
                cte_rms_m=cte_rms, time_xy_rms_m=xy_rms, speed_rms_mps=speed_rms,
                primary=primary, extra_jerk_rms_mps3=float(np.sqrt(np.mean(delta_jerk**2))),
                final_overshoot_m=overshoot, start_convergence_s=convergence,
                terminal_hold=terminal, collision_count=len(summary['collisions']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--expert', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--cruises', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    refs = json.loads(a.references.read_text())
    cruises = json.loads(a.cruises.read_text())
    results = []
    starts = {}
    for path in sorted(a.oracle.glob('p0[1-5]/*/A/*/validation_trace.json')):
        rows = json.loads(path.read_text())
        starts[(path.parents[3].name, path.parents[2].name)] = np.asarray(rows[0]['truth_xy'], float)
    if len(starts) != 8*5:
        raise ValueError(f'Expected 40 paired oracle start anchors; got {len(starts)}')
    for kind, root in [('oracle', a.oracle), ('expert', a.expert)]:
        paths = sorted(root.glob('p0[1-5]/*/[ABCD]/*/validation.json'))
        if len(paths) != 8*5*4:
            raise ValueError(f'Expected 160 {kind} L1 cases; got {len(paths)}')
        for path in paths:
            route = path.parents[2].name
            reference = Path(refs[route]) if kind == 'expert' else None
            seed = path.parents[3].name
            results.append(one(path, kind, reference, cruises.get(route, cruises['default']),
                               starts[(seed, route)] if kind == 'oracle' else None))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(results)
    print(f'Wrote {len(results)} L1 scores to {a.out}')


if __name__ == '__main__':
    main()
