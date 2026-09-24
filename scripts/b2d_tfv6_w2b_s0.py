"""W2b S0 replay on all valid W2/D2 logs and B standstill/launch coordinate check."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from b2d_tfv6_coordinates import REAR_OFFSET_M, rear_waypoints
from b2d_tfv6_tangent import case_inputs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'todos/2026-09-23-tfv6-controller/results/w2b'


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def rear_pose(frame):
    truth = frame['truth']
    yaw = math.radians(truth['rotation'][2])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(truth['location'][:2]) - REAR_OFFSET_M * np.array([c,s]), c, s


def coordinate_errors(frames, source, repeat, level, route, seed):
    by_step = {f['step']:f for f in frames}
    result = []
    for frame in frames:
        if frame.get('waypoint') is None or frame.get('truth') is None:
            continue
        future1 = by_step.get(frame['step'] + 20)
        if future1 is None or future1.get('truth') is None:
            continue
        speed = abs(frame['truth']['forward_speed_mps'])
        future_speed = abs(future1['truth']['forward_speed_mps'])
        if speed < .2 and future_speed < .5:
            context = 'standstill'
        elif speed < .5 and future_speed >= 1.:
            context = 'launch'
        else:
            continue
        current,c,s = rear_pose(frame)
        old = np.asarray(frame['rear_waypoint'],dtype=float)
        new = rear_waypoints(frame['waypoint'])
        for horizon,offset in ((.25,5),(.5,10),(1.,20)):
            future = by_step.get(frame['step']+offset)
            if future is None or future.get('truth') is None:
                continue
            actual,_,_ = rear_pose(future)
            for rule,points in (('W2',old),('W2b',new)):
                point = points[int(horizon/.25)-1]
                world = current + (c*point[0]+s*point[1],s*point[0]-c*point[1])
                delta = world-actual
                result.append(dict(source=source,repeat=repeat,level=level,route=route,
                                   seed=seed,context=context,horizon_s=horizon,rule=rule,
                                   long_error_m=float(c*delta[0]+s*delta[1]),
                                   lateral_error_m=float(s*delta[0]-c*delta[1])))
    return result


def main():
    cases=[]; values=[]
    for source,repeat,path in case_inputs():
        done=json.loads(Path(path).read_text())
        run=Path(done['run_dir'])
        with (run.parent/'frames.jsonl').open() as f:
            frames=[json.loads(line) for line in f]
        old_phantom=new_phantom=eligible=matched=0
        for frame in frames:
            actor,old=frame.get('waypoint'),frame.get('rear_waypoint')
            if actor is None or old is None:
                continue
            p=np.asarray(actor,dtype=float)*[1.,-1.]
            old=np.asarray(old,dtype=float)
            new=rear_waypoints(actor)
            arc=np.cumsum(np.linalg.norm(np.diff(np.vstack((np.zeros(2),p)),axis=0),axis=1))
            mask=arc>=REAR_OFFSET_M
            if not np.array_equal(new[mask],old[mask]):
                raise AssertionError(f'W2b differs past L: {path} step {frame["step"]}')
            matched += int(mask.sum())
            near=np.linalg.norm(p[0])<.3
            eligible+=int(near)
            old_phantom+=int(near and np.linalg.norm(old[0]-p[0])>.3)
            new_phantom+=int(near and np.linalg.norm(new[0]-p[0])>.3)
        cases.append(dict(source=source,repeat=repeat,level=done['level'],route=done['route'],
                          seed=done['seed'],arm=done['arm'],frames=len(frames),
                          eligible_ticks=eligible,W2_phantom_ticks=old_phantom,
                          W2b_phantom_ticks=new_phantom,bit_identical_points_past_L=matched))
        if done['arm']=='B':
            values.extend(coordinate_errors(frames,source,repeat,done['level'],done['route'],done['seed']))
    if len(cases)!=218 or sum(x['W2b_phantom_ticks'] for x in cases)!=0:
        raise AssertionError('S0 case count or phantom gate failed')
    write('s0-replay.csv',cases)
    summary=[]
    for context in ('standstill','launch'):
        for horizon in (.25,.5,1.):
            for rule in ('W2','W2b'):
                z=[x for x in values if x['context']==context and x['horizon_s']==horizon and x['rule']==rule]
                def stat(axis,pct):
                    return float(np.percentile(np.abs([x[axis] for x in z]),pct)) if z else math.nan
                summary.append(dict(context=context,horizon_s=horizon,rule=rule,n=len(z),
                                    long_abs_median_m=stat('long_error_m',50),
                                    long_abs_p95_m=stat('long_error_m',95),
                                    lateral_abs_median_m=stat('lateral_error_m',50),
                                    lateral_abs_p95_m=stat('lateral_error_m',95)))
    if not all(x['n'] for x in summary):
        raise AssertionError('S0 coordinate context empty')
    write('s0-coordinate.csv',summary)
    print(json.dumps({'cases':len(cases),'W2_phantom':sum(x['W2_phantom_ticks'] for x in cases),
                      'W2b_phantom':0,'bit_identical_points_past_L':sum(x['bit_identical_points_past_L'] for x in cases),
                      'coordinate_samples':len(values)}))


if __name__=='__main__':
    main()
