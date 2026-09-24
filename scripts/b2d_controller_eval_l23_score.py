"""Frozen plan-conditioned Task 10 L2 and route-cluster L3 scoring."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def rms(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if len(values) else None


def curvature(points):
    if points is None or len(points) < 4:
        return None
    p = np.asarray(points, float)
    a, b, c = p[1:4]
    ab, bc, ac = b-a, c-b, c-a
    denom = np.linalg.norm(ab)*np.linalg.norm(bc)*np.linalg.norm(ac)
    return float(2*abs(np.linalg.det(np.stack((ab,bc))))/denom) if denom > 1e-6 else 0.


def rear_truth(row, planner):
    if planner == 'TFv6':
        truth = row['truth']
        if truth is None:
            return None
        yaw = math.radians(truth['rotation'][2])
        xy = np.asarray(truth['location'][:2], float)
        velocity = np.asarray(truth['velocity'][:2], float)
    else:
        truth = row['truth']
        if truth.get('error'):
            return None
        yaw = math.radians(truth['rotation_deg'][2])
        xy = np.asarray(truth['xyz'][:2], float)
        velocity = np.asarray(truth['velocity'][:2], float)
    forward = np.array([math.cos(yaw),math.sin(yaw)])
    return xy - 1.388633220199954*forward, float(velocity@forward), yaw


def plan(row, planner, arm, speed):
    if planner == 'TFv6':
        if arm == 'A':
            points = row['route_prediction']
            value = row['target_speed']
            if points is None or value is None:
                return None
            points = np.asarray(points, float)*[1.,-1.]
            internal = None
            timed = None
        else:
            points = row['rear_waypoint']
            if points is None:
                return None
            points = np.asarray(points, float)
            value = float(np.linalg.norm(points[3]-points[1])/.5)
            segments = np.linalg.norm(np.diff(points,axis=0),axis=1)/.25
            internal = float(np.max(np.abs(np.diff(segments)/.25)))
            timed = points[1]
    else:
        prediction = row['prediction']
        if prediction is None:
            return None
        points = np.asarray(prediction['raw_waypoints'],float)*[1.,-1.]
        value = float(sum(np.linalg.norm(points[i+1]-points[i]) for i in range(3))/1.5)
        segments = np.linalg.norm(np.diff(points,axis=0),axis=1)/.5
        internal = float(np.max(np.abs(np.diff(segments)/.5)))
        timed = points[0]
    if not np.isfinite(points).all() or not np.isfinite(value):
        raise ValueError('Nonfinite planner prediction')
    curve = curvature(points)
    if curve is None:
        raise ValueError('Planner prediction has fewer than four points')
    demand = (value-speed)/.5
    lateral = value*value*curve if speed >= .5 else 0.
    feasible = (-4. <= demand <= 3. and (internal is None or internal <= 6.) and lateral <= 6.)
    return dict(speed=value, a_req=demand, a_internal=internal, a_lat=lateral,
                feasible=feasible, timed=timed)


def contiguous_events(mask):
    events=[];start=None
    for i,active in enumerate(list(mask)+[False]):
        if active and start is None:start=i
        if not active and start is not None:
            events.append((start,i));start=None
    return events


def smooth_jerk(values):
    a=np.asarray(values,float)
    valid=np.isfinite(a)
    if len(a)<25:return np.full_like(a,np.nan)
    a[~valid]=np.interp(np.flatnonzero(~valid),np.flatnonzero(valid),a[valid]) if valid.any() else 0.
    sm=np.convolve(a,np.ones(5)/5,mode='same')
    accel=(np.roll(sm,-5)-np.roll(sm,5))/.5
    jerk=(np.roll(accel,-5)-np.roll(accel,5))/.5
    jerk[:12]=jerk[-12:]=np.nan
    return jerk


def front_gap(row, planner, ego_xy, yaw):
    if planner=='TFv6':
        nearby=row.get('nearby_actors')
        if not isinstance(nearby,list):return None
        gaps=[x['distance_m'] for x in nearby if x['relative_xy_m'][0]>0 and
              abs(x['relative_xy_m'][1])<2.5 and x['type'].startswith('vehicle.')]
        return min(gaps) if gaps else None
    nearby=row['truth'].get('nearby_actors',[])
    if not nearby:return None
    co,si=math.cos(yaw),math.sin(yaw)
    gaps=[]
    for actor in nearby:
        if not actor['type'].startswith('vehicle.'):continue
        delta=np.asarray(actor['xyz'][:2])-ego_xy
        x=co*delta[0]+si*delta[1];y=-si*delta[0]+co*delta[1]
        if x>0 and abs(y)<2.5:gaps.append(float(np.linalg.norm(delta)))
    return min(gaps) if gaps else None


def load_case(done_path,planner):
    done=json.loads(done_path.read_text())
    arm=done['arm'];rid=done['route'];seed=int(done['seed'])
    if planner=='TFv6':
        attempt=done_path.parent/f"attempt-{done['attempt']}"
        rows=[json.loads(s) for s in (attempt/'frames.jsonl').open()]
        official_path=Path(done['run_dir'])/'attempts'/rid/'1'/'results.json'
        events=json.loads((attempt/'infractions.json').read_text())['infractions']
        hit=min((x['step'] for x in events if 'COLLISION' in x['event_type']),default=len(rows))
        rows=[r for r in rows if r['step']<hit]
    else:
        attempt=Path(done['run_dir'])/'attempts'/rid/'1'
        rows=[json.loads(s) for s in (attempt/'tcp-control.jsonl').open()]
        official_path=attempt/'results.json'
        criterion=json.loads((attempt/'criterion_events.json').read_text())
        events=criterion['events']
        hit=min((x['frame'] for x in events if 'COLLISION' in x['type']),default=10**18)
        rows=[r for r in rows if r['frame']<hit]
    record=json.loads(official_path.read_text())['_checkpoint']['records'][0]
    telemetry=[]
    for row in rows:
        truth=rear_truth(row,planner)
        if truth is None:continue
        xy,speed,yaw=truth
        selected=(row['executed_control'] if planner=='TFv6' else
                  dict(zip(('throttle','steer','brake'),row['selected_control'])))
        item=plan(row,planner,arm,speed)
        if item is None:continue
        item.update(xy=xy,speed_actual=speed,yaw=yaw,time=row['sim_time'] if planner=='TFv6' else row['timestamp'],
                    throttle=float(selected['throttle']),brake=float(selected['brake']),
                    gap=front_gap(row,planner,xy,yaw))
        telemetry.append(item)
    times=np.asarray([r['time'] for r in telemetry])
    if len(times)>1 and (np.any(np.diff(times)<=0) or np.max(np.abs(np.diff(times)-.05))>.01):
        raise ValueError(f'L2 telemetry not contiguous 20 Hz: {planner}/{rid}/{seed}/{arm}')
    n=len(telemetry)
    feasible=np.asarray([r['feasible'] for r in telemetry],bool)
    speeds=np.asarray([r['speed_actual'] for r in telemetry])
    targets=np.asarray([r['speed'] for r in telemetry])
    demand=np.asarray([r['a_req'] for r in telemetry])
    throttle=np.asarray([r['throttle'] for r in telemetry])
    brake=np.asarray([r['brake'] for r in telemetry])
    position=np.asarray([r['xy'] for r in telemetry])
    jerk_delta=smooth_jerk(speeds)-smooth_jerk(targets)
    timed=[]
    for i in range(n-10):
        if not feasible[i] or telemetry[i]['timed'] is None:continue
        delta=position[i+10]-position[i]
        yaw=telemetry[i]['yaw'];co,si=math.cos(yaw),math.sin(yaw)
        local=np.array([co*delta[0]+si*delta[1],si*delta[0]-co*delta[1]])
        timed.append(float(np.linalg.norm(local-telemetry[i]['timed'])))
    flip=0
    state=np.where(throttle>=.95,1,np.where(brake>=.95,-1,0))
    for i in range(5,n-5):
        if not feasible[i] or not state[i] or state[i]==state[i-1]:continue
        if -state[i] in state[i-5:i] and (np.all(demand[i-5:i+6]>=0) or np.all(demand[i-5:i+6]<=0)):
            flip+=1
    low_req=feasible&(np.abs(demand)<=1.5)
    false_stop=feasible&(targets>=2)&(speeds<.5)
    missed_stop=feasible&(targets<=.2)&(speeds>.5)
    bad_events=contiguous_events(~feasible)
    event_jerks=[];stop_dist=[];gaps=[]
    for start,end in bad_events:
        upper=min(n,start+40)
        event_jerks.extend(jerk_delta[start:upper])
        gaps.extend(x['gap'] for x in telemetry[start:upper] if x['gap'] is not None)
        if targets[start]<=.2 and speeds[start]>2:
            end_stop=next((i for i in range(start,upper) if speeds[i]<=.5),None)
            if end_stop is not None:
                stop_dist.append(float(np.sum(np.linalg.norm(np.diff(position[start:end_stop+1],axis=0),axis=1))))
    result=dict(planner=planner,route=rid,seed=seed,arm=arm,
        ds=float(record['scores']['score_composed']),rc=float(record['scores']['score_route']),
        official_status=record['status'],
        collisions_total=sum(len(value) for key,value in record['infractions'].items()
                             if key.startswith('collisions_')),
        completion=float(record['scores']['score_route']),ticks=len(rows),l2_frames=n,
        feasible_frames=int(feasible.sum()),feasible_share=float(feasible.mean()) if n else None,
        feasible_disp05_mean_m=float(np.mean(timed)) if timed else None,
        feasible_speed_mae_mps=float(np.mean(np.abs(speeds[feasible]-targets[feasible]))) if feasible.any() else None,
        feasible_extra_jerk_rms_mps3=rms(jerk_delta[feasible]),
        low_demand_full_pedal_share=float(np.mean((throttle[low_req]>=.95)|(brake[low_req]>=.95))) if low_req.any() else None,
        plan_unjustified_pedal_flips_per_min=flip/(n*.05/60) if n else None,
        false_stop_events_2s=sum(end-start>=40 for start,end in contiguous_events(false_stop)),
        missed_stop_events_2s=sum(end-start>=40 for start,end in contiguous_events(missed_stop)),
        infeasible_events=len(bad_events),infeasible_frames=int((~feasible).sum()),
        infeasible_extra_jerk_rms_mps3=rms(event_jerks),
        infeasible_stop_distance_m=float(np.median(stop_dist)) if stop_dist else None,
        infeasible_stop_distance_n=len(stop_dist),
        infeasible_min_front_gap_m=min(gaps) if gaps else None,
        infeasible_front_gap_samples=len(gaps))
    return result


def bootstrap_diff(rows,left,right,metric):
    lookup={(r['route'],r['seed'],r['arm']):r for r in rows}
    routes=sorted({r['route'] for r in rows})
    per_route=[];missing=[]
    for route in routes:
        vals=[]
        for seed in (0,1):
            a=lookup.get((route,seed,left));b=lookup.get((route,seed,right))
            if not a or not b or a[metric] is None or b[metric] is None:
                missing.append(f'{route}/{seed}/{left}-{right}')
                continue
            vals.append(a[metric]-b[metric])
        if len(vals)==2:per_route.append(np.mean(vals))
    if missing:
        return dict(mean=None,ci95=None,routes=len(per_route),missing_pairs=missing,
                    conclusion='not established: incomplete paired metric')
    x=np.asarray(per_route)
    rng=np.random.default_rng(20260924)
    samples=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1)
    return dict(mean=float(np.mean(x)),ci95=np.percentile(samples,[2.5,97.5]).tolist(),routes=len(x))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tfv6',type=Path,required=True)
    p.add_argument('--tcp',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for planner,root in [('TFv6',a.tfv6),('TCP',a.tcp)]:
        paths=sorted(root.glob('cases/*/route-*/seed-*/*/done.json')) if planner=='TFv6' else sorted(root.glob('cases/route-*/seed-*/*/done.json'))
        expected=64 if planner=='TFv6' else 80
        if len(paths)!=expected:raise ValueError(f'Expected {expected} {planner} cases, got {len(paths)}')
        rows.extend(load_case(path,planner) for path in paths)
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'l23-cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    summaries={}
    for planner in ('TFv6','TCP'):
        group=[r for r in rows if r['planner']==planner]
        native='A' if planner=='TFv6' else 'N'
        summaries[planner]={}
        for arm in sorted({r['arm'] for r in group}):
            s=[r for r in group if r['arm']==arm]
            metrics={key:(float(np.median([r[key] for r in s if r[key] is not None])) if any(r[key] is not None for r in s) else None)
                     for key in ('ds','feasible_share','feasible_disp05_mean_m','feasible_speed_mae_mps',
                                 'feasible_extra_jerk_rms_mps3','low_demand_full_pedal_share',
                                 'plan_unjustified_pedal_flips_per_min','infeasible_extra_jerk_rms_mps3',
                                 'infeasible_stop_distance_m','infeasible_min_front_gap_m')}
            metrics.update(cases=len(s),ds_mean=float(np.mean([r['ds'] for r in s])),
                           collisions=sum(r['collisions_total'] for r in s),
                           complete=sum(r['official_status']=='Completed' for r in s),
                           feasible_cases_at_least_50=sum(r['feasible_frames']>=50 for r in s),
                           feasible_cases_below_50=[f"{r['route']}/{r['seed']}"
                                                    for r in s if r['feasible_frames']<50])
            if arm!=native:
                metrics['l3_ds_difference']=bootstrap_diff(group,arm,native,'ds')
                baseline='B' if planner=='TFv6' else 'N'
                for metric in ('feasible_disp05_mean_m','feasible_speed_mae_mps',
                               'feasible_extra_jerk_rms_mps3'):
                    if not (planner=='TFv6' and arm=='A') and arm!=baseline:
                        metrics[metric+'_difference']=bootstrap_diff(group,arm,baseline,metric)
            summaries[planner][arm]=metrics
    (a.out/'l23-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(f'Wrote {len(rows)} cases and summaries to {a.out}')


if __name__=='__main__':main()
