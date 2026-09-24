"""D3a: scan canonical W2/W2b logs without starting CARLA."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from b2d_controller import ConditionalPI
from b2d_tfv6_campaign import DEV10, HOLDOUT
from b2d_tfv6_coordinates import rear_waypoints

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'todos/2026-09-23-tfv6-controller/results/diagnosis/d3'
W2 = Path('/data/runs/b2d/tfv6-w2/formal')
W2B = Path('/data/runs/b2d/tfv6-w2b')
ROUTES = {'1': DEV10, '2': HOLDOUT}


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        (OUT/name).write_text('')
        return
    with (OUT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def source(level, route, seed, arm):
    root = W2 if arm in 'AB' else W2B
    return root/f'level{level}'/'cases'/level/f'route-{route}'/f'seed-{seed}'/arm/'done.json'


def case(level, route, seed, arm):
    path = source(level, route, seed, arm)
    done = json.loads(path.read_text())
    run = Path(done['run_dir'])
    frames = [json.loads(s) for s in (run.parent/'frames.jsonl').open()]
    record = json.loads((run/'attempts'/route/'1'/'results.json').read_text())['_checkpoint']['records'][0]
    return done, frames, record


def route_geometry(level, route):
    node = next(n for n in ET.parse(ROUTES[level]).getroot() if n.attrib['id']==route)
    points = np.asarray([[float(n.attrib['x']),float(n.attrib['y'])]
                         for n in node.find('waypoints')], dtype=float)
    gaps = np.linalg.norm(np.diff(points,axis=0),axis=1)
    jumps = [i for i,d in enumerate(gaps) if d>6.]
    turn = points[jumps[0]] if jumps else None
    junctions = [points[i] for i in jumps]
    for trigger in node.findall('.//trigger_point'):
        junctions.append(np.asarray([float(trigger.attrib['x']),float(trigger.attrib['y'])]))
    if turn is None:
        return {'points':points,'turn':None,'junctions':junctions,'correct':None,'wrong':None}
    i = jumps[0]
    heading = points[i]-points[i-1]
    heading = heading/np.linalg.norm(heading)
    wrong = np.stack([turn+heading*t for t in np.linspace(0,70,71)])
    correct = points[i:]
    return {'points':points,'turn':turn,'junctions':junctions,'correct':correct,'wrong':wrong}


def nearest_junction(xy, geo):
    if not geo['junctions']:
        return None
    return float(min(np.linalg.norm(xy-p) for p in geo['junctions']))


def polyline_distance(points, line):
    if line is None or len(points)==0:
        return None
    p=np.asarray(points,dtype=float)
    a=line[:-1];b=line[1:];v=b-a
    t=np.clip(np.sum((p[:,None,:]-a[None,:,:])*v[None,:,:],axis=2)
              /np.maximum(np.sum(v*v,axis=1),1e-8)[None,:],0,1)
    proj=a[None,:,:]+t[:,:,None]*v[None,:,:]
    return float(np.mean(np.min(np.linalg.norm(p[:,None,:]-proj,axis=2),axis=1)))


def world_plan(frame, key):
    local=frame.get(key)
    if not local or not frame.get('truth'):
        return np.empty((0,2))
    xy=np.asarray(frame['truth']['location'][:2],dtype=float)
    yaw=math.radians(frame['truth']['rotation'][2]); c,s=math.cos(yaw),math.sin(yaw)
    p=np.asarray(local,dtype=float)[:,:2]
    # TFv6 local axes: x forward, y right; CARLA world y points left at yaw=0.
    return xy + np.column_stack((c*p[:,0]+s*p[:,1],s*p[:,0]-c*p[:,1]))


def stall_rows(level,route,seed,arm,frames,geo):
    rows=[];start=None
    for i in range(len(frames)+1):
        moving = i==len(frames) or not frames[i].get('truth') or abs(frames[i]['truth']['forward_speed_mps'])>=.5
        if not moving and start is None:
            start=i
        if moving and start is not None:
            end=i-1
            if end-start+1>=100:
                chunk=frames[start:end+1]
                mid=chunk[len(chunk)//2]; xy=np.asarray(mid['truth']['location'][:2])
                at_start=frames[start]['sim_time']; at_end=frames[end]['sim_time']
                rows.append(dict(level=level,route=route,seed=seed,arm=arm,
                    start_step=frames[start]['step'],end_step=frames[end]['step'],
                    start_s=round(at_start,3),end_s=round(at_end,3),
                    duration_s=round(at_end-at_start+.05,3),x=float(xy[0]),y=float(xy[1]),
                    nearest_xml_turn_or_trigger_m=nearest_junction(xy,geo),
                    first_motion_before=any(abs(f['truth']['forward_speed_mps'])>=1 for f in frames[:start] if f.get('truth')),
                    median_speed_mps=float(np.median([abs(f['truth']['forward_speed_mps']) for f in chunk]))))
            start=None
    return rows


def plan_timeline(level,route,seed,arm,frames,geo,first,last):
    result=[];first_wrong={}
    for frame in frames[first:last+1]:
        if not frame.get('truth'):continue
        ego=np.asarray(frame['truth']['location'][:2]); near=nearest_junction(ego,geo)
        row=dict(level=level,route=route,seed=seed,arm=arm,step=frame['step'],
            sim_time_s=frame['sim_time'],x=float(ego[0]),y=float(ego[1]),
            speed_mps=frame['truth']['forward_speed_mps'],nearest_xml_turn_or_trigger_m=near,
            steer=(frame.get('executed_control') or {}).get('steer'),
            c_raw_steer=(frame.get('raw_control') or {}).get('C',{}).get('steer'),
            c_steer_saturation=None,command='unlogged',target_point='unlogged')
        raw=(frame.get('raw_control') or {}).get('C')
        if raw:row['c_steer_saturation']=abs(raw['steer'])>=.8-1e-4
        for key,prefix in [('waypoint','waypoint'),('route_prediction','route')]:
            world=world_plan(frame,key)
            row[f'{prefix}_correct_m']=polyline_distance(world,geo['correct'])
            row[f'{prefix}_wrong_m']=polyline_distance(world,geo['wrong'])
            row[f'{prefix}_world_xy_json']=json.dumps(world.tolist())
            # Compare the future endpoint only after the plan is near the bend;
            # both branches share the approach and should not be classified there.
            endpoint=world[-1:] if len(world) else world
            dc=polyline_distance(endpoint,geo['correct'])
            dw=polyline_distance(endpoint,geo['wrong'])
            row[f'{prefix}_endpoint_correct_m']=dc
            row[f'{prefix}_endpoint_wrong_m']=dw
            row[f'{prefix}_wrong_preferred']=bool(dc is not None and dw is not None and
                                                  near is not None and near<35 and dw+2<dc)
            if row[f'{prefix}_wrong_preferred'] and prefix not in first_wrong:
                first_wrong[prefix]=frame['step']
        result.append(row)
    return result,first_wrong


def speed_options(frame):
    wp=frame.get('waypoint')
    if not wp:return None
    p=np.asarray(wp,dtype=float);r=rear_waypoints(p)
    points=np.vstack((np.zeros(2),r))
    segments=np.linalg.norm(np.diff(points,axis=0),axis=1)
    return {'first':float(segments[0]/.25),
            'author':float(2*np.linalg.norm(p[1]-p[3])),
            'mean8':float(np.sum(segments)/2.)}


def stall_timeline(level,route,seed,arm,frames,stalls):
    rows=[];counter=[]
    for segment in stalls:
        if segment['duration_s']<5:continue
        start,end=segment['start_step'],segment['end_step']
        pis={key:ConditionalPI(-1,.75,kp=.5,ki=.25) for key in ('first','author','mean8')}
        last_time=None
        counts={key:0 for key in pis};eligible=0;matches=0
        for frame in frames[start:end+1]:
            speeds=speed_options(frame)
            if speeds is None:continue
            eligible+=1;now=frame['sim_time'];elapsed=.05 if last_time is None else max(.0001,now-last_time)
            last_time=now;speed=frame['controller_speed_mps']
            if not isinstance(speed,(float,int)):continue
            raw=frame['raw_control'];final=frame['final_control'];actual=frame['executed_control']
            c=raw['C'];b=raw['B'];h=frame.get('heuristic') or {}
            calc={}
            for key,desired in speeds.items():
                effort=pis[key].step(desired-speed,elapsed)
                throttle=max(effort,0.);brake=max(-effort,0.)
                if desired<.05 and speed<.1:
                    throttle=0.;brake=1.;pis[key].reset()
                calc[key]=(throttle,brake)
                counts[key]+=int(throttle>.05 and brake<.05)
            matches+=int(abs(calc['first'][0]-c['throttle'])<.05 and
                         abs(calc['first'][1]-c['brake'])<.05)
            changed=any(abs(final[arm][k]-raw[arm][k])>1e-5 for k in ('throttle','brake'))
            rows.append(dict(level=level,route=route,seed=seed,arm=arm,
                segment_start_step=start,step=frame['step'],sim_time_s=now,
                x=frame['truth']['location'][0],y=frame['truth']['location'][1],
                speed_mps=speed,tfv6_target_mps=frame['target_speed'],
                first_mps=speeds['first'],author_mps=speeds['author'],mean8_mps=speeds['mean8'],
                c_reason=frame['controller_reason'].get('C'),
                c_raw_throttle=c['throttle'],c_raw_brake=c['brake'],
                b_shadow_throttle=b['throttle'],b_shadow_brake=b['brake'],
                actual_throttle=actual['throttle'],actual_brake=actual['brake'],
                postprocessor_changed=changed,creep_active=h.get('creep_active'),
                stop_sign_active=h.get('stop_sign_active'),
                offline_first_throttle=calc['first'][0],offline_first_brake=calc['first'][1],
                offline_author_throttle=calc['author'][0],offline_author_brake=calc['author'][1],
                offline_mean8_throttle=calc['mean8'][0],offline_mean8_brake=calc['mean8'][1]))
        counter.append(dict(level=level,route=route,seed=seed,arm=arm,start_step=start,
            end_step=end,duration_s=segment['duration_s'],eligible_ticks=eligible,
            offline_first_matches_logged_raw_within_0_05=matches,
            offline_first_go_ticks=counts['first'],offline_author_go_ticks=counts['author'],
            offline_mean8_go_ticks=counts['mean8']))
    return rows,counter


def main():
    deviations=[];stalls=[];route_rows=[];stall_rows_out=[];counter=[];firsts=[]
    cache={}
    count=0
    for level,xml in ROUTES.items():
        for node in ET.parse(xml).getroot():
            route=node.attrib['id'];geo=route_geometry(level,route)
            for seed in range(3):
                for arm in 'ABCD':
                    done,frames,record=case(level,route,seed,arm);count+=1
                    if route in ('2084','27529','2091'):
                        cache[(level,route,seed,arm)]=(frames,geo)
                    local_stalls=stall_rows(level,route,seed,arm,frames,geo)
                    stalls.extend(local_stalls)
                    if 'deviated from the route' in record['status']:
                        frame=frames[-1];xy=np.asarray(frame['truth']['location'][:2])
                        deviations.append(dict(level=level,route=route,seed=seed,arm=arm,
                            step=frame['step'],sim_time_s=frame['sim_time'],ds=record['scores']['score_composed'],
                            rc=record['scores']['score_route'],x=float(xy[0]),y=float(xy[1]),
                            nearest_xml_turn_or_trigger_m=nearest_junction(xy,geo),status=record['status']))
                    if route=='2091':
                        selected=[s for s in local_stalls if s['duration_s']>=5]
                        rr,cc=stall_timeline(level,route,seed,arm,frames,selected)
                        stall_rows_out.extend(rr);counter.extend(cc)
    assert count==192
    visited=set()
    for dev in deviations:
        if dev['route'] not in ('2084','27529'):continue
        level,route,seed=dev['level'],dev['route'],dev['seed']
        for arm in (dev['arm'],'B'):
            key=(level,route,seed,arm)
            if key in visited:continue
            visited.add(key)
            frames,geo=cache[(level,route,seed,arm)]
            if arm=='B':
                turn=geo['turn'];i=min(range(len(frames)),key=lambda k:
                    np.linalg.norm(np.asarray(frames[k]['truth']['location'][:2])-turn))
                windows=[('approach',max(0,i-200),min(len(frames)-1,i+80))]
            else:
                turn=geo['turn'];i=min(range(min(len(frames),600)),key=lambda k:
                    np.linalg.norm(np.asarray(frames[k]['truth']['location'][:2])-turn))
                windows=[('approach',max(0,i-200),min(len(frames)-1,i+80)),
                         ('pre_official_end',max(0,dev['step']-200),dev['step'])]
            for kind,first,last in windows:
                rows,found=plan_timeline(level,route,seed,arm,frames,geo,first,last)
                for row in rows:row['window_kind']=kind
                route_rows.extend(rows)
                firsts.append(dict(level=level,route=route,seed=seed,arm=arm,
                    window_kind=kind,window_start_step=first,window_end_step=last,
                    first_wrong_waypoint_step=found.get('waypoint'),
                    first_wrong_route_step=found.get('route'),
                    command_and_target_point='not logged in W2/W2b; D3b required'))
    write('all-deviations.csv',deviations)
    write('all-stalls.csv',stalls)
    write('route-timeline.csv',route_rows)
    write('route-first-wrong.csv',firsts)
    write('2091-stall-timeline.csv',stall_rows_out)
    write('2091-openloop.csv',counter)
    summary={'cases':count,'deviations':len(deviations),'deviation_routes':sorted(set(r['route'] for r in deviations),key=int),
             'stalls':len(stalls),'stall_routes':sorted(set(r['route'] for r in stalls),key=int),
             'route_timeline_ticks':len(route_rows),'2091_stall_ticks':len(stall_rows_out),
             'geometry_note':'Junction distance uses nearest route-XML turn or scenario trigger; route branches use evaluator XML nodes as piecewise-linear approximation to its dense global plan.'}
    (OUT/'d3a-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
