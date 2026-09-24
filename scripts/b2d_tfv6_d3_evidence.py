"""Tick-level D3 deviation evidence and same-seed B time/progress matches."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from b2d_tfv6_d3a import OUT, polyline_distance, world_plan
from b2d_tfv6_d3_analyze import ROOT, BEND, rows_for_case


def load(level,route,seed,arm):
    base=ROOT/'factorial/cases'/level/f'route-{route}'/f'seed-{seed}'/arm
    done=json.loads((base/'done.json').read_text())
    attempt=base/f"attempt-{done['attempt']}"
    frames={f['step']:f for f in (json.loads(x) for x in (attempt/'frames.jsonl').open())}
    semantic={f['step']:f for f in (json.loads(x) for x in (attempt/'d3_nav_semantic.jsonl').open())}
    return done,frames,semantic


def geometry(level,route):
    package=json.loads((ROOT/'dense'/f'{level}-{route}.json').read_text())
    dense=np.asarray([p['xyz'][:2] for p in package['dense']],dtype=float)
    bend=BEND[route]
    direction=dense[bend]-dense[bend-5];direction/=np.linalg.norm(direction)
    wrong=np.asarray([dense[bend]+direction*t for t in np.linspace(0,70,141)])
    return dense[bend:],wrong


def measurements(frame,semantic,correct,wrong):
    nav=frame.get('d3_nav') or {}
    route=world_plan(frame,'route_prediction')
    wp=world_plan(frame,'waypoint')
    kalman=np.asarray(semantic.get('kalman_world_xy'),dtype=float)
    gps=np.asarray(semantic.get('gps_world_xy'),dtype=float)
    result={'step':frame['step'],'sim_time_s':frame['sim_time'],
            'ego_x':frame['truth']['location'][0],'ego_y':frame['truth']['location'][1],
            'ego_speed_mps':frame['truth']['forward_speed_mps'],
            'ego_dense_distance_m':semantic.get('ego_dense_distance_m'),
            'rc_dense_index':semantic.get('rc_dense_index'),
            'target_dense_index':semantic.get('target_dense_index'),
            'target_dense_distance_m':semantic.get('target_dense_distance_m'),
            'target_world_xy':json.dumps(semantic.get('tfv6_target_world_xy')),
            'selected_target_world_xy':json.dumps(semantic.get('selected_target_world_xy')),
            'kalman_world_xy':json.dumps(semantic.get('kalman_world_xy')),
            'gps_world_xy':json.dumps(semantic.get('gps_world_xy')),
            'kalman_gps_delta_m':float(np.linalg.norm(kalman-gps)),
            'selected_pop_distance_m':nav.get('selected_pop_distance'),
            'command_1based':1+(nav['command'].index(max(nav['command']))) if nav.get('command') else None,
            'next_command_1based':1+(nav['next_command'].index(max(nav['next_command']))) if nav.get('next_command') else None,
            'steer':(frame.get('executed_control') or {}).get('steer'),
            'route_endpoint_world_xy':json.dumps(route[-1].tolist()) if len(route) else None,
            'waypoint_endpoint_world_xy':json.dumps(wp[-1].tolist()) if len(wp) else None,
            'route_endpoint_correct_dense_m':polyline_distance(route[-1:],correct) if len(route) else None,
            'route_endpoint_wrong_straight_m':polyline_distance(route[-1:],wrong) if len(route) else None,
            'waypoint_endpoint_correct_dense_m':polyline_distance(wp[-1:],correct) if len(wp) else None,
            'waypoint_endpoint_wrong_straight_m':polyline_distance(wp[-1:],wrong) if len(wp) else None,
            'light_state':(frame.get('d3_traffic_light') or {}).get('state'),
            'nearest_actor_json':json.dumps((frame.get('nearby_actors') or [])[:1])}
    return result


def main():
    rows=[];timeline=[]
    for done_path in sorted((ROOT/'factorial/cases').glob('*/*/*/*/done.json')):
        done=json.loads(done_path.read_text())
        level,route,seed,arm=done['level'],done['route'],done['seed'],done['arm']
        if route not in BEND or 'deviated from the route' not in done['official_status']:
            continue
        _,frames,semantic=load(level,route,seed,arm)
        _,bframes,bsemantic=load(level,route,seed,'B')
        correct,wrong=geometry(level,route)
        summary=rows_for_case('factorial',level,route,seed,arm)
        event_steps={'first_target_postturn':summary['first_target_postturn_step'],
                     'first_dense_wrong_plan':summary['first_dense_wrong_plan_step'],
                     'first_ego_offroute_3m':summary['first_ego_offroute_3m_step'],
                     'official_terminal':max(frames)}
        for event,step in event_steps.items():
            if step is None or step not in semantic:continue
            at=semantic[step]
            candidates=[s for s in bsemantic if s in bframes]
            same_time=min(candidates,key=lambda s:abs(s-step))
            same_progress=min(candidates,key=lambda s:(abs(bsemantic[s]['rc_dense_index']-at['rc_dense_index']),
                                                     abs(s-step)))
            for role,selected,ff,ss in [('deviation',step,frames,semantic),
                                        ('B_same_time',same_time,bframes,bsemantic),
                                        ('B_same_progress',same_progress,bframes,bsemantic)]:
                row={'level':level,'route':route,'seed':seed,'deviation_arm':arm,
                     'official_status':done['official_status'],'event':event,
                     'event_step':step,'role':role}
                row.update(measurements(ff[selected],ss[selected],correct,wrong))
                rows.append(row)
        anchor=summary['first_ego_offroute_3m_step'] or max(frames)
        for step in range(max(0,anchor-200),min(max(frames),anchor+20)+1):
            if step not in frames or step not in semantic:continue
            at=semantic[step]
            candidates=[s for s in bsemantic if s in bframes]
            same_time=min(candidates,key=lambda s:abs(s-step))
            same_progress=min(candidates,key=lambda s:(abs(bsemantic[s]['rc_dense_index']-at['rc_dense_index']),
                                                     abs(s-step)))
            for role,selected,ff,ss in [('deviation',step,frames,semantic),
                                        ('B_same_time',same_time,bframes,bsemantic),
                                        ('B_same_progress',same_progress,bframes,bsemantic)]:
                row={'level':level,'route':route,'seed':seed,'deviation_arm':arm,
                     'official_status':done['official_status'],'anchor_step':anchor,
                     'deviation_step':step,'role':role}
                row.update(measurements(ff[selected],ss[selected],correct,wrong))
                timeline.append(row)
    if not rows:raise RuntimeError('No completed D3b deviation case')
    path=OUT/'d3b-deviation-evidence.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    path=OUT/'d3b-deviation-timeline.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(timeline[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(timeline)
    print({'deviation_event_comparisons':len(rows)//3,'timeline_rows':len(timeline)})


if __name__=='__main__':main()
