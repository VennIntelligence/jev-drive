"""Summarize the D3b factorial and Kalman runs by route, seed and arm."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from b2d_tfv6_d3a import OUT, polyline_distance, world_plan

ROOT=Path('/data/runs/b2d/tfv6-d3')
BEND={'2084':33,'27529':16}


def rows_for_case(phase,level,route,seed,arm):
    case=ROOT/phase/'cases'/level/f'route-{route}'/f'seed-{seed}'/arm
    done=json.loads((case/'done.json').read_text())
    attempt=case/f"attempt-{done['attempt']}"
    frames=[json.loads(line) for line in (attempt/'frames.jsonl').open()]
    nav=[json.loads(line) for line in (attempt/'d3_nav_semantic.jsonl').open()]
    nav_by_step={n['step']:n for n in nav}
    package=json.loads((ROOT/'dense'/f'{level}-{route}.json').read_text())
    dense=np.asarray([p['xyz'][:2] for p in package['dense']],dtype=float)
    bend=BEND.get(route)
    correct=wrong=None
    if bend is not None:
        correct=dense[bend:]
        direction=dense[bend]-dense[bend-5];direction/=np.linalg.norm(direction)
        wrong=np.asarray([dense[bend]+direction*t for t in np.linspace(0,70,141)])
    step_after_turn=None;distance_at_jump=None;first_plan_wrong=None;first_offroute=None
    max_ego_dense_distance_m=max((n['ego_dense_distance_m'] for n in nav),default=None)
    first_motion=None;nearby_vehicle_ticks=red_light_ticks=post_changed=hold_ticks=0
    plan_ticks=nav_ticks=0
    for f in frames:
        n=nav_by_step.get(f['step'])
        if n:
            nav_ticks+=1
            if bend is not None:
                after=package['agent_sparse_dense_indices'][2]
                if step_after_turn is None and n['target_dense_index']>=after:
                    step_after_turn=f['step']
                    distance_at_jump=float(np.linalg.norm(np.asarray(n['ego_world_xy'])-dense[bend]))
                if first_offroute is None and n['ego_dense_distance_m']>3:
                    first_offroute=f['step']
        if first_motion is None and f.get('truth') and f['truth']['forward_speed_mps']>.5:
            first_motion=f['step']
        actors=f.get('nearby_actors') or []
        if any(x['type'].startswith(('vehicle.','walker.')) and x['distance_m']<10 for x in actors):
            nearby_vehicle_ticks+=1
        if (f.get('d3_traffic_light') or {}).get('state')=='Red':red_light_ticks+=1
        if (f.get('controller_reason') or {}).get('C')=='stop_hold':hold_ticks+=1
        raw=f.get('raw_control') or {};final=f.get('final_control') or {}
        if arm in raw and arm in final and any(abs(raw[arm][k]-final[arm][k])>1e-5 for k in ('steer','throttle','brake')):
            post_changed+=1
        if bend is not None and first_plan_wrong is None and f.get('truth'):
            ego=np.asarray(f['truth']['location'][:2]);dist=float(np.linalg.norm(ego-dense[bend]))
            if dist<35:
                route_world=world_plan(f,'route_prediction')
                if len(route_world):
                    endpoint=route_world[-1:]
                    dc=polyline_distance(endpoint,correct);dw=polyline_distance(endpoint,wrong)
                    if dc is not None and dw is not None:
                        plan_ticks+=1
                        if dw+2<dc:first_plan_wrong=f['step']
    record=json.loads((Path(done['run_dir'])/'attempts'/route/'1'/'results.json').read_text())['_checkpoint']['records'][0]
    return {'phase':phase,'level':level,'route':route,'seed':seed,'arm':arm,
            'official_status':done['official_status'],'ds':record['scores']['score_composed'],
            'rc':record['scores']['score_route'],'ticks':len(frames),'nav_ticks':nav_ticks,
            'first_motion_step':first_motion,'first_target_postturn_step':step_after_turn,
            'distance_to_bend_at_target_jump_m':distance_at_jump,
            'first_dense_wrong_plan_step':first_plan_wrong,'first_ego_offroute_3m_step':first_offroute,
            'max_ego_dense_distance_m':max_ego_dense_distance_m,
            'red_light_ticks':red_light_ticks,'near_actor_10m_ticks':nearby_vehicle_ticks,
            'c_stop_hold_ticks':hold_ticks,'postprocessor_changed_ticks':post_changed}


def main():
    rows=[]
    for phase in ('factorial','kalman'):
        for done_path in sorted((ROOT/phase/'cases').glob('*/*/*/*/done.json')):
            d=json.loads(done_path.read_text())
            rows.append(rows_for_case(phase,d['level'],d['route'],d['seed'],d['arm']))
    if not rows:raise RuntimeError('No D3b completed cases')
    path=OUT/'d3b-cases.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print({'cases':len(rows),'path':str(path)})


if __name__=='__main__':main()
