"""Read logged W2/D3b truth and controls; diagnose MKZ steer-to-curvature map."""
from pathlib import Path
import csv
import glob
import json
import math
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from b2d_controller import Controller
from b2d_tfv6_steer_diagnosis import curvature,write_csv

OUT=Path('results/diagnosis/plant')
FULL=Path('/data/runs/b2d/tfv6-w2/diagnosis/plant')
L=2.8604714913890885
TRACK=1.5929
MAX=math.radians(69.99999237060547)
CURVE=np.array([[0.,1.],[20.,.9],[60.,.8],[120.,.7]])
COMMON=dict(longitudinal_mode='pi',lookahead='max',pi_kp=.5,pi_ki=.25,max_lookahead_time_s=.5)


def cases():
    items=[]
    for root,source in (('/data/runs/b2d/tfv6-w2/formal','W2'),
                        ('/data/runs/b2d/tfv6-d3/factorial/cases/2','D3b')):
        for p in glob.glob(root+'/**/done.json',recursive=True):
            if '/aborted/' in p:continue
            parts=Path(p).parts
            arm=Path(p).parent.name
            if arm not in ('A','B','C','D','E','F'):continue
            route=next((x[6:] for x in parts if x.startswith('route-')),None)
            seed=next((int(x[5:]) for x in parts if x.startswith('seed-')),None)
            if route is None or seed is None:continue
            done=json.loads(Path(p).read_text())
            frames=Path(p).parent/f"attempt-{done['attempt']}"/'frames.jsonl'
            if frames.exists():items.append((source,route,seed,arm,str(frames)))
    return sorted(items)


def scale(speed):
    return float(np.interp(abs(speed)*3.6,CURVE[:,0],CURVE[:,1]))


def map_curvature(steer,speed):
    # CARLA frame, positive means right; CARLA applies nominal to inner wheel.
    k_nom=math.tan(float(steer)*MAX*scale(speed))/L
    k_ack=k_nom/(1.+TRACK*abs(k_nom)/2.)
    k_unscaled=math.tan(float(steer)*MAX)/L
    return k_nom,k_ack,k_unscaled


def replay_controller(arm):
    if arm=='C':return Controller(preset='pursuit',**COMMON)
    if arm=='D':return Controller(preset='pursuit',pursuit_frame='rear_slip',
                                  rear_slip_c_per_rad=11.,steer_inverse='ackermann',
                                  track_width_m=TRACK,**COMMON)
    return None


def read_case(case):
    source,route,seed,arm,p=case
    ctl=replay_controller(arm)
    rows=[]
    for line in open(p):
        f=json.loads(line)
        truth=f.get('truth'); executed=f.get('executed_control')
        if not truth or not executed:continue
        speed=float(truth['forward_speed_mps']); actual=float(truth['speed_mps'])
        steer=float(executed['steer'])
        k_nom,k_ack,k_unscaled=map_curvature(steer,speed)
        yaw_rate=math.radians(float(truth['angular_velocity'][2]))
        k_ach=yaw_rate/speed if speed>1. else float('nan')
        plan=curvature(f['waypoint']) if f.get('waypoint') else float('nan')
        cmd=float('nan'); replay_diff=float('nan'); reason=''
        if ctl is not None and f.get('rear_waypoint') and f.get('raw_control'):
            ctl.update(f['rear_waypoint'],f['sim_time'],trajectory_dt=.25)
            th,st,br=ctl.step(f['sim_time'],f['controller_speed_mps'],
                              -yaw_rate if arm=='D' else 0.)
            diag=ctl.diagnostics
            if br>0 and f['raw_signed_speed_mps']<.01:st=0.
            replay_diff=abs(st-f['raw_control'][arm]['steer'])
            if diag.get('aim_xy') is not None:
                if arm=='D':cmd=-float(diag['pursuit_curvature_inv_m'])
                else:
                    a=np.asarray(diag['aim_xy'],float)
                    cmd=-2.*a[1]/max(float(a@a),1e-8)
            reason=diag['reason']
        br=f.get('raw_control',{}).get('B',{}).get('steer',float('nan')) if f.get('raw_control') else float('nan')
        rows.append(dict(source=source,route=route,seed=seed,arm=arm,step=f['step'],t=f['sim_time'],
                         speed=speed,speed_3d=actual,steer=steer,b_shadow_steer=br,
                         plan_k_right=plan,k_cmd_right=cmd,k_ach_right=k_ach,
                         k_nom_right=k_nom,k_ack_right=k_ack,k_unscaled_right=k_unscaled,
                         replay_diff=replay_diff,reason=reason,
                         yaw_rate_right=yaw_rate,world_x=float(truth['location'][0]),
                         world_y=float(truth['location'][1]),yaw_deg=float(truth['rotation'][2])))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);FULL.mkdir(parents=True,exist_ok=True)
    all_rows=[];run_rows=[]
    for case in cases():
        rr=read_case(case)
        if rr:
            run_rows.append(dict(source=case[0],route=case[1],seed=case[2],arm=case[3],
                                 ticks=len(rr),max_replay_diff=max((r['replay_diff'] for r in rr if math.isfinite(r['replay_diff'])),default=float('nan'))))
        all_rows+=rr
    write_csv(FULL/'ticks.csv',all_rows)
    write_csv(OUT/'runs.csv',run_rows)
    turn=[r for r in all_rows if r['speed']>=1 and abs(r['plan_k_right'])>=.03 and
          abs(r['b_shadow_steer'])>=.05 and r['arm'] in ('B','C','D')]
    write_csv(OUT/'own_turn_ticks.csv',turn)
    print('runs',len(run_rows),'ticks',len(all_rows),'turn',len(turn))
    for arm in ('C','D'):
        e=[r['replay_diff'] for r in all_rows if r['arm']==arm and math.isfinite(r['replay_diff'])]
        print('replay',arm,len(e),max(e),np.percentile(e,[50,90,99]))


if __name__=='__main__':main()
