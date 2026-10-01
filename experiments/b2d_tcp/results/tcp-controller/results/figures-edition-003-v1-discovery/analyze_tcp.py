#!/usr/bin/env python3
"""Offline native_common/pi_common telemetry analysis; stdlib only, exclusive editions.

--case ROUTE:ARM=/absolute/attempt may be repeated. Never runs/imports CARLA/TCP.
World acceleration is differentiated before body projection for primary jerk.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS=('native_common','pi_common')
ROUTES=('24211','1711','1773')
REQUIRED_SENSORS=('GPS','IMU','SPEED','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')
CONTROL_TOLERANCE=1e-7

def finite(x):return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
def vector(x,n=3):return isinstance(x,list) and len(x)==n and all(finite(v) for v in x)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def stats(values):
    a=sorted(float(x) for x in values if finite(x))
    if not a:return dict(n=0,mean=None,rms=None,p05=None,p95=None,abs_p95=None,max_abs=None)
    def percentile(v,q):
        pos=(len(v)-1)*q;i=int(pos);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
    return dict(n=len(a),mean=sum(a)/len(a),rms=math.sqrt(sum(x*x for x in a)/len(a)),p05=percentile(a,.05),p95=percentile(a,.95),abs_p95=percentile(sorted(map(abs,a)),.95),max_abs=max(map(abs,a)))
def basis(rotation):
    # CARLA0.9.15 Math::GetForwardVector/GetRightVector, input roll/pitch/yaw degrees.
    roll,pitch,yaw=map(math.radians,rotation);cr,sr=math.cos(roll),math.sin(roll);cp,sp=math.cos(pitch),math.sin(pitch);cy,sy=math.cos(yaw),math.sin(yaw)
    return [cy*cp,sy*cp,sp],[cy*sp*sr-sy*cr,sy*sp*sr+cy*cr,-cp*sr]
def recompute_desired(points):
    if not isinstance(points,list) or len(points)!=4 or not all(vector(p,2) for p in points):return None
    return sum(math.hypot(b[0]-a[0],b[1]-a[1])/.5 for a,b in zip(points,points[1:]))/3

def frame_metrics(rows):
    output=[];prev=None
    for raw in rows:
        frame,t=raw.get('frame'),raw.get('timestamp');pred=raw.get('prediction') or {};meta=pred.get('metadata') or {};truth=raw.get('truth') or {};control=raw.get('selected_control');desired=meta.get('desired_speed');independent=recompute_desired(pred.get('raw_waypoints'))
        row=dict(frame=frame,timestamp=t,arm=raw.get('arm'),desired_speed_mps=desired if finite(desired) else None,desired_recomputed_mps=independent,
                 desired_recompute_error_mps=desired-independent if finite(desired) and finite(independent) else None,
                 input_speed_mps=raw.get('raw_speed_mps'),forward_count=raw.get('forward_count'),native_pid_calls=pred.get('native_pid_calls',raw.get('native_pid_calls')),
                 reason=((raw.get('comparison') or {}).get('diagnostics') or {}).get('reason','initialization_or_missing'),truth_aligned=False,
                 speed_truth_mps=None,speed_3d_mps=None,speed_error_mps=None,input_speed_error_mps=None,accel_lon_mps2=None,accel_lat_mps2=None,
                 jerk_lon_mps3=None,jerk_lat_mps3=None,jerk_world_norm_mps3=None,body_accel_lon_derivative_mps3=None,target_change_rate_mps2=None,
                 throttle=None,steer=None,brake=None,steer_rate_per_s=None,steer_saturated=None,gear=truth.get('gear'),applied_pedals_previous_command_max_diff=None,
                 native_steer_difference=None,selected_branch_max_diff=None,nominal_interval=None,
                 prediction_present=bool(pred),prediction_fault=pred.get('fault'),raw_speed_finite=finite(raw.get('raw_speed_mps')),
                 raw_waypoints_finite=independent is not None if pred else None,raw_target_finite=vector(pred.get('raw_target'),2) if pred else None,
                 raw_imu_finite=vector(raw.get('raw_imu'),7),sensor_frames_aligned=all((raw.get('sensor_frames') or {}).get(k)==frame for k in REQUIRED_SENSORS),
                 model_rgb_hash_present=isinstance((raw.get('model_input') or {}).get('rgb_sha256'),str),
                 official_tail_throttle_difference=None,official_tail_steer_difference=None,official_tail_brake_difference=None)
        if finite(raw.get('raw_speed_mps')) and finite(desired):row['input_speed_error_mps']=raw['raw_speed_mps']-desired
        if vector(control):
            row.update(throttle=control[0],steer=control[1],brake=control[2],steer_saturated=abs(control[1])>=1-1e-6)
            branch=(raw.get('comparison') or {}).get(str(raw.get('arm'))+'_control')
            if vector(branch):row['selected_branch_max_diff']=max(abs(x-y) for x,y in zip(control,branch))
            official=raw.get('official_tail_control')
            if vector(official):
                for i,name in enumerate(('throttle','steer','brake')):row['official_tail_'+name+'_difference']=control[i]-official[i]
            native=pred.get('native_control')
            if vector(native):row['native_steer_difference']=control[1]-native[1]
        valid=(truth.get('frame')==frame and not truth.get('error') and vector(truth.get('rotation_deg')) and vector(truth.get('velocity')) and vector(truth.get('acceleration')))
        fwd=right=None
        if valid:
            fwd,right=basis(truth['rotation_deg']);v=dot(truth['velocity'],fwd)
            row.update(truth_aligned=True,speed_truth_mps=v,speed_3d_mps=norm(truth['velocity']),speed_error_mps=v-desired if finite(desired) else None,
                       accel_lon_mps2=dot(truth['acceleration'],fwd),accel_lat_mps2=dot(truth['acceleration'],right))
        if prev and finite(t) and finite(prev[0].get('timestamp')):
            old,oldrow=prev;dt=t-old['timestamp'];cont=(finite(frame) and finite(old.get('frame')) and frame==old['frame']+1 and dt>0 and abs(dt-.05)<=1e-4)
            row['nominal_interval']=cont
            if cont:
                if finite(row['steer']) and finite(oldrow['steer']):row['steer_rate_per_s']=(row['steer']-oldrow['steer'])/dt
                if finite(desired) and finite(oldrow['desired_speed_mps']):row['target_change_rate_mps2']=(desired-oldrow['desired_speed_mps'])/dt
                if valid and oldrow['truth_aligned']:
                    jerk=[(a-b)/dt for a,b in zip(truth['acceleration'],old['truth']['acceleration'])]
                    row.update(jerk_lon_mps3=dot(jerk,fwd),jerk_lat_mps3=dot(jerk,right),jerk_world_norm_mps3=norm(jerk),
                               body_accel_lon_derivative_mps3=(row['accel_lon_mps2']-oldrow['accel_lon_mps2'])/dt)
                applied=truth.get('applied_control');oldctrl=old.get('selected_control')
                if vector(applied) and vector(oldctrl):row['applied_pedals_previous_command_max_diff']=max(abs(a-b) for a,b in zip(applied,oldctrl))
        output.append(row);prev=(raw,row)
    return output

FIELDS=('speed_error_mps','input_speed_error_mps','speed_truth_mps','desired_speed_mps','desired_recompute_error_mps','target_change_rate_mps2','accel_lon_mps2','accel_lat_mps2','jerk_lon_mps3','jerk_lat_mps3','jerk_world_norm_mps3','body_accel_lon_derivative_mps3','steer_rate_per_s','throttle','brake','native_steer_difference','selected_branch_max_diff','applied_pedals_previous_command_max_diff','official_tail_throttle_difference','official_tail_steer_difference','official_tail_brake_difference')
def summary(rows):
    metrics={k:stats(r[k] for r in rows) for k in FIELDS}
    metrics.update(rows=len(rows),speed_error_coverage=sum(finite(r['speed_error_mps']) for r in rows)/len(rows) if rows else None,
                   simultaneous_pedals=sum((r['throttle'] or 0)>0 and (r['brake'] or 0)>0 for r in rows),
                   steer_saturated_ticks=sum(r['steer_saturated'] is True for r in rows),
                   throttle_saturated_ticks=sum(finite(r['throttle']) and r['throttle']>=.75-1e-6 for r in rows),
                   brake_positive_ticks=sum(finite(r['brake']) and r['brake']>0 for r in rows),
                   gear_transitions=sum(a['gear'] is not None and b['gear'] is not None and a['gear']!=b['gear'] and b['nominal_interval'] is True for a,b in zip(rows,rows[1:])))
    return metrics

def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None

def analyze(path,route,arm):
    source=path/'tcp-control.jsonl';raw=[];errors=[]
    before={str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in path.iterdir() if p.is_file()}
    if source.exists():
        for i,line in enumerate(source.read_text().splitlines(),1):
            try:
                value=json.loads(line)
                if not isinstance(value,dict):raise ValueError('not an object')
                raw.append(value)
            except ValueError as e:errors.append(dict(line=i,error=str(e)))
    else:errors.append(dict(error='missing telemetry'))
    rows=frame_metrics(raw);events=read_json(path/'criterion_events.json');record=read_json(path/'results.json');records=(record or {}).get('_checkpoint',{}).get('records',[]);official=records[0] if len(records)==1 and isinstance(records[0],dict) else None
    attempt=read_json(path/'attempt.json') or {};event_list=(events or {}).get('events',[])
    collisions=[e for e in event_list if str(e.get('type','')).startswith('COLLISION') and finite(e.get('frame'))]
    first=min((e['frame'] for e in collisions),default=None);event_ready=isinstance(events,dict) and events.get('state')=='ok'
    official_collisions=sum(len(v) for k,v in ((official or {}).get('infractions') or {}).items() if k.startswith('collisions_') and isinstance(v,list))
    if official_collisions and first is None:event_ready=False
    prefix=[r for r in rows if first is None or r['frame']<first] if event_ready else []
    terminal=bool(official and official.get('status') not in [None,'Started'])
    low=[]
    for r in rows:
        if finite(r['speed_truth_mps']) and abs(r['speed_truth_mps'])<.5:
            if low and r['nominal_interval'] is True and low[-1][-1]['frame']==r['frame']-1:low[-1].append(r)
            else:low.append([r])
    low=[dict(start_frame=x[0]['frame'],end_frame=x[-1]['frame'],duration_s=x[-1]['timestamp']-x[0]['timestamp'],sample_bin_occupancy_s=x[-1]['timestamp']-x[0]['timestamp']+.05,
              desired_min=min((r['desired_speed_mps'] for r in x if finite(r['desired_speed_mps'])),default=None),
              missing_desired=sum(not finite(r['desired_speed_mps']) for r in x)) for x in low if finite(x[0]['timestamp']) and finite(x[-1]['timestamp']) and x[-1]['timestamp']-x[0]['timestamp']>=5-1e-6]
    result=dict(route_id=route,arm=arm,attempt_path=str(path.resolve()),official=official,attempt=attempt,terminal_official_record=terminal,
                full=summary(rows),before_collision=summary(prefix) if event_ready else None,first_collision_frame=first,events_state=(events or {}).get('state','missing'),
                moving_diagnostic=summary([r for r in rows if finite(r['speed_truth_mps']) and abs(r['speed_truth_mps'])>.5]),
                predicted_stop_diagnostic=summary([r for r in rows if finite(r['desired_speed_mps']) and r['desired_speed_mps']<.4]),
                low_speed_runs_ge_5s=low,quality=dict(parse_errors=errors,truth_unavailable_ticks=sum(not r['truth_aligned'] for r in rows),
                nonnominal_intervals=sum(r['nominal_interval'] is False for r in rows),unexpected_arm_ticks=sum(r['arm']!=arm for r in rows),
                native_pid_calls_recorded_ticks=sum(r['native_pid_calls'] is not None for r in rows),
                prediction_missing_ticks=sum(not r['prediction_present'] for r in rows),
                initialization_neutral_ticks=sum(not r['prediction_present'] and r['forward_count']==0 and all(r.get(k)==0 for k in ('throttle','steer','brake')) for r in rows),
                prediction_fault_ticks=sum(bool(r['prediction_fault']) for r in rows),
                reason_counts={reason:sum(r['reason']==reason for r in rows) for reason in sorted(set(r['reason'] for r in rows))},
                native_pid_call_counts={str(value):sum(r['native_pid_calls']==value for r in rows) for value in (None,0,1)},
                forward_call_counts={str(value):sum(r['forward_count']==value for r in rows) for value in (None,0,1)},
                raw_speed_nonfinite_ticks=sum(not r['raw_speed_finite'] for r in rows),
                raw_imu_missing_or_nonfinite_ticks=sum(not r['raw_imu_finite'] for r in rows),
                raw_waypoints_invalid_ticks=sum(r['raw_waypoints_finite'] is False for r in rows),
                raw_target_invalid_ticks=sum(r['raw_target_finite'] is False for r in rows),
                negative_raw_speed_ticks=sum(finite(r['input_speed_mps']) and r['input_speed_mps']<0 for r in rows),
                tiny_negative_raw_speed_ticks=sum(finite(r['input_speed_mps']) and -.01<r['input_speed_mps']<0 for r in rows),
                required_sensor_alignment_ticks=sum(r['sensor_frames_aligned'] for r in rows),
                required_sensor_frame_coverage={k:dict(missing=sum(k not in (r.get('sensor_frames') or {}) for r in raw),mismatch=sum(k in (r.get('sensor_frames') or {}) and r['sensor_frames'][k]!=r.get('frame') for r in raw)) for k in REQUIRED_SENSORS},
                model_rgb_hash_missing_ticks=sum(not r['model_rgb_hash_present'] for r in rows),
                prediction_frame_provenance='Same run_step row; sensor frames and exactly-one forward/captured PID are logged. No independent network source-frame field exists; optional asynchronous bev excluded.',
                control_float_tolerance=CONTROL_TOLERANCE,
                native_steer_over_tolerance_ticks=sum(finite(r['native_steer_difference']) and abs(r['native_steer_difference'])>CONTROL_TOLERANCE for r in rows),
                selected_branch_over_tolerance_ticks=sum(finite(r['selected_branch_max_diff']) and r['selected_branch_max_diff']>CONTROL_TOLERANCE for r in rows),
                official_tail_difference_note='Selected minus official-tail throttle/brake can differ intentionally due to shared envelope. Only small steer/selected-branch differences within1e-7 are treated as numeric storage tolerance; raw differences retained.',
                source_changed_during_read=[name for name,state in before.items() if not Path(name).exists() or (Path(name).stat().st_size,Path(name).stat().st_mtime_ns)!=state]),
                source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file() and p.name in ['tcp-control.jsonl','tcp-setup.json','tcp-route-reference.json','criterion_events.json','results.json','attempt.json']})
    return result,rows


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--case',action='append',required=True);parser.add_argument('--out',required=True);a=parser.parse_args();out=Path(a.out)
    if out.exists():parser.error('Output exists; use a new edition')
    parsed=[]
    for value in a.case:
        identity,path=value.split('=',1);route,arm=identity.split(':',1)
        if route not in ROUTES or arm not in ARMS:parser.error('Unexpected route/arm')
        if identity in [r[0] for r in parsed]:parser.error('Pass one explicitly selected attempt per route/arm; preserve retries separately')
        lifecycle=read_json(Path(path)/'attempt.json') or {}
        if lifecycle.get('status') in (None,'running','started','Started'):
            parser.error('Only closed attempts with final attempt.json status may be analyzed: '+path)
        parsed.append((identity,route,arm,Path(path)))
    out.mkdir(parents=True);cases={}
    for identity,route,arm,path in parsed:
        result,rows=analyze(path,route,arm);cases[identity]=result
        (out/(route+'-'+arm+'.json')).write_text(json.dumps(result,indent=2))
        if rows:
            with (out/(route+'-'+arm+'-frames.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pairs=[]
    for route in ROUTES:
        b,c=[cases.get(route+':'+arm) for arm in ARMS]
        ready=bool(b and c and b['terminal_official_record'] and c['terminal_official_record'])
        quality=lambda x: bool(x['full']['rows']) and not x['quality']['parse_errors'] and not x['quality']['truth_unavailable_ticks'] and not x['quality']['nonnominal_intervals'] and not x['quality']['unexpected_arm_ticks'] and not x['quality']['source_changed_during_read'] and x['full']['speed_error_mps']['n']>0 and x['full']['jerk_lon_mps3']['n']>0
        pair=dict(route_id=route,pair_terminal=ready,behavior_data_available=ready and quality(b) and quality(c),deltas_c_minus_b={})
        if ready:
            for key,stat in [('speed_error_mps','rms'),('accel_lon_mps2','abs_p95'),('jerk_lon_mps3','abs_p95')]:
                x,y=b['full'][key][stat],c['full'][key][stat];pair['deltas_c_minus_b'][key+'_'+stat]=y-x if finite(x) and finite(y) else None
        pairs.append(pair)
    result=dict(cases=cases,paired=pairs,all_six_terminal=len(cases)==6 and all(p['pair_terminal'] for p in pairs),new_default_qualified=False,
                all_six_behavior_data_available=len(cases)==6 and all(p['behavior_data_available'] for p in pairs),
                main_metric='Full route; no warmup removal. Prefix/moving/stop are supplemental.',jerk='diff world acceleration / actual dt, then project current body axes. Noncontiguous/non20Hz gaps excluded and counted.',
                limitations=['No identical-world assumption between arms. Model target changes and execution response both contribute.','Missing data are not zero; official terminal records alone do not establish valid behavior comparison.','This helper does not choose a favorable retry or infer collision responsibility.','No model-origin CTE is synthesized; prediction physical origin remains unconfirmed.'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2));(out/'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},inputs={name:digest for case in cases.values() for name,digest in case['source_sha256'].items()},outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(dict(cases=len(cases),all_six_terminal=result['all_six_terminal'],out=str(out))))

if __name__=='__main__':main()
