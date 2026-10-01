#!/usr/bin/env python3
"""Closed TCP case segmentation: fixed 5s bins, event prefix, all >=5s low-speed runs."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from analyze_tcp import frame_metrics,summary,finite,vector,stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--route',default='1773');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('New output edition required')
    data=json.loads((a.analysis/'comparison.json').read_text());a.out.mkdir(parents=True);sources={str((a.analysis/'comparison.json').resolve()):sha(a.analysis/'comparison.json')};allframes=[];segments=[]
    for key,case in data['cases'].items():
        if case['route_id']!=a.route:continue
        source=Path(case['attempt_path'])/'tcp-control.jsonl';assert sha(source)==case['source_sha256'][str(source.resolve())];sources[str(source.resolve())]=sha(source)
        raw=[json.loads(l) for l in source.read_text().splitlines()];rows=frame_metrics(raw);first=case['first_collision_frame'];t0=rows[0]['timestamp'];enriched=[]
        for original,row in zip(raw,rows):
            meta=(original.get('prediction') or {}).get('metadata') or {};angle,target,last,final=[meta.get(k) for k in ('angle','angle_target','angle_last','angle_final')];targetxy=meta.get('target');valid=all(finite(x) for x in (angle,target,last,final)) and vector(targetxy,2)
            use_target=(abs(target)<abs(angle) or (abs(target-last)>.3 and targetxy[1]<10)) if valid else None
            r=dict(row,case=key,time_from_first_s=row['timestamp']-t0,phase='precontact' if first is None or row['frame']<first else 'contact_and_after',angle_prediction_deg=angle*90 if finite(angle) else None,angle_target_deg=target*90 if finite(target) else None,angle_final_deg=final*90 if finite(final) else None,
                   target_rule=use_target,target_rule_final_diff=final-(target if use_target else angle) if valid else None,arbitration_attenuation_deg=(abs(angle)-abs(final))*90 if valid else None,
                   model_stop_intent=finite(row['desired_speed_mps']) and row['desired_speed_mps']<.4,command_drive=finite(row['throttle']) and row['throttle']>0 and row['brake']==0,
                   low_speed=finite(row['speed_truth_mps']) and abs(row['speed_truth_mps'])<.5,nearest_actor_center_m=None,nearest_actor_id=None,nearest_actor_type=None)
            truth=original.get('truth') or {};nearby=truth.get('nearby_actors') or [];xyz=truth.get('xyz')
            if vector(xyz) and nearby:
                distances=[(math.sqrt(sum((x-y)**2 for x,y in zip(xyz,o['xyz']))),o) for o in nearby if vector(o.get('xyz'))]
                if distances:
                    distance,actor=min(distances,key=lambda x:x[0]);r.update(nearest_actor_center_m=distance,nearest_actor_id=actor['id'],nearest_actor_type=actor['type'])
            enriched.append(r)
        groups=[('full',enriched),('precontact',[r for r in enriched if r['phase']=='precontact']),('contact_and_after',[r for r in enriched if r['phase']=='contact_and_after'])]
        for start in range(0,int(enriched[-1]['time_from_first_s'])+1,5):groups.append(('fixed_5s_%03d'%start,[r for r in enriched if start<=r['time_from_first_s']<start+5]))
        for i,run in enumerate(case['low_speed_runs_ge_5s']):groups.append(('low_speed_run_%02d'%i,[r for r in enriched if run['start_frame']<=r['frame']<=run['end_frame']]))
        for label,rs in groups:
            if not rs:continue
            metrics=summary(rs);segments.append(dict(case=key,segment=label,start_frame=rs[0]['frame'],end_frame=rs[-1]['frame'],start_s=rs[0]['time_from_first_s'],end_s=rs[-1]['time_from_first_s'],span_s=rs[-1]['timestamp']-rs[0]['timestamp'],rows=len(rs),speed_error_rms=metrics['speed_error_mps']['rms'],jerk_lon_abs_p95=metrics['jerk_lon_mps3']['abs_p95'],speed_mean=metrics['speed_truth_mps']['mean'],desired_mean=metrics['desired_speed_mps']['mean'],model_stop_ticks=sum(r['model_stop_intent'] for r in rs),drive_ticks=sum(r['command_drive'] for r in rs),low_speed_ticks=sum(r['low_speed'] for r in rs),low_speed_drive_ticks=sum(r['low_speed'] and r['command_drive'] for r in rs),target_rule_ticks=sum(r['target_rule'] is True for r in rs),arbitration_available_ticks=sum(r['target_rule'] is not None for r in rs),arbitration_attenuation_mean_deg=stats(r['arbitration_attenuation_deg'] for r in rs)['mean'],target_rule_mismatch_ticks=sum(finite(r['target_rule_final_diff']) and abs(r['target_rule_final_diff'])>1e-6 for r in rs),reverse_guard_ticks=sum(r['reason']=='reverse_motion' for r in rs),selected_vs_applied_previous_max=metrics['applied_pedals_previous_command_max_diff']['max_abs'],steer_saturated_ticks=metrics['steer_saturated_ticks']))
        allframes.extend(enriched)
    for name,rs in [('frames.csv',allframes),('segments.csv',segments)]:
        if rs:
            with (a.out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    for source in [Path(__file__),Path(__file__).with_name('analyze_tcp.py'),Path('/data/third_party/Bench2DriveZoo/TCP/model.py'),Path('/data/third_party/Bench2DriveZoo/TCP/config.py')]:
        sources[str(source.resolve())]=sha(source);(a.out/('vendor-'+source.name if 'Bench2DriveZoo' in str(source) else source.name)).write_bytes(source.read_bytes())
    (a.out/'README.md').write_text('''# TCP停滞与原生目标点仲裁审计

全程、严格首次collision前、collision及之后、从首受控tick起每5s固定箱，以及全部实际跨度>=5s低速(|truth longitudinal speed|<.5m/s)连续区间。区间仅诊断，不替代完整路线主指标；边界规则不依结果择优。表保留各箱的实际跨度与全部低速样本，停滞可稀释全程jerk，必须连同prefix看。

model_stop_intent为desired<.4m/s；command_drive为selected throttle>0且brake0。两者与实际低速分别计数。原生target仲裁按已归档vendor source重算：|angle_target|<|angle|，或|angle_target-angle_last|>.3且target forward<10m；归一化angle乘90转degree，与metadata.angle_final核对。attenuation正数表示绝对角减小，负数表示增大；不是控制效果因果证据。

附近actor仅5Hz中心距离，不能据此判bbox碰撞/责任。模型物理原点未确认；局部waypoint/aim/target不能伪装精确监督真值。applied_control对上一条命令仅检验API控制值交付，不证明物理驱动力或可通行空间。
''')
    (a.out/'manifest.json').write_text(json.dumps(dict(sources=sources,outputs={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}),indent=2))
    print(json.dumps(dict(cases=len(set(r['case'] for r in allframes)),rows=len(allframes),segments=len(segments),out=str(a.out))))
if __name__=='__main__':main()
