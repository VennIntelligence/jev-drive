#!/usr/bin/env python3
"""冻结控制器的逐帧重放与单次轨迹更新反事实；不启动仿真、不改变输入记录。"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROUTES=('26966','24240','17563')
VARIANTS=('baseline-max','short-max')
ADAPTER_KEYS=('adapter','rear_axle_offset_m','gnss_x_m','pose_gnss_gain','pose_heading_gain',
              'route_end_extension_m','truth_logging','route_stop_deceleration','metadata','preset')


def finite(value):
    return isinstance(value,(int,float,np.number)) and np.isfinite(value)


def compare(left,right):
    """返回逐值不等数和数值最大绝对差；None/缺字段不转成0。"""
    if isinstance(left,list) and isinstance(right,list):
        if len(left)!=len(right):return 1,None
        pairs=[compare(a,b) for a,b in zip(left,right)]
        return sum(x[0] for x in pairs),max([x[1] for x in pairs if x[1] is not None] or [0.])
    if finite(left) and finite(right):return int(left!=right),float(abs(left-right))
    return int(left!=right),None


def station(point,xy):
    d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=1)
    good=length>1e-8;starts=xy[:-1][good];d=d[good];length=length[good]
    s=np.r_[0.,np.cumsum(length)]
    u=np.clip(np.sum((point-starts)*d,axis=1)/(length*length),0,1)
    feet=starts+u[:,None]*d;i=int(np.argmin(np.sum((feet-point)**2,axis=1)))
    return float(s[i]+u[i]*length[i])


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path)
    ap.add_argument('--windows',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('拒绝覆盖已有诊断。')
    source=args.source or args.run_root/'provenance/source/scripts/b2d_controller.py'
    spec=importlib.util.spec_from_file_location('frozen_replay_controller',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources={}
    def record(p):
        p=Path(p);sources[str(p.resolve())]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    for p in (source,Path(__file__),args.windows):record(p)
    windows={r:[] for r in ROUTES}
    for row in csv.DictReader(args.windows.open()):
        if row['dataset']=='G2-v4' and row['route'] in windows:
            windows[row['route']].append(dict(segment=int(row['segment']),start=float(row['start_m']),end=float(row['end_m'])))
    if [len(windows[r]) for r in ROUTES]!=[1,1,2]:raise SystemExit('需保留原来的四个几何窗口。')
    args.out.mkdir(parents=True)
    full=(args.out/'full-replay.jsonl').open('w')
    verification=[];events=[]
    for route in ROUTES:
        for variant in VARIANTS:
            directory=args.run_root/route/variant/'pursuit'
            for name in ('control.jsonl','trajectories.jsonl','agent_config.json','route_reference.json'):record(directory/name)
            cfg=json.loads((directory/'agent_config.json').read_text());configpath=Path(cfg['controller_config']);record(configpath)
            config=json.loads(configpath.read_text())
            parameters={k:v for k,v in config.items() if k not in ADAPTER_KEYS}
            controller=module.Controller(preset=cfg['controller_preset'],**parameters)
            motion=[json.loads(x) for x in (directory/'control.jsonl').read_text().splitlines() if x]
            plans=[json.loads(x) for x in (directory/'trajectories.jsonl').read_text().splitlines() if x]
            trajectory={r['frame']:r for r in plans}
            if len(trajectory)!=len(plans):raise SystemExit('重复轨迹frame，需显式重放顺序，不能猜测。')
            xy=np.asarray(json.loads((directory/'route_reference.json').read_text())['world_xy'],float)
            result=dict(route=route,variant=variant,frames=len(motion),trajectory_updates=len(plans),
                output_unequal_values=0,diagnostic_unequal_values=0,acceptance_mismatches=0,
                maximum_output_abs_error=0.,maximum_diagnostic_abs_error=0.,field_mismatches={},
                all_frames_consecutive=all(b['frame']==a['frame']+1 for a,b in zip(motion,motion[1:])))
            previous=None
            for row in motion:
                frame=row['frame'];plan=trajectory.get(frame)
                prior_diag=controller.diagnostics
                old=copy.deepcopy(controller) if plan is not None else None
                accepted=None
                if plan is not None:
                    accepted=controller.update(plan['trajectory_xy'],plan['sim_time'])
                    result['acceptance_mismatches']+=accepted!=plan['accepted']
                out=controller.step(row['sim_time'],row['speed_mps'],row['yaw_rate_rps'])
                diag=controller.diagnostics
                for key,value in zip(('throttle','steer','brake'),out):
                    bad,error=compare(value,row[key]);result['output_unequal_values']+=bad
                    result['maximum_output_abs_error']=max(result['maximum_output_abs_error'],error or 0.)
                for key,value in diag.items():
                    bad,error=compare(value,row.get(key,'__missing__'));result['diagnostic_unequal_values']+=bad
                    result['maximum_diagnostic_abs_error']=max(result['maximum_diagnostic_abs_error'],error or 0.)
                    if bad:result['field_mismatches'][key]=result['field_mismatches'].get(key,0)+bad
                full.write(json.dumps(dict(route=route,variant=variant,frame=frame,sim_time=row['sim_time'],
                    speed_mps=row['speed_mps'],yaw_rate_rps=row['yaw_rate_rps'],trajectory_update=plan is not None,
                    replay_output=out,recorded_output=[row[k] for k in ('throttle','steer','brake')],diagnostics=diag))+'\n')
                if old is not None:
                    old_out=old.step(row['sim_time'],row['speed_mps'],row['yaw_rate_rps']);old_diag=old.diagnostics
                    s=station(np.asarray(row['truth_xy']),xy) if row.get('truth_xy') is not None else None
                    segment=next((w['segment'] for w in windows[route] if s is not None and w['start']<=s<=w['end']),None)
                    event=dict(route=route,variant=variant,frame=frame,sim_time=row['sim_time'],station_m=s,
                        window_segment=segment,accepted=accepted,previous_reason=prior_diag.get('reason'),
                        new_reason=diag.get('reason'),old_continued_reason=old_diag.get('reason'),
                        new_source_time=diag.get('trajectory_time'),old_source_time=old_diag.get('trajectory_time'),
                        new_raw=diag.get('raw_steer'),old_continued_raw=old_diag.get('raw_steer'),previous_raw=prior_diag.get('raw_steer'),
                        new_emitted=out[1],old_continued_emitted=old_out[1],previous_emitted=previous['steer'] if previous else None,
                        motion_pose_branch_max_difference=float(np.max(abs(controller._pose-old._pose))),
                        new_aim_x=None,new_aim_y=None,old_aim_x=None,old_aim_y=None,aim_update_distance_m=None)
                    for label,dd in [('new',diag),('old',old_diag)]:
                        aim=dd.get('aim_xy')
                        if aim is not None:event[label+'_aim_x'],event[label+'_aim_y']=aim
                    if diag.get('aim_xy') is not None and old_diag.get('aim_xy') is not None:
                        event['aim_update_distance_m']=float(np.linalg.norm(np.asarray(diag['aim_xy'])-old_diag['aim_xy']))
                    for kind in ('raw','emitted'):
                        a,b,p=event['new_'+kind],event['old_continued_'+kind],event['previous_'+kind]
                        if all(finite(x) for x in (a,b,p)) and previous is not None:
                            total=a-p;motion_delta=b-p;update_delta=a-b;dt=row['sim_time']-previous['sim_time']
                            event.update({kind+'_total_delta':total,kind+'_motion_delta':motion_delta,kind+'_update_delta':update_delta,
                                kind+'_decomposition_residual':total-motion_delta-update_delta,
                                kind+'_update_delta_per_s':update_delta/dt})
                    events.append(event)
                previous=row
            result['exact_replay']=result['output_unequal_values']==0 and result['diagnostic_unequal_values']==0 and result['acceptance_mismatches']==0
            verification.append(result)
    full.close()
    csv_write(args.out/'per-update-counterfactual.csv',events)
    all_exact=all(v['exact_replay'] for v in verification)
    summaries=[];peaks=[]
    for route in ROUTES:
        for variant in VARIANTS:
            for window in windows[route]:
                rows=[e for e in events if (e['route'],e['variant'],e['window_segment'])==(route,variant,window['segment'])]
                eligible=[e for e in rows if 'raw_total_delta' in e]
                summary=dict(route=route,variant=variant,segment=window['segment'],updates=len(rows),
                    decomposable_updates=len(eligible),exact_replay_prerequisite=all_exact)
                for field in ('raw_total_delta','raw_motion_delta','raw_update_delta','emitted_total_delta','emitted_motion_delta',
                              'emitted_update_delta','aim_update_distance_m'):
                    values=np.asarray([e[field] for e in eligible if finite(e.get(field))],float)
                    summary[field+'_rms']=float(np.sqrt(np.mean(values**2))) if len(values) else None
                    summary[field+'_p95_abs']=float(np.percentile(abs(values),95)) if len(values) else None
                    summary[field+'_max_abs']=float(max(abs(values))) if len(values) else None
                summary['update_larger_than_motion_count']=sum(abs(e['raw_update_delta'])>abs(e['raw_motion_delta']) for e in eligible)
                summary['update_opposes_motion_count']=sum(e['raw_update_delta']*e['raw_motion_delta']<0 for e in eligible)
                summaries.append(summary)
                peaks.extend(sorted(eligible,key=lambda e:abs(e['raw_update_delta']),reverse=True)[:5])
    csv_write(args.out/'window-summary.csv',summaries);csv_write(args.out/'top5-update-effects.csv',peaks)
    (args.out/'replay-verification.json').write_text(json.dumps(dict(all_exact=all_exact,cases=verification,
        max_motion_pose_branch_difference=max(e['motion_pose_branch_max_difference'] for e in events),
        max_raw_decomposition_residual=max(abs(e.get('raw_decomposition_residual',0)) for e in events)),indent=2)+'\n')
    snapshot=args.out/'source';snapshot.mkdir()
    (snapshot/'b2d_controller.py').write_bytes(source.read_bytes())
    (snapshot/'replay_controller_updates.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(inputs=sources,outputs={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.rglob('*') if p.is_file()},
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        counterfactual='At every update copy identical pre-tick controller state; actual branch accepts recorded trajectory then steps recorded current motion; alternate skips ONLY this update and steps same motion. Alternate discarded immediately.',
        identity='new_raw-previous_raw=(new_raw-old_continued_raw)+(old_continued_raw-previous_raw). First/no-reference/safe branches retain missing raw; never substitute emitted command as raw demand.',
        truth_boundary='Recorded truth_xy only assigns already-frozen station/window; controller sees only original trajectory/time/speed/yaw_rate.',
        interpretation='Immediate controller output causal intervention on trajectory acceptance, conditional on recorded state. Not closed-loop vehicle outcome, adapter diagnosis, or future continuation. Motion term includes time/reference-age advancement and speed-dependent lookahead.',
        invalid_replay_policy='If replay is not exact, summaries are diagnostic-only and exact_replay_prerequisite is false; no exact-replay claim permitted.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(all_exact=all_exact,frames=sum(v['frames'] for v in verification),updates=len(events),windows=len(summaries),out=str(args.out))))
    if not all_exact:raise SystemExit(2)


if __name__=='__main__':main()
