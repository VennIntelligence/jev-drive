#!/usr/bin/env python3
"""固定进度与Controller状态，只替换RouteAdapter来源位姿的只读理想定位诊断。"""
import argparse,copy,csv,hashlib,importlib.util,json,runpy,sys
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--windows',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise SystemExit('拒绝覆盖已有pose shadow。')
    helper=BASE/'replay_controller_updates.py';h=runpy.run_path(str(helper))
    sources=args.run_root/'provenance/source/scripts'
    controllerfile=sources/'b2d_controller.py';adapterfile=sources/'b2d_controller_adapter.py'
    ctrl=load('pose_shadow_controller',controllerfile);adapt=load('pose_shadow_adapter',adapterfile)
    hashes={}
    def record(p):hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path(__file__),helper,controllerfile,adapterfile,args.windows):record(p)
    windows={r:[] for r in h['ROUTES']}
    for w in csv.DictReader(args.windows.open()):
        if w['dataset']=='G2-v4' and w['route'] in windows:
            windows[w['route']].append(dict(segment=int(w['segment']),start=float(w['start_m']),end=float(w['end_m'])))
    verification=[];events=[];trajectory_rows=[]
    for route in h['ROUTES']:
        for variant in h['VARIANTS']:
            directory=args.run_root/route/variant/'pursuit'
            for name in ('control.jsonl','trajectories.jsonl','route_reference.json','agent_config.json'):record(directory/name)
            cfg=json.loads((directory/'agent_config.json').read_text());configpath=Path(cfg['controller_config']);record(configpath)
            parameters={k:v for k,v in json.loads(configpath.read_text()).items() if k not in h['ADAPTER_KEYS']}
            c=ctrl.Controller(preset=cfg['controller_preset'],**parameters)
            ref=json.loads((directory/'route_reference.json').read_text());xy=np.asarray(ref['world_xy'],float)
            settings=ref['adapter']
            base_adapter=adapt.RouteAdapter(xy,cruise_mps=cfg['cruise_mps'],
                end_extension_m=settings['route_end_extension_m'],stop_deceleration=settings['route_stop_deceleration'])
            plans={t['frame']:t for t in map(json.loads,(directory/'trajectories.jsonl').read_text().splitlines())}
            result=dict(route=route,variant=variant,updates=len(plans),adapter_nonexact_trajectories=0,
                adapter_max_abs_error=0.,controller_output_unequal_values=0,controller_diagnostic_unequal_values=0,
                missing_truth_updates=0,branch_motion_pose_max_difference=0.)
            for r in map(json.loads,(directory/'control.jsonl').read_text().splitlines()):
                t=plans.get(r['frame']);branches={};adapter_diagnostics={}
                if t is not None:
                    a=copy.deepcopy(base_adapter);a.progress=r['route_progress_m'];a.terminal_hold=r['route_terminal_hold']
                    actual=a.trajectory(t['pose_xy'],t['pose_yaw'])
                    error=float(np.max(abs(actual-np.asarray(t['trajectory_xy']))))
                    result['adapter_nonexact_trajectories']+=error!=0.;result['adapter_max_abs_error']=max(result['adapter_max_abs_error'],error)
                    truth_valid=(r.get('truth_frame')==r['frame'] and r.get('truth_xy') is not None and
                        r.get('truth_yaw') is not None and np.isfinite(np.r_[r['truth_xy'],r['truth_yaw']]).all())
                    if not truth_valid:result['missing_truth_updates']+=1
                    if error==0 and truth_valid:
                        for label,position,yaw in [('truth_position',r['truth_xy'],t['pose_yaw']),
                            ('truth_heading',t['pose_xy'],r['truth_yaw']),('truth_both',r['truth_xy'],r['truth_yaw'])]:
                            aa=copy.deepcopy(base_adapter);aa.progress=r['route_progress_m'];aa.terminal_hold=r['route_terminal_hold']
                            trajectory=aa.trajectory(position,yaw)
                            b=copy.deepcopy(c);accepted=b.update(trajectory,t['sim_time'])
                            output=b.step(r['sim_time'],r['speed_mps'],r['yaw_rate_rps'])
                            branches[label]=(b,output,accepted)
                            adapter_diagnostics[label]=aa.rejoin_diagnostics.copy()
                            trajectory_rows.append(dict(route=route,variant=variant,frame=r['frame'],mode=label,
                                progress_m=r['route_progress_m'],terminal_hold=r['route_terminal_hold'],source_position=position,
                                source_yaw=yaw,trajectory_xy=trajectory.tolist(),adapter=aa.rejoin_diagnostics))
                    c.update(t['trajectory_xy'],t['sim_time'])
                actual_output=c.step(r['sim_time'],r['speed_mps'],r['yaw_rate_rps']);actual_diag=c.diagnostics
                result['controller_output_unequal_values']+=sum(v!=r[k] for v,k in zip(actual_output,('throttle','steer','brake')))
                result['controller_diagnostic_unequal_values']+=sum(h['compare'](v,r.get(k,'__missing__'))[0] for k,v in actual_diag.items())
                if t is not None:
                    s=h['station'](np.asarray(r['truth_xy']),xy) if r.get('truth_xy') is not None else None
                    segment=next((w['segment'] for w in windows[route] if s is not None and w['start']<=s<=w['end']),None)
                    for label,(b,output,accepted) in branches.items():
                        dd=b.diagnostics;pose_delta=float(np.max(abs(b._pose-c._pose)))
                        result['branch_motion_pose_max_difference']=max(result['branch_motion_pose_max_difference'],pose_delta)
                        row=dict(route=route,variant=variant,frame=r['frame'],sim_time=r['sim_time'],window_segment=segment,
                            mode=label,station_m=s,fixed_progress_m=r['route_progress_m'],fixed_hold=r['route_terminal_hold'],
                            source_position_error_m=float(np.linalg.norm(np.asarray(t['pose_xy'])-r['truth_xy'])),
                            source_heading_error_deg=float(np.degrees(adapt.wrap(t['pose_yaw']-r['truth_yaw']))),
                            recorded_raw=actual_diag.get('raw_steer'),shadow_raw=dd.get('raw_steer'),raw_delta=None,
                            emitted_delta=output[1]-actual_output[1],aim_delta_m=None,recorded_reason=actual_diag.get('reason'),
                            shadow_reason=dd.get('reason'),accepted=accepted,
                            original_join_length_m=t['route_rejoin'].get('join_length_m'),
                            shadow_join_length_m=adapter_diagnostics[label].get('join_length_m'))
                        if h['finite'](row['recorded_raw']) and h['finite'](row['shadow_raw']):row['raw_delta']=row['shadow_raw']-row['recorded_raw']
                        if actual_diag.get('aim_xy') is not None and dd.get('aim_xy') is not None:
                            row['aim_delta_m']=float(np.linalg.norm(np.asarray(dd['aim_xy'])-actual_diag['aim_xy']))
                        events.append(row)
            verification.append(result)
    summaries=[]
    for route in h['ROUTES']:
        for variant in h['VARIANTS']:
            for w in windows[route]:
                for mode in ('truth_position','truth_heading','truth_both'):
                    rr=[r for r in events if (r['route'],r['variant'],r['window_segment'],r['mode'])==(route,variant,w['segment'],mode)]
                    summary=dict(route=route,variant=variant,segment=w['segment'],mode=mode,updates=len(rr),
                        join_length_changes=sum(r['original_join_length_m']!=r['shadow_join_length_m'] for r in rr))
                    for field in ('raw_delta','emitted_delta','aim_delta_m','source_position_error_m','source_heading_error_deg'):
                        v=np.asarray([r[field] for r in rr if h['finite'](r.get(field))],float)
                        summary.update({field+'_count':len(v),field+'_rms':float(np.sqrt(np.mean(v*v))) if len(v) else None,
                            field+'_p95_abs':float(np.percentile(abs(v),95)) if len(v) else None,
                            field+'_max_abs':float(np.max(abs(v))) if len(v) else None})
                    summaries.append(summary)
    args.out.mkdir(parents=True)
    h['csv_write'](args.out/'per-update-pose-shadow.csv',events);h['csv_write'](args.out/'window-summary.csv',summaries)
    with (args.out/'shadow-trajectories.jsonl').open('w') as f:
        for row in trajectory_rows:f.write(json.dumps(row)+'\n')
    (args.out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    snapshot=args.out/'source';snapshot.mkdir()
    for p in (Path(__file__),helper,controllerfile,adapterfile):(snapshot/p.name).write_bytes(p.read_bytes())
    manifest=dict(inputs=hashes,outputs={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.rglob('*') if p.is_file()},
        numpy=np.__version__,python=sys.version,
        scope='Exact original adapter trajectory reconstruction first. Fixed recorded progress and terminal_hold, unchanged settings. Separately replace source rear-axle position/heading/both with same-frame logger truth. Controller receives generated local trajectory only; same original motion/history pre-update state. Branch discarded each tick.',
        boundary='Ideal localization INPUT diagnostic only. Truth never enters actual control. Progress is NOT reprojected from truth. Internal join-length search may react to changed pose and is recorded. Does not isolate localization noise from fixed-progress inconsistency or prove closed-loop improvement.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(verification=verification,summaries=summaries)))
    if any(v['adapter_nonexact_trajectories'] or v['controller_output_unequal_values'] or v['controller_diagnostic_unequal_values'] or v['missing_truth_updates'] for v in verification):raise SystemExit(2)


if __name__=='__main__':main()
