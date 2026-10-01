#!/usr/bin/env python3
"""同历史state/path的Hermite aim理论shadow，不改变车辆、控制器状态或纵向输出。"""
import argparse,ast,copy,hashlib,importlib.util,json,math,runpy,sys
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--windows',type=Path,required=True)
    ap.add_argument('--prototype',type=Path,default=BASE/'circle_sampling.py')
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('拒绝覆盖已有shadow。')
    import csv
    helpers=runpy.run_path(str(BASE/'replay_controller_updates.py'))
    source=args.run_root/'provenance/source/scripts/b2d_controller.py'
    spec=importlib.util.spec_from_file_location('shadow_frozen_controller',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    prototype_text=args.prototype.read_text()
    node=next(n for n in ast.parse(prototype_text).body if isinstance(n,ast.FunctionDef) and n.name=='hermite_aim')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.prototype),'exec'),namespace)
    hermite_aim=namespace['hermite_aim']
    inputs={}
    def record(p):inputs[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path(__file__),BASE/'replay_controller_updates.py',source,args.windows,args.prototype):record(p)
    windows={r:[] for r in helpers['ROUTES']}
    for w in csv.DictReader(args.windows.open()):
        if w['dataset']=='G2-v4' and w['route'] in windows:
            windows[w['route']].append(dict(segment=int(w['segment']),start=float(w['start_m']),end=float(w['end_m'])))
    def shadow(c,speed):
        if c.diagnostics.get('raw_steer') is None:return None
        points,geometry=c._geometry(max(0.,c._last_time-c._source_time))
        if geometry is None:return None
        location=min(geometry[0]+max(3.,.5*speed),c._arc[-1])
        aim=hermite_aim(points,c._arc,location)
        curvature=2.*aim[1]/max(float(aim@aim),1e-8)
        scale=float(np.interp(speed*3.6,c.steering_curve[:,0],c.steering_curve[:,1]))
        return dict(raw=-math.atan(c.wheelbase*curvature)/(c.max_steer_rad*scale),
                    aim_x=float(aim[0]),aim_y=float(aim[1]),forward=bool(aim[0]>0))
    frames=[];updates=[];checks=[]
    for route in helpers['ROUTES']:
        directory=args.run_root/route/'baseline-max/pursuit'
        for name in ('control.jsonl','trajectories.jsonl','agent_config.json','route_reference.json'):record(directory/name)
        config=json.loads((directory/'agent_config.json').read_text());configpath=Path(config['controller_config']);record(configpath)
        cfg=json.loads(configpath.read_text());parameters={k:v for k,v in cfg.items() if k not in helpers['ADAPTER_KEYS']}
        assert parameters.get('max_lookahead_time_s',.5)==.5 and parameters.get('lookahead')=='max'
        c=mod.Controller(preset='pursuit',**parameters)
        plans={r['frame']:r for r in map(json.loads,(directory/'trajectories.jsonl').read_text().splitlines())}
        controls=[json.loads(r) for r in (directory/'control.jsonl').read_text().splitlines()]
        xy=np.asarray(json.loads((directory/'route_reference.json').read_text())['world_xy'],float)
        previous=None;mismatch=0;diagnostic_mismatch=0
        for r in controls:
            plan=plans.get(r['frame']);old=copy.deepcopy(c) if plan else None
            if plan:c.update(plan['trajectory_xy'],plan['sim_time'])
            output=c.step(r['sim_time'],r['speed_mps'],r['yaw_rate_rps']);diag=c.diagnostics
            mismatch+=sum(a!=r[k] for a,k in zip(output,['throttle','steer','brake']))
            diagnostic_mismatch+=sum(helpers['compare'](v,r.get(k,'__missing__'))[0] for k,v in diag.items())
            h=shadow(c,r['speed_mps']);s=helpers['station'](np.asarray(r['truth_xy']),xy)
            seg=next((w['segment'] for w in windows[route] if w['start']<=s<=w['end']),None)
            row=dict(route=route,frame=r['frame'],sim_time=r['sim_time'],station_m=s,window_segment=seg,
                linear_raw=diag.get('raw_steer'),hermite_raw=h['raw'] if h else None,
                hermite_aim_x=h['aim_x'] if h else None,hermite_aim_y=h['aim_y'] if h else None,
                hermite_forward=h['forward'] if h else None,reason=diag['reason'],
                throttle_unchanged=output[0],brake_unchanged=output[2],trajectory_update=plan is not None)
            if previous and r['frame']==previous['frame']+1:
                dt=r['sim_time']-previous['sim_time']
                for kind in ('linear','hermite'):
                    if helpers['finite'](row[kind+'_raw']) and helpers['finite'](previous[kind+'_raw']):
                        row[kind+'_raw_rate']=(row[kind+'_raw']-previous[kind+'_raw'])/dt
            frames.append(row)
            if old:
                old.step(r['sim_time'],r['speed_mps'],r['yaw_rate_rps']);oh=shadow(old,r['speed_mps'])
                event=dict(route=route,frame=r['frame'],station_m=s,window_segment=seg,
                    linear_update_delta=None,hermite_update_delta=None)
                if helpers['finite'](diag.get('raw_steer')) and helpers['finite'](old.diagnostics.get('raw_steer')):
                    event['linear_update_delta']=diag['raw_steer']-old.diagnostics['raw_steer']
                if h and oh:event['hermite_update_delta']=h['raw']-oh['raw']
                updates.append(event)
            previous=row
        checks.append(dict(route=route,frames=len(controls),output_unequal_values=mismatch,
                           diagnostic_unequal_values=diagnostic_mismatch,longitudinal_changed=False))
    summaries=[]
    for route in helpers['ROUTES']:
        for w in windows[route]:
            rr=[r for r in frames if r['route']==route and r['window_segment']==w['segment']]
            uu=[r for r in updates if r['route']==route and r['window_segment']==w['segment']]
            result=dict(route=route,segment=w['segment'],frames=len(rr),updates=len(uu))
            for label in ('linear','hermite'):
                rate=np.array([r[label+'_raw_rate'] for r in rr if helpers['finite'](r.get(label+'_raw_rate'))])
                delta=np.array([r[label+'_update_delta'] for r in uu if helpers['finite'](r.get(label+'_update_delta'))])
                result.update({label+'_rate_count':len(rate),label+'_raw_rate_p95':float(np.percentile(abs(rate),95)),
                    label+'_raw_rate_rms':float(np.sqrt(np.mean(rate**2))),label+'_update_count':len(delta),
                    label+'_update_delta_rms':float(np.sqrt(np.mean(delta**2))),
                    label+'_update_delta_p95':float(np.percentile(abs(delta),95))})
            result['hermite_nonforward_count']=sum(r['hermite_forward'] is False for r in rr)
            summaries.append(result)
    args.out.mkdir(parents=True)
    for name,rows in [('per-frame-shadow.csv',frames),('per-update-shadow.csv',updates),('window-summary.csv',summaries)]:helpers['csv_write'](args.out/name,rows)
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    src=args.out/'source';src.mkdir();(src/'circle_sampling.py').write_text(prototype_text)
    for p in (Path(__file__),source,BASE/'replay_controller_updates.py'):(src/p.name).write_bytes(p.read_bytes())
    manifest=dict(inputs=inputs,outputs={str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.rglob('*') if p.is_file()},
        numpy=np.__version__,python=sys.version,prototype='Exact hermite_aim AST extracted from circle_sampling.py; does not import its mutable runtime Controller',
        scope='Historical baseline .5 states/paths/motion unchanged. Only alternate aim/raw computed; no shadow output advances controller or vehicle. Original longitudinal replay unchanged. Not formal guard implementation or closed-loop improvement.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,windows=summaries)))
    if any(c['output_unequal_values'] or c['diagnostic_unequal_values'] for c in checks):raise SystemExit(2)


if __name__=='__main__':main()
