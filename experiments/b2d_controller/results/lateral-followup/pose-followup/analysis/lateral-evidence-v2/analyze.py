#!/usr/bin/env python3
"""Frozen v1 geometry windows; saved control telemetry only. No simulator imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(values, prefix):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return {prefix+'_n':len(a), prefix+'_rms':float(np.sqrt(np.mean(a*a))) if len(a) else None,
            prefix+'_p95_abs':float(np.percentile(abs(a),95)) if len(a) else None,
            prefix+'_max_abs':float(max(abs(a))) if len(a) else None}


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w = csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--runs',type=Path,default=Path('/data/runs/b2d/controller'))
    args=ap.parse_args()
    if args.out.exists():raise SystemExit('Refusing existing output directory.')
    args.out.mkdir(parents=True)
    base=Path(__file__).resolve().parents[1]
    hashes={}
    def record(p):
        p=Path(p);hashes[str(p.resolve())]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    record(__file__)
    window_file=base/'lateral-evidence-v1/turn-windows.csv'
    route_file=base/'lateral-evidence-v1/routes.csv'
    record(window_file);record(route_file)
    windows={}
    for r in csv.DictReader(window_file.open()):
        key=(r['dataset'],r['route'])
        windows.setdefault(key,[]).append({k:(v if k in ('dataset','route','kind') else float(v)) for k,v in r.items()})
    references={(r['dataset'],r['route']):Path(r['source']) for r in csv.DictReader(route_file.open())}
    cases=[];attempt_inventory=[]
    for (dataset,route),ws in windows.items():
        if dataset=='G2-v4':
            for log in sorted((args.runs/'development-v4'/route).glob('*/*/control.jsonl')):
                cases.append(dict(dataset=dataset,route=route,variant='/'.join(log.parts[-3:-1]),seed='',attempt=1,
                                  directory=log.parent,official_status='G2 diagnostic',completion=None,collisions=None))
    for report in sorted((args.runs/'formal-v4').glob('*-seed*/controller-report.json')):
        record(report)
        entries=json.loads(report.read_text())['attempts']
        byroute={}
        for a in entries:
            byroute.setdefault(str(a['route_id']),[]).append(a)
            attempt_inventory.append(dict(group=report.parent.name,route=a['route_id'],attempt=a['attempt'],
                harness_finished=a.get('harness_finished'),status=a.get('status'),completion=a.get('completion')))
        for route,aa in sorted(byroute.items()):
            if ('Dev10-v4-pursuit-seed0-attempt1',route) not in windows:continue
            ordered=sorted(aa,key=lambda a:a['attempt'])
            a=next((a for a in ordered if a.get('harness_finished')),ordered[-1])
            cases.append(dict(dataset='Dev10-v4-pursuit-seed0-attempt1',route=route,variant=a['preset'],seed=a['seed'],
                attempt=a['attempt'],directory=report.parent/'attempts'/route/str(a['attempt']),
                official_status=a.get('official_status'),completion=a.get('completion'),
                collisions=sum(len(v) for k,v in a.get('infractions',{}).items() if k.startswith('collisions_'))))
    metrics=[];recovery=[];case_status=[];pooled={}
    for case in cases:
        directory=case.pop('directory');ref=references[(case['dataset'],case['route'])]
        log=directory/'control.jsonl'
        if not log.exists():
            case_status.append(dict(**case,status='missing_control',source=str(log)));continue
        record(ref);record(log)
        agent_config=directory/'agent_config.json';record(agent_config)
        config=json.loads(agent_config.read_text());configpath=Path(config['controller_config']);record(configpath)
        controller_config=json.loads(configpath.read_text());limit=controller_config.get('max_steer',.8)
        xy=np.asarray(json.loads(ref.read_text())['world_xy'],float)
        xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-8]]
        d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=1);s=np.r_[0.,np.cumsum(length)]
        records=[json.loads(line) for line in log.read_text().splitlines() if line]
        rows=[]
        previous=None
        for r in records:
            rate=np.nan
            if previous is not None:
                dt=r['sim_time']-previous['sim_time']
                if dt>0 and r['frame']==previous['frame']+1:rate=(r['steer']-previous['steer'])/dt
            previous=r
            if r.get('truth_xy') is None:continue
            p=np.asarray(r['truth_xy']);u=np.clip(np.sum((p-xy[:-1])*d,axis=1)/(length*length),0,1)
            foot=xy[:-1]+u[:,None]*d;dist2=np.sum((foot-p)**2,axis=1);i=int(np.argmin(dist2))
            progress=s[i]+u[i]*length[i];delta=p-foot[i]
            cte=(d[i,0]*delta[1]-d[i,1]*delta[0])/length[i]
            interp=lambda q:np.array([np.interp(q,s,xy[:,j]) for j in range(2)])
            tangent=interp(progress+2.5)-interp(progress-2.5)
            path_yaw=np.arctan2(tangent[1],tangent[0]);yaw=r.get('truth_yaw',np.nan)
            heading=np.degrees((yaw-path_yaw+np.pi)%(2*np.pi)-np.pi)
            rows.append(dict(frame=r['frame'],time=r['sim_time'],progress=progress,cte=cte,heading=heading,
                speed=r['speed_mps'],rate=rate,saturated=abs(r['steer'])>=limit-1e-6,
                limited=bool(r.get('steer_limited',False)),steer=r['steer'],reason=r.get('reason')))
        case_status.append(dict(**case,status='read',source=str(log),rows=len(records),truth_rows=len(rows),
            longitudinal_mode=controller_config.get('longitudinal_mode','vendor'),steer_limit=limit,
            lookahead=controller_config.get('lookahead','additive'),reference=str(ref)))
        ws=windows[(case['dataset'],case['route'])]
        for w in ws:
            selected=[r for r in rows if w['start_m']<=r['progress']<=w['end_m']]
            for subset in ('all','moving_ge_2mps'):
                rr=selected if subset=='all' else [r for r in selected if r['speed']>=2]
                output=dict(**case,segment=int(w['segment']),kind=w['kind'],subset=subset,count=len(rr),
                    frame_start=rr[0]['frame'] if rr else None,frame_end=rr[-1]['frame'] if rr else None,
                    low_speed_count=sum(r['speed']<2 for r in rr),
                    saturated_fraction=np.mean([r['saturated'] for r in rr]) if rr else None,
                    limited_fraction=np.mean([r['limited'] for r in rr]) if rr else None)
                for field,unit in (('cte','cte_m'),('heading','heading_deg'),('rate','steer_rate_per_s'),('steer','steer')):
                    output.update(stats([r[field] for r in rr],unit))
                metrics.append(output)
                key=(case['dataset'],case['variant'],case['seed'],subset)
                pooled.setdefault(key,[]).extend(rr)
            before=[r for r in rows if max(0,w['start_m']-5)<=r['progress']<w['start_m']]
            after=[r for r in rows if w['end_m']<r['progress']<=min(s[-1],w['end_m']+10)]
            # Recovery starts at first observed core exit; failure/stall remains censored.
            exits=[r for r in rows if r['progress']>=w['core_end_m']]
            recovery_time=None;streak_start=None
            if exits:
                start=exits[0]['time']
                candidates=[r for r in rows if r['time']>=start and r['progress']<=w['end_m']+10]
                prev=None
                for r in candidates:
                    good=abs(r['cte'])<=.25 and abs(r['heading'])<=5
                    if not good or (prev is not None and r['frame']!=prev['frame']+1):streak_start=None
                    if good and streak_start is None:streak_start=r['time']
                    if good and r['time']-streak_start>=.5-1e-7:
                        recovery_time=streak_start-start;break
                    prev=r
            recovery.append(dict(**case,segment=int(w['segment']),kind=w['kind'],core_exited=bool(exits),
                padded_window_exited=bool(after),recovery_time_s=recovery_time,
                recovery_censored=recovery_time is None,entry_count=len(before),post_count=len(after),
                **stats([r['cte'] for r in before],'entry_cte_m'),**stats([r['cte'] for r in after],'post_cte_m'),
                **stats([r['heading'] for r in after],'post_heading_deg')))
    aggregates=[]
    for key,rr in pooled.items():
        row=dict(zip(('dataset','variant','seed','subset'),key),count=len(rr),
            low_speed_count=sum(r['speed']<2 for r in rr),
            saturated_fraction=np.mean([r['saturated'] for r in rr]) if rr else None)
        for field,unit in (('cte','cte_m'),('heading','heading_deg'),('rate','steer_rate_per_s')):
            row.update(stats([r[field] for r in rr],unit))
        aggregates.append(row)
    for name,rows in [('per-window.csv',metrics),('recovery.csv',recovery),('cases.csv',case_status),
                      ('pooled-descriptive.csv',aggregates),('all-formal-attempts.csv',attempt_inventory)]:
        csv_write(args.out/name,rows)
    manifest=dict(inputs=hashes,outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.csv')},
        sampling='One emitted control per recorded consecutive frame, nominal .05s. Rate uses actual sim_time delta; gaps unavailable.',
        geometry='Frozen v1 turn windows; shared unmodified reference per route. Truth rear axle nearest segment projection, right-positive perpendicular CTE. Heading uses 5m centered chord; endpoint projection can hide along-track error.',
        saturation='abs(emitted steer)>=configured max_steer minus1e-6; default .8. steer_limited may also include rate limit.',
        recovery='Diagnostic only: after first core exit, first >=.5s consecutive samples with |CTE|<=.25m and |heading|<=5deg, observed until padded end+10m. No recovery is censored, not passed.',
        selection='First harness-finished attempt, otherwise last. All formal attempts indexed. All window frames including reverse/stall retained; moving>=2m/s auxiliary.',
        caveats=['Oracle route adapter/policy none, not actual TCP model. TCP preset omits native target arbitration.',
                 'Formal variants differ in longitudinal mode and actor realizations; descriptive comparison cannot identify lateral causal effect.',
                 'Pooled time-weighted values can be dominated by stalls; per-window values are primary. No new acceptance gate.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(cases=len(case_status),windows=len(recovery),metrics=len(metrics),out=str(args.out))))


if __name__=='__main__':main()
