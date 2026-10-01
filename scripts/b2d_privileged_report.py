"""Current-state event readouts and route-cluster paired bootstrap for privileged ceilings.

Debug and diagnostic runs never enter formal estimates. All files stay in the new run root.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from jevdrive.op_arb_report import attempts, drive_row
from b2d_privileged_geometry import ARMS, project


def jsonlines(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def waiting(rows):
    durations=[];start=None;last=None
    for r in rows:
        t=r['t'];v=r['ego']['v']
        if v<.2:
            if start is None:start=t
        elif start is not None:
            durations.append(max(0.,t-start));start=None
        last=t
    if start is not None:durations.append(max(0.,last-start+.2))
    durations=[d for d in durations if d>=1.-1e-6]
    return sum(durations),max(durations,default=0.)


def event_rows(path,meta):
    scenes=jsonlines(path/'privileged.jsonl')
    live=[r for r in scenes if not r['pc']['warm']]
    plans=[r for r in jsonlines(path/'plans.jsonl') if not r['warm']]
    contacts=[];last_contact={}
    for c in sorted(jsonlines(path/'contacts.jsonl'),key=lambda row:row['t']):
        if c['impulse']<1.:continue
        if c['t']-last_contact.get(c['id'],-1e9)>1.:contacts.append(c)
        last_contact[c['id']]=c['t']
    route=np.asarray(json.loads((path/'route.json').read_text())['xy'])
    events=[]
    for kind,field in [('junction','junctions'),('obstacle','obstacles')]:
        groups=[]
        for i,row in enumerate(live):
            for candidate in row['pc'][field]:
                ids=set(candidate['ids'])
                matching=[g for g in groups if (kind=='junction' and g['key']==candidate['key']) or
                          (kind=='obstacle' and g['ids'] & ids)]
                if matching:
                    g=matching[0]
                    # One observation can join previously separate obstacle groups. Keep one union event.
                    for other in matching[1:]:
                        g['ids']|=other['ids'];g['end_s']=max(g['end_s'],other['end_s'])
                        g['start_s']=min(g['start_s'],other['start_s']);g['first']=min(g['first'],other['first'])
                        groups.remove(other)
                    g['ids']|=ids;g['end_s']=max(g['end_s'],candidate['end_s'])
                    g['start_s']=min(g['start_s'],candidate['start_s'])
                else:
                    g=dict(key=candidate['key'],ids=ids,start_s=candidate['start_s'],end_s=candidate['end_s'],first=i)
                    groups.append(g)
        for g in groups:
            first=g['first'];tail=live[first:]
            # The event ends at passing the junction; obstacle completion includes return geometry.
            threshold=g['end_s']+(5 if kind=='junction' else 23)
            indices=[i for i,r in enumerate(tail) if r['pc']['ego_s']>threshold]
            end=indices[0] if indices else len(tail)-1
            rows=tail[:end+1];t0,t1=rows[0]['t'],rows[-1]['t']
            relevant=[c for c in contacts if c['id'] in g['ids'] and t0-.2<=c['t']<=t1+.2]
            wait_s,wait_max=waiting(rows)
            ego_s=rows[-1]['pc']['ego_s'];passed=ego_s>g['end_s']+(5 if kind=='junction' else 6)
            lateral=float(project([rows[-1]['ego']['xyz'][:2]],route)[1][0])
            returned=bool(kind=='junction' or ego_s>g['end_s']+23 and abs(lateral)<=.75)
            success=bool(passed and returned and not relevant and wait_max<60)
            enabled=any(r['pc'].get('bypass') if kind=='obstacle' else 'pjunc' in r['pc'].get('controls',{}) for r in rows)
            go=any(r['ego']['v']>1 and not r['pc'].get('hold') for r in rows[1:])
            no_adj=any(r['pc'].get('no_adjacent_lane',False) for r in rows)
            wrong_return=kind=='obstacle' and passed and not returned
            trace=('contact' if relevant else 'waiting_60s' if wait_max>=60 else 'no_adjacent_lane' if no_adj else
                   'no_constraint_or_geometry' if not enabled and not success else 'return_incomplete' if wrong_return else
                   'passage_incomplete' if not success else 'completed')
            opposing=[]
            if kind=='obstacle':
                for r in rows:
                    st=r['pc'].get('bypass_state',{})
                    if not st.get('borrow'):continue
                    for a in r['actors']:
                        if not a['type'].startswith('vehicle.'):continue
                        _,d,tangent=project([a['xyz'][:2]],route)
                        if np.asarray(a['velocity'])@tangent[0]<-.5 and abs(d[0]-st['offset'])<=2:
                            opposing.append(a['surface_gap'])
            events.append(dict(**meta,kind=kind,key=str(g['key']),ids=json.dumps(sorted(g['ids'])),start_t=t0,end_t=t1,
                    start_s=g['start_s'],end_s=g['end_s'],opportunity=1,failed=int(not success),success=int(success),
                    passed=int(passed),returned=int(returned),contact=int(bool(relevant)),contacts=len(relevant),
                    wait_s=wait_s,wait_max_s=wait_max,blocked=int(wait_max>=60 or not passed),
                    enabled=int(enabled),clear_go=int(go),execution_failed=int(enabled and not success),
                    no_adjacent=int(no_adj),failure_stage=trace,opposing_gap_m=min(opposing) if opposing else np.nan,
                    green_latency_s=np.nan,green_censored=0))
    # Red opportunities use logged light identity, not a control trigger or infraction count.
    lights={}
    for i,r in enumerate(plans):
        c=r.get('ctx',{});lid=c.get('tl_id');d=c.get('tl_dist',1e9)
        if lid is not None and c.get('tl')==2 and 0<d<=50:
            lights.setdefault(lid,dict(first=i,line_s=r['pc']['ego_s']+3.8394+d))
    for lid,g in lights.items():
        rows=plans[g['first']:];end=next((i for i,r in enumerate(rows) if r['pc']['ego_s']>g['line_s']+5),len(rows)-1)
        rows=rows[:end+1];first,last=rows[0],rows[-1]
        green=next((r for r in rows if r.get('ctx',{}).get('tl_id')==lid and r['ctx'].get('tl')==0),None)
        go=next((r for r in rows if green is not None and r['t']>=green['t'] and r['v']>1),None)
        violation=any(a.get('ctx',{}).get('tl_id')==lid and b.get('ctx',{}).get('tl_id')==lid and
                      a['ctx'].get('tl_dist',-1)>0 and b['ctx'].get('tl_dist',1)<=0 and b['ctx'].get('tl')==2
                      for a,b in zip(rows[:-1],rows[1:]))
        # The official violation names provide exact stop-line crossing when the sparse observation misses it.
        result=json.loads((path/'results.json').read_text())['_checkpoint']['records'][0]
        violation=violation or any(re.search(r'\b'+str(lid)+r'\b',s) for s in result['infractions'].get('red_light',[]))
        passed=last['pc']['ego_s']>g['line_s']+5
        success=passed and not violation
        stopped=any(r['v']<.2 and r.get('ctx',{}).get('tl_id')==lid and r['ctx'].get('tl')==2 for r in rows)
        enabled=any('pred' in r['pc'].get('controls',{}) for r in rows)
        events.append(dict(**meta,kind='red',key=str(lid),ids='[]',start_t=first['t'],end_t=last['t'],
             start_s=g['line_s'],end_s=g['line_s'],opportunity=1,failed=int(not success),success=int(success),
             passed=int(passed),returned=int(passed),contact=0,contacts=0,wait_s=sum(r['v']<.2 for r in rows)*.05,
             wait_max_s=np.nan,blocked=int(not passed),enabled=int(enabled),clear_go=int(go is not None),
             execution_failed=int(enabled and not success),no_adjacent=0,
             failure_stage='red_crossing' if violation else 'release_or_passage_incomplete' if not passed else 'completed',
             opposing_gap_m=np.nan,red_stop=int(stopped),red_cross=int(violation),
             green_latency_s=go['t']-green['t'] if go is not None else np.nan,
             green_censored=int(green is None or go is None)))
    visibility=[]
    if meta['arm']=='drive':
        for e in events:
            if e['kind']=='red':continue
            for aid in json.loads(e['ids']):
                history=[(r,a) for r in scenes for a in r['actors'] if a['id']==aid and r['t']<=e['end_t']+.2]
                if not history:continue
                # Closest approach is restricted to the recorded event window; earlier visibility uses the full history.
                window=[(r,a) for r,a in history if r['t']>=e['start_t']-.2]
                if not window:continue
                touching=[c for c in contacts if c['id']==aid and e['start_t']-.2<=c['t']<=e['end_t']+.2]
                end_t=min(c['t'] for c in touching) if touching else min(window,key=lambda pair:pair[1]['surface_gap'])[0]['t']
                for camera in ('road','wide','union'):
                    visible=lambda a:any(v[0] for v in a['view'].values()) if camera=='union' else a['view'][camera][0]
                    seen=next(((r,a) for r,a in history if r['t']<=end_t and visible(a)),None)
                    left=bool(seen is not None and visible(history[0][1]))
                    never=seen is None;lead=0. if never else max(0.,end_t-seen[0]['t'])
                    box=(max(seen[1]['view'].values(),key=lambda v:v[3]) if camera=='union' else seen[1]['view'][camera]) if seen else [False,0,0,0]
                    visibility.append(dict(**meta,kind=e['kind'],event_key=e['key'],actor_id=aid,camera=camera,
                         endpoint_t=end_t,endpoint_contact=int(bool(touching)),first_t=seen[0]['t'] if seen else np.nan,
                         lead_s=lead,left_censored=int(left),never_seen=int(never),
                         definitely_under_2s=int(not left and lead<2),possibly_under_2s=int(lead<2),
                         first_distance_m=seen[1]['distance'] if seen else np.nan,width_px=box[1],height_px=box[2],area_px2=box[3]))
    return events,visibility


def read_unit(adir,tag,arm,seed,log):
    root=adir.parents[1];out=root/'readouts'/adir.name;out.mkdir(parents=True,exist_ok=True)
    routes=[];events=[];visible=[]
    for rid,path in attempts(adir).items():
        meta=dict(unit=adir.name,group=tag.split('-')[1],route=str(rid),arm=arm,seed=seed,attempt=str(path))
        r=drive_row(path);r.update(meta)
        counts=np.array([r[k] for k in ('coll_ped','coll_veh','coll_layout','red_light')])
        penalties=np.power(np.array([.5,.6,.65,.7]),counts)
        product=penalties.prod();reconstructed=r['RC']*product
        r['DS_reconstructed']=reconstructed;r['DS_residual']=r['DS']-reconstructed
        for k,p in zip(('pedestrian','vehicle','layout','red'),penalties):r['recover_'+k]=reconstructed*(1/p-1)
        r['recover_route_completion']=(100-r['RC'])*product
        routes.append(r);ee,vv=event_rows(path,meta);events.extend(ee);visible.extend(vv)
    pd.DataFrame(routes).to_csv(out/'routes.csv',index=False)
    pd.DataFrame(events).to_csv(out/'events.csv',index=False)
    pd.DataFrame(visible).to_csv(out/'visibility.csv',index=False)
    log.event('readout',unit=adir.name,routes=len(routes),events=len(events),visibility_pairs=len(visible))
    log.info(f'Read {adir.name}: DS={np.mean([r["DS"] for r in routes]):.2f}, events={len(events)}')


def concatenate(root,name):
    rows=[]
    for p in sorted((root/'readouts').glob('eval-*/*'+name)):
        if p.stat().st_size>1:
            try:rows.append(pd.read_csv(p,dtype={'route':str}))
            except pd.errors.EmptyDataError:pass
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def bootstrap_delta(routes,events,arm,group,kind):
    r=routes[routes['group']==group] if group!='all' else routes
    ids=sorted(r.route.unique());n=len(ids);draw=np.random.default_rng(0).integers(0,n,(2000,n))
    pairs=r[r.arm.isin([arm,'drive'])].groupby(['route','arm']).DS.mean().unstack().reindex(ids)
    assert pairs[[arm,'drive']].notna().all().all(),'Incomplete paired routes'
    dd=(pairs[arm]-pairs.drive).to_numpy();boot=dd[draw].mean(1)
    e=events[(events['group']==group) & (events.kind==kind)] if group!='all' else events[events.kind==kind] if kind!='all' else events
    counts=[]
    for a in ('drive',arm):
        c=e[e.arm==a].groupby('route')[['failed','opportunity']].sum().reindex(ids,fill_value=0).to_numpy()
        counts.append(c)
    def ratio(c):return c[...,0]/np.where(c[...,1]>0,c[...,1],np.nan)
    effect=ratio(counts[1].sum(0))-ratio(counts[0].sum(0))
    effects=ratio(counts[1][draw].sum(1))-ratio(counts[0][draw].sum(1))
    return dict(arm=arm,group=group,kind=kind,routes=n,dDS=float(dd.mean()),DS_lo=np.quantile(boot,.025),DS_hi=np.quantile(boot,.975),
                fail_delta=effect,fail_lo=np.nanquantile(effects,.025),fail_hi=np.nanquantile(effects,.975),
                bootstrap_undefined=int(np.isnan(effects).sum()),drive_opps=int(counts[0][:,1].sum()),arm_opps=int(counts[1][:,1].sum()))


def absolute_intervals(routes,events):
    route_rows=[];event_rates=[]
    for group in ('all','junction','obstacle','dev'):
        rr=routes if group=='all' else routes[routes['group']==group]
        ee=events if group=='all' else events[events['group']==group]
        ids=sorted(rr.route.unique());draw=np.random.default_rng(0).integers(0,len(ids),(2000,len(ids)))
        for arm in ARMS:
            a=rr[rr.arm==arm].groupby('route')[['DS','RC','v_mean']].mean().reindex(ids)
            for metric in ('DS','RC','v_mean'):
                values=a[metric].to_numpy();boot=values[draw].mean(1)
                route_rows.append(dict(group=group,arm=arm,metric=metric,mean=values.mean(),
                         lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),routes=len(ids)))
            for kind in ('junction','obstacle','red'):
                b=ee[(ee.arm==arm)&(ee.kind==kind)].groupby('route')[['failed','opportunity']].sum().reindex(ids,fill_value=0).to_numpy()
                sums=b[draw].sum(1);valid=sums[:,1]>0
                boot=sums[valid,0]/sums[valid,1]
                event_rates.append(dict(group=group,arm=arm,kind=kind,opportunities=int(b[:,1].sum()),failures=int(b[:,0].sum()),
                     rate=b[:,0].sum()/b[:,1].sum() if b[:,1].sum()>0 else np.nan,
                     lo=np.quantile(boot,.025) if len(boot) else np.nan,hi=np.quantile(boot,.975) if len(boot) else np.nan,
                     bootstrap_undefined=int((~valid).sum())))
    return pd.DataFrame(route_rows),pd.DataFrame(event_rates)


def summarize(root,log):
    out=root/'summary';out.mkdir(exist_ok=True)
    r=concatenate(root,'routes.csv');e=concatenate(root,'events.csv');v=concatenate(root,'visibility.csv')
    assert len(r)==696 and not r.duplicated(['route','arm','seed']).any(),'Formal run completeness failed'
    for df,name in ((r,'routes'),(e,'events'),(v,'visibility')):df.to_csv(out/(name+'.csv'),index=False)
    r.groupby('arm').mean(numeric_only=True).reindex(ARMS).to_csv(out/'arm_means.csv')
    raw_ci,event_ci=absolute_intervals(r,e)
    raw_ci.to_csv(out/'absolute_route_intervals.csv',index=False);event_ci.to_csv(out/'event_rate_intervals.csv',index=False)
    e.groupby(['group','arm','kind']).agg(opportunities=('opportunity','sum'),failures=('failed','sum'),
       successes=('success','sum'),contacts=('contact','sum'),blocked=('blocked','sum'),
       wait_s=('wait_s','sum'),enabled=('enabled','sum'),execution_failed=('execution_failed','sum')).to_csv(out/'event_counts.csv')
    spec=[('pjunc','junction','junction'),('pbyp','obstacle','obstacle'),('pbypgap','obstacle','obstacle'),('pred','all','red'),('pall','all','all')]
    pairs=pd.DataFrame([bootstrap_delta(r,e,*s) for s in tqdm(spec,desc='Paired route bootstrap')]);pairs.to_csv(out/'paired.csv',index=False)
    gates={};complete=len(r)==696 and not r.status.isin(['Failed','Simulation crashed','Agent crashed',"Agent couldn't be set up"]).any()
    power=all(len(e[(e.arm==a)&(e['group']==g)&(e.kind==g)])>=30 for a in ARMS for g in ('junction','obstacle'))
    gates['I0']=bool(complete and power)
    for label,row in zip(('J1','B1','B2','R1','A1'),pairs.to_dict('records')):
        gates[label]=bool(gates['I0'] and row['arm_opps']>=30 and row['drive_opps']>=30 and row['dDS']>0 and row['fail_hi']<0)
    xrows=[]
    for arm,group,kind in spec[:-1]:
        ee=e[(e.arm==arm)&(e.kind==kind)]
        if group!='all':ee=ee[ee['group']==group]
        success=ee.success.mean();execution=ee.execution_failed.mean()
        xrows.append(dict(arm=arm,opportunities=len(ee),success_rate=success,execution_failure_rate=execution,
                          passed=bool(gates['I0'] and success>=.8 and execution<=.1)))
    pd.DataFrame(xrows).to_csv(out/'execution_gates.csv',index=False)
    vis=[]
    for group in ('junction','obstacle'):
        vv=v[(v['group']==group)&(v.kind==group)&(v.camera=='union')];ids=sorted(r[r['group']==group].route.unique())
        c=vv.groupby('route').agg(short=('definitely_under_2s','sum'),possible=('possibly_under_2s','sum'),n=('actor_id','size')).reindex(ids,fill_value=0)
        draws=np.random.default_rng(0).integers(0,len(ids),(2000,len(ids)));s=c.to_numpy()[draws].sum(1)
        boot=s[:,0]/np.where(s[:,2]>0,s[:,2],np.nan)
        lo,hi=np.nanquantile(boot,[.025,.975]);n=len(vv)
        opps=len(e[(e.arm=='drive')&(e['group']==group)&(e.kind==group)])
        gates['V1_'+group]=bool(opps>=30 and n>=30 and lo>.5)
        vis.append(dict(group=group,opportunities=opps,objects=n,definite_short=vv.definitely_under_2s.mean(),possible_short=vv.possibly_under_2s.mean(),
              short_lo=lo,short_hi=hi,left_censored=int(vv.left_censored.sum()),never_seen=int(vv.never_seen.sum()),
              median_observed_upper_lead_s=vv.lead_s.median(),passed=gates['V1_'+group]))
    pd.DataFrame(vis).to_csv(out/'visibility_gates.csv',index=False)
    wide=r.groupby(['route','arm']).DS.mean().unstack().reindex(columns=ARMS)
    interaction=(wide.pall-wide.drive)-(wide.pjunc+wide.pbypgap+wide.pred-3*wide.drive)
    draws=np.random.default_rng(0).integers(0,len(wide),(2000,len(wide)))
    ci=np.quantile(interaction.to_numpy()[draws].mean(1),[.025,.975])
    (out/'gates.json').write_text(json.dumps(dict(gates=gates,interaction_delta=float(interaction.mean()),interaction_lo=float(ci[0]),interaction_hi=float(ci[1])),indent=2)+'\n')
    log.event('summary',gates=gates)
    return out


if __name__=='__main__':
    from jevdrive.runlog import RunLog
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(os.environ['DATA_DIR'])/'runs/b2d_privileged_ceiling')
    args=p.parse_args();log=RunLog('b2d_privileged_ceiling','report')
    try:summarize(args.root,log)
    finally:log.close()
