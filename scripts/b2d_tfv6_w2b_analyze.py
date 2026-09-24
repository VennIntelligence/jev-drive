"""Pair W2b C/D with canonical W2 A/B; compare old/new C/D and repeat D1 logs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import numpy as np

import b2d_tfv6_analyze as w2
import b2d_tfv6_diagnose as d1
import b2d_tfv6_mechanism as mechanism
from b2d_tfv6_coordinates import rear_waypoints

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/runs/b2d/tfv6-w2/formal')
NEW = Path('/data/runs/b2d/tfv6-w2b')
OUT = ROOT / 'todos/2026-09-23-tfv6-controller/results/w2b'
OLD_TANGENT = ROOT / 'todos/2026-09-23-tfv6-controller/results/diagnosis/tangent/outcomes.csv'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def done_path(root, level, route, seed, arm):
    return root / ('level'+level) / 'cases' / level / ('route-'+route) / ('seed-'+str(seed)) / arm / 'done.json'


def result(done):
    run = Path(done['run_dir'])
    p = run / 'attempts' / done['route'] / '1' / 'results.json'
    records = json.loads(p.read_text())['_checkpoint']['records']
    if len(records) != 1:
        raise ValueError(p)
    return records[0]


def events(done):
    p = Path(done['run_dir']).parent / 'infractions.json'
    return [(int(x['step']),x['event_type'].split('.')[-1])
            for x in json.loads(p.read_text())['infractions']]


def same_trajectory_control(done):
    """Executed new C/D control against B shadow on that same realized path."""
    with (Path(done['run_dir']).parent/'frames.jsonl').open() as f:
        frames=[json.loads(line) for line in f]
    groups={band:[] for band in ('<0.5','0.5-3','3-8','>=8')}
    for row in frames:
        if not row.get('final_control') or not row.get('executed_control') or not row.get('truth'):
            continue
        b=row['final_control']['B']
        a=row['executed_control']
        speed=abs(row['truth']['forward_speed_mps'])
        band=str(d1.speed_bin(speed))
        groups[band].append(((a['throttle']-a['brake'])-(b['throttle']-b['brake']),
                             a['steer']-b['steer']))
    rows=[]
    for band,values in groups.items():
        arr=np.asarray(values,dtype=float).reshape(-1,2)
        def pct(col,p):
            return float(np.percentile(arr[:,col],p)) if len(arr) else None
        rows.append(dict(level=done['level'],route=done['route'],seed=done['seed'],
                         arm=done['arm'],speed_band=band,n=len(arr),
                         long_delta_median=pct(0,50),long_delta_p10=pct(0,10),
                         long_delta_p90=pct(0,90),steer_delta_median=pct(1,50),
                         steer_delta_p10=pct(1,10),steer_delta_p90=pct(1,90),
                         long_abs_gt_0_05=int(np.sum(np.abs(arr[:,0])>.05)),
                         steer_abs_gt_0_05=int(np.sum(np.abs(arr[:,1])>.05))))
    return rows


def plot_change(changes):
    import matplotlib.pyplot as plt
    from research import plot_style as style
    style.apply()
    routes=sorted({r['route'] for r in changes},key=int)
    fig,ax=plt.subplots(figsize=(style.DOUBLE_COLUMN_IN,5.7))
    for arm,offset,color in (('C',-.12,style.PALETTE['blue']),
                             ('D',.12,style.PALETTE['green'])):
        means=[];lo=[];hi=[]
        for route in routes:
            vals=np.asarray([r['ds_change'] for r in changes if r['route']==route and r['arm']==arm])
            means.append(float(np.mean(vals)))
            lo.append(float(np.mean(vals)-np.min(vals)))
            hi.append(float(np.max(vals)-np.mean(vals)))
        ax.errorbar(means,np.arange(len(routes))+offset,xerr=[lo,hi],fmt='o',
                    markersize=3,elinewidth=.7,capsize=1.5,color=color,label=arm)
    ax.axvline(0,color=style.BASELINE,lw=.7)
    ax.set_yticks(np.arange(len(routes)),routes)
    ax.set(xlabel='W2b − W2 Driving Score (route mean; seed range)',ylabel='Route')
    ax.legend()
    fig.tight_layout()
    style.save(fig,OUT/'cd-change')
    plt.close(fig)


def corrected_tracking(case_rows):
    """Use the W2b geometry on old B logs while retaining their realized W2 drive."""
    import matplotlib.pyplot as plt
    from research import plot_style as style
    rows=read_csv(OUT/'tracking.csv')
    rows=[x for x in rows if x['arm']!='B']
    with tempfile.TemporaryDirectory(prefix='w2b-B-tracking-',dir='/data/runs/b2d/tfv6-w2b') as temporary:
        tmp=Path(temporary)/'frames.jsonl'
        for item in case_rows:
            if item['arm']!='B':
                continue
            done=json.loads(done_path(OLD,item['level'],item['route'],item['seed'],'B').read_text())
            run=Path(done['run_dir'])
            with (run.parent/'frames.jsonl').open() as f:
                frames=[json.loads(line) for line in f]
            for frame in frames:
                if frame.get('waypoint') is not None:
                    frame['rear_waypoint']=rear_waypoints(frame['waypoint']).tolist()
            with tmp.open('w') as f:
                for frame in frames:
                    f.write(json.dumps(frame)+'\n')
            for metric in w2.frame_metrics(tmp,run.parent/'infractions.json','B'):
                rows.append({'level':item['level'],'route':item['route'],
                             'seed':item['seed'],**metric})
    w2.csv_write(OUT/'tracking.csv',rows)
    style.apply()
    fig,ax=plt.subplots(figsize=(style.SINGLE_COLUMN_IN,2.55))
    for arm,color in (('B',style.PALETTE['orange']),('C',style.PALETTE['blue']),
                      ('D',style.PALETTE['green'])):
        values=[r for r in rows if r['arm']==arm and r['segment']=='pre' and int(r['n'])]
        medians=[]
        for h in w2.HORIZONS:
            sub=[float(r['lateral_median_m']) for r in values if float(r['horizon_s'])==h]
            medians.append(float(np.median(sub)) if sub else np.nan)
        ax.plot(w2.HORIZONS,medians,marker='o',label=arm,color=color)
    ax.set(xlabel='Horizon (s)',ylabel='Pre-collision median lateral error (m)')
    ax.legend();fig.tight_layout()
    style.save(fig,OUT/'tracking')
    plt.close(fig)


def main():
    extra = [p for level in ('1','2') for p in OLD.glob(
        f'level{level}/cases/{level}/route-*/seed-*/*/done.json')
             if p.parent.name in 'AB']
    if len(extra) != 96:
        raise ValueError(f'Expected 96 W2 A/B baselines, found {len(extra)}')
    OUT.mkdir(parents=True,exist_ok=True)
    summary = w2.analyze([NEW/'level1',NEW/'level2'],OUT,extra=extra)
    if summary['missing_groups']:
        raise ValueError(f'Missing W2b paired groups: {summary["missing_groups"]}')
    case_rows=read_csv(OUT/'cases.csv')
    corrected_tracking(case_rows)
    old_tangent = {(r['level'],r['route'],int(r['seed']),r['arm']):r
                   for r in read_csv(OLD_TANGENT) if r['source']=='formal' and r['arm'] in 'CD'}
    changes=[]; control_rows=[]
    for level in ('1','2'):
        for p in sorted((NEW/('level'+level)).glob(f'cases/{level}/route-*/seed-*/*/done.json')):
            new = json.loads(p.read_text())
            if new['arm'] not in 'CD':
                continue
            key=(level,new['route'],new['seed'],new['arm'])
            prior=json.loads(done_path(OLD,*key).read_text())
            a,b=result(prior),result(new)
            old_o=old_tangent[key]
            new_events=events(new)
            old_event=old_o['first_linked_event_type']
            old_step=int(old_o['first_linked_event_step']) if old_event else None
            same_near=bool(old_event and any(kind==old_event and abs(step-old_step)<=60
                                             for step,kind in new_events))
            same_any=bool(old_event and any(kind==old_event for _,kind in new_events))
            changes.append(dict(level=level,route=new['route'],seed=new['seed'],arm=new['arm'],
                                old_ds=float(a['scores']['score_composed']),
                                new_ds=float(b['scores']['score_composed']),
                                ds_change=float(b['scores']['score_composed'])-float(a['scores']['score_composed']),
                                old_rc=float(a['scores']['score_route']),
                                new_rc=float(b['scores']['score_route']),
                                old_status=a['status'],new_status=b['status'],
                                old_phantom_ticks=int(old_o['phantom_ticks']),
                                old_linked_event=old_event,
                                old_linked_event_step=old_step if old_step is not None else '',
                                new_same_event_near_old_step_3s=same_near,
                                new_same_event_anytime=same_any,
                                new_event_types='|'.join(sorted(set(kind for _,kind in new_events)))))
            control_rows.extend(same_trajectory_control(new))
    if len(changes)!=96:
        raise ValueError(f'Expected 96 new C/D, found {len(changes)}')
    d1.write_csv(OUT/'cd-change.csv',changes)
    d1.write_csv(OUT/'same-trajectory-control.csv',control_rows)
    plot_change(changes)
    pairs_by_route={}
    case_index={(r['level'],r['route'],int(r['seed']),r['arm']):r for r in case_rows}
    for key,row in case_index.items():
        level,route,seed,arm=key
        if arm in 'CD':
            b=case_index[(level,route,seed,'B')]
            pairs_by_route.setdefault((route,arm),[]).append(float(row['ds'])-float(b['ds']))
    selected={route for (route,arm),vals in pairs_by_route.items()
              if len(vals)==3 and abs(float(np.mean(vals)))>=10}
    stretches=[]
    for item in changes:
        if item['route'] not in selected:
            continue
        done=json.loads(done_path(NEW,item['level'],item['route'],item['seed'],item['arm']).read_text())
        with (Path(done['run_dir']).parent/'frames.jsonl').open() as f:
            frames=[json.loads(line) for line in f]
        for span in mechanism.stretches(frames):
            row=mechanism.summarize(item['level'],item['route'],item['seed'],item['arm'],span)
            if row:
                stretches.append(row)
    d1.write_csv(OUT/'mechanism-stretches.csv',stretches)
    # Load new C/D and old A/B through D1's canonical path parser, then run the
    # exact same same-trajectory, first-divergence and log-score decomposition.
    loaded=[]
    for row in case_rows:
        d1.FORMAL = OLD if row['arm'] in 'AB' else NEW
        loaded.append(d1.load_case(row))
    pairs,divergences,tracking,lateral,controls,drift=d1.diagnose(loaded)
    for name,rows in [('d1-ds-pairs',pairs),('d1-divergences',divergences),
                      ('d1-speed-tracking',tracking),('d1-route-offset',lateral),
                      ('d1-control-bins',[x for x in controls if x['executed_arm']=='C'])]:
        d1.write_csv(OUT/(name+'.csv'),rows)
    side={}
    for arm in 'CD':
        x=[r for r in changes if r['arm']==arm]
        side[arm]={'n':len(x),'old_mean_ds':float(np.mean([r['old_ds'] for r in x])),
                   'new_mean_ds':float(np.mean([r['new_ds'] for r in x])),
                   'mean_change':float(np.mean([r['ds_change'] for r in x])),
                   'old_linked_cases':sum(bool(r['old_linked_event']) for r in x),
                   'new_same_near_old_step':sum(r['new_same_event_near_old_step_3s'] for r in x),
                   'new_same_anytime':sum(r['new_same_event_anytime'] for r in x)}
    (OUT/'comparison-summary.json').write_text(json.dumps(side,indent=2)+'\n')
    # The shared CSV writers use csv.writer's CRLF default. Keep repo artifacts
    # in LF format so git whitespace checks stay meaningful on Linux.
    for path in OUT.glob('*.csv'):
        raw = path.read_bytes()
        if b'\r\n' in raw:
            path.write_bytes(raw.replace(b'\r\n', b'\n'))
    print(json.dumps({'cases':len(case_rows),'paired':len(read_csv(OUT/'paired.csv')),
                      'changes':len(changes),'d1_pairs':len(pairs),
                      'mechanism_routes':len(selected),'mechanism_stretches':len(stretches),
                      'summary':side},indent=2))


if __name__=='__main__':
    main()
