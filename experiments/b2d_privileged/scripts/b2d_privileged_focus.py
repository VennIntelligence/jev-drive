"""Run only previously low-scoring dev routes with the relevant privileged arms.

Reuse the frozen controller, queue, event parser and route bootstrap. No training.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/b2d_privileged/scripts",)]
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from tqdm import tqdm

from b2d_privileged_chain import Chain, ROOT, write
from b2d_privileged_checks import route_checks
from b2d_privileged_checks import source_check
from b2d_privileged_report import read_unit, bootstrap_delta
from b2d_privileged_plots import figures

COHORTS=dict(junction=['27043','9196','37969'],obstacle=['24497'],red=['24944','27870','27297'])
SPECS=[('pjunc','junction','junction'),('pbyp','obstacle','obstacle'),
       ('pbypgap','obstacle','obstacle'),('pred','red','red'),('pall','all','all')]


class Focus(Chain):
    def full(self):
        assert (ROOT/'DONE-repair').exists(), 'Single-unit repair checklist must pass first'
        proof=source_check()
        assert proof['new_control_sha256']==self.control_hash
        # The control calculation is byte-identical after removing the logging-only helper.
        # Recheck the original completed debug units with the corrected crash classifier.
        checks={}
        for rid,arm,kind in [('27787','pred','drive'),('334','pred','pred'),('26872','pjunc','pjunc'),
                             ('25169','pbyp','pbyp'),('24955','pbyp','pbyp')]:
            adir=ROOT/'arms'/f'debug-v4-{rid}-{arm}-s0'
            checks[rid+'-'+arm]=route_checks(self.attempt(adir,rid),kind)
        for p in sorted((ROOT/'arms').glob('debug-v4-*/done/*.json')):
            basic=route_checks(self.attempt(p.parent.parent,p.stem),'drive')
            assert basic['checks']['no_crash'] and basic['checks']['finite'],basic
        assert all(c['passed'] for c in checks.values()), 'Existing debug execution checklist failed'
        focus=ROOT/'focus';out=focus/'summary';out.mkdir(parents=True,exist_ok=True)
        ids=[rid for routes in COHORTS.values() for rid in routes]
        spec=[('drive',ids),('pall',ids),('pjunc',COHORTS['junction']),('pred',COHORTS['red']),
              ('pbyp',COHORTS['obstacle']),('pbypgap',COHORTS['obstacle'])]
        tasks=[(f'focus-low-v6-{arm}',arm,seed,routes) for arm,routes in spec for seed in (0,1)]
        write(focus/'plan.json',dict(cohorts=COHORTS,tasks=tasks,runs=sum(len(t[3]) for t in tasks),
              source_checks=proof,debug_checks=checks,selection='Prior ld-drive two-seed mean DS < 100; relevant arms only',
              inference='Selected low-score diagnostic; original 696-run confirmation discontinued by user'))
        start=__import__('time').time()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(self.unit,*task,record=True):task for task in tasks}
            for f in tqdm(as_completed(futures),total=len(futures),desc='Low-score diagnostic shards'):
                tag,arm,seed,_=futures[f];adir=f.result();read_unit(adir,tag,arm,seed,self.log)
        tables={}
        group_of={rid:group for group,routes in COHORTS.items() for rid in routes}
        for name in ('routes','events','visibility','yellow'):
            frames=[]
            for tag,arm,seed,_ in tasks:
                p=ROOT/'readouts'/f'{tag}-{arm}-s{seed}'/(name+'.csv')
                try:frames.append(pd.read_csv(p,dtype={'route':str}))
                except pd.errors.EmptyDataError:pass
            frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
            if len(frame):frame['group']=frame.route.map(group_of)
            frame.to_csv(out/(name+'.csv'),index=False);tables[name]=frame
        r,e=tables['routes'],tables['events']
        expected={(rid,arm,seed) for _,arm,seed,routes in tasks for rid in routes}
        assert len(r)==44 and set(zip(r.route,r.arm,r.seed))==expected and not r.duplicated(['route','arm','seed']).any()
        paired=pd.DataFrame([bootstrap_delta(r,e,*s) for s in tqdm(SPECS,desc='Route-cluster diagnostic bootstrap')])
        paired.to_csv(out/'paired.csv',index=False)
        r.groupby(['group','arm']).mean(numeric_only=True).to_csv(out/'arm_means.csv')
        e.groupby(['group','arm','kind']).agg(opportunities=('opportunity','sum'),failures=('failed','sum'),
             successes=('success','sum'),contacts=('contact','sum'),blocked=('blocked','sum'),wait_s=('wait_s','sum'),
             enabled=('enabled','sum'),execution_failed=('execution_failed','sum')).to_csv(out/'event_counts.csv')
        videos=[]
        for arm,group,kind in SPECS:
            ee=e[e.arm==arm]
            if group!='all':ee=ee[ee['group']==group]
            if kind!='all':ee=ee[ee.kind==kind]
            runs=ee.groupby(['route','seed']).failed.max()
            for outcome,value in [('success',0),('failure',1)]:
                choices=sorted([(rid,int(seed)) for (rid,seed),failed in runs.items() if failed==value],key=lambda x:(int(x[0]),x[1]))
                if not choices:videos.append(dict(arm=arm,outcome=outcome,available=False));continue
                rid,seed=choices[0];paths={}
                for a in (arm,'drive'):
                    row=r[(r.arm==a)&(r.route==rid)&(r.seed==seed)].iloc[0];p=Path(row.attempt)
                    paths[a]=dict(path=str(p/'chase.mp4'),DS=float(row.DS),qa=json.loads((p/'video_qa.json').read_text()))
                videos.append(dict(arm=arm,outcome=outcome,available=True,route=rid,seed=seed,paths=paths))
        write(out/'videos.json',dict(pairs=videos,selection='First numeric route and seed by actual recorded outcome; no reruns'))
        end=__import__('time').time();util=pd.read_csv(ROOT/'util.csv');util=util[(util.time>=start)&(util.time<=end)]
        write(out/'timing.json',dict(start=start,end=end,wall_s=end-start,runs=44,routes_per_hour=44*3600/(end-start),
              gpu={str(g):dict(mean_utilization_pct=float(util[util.gpu==g].utilization_pct.mean()),
                     peak_memory_mib=int(util[util.gpu==g].memory_used_mib.max())) for g in range(3)}))
        figures(focus,diagnostic=True)
        self.log.event('focus_complete',runs=44,summary=str(out))


if __name__=='__main__':
    Focus(SimpleNamespace(phase='focus',slots=2,workers=4,chunk_size=12,
                          verification_version='v6',after_slot='')).run()
