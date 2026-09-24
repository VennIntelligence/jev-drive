"""Route-cluster bootstrap summary of frozen L1 case scores."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def paired(rows, arm, baseline, metric):
    table={(r['route'],r['seed'],r['arm']):r for r in rows}
    routes=sorted({r['route'] for r in rows})
    values=[]
    for route in routes:
        differences=[]
        for seed in ('p01','p02','p03','p04','p05'):
            a=table[(route,seed,arm)][metric]
            b=table[(route,seed,baseline)][metric]
            differences.append(float(a)-float(b))
        values.append(float(np.mean(differences)))
    x=np.asarray(values)
    rng=np.random.default_rng(20260924)
    samples=np.mean(x[rng.integers(0,len(x),(10000,len(x)))],axis=1)
    return dict(mean=float(np.mean(x)), ci95=np.percentile(samples,[2.5,97.5]).tolist(),
                route_differences={route:float(value) for route,value in zip(routes,x)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=list(csv.DictReader(a.cases.open()))
    if len(rows)!=320:raise ValueError(f'Expected 320 L1 cases, got {len(rows)}')
    result={}
    for reference in ('oracle','expert'):
        subset=[r for r in rows if r['reference']==reference]
        result[reference]={}
        for arm in 'ABCD':
            group=[r for r in subset if r['arm']==arm]
            if len(group)!=40:raise ValueError(f'Expected 40 {reference}/{arm} cases')
            metrics={key:float(np.median([float(r[key]) for r in group])) for key in
                     ('primary','cte_rms_m','time_xy_rms_m','speed_rms_mps',
                      'extra_jerk_rms_mps3','final_overshoot_m')}
            metrics.update(cases=len(group),completed=sum(r['status']=='completed' for r in group),
                           collisions=sum(int(r['collision_count']) for r in group),
                           terminal_hold=sum(r['terminal_hold']=='True' for r in group))
            if arm!='A':metrics['primary_vs_A']=paired(subset,arm,'A','primary')
            if arm in 'CD':metrics['primary_vs_B']=paired(subset,arm,'B','primary')
            result[reference][arm]=metrics
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote L1 summary to {a.out}')


if __name__=='__main__':main()
