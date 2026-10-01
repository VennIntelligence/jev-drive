"""One descriptive signed through-origin LS term, baseline-only fit, held-out windows/arm."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


WINDOWS={'26966':[(1,24.,51.)],'24240':[(1,18.,64.)],
         '17563':[(1,27.5,48.5),(2,73.5,95.)]}


def fit(rows):
    x=np.array([r['v2_omega'] for r in rows]);y=np.array([r['model_minus_rear_right_mps'] for r in rows])
    assert len(x)>0 and np.isfinite(x).all() and np.isfinite(y).all() and x@x>0
    return float(x@y/(x@x))


def metrics(rows,k):
    x=np.array([r['v2_omega'] for r in rows]);y=np.array([r['model_minus_rear_right_mps'] for r in rows])
    prediction=k*x;error=y-prediction
    return dict(n=len(rows),coefficient=k,target_mean_mps=float(y.mean()),prediction_mean_mps=float(prediction.mean()),
        target_rms_mps=float(np.sqrt(np.mean(y*y))),residual_rms_mps=float(np.sqrt(np.mean(error*error))),
        residual_p95_abs_mps=float(np.percentile(abs(error),95)),residual_mean_mps=float(error.mean()),
        rms_reduction_fraction=float(1-np.sqrt(np.mean(error*error)/np.mean(y*y))),
        prediction_target_correlation=float(np.corrcoef(prediction,y)[0,1]),
        sign_agreement_fraction=float(np.mean(prediction*y>0)),
        count_abs_world_gyro_lt_point1=sum(abs(r['world_gyro_rps'])<.1 for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--propagation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def record(path):hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    record(args.propagation/'frames.csv');record(Path(__file__))
    motion={}
    for route in WINDOWS:
        for variant in ('baseline-max','short-max'):
            path=args.run_root/route/variant/'pursuit/control.jsonl';record(path)
            c=[json.loads(s) for s in path.read_text().splitlines()]
            for a,b in zip(c,c[1:]):
                assert b['frame']==a['frame']+1
                speed=.5*(a['speed_mps']+b['speed_mps'])
                omega=-.5*(a['yaw_rate_rps']+b['yaw_rate_rps'])
                motion[(route,variant,b['frame'])]=(speed,omega)
    rows=[]
    for r in csv.DictReader((args.propagation/'frames.csv').open()):
        route=r['route'];station=float(r['station_m'])
        for segment,start,end in WINDOWS[route]:
            if start<=station<=end:
                speed,omega=motion[(route,r['variant'],int(r['frame']))]
                rows.append(dict(route=route,variant=r['variant'],segment=segment,frame=int(r['frame']),
                    station_m=station,interval_speed_mps=speed,world_gyro_rps=omega,v2_omega=speed*speed*omega,
                    model_minus_rear_right_mps=float(r['true_heading_model_minus_rear_right_mps'])))
    baseline=[r for r in rows if r['variant']=='baseline-max'];holdout=[r for r in rows if r['variant']=='short-max']
    k=fit(baseline)
    summaries=[]
    for variant in ('baseline-max','short-max'):
        for route in WINDOWS:
            for segment,_,_ in WINDOWS[route]:
                window=[r for r in rows if r['variant']==variant and r['route']==route and r['segment']==segment]
                summaries.append(dict(route=route,variant=variant,segment=segment,fit_scope='global_baseline_four_windows',**metrics(window,k)))
    loo=[]
    for route in WINDOWS:
        for segment,_,_ in WINDOWS[route]:
            train=[r for r in baseline if (r['route'],r['segment'])!=(route,segment)]
            test=[r for r in baseline if (r['route'],r['segment'])==(route,segment)]
            lk=fit(train)
            loo.append(dict(route=route,variant='baseline-max',segment=segment,fit_scope='leave_this_window_out',
                            training_count=len(train),**metrics(test,lk)))
    for row in rows:
        row.update(global_prediction_mps=k*row['v2_omega'],global_error_mps=row['model_minus_rear_right_mps']-k*row['v2_omega'])
    for name,values in [('frames.csv',rows),('windows.csv',summaries),('leave-one-window-out.csv',loo)]:
        with (args.out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summary=dict(coefficient=k,coefficient_units='s^2/m (radians dimensionless)',
        fit_definition='Single signed through-origin ordinary least squares minimizing sum((y-k*x)^2) over all 328 baseline fixed-window samples. No window weights, intercept, lag search, filtering, or short-arm fit.',
        x='endpoint-mean signed SPEED squared times endpoint-mean world gyro (right-positive rad/s); prior/current data only',
        y='no-lateral-motion true-midpoint-heading model minus directly differenced rear displacement, projected interval-midpoint RIGHT, divided by actualdt',
        sign='Positive k means model predicts too much rightward displacement for a right turn. A compensating rear velocity would be -k*v^2*omega*right. This is NOT implemented.',
        baseline=metrics(baseline,k),held_out_short_arm=metrics(holdout,k),windows=summaries,leave_one_window_out=loo,
        limits='Descriptive fitted motion term, not identified tire parameter or validated localization/controller improvement. Finite-difference interval target, lever geometry, pitch, timing, transient forces, GNSS and model feedback remain confounds; all fixed-window samples retained.',
        inputs=hashes)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(__file__,str(args.out/'recompute.py'))
    (args.out/'outputs-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('coefficient','baseline','held_out_short_arm','leave_one_window_out')},indent=2))


if __name__=='__main__':main()
