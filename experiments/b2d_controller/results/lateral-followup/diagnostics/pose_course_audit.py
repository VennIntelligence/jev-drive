"""Body-heading versus rear-axle interval course; never substitutes frozen gates."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np

WINDOWS={'26966':[(1,29.,46.,24.,51.)],'24240':[(1,23.,59.,18.,64.)],
         '17563':[(1,32.5,43.5,27.5,48.5),(2,78.5,90.,73.5,95.)]}
K=.010659832
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def csvout(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    hashes={};frames=[];summaries=[];verification=[]
    def record(p):hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    record(args.analysis/'frames.csv');record(args.analysis/'required-conditions.json')
    frozen={(r['route'],r['variant'],int(r['frame'])):r for r in csv.DictReader((args.analysis/'frames.csv').open())}
    index=Path(__file__).resolve().parents[1]/'pose-followup/analysis/lateral-evidence-v1/routes.csv'
    record(index)
    refs={r['route']:Path(r['source']) for r in csv.DictReader(index.open()) if r['dataset']=='G2-v4'}
    for route in WINDOWS:
        for variant in ('baseline-zero','candidate-fixed-k'):
            case=args.run_root/route/variant/'pursuit'
            for n in ('control.jsonl','validation_trace.json','route_reference.json'):record(case/n)
            controls=[json.loads(s) for s in (case/'control.jsonl').read_text().splitlines()]
            trace=json.loads((case/'validation_trace.json').read_text())
            assert [r['frame'] for r in controls]==[r['frame'] for r in trace]
            record(refs[route]);xy=np.asarray(json.loads(refs[route].read_text())['world_xy'])
            actual_reference=np.asarray(json.loads((case/'route_reference.json').read_text())['world_xy'])
            assert actual_reference.shape==xy.shape and np.allclose(actual_reference,xy,atol=.01,rtol=0)
            xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-8]]
            d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=1);arc=np.r_[0.,np.cumsum(length)]
            def projection(point):
                u=np.clip(np.sum((point-xy[:-1])*d,axis=1)/(length*length),0,1)
                foot=xy[:-1]+u[:,None]*d;i=int(np.argmin(np.sum((foot-point)**2,axis=1)))
                return arc[i]+u[i]*length[i]
            def reference_heading(s):
                interp=lambda q:np.array([np.interp(q,arc,xy[:,j]) for j in range(2)])
                tangent=interp(s+2.5)-interp(s-2.5)
                return np.arctan2(tangent[1],tangent[0])
            max_error=0.;case_rows=[]
            for i,(c,t) in enumerate(zip(controls,trace)):
                old=frozen[(route,variant,c['frame'])];s=projection(np.asarray(c['truth_xy']))
                ref=reference_heading(s);body=float(np.rad2deg(wrap(c['truth_yaw']-ref)))
                max_error=max(max_error,abs(body-float(old['heading'])),abs(s-float(old['progress'])))
                row=dict(route=route,variant=variant,frame=c['frame'],station_m=float(s),sim_time=t['sim_time'],
                    actual_speed_mps=t['speed'],low_speed_lt2=t['speed']<2.,frozen_body_error_deg=body,
                    course_valid=False,course_invalid_reason='first_frame',interval_dt_s=None,
                    interval_center_station_m=None,course_error_on_current_reference_deg=None,
                    midpoint_body_error_deg=None,midpoint_course_error_deg=None,body_minus_course_deg=None,
                    residual_model_prediction_offset_deg=None,rear_right_interval_mps=None,
                    no_lateral_model_minus_rear_right_mps=None,fixed_k_predicted_residual_mps=None)
                if i:
                    p,pc=trace[i-1],controls[i-1];dt=t['sim_time']-p['sim_time']
                    delta=np.asarray(c['truth_xy'])-pc['truth_xy'];distance=np.linalg.norm(delta)
                    row['course_invalid_reason']='gap_or_nonpositive_dt' if c['frame']!=pc['frame']+1 or dt<=0 else 'zero_displacement'
                    if c['frame']==pc['frame']+1 and dt>0 and np.isfinite(delta).all() and distance>1e-8:
                        course=np.arctan2(delta[1],delta[0]);middle=pc['truth_yaw']+.5*wrap(c['truth_yaw']-pc['truth_yaw'])
                        center=.5*(np.asarray(c['truth_xy'])+pc['truth_xy']);ms=projection(center);mr=reference_heading(ms)
                        right=np.array([-np.sin(middle),np.cos(middle)]);forward=np.array([np.cos(middle),np.sin(middle)])
                        velocity=delta/dt;v=.5*(c['speed_mps']+pc['speed_mps']);omega=-.5*(c['yaw_rate_rps']+pc['yaw_rate_rps'])
                        expected=K*v*v*omega
                        row.update(course_valid=True,course_invalid_reason=None,interval_dt_s=dt,
                            interval_center_station_m=float(ms),course_error_on_current_reference_deg=float(np.rad2deg(wrap(course-ref))),
                            midpoint_body_error_deg=float(np.rad2deg(wrap(middle-mr))),
                            midpoint_course_error_deg=float(np.rad2deg(wrap(course-mr))),
                            body_minus_course_deg=float(np.rad2deg(wrap(middle-course))),
                            residual_model_prediction_offset_deg=float(np.rad2deg(np.arctan2(expected,velocity@forward))),
                            rear_right_interval_mps=float(velocity@right),no_lateral_model_minus_rear_right_mps=float(-velocity@right),
                            fixed_k_predicted_residual_mps=float(expected))
                case_rows.append(row)
            assert max_error<1e-8
            verification.append(dict(route=route,variant=variant,frames=len(case_rows),frozen_heading_station_max_error=max_error))
            frames.extend(case_rows)
            for segment,core_start,core_end,start,end in WINDOWS[route]:
                selections={'window':[r for r in case_rows if start<=r['station_m']<=end],
                    'entry':[r for r in case_rows if start-5<=r['station_m']<start],
                    'pad_before_core':[r for r in case_rows if start<=r['station_m']<core_start],
                    'core':[r for r in case_rows if core_start<=r['station_m']<=core_end],
                    'exit':[r for r in case_rows if core_end<r['station_m']<=end],
                    'post10m':[r for r in case_rows if end<r['station_m']<=end+10]}
                for band,rr in selections.items():
                    m=dict(route=route,variant=variant,segment=segment,band=band,n=len(rr),
                           invalid_course_count=sum(not r['course_valid'] for r in rr),low_speed_count=sum(r['low_speed_lt2'] for r in rr))
                    for field in ('frozen_body_error_deg','course_error_on_current_reference_deg','midpoint_body_error_deg',
                                  'midpoint_course_error_deg','body_minus_course_deg','residual_model_prediction_offset_deg',
                                  'no_lateral_model_minus_rear_right_mps','fixed_k_predicted_residual_mps'):
                        a=np.array([r[field] for r in rr if r[field] is not None])
                        m[field+'_n']=len(a);m[field+'_mean']=float(a.mean()) if len(a) else None
                        m[field+'_rms']=float(np.sqrt(np.mean(a*a))) if len(a) else None
                        m[field+'_p95_abs']=float(np.percentile(abs(a),95)) if len(a) else None
                    valid=[r for r in rr if r['course_valid']]
                    pred=np.array([r['fixed_k_predicted_residual_mps'] for r in valid]);target=np.array([r['no_lateral_model_minus_rear_right_mps'] for r in valid])
                    m['fixed_k_residual_rms_mps']=float(np.sqrt(np.mean((pred-target)**2))) if len(valid) else None
                    m['fixed_k_sign_agreement']=float(np.mean(pred*target>0)) if len(valid) else None
                    summaries.append(m)
    baseline=next(m for m in summaries if (m['route'],m['variant'],m['band'])==('26966','baseline-zero','window'))
    threshold=baseline['frozen_body_error_deg_p95_abs']+1.
    exceed=[dict(r) for r in frames if r['route']=='26966' and r['variant']=='candidate-fixed-k' and 24<=r['station_m']<=51 and abs(r['frozen_body_error_deg'])>threshold]
    for r in exceed:r['frozen_body_guard_threshold_deg']=threshold
    csvout(args.out/'frames.csv',frames);csvout(args.out/'phase-metrics.csv',summaries);csvout(args.out/'right-threshold-exceedances.csv',exceed)
    record(Path(__file__));shutil.copyfile(__file__,str(args.out/'recompute.py'))
    manifest=dict(verification=verification,inputs=hashes,coefficient_not_refitted=K,
        body_gate_unchanged=True,right_baseline_p95_plus1_deg=threshold,right_exceeding_frames=len(exceed),
        convention='CARLA right-positive angles. Frozen body error is current yaw/current5m-centered-reference. Interval course is atan2(current rearXY - previous rearXY), naturally centered between ticks. Aligned body and reference use wrapped midpoint yaw and nearest projection of midpoint rearXY. Course/current-reference metric retained separately to expose timing mismatch.',
        retention='All fixed-window samples retained. Invalid first/gap/zero-displacement course remains in full frames with explicit reason; per-metric count shown. No low-speed rejection. No added performance gate or substitution for failed frozen body-heading condition.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.out/'outputs-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(dict(right_threshold=threshold,exceedance_count=len(exceed),ranges=[(r['frame'],r['station_m'],r['frozen_body_error_deg'],r['midpoint_course_error_deg']) for r in exceed]),indent=2))

if __name__=='__main__':main()
