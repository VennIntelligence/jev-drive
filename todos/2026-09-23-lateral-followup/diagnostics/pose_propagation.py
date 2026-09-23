"""Read-only rear-axle interval kinematics and exact pose-error recurrence audit.

Old trace lacks a direct actor velocity vector. Position differences are labelled
interval finite-difference estimates throughout; no unobserved vector is invented.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    hashes, all_rows, summaries, checks = {}, [], [], []
    def read(path, jsonl=False):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(x) for x in path.read_text().splitlines()] if jsonl else json.loads(path.read_text())
    windows = {'26966': [(1,24.,51.)], '24240': [(1,18.,64.)],
               '17563': [(1,27.5,48.5),(2,73.5,95.)]}
    for route in windows:
        for variant in ('baseline-max','short-max'):
            case = args.run_root / route / variant / 'pursuit'
            trace = read(case/'validation_trace.json')
            controls = read(case/'control.jsonl', True)
            conf = read(Path(read(case/'agent_config.json')['controller_config']))
            rear, gain = conf['rear_axle_offset_m'], conf.get('pose_gnss_gain',.05)
            assert [r['frame'] for r in trace] == [r['frame'] for r in controls]
            assert all('velocity_mps' not in r['plant_kinematics'] for r in trace)
            initial = np.asarray(controls[0]['pose_xy'])-trace[0]['truth_xy']
            model_term, gnss_term, initial_term = np.zeros(2), np.zeros(2), initial.copy()
            rows, error_residuals, truth_residuals = [], [], []
            for i in range(1,len(trace)):
                t,p,c,cp = trace[i],trace[i-1],controls[i],controls[i-1]
                dt = t['sim_time']-p['sim_time']; cdt=c['sim_time']-cp['sim_time']
                assert t['frame']==p['frame']+1 and 0 < dt <= .2 and abs(dt-cdt)<1e-7
                k,kp=t['plant_kinematics'],p['plant_kinematics']
                loc,prevloc=np.asarray(k['location_m']),np.asarray(kp['location_m'])
                f,fp=np.asarray(k['forward_vector']),np.asarray(kp['forward_vector'])
                right=np.asarray(k['right_vector'])
                lever,prevlever=rear*f,rear*fp
                actor_fd=(loc-prevloc)/dt
                rear_fd=(loc+lever-prevloc-prevlever)/dt
                omega,op=np.deg2rad(k['angular_velocity_deg_s']),np.deg2rad(kp['angular_velocity_deg_s'])
                rear_omega=actor_fd+.5*(np.cross(omega,lever)+np.cross(op,prevlever))
                dy=np.asarray(t['truth_xy'])-p['truth_xy']
                truth_residuals.append(np.max(np.abs((loc+lever)[:2]-t['truth_xy'])))
                # Controller logs yaw_rate left-positive; PoseFilter uses its negation.
                middle=cp['pose_yaw']-.5*c['yaw_rate_rps']*cdt
                distance=.5*(c['speed_mps']+cp['speed_mps'])*cdt
                pred=distance*np.array([np.cos(middle),np.sin(middle)])
                error=np.asarray(c['pose_xy'])-t['truth_xy']
                raw=np.asarray(c['raw_pose_xy'])-t['truth_xy']
                forcing=pred-dy
                model_term=(1-gain)*(model_term+forcing)
                gnss_term=(1-gain)*gnss_term+gain*raw
                initial_term=(1-gain)*initial_term
                error_residuals.append(np.max(np.abs(error-model_term-gnss_term-initial_term)))
                yaw=c['truth_yaw']; bodyright=np.array([-np.sin(yaw),np.cos(yaw)])
                heading=t['path_heading_rad']; refleft=np.array([np.sin(heading),-np.cos(heading)])
                turn=(c['truth_yaw']-cp['truth_yaw']+np.pi)%(2*np.pi)-np.pi
                true_middle=cp['truth_yaw']+.5*turn
                middle_forward=np.array([np.cos(true_middle),np.sin(true_middle)])
                middle_right=np.array([-np.sin(true_middle),np.cos(true_middle)])
                true_noslip=distance*middle_forward
                interval_right=np.r_[middle_right,0.]
                interval_forward=np.r_[middle_forward,0.]
                rows.append(dict(route=route,variant=variant,frame=t['frame'],station_m=t['progress_m'],dt_s=dt,
                    actor_interval_right_mps=float(actor_fd@interval_right),rear_interval_right_mps=float(rear_fd@interval_right),
                    rear_omega_right_mps=float(rear_omega@interval_right),
                    omega_vs_rear_fd_right_mps=float((rear_omega-rear_fd)@interval_right),
                    actor_interval_forward_mps=float(actor_fd@interval_forward),rear_interval_forward_mps=float(rear_fd@interval_forward),
                    actor_fd_norm_minus_endpoint_mean_speed_mps=float(np.linalg.norm(actor_fd)-.5*(t['speed']+p['speed'])),
                    prediction_minus_rear_right_mps=float((pred-dy)@bodyright/dt),
                    prediction_minus_rear_left_m=float(forcing@refleft),
                    true_heading_model_minus_rear_right_mps=float((true_noslip-dy)@middle_right/dt),
                    true_heading_model_minus_rear_forward_mps=float((true_noslip-dy)@middle_forward/dt),
                    true_heading_model_minus_rear_right_m=float((true_noslip-dy)@middle_right),
                    true_heading_noslip_minus_rear_left_m=float((true_noslip-dy)@refleft),
                    pose_error_left_m=float(error@refleft),raw_pose_error_left_m=float(raw@refleft),
                    model_history_error_left_m=float(model_term@refleft),
                    gnss_history_error_left_m=float(gnss_term@refleft),
                    initial_history_error_left_m=float(initial_term@refleft)))
            assert max(error_residuals)<1e-8
            checks.append(dict(route=route,variant=variant,frames=len(trace),
                direct_actor_velocity_vector_archived=False,pose_recurrence_max_error_m=max(error_residuals),
                rear_origin_max_error_m=max(truth_residuals),gain=gain))
            all_rows.extend(rows)
            for segment,start,end in windows[route]:
                selected=[r for r in rows if start<=r['station_m']<=end]
                summary=dict(route=route,variant=variant,segment=segment,start_m=start,end_m=end,n=len(selected))
                fields=[k for k in rows[0] if k.endswith(('_mps','_m')) and k!='station_m']
                for field in fields:
                    a=np.array([r[field] for r in selected])
                    summary[field+'_mean']=float(a.mean())
                    summary[field+'_rms']=float(np.sqrt(np.mean(a*a)))
                model=np.array([r['model_history_error_left_m'] for r in selected])
                actual=np.array([r['pose_error_left_m'] for r in selected])
                summary['model_pose_left_sign_agreement_fraction']=float(np.mean(model*actual>0))
                summary['model_pose_left_correlation']=float(np.corrcoef(model,actual)[0,1])
                summary['constant_forcing_lag_scale_s']=(1-gain)/gain*float(np.mean([r['dt_s'] for r in selected]))
                summaries.append(summary)
    source=args.run_root/'provenance/source/scripts'
    for name in ('b2d_controller_validate.py','b2d_controller_adapter.py','b2d_agent.py'):
        path=source/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    speed_source=Path('/data/third_party/Bench2Drive/leaderboard/leaderboard/envs/sensor_interface.py')
    hashes[str(speed_source)]=hashlib.sha256(speed_source.read_bytes()).hexdigest()
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_csv(args.out/'frames.csv',all_rows);write_csv(args.out/'windows.csv',summaries)
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    (args.out/'inputs-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    shutil.copyfile(__file__,str(args.out/'recompute.py'))
    (args.out/'outputs-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
