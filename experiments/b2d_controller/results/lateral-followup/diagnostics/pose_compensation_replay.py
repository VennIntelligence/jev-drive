"""Frozen-k sensor-only PoseFilter counterfactual; archived physical motion is unchanged."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

K=0.010659832  # Parent-frozen rounded coefficient: never fitted in this script.
WINDOWS={'26966':[(1,24.,51.)],'24240':[(1,18.,64.)],
         '17563':[(1,27.5,48.5),(2,73.5,95.)]}
APPROACH={'26966':[(0.,19.)],'24240':[(0.,13.)],
          '17563':[(0.,22.5),(53.5,68.5)]}


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=args.run_root/'provenance/source/scripts/b2d_controller_adapter.py'
    spec=importlib.util.spec_from_file_location('frozen_pose_adapter',str(source))
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    hashes={};frames=[];summaries=[];checks=[]
    def read(path,jsonl=False):
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text().splitlines()] if jsonl else json.loads(path.read_text())
    class ReplayFilter(adapter.PoseFilter):
        def __init__(self,*a,coefficient,**kw):
            self.coefficient=coefficient
            super().__init__(*a,**kw)
        def reset(self):
            super().reset();self.previous_world_gyro=None
        def update(self,gps,compass,speed,world_yaw_rate,timestamp):
            correction=np.zeros(2)
            if self.t is not None and self.coefficient!=0:
                dt=timestamp-self.t
                assert 0<dt<=.2 and np.isfinite([speed,world_yaw_rate,self.previous_world_gyro]).all()
                v=.5*(speed+self.previous_speed)
                omega=.5*(world_yaw_rate+self.previous_world_gyro)
                middle=self.yaw+.5*world_yaw_rate*dt
                right=np.array([-np.sin(middle),np.cos(middle)])
                correction=-self.coefficient*v*v*omega*dt*right
                # Inject into the predicted position before the unchanged GNSS
                # correction, so this term is multiplied by (1-gain) exactly once.
                self.xy+=correction
            xy,yaw=super().update(gps,compass,speed,world_yaw_rate,timestamp)
            self.previous_world_gyro=world_yaw_rate
            return xy,yaw,correction
    for route in WINDOWS:
        for variant in ('baseline-max','short-max'):
            p=args.run_root/route/variant/'pursuit'
            ref=read(p/'route_reference.json');control=read(p/'control.jsonl',True)
            motion=read(p/'motion.jsonl',True);trace=read(p/'validation_trace.json')
            assert [r['frame'] for r in control]==[r['frame'] for r in motion]==[r['frame'] for r in trace]
            settings=ref['adapter']
            projector=adapter.GPSProjector(ref['gps_lat_lon'],ref['world_xy'])
            pars=dict(rear_axle_offset_m=ref['rear_axle_offset_m'],gnss_x_m=settings['gnss_x_m'],
                      gnss_gain=settings['pose_gnss_gain'],heading_gain=settings['pose_heading_gain'])
            filters=[ReplayFilter(projector,coefficient=k,**pars) for k in (0.,K)]
            verification=dict(route=route,variant=variant,ticks=len(control),zero_pose_unequal_values=0,
                zero_yaw_unequal_values=0,zero_raw_unequal_values=0,compensated_heading_unequal_values=0)
            rows=[]
            for c,m,t in zip(control,motion,trace):
                sensors=m['sensors'];gps=sensors['GPS']['data'];imu=sensors['IMU']['data']
                speed=adapter.controller_speed(sensors['SPEED']['data']['speed'])
                assert speed==c['speed_mps'] and -float(imu[5])==c['yaw_rate_rps']
                baseline,changed=[f.update(gps,float(imu[6]),speed,float(imu[5]),m['sim_time']) for f in filters]
                xy,yaw,_=baseline;cx,cy,correction=changed
                verification['zero_pose_unequal_values']+=int(np.count_nonzero(xy!=c['pose_xy']))
                verification['zero_yaw_unequal_values']+=int(yaw!=c['pose_yaw'])
                verification['zero_raw_unequal_values']+=int(np.count_nonzero(filters[0].raw_xy!=c['raw_pose_xy']))
                verification['compensated_heading_unequal_values']+=int(yaw!=cy)
                heading=t['path_heading_rad'];left=np.array([np.sin(heading),-np.cos(heading)])
                actual=np.asarray(t['truth_xy']);e=xy-actual;ce=cx-actual
                rows.append(dict(route=route,variant=variant,frame=c['frame'],sim_time=c['sim_time'],
                    station_m=t['progress_m'],speed_mps=speed,
                    baseline_x_m=float(xy[0]),baseline_y_m=float(xy[1]),candidate_x_m=float(cx[0]),candidate_y_m=float(cx[1]),
                    truth_x_m=float(actual[0]),truth_y_m=float(actual[1]),raw_x_m=float(filters[0].raw_xy[0]),raw_y_m=float(filters[0].raw_xy[1]),
                    baseline_position_error_m=float(np.linalg.norm(e)),candidate_position_error_m=float(np.linalg.norm(ce)),
                    baseline_left_error_m=float(e@left),candidate_left_error_m=float(ce@left),
                    baseline_heading_error_deg=float(np.rad2deg(adapter.wrap(yaw-c['truth_yaw']))),
                    candidate_heading_error_deg=float(np.rad2deg(adapter.wrap(cy-c['truth_yaw']))),
                    correction_world_x_m=float(correction[0]),correction_world_y_m=float(correction[1])))
            assert all(verification[k]==0 for k in verification if k.endswith('unequal_values'))
            frames.extend(rows);checks.append(verification)
            bands=[('full',rows),('fixed_straight_approach_and_gap',[r for r in rows if any(a<=r['station_m']<=b for a,b in APPROACH[route])]),
                   ('final5s',[r for r in rows if r['sim_time']>=rows[-1]['sim_time']-5.])]
            bands += [('window_%d'%segment,[r for r in rows if a<=r['station_m']<=b]) for segment,a,b in WINDOWS[route]]
            for band,rr in bands:
                summary=dict(route=route,variant=variant,band=band,n=len(rr))
                for mode in ('baseline','candidate'):
                    for field in ('position_error_m','left_error_m','heading_error_deg'):
                        a=np.array([r[mode+'_'+field] for r in rr]);key=mode+'_'+field
                        summary[key+'_rms']=float(np.sqrt(np.mean(a*a)))
                        summary[key+'_p95_abs']=float(np.percentile(abs(a),95))
                        summary[key+'_max_abs']=float(max(abs(a)))
                        summary[key+'_mean']=float(a.mean())
                summary['position_rms_change_m']=summary['candidate_position_error_m_rms']-summary['baseline_position_error_m_rms']
                summaries.append(summary)
    for p in (source,Path(__file__)):
        hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_csv(args.out/'frames.csv',frames);write_csv(args.out/'metrics.csv',summaries)
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    manifest=dict(coefficient=K,fit_performed=False,inputs=hashes,
        interval='Use prior/current sensor signed speed average squared and prior/current world gyro average. Direction uses previous estimated yaw + half current gyro*dt, matching unchanged propagation midpoint. No future samples or truth input to either filter.',
        injection='-k*v^2*omega*right*dt into predicted XY before original .05 GNSS correction; original yaw and all other filter updates unchanged.',
        baseline_equivalence='Raw original GPS/IMU/SPEED plus original route-derived GPS projector; zero coefficient pose/yaw/raw compared exactly to archived controls.',
        bands=dict(windows=WINDOWS,fixed_straight_approach_and_gap=APPROACH,final5s='All final five seconds, no speed selection'),
        boundary='Open-loop localization replay on existing physical trajectories; no new controller/route-adapter commands, no closed-loop driving benefit claim. Straight guard is predeclared station approach/gap, not error-selected samples.')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copyfile(str(source),str(args.out/'frozen-adapter.py'));shutil.copyfile(__file__,str(args.out/'recompute.py'))
    (args.out/'outputs-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
