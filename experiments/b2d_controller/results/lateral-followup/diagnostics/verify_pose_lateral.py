"""Verify production opt-in against frozen raw inputs and pre-implementation shadow."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts'))
from b2d_controller_adapter import GPSProjector,PoseFilter,controller_speed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--shadow',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    hashes={};checks=[];frames=[]
    def read(path,jsonl=False):
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(s) for s in path.read_text().splitlines()] if jsonl else json.loads(path.read_text())
    path=args.shadow/'frames.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    shadow={(r['route'],r['variant'],int(r['frame'])):r for r in csv.DictReader(path.open())}
    for route in ('26966','24240','17563'):
        for variant in ('baseline-max','short-max'):
            case=args.run_root/route/variant/'pursuit'
            ref=read(case/'route_reference.json');motion=read(case/'motion.jsonl',True);control=read(case/'control.jsonl',True)
            projector=GPSProjector(ref['gps_lat_lon'],ref['world_xy']);s=ref['adapter']
            parameters=dict(rear_axle_offset_m=ref['rear_axle_offset_m'],gnss_x_m=s['gnss_x_m'],
                            gnss_gain=s['pose_gnss_gain'],heading_gain=s['pose_heading_gain'])
            filters=[PoseFilter(projector,**parameters),PoseFilter(projector,lateral_coefficient_s2_per_m=.010659832,**parameters)]
            count=dict(route=route,variant=variant,frames=len(motion),default_pose_unequal_values=0,
                       candidate_shadow_unequal_values=0,heading_unequal_values=0,raw_pose_unequal_values=0,
                       candidate_shadow_max_abs_error_m=0.)
            for c,m in zip(control,motion):
                assert c['frame']==m['frame']
                sensors=m['sensors'];imu=sensors['IMU']['data'];speed=controller_speed(sensors['SPEED']['data']['speed'])
                b,candidate=[f.update(sensors['GPS']['data'],float(imu[6]),speed,float(imu[5]),m['sim_time']) for f in filters]
                old=shadow[(route,variant,c['frame'])]
                expected=[float(old['candidate_x_m']),float(old['candidate_y_m'])]
                count['default_pose_unequal_values']+=int(np.count_nonzero(b[0]!=c['pose_xy']))
                count['candidate_shadow_unequal_values']+=int(np.count_nonzero(candidate[0]!=expected))
                count['heading_unequal_values']+=int(b[1]!=c['pose_yaw'])+int(candidate[1]!=c['pose_yaw'])
                count['raw_pose_unequal_values']+=sum(int(np.count_nonzero(f.raw_xy!=c['raw_pose_xy'])) for f in filters)
                count['candidate_shadow_max_abs_error_m']=max(count['candidate_shadow_max_abs_error_m'],float(np.max(abs(candidate[0]-expected))))
                frames.append(dict(route=route,variant=variant,frame=c['frame'],
                    baseline_pose=b[0].tolist(),candidate_pose=candidate[0].tolist(),
                    baseline_status=filters[0].diagnostics,candidate_status=filters[1].diagnostics))
            assert all(v==0 for k,v in count.items() if 'unequal_values' in k)
            checks.append(count)
    source=args.out/'source';source.mkdir()
    for path in [REPO/'scripts'/name for name in ('b2d_controller_adapter.py','b2d_agent.py','test_b2d_controller_pose_lateral.py')]+[Path(__file__)]:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();shutil.copyfile(str(path),str(source/path.name))
    with (args.out/'frames.jsonl').open('w') as f:
        for row in frames:f.write(json.dumps(row,allow_nan=False)+'\n')
    summary=dict(total_frames=len(frames),exactly_equal=True,checks=checks,
        coefficient=.010659832,shadow=str(args.shadow),inputs=hashes,
        scope='Production PoseFilter only, using complete original GPS/IMU/SPEED. Exact default vs original log; fixed k vs independently written pre-implementation shadow. No new physical/controller execution.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.out/'outputs-sha256.json').write_text(json.dumps({str(p.relative_to(args.out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
