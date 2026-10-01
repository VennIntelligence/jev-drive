"""Offline contract fixture from raw sensors and PoseFilter only; no CARLA imports."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from pose_contract import audit_pose,FIXED_K,KEY


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise SystemExit('Refusing existing output')
    spec=importlib.util.spec_from_file_location('pose_production_fixture',args.adapter)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inputs={};cases=[]
    def record(p):inputs[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (args.adapter,Path(__file__),Path(__file__).with_name('pose_contract.py')):record(p)
    for route in ('26966','24240','17563'):
        directory=args.raw/route/'baseline-max/pursuit'
        for name in ('control.jsonl','motion.jsonl','route_reference.json'):record(directory/name)
        original=[json.loads(x) for x in (directory/'control.jsonl').read_text().splitlines()]
        motion=[json.loads(x) for x in (directory/'motion.jsonl').read_text().splitlines()]
        ref=json.loads((directory/'route_reference.json').read_text());settings=ref['adapter']
        for variant,k in (('baseline-zero',0.),('candidate-fixed-k',FIXED_K)):
            projector=module.GPSProjector(ref['gps_lat_lon'],ref['world_xy'])
            pose=module.PoseFilter(projector,settings['rear_axle_offset_m'],gnss_x_m=settings['gnss_x_m'],
                gnss_gain=settings['pose_gnss_gain'],heading_gain=settings['pose_heading_gain'],lateral_coefficient_s2_per_m=k)
            rows=copy.deepcopy(original)
            for row,m in zip(rows,motion):
                sensors=m['sensors'];imu=sensors['IMU']['data']
                xy,yaw=pose.update(sensors['GPS']['data'],imu[6],module.controller_speed(sensors['SPEED']['data']['speed']),imu[5],m['sim_time'])
                row.update(pose_xy=xy.tolist(),pose_yaw=yaw,raw_pose_xy=pose.raw_xy.tolist(),pose_status=pose.diagnostics)
            result=audit_pose(rows,motion,{KEY:k,'aim_interpolation':'linear'},variant,reference=ref)
            # Negative mutation targets otherwise valid current-schema producer output.
            broken=copy.deepcopy(rows);broken[100]['pose_status'].pop('lateral_dt_s')
            missing=audit_pose(broken,motion,{KEY:k,'aim_interpolation':'linear'},variant,reference=ref)
            cases.append(dict(route=route,variant=variant,frames=len(rows),complete=result['complete'],
                sensor_replay_complete=result['sensor_replay_complete'],errors=result['errors'],
                missing_single_field_rejected=not missing['complete'],
                max_independent_pose_difference_m=max(r.get('position_difference_m',0.) for r in result['frames'])))
    output=dict(complete=all(r['complete'] and r['missing_single_field_rejected'] for r in cases),cases=cases,inputs=inputs,
        scope='Offline PoseFilter-produced pose/status fixture only. Historical controls are placeholders, not counterfactual closed-loop controls. No truth fed to producer or independent audit; no CARLA.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(complete=output['complete'],cases=len(cases),frames=sum(r['frames'] for r in cases))))
    if not output['complete']:raise SystemExit('Contract verification failed')


if __name__=='__main__':main()
