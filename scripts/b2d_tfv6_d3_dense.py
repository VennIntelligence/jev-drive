"""Rebuild evaluator-dense routes offline and validate them on completed W2 A/B paths."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from b2d_route import add_bench2drive_to_path

os.environ.setdefault('DATA_DIR','/data')
add_bench2drive_to_path(Path('/data/runs/b2d/tfv6-repro/runtime/Bench2Drive'))

import carla
import numpy as np
from leaderboard.utils.route_manipulation import downsample_route, interpolate_trajectory
from leaderboard.utils.route_parser import RouteParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from b2d_run import Server
from b2d_tfv6_campaign import DEV10, HOLDOUT
from lead.common.common_utils import convert_gps_to_carla, find_gps_ref

ROOT=Path('/data/runs/b2d/tfv6-d3')
OUT=ROOT/'dense'
REPO=Path(__file__).resolve().parents[1]
TABLE=REPO/'todos/2026-09-23-tfv6-controller/results/diagnosis/d3/dense-validation.csv'
ROUTES=(('2','2084',HOLDOUT),('2','27529',HOLDOUT),('1','2091',DEV10))
OLD=Path('/data/runs/b2d/tfv6-w2/formal')


def closest_polyline(points,line):
    """Exact XY distance to the piecewise-linear dense route."""
    points=np.asarray(points,dtype=float)
    line=np.asarray(line,dtype=float)
    a=line[:-1];v=line[1:]-a;den=np.maximum(np.sum(v*v,axis=1),1e-12)
    result=[]
    for chunk in np.array_split(points,max(1,math.ceil(len(points)/256))):
        if not len(chunk):continue
        rel=chunk[:,None,:]-a[None,:,:]
        t=np.clip(np.sum(rel*v[None,:,:],axis=2)/den[None,:],0,1)
        proj=a[None,:,:]+t[:,:,None]*v[None,:,:]
        result.extend(np.min(np.linalg.norm(chunk[:,None,:]-proj,axis=2),axis=1))
    return np.asarray(result)


def fit_planner_transform(planner_xy,world_xy):
    """Fit a similarity, then require its scale/rotation to be identity-like."""
    p=np.asarray(planner_xy,dtype=float);w=np.asarray(world_xy,dtype=float)
    pc=p.mean(axis=0);wc=w.mean(axis=0)
    u,s,vt=np.linalg.svd((p-pc).T@(w-wc))
    rotation=u@vt
    if np.linalg.det(rotation)<0:
        u[:,-1]*=-1;rotation=u@vt
    scale=float(np.sum((p-pc)@rotation*(w-wc))/np.sum((p-pc)**2))
    translation=wc-scale*pc@rotation
    residual=np.linalg.norm(scale*p@rotation+translation-w,axis=1)
    angle=math.atan2(rotation[0,1],rotation[0,0])
    if abs(scale-1)>1e-3 or abs(angle)>1e-3 or np.max(residual)>=.1:
        raise ValueError(f'Planner/world alignment failed: scale={scale}, angle={angle}, max_residual={max(residual)}')
    return {'scale':scale,'rotation':rotation.tolist(),'rotation_rad':angle,
            'translation_xy':translation.tolist(),'max_residual_m':float(np.max(residual))}


def completed_ab_validation(level,route,dense_xy):
    rows=[]
    for seed in range(3):
        for arm in 'AB':
            p=OLD/f'level{level}'/'cases'/level/f'route-{route}'/f'seed-{seed}'/arm/'done.json'
            done=json.loads(p.read_text())
            if done['official_status'] not in ('Completed','Perfect'):
                continue
            frame_path=Path(done['run_dir']).parent/'frames.jsonl'
            xy=[]
            with frame_path.open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row.get('truth'):xy.append(row['truth']['location'][:2])
            distance=closest_polyline(xy,dense_xy)
            within=float(np.mean(distance<=2.))
            rows.append({'level':level,'route':route,'seed':seed,'arm':arm,
                         'ticks':len(distance),'within_2m_fraction':within,
                         'max_distance_m':float(np.max(distance)),
                         'p99_distance_m':float(np.percentile(distance,99))})
    if not rows:
        raise ValueError(f'No completed W2 A/B cases to validate route {route}')
    if any(r['within_2m_fraction']<.99 for r in rows):
        bad=[r for r in rows if r['within_2m_fraction']<.99]
        raise ValueError(f'Dense route validation <99% within 2m for {route}: {bad}')
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    server=Server(98,OUT/'server','Epic',gpu_rank=0)
    server.start()
    client=carla.Client('127.0.0.1',server.port)
    client.set_timeout(60.)
    all_rows=[];summary=[]
    try:
        for level,route_id,xml in ROUTES:
            config=RouteParser.parse_routes_file(str(xml),route_id)[0]
            world=client.load_world(config.town)
            CarlaDataProvider.set_client(client)
            CarlaDataProvider.set_world(world)
            gps,dense=interpolate_trajectory(config.keypoints,hop_resolution=1.0)
            dense_json=[{'xyz':[float(t.location.x),float(t.location.y),float(t.location.z)],
                         'command':int(command)} for t,command in dense]
            indices=downsample_route(dense,50)
            sparse=[dense[i] for i in indices];sparse_gps=[gps[i] for i in indices]
            lat_ref,lon_ref=find_gps_ref(sparse,sparse_gps)
            planner=np.asarray([convert_gps_to_carla(
                np.asarray([x['lat'],x['lon'],x['z']]),lat_ref,lon_ref)[:2]
                for x,_ in sparse_gps])
            world_xy=np.asarray([[t.location.x,t.location.y] for t,_ in sparse])
            transform=fit_planner_transform(planner,world_xy)
            dense_xy=np.asarray([r['xyz'][:2] for r in dense_json])
            checks=completed_ab_validation(level,route_id,dense_xy)
            all_rows.extend(checks)
            package={'level':level,'route':route_id,'town':config.town,
                     'hop_resolution_m':1.0,'downsample_distance_m':50,
                     'dense':dense_json,'agent_sparse_dense_indices':indices,
                     'agent_sparse_planner_xy':planner.tolist(),
                     'agent_sparse_world_xy':world_xy.tolist(),
                     'planner_to_world':transform}
            (OUT/f'{level}-{route_id}.json').write_text(json.dumps(package)+'\n')
            summary.append({'route':route_id,'town':config.town,'dense_points':len(dense_json),
                            'sparse_points':len(indices),'max_completed_ab_distance_m':max(x['max_distance_m'] for x in checks),
                            'min_completed_ab_within_2m':min(x['within_2m_fraction'] for x in checks),
                            'alignment_max_residual_m':transform['max_residual_m']})
            print(summary[-1],flush=True)
    finally:
        server.stop()
    TABLE.parent.mkdir(parents=True,exist_ok=True)
    with TABLE.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(all_rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
