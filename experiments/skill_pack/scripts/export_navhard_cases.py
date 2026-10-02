"""Export existing NAVHARD inputs, cached plans and official scoring rollouts. No model inference."""
import os, json, pickle, lzma, glob
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_coords_array

D = Path(os.environ['DATA_DIR'])
O = D / 'runs/openloop_visual_review_20261002'
O.mkdir(exist_ok=True)
idx = pickle.load(open(D/'runs/navsim_zs/index/navhard_two_stage_slim.pkl','rb'))
idx = {e['token']: e for e in idx}
def scores(name):
    p = sorted(glob.glob(str(D/f'runs/navsim/eval/v2_navhard_two_stage_{name}/*/*.csv')))[-1]
    t = pd.read_csv(p)
    return t[t.token.isin(idx)].set_index('token'), p
native, native_csv = scores('opi_lb_navhard_gimm-cinque__base')
n4, n4_csv = scores('sp_n4_navhard')
def plans(p):
    z=np.load(p); return dict(zip(z['tokens'].tolist(),z['poses']))
pn=plans(D/'runs/op_lb/lb_navhard/preds/gimm-cinque__base.npz')
p4=plans(D/'runs/skill_pack/raise/n4/navhard_n4.npz')
mcache={Path(p).parent.name:p for p in glob.glob(str(D/'runs/navsim/metric_cache/v2_navhard_two_stage/*/*/*/metric_cache.pkl'))}
sampling=TrajectorySampling(num_poses=40,interval_length=.1)
sim=PDMSimulator(sampling)
def metric(df,t,k): return float(df.loc[t,k+'_stage_two'])
tokens=[t for t,e in idx.items() if e['stage']=='two' and np.linalg.norm(e['vel'][-1])>3 and t in mcache]
dac='drivable_area_compliance';ddc='driving_direction_compliance'
groups=[('both_offroad',lambda t:metric(native,t,dac)==0 and metric(n4,t,dac)==0,3),
        ('n4_regression',lambda t:metric(native,t,dac)==1 and metric(n4,t,dac)==0,1),
        ('n4_recovery',lambda t:metric(native,t,dac)==0 and metric(n4,t,dac)==1,1),
        ('direction',lambda t:metric(n4,t,dac)==1 and metric(n4,t,ddc)<1,1),
        ('control',lambda t:metric(native,t,dac)==1 and metric(n4,t,dac)==1 and metric(n4,t,ddc)==1 and n4.loc[t,'score']>.85,1)]
seen=set();cases=[]
for group,pred,count in groups:
    eligible=sorted([t for t in tokens if pred(t)], key=lambda t:(int(np.argmax(idx[t]['cmd'][-1])) if group=='both_offroad' else 0,t))
    got=0
    for t in eligible:
        e=idx[t]
        if e['log_name'] in seen: continue
        if group=='both_offroad' and int(np.argmax(e['cmd'][-1])) != got: continue
        with lzma.open(mcache[t],'rb') as f: mc=pickle.load(f)
        ego=mc.ego_state; origin=np.array(ego.rear_axle.serialize());c,s=np.cos(origin[2]),np.sin(origin[2]);R=np.array([[c,-s],[s,c]])
        local=lambda p: (np.asarray(p)-origin[:2])@R
        tracks=np.stack([get_trajectory_as_array(mc.trajectory,sampling,ego.time_point),
                        get_trajectory_as_array(transform_trajectory(Trajectory(pn[t].astype(float)),ego),sampling,ego.time_point),
                        get_trajectory_as_array(transform_trajectory(Trajectory(p4[t].astype(float)),ego),sampling,ego.time_point)])
        states=sim.simulate_proposals(tracks,ego)
        corners=state_array_to_coords_array(states,ego.car_footprint.vehicle_parameters)
        amap=mc.drivable_area_map
        area_ids=amap.get_indices_of_map_type([L.ROADBLOCK,L.INTERSECTION,L.DRIVABLE_AREA,L.CARPARK_AREA])
        lane_ids=amap.get_indices_of_map_type([L.LANE,L.LANE_CONNECTOR])
        inside=amap.points_in_polygons(corners[...,:-1,:])[area_ids].any(axis=0).all(axis=-1)
        # Skip starts already outside the map: show a subsequent departure, not an initial-state penalty.
        if not inside[:,0].all(): continue
        if group=='both_offroad' and not inside[0].all(): continue
        checks=[int(inside[1].all()),int(inside[2].all())]
        expected=[int(metric(native,t,dac)),int(metric(n4,t,dac))]
        if checks!=expected: print('DAC MISMATCH',t,checks,expected,flush=True);continue
        polys=[]
        for k in area_ids+lane_ids:
            g=amap._geometries[k]
            if g.distance(ego.car_footprint.oriented_box.geometry)>70:continue
            for p in (list(g.geoms) if hasattr(g,'geoms') else [g]):
                polys.append({'kind':'area' if k in area_ids else 'route' if amap.tokens[k] in mc.route_lane_ids else 'lane',
                              'exterior':local(p.exterior.coords).round(4).tolist(),
                              'holes':[local(h.coords).round(4).tolist() for h in p.interiors]})
        case_id=f'{len(cases)+1:02d}-{group}'
        imgs=[]
        for k,cam in enumerate(e['cams']):
            ip=Path(cam['CAM_F0']['path']);im=Image.open(ip).convert('RGB');im.thumbnail((960,540))
            fn=f'{case_id}-history-{k}.jpg';im.save(O/fn,quality=90);imgs.append(fn)
        scores_out={}
        for name,df in [('native',native),('n4',n4)]:
            scores_out[name]={'score':float(df.loc[t,'score']),**{k:metric(df,t,k) for k in [dac,ddc,'no_at_fault_collisions','ego_progress','time_to_collision_within_bound']}}
        case={'id':case_id,'group':group,'token':t,'stage':2,'log':e['log_name'],'map':e['map'],
              'command':['left','straight','right','unknown'][int(np.argmax(e['cmd'][-1]))],
              'speed':float(np.linalg.norm(e['vel'][-1])),'images':imgs,'polygons':polys,'scores':scores_out,
              'states_xy':local(states[...,:2]).round(4).tolist(),'corners_xy':local(corners).round(4).tolist(),
              'inside':inside.tolist(),'first_departure_s':[None if a.all() else round(float(np.where(~a)[0][0])*.1,1) for a in inside],
              'source_metric_cache':mcache[t],'source_camera_paths':[x['CAM_F0']['path'] for x in e['cams']]}
        cases.append(case);seen.add(e['log_name']);got+=1
        print(case_id,t,case['command'],case['first_departure_s'],flush=True)
        if got>=count:break
(O/'cases.json').write_text(json.dumps({'selection':'Descriptive examples selected from existing stage-two scores; distinct logs; speed >3 m/s; starts inside drivable area. Not representative sampling.',
    'native_csv':native_csv,'n4_csv':n4_csv,'sampling_dt':.1,'cases':cases},ensure_ascii=False))
print('DONE',len(cases),flush=True)
