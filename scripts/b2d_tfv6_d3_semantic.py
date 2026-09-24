"""World-frame D3 route telemetry and evaluator-style completion checks."""
from __future__ import annotations

import math
import numpy as np


class SemanticFailure(ValueError):
    pass


def planner_to_world(point, transform):
    p=np.asarray(point[:2],dtype=float)
    return (float(transform['scale'])*p@np.asarray(transform['rotation'])+
            np.asarray(transform['translation_xy'])).tolist()


def local_to_planner(local, origin, compass):
    c,s=math.cos(compass),math.sin(compass)
    return (np.asarray([[c,-s],[s,c]])@np.asarray(local[:2],dtype=float)+
            np.asarray(origin[:2],dtype=float)).tolist()


def route_projection(point, dense):
    p=np.asarray(point[:2],dtype=float)
    line=np.asarray(dense,dtype=float)[:,:2]
    v=line[1:]-line[:-1]
    length2=np.sum(v*v,axis=1)
    frac=np.clip(np.sum((p-line[:-1])*v,axis=1)/np.maximum(length2,1e-12),0,1)
    projected=line[:-1]+frac[:,None]*v
    distances=np.linalg.norm(projected-p,axis=1)
    segment=int(np.argmin(distances))
    return float(distances[segment]),segment,float(frac[segment])


class SemanticSequence:
    """Mirror RouteCompletionTest's 10-point forward window on dense route."""
    def __init__(self, package):
        self.package=package
        self.dense=np.asarray([p['xyz'] for p in package['dense']],dtype=float)
        self.transform=package['planner_to_world']
        self.rc_index=0
        self.target_index=-1
        self.sparse=np.asarray(package['agent_sparse_world_xy'],dtype=float)
        self.sparse_indices=package['agent_sparse_dense_indices']

    def check(self, frame):
        nav=frame['d3_nav'];step=frame['step']
        ego=np.asarray(frame['truth']['location'][:2],dtype=float)
        target_planner=nav.get('selected_target_planner_xyz')
        if target_planner is None:
            raise SemanticFailure(f'step {step}: no selected planner target')
        target=np.asarray(planner_to_world(target_planner,self.transform))
        target_dist,_,_=route_projection(target,self.dense)
        sparse_dist=np.linalg.norm(self.sparse-target,axis=1)
        sparse_pos=int(np.argmin(sparse_dist))
        if sparse_dist[sparse_pos]>.1:
            raise SemanticFailure(f'step {step}: target absent from aligned sparse plan, {sparse_dist[sparse_pos]:.3f}m')
        index=int(self.sparse_indices[sparse_pos])
        if index<self.target_index:
            raise SemanticFailure(f'step {step}: dense target index regressed {self.target_index}->{index}')
        if target_dist>1.0:
            raise SemanticFailure(f'step {step}: target {target_dist:.3f}m from evaluator dense route')
        self.target_index=index
        origin=nav['filtered_state'] if self.package.get('uses_kalman',True) else nav['noisy_state']
        local_targets={key:planner_to_world(local_to_planner(nav[key],origin,float(nav['compass_rad'])),self.transform)
                       for key in ('target_point_previous','target_point','target_point_next')}
        local_dist,_,_=route_projection(local_targets['target_point'],self.dense)
        if local_dist>1.0:
            raise SemanticFailure(f'step {step}: TFv6 local target {local_dist:.3f}m from evaluator dense route')
        if np.linalg.norm(np.asarray(local_targets['target_point'])-target)>1.0:
            raise SemanticFailure(f'step {step}: TFv6 target differs from selected planner target')
        old_rc=self.rc_index
        for i in range(old_rc,min(old_rc+11,len(self.dense))):
            forward=self.dense[min(i+1,len(self.dense)-1),:2]-self.dense[max(i-1,0),:2]
            if np.dot(ego-self.dense[i,:2],forward)>0:
                self.rc_index=i
        ego_dist,_,_=route_projection(ego,self.dense)
        if self.rc_index>old_rc and ego_dist>3.0:
            raise SemanticFailure(f'step {step}: RC advanced {old_rc}->{self.rc_index} with ego {ego_dist:.3f}m off dense route')
        return {'step':step,'ego_world_xy':ego.tolist(),'ego_dense_distance_m':ego_dist,
                'rc_dense_index':self.rc_index,'rc_increasing':self.rc_index>old_rc,
                'selected_target_world_xy':target.tolist(),'target_dense_index':index,
                'target_dense_distance_m':target_dist,'tfv6_target_dense_distance_m':local_dist,
                'tfv6_target_world_xy':local_targets['target_point'],
                'tfv6_previous_world_xy':local_targets['target_point_previous'],
                'tfv6_next_world_xy':local_targets['target_point_next'],
                'kalman_world_xy':planner_to_world(nav['filtered_state'],self.transform),
                'gps_world_xy':planner_to_world(nav['noisy_state'],self.transform)}
