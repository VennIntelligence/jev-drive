"""Current-state privileged arbitration and evaluation geometry; CARLA agent compatible (Python 3.8).

No scenario registry, scripted future, model input, or model weight is accessed here.
Protocol: experiments/b2d_privileged/plans/2026-10-01-b2d-privileged-ceiling.md.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/op_closed_loop/lib", "scripts",)]
import json
import math
from pathlib import Path
import time

import numpy as np

ARMS = ("drive", "pjunc", "pbyp", "pbypgap", "pred", "pall")
PARAMS = dict(snapshot_s=.20, radius=100., horizon=5., margin=.5, clear_s=.8,
              release_s=2., static_speed=.2, static_s=2., obstacle_m=50.,
              merge_m=20., enter_before_m=20., transition_m=15., return_after_m=8.,
              gap_margin_s=2., gap_min_m=15., tl_m=50., tl_margin=.5)


def project(points, path, offset=0.):
    """Exact nearest polyline segment, vectorized over query points."""
    points, path = np.asarray(points, float).reshape(-1, 2), np.asarray(path, float)
    delta = np.diff(path, axis=0)
    length = np.linalg.norm(delta, axis=1)
    w = points[:, None] - path[None, :-1]
    fraction = np.clip(np.einsum("nki,ki->nk", w, delta) / np.maximum(length**2, 1e-12), 0, 1)
    closest = path[None, :-1] + fraction[..., None] * delta
    distance2 = ((points[:, None] - closest)**2).sum(-1)
    index = distance2.argmin(1)
    j = np.arange(len(points))
    unit = delta[index] / np.maximum(length[index, None], 1e-12)
    normal = np.stack([-unit[:, 1], unit[:, 0]], -1)
    signed = ((points - closest[j, index]) * normal).sum(-1)
    arc = np.r_[0., np.cumsum(length)]
    return offset + arc[index] + fraction[j, index] * length[index], signed, unit


def ego_boxes(rear,observed_yaw):
    """Future body center follows the future tangent of the rear-axle trajectory."""
    rear=np.asarray(rear,float)
    delta=np.gradient(rear,axis=0)
    headings=np.where(np.linalg.norm(delta,axis=1)>1e-3,np.arctan2(delta[:,1],delta[:,0]),observed_yaw)
    forward=np.stack([np.cos(headings),np.sin(headings)],-1)
    return rear+1.3886*forward,headings


def overlap(a, ah, ae, b, bh, be, margin=.5):
    """Broadcasted oriented-rectangle SAT, including a separating-axis margin."""
    a, b, ah, bh, ae, be = [np.asarray(v, float) for v in (a, b, ah, bh, ae, be)]
    af = np.stack([np.cos(ah), np.sin(ah)], -1)
    al = np.stack([-np.sin(ah), np.cos(ah)], -1)
    bf = np.stack([np.cos(bh), np.sin(bh)], -1)
    bl = np.stack([-np.sin(bh), np.cos(bh)], -1)
    valid = np.ones(np.broadcast_shapes(a.shape[:-1], b.shape[:-1]), bool)
    for axis in (af, al, bf, bl):
        ra = np.abs((af*axis).sum(-1))*ae[..., 0] + np.abs((al*axis).sum(-1))*ae[..., 1]
        rb = np.abs((bf*axis).sum(-1))*be[..., 0] + np.abs((bl*axis).sum(-1))*be[..., 1]
        valid &= np.abs(((b-a)*axis).sum(-1)) <= ra+rb+margin
    return valid


def corners(p, yaw, ext):
    signs = np.array([[-1,-1],[-1,1],[1,1],[1,-1]], float)
    rot = np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    return np.asarray(p) + (signs*np.asarray(ext)) @ rot.T


def rectangle_gap(a, ah, ae, b, bh, be):
    if overlap(a, ah, ae, b, bh, be, margin=0.): return 0.
    ac, bc = corners(a, ah, ae), corners(b, bh, be)
    distances = []
    for p, poly in ((ac,bc),(bc,ac)):
        for left,right in zip(poly,np.roll(poly,-1,axis=0)):
            d=right-left
            u=np.clip(((p-left)*d).sum(-1)/max(float(d@d),1e-12),0,1)
            distances.append(np.linalg.norm(p-left-u[:,None]*d,axis=1).min())
    return float(min(distances))


def visibility(actor, hero):
    """Native-camera bounding-box visibility upper bound; intentionally no occlusion test."""
    import zeroshot_rigs as rigs
    c = np.asarray(actor["corners"],float)
    origin = np.asarray(hero["xyz"],float)
    delta=c-origin
    local=delta@np.asarray(hero["rotation_matrix"],float)
    # Mount is in the rear-axle rig; hero xyz is the actor origin.
    cam=np.array([rigs.OP_MOUNT_RIG[0]+rigs.REAR_AXLE_X,0.,rigs.OP_MOUNT_RIG[2]])
    ray=local-cam
    ray=ray[ray[:,0]>.05]
    out={}
    w,h=rigs.OP_CAMERA_WH
    for name,focal in rigs.OP_FOCAL.items():
        if not len(ray): out[name]=[False,0.,0.,0.];continue
        u=w/2+focal*ray[:,1]/ray[:,0]; v=h/2-focal*ray[:,2]/ray[:,0]
        width=max(0.,min(float(u.max()),w)-max(float(u.min()),0.))
        height=max(0.,min(float(v.max()),h)-max(float(v.min()),0.))
        area=width*height
        out[name]=[bool(area>=4.),round(width,2),round(height,2),round(area,2)]
    return out


class Privileged:
    def __init__(self, agent, arm):
        assert arm in ARMS, arm
        self.agent,self.arm=agent,arm
        self.junction=arm in ("pjunc","pall")
        self.bypass=arm in ("pbyp","pbypgap","pall")
        self.gap_check=arm in ("pbypgap","pall")
        self.red=arm in ("pred","pall")
        self.last=-1e9;self.actors=[];self.static_since={};self.actor_cache={}
        self.hold=False;self.clear_since=None;self.release_until=-1e9;self.light_hold=False;self.junction_stop_s=None
        self.bypass_state=None;self.meta={};self.sensor=None;self.prepared=False
        root=Path(agent.out)
        self.scene=(root/"privileged.jsonl").open("w",buffering=1)
        self.contacts=(root/"contacts.jsonl").open("w",buffering=1)

    def prepare(self):
        import carla
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider as CDP
        self.world=CDP.get_world();self.map=CDP.get_map();self.hero=CDP.get_hero_actor()
        r=self.agent.route
        self.flags=[];self.jids=[]
        for p in r.xy:
            w=self.map.get_waypoint(carla.Location(x=float(p[0]),y=float(p[1])),project_to_road=True)
            self.flags.append(bool(w.is_junction));self.jids.append(int(w.junction_id) if w.is_junction else -1)
        self.flags=np.asarray(self.flags);self.jids=np.asarray(self.jids)
        self.sensor=self.world.spawn_actor(self.world.get_blueprint_library().find("sensor.other.collision"),
                                          carla.Transform(),attach_to=self.hero)
        def contact(e):
            impulse=e.normal_impulse
            self.contacts.write(json.dumps(dict(frame=e.frame,t=e.timestamp,id=e.other_actor.id,
                 type=e.other_actor.type_id,impulse=math.sqrt(impulse.x**2+impulse.y**2+impulse.z**2)))+"\n")
        self.sensor.listen(contact)
        self.prepared=True

    def snapshot(self,t):
        if not self.prepared:self.prepare()
        if t-self.last < PARAMS["snapshot_s"]-1e-6:return
        start=time.perf_counter();snap=self.world.get_snapshot(); hs=snap.find(self.hero.id)
        ht=hs.get_transform();hv=hs.get_velocity()
        origin=np.array([ht.location.x,ht.location.y,ht.location.z]); yaw=math.radians(ht.rotation.yaw)
        hero=dict(xyz=origin.tolist(),yaw=yaw,rotation_matrix=np.asarray(ht.get_matrix())[:3,:3].tolist(),
                  extent=[self.hero.bounding_box.extent.x,self.hero.bounding_box.extent.y],
                  v=math.hypot(hv.x,hv.y),id=self.hero.id)
        actors=[]
        for actor in self.world.get_actors():
            if actor.id==self.hero.id or not actor.type_id.startswith(("vehicle.","static.prop.")):continue
            st=snap.find(actor.id)
            if st is None:continue
            tf=st.get_transform();vel=st.get_velocity()
            p=np.array([tf.location.x,tf.location.y,tf.location.z])
            if np.linalg.norm(p[:2]-origin[:2])>PARAMS["radius"] or abs(p[2]-origin[2])>2.:continue
            box=actor.bounding_box
            vertices=box.get_world_vertices(tf)
            cr=np.array([[v.x,v.y,v.z] for v in vertices])
            center=cr.mean(0)
            speed=math.hypot(vel.x,vel.y)
            if speed<=PARAMS["static_speed"]:self.static_since.setdefault(actor.id,t)
            else:self.static_since.pop(actor.id,None)
            row=dict(id=actor.id,type=actor.type_id,xyz=center.tolist(),yaw=math.radians(tf.rotation.yaw+box.rotation.yaw),
                     velocity=[vel.x,vel.y],extent=[max(box.extent.x,.05),max(box.extent.y,.05)],
                     stationary_s=t-self.static_since.get(actor.id,t),corners=cr.tolist())
            row["view"]=visibility(row,hero)
            row["distance"]=float(np.linalg.norm(center[:2]-origin[:2]))
            row["surface_gap"]=rectangle_gap(origin[:2],yaw,hero["extent"],center[:2],row["yaw"],row["extent"])
            actors.append(row)
        self.hero_row,self.actors=hero,actors;self.last=t
        self.snapshot_ms=(time.perf_counter()-start)*1000

    def geometry(self,speed,t,xy,yaw,world,warm):
        self.snapshot(t)
        r=self.agent.route
        ego_s=float(project([xy],r.xy)[0][0]);self.ego_s=ego_s
        self.meta=dict(junctions=[],obstacles=[],bypass=False,gap_open=False,borrow=False,
                       snapshot_ms=round(self.snapshot_ms,2),ego_s=round(ego_s,3),warm=warm)
        if warm:return world
        ahead=np.flatnonzero(self.flags & (r.s>=ego_s-5)&(r.s<=ego_s+60))
        jids=set(int(self.jids[i]) for i in ahead)
        for jid in jids:
            indices=np.flatnonzero(self.jids==jid)
            lo,hi=float(r.s[indices[0]]),float(r.s[indices[-1]])
            path=r.xy[max(0,indices[0]-1):min(len(r.xy),indices[-1]+2)]
            if len(path)<2:continue
            related=[]
            for a in self.actors:
                vel=np.asarray(a["velocity"]);v=float(np.linalg.norm(vel))
                if not a["type"].startswith("vehicle.") or v<.5:continue
                points=np.asarray(a["xyz"][:2])+np.arange(0,5.01,.25)[:,None]*vel
                along,distance,tangent=project(points,path)
                path_s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
                closest=np.stack([np.interp(along,path_s,path[:,k]) for k in (0,1)],-1)
                distance=np.linalg.norm(points-closest,axis=1)
                k=int(np.abs(distance).argmin())
                angle=abs(float(vel@tangent[k]/v))
                opposing=float(vel@tangent[k])<0
                if abs(distance[k])<=a["extent"][1]+self.hero_row["extent"][1]+.15 and (angle<=math.cos(math.pi/6) or opposing):
                    related.append(a["id"])
            if related:self.meta["junctions"].append(dict(key=jid,start_s=lo,end_s=hi,ids=related))
        blockers=[]
        if self.actors:
            positions=np.array([a["xyz"][:2] for a in self.actors])
            longitudinal,lateral,tangent=project(positions,r.xy)
            for i,a in enumerate(self.actors):
                s=float(longitudinal[i]);j=int(np.searchsorted(r.s,s).clip(0,len(r.s)-1))
                h=np.array([math.cos(a["yaw"]),math.sin(a["yaw"])])
                normal=np.array([-tangent[i,1],tangent[i,0]])
                extent=abs(float(h@normal))*a["extent"][0]+abs(float(np.array([-h[1],h[0]])@normal))*a["extent"][1]
                red_queue=a["type"].startswith("vehicle.") and self.agent._ctx().get("tl") in (1,2) and self.agent._ctx().get("tl_dist",99)<50
                if a["stationary_s"]>=2 and -5<=s-ego_s-3.8394<=50 and not self.flags[j] and not red_queue and abs(lateral[i])<=extent+self.hero_row["extent"][1]+.15:
                    blockers.append(dict(id=a["id"],s=s,extent=a["extent"][0]))
        blockers.sort(key=lambda b:b["s"])
        groups=[]
        for b in blockers:
            if not groups or b["s"]-groups[-1][-1]["s"]>20:groups.append([b])
            else:groups[-1].append(b)
        for group in groups:
            start=min(b["s"]-b["extent"] for b in group);end=max(b["s"]+b["extent"] for b in group)
            self.meta["obstacles"].append(dict(key=round(start/5)*5,start_s=start,end_s=end,ids=[b["id"] for b in group]))
        if not self.bypass:return world
        if self.bypass_state is not None and ego_s>self.bypass_state["end_s"]+23:
            self.bypass_state=None
        if self.bypass_state is None and self.meta["obstacles"]:
            candidate=self.meta["obstacles"][0]
            candidate=self.adjacent(candidate)
            if candidate is not None:self.bypass_state=candidate
        state=self.bypass_state
        if state is None:return world
        gap=self.gap_open(state,speed)
        self.meta.update(gap_open=gap,borrow=state["borrow"],bypass_state=state)
        if self.gap_check and state["borrow"] and not state.get("started") and not gap:return world
        state["started"]=True
        s=r.s;enter=np.clip((s-(state["start_s"]-20))/15,0,1);leave=np.clip((s-(state["end_s"]+8))/15,0,1)
        smooth=lambda v:v*v*(3-2*v)
        weight=smooth(enter)*(1-smooth(leave))
        tangent=np.gradient(r.xy,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
        normal=np.stack([-tangent[:,1],tangent[:,0]],-1)
        shifted=r.xy+normal*(weight*state["offset"])[:,None]
        mask=(s>=ego_s-1)&(s<=ego_s+85)
        selected=shifted[mask]
        direction=np.array([math.cos(yaw),math.sin(yaw)])
        selected=selected[((selected-xy)*direction).sum(-1)>.5]
        if len(selected)<2:return world
        self.meta["bypass"]=True
        return np.r_[np.asarray(xy)[None],selected]

    def adjacent(self,state):
        import carla
        r=self.agent.route;j=int(np.searchsorted(r.s,state["start_s"]).clip(0,len(r.s)-1))
        p=r.xy[j];wp=self.map.get_waypoint(carla.Location(x=float(p[0]),y=float(p[1])))
        options=[]
        forward=wp.transform.get_forward_vector();f=np.array([forward.x,forward.y])
        normal=np.array([-f[1],f[0]])
        for order,other in enumerate((wp.get_left_lane(),wp.get_right_lane())):
            if other is None or other.lane_type!=carla.LaneType.Driving:continue
            loc=other.transform.location;d=np.array([loc.x-p[0],loc.y-p[1]])
            offset=float(d@normal);unit=other.transform.get_forward_vector()
            borrow=bool(unit.x*f[0]+unit.y*f[1]<0)
            if 2.5<=abs(offset)<=4.5 and abs(loc.z-wp.transform.location.z)<=.75:
                options.append((borrow,order,offset))
        if not options:self.meta["no_adjacent_lane"]=True;return None
        borrow,_,offset=sorted(options)[0]
        return dict(state,offset=offset,borrow=borrow)

    def gap_open(self,state,speed):
        if not state["borrow"]:return True
        r=self.agent.route;need=max(0.,state["end_s"]+23-self.ego_s)/max(speed,2.)+2
        for a in self.actors:
            if not a["type"].startswith("vehicle."):continue
            s,d,tangent=project([a["xyz"][:2]],r.xy)
            velocity=float(np.asarray(a["velocity"])@tangent[0])
            if velocity>=-.5 or abs(d[0]-state["offset"])>2.:continue
            distance=float(s[0]-self.ego_s)
            if distance>=-5 and (distance<15 or distance/max(-velocity,.5)<need):return False
        return True

    def constraints(self,s_base,speed,path,t,warm):
        from op_arb_agent import TIMES, place, idm, REAR_TO_BUMPER
        # Use observed pose and route geometry, never the actor's scripted future.
        agent=self.agent;xy,yaw=agent.poses[-1][1],agent.poses[-1][2]
        heading=np.array([math.cos(yaw),math.sin(yaw)]);normal=np.array([-heading[1],heading[0]])
        local=place(path,s_base)
        future=np.asarray(xy)+local[:,0,None]*heading-local[:,1,None]*normal
        future_center,headings=ego_boxes(future,yaw)
        conflict_s=[];self.meta["conflict_ids"]=[]
        ids=set(i for j in self.meta["junctions"] for i in j["ids"])
        for a in self.actors:
            if a["id"] not in ids:continue
            other=np.asarray(a["xyz"][:2])+(TIMES+t-self.last)[:,None]*np.asarray(a["velocity"])
            hit=overlap(future_center,headings,self.hero_row["extent"],other,a["yaw"],a["extent"])
            if hit.any():
                position=future[np.flatnonzero(hit)[0]]
                conflict_s.append(float(project([position],agent.route.xy)[0][0]));self.meta["conflict_ids"].append(a["id"])
        out={};active=False
        def stop(distance):
            A=agent.arb
            return idm(speed,distance+A["idm_s0"],0.,0.,A["cruise"],A["amax"],A["idm_b"],A["idm_s0"],A["idm_T"])
        if self.junction and conflict_s and not warm:
            stop_s=min(conflict_s)-REAR_TO_BUMPER-2
            # Retain the earliest stop target during this conflict episode instead of chasing it forward.
            self.junction_stop_s=stop_s if self.junction_stop_s is None else min(self.junction_stop_s,stop_s)
            gap=self.junction_stop_s-self.ego_s
            out["pjunc"]=stop(gap);self.meta.update(junction_stop_gap=gap,junction_stop_s=self.junction_stop_s,late=gap<0)
            active=True;self.clear_since=None
        elif self.junction and self.hold and not warm:
            if self.clear_since is None:self.clear_since=t
            if t-self.clear_since<.8:
                # Keep the same IDM stop location while checking clearance; zeroing a moving profile is a hard brake.
                gap=self.junction_stop_s-self.ego_s
                out["pjunc"]=stop(gap);active=True
                self.meta.update(junction_stop_gap=gap,junction_stop_s=self.junction_stop_s,late=gap<0)
            else:self.release_until=t+2;self.clear_since=None;self.junction_stop_s=None
        self.hold=active
        c=agent._ctx();self.meta["light"]={k:c[k] for k in ("tl","tl_dist","tl_id") if k in c}
        d=c.get("tl_dist",1e9);red=c.get("tl") in (1,2)
        if self.red and red and d<50 and (d>max(.5,speed*speed/8-1) or speed<1 and d>-1) and not warm:
            out["pred"]=stop(d-.5);self.light_hold=True;self.held_light_id=c.get("tl_id")
        elif self.red and self.light_hold and c.get("tl")==0 and d<50 and c.get("tl_id")==self.held_light_id:
            self.release_until=t+2;self.light_hold=False
        if out and speed<.2:self.owned_stop=True
        release=t<self.release_until and speed<1 and getattr(self,"owned_stop",False) and not out and not warm
        if speed>=1:self.release_until=-1e9;self.owned_stop=False
        self.meta.update(release=release,hold=bool(out),controls={k:float(v[-1]) for k,v in out.items()})
        if t>=self.last and getattr(self,"logged",-1e9)<self.last:
            self.scene.write(json.dumps(dict(t=t,frame=agent.cam_sets[-1][0],ego=self.hero_row,
                   actors=self.actors,pc=self.meta))+"\n")
            self.logged=self.last
        return out,release

    def close(self):
        if self.sensor is not None:self.sensor.stop();self.sensor.destroy();self.sensor=None
        self.scene.close();self.contacts.close()
