"""Independent synthetic geometry checks and debug route checklists. No score-based tuning."""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/b2d_privileged/lib",)]
import argparse
import ast
import hashlib
import subprocess
import json
from pathlib import Path
import sys
import time

import numpy as np
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from b2d_privileged_geometry import overlap, project, corners, rectangle_gap, visibility, ego_boxes


def is_crash(status):
    """Official statuses append the failure message to 'Failed - '."""
    return status == "Failed" or any(message in status for message in
        ("Simulation crashed", "Agent crashed", "Agent couldn't be set up", "Agent's sensors were invalid"))


BASE_COMMIT='c1ab219'
OLD_WRITE='self.plan_log.write(json.dumps(rec) + "\\n")'
NEW_WRITE='self.plan_log.write(json.dumps(rec, default=_json_scalar) + "\\n")'
HELPER='''

def _json_scalar(value):
    """Normalize NumPy scalars only at the log boundary, including nested flags."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Object of type %s is not JSON serializable' % type(value).__name__)

'''


def source_check():
    paths=('lib/op_arb_agent.py','experiments/b2d_privileged/lib/b2d_privileged_geometry.py')
    old=[subprocess.check_output(['git','show',BASE_COMMIT+':'+p],cwd=REPO) for p in paths]
    new=[(REPO/p).read_bytes() for p in paths]
    assert old[1]==new[1], 'Privileged geometry changed during logging repair'
    assert old[0].decode().count(OLD_WRITE)==1
    normalized=new[0].decode().replace(HELPER,'').replace(NEW_WRITE,OLD_WRITE)
    assert normalized.encode()==old[0], 'Repair changed more than the logging serializer'
    tree=ast.parse(new[0])
    helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_json_scalar']
    assert len(helpers)==1
    namespace={'np':np}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<actual plan-log serializer>','exec'),namespace)
    serializer=namespace['_json_scalar']
    for value in (None,False,True,np.bool_(False),np.bool_(True),np.int64(3),np.float32(.25),'timeout','privileged_clear'):
        decoded=json.loads(json.dumps({'rel':value,'ctx':{'flag':value}},default=serializer))
        assert decoded['rel']==value and decoded['ctx']['flag']==value
    native={'rel':None,'rb':False,'pc':{'controls':{'pjunc':3.5},'hold':True},'t':.05}
    assert json.dumps(native)==json.dumps(native,default=serializer)
    try:json.dumps({'unsupported':object()},default=serializer)
    except TypeError:pass
    else:raise AssertionError('Unknown objects must still fail serialization')
    for status in ('Failed - Agent crashed','Failed - Simulation crashed',"Failed - Agent couldn't be set up",
                   "Failed - Agent's sensors were invalid",'Agent crashed','Failed'):
        assert is_crash(status),status
    for status in ('Perfect','Completed','Failed - TickRuntime','Failed - Agent got blocked'):
        assert not is_crash(status),status
    return dict(base_commit=BASE_COMMIT,old_control_sha256=hashlib.sha256(b''.join(old)).hexdigest(),
                new_control_sha256=hashlib.sha256(b''.join(new)).hexdigest(),
                geometry_identical=True,only_plan_log_serializer_changed=True,
                nested_numpy_scalars_checked=True,native_log_bytes_unchanged=True,unknown_objects_rejected=True,
                actual_log_serializer_checked=True,prefixed_crash_classifier_checked=True,
                model_and_control_computation_unchanged=True)


def scalar_project(point,path):
    best=None;s=0.
    for left,right in zip(path[:-1],path[1:]):
        delta=right-left;length=np.linalg.norm(delta)
        fraction=np.clip((point-left)@delta/max(length*length,1e-12),0,1)
        closest=left+fraction*delta;distance=np.linalg.norm(point-closest)
        tangent=delta/max(length,1e-12)
        signed=(point-closest)@np.array([-tangent[1],tangent[0]])
        if best is None or distance<best[0]:best=(distance,s+fraction*length,signed,tangent)
        s+=length
    return best[1:]


def scalar_overlap(a,ah,ae,b,bh,be,margin):
    ac,bc=corners(a,ah,ae),corners(b,bh,be)
    for angle in (ah,ah+np.pi/2,bh,bh+np.pi/2):
        unit=np.array([np.cos(angle),np.sin(angle)])
        x,y=ac@unit,bc@unit
        if x.max()+margin<y.min() or y.max()+margin<x.min():return False
    return True


def numeric(log,out):
    rng=np.random.default_rng(0)
    path=np.cumsum(rng.normal(size=(60,2)),axis=0)
    points=rng.normal(size=(100,2))*5
    start=time.perf_counter();reference=[scalar_project(p,path) for p in tqdm(points,desc="Scalar projection")]
    reference_s=time.perf_counter()-start
    start=time.perf_counter();actual=project(points,path);vector_s=time.perf_counter()-start
    errors=[float(np.max(np.abs(actual[k]-np.array([p[k] for p in reference])))) for k in range(3)]
    maxerr=max(errors);assert maxerr<=1e-6
    rear=np.array([[0.,0.],[1.,0.],[2.,0.],[2.,1.],[2.,2.]])
    body,angles=ego_boxes(rear,0.)
    assert np.allclose(body[-1],[2.,3.3886],rtol=0,atol=1e-12)
    stationary,angles=ego_boxes(np.tile([2.,3.],(3,1)),np.pi/2)
    assert np.allclose(stationary,np.tile([2.,4.3886],(3,1)),rtol=0,atol=1e-12)
    n=200;a=rng.normal(size=(n,2));b=rng.normal(size=(n,2))*5
    ah=rng.uniform(-np.pi,np.pi,n);bh=rng.uniform(-np.pi,np.pi,n)
    ae=rng.uniform(.1,3,(n,2));be=rng.uniform(.1,3,(n,2))
    expected=[scalar_overlap(a[i],ah[i],ae[i],b[i],bh[i],be[i],.5) for i in tqdm(range(n),desc="Reference SAT")]
    found=overlap(a,ah,ae,b,bh,be,.5)
    assert np.array_equal(found,expected)
    assert abs(rectangle_gap([0,0],0,[1,1],[5,0],0,[1,1])-3)<1e-9
    assert rectangle_gap([0,0],0,[1,1],[1,0],0,[1,1])==0
    hero=dict(xyz=[0,0,0],yaw=0,rotation_matrix=np.eye(3).tolist())
    box=lambda x:dict(corners=[[x+sx,sy,1.0+sz] for sx in (-1,1) for sy in (-.5,.5) for sz in (-.5,.5)])
    assert all(v[0] for v in visibility(box(20),hero).values())
    assert not any(v[0] for v in visibility(box(-20),hero).values())
    # Disabled privilege returns the exact original geometry object, with no coefficient or precision change.
    from b2d_privileged_geometry import Privileged
    from types import SimpleNamespace
    engine=Privileged.__new__(Privileged)
    engine.agent=SimpleNamespace(route=SimpleNamespace(xy=np.array([[0.,0.],[20.,0.],[40.,0.]]),s=np.array([0.,20.,40.])))
    engine.snapshot=lambda t:None;engine.snapshot_ms=0;engine.actors=[]
    engine.flags=np.zeros(3,bool);engine.jids=np.full(3,-1);engine.bypass=False
    # Exercise adjacent-lane state with NumPy inputs; JSON must remain valid in every arm.
    import sys as _sys
    prior=_sys.modules.get("carla")
    _sys.modules["carla"]=SimpleNamespace(Location=lambda **kw:SimpleNamespace(**kw),LaneType=SimpleNamespace(Driving="Driving"))
    try:
        other=SimpleNamespace(lane_type="Driving",transform=SimpleNamespace(location=SimpleNamespace(x=0.,y=3.5,z=0.),
                           get_forward_vector=lambda:SimpleNamespace(x=-1.,y=0.)))
        waypoint=SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(z=0.),
                    get_forward_vector=lambda:SimpleNamespace(x=1.,y=0.)),get_left_lane=lambda:other,get_right_lane=lambda:None)
        engine.map=SimpleNamespace(get_waypoint=lambda location:waypoint);engine.meta={}
        state=engine.adjacent(dict(start_s=0.,end_s=20.,ids=[7]))
        assert type(state["borrow"]) is bool and state["borrow"]
        json.dumps(state,allow_nan=False)
    finally:
        if prior is None:_sys.modules.pop("carla",None)
        else:_sys.modules["carla"]=prior
    original=np.array([[0.,0.],[10.,0.],[30.,0.]])
    assert engine.geometry(5,0,np.zeros(2),0,original,False) is original
    result=dict(projection_maxabs_m=maxerr,sat_cases=n,sat_disagreements=0,disabled_geometry_identity=True,
                exact_gap_checks=True,native_camera_direction_checks=True,adjacent_state_json=True,future_body_center_checks=True,
                scalar_projection_points_per_s=len(points)/reference_s,vector_projection_points_per_s=len(points)/vector_s)
    (out/"numeric_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    log.info("Numeric checks passed: "+json.dumps(result))
    log.scalar("profile/projection_vector_points_per_s",result["vector_projection_points_per_s"],0)
    return result


def route_checks(attempt,kind):
    rows=[json.loads(l) for l in (attempt/"plans.jsonl").open()]
    live=[r for r in rows if not r["warm"]]
    record=json.loads((attempt/"results.json").read_text())["_checkpoint"]["records"][0]
    crash=is_crash(record["status"])
    ms=np.array([r["ms"] for r in live]);v=np.array([r["v"] for r in live])
    assert len(live)>0,f"No post-warmup plans: status={record['status']}, attempt={attempt}; inspect route.log"
    checks=dict(files=(attempt/"privileged.jsonl").exists() and (attempt/"contacts.jsonl").exists(),
                no_crash=not crash,finite=bool(np.isfinite(ms).all() and np.isfinite(v).all()),
                median_latency=float(np.median(ms))<=75,p99_latency=float(np.quantile(ms,.99))<=200)
    if kind=="pred":
        stops=[i for i,r in enumerate(live) if "pred" in r.get("pc",{}).get("controls",{}) and r["v"]<.2]
        go=[r for r in live if r["pc"].get("light",{}).get("tl")==0 and r["v"]>1 and stops and r["t"]>live[stops[0]]["t"]]
        checks.update(red_stop=bool(stops),green_go=bool(go))
    if kind=="pjunc":
        stop=[r for r in live if "pjunc" in r.get("pc",{}).get("controls",{})]
        checks.update(conflict_constraint=bool(stop),clear_go=any(r["v"]>1 and r["t"]>stop[0]["t"] and not r["pc"].get("hold") for r in live) if stop else False)
    if kind=="pbyp":
        scene=[json.loads(l) for l in (attempt/"privileged.jsonl").open()]
        enabled=[r for r in scene if r["pc"].get("bypass")]
        checks.update(obstacle_detected=any(r["pc"].get("obstacles") for r in scene),path_enabled=bool(enabled),
             valid_shift=all(2.5<=abs(r["pc"]["bypass_state"]["offset"])<=4.5 for r in enabled) if enabled else False,
             passed_and_returned=any(r["pc"]["ego_s"]>enabled[0]["pc"]["bypass_state"]["end_s"]+23 for r in scene) if enabled else False)
    return dict(checks=checks,status=record["status"],ms_med=float(np.median(ms)),ms_p99=float(np.quantile(ms,.99)),
                passed=all(checks.values()),attempt=str(attempt))
