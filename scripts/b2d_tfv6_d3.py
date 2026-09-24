"""D3b factorial/Kalman diagnostic reruns, with per-case W2b invariants."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

from b2d_tfv6_campaign import DATA, DEV10, HOLDOUT, RUNTIME, case, record

# The route launcher adds these three source roots before loading the agent.
# This runner also imports an agent helper while building its own invariants.
for source_root in (DATA/'third_party/lead-cvpr2026',
                    RUNTIME/'leaderboard',RUNTIME/'scenario_runner'):
    sys.path.insert(0,str(source_root))

from b2d_tfv6_controller_agent import _compose_d3_arm
from b2d_tfv6_d3_semantic import SemanticFailure, SemanticSequence
from b2d_tfv6_w2b import InvariantFailure, validate_attempt

ROOT=Path('/data/runs/b2d/tfv6-d3')
BUS=Path('/data/runs/b2d/tfv6-w2/bus')
GROUPS=(('2','2084',0),('2','2084',1),('2','2084',2),
        ('2','27529',0),('2','27529',1),('2','27529',2),
        ('1','2091',0),('1','2091',1),('1','2091',2))
LOCK=threading.Lock()


def milestone(step,message,**numbers):
    row={'t':datetime.now(timezone.utc).isoformat(),'phase':'D3','step':step,
         'msg':message,'numbers':numbers}
    with LOCK,(BUS/'status.jsonl').open('a') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def block(error):
    now=datetime.now(timezone.utc).isoformat()
    message=f'D3b stopped on invariant, driving, runner or classification failure: {type(error).__name__}: {error}'
    (BUS/'question.md').write_text('# D3 BLOCKED\n\n'+message+'\n\nOnly D3-owned route/CARLA workers were cancelled; no patch or restart was made. Inspect /data/runs/b2d/tfv6-d3/.\n')
    milestone('blocked',message)
    (BUS/'SIGNAL').write_text(f'BLOCKED {now} {message}\n')


def _same(left,right,tol=1e-5):
    return all(abs(float(left[k])-float(right[k]))<=tol for k in ('steer','throttle','brake'))


def validate_d3(result,attempt_dir,run_dir):
    info=validate_attempt(result,attempt_dir,run_dir)
    if info['infra_attempt']:
        return info
    dense_path=ROOT/'dense'/f'{result["level"]}-{result["route"]}.json'
    sparse_path=attempt_dir/'d3_agent_sparse_plan.json'
    if not dense_path.exists() or not sparse_path.exists():
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: missing evaluator dense or agent sparse plan')
    package=json.loads(dense_path.read_text())
    sparse=json.loads(sparse_path.read_text())
    if len(package['dense'])<2 or len(sparse)!=len(package['agent_sparse_world_xy']):
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: invalid dense/sparse plan')
    fit=package['planner_to_world']
    if fit['max_residual_m']>=.1 or abs(fit['scale']-1)>=.001 or abs(fit['rotation_rad'])>=.001:
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: planner/world alignment failed')
    for j,(actual,expected) in enumerate(zip(sparse,package['agent_sparse_world_xy'])):
        if math.dist(actual['xyz'][:2],expected)>.1 or actual['command']!=package['dense'][package['agent_sparse_dense_indices'][j]]['command']:
            raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: agent sparse waypoint {j} differs from reconstructed evaluator route')
    checker=SemanticSequence(package)
    arm=result['arm']
    path=attempt_dir/'frames.jsonl'
    nav_count=actor_count=light_count=kalman_count=0
    first_ego_offroute_step=None
    max_ego_dense_distance_m=0.0
    with path.open() as stream,(attempt_dir/'d3_nav_semantic.jsonl').open('w') as semantic_out:
        for line in stream:
            f=json.loads(line)
            if f.get('d3_nav'):
                nav_count+=1
                if not f['d3_nav'].get('planners'):
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: missing planner trace at step {f["step"]}')
                try:
                    semantic=checker.check(f)
                except (SemanticFailure,KeyError,ValueError) as error:
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: D3 semantic telemetry: {error}') from error
                semantic_out.write(json.dumps(semantic)+'\n')
                distance=semantic['ego_dense_distance_m']
                max_ego_dense_distance_m=max(max_ego_dense_distance_m,distance)
                if distance>3 and first_ego_offroute_step is None:
                    first_ego_offroute_step=f['step']
            if isinstance(f.get('nearby_actors'),list):actor_count+=1
            if f.get('d3_traffic_light'):light_count+=1
            if f.get('d3_kalman'):kalman_count+=1
            raw=f.get('raw_control');final=f.get('final_control');actual=f.get('executed_control')
            if raw and arm in 'EFK':
                speed=float(f['raw_signed_speed_mps'])
                # The author brake interlock uses the logged speedometer value.
                expected=_compose_d3_arm(arm,raw,speed)
                if not _same(raw[arm],expected):
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: hybrid raw composition at step {f["step"]}')
                if final and actual and not _same(final[arm],actual):
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: hybrid executed mismatch at step {f["step"]}')
            if arm=='K' and f.get('d3_kalman'):
                k=f['d3_kalman']
                if k['used_b_shadow'] and not raw:
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/K: Kalman replacement without B shadow at step {f["step"]}')
    if nav_count<info['frames']-2 or actor_count<info['frames']-2 or light_count<info['frames']-2:
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: D3 telemetry coverage nav={nav_count}, actor={actor_count}, light={light_count}, ticks={info["frames"]}')
    if arm=='K' and kalman_count<info['frames']-2:
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/K: Kalman telemetry coverage {kalman_count}/{info["frames"]}')
    recorder=run_dir/'attempts'/result['route']/'1'/'recorder'
    if not recorder.exists() or not any(p.stat().st_size>0 for p in recorder.glob('*.log')):
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: missing CARLA recorder')
    return {**info,'d3_nav_ticks':nav_count,'d3_actor_ticks':actor_count,
            'd3_light_ticks':light_count,'d3_kalman_ticks':kalman_count,
            'first_ego_offroute_3m_step':first_ego_offroute_step,
            'max_ego_dense_distance_m':max_ego_dense_distance_m,
            'official_status':result['official_status']}


def _check_done_tree(root):
    seen=set()
    for p in root.glob('cases/*/route-*/seed-*/*/done.json'):
        done=json.loads(p.read_text())
        key=(done['level'],done['route'],done['seed'],done['arm'])
        if key in seen:
            raise InvariantFailure(f'I3 duplicate key {key}')
        seen.add(key)
        if Path(done['run_dir']).resolve()!=(p.parent/f"attempt-{done['attempt']}"/'run').resolve():
            raise InvariantFailure(f'I3 selected attempt mismatch {p}')
    return seen


def run_one(item,slot,phase,abort):
    level,route,seed,arm=item
    out=ROOT/phase
    out.mkdir(parents=True,exist_ok=True)
    directory=out/'cases'/level/f'route-{route}'/f'seed-{seed}'/arm
    if (directory/'done.json').exists():
        done=json.loads((directory/'done.json').read_text())
        info=validate_d3(done,directory/f"attempt-{done['attempt']}",Path(done['run_dir']))
        return item,done,info
    def check(result,attempt_dir,run_dir):
        try:return validate_d3(result,attempt_dir,run_dir)
        except InvariantFailure:
            abort.set();raise
    xml=HOLDOUT if level=='2' else DEV10
    result=case(out,xml,level,route,seed,arm,90+slot,0,record_carla=True,
                invariant_check=check,abort_event=abort)
    _check_done_tree(out)
    info=validate_d3(result,directory/f"attempt-{result['attempt']}",Path(result['run_dir']))
    return item,result,info


def execute(phase,items):
    abort=threading.Event();start=time.monotonic();results=[]
    out=ROOT/phase;out.mkdir(parents=True,exist_ok=True)
    record(out,'phase_start',phase=phase,cases=len(items))
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=iter(items)
        current={}
        for slot in range(min(2,len(items))):
            item=next(pending);current[pool.submit(run_one,item,slot,phase,abort)]=slot
        while current:
            for future in as_completed(list(current)):
                slot=current.pop(future)
                try:
                    item,result,info=future.result()
                except Exception:
                    abort.set();raise
                results.append((item,result,info))
                milestone('D3b_progress',f'{phase} {len(results)}/{len(items)} cases valid',
                    phase=phase,done=len(results),total=len(items),case=item,
                    wall_s=round(time.monotonic()-start,1),official_status=result.get('official_status'))
                next_item=next(pending,None)
                if next_item and not abort.is_set():
                    current[pool.submit(run_one,next_item,slot,phase,abort)]=slot
                break
    record(out,'phase_end',phase=phase,cases=len(results),wall_s=round(time.monotonic()-start,1))
    milestone('D3b_phase_end',f'{phase} {len(results)}/{len(items)} valid',
              phase=phase,done=len(results),total=len(items),wall_s=round(time.monotonic()-start,1))
    return results


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=('factorial','kalman'),required=True)
    parser.add_argument('--groups',nargs='*',help='kalman group ids such as 2-2084-0')
    args=parser.parse_args()
    os.environ['B2D_D3_DIAGNOSTIC']='1'
    os.environ['B2D_D2_ACTORS']='1'
    if args.phase=='factorial':
        items=[(*group,arm) for group in GROUPS for arm in 'BCEF']
    else:
        selected=set(args.groups or [])
        if not selected:
            parser.error('kalman requires explicit groups selected from logged R1 signs')
        groups=[g for g in GROUPS if f'{g[0]}-{g[1]}-{g[2]}' in selected]
        if len(groups)!=len(selected) or any(g[1]=='2091' for g in groups):
            parser.error('kalman selection must be 2084/27529 groups only')
        items=[(*g,'K') for g in groups]
    try:
        execute(args.phase,items)
    except Exception as error:
        block(error)
        raise


if __name__=='__main__':main()
