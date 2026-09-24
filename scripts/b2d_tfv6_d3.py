"""D3b factorial/Kalman diagnostic reruns, with per-case W2b invariants."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from b2d_tfv6_campaign import DEV10, HOLDOUT, case, record
from b2d_tfv6_controller_agent import _compose_d3_arm
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


def _same(left,right,tol=1e-5):
    return all(abs(float(left[k])-float(right[k]))<=tol for k in ('steer','throttle','brake'))


def validate_d3(result,attempt_dir,run_dir):
    info=validate_attempt(result,attempt_dir,run_dir)
    if info['infra_attempt']:
        return info
    dense_path=attempt_dir/'d3_global_plan.json'
    if not dense_path.exists():
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: missing evaluator dense plan')
    dense=json.loads(dense_path.read_text())
    if len(dense)<2:
        raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{result["arm"]}: empty evaluator dense plan')
    arm=result['arm']
    path=attempt_dir/'frames.jsonl'
    nav_count=actor_count=light_count=kalman_count=0
    with path.open() as stream:
        for line in stream:
            f=json.loads(line)
            if f.get('d3_nav'):
                nav_count+=1
                if not f['d3_nav'].get('planners'):
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: missing planner trace at step {f["step"]}')
                index=f['d3_nav'].get('selected_target_global_index')
                if index is None or not 0<=int(index)<len(dense):
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: invalid dense plan index at step {f["step"]}')
            if isinstance(f.get('nearby_actors'),list):actor_count+=1
            if f.get('d3_traffic_light'):light_count+=1
            if f.get('d3_kalman'):kalman_count+=1
            raw=f.get('raw_control');final=f.get('final_control');actual=f.get('executed_control')
            if raw and arm in 'EFK':
                speed=float(f['raw_signed_speed_mps'])
                # Author brake interlock uses the speedometer magnitude, not
                # the production controller's signed-motion contract.
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
            'd3_light_ticks':light_count,'d3_kalman_ticks':kalman_count}


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
    execute(args.phase,items)


if __name__=='__main__':main()
