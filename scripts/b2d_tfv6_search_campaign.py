"""Staged Dev10 controller search and frozen v1 holdout runner."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import threading
import time

from tqdm import tqdm

import b2d_tfv6_campaign as base
from b2d_tfv6_search_control import SEARCH_ARMS, compose_search_control
from b2d_tfv6_w2b import InvariantFailure, validate_attempt

ROOT=Path(__file__).resolve().parents[1]
BUS=Path('/data/runs/b2d/tfv6-w2/bus')
PROTOCOL='todos/2026-09-23-tfv6-controller/protocol-search.md'


def status(step,msg,**numbers):
    t=datetime.now(timezone.utc).isoformat()
    with (BUS/'status.jsonl').open('a') as f:
        f.write(json.dumps({'t':t,'phase':'TASK-8','step':step,'msg':msg,'numbers':numbers},ensure_ascii=False)+'\n')


def same(a,b,tol=1e-5):
    return all(abs(float(a[k])-float(b[k]))<=tol for k in ('steer','throttle','brake'))


def check(result,attempt_dir,run_dir):
    info=validate_attempt(result,attempt_dir,run_dir)
    if info['infra_attempt']:
        return info
    arm=result['arm']
    with (attempt_dir/'frames.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            raw=row.get('raw_control');final=row.get('final_control');actual=row.get('executed_control')
            if not raw or not final or not actual:
                # Initial delay still logs all controls after the model starts.
                if row.get('waypoint') is not None:
                    raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: missing search controls at {row["step"]}')
                continue
            if any(k not in raw for k in ('A','B','C','D',arm)):
                raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: incomplete raw controls at {row["step"]}')
            expected=compose_search_control(arm,raw)
            if expected['brake']>0 and float(row['speed'])<.01:
                expected['steer']=0.
            if expected['brake']>0:expected['throttle']=0.
            if not same(expected,raw[arm]) or not same(final[arm],actual):
                raise InvariantFailure(f'{result["route"]}/{result["seed"]}/{arm}: composition/execution mismatch at {row["step"]}')
    return info


def run_one(item,slot,out,xml,level,abort):
    route,seed,arm=item
    directory=out/'cases'/level/f'route-{route}'/f'seed-{seed}'/arm
    done_path=directory/'done.json'
    if done_path.exists():
        done=json.loads(done_path.read_text())
        check(done,directory/f"attempt-{done['attempt']}",Path(done['run_dir']))
        return item,done
    def invariant(result,attempt_dir,run_dir):
        try:return check(result,attempt_dir,run_dir)
        except InvariantFailure:
            abort.set();raise
    result=base.case(out,xml,level,route,seed,arm,100+slot,0,
                     invariant_check=invariant,abort_event=abort,
                     extra_env={'B2D_TFV6_SEARCH':'1'})
    if result['status']!='finished':
        abort.set();raise InvariantFailure(f'infrastructure retry exhausted for {route}/{seed}/{arm}')
    return item,result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=('dev10','holdout'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arms',required=True)
    p.add_argument('--route-ids',default='')
    p.add_argument('--seeds',default='0')
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--frozen-protocol-commit',default='')
    args=p.parse_args()
    if not args.arms or len(set(args.arms))!=len(args.arms) or any(a not in SEARCH_ARMS for a in args.arms):
        p.error('arms must be distinct registered search arms')
    if args.concurrency not in (1,2):p.error('concurrency must be 1 or 2')
    if args.dataset=='holdout':
        if not args.frozen_protocol_commit:p.error('holdout requires frozen protocol commit')
        frozen=subprocess.check_output(['git','show',f'{args.frozen_protocol_commit}:{PROTOCOL}'],cwd=ROOT)
        if frozen!=(ROOT/PROTOCOL).read_bytes():p.error('holdout protocol changed since frozen commit')
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
            p.error('holdout requires clean worktree')
    xml=base.DEV10 if args.dataset=='dev10' else base.HOLDOUT
    available=base.routes_in(xml)
    routes=args.route_ids.split(',') if args.route_ids else available
    if any(r not in available for r in routes) or len(set(routes))!=len(routes):p.error('invalid route list')
    seeds=[int(s) for s in args.seeds.split(',')]
    if not seeds or any(s not in (0,1,2) for s in seeds) or len(set(seeds))!=len(seeds):p.error('invalid seeds')
    if args.dataset=='holdout' and (routes!=available or seeds!=[0,1,2]):
        p.error('holdout must run all six routes and all three seeds')
    items=[(route,seed,arm) for route in routes for seed in seeds for arm in args.arms]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    level='1' if args.dataset=='dev10' else '2'
    base.record(out,'search_stage_start',dataset=args.dataset,cases=len(items),arms=args.arms)
    status('run_start',f'{args.dataset} search batch starting',cases=len(items),arms=args.arms,
           routes=routes,seeds=seeds,out=str(out))
    abort=threading.Event();done=0;begun=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            pending=iter(items)
            active={pool.submit(run_one,item,slot,out,xml,level,abort):slot
                    for slot,item in enumerate([next(pending,None) for _ in range(args.concurrency)]) if item}
            with tqdm(total=len(items),desc=f'TFv6 search {args.dataset}') as progress:
                while active:
                    for future in as_completed(list(active)):
                        slot=active.pop(future)
                        item,result=future.result()
                        done+=1;progress.update(1)
                        status('case_done',f'{args.dataset} {item} complete',done=done,total=len(items),
                               wall_s=round(time.monotonic()-begun,1),case_wall_s=round(result['wall_s'],1),
                               infra_retries=result.get('infra_retries',0),official_status=result['official_status'])
                        next_item=next(pending,None)
                        if next_item and not abort.is_set():
                            active[pool.submit(run_one,next_item,slot,out,xml,level,abort)]=slot
                        break
    except Exception as e:
        abort.set()
        message=f'{args.dataset} stopped on runner/invariant error: {type(e).__name__}: {e}'
        status('stopped',message,done=done,total=len(items))
        now=datetime.now(timezone.utc).isoformat()
        (BUS/'question.md').write_text('# TASK-8 BLOCKED\n\n'+message+'\n')
        (BUS/'SIGNAL').write_text(f'BLOCKED {now} {message}\n')
        raise
    base.record(out,'search_stage_end',dataset=args.dataset,cases=done,wall_s=round(time.monotonic()-begun,1))
    status('run_end',f'{args.dataset} batch complete',done=done,total=len(items),wall_s=round(time.monotonic()-begun,1))


if __name__=='__main__':main()
