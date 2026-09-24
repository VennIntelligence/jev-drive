"""W2b C/D staged campaign with immediate I1-I3 checks and mechanical gates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from tqdm import tqdm

import b2d_tfv6_campaign as base
from b2d_tfv6_coordinates import rear_waypoints

ROOT = Path(__file__).resolve().parents[1]
BUS = Path('/data/runs/b2d/tfv6-w2/bus')
OUT = Path('/data/runs/b2d/tfv6-w2b')
OLD = Path('/data/runs/b2d/tfv6-w2/formal')
PROTOCOL = 'todos/2026-09-23-tfv6-controller/protocol-w2b.md'
STAGES = ('S1', 'S2', 'S3')


class InvariantFailure(RuntimeError):
    pass

REAR_WAYPOINT_ABS_TOL_M = 1e-12


def bus(state, stage, message, numbers=None):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    entry = {'t': now, 'phase': 'W2b', 'step': stage, 'msg': message,
             'numbers': numbers or {}}
    with (BUS / 'status.jsonl').open('a', buffering=1) as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    if state in ('BLOCKED', 'DONE'):
        (BUS / 'SIGNAL').write_text(f'{state} {now} {message}\n')


def block(message):
    (BUS / 'question.md').write_text('# W2b BLOCKED\n\n' + message + '\n')
    bus('BLOCKED', 'blocked', message)


def load_frames(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def validate_attempt(result, attempt_dir, run_dir):
    """Run directly after a case attempt; infra failures may retry, I1-I3 may not."""
    label = f"{result['level']}/{result['route']}/{result['seed']}/{result['arm']} attempt {result['attempt']}"
    frame_path = attempt_dir / 'frames.jsonl'
    frames = load_frames(frame_path) if frame_path.exists() else []
    for frame in frames:
        diagnostic = frame.get('diagnostic')
        if diagnostic:
            invariant = 'I1' if diagnostic == 'I1_phantom' else 'I2'
            raise InvariantFailure(f'{label}: {invariant} agent diagnostic at step {frame.get("step")}: {diagnostic}')
    official_status = result.get('official_status') or ''
    if 'agent crashed' in official_status.lower() or 'agent error' in official_status.lower():
        raise InvariantFailure(f'{label}: I2 official agent crash: {official_status}')
    if result['status'] != 'finished':
        # A simulator/RPC/load failure is an infrastructure retry under W2 rules.
        return {'infra_attempt': True, 'label': label}
    if not official_status:
        raise InvariantFailure(f'{label}: I3 missing official status')
    result_path = run_dir / 'attempts' / result['route'] / '1' / 'results.json'
    infractions_path = attempt_dir / 'infractions.json'
    summary_path = run_dir / 'summary.json'
    if not all(p.exists() for p in (frame_path, result_path, infractions_path, summary_path)):
        raise InvariantFailure(f'{label}: I3 missing frames/results/infractions/summary')
    try:
        records = json.loads(result_path.read_text())['_checkpoint']['records']
        infractions = json.loads(infractions_path.read_text())['infractions']
        summary = json.loads(summary_path.read_text())
        ticks = int(summary['attempts'][result['route']][-1]['ticks'])
    except (ValueError, KeyError, TypeError, IndexError) as e:
        raise InvariantFailure(f'{label}: I3 malformed official/log metadata: {e}') from e
    if len(records) != 1 or not isinstance(infractions, list) or 'infractions' not in records[0]:
        raise InvariantFailure(f'{label}: I3 malformed official result or infractions')
    if len(frames) != ticks or [f.get('step') for f in frames] != list(range(ticks)):
        raise InvariantFailure(f'{label}: I3 frame coverage {len(frames)} != {ticks} or non-contiguous steps')
    if any(f.get('arm') != result['arm'] or f.get('route') != result['route'] for f in frames):
        raise InvariantFailure(f'{label}: I3 frame identity mismatch')
    phantom = valid = roundoff_ticks = 0
    max_rear_waypoint_residual_m = 0.0
    for frame in frames:
        actor, rear = frame.get('waypoint'), frame.get('rear_waypoint')
        if actor is None and rear is None:
            continue
        if actor is None or rear is None:
            raise InvariantFailure(f'{label}: I3 incomplete waypoint pair at step {frame["step"]}')
        p = np.asarray(actor, dtype=float)
        r = np.asarray(rear, dtype=float)
        if p.shape != (8, 2) or r.shape != (8, 2) or not np.isfinite(p).all() or not np.isfinite(r).all():
            raise InvariantFailure(f'{label}: I3 invalid waypoint shape/value at step {frame["step"]}')
        p0 = p[0] * (1., -1.)
        valid += 1
        phantom += int(np.linalg.norm(p0) < .3 and np.linalg.norm(r[0] - p0) > .3)
        residual = float(np.max(np.abs(r - rear_waypoints(p))))
        max_rear_waypoint_residual_m = max(max_rear_waypoint_residual_m, residual)
        roundoff_ticks += int(residual > 0)
        if residual > REAR_WAYPOINT_ABS_TOL_M:
            if phantom:
                raise InvariantFailure(f'{label}: I1 offline phantom at step {frame["step"]}')
            raise InvariantFailure(f'{label}: I3 logged rear waypoint differs from pinned transform at step {frame["step"]}: max residual {residual:.3g}m > {REAR_WAYPOINT_ABS_TOL_M:.3g}m')
    if phantom:
        raise InvariantFailure(f'{label}: I1 offline phantom count {phantom}/{valid}')
    return {'infra_attempt': False, 'label': label, 'frames': len(frames),
            'valid_waypoint_ticks': valid, 'phantom': phantom,
            'rear_waypoint_abs_tolerance_m': REAR_WAYPOINT_ABS_TOL_M,
            'max_rear_waypoint_residual_m': max_rear_waypoint_residual_m,
            'rear_waypoint_roundoff_ticks': roundoff_ticks,
            'official_status': official_status,
            'ds': float(records[0]['scores']['score_composed'])}


def items_for_stage(stage):
    dev = base.routes_in(base.DEV10)
    hold = base.routes_in(base.HOLDOUT)
    all_cd = [(level, route, seed, arm) for level, routes in (('1', dev), ('2', hold))
              for route in routes for seed in range(3) for arm in 'CD']
    canary_cd = [('1','3514',seed,'C') for seed in range(3)] + [
        ('2','28154',0,'C'), ('2','25318',0,'C')]
    canary_ab = [('2','28154',0,'B'), ('1','26405',1,'A')]
    small = [('1',route,0,arm) for route in dev for arm in 'CD'
             if ('1',route,0,arm) not in canary_cd]
    rest = [key for key in all_cd if key not in canary_cd and key not in small]
    assert (len(canary_cd),len(canary_ab),len(small),len(rest)) == (5,2,19,72)
    return {'S1':canary_cd + canary_ab, 'S2':small, 'S3':rest}[stage]


def case_dir(level, route, seed, arm):
    root = OUT / ('canary' if arm in 'AB' else 'level' + level)
    return root, root / 'cases' / level / f'route-{route}' / f'seed-{seed}' / arm


def _check_done_tree():
    seen = set()
    for path in OUT.glob('*/cases/*/route-*/seed-*/*/done.json'):
        d = json.loads(path.read_text())
        key = (str(d['level']),str(d['route']),int(d['seed']),d['arm'])
        root, expected = case_dir(*key)
        if path.parent.resolve() != expected.resolve():
            raise InvariantFailure(f'I3 wrong case path: {path}')
        run = expected / f"attempt-{d['attempt']}" / 'run'
        if Path(d['run_dir']).resolve() != run.resolve():
            raise InvariantFailure(f'I3 selected attempt mismatch: {path}')
        if key in seen:
            raise InvariantFailure(f'I3 duplicate case key: {key}')
        seen.add(key)
    return seen


def run_one(item, slot, abort):
    level, route, seed, arm = item
    root, directory = case_dir(*item)
    root.mkdir(parents=True, exist_ok=True)
    if (directory / 'done.json').exists():
        done = json.loads((directory / 'done.json').read_text())
        attempt = directory / f"attempt-{done['attempt']}"
        check = validate_attempt(done, attempt, Path(done['run_dir']))
        _check_done_tree()
        return {'item':item, 'result':done, 'check':{**check,'resumed':True}}
    checks = []
    def check(result, attempt_dir, run_dir):
        try:
            value = validate_attempt(result, attempt_dir, run_dir)
            checks.append(value)
            return value
        except InvariantFailure:
            abort.set()
            raise
    xml = base.DEV10 if level == '1' else base.HOLDOUT
    result = base.case(root, xml, level, route, seed, arm, 90 + slot, 0,
                       invariant_check=check, abort_event=abort)
    _check_done_tree()
    return {'item':item, 'result':result, 'check':checks[-1] if checks else {}}


def check_s1_gate(rows):
    if len(rows) != 7 or any(x['result']['status'] != 'finished' for x in rows):
        raise InvariantFailure('S1 gate: all 7 canary cases must finish with I1-I3 valid')
    by = {x['item']:x for x in rows}
    for seed in range(3):
        item = ('1','3514',seed,'C')
        done = by[item]['result']
        inf = Path(done['run_dir']).parent / 'infractions.json'
        events = json.loads(inf.read_text())['infractions']
        collisions = [e for e in events if e['event_type'].split('.')[-1] == 'COLLISION_STATIC'
                      and int(e['step']) <= 60]
        if collisions:
            raise InvariantFailure(f'S1 gate: 3514 seed {seed} C static collision within 3 s: {collisions}')
    for item in (('2','28154',0,'B'), ('1','26405',1,'A')):
        new = by[item]['result']
        oldpath = (OLD / ('level' + item[0]) / 'cases' / item[0] /
                   ('route-' + item[1]) / ('seed-' + str(item[2])) / item[3] / 'done.json')
        old = json.loads(oldpath.read_text())
        def official(d):
            p = Path(d['run_dir']) / 'attempts' / d['route'] / '1' / 'results.json'
            record = json.loads(p.read_text())['_checkpoint']['records'][0]
            return float(record['scores']['score_composed']), record['status']
        if official(new) != official(old):
            raise InvariantFailure(f'S1 gate: A/B invariance failed for {item}: {official(new)} vs {official(old)}')


def run_stage(stage):
    items = items_for_stage(stage)
    begun = time.monotonic()
    abort = threading.Event()
    results = []
    base.record(OUT, 'stage_start', stage=stage, cases=len(items))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = iter(items)
            future_map = {pool.submit(run_one, item, slot, abort): (item,slot)
                          for slot,item in enumerate([next(pending),next(pending)])}
            with tqdm(total=len(items), desc=f'W2b {stage}') as progress:
                while future_map:
                    for future in as_completed(list(future_map)):
                        item,slot = future_map.pop(future)
                        value = future.result()
                        results.append(value)
                        progress.update(1)
                        if stage == 'S3' and time.monotonic() - run_stage.last_progress >= 6900:
                            bus('WORKING', 'S3_progress', 'S3 running',
                                {'done':len(results),'total':len(items),
                                 'wall_s':time.monotonic()-begun})
                            run_stage.last_progress = time.monotonic()
                        next_item = next(pending,None)
                        if next_item and not abort.is_set():
                            future_map[pool.submit(run_one,next_item,slot,abort)] = (next_item,slot)
                        break
        if stage == 'S1':
            check_s1_gate(results)
        _check_done_tree()
        info = {'done':len(results),'total':len(items),
                'valid':sum(x['result']['status']=='finished' for x in results),
                'infra_exhausted':sum(x['result']['status']!='finished' for x in results),
                'I1_phantom':0,'I2_failed':0,'I3_failed':0,
                'wall_s':round(time.monotonic()-begun,1)}
        bus('WORKING',stage,f'{stage} gate passed: {info["valid"]}/{len(items)} valid',info)
        base.record(OUT,'stage_end',stage=stage,**info)
        return results
    except Exception:
        abort.set()
        raise


run_stage.last_progress = time.monotonic()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen-protocol-commit',required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--start-stage',choices=STAGES,default='S1')
    args = parser.parse_args()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head != args.code_commit or subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT).returncode:
        parser.error('Pinned code commit does not match clean HEAD')
    frozen = subprocess.check_output(['git','show',f'{args.frozen_protocol_commit}:{PROTOCOL}'],cwd=ROOT)
    if frozen != (ROOT / PROTOCOL).read_bytes():
        parser.error('Frozen W2b protocol differs from pinned commit')
    OUT.mkdir(parents=True,exist_ok=True)
    bus('WORKING','campaign_start','W2b staged campaign running',
        {'code_commit':head,'protocol_commit':args.frozen_protocol_commit})
    try:
        for stage in STAGES[STAGES.index(args.start_stage):]:
            run_stage(stage)
    except Exception as e:
        block(f'W2b stopped after an invariant/gate/runner failure: {type(e).__name__}: {e}. '
              'Only W2b-owned route/CARLA processes were terminated; inspect W2b log.txt and events.jsonl. '
              'No patch or restart was made.')
        raise
    bus('WORKING','campaign_end','S1-S3 passed; analysis/report pending',
        {'CD_cases':96,'canary_AB':2})


if __name__ == '__main__':
    main()
