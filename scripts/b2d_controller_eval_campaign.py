"""Resumable Task 10 L2/L3 CARLA campaign with immediate attempt checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data')
ROUTES = ROOT / 'todos/2026-09-23-tfv6-controller/controller-eval/l23-v2-heldout.xml'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def event(root, kind, **data):
    row = {'t': datetime.now(timezone.utc).isoformat(), 'kind': kind, **data}
    with (root/'events.jsonl').open('a') as file:
        file.write(json.dumps(row)+'\n')
    with (root/'log.txt').open('a') as file:
        file.write(f"{row['t']} {kind} {data}\n")
    print(f'{kind}: {data}', flush=True)


def validate_tcp_attempt(result, attempt, run):
    rid, arm = result['route'], result['arm']
    output = run/'attempts'/rid/'1'
    rows_path = output/'tcp-control.jsonl'
    official_path = output/'results.json'
    summary_path = run/'summary.json'
    if result['status'] != 'finished' and not official_path.exists():
        return False  # infrastructure fault; permitted retry
    if not all(p.exists() for p in (rows_path, official_path, summary_path)):
        raise RuntimeError(f'TCP {rid}/{arm}: missing driving telemetry/result')
    summary = json.loads(summary_path.read_text())
    ticks = int(summary['attempts'][rid][-1]['ticks'])
    official = json.loads(official_path.read_text())['_checkpoint']['records']
    if len(official) != 1 or not official[0]['status']:
        raise RuntimeError(f'TCP {rid}/{arm}: invalid official result')
    if 'agent crashed' in official[0]['status'].lower() or 'agent error' in official[0]['status'].lower():
        raise RuntimeError(f'TCP {rid}/{arm}: agent crash: {official[0]["status"]}')
    rows = [json.loads(s) for s in rows_path.open()]
    mirror = [json.loads(s) for s in (output/'controller-eval.jsonl').open()]
    if len(rows) != ticks or rows != mirror or not rows:
        raise RuntimeError(f'TCP {rid}/{arm}: incomplete or inconsistent frame log {len(rows)}/{ticks}')
    frames = [r['frame'] for r in rows]
    if frames != list(range(frames[0], frames[0]+ticks)):
        raise RuntimeError(f'TCP {rid}/{arm}: noncontiguous frame IDs')
    for index, row in enumerate(rows):
        if row['arm'] != arm or row['truth'].get('frame') != row['frame']:
            raise RuntimeError(f'TCP {rid}/{arm}: identity/truth mismatch tick {index}')
        if any(row['sensor_frames'].get(k) != row['frame'] for k in
               ('GPS','IMU','SPEED','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')):
            raise RuntimeError(f'TCP {rid}/{arm}: model sensor mismatch tick {index}')
        if row['prediction'] is not None:
            if row['forward_count'] != 1 or row['prediction']['native_pid_calls'] != 1:
                raise RuntimeError(f'TCP {rid}/{arm}: inference/native PID count tick {index}')
            if set(row['shadow_control']) != set('ABCD' + ('P' if os.environ.get('B2D_P_CONFIG') else '')):
                raise RuntimeError(f'TCP {rid}/{arm}: missing shadow arm tick {index}')
            selected = row['official_tail_control'] if arm == 'N' else row['shadow_control'][arm]
            if max(abs(a-b) for a,b in zip(selected,row['selected_control'])) > 1e-6:
                raise RuntimeError(f'TCP {rid}/{arm}: selected control mismatch tick {index}')
    result['official_status'] = official[0]['status']
    result['ds'] = float(official[0]['scores']['score_composed'])
    result['rc'] = float(official[0]['scores']['score_route'])
    result['ticks'] = ticks
    return True


def tcp_case(out, rid, seed, arm, server_index):
    directory = out/'cases'/f'route-{rid}'/f'seed-{seed}'/arm
    done = directory/'done.json'
    if done.exists():
        event(out, 'skip', route=rid, seed=seed, arm=arm)
        return json.loads(done.read_text())
    for number in range(1, 5):
        attempt = directory/f'attempt-{number}'
        if (attempt/'result.json').exists():
            prior = json.loads((attempt/'result.json').read_text())
            if prior['status'] == 'finished':
                raise RuntimeError(f'Completed TCP attempt lacks done.json: {rid}/{seed}/{arm}')
            continue
        attempt.mkdir(parents=True, exist_ok=True)
        run = attempt/'run'
        env = os.environ.copy()
        zoo = DATA/'third_party/Bench2DriveZoo'
        env.update(DATA_DIR=str(DATA), BENCH2DRIVE_ROOT=str(DATA/'runs/b2d/tfv6-repro/runtime/Bench2Drive'),
                   IS_BENCH2DRIVE='1', PLANNER_TYPE='only_traj', TORCH_HOME=str(DATA/'models/torch'),
                   B2D_TCP_CONTROL_ARM=arm, B2D_TCP_OPTIMIZE='1', B2D_TCP_PIPELINE='1',
                   B2D_TCP_FAST_COLOR='1', B2D_TCP_DEBUG_VIEWS='1', B2D_TCP_EARLY_RGB='1',
                   B2D_ASYNC_DISPLAY='1', B2D_CAPTURE_CRITERION_EVENTS='1',
                   B2D_PREVIEW_DIR=str(attempt/'live'), SAVE_PATH=str(attempt/'vendor-save'),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_CORETYPE='Barcelona',
                   PYTHONPATH=':'.join(map(str, (ROOT/'scripts', zoo, zoo/'TCP'))))
        python = DATA/'envs/b2d-tcp/bin/python'
        command = [str(python), str(ROOT/'scripts/b2d_run.py'), '--routes', str(ROUTES),
                   '--route-ids', rid, '--out', str(run), '--workers', '1',
                   '--server-index', str(server_index), '--gpu-rank', '0', '--quality', 'Epic',
                   '--max-attempts', '1', '--no-spectator', '--zero-copy',
                   '--agent', str(ROOT/'scripts/b2d_tcp_eval_agent.py'),
                   '--agent-config', str(DATA/'models/bench2drive/tcp/tcp_b2d.ckpt'),
                   '--python', str(python), '--tm-seed', str(seed), '--route-timeout-s', '600',
                   '--stall-s', '120', '--max-ticks', '4000']
        event(out, 'case_start', route=rid, seed=seed, arm=arm, attempt=number)
        started = time.monotonic()
        with (attempt/'runner.log').open('w') as log:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        summary_path = run/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        reports = summary.get('attempts', {}).get(rid, [])
        status = reports[-1]['status'] if reports else 'missing_summary'
        result = dict(planner='TCP', route=rid, seed=seed, arm=arm, attempt=number,
                      returncode=rc, status=status, wall_s=time.monotonic()-started,
                      run_dir=str(run))
        put(attempt/'result.json', result)
        try:
            valid = validate_tcp_attempt(result, attempt, run)
        except Exception as error:
            event(out, 'invariant_failure', route=rid, seed=seed, arm=arm, error=repr(error))
            raise
        put(attempt/'result.json', result)
        event(out, 'case_attempt_end', **result)
        if valid:
            put(done, {**result, 'infra_retries': number-1})
            return result
        event(out, 'infra_retry', route=rid, seed=seed, arm=arm, attempt=number)
    raise RuntimeError(f'TCP infrastructure retry limit reached: {rid}/{seed}/{arm}')


def main():
    global ROUTES
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', required=True, choices=('tfv6','tcp'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server-index', type=int, default=110)
    parser.add_argument('--routes', type=Path, default=ROUTES)
    parser.add_argument('--route-ids', help='default: every route in --routes')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--arms', help='default ABCD for TFv6, NABCD for TCP')
    a = parser.parse_args()
    ROUTES = a.routes
    a.out.mkdir(parents=True, exist_ok=True)
    all_ids = [r.get('id') for r in ET.parse(ROUTES).getroot().findall('route')]
    route_ids = a.route_ids.split(',') if a.route_ids else all_ids
    seeds = [int(x) for x in a.seeds.split(',')]
    candidate = 'P' if os.environ.get('B2D_P_CONFIG') else ''
    arms = a.arms or ('ABCD' if a.planner == 'tfv6' else 'NABCD') + candidate
    if not set(route_ids) <= set(all_ids) or not set(arms) <= set(('ABCD' if a.planner=='tfv6' else 'NABCD') + candidate):
        raise ValueError('Invalid route/arm selection')
    if a.planner == 'tfv6':
        sys.path.insert(0, str(ROOT/'scripts'))
        import b2d_tfv6_campaign as base
        from b2d_tfv6_w2b import validate_attempt
    total = len(route_ids)*len(seeds)*len(arms)
    event(a.out, 'start', planner=a.planner, route_ids=route_ids, seeds=seeds,
          arms=arms, total=total, protocol_commit='v2')
    count = 0
    for rid in route_ids:
        for seed in seeds:
            for arm in arms:
                if a.planner == 'tfv6':
                    base.case(a.out, ROUTES, '10', rid, seed, arm,
                              a.server_index, 4000, invariant_check=validate_attempt,
                              extra_env={'B2D_D2_ACTORS': '1',
                                         **({'B2D_P_CONFIG': os.environ['B2D_P_CONFIG']} if candidate else {})})
                else:
                    tcp_case(a.out, rid, seed, arm, a.server_index)
                count += 1
                event(a.out, 'progress', done=count, total=total, route=rid, seed=seed, arm=arm)
    event(a.out, 'end', status='completed', done=count, total=total)


if __name__ == '__main__':
    main()
