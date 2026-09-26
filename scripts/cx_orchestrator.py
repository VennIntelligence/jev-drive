#!/usr/bin/env python3
"""Durable dispatcher for registered independent D children; no live-chain edits.

Run in jev tmux: python3 scripts/cx_orchestrator.py run
Only this manifest's non-CARLA commands run. GPU 1 is never admitted. Failures
stop their dependants, not other branches. DONE files belong to this dispatcher.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / 'scripts/cx_orchestration.json'
DATA = Path(os.environ.get('DATA_DIR', '/nonexistent'))
OUT = DATA / 'runs/nq4/cx/orchestration'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def cpuset(spec):
    result = set()
    for part in spec.split(','):
        pair = list(map(int, part.split('-')))
        result.update(range(pair[0], pair[-1] + 1))
    return result


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return {'pid': int(pid), 'start_ticks': stat[19], 'pgid': int(stat[2])}
    except (OSError, ValueError): return None


def alive(record):
    return bool(record) and identity(record['pid']) == record


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def expand(value):
    return os.path.expandvars(value.replace('$REPO', str(REPO)))


def artifacts(job):
    """All required patterns must have at least one nonempty file; alternatives OR."""
    import glob
    def found(pattern):
        return any(Path(p).is_file() and Path(p).stat().st_size > 0 for p in glob.glob(expand(pattern)))
    return all(found(p) for p in job.get('artifacts', [])) and all(
        any(found(p) for p in group) for group in job.get('artifact_any', []))


def complete(job, result):
    return bool(result and result.get('status') == 'DONE' and artifacts(job))


def provenance(branch):
    root = DATA / f'runs/sched/pull_forward/{branch}'
    assert (root / 'DONE').exists(), f'{branch} cache producer unfinished'
    for line in (root / 'provenance.txt').read_text().splitlines():
        if line.startswith('commit '): continue
        digest, name = line.split(maxsplit=1)
        assert hashlib.md5((REPO / name.lstrip('*')).read_bytes()).hexdigest() == digest, name


def preflight(branch):
    """Read real consumer inputs, including token coverage, before fitting/scoring."""
    import numpy as np
    from jevdrive import nq3_q6 as Q6
    if branch == 'q4b':
        from jevdrive import nq3_q4b as Q, night2_n3 as N3
        provenance('q4b')
        data = N3.p5_data()
        for model in Q.MODELS:
            _, x, names = Q.p5_sets(model, data)['p5']
            assert len(names) == len(x) and np.isfinite(x).all()
            for seed in Q.SEEDS:
                with np.load(N3._sel_files()[(model, seed)], allow_pickle=True) as z:
                    assert len(z['navtest_hydra']) > 0
    else:
        from jevdrive import elicit_e1 as E1
        provenance('q6')
        data = E1.wod_frames()
        assert np.isfinite(Q6._wod_vjepa(data['frame_name'])).all()
        dst = DATA / 'processed/navsim_vjepa2/navtest'
        tokens, values = np.load(dst / 'tokens.npy'), np.load(dst / 'mean.npy', mmap_mode='r')
        assert len(tokens) == len(values) and np.isfinite(values).all()
        for model in Q6.MODELS:
            with np.load(DATA / 'runs/navsim_zs/openpilot/navtest' / f'{model}_temporal.npz') as z:
                assert set(z['tokens']) <= set(tokens) and np.isfinite(z['temporal']).all()
                with np.load(DATA / E1.NAV_HEADS / f'navtest_ridge_late_{model}_temporal.npz') as p:
                    assert np.array_equal(z['tokens'], p['tokens']) and np.isfinite(p['poses']).all()
            for seed in Q6.SEEDS:
                assert len(Q6._heads(model, seed)) > 0
                Q6._tau(model, seed)
    print(f'{branch} input coverage, arrays and provenance verified', flush=True)


def external_writers(active):
    groups = {v['identity']['pgid'] for v in active if v.get('identity')}
    hits = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            ident = identity(int(p.parent.name))
            if not ident or ident['pgid'] in groups: continue
            argv = p.read_bytes().decode(errors='replace').split('\0')
            if any(a in ('jevdrive.nq3_q4b', 'jevdrive.nq3_q5', 'jevdrive.nq3_q6') or
                   a.endswith(('scripts/nq3_d.sh', 'scripts/nq3_d/q4b.sh', 'scripts/nq3_d/q5.sh', 'scripts/nq3_d/q6.sh')) for a in argv):
                hits.append(ident['pid'])
        except OSError: pass
    return hits


def admission(job, active, probe, rows):
    import sch_table as sch
    cores = cpuset(job['cpus'])
    if not cores <= os.sched_getaffinity(0): return None, 'CPU set unavailable'
    if any(cores & cpuset(v['cpus']) for v in active): return None, 'CPU slot reserved'
    # Inspect all specifically pinned live processes; broad system affinity is
    # not a reservation. Exclude our admitted process groups.
    groups = {v['identity']['pgid'] for v in active if v.get('identity')}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name); ident = identity(pid)
            affinity = os.sched_getaffinity(pid)
            if ident and ident['pgid'] not in groups and len(affinity) <= 40 and affinity & cores:
                return None, f'CPU affinity owned by PID {pid}'
        except (OSError, ValueError): pass
    if probe['cores_used'] + sum(len(cpuset(v['cpus'])) for v in active) + len(cores) > sch.CPU_CAP:
        return None, 'CPU headroom'
    promised = sum(int(r['workers']) * len(sch.gpus(r)) for r in rows
                   if r['workers'].isdigit() and not r['status'].startswith(('done', 'revoked')))
    pending = max(0, promised - sum(g['carla'] for g in probe['gpus']))
    projected = probe['pids'] + pending * sch.PIDS_PER_WORKER + 600 + job['pids'] + sum(v['pids'] for v in active)
    # 16000 is the CARLA admission cap. These jobs add no CARLA servers;
    # include promised CARLA plus X headroom, leaving >=1480 below hard max.
    if projected > min(probe['pids_max'] - 1480, 19000): return None, f'pending-inclusive threads {projected}'
    if not job.get('gpu_gb'): return -1, 'CPU admitted'
    candidates = []
    for g in probe['gpus']:
        if g['gpu'] == 1: continue
        promised_gpu = sum(int(r['workers']) for r in rows if r['workers'].isdigit()
                           and g['gpu'] in sch.gpus(r) and not r['status'].startswith(('done', 'revoked')))
        reserve = max(0, promised_gpu - g['carla']) * sch.VRAM_PER_WORKER_GB
        reserve += sum(v.get('gpu_gb', 0) for v in active if v.get('gpu') == g['gpu'])
        if g['used_gb'] + reserve + job['gpu_gb'] <= min(g['total_gb'] - 8, sch.VRAM_CAP_GB):
            candidates.append(g)
    if not candidates: return None, 'VRAM including pending reservations'
    return min(candidates, key=lambda g: (g['util'], g['used_gb']))['gpu'], 'GPU admitted'


def worker(manifest, name, gpu):
    job = next(j for j in json.loads(manifest.read_text())['jobs'] if j['id'] == name)
    dst = OUT / 'jobs' / name; dst.mkdir(parents=True, exist_ok=True)
    locks = []
    for key in sorted(job['locks']):
        lock = claim(OUT / 'claims' / f'{key}.lock')
        if lock is None:
            atomic(dst / 'result.json', {'status': 'WAIT', 'reason': 'output claim held'})
            return 3
        locks.append(lock)
    env = dict(os.environ, P5_SET='carla_p5v1_ba', PYTHONPATH=str(REPO),
               CUDA_VISIBLE_DEVICES=str(gpu) if gpu >= 0 else '', GPU=str(gpu), CPUS=job['cpus'],
               OMP_NUM_THREADS=str(job.get('threads', 8)), MKL_NUM_THREADS=str(job.get('threads', 8)),
               OPENBLAS_NUM_THREADS=str(job.get('threads', 8)), NUMBA_NUM_THREADS=str(job.get('threads', 8)),
               **job.get('env', {}))
    with (dst / 'log.txt').open('a') as log:
        child = subprocess.Popen(['taskset', '-c', job['cpus'], 'bash', '-euo', 'pipefail', '-c', job['command']],
                                 cwd=REPO, env=env, stdout=log, stderr=log,
                                 pass_fds=tuple(f.fileno() for f in locks))
        atomic(dst / 'command.pid.json', identity(child.pid))
        try: rc = child.wait(timeout=job['timeout_s'])
        except subprocess.TimeoutExpired:
            atomic(dst / 'result.json', {'status': 'WAIT', 'reason': 'timeout; recorded group terminated'})
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return 124
    status = 'DONE' if rc == 0 and artifacts(job) else 'WAIT'
    result = {'status': status, 'rc': rc, 'artifacts_ok': artifacts(job), 'finished': time.time()}
    atomic(dst / 'result.json', result)
    if status == 'DONE': atomic(dst / 'DONE', result)
    return rc


def run(manifest, once=False):
    import sch_table as sch
    config = json.loads(manifest.read_text()); jobs = config['jobs']
    OUT.mkdir(parents=True, exist_ok=True)
    lock = claim(OUT / 'dispatcher.lock')
    if lock is None: raise SystemExit('dispatcher already running')
    (OUT / 'pid').write_text(str(os.getpid()) + '\n')
    atomic(OUT / 'identity.json', identity(os.getpid()))
    state_path = OUT / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    known = OUT / 'manifest.sha256'
    if known.exists() and known.read_text().strip() != manifest_hash:
        raise SystemExit('manifest changed: explicit review required')
    known.write_text(manifest_hash + '\n')
    deadline = dt.datetime.fromisoformat(config['deadline']).timestamp()
    children = {}
    def event(name, value):
        if state.get(name) == value: return
        state[name] = value
        row = {'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'job': name, **value}
        for path in (OUT / 'log.txt', OUT / 'events.jsonl'):
            with path.open('a') as stream: stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        atomic(state_path, state)
    while True:
        for name, child in list(children.items()):
            if child.poll() is not None: del children[name]
        active = []
        for job in jobs:
            name = job['id']; previous = state.get(name, {})
            result_path = OUT / 'jobs' / name / 'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            if previous.get('status') == 'RUNNING' and alive(previous.get('identity')):
                active.append(previous); continue
            if complete(job, result): event(name, {'status': 'DONE'}); continue
            if previous.get('status') == 'RUNNING' or result:
                event(name, {'status': 'WAIT', 'reason': result or 'worker exited without result'}); continue
            if previous.get('status') == 'WAIT': continue
            # Reuse only explicit pre-existing markers AND complete output sets.
            if job.get('existing_done') and Path(expand(job['existing_done'])).exists() and artifacts(job):
                event(name, {'status': 'DONE', 'source': 'existing marker and artifacts'})
        pending = [j for j in jobs if state.get(j['id'], {}).get('status') not in ('DONE', 'WAIT', 'RUNNING')]
        if time.time() < deadline and pending:
            probe = sch.probe(); rows = sch.load()
            atomic(OUT / 'resources.json', {'time': time.time(), 'probe': probe, 'rows': rows})
            writers = external_writers(active)
            for job in pending:
                deps = [state.get(d, {}).get('status') for d in job.get('after', [])]
                if any(s == 'WAIT' for s in deps):
                    event(job['id'], {'status': 'WAIT', 'reason': 'dependency needs review'}); continue
                if any(s != 'DONE' for s in deps): continue
                if writers:
                    event(job['id'], {'status': 'PENDING', 'reason': f'external D writers {writers}'}); continue
                gpu, reason = admission(job, active, probe, rows)
                if gpu is None:
                    event(job['id'], {'status': 'PENDING', 'reason': reason}); continue
                dst = OUT / 'jobs' / job['id']; dst.mkdir(parents=True, exist_ok=True)
                with (dst / 'worker.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                        '--manifest', str(manifest), '--job', job['id'], '--gpu', str(gpu)],
                        cwd=REPO, stdout=log, stderr=log, start_new_session=True)
                value = {'status': 'RUNNING', 'identity': identity(child.pid), 'cpus': job['cpus'],
                         'gpu': gpu, 'gpu_gb': job.get('gpu_gb', 0), 'pids': job['pids'], 'started': time.time()}
                event(job['id'], value); active.append(value); children[job['id']] = child
        lines = ['# Independent queue', '', f'Updated UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}', '',
                 '| Job | State | Detail |', '|---|---|---|']
        lines += [f"| {j['id']} | {state.get(j['id'], {}).get('status', 'PENDING')} | {json.dumps(state.get(j['id'], {}))} |" for j in jobs]
        (OUT / 'STATUS.md').write_text('\n'.join(lines) + '\n')
        if once or (not active and (not pending or time.time() >= deadline)): break
        time.sleep(config.get('poll_s', 45))
    atomic(OUT / 'EXIT.json', {'time': time.time(), 'all_done': all(state.get(j['id'], {}).get('status') == 'DONE' for j in jobs)})
    lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'worker', 'preflight'])
    ap.add_argument('--manifest', type=Path, default=DEFAULT)
    ap.add_argument('--job'); ap.add_argument('--gpu', type=int, default=-1)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if args.mode == 'preflight': preflight(args.job)
    elif args.mode == 'worker': sys.exit(worker(args.manifest, args.job, args.gpu))
    else: run(args.manifest, args.once)


if __name__ == '__main__': main()
