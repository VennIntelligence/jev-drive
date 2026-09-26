#!/usr/bin/env python3
"""Capacity-aware successor; adopts existing workers without restarting them.

Only registered non-CARLA jobs run. Allocations use free cores in the reviewed
pool, resident resources plus unmaterialized claims, and bounded launch bursts.
The original dispatcher/manifest remain immutable for their live workers.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cx_orchestrator as base
import sch_table as sch

REPO, DATA, OUT = base.REPO, base.DATA, base.OUT
DEFAULT = REPO / 'scripts/cx_orchestration_v2.json'


def processes():
    rows = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z': continue
            rows.append(dict(pid=int(path.name), pgid=int(stat[2]), threads=int(stat[17]),
                             cores=sorted(os.sched_getaffinity(int(path.name)))))
        except (OSError, ValueError): pass
    return rows


def resident(active, procs, gpu_apps):
    """Count owned groups, and pinned Ray processes even if they detached."""
    result = []
    for value in active:
        cores = base.cpuset(value['cpus'])
        group = value['identity']['pgid']
        members = [p for p in procs if p['pgid'] == group or (p['cores'] and set(p['cores']) <= cores)]
        pids = {p['pid'] for p in members}
        result.append(dict(value, resident_pids=sum(p['threads'] for p in members),
                           resident_gpu_gb=sum(gb for pid, gb in gpu_apps if pid in pids)))
    return result


def gpu_apps():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in raw.splitlines():
        try:
            pid, mb = line.split(','); result.append((int(pid), float(mb) / 1024))
        except ValueError: pass
    return result


def choose_cores(job, active, procs):
    pool = base.cpuset(job['cpu_pool']) & os.sched_getaffinity(0)
    for value in active: pool -= base.cpuset(value['cpus'])
    groups = {v['identity']['pgid'] for v in active}
    for proc in procs:
        if proc['pgid'] not in groups and len(proc['cores']) <= 40:
            pool -= set(proc['cores'])
    if len(pool) < job['min_cores']: return None
    return sorted(pool)[:min(job['max_cores'], len(pool))]


def budget(job, cores, active, probe, rows):
    """Admission for CPU/GPU jobs, never admission for new CARLA servers."""
    fresh = [v for v in active if time.time() - v['started'] < 30]
    if probe['cores_used'] + len(cores) + sum(len(base.cpuset(v['cpus'])) for v in fresh) > sch.CPU_CAP:
        return None, 'CPU load plus unmaterialized launches'
    promised = sum(int(r['workers']) * len(sch.gpus(r)) for r in rows
                   if r['workers'].isdigit() and not r['status'].startswith(('done', 'revoked')))
    pending = max(0, promised - sum(g['carla'] for g in probe['gpus']))
    # Existing CARLA runners wait at their own 17000 thread gate. They cannot
    # all realize stale peak reservations simultaneously. Budget the admissible
    # backlog up to that gate plus one 400-thread race; retain 600 for debug.
    carla_burst = min(pending * sch.PIDS_PER_WORKER,
                      max(0, 17000 - probe['pids']) + sch.PIDS_PER_WORKER)
    unrealized = sum(max(0, v['pids'] - v.get('resident_pids', 0)) for v in active)
    projected = probe['pids'] + carla_burst + 600 + unrealized + job['pids']
    if projected > probe['pids_max'] - 1024:
        return None, f'resident + bounded CARLA burst + unrealized claims {projected}'
    if not job.get('gpu_gb'): return -1, 'CPU admitted'
    candidates = []
    for gpu in probe['gpus']:
        if gpu['gpu'] == 1: continue
        target = sum(int(r['workers']) for r in rows if r['workers'].isdigit() and
                     gpu['gpu'] in sch.gpus(r) and not r['status'].startswith(('done', 'revoked')))
        reserve = max(0, target - gpu['carla']) * sch.VRAM_PER_WORKER_GB
        reserve += sum(max(0, v.get('gpu_gb', 0) - v.get('resident_gpu_gb', 0))
                       for v in active if v.get('gpu') == gpu['gpu'])
        if gpu['used_gb'] + reserve + job['gpu_gb'] <= min(gpu['total_gb'] - 8, sch.VRAM_CAP_GB):
            candidates.append(gpu)
    if not candidates: return None, 'VRAM including unmaterialized claims'
    return min(candidates, key=lambda g: (g['util'], g['used_gb']))['gpu'], 'GPU admitted'


def measure(name):
    """One bounded 30-second before/after window, not model-driven polling."""
    samples = []
    for i in range(7):
        cpu = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        samples.append(dict(t=time.time(), cpu=int(cpu['usage_usec']),
                            pids=int(Path('/sys/fs/cgroup/pids.current').read_text()),
                            gpu=[list(map(int, line.split(','))) for line in raw.splitlines()]))
        if i < 6: time.sleep(5)
    state = json.loads((OUT / 'state.json').read_text())
    result = dict(samples=samples, elapsed_s=samples[-1]['t'] - samples[0]['t'], cores=(samples[-1]['cpu'] - samples[0]['cpu']) /
                  (samples[-1]['t'] - samples[0]['t']) / 1e6, state=state,
                  gpu_mean={g: sum(next(r[2] for r in s['gpu'] if r[0] == g) for s in samples) / len(samples)
                            for g in range(7)})
    base.atomic(OUT / (name + '.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))


def validate_adoption(job, previous, old):
    """A successor adopts the immutable command actually launched, not a template."""
    if previous.get('launch'):
        launched = json.loads(Path(previous['launch']).read_text())['jobs']
        original = next(j for j in launched if j['id'] == job['id'])
    else:
        original = old[job['id']]
    assert job['command'] == original['command'], 'live command changed'


def run(manifest):
    config = json.loads(manifest.read_text()); jobs = config['jobs']
    lock = base.claim(OUT / 'dispatcher.lock')
    if lock is None: raise SystemExit('another dispatcher still owns the lock')
    (OUT / 'pid').write_text(str(os.getpid()) + '\n')
    base.atomic(OUT / 'identity.json', base.identity(os.getpid()))
    state_path = OUT / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Adopt old workers by stable job ID and exact command, never restart them.
    old = {j['id']: j for j in json.loads(base.DEFAULT.read_text())['jobs']}
    for job in jobs:
        if state.get(job['id'], {}).get('status') == 'RUNNING':
            validate_adoption(job, state[job['id']], old)
    base.atomic(OUT / 'successor.json', dict(pid=os.getpid(), manifest=str(manifest),
        sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), adopted=state))
    deadline = dt.datetime.fromisoformat(config['deadline']).timestamp()
    children = {}
    def event(name, value):
        if state.get(name) == value: return
        state[name] = value
        row = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), job=name, **value)
        for path in (OUT / 'log.txt', OUT / 'events.jsonl'):
            with path.open('a') as stream: stream.write(json.dumps(row) + '\n')
        base.atomic(state_path, state)
        print(json.dumps(row), flush=True)
    while True:
        for name, child in list(children.items()):
            if child.poll() is not None: del children[name]
        active = []
        for job in jobs:
            name = job['id']; previous = state.get(name, {})
            result_path = OUT / 'jobs' / name / 'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            if previous.get('status') == 'RUNNING' and base.alive(previous.get('identity')):
                active.append(previous); continue
            if base.complete(job, result): event(name, dict(status='DONE')); continue
            if previous.get('status') == 'RUNNING' or result:
                event(name, dict(status='WAIT', reason=result or 'worker exited without result')); continue
            if previous.get('status') == 'WAIT': continue
            if job.get('existing_done') and Path(base.expand(job['existing_done'])).exists() and base.artifacts(job):
                event(name, dict(status='DONE', source='existing marker and artifacts'))
        pending = [j for j in jobs if state.get(j['id'], {}).get('status') not in ('DONE', 'WAIT', 'RUNNING')]
        if pending and time.time() < deadline:
            probe = sch.probe(); rows = base.effective_rows(sch.load()); procs = processes()
            active = resident(active, procs, gpu_apps())
            base.atomic(OUT / 'resources.json', dict(time=time.time(), probe=probe, rows=rows, active=active))
            writers = base.external_writers(active)
            for job in pending:
                deps = [state.get(d, {}).get('status') for d in job.get('after', [])]
                if 'WAIT' in deps:
                    event(job['id'], dict(status='WAIT', reason='dependency needs review')); continue
                if any(s != 'DONE' for s in deps): continue
                if writers:
                    event(job['id'], dict(status='PENDING', reason=f'external D writers {writers}')); continue
                cores = choose_cores(job, active, procs)
                if cores is None:
                    event(job['id'], dict(status='PENDING', reason='insufficient free pool cores')); continue
                gpu, reason = budget(job, cores, active, probe, rows)
                if gpu is None:
                    event(job['id'], dict(status='PENDING', reason=reason)); continue
                dst = OUT / 'jobs' / job['id']; dst.mkdir(parents=True, exist_ok=True)
                launch = dict(job, cpus=','.join(map(str, cores)), threads=min(job['threads'], len(cores)))
                launch_file = dst / f'launch-{time.time_ns()}.json'
                base.atomic(launch_file, dict(jobs=[launch]))
                with (dst / 'worker.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, str(REPO / 'scripts/cx_orchestrator.py'),
                        'worker', '--manifest', str(launch_file), '--job', job['id'], '--gpu', str(gpu)],
                        cwd=REPO, stdout=log, stderr=log, start_new_session=True)
                value = dict(status='RUNNING', identity=base.identity(child.pid), cpus=launch['cpus'],
                    gpu=gpu, gpu_gb=job.get('gpu_gb', 0), pids=job['pids'], started=time.time(), launch=str(launch_file))
                event(job['id'], value); active.append(value); children[job['id']] = child
        lines = ['# Independent queue v2', '', f'Updated UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}', '',
                 '| Job | State | Detail |', '|---|---|---|']
        lines += [f"| {j['id']} | {state.get(j['id'], {}).get('status', 'PENDING')} | {json.dumps(state.get(j['id'], {}))} |" for j in jobs]
        (OUT / 'STATUS.md').write_text('\n'.join(lines) + '\n')
        if not active and (not pending or time.time() >= deadline): break
        time.sleep(config.get('poll_s', 15))
    base.atomic(OUT / 'EXIT.json', dict(time=time.time(), all_done=all(state.get(j['id'], {}).get('status') == 'DONE' for j in jobs)))
    lock.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'measure'])
    ap.add_argument('--manifest', type=Path, default=DEFAULT)
    ap.add_argument('--name', default='measurement')
    args = ap.parse_args()
    if args.mode == 'measure': measure(args.name)
    else: run(args.manifest)
