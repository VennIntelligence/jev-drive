#!/usr/bin/env python3
"""Authorized GPU1 K/G staged pilots only; no full batch, GO or prep marker.

Run in tmux. Each original pilot performs 1 then 10 routes with the registered
checks. Recheck capacity before both stages, preserve failures per candidate,
and immediately continue the next independent candidate. World reuse stays off.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/night_queue_4/archive", "scripts",)]
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import sch_table as sch
from cx_controller import atomic, claim, process_snapshot, same, identity
from cx_owned_process import refresh

DATA = Path(os.environ.get('DATA_DIR', '/root/autodl-tmp/ujs'))
OUT = DATA / 'runs/nq4/cx/gk-pilots'
G = DATA / 'runs/nq4/gk'
ARMS = 'cx_pilot_20260927'
DEADLINE = dt.datetime.fromisoformat('2026-09-27T23:59:00+09:00').timestamp()
QUEUE = [(f'k{i}_unseen', 'k', 'k220') for i in range(4)] + [
    ('pdm', 'ghost', 'g'), ('bridgedrive', 'ghost', 'g'), ('blue', 'ghost', 'g'),
    ('q2', 'ghost', 'gob'), ('pdm', 'shift', 'g'), ('pdm', 'swap', 'g')]


def event(kind, **detail):
    OUT.mkdir(parents=True, exist_ok=True)
    row = dict(time=time.time(), kind=kind, **detail)
    for name in ('events.jsonl', 'log.txt'):
        with (OUT / name).open('a') as f:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def status(message):
    (OUT / 'STATUS.md').write_text(f'# G/K staged pilots\n\n{dt.datetime.now(dt.timezone.utc).isoformat()}\n\n{message}\n')


def bound_ports():
    ports = set()
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/net/' + name).read_text().splitlines()[1:]:
            ports.add(int(line.split()[1].rsplit(':', 1)[1], 16))
    return ports


def admission(probe, pending, workers, conflicts, clean_indices):
    gpu = next(g for g in probe['gpus'] if g['gpu'] == 1)
    reasons = list(conflicts)
    if probe['pids'] + 400 * (pending + workers) + 100 > 16000:
        reasons.append('pids including pending B servers and pilot reservation')
    if probe['cores_used'] + 2.5 * (pending + workers) + 2 > 165:
        reasons.append('CPU including pending workers')
    if gpu['used_gb'] + 7.5 * workers + 2 > 88:
        reasons.append('GPU1 VRAM')
    if gpu['carla'] + workers > 4:
        reasons.append('debug CARLA limit')
    if clean_indices < min(workers + 2, 10):
        reasons.append('reserved ports not free')
    return reasons


def admit(workers):
    if workers not in (1, 2):
        raise ValueError('Pilot worker grant is 1..2')
    while time.time() < DEADLINE:
        try:
            probe = sch.probe()
            rows = sch.load()
            grant = next(r for r in rows if r['lane'] == 'nq4-gk-pilot')
            if (grant['gpus'], grant['idx0'], grant['idx_span'], grant['cpus']) != ('1', '150', '10', '204-207'):
                raise RuntimeError('Registered pilot allocation changed; waiting for owner review')
            if int(grant['workers']) < workers:
                raise RuntimeError('Insufficient registered pilot workers')
            current = (DATA / 'runs/nq3/b/CURRENT').read_text().split()
            claims = len(list((DATA / f'runs/nq3/b/arms/{current[0]}/s{current[1]}/claims').glob('*.lock')))
            batch_servers = sum(g['carla'] for g in probe['gpus'] if g['gpu'] in (0, 2, 3, 4, 5))
            pending = max(0, claims - batch_servers)
            ports = bound_ports()
            clean = sum(not (ports & {2000+50*i, 2001+50*i, 2002+50*i, *range(8000+50*i, 8050+50*i)})
                        for i in range(150, 160))
            reasons = admission(probe, pending, workers, sch.conflicts(rows), clean)
            if not reasons:
                event('ADMITTED', workers=workers, pending_B=pending, probe=probe, clean_indices=clean)
                status(f'ADMITTED {workers} pilot worker(s); GPU1, indices150–159, CPU204–207.')
                return 0
            event('WAIT_CAPACITY', reasons=reasons, workers=workers, pending_B=pending, probe=probe)
            status(f'WAIT_CAPACITY: {reasons}; pids={probe["pids"]}, pending_B={pending}.')
        except Exception as exc:
            event('WAIT_UNKNOWN', detail=f'{type(exc).__name__}: {exc}')
            status(f'WAIT_UNKNOWN: {exc}')
        time.sleep(30)
    event('DEADLINE_NO_NEW_STAGE')
    return 4


def result(cand, variant):
    directory = G / f'{ARMS}_pilot/{cand}.{variant}'
    if (directory / 'PASS').exists():
        checks = [json.loads((directory / f'stage{i}.json').read_text()) for i in (1, 2)]
        if all(c.get('pass') is True for c in checks):
            return 'PILOT_PASS'
        return 'INVALID_PASS_EVIDENCE'
    if (G / f'{ARMS}_blocked/{cand}').exists():
        return 'PILOT_FAIL'
    return 'TECHNICAL_WAIT'


def historical_pilot(cand, variant):
    """Only reuse an existing complete ten-route staged pilot, never a two-route smoke."""
    d = G / f'smoke_pilot/{cand}.{variant}'
    if not (d / 'PASS').exists():
        return False
    try:
        checks = [json.loads((d / f'stage{i}.json').read_text()) for i in (1, 2)]
        n = checks[1]['facts']['routes']
        done = list((G / f'smoke/{cand}/{variant}/s0/done').glob('*.json'))
        return all(c.get('pass') is True for c in checks) and n == 10 and len(done) >= n
    except (OSError, ValueError, KeyError):
        return False


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = claim(OUT / 'owner.lock')
    (OUT / 'pid').write_text(str(os.getpid()))
    path = OUT / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    env = dict(os.environ, ARMS=ARMS, GPUS='1', WORKERS='2', SIDX0='150', SPAN='10', CPUS='204-207', CTX='pilot',
               CX_GK_PILOT_GUARD='1', B2D_NQ4_REUSE_WORLD='0', PYTHONUNBUFFERED='1')
    with lock:
        event('OWNER_START', pid=os.getpid(), queue=QUEUE, full_batch=False)
        for cand, variant, routeset in QUEUE:
            key = f'{cand}.{variant}'
            previous = state.get(key, {})
            if previous.get('state') in ('PILOT_PASS', 'REUSED_PILOT_PASS', 'PILOT_FAIL', 'TECHNICAL_WAIT', 'INVALID_PASS_EVIDENCE'):
                continue
            if not previous and historical_pilot(cand, variant):
                state[key] = dict(state='REUSED_PILOT_PASS', source=str(G / f'smoke_pilot/{key}'))
                atomic(path, state)
                event('REUSED_PILOT_PASS', candidate=key, **state[key])
                continue
            if previous.get('process'):
                while same(previous['process'], process_snapshot()):
                    status(f'ADOPTED {key}: PID{previous["process"]["pid"]}; no duplicate launch.')
                    time.sleep(30)
                previous['state'] = result(cand, variant)
                atomic(path, state)
                continue
            if time.time() >= DEADLINE:
                status('DEADLINE: no new pilot. Existing outputs retained.')
                return
            if cand.startswith('k') and not (DATA / 'runs/nq4/k/READY').exists():
                state[key] = dict(state='TECHNICAL_WAIT', reason='K READY missing')
                atomic(path, state)
                continue
            if cand == 'q2' and not (G / 'heads_xfit/q2/READY').exists():
                state[key] = dict(state='TECHNICAL_WAIT', reason='Formal crossfit Q2 READY missing')
                atomic(path, state)
                continue
            # Never reuse an old partial directory or reset its claims/attempt budget.
            if (G / f'{ARMS}_pilot/{key}').exists():
                state[key] = dict(state=result(cand, variant))
                atomic(path, state)
                continue
            command = ['bash', 'experiments/night_queue_4/archive/nq4_gk.sh', 'pilot', cand, variant, '0', routeset]
            with (OUT / f'{key}.log').open('a') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record, _ = refresh(OUT / f'{key}.owned.json', child.pid)
            state[key] = dict(state='RUNNING', process=record['root'], command=command)
            atomic(path, state)
            event('PILOT_OWNER', candidate=key, process=record['root'], command=command)
            while child.poll() is None:
                refresh(OUT / f'{key}.owned.json')
                time.sleep(10)
            state[key].update(state=result(cand, variant), rc=child.returncode)
            atomic(path, state)
            event('PILOT_RESULT', candidate=key, **state[key])
        status('Pilot queue settled; inspect per-candidate evidence. No full batch authorized by this runner.')
        event('QUEUE_SETTLED', state=state)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'admit'])
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    raise SystemExit(admit(args.workers) if args.mode == 'admit' else run())
