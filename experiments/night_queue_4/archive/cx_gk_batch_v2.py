#!/usr/bin/env python3
"""Fill idle batch GPUs with registered K seeds as each staged pilot passes.

One owner and one writer per candidate/seed; official 220 routes, original
agents/P7, seed set 0/1/2 and pilot thresholds. Independent GPU1 pilots continue.
No G prep marker or scientific approval is manufactured by this scheduler.
"""
import sys as _sys, pathlib as _pl  # restructure: dirs of the script modules this file imports by bare name
_sys.path[:0] = [str(_pl.Path(__file__).resolve().parents[3] / _d) for _d in ("experiments/night_queue_4/archive", "scripts",)]
import argparse
import fcntl
import sys
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import statistics
import time

import sch_table as sch
from cx_controller import atomic, claim, identity, process_snapshot, same
from cx_owned_process import refresh

DATA = Path(os.environ.get('DATA_DIR', '/root/autodl-tmp/ujs'))
G = DATA / 'runs/nq4/gk'
OUT = DATA / 'runs/nq4/cx/gk-batch'
DEADLINE = dt.datetime.fromisoformat('2026-09-27T23:59:00+09:00').timestamp()
# K0/K3 then K1/K2 is the original registered K batch order.
CANDIDATES = ['k0_unseen', 'k3_unseen', 'k1_unseen', 'k2_unseen']


def registered_k_checks(cand, done):
    """Apply the additional existing K READY checklist, without changing its limits."""
    failed, speeds, activations = [], [], []
    blocked, ref_blocked, changed, rule_ticks = 0, 0, 0, 0
    for f in done:
        rec = json.loads(f.read_text())
        attempt = f.parent.parent / 'attempts' / f.stem / str(rec['attempt'])
        summary = json.loads((attempt / 'agent_summary.json').read_text())
        cfg = summary['config']
        if cfg.get('arm') != cand[:2] or cfg.get('k_view') != 'unseen' or summary.get('n_plans', 0) <= 0:
            failed.append(f'{f.stem}: agent identity or plans')
        for log in (attempt.parent).glob('*/route.log'):
            if 'Traceback (most recent call last)' in log.read_text(errors='replace'):
                failed.append(f'{f.stem}: route traceback')
        mine = json.loads((attempt / 'results.json').read_text())['_checkpoint']['records'][0]
        refdir = DATA / f'runs/nq3/b/arms/cl3/s0'
        ref = json.loads((refdir / 'done' / f.name).read_text())
        theirs = json.loads((refdir / 'attempts' / f.stem / str(ref['attempt']) / 'results.json').read_text())['_checkpoint']['records'][0]
        blocked += bool(mine.get('infractions', {}).get('vehicle_blocked'))
        ref_blocked += bool(theirs.get('infractions', {}).get('vehicle_blocked'))
        for line in (attempt / 'plans.jsonl').read_text().splitlines():
            plan = json.loads(line)
            if not all(math.isfinite(float(v)) for point in plan['path'] for v in point):
                failed.append(f'{f.stem}: nonfinite path')
            if cand[:2] != 'k0':
                speed = float(plan['v_target'])
                if not 0 <= speed <= 20:
                    failed.append(f'{f.stem}: v_target range')
                if plan.get('speed', 0) > 2:
                    speeds.append(speed)
            if cand[:2] == 'k3':
                value = float(plan['g3'])
                if not 0 <= value <= 1:
                    failed.append(f'{f.stem}: g3 range')
                activations.append(value > .5)
        if cand[:2] in ('k2', 'k3'):
            ticks = [json.loads(line) for line in (attempt / 'ticks.jsonl').read_text().splitlines()]
            if any('rules' not in tick for tick in ticks):
                failed.append(f'{f.stem}: missing rules record')
            rules = [tick['rules'] for tick in ticks if 'rules' in tick]
            if not any(r.get('stop_det') is not None for r in rules):
                changed += sum(r['rule_in'] != r['rule_out'] for r in rules)
                rule_ticks += len(rules)
    if blocked > ref_blocked + 2:
        failed.append('blocked exceeds CL3 + 2')
    if cand[:2] != 'k0' and (not speeds or statistics.median(speeds) <= 2):
        failed.append('v_target moving median')
    if cand[:2] == 'k3' and (not activations or not .005 <= statistics.mean(activations) <= .2):
        failed.append('g3 activation share')
    if rule_ticks and changed / rule_ticks >= .05:
        failed.append('rules change >=5% outside stop-sign routes')
    return dict(pass_check=not failed, failed=failed, blocked=blocked, reference_blocked=ref_blocked,
                moving_target_median=statistics.median(speeds) if speeds else None,
                g3_share=statistics.mean(activations) if activations else None,
                rules_changed=changed, rules_ticks=rule_ticks)


def gate(cand):
    if cand not in CANDIDATES or not (DATA / 'runs/nq4/k/READY').is_file():
        return False
    p = G / f'cx_pilot_20260927_pilot/{cand}.k'
    if not (p / 'PASS').is_file():
        return False
    try:
        stages = [json.loads((p / f'stage{i}.json').read_text()) for i in (1, 2)]
        done = list((G / f'cx_pilot_20260927_k/{cand}/s0/done').glob('*.json'))
        valid = (all(s.get('pass') is True for s in stages) and stages[1]['facts']['routes'] == 10
                and stages[1]['facts']['finished'] == 10 and len(done) == 10
                and all(json.loads(f.read_text()).get('status') == 'finished' for f in done))
        if not valid:
            return False
        check = registered_k_checks(cand, done)
        atomic(OUT / 'gates' / f'{cand}.json', check)
        return check['pass_check']
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        atomic(OUT / 'gates' / f'{cand}.json', dict(pass_check=False, state='WAIT_MISSING_EVIDENCE', reason=str(exc)))
        return False


def event(kind, **detail):
    row = dict(time=time.time(), kind=kind, **detail)
    for name in ('events.jsonl', 'log.txt'):
        with (OUT / name).open('a') as f:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def prepare(cand, seed):
    output = G / f'arms_k/{cand}/s{seed}'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Existing unowned batch output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source = G / f'cx_pilot_20260927_k/{cand}/s0'
    requested = json.loads((source / 'requested.json').read_text())
    if len(requested) != 220 or len(set(requested)) != 220:
        raise RuntimeError('Pilot parent does not identify the registered 220 routes')
    atomic(output / 'requested.json', requested)
    if seed == 0:
        for finished in (source / 'done').glob('*.json'):
            rid = finished.stem
            shutil.copytree(source / 'attempts' / rid, output / 'attempts' / rid, copy_function=os.link)
            (output / 'done').mkdir(exist_ok=True)
            os.link(finished, output / 'done' / finished.name)
    return output


def free_block(rows, ports):
    for first in range(0, 487):
        candidate = dict(lane='candidate', gpus='0', workers='6', idx0=str(first), idx_span='8',
                         cpus='118-133', status='batch', go='-')
        if sch.conflicts(rows + [candidate]):
            continue
        used = {p for i in range(first, first+8) for p in
                (2000+50*i, 2001+50*i, 2002+50*i, *range(8000+50*i, 8050+50*i))}
        if not ports & used:
            return first
    return None


def ports_bound():
    return {int(line.split()[1].rsplit(':', 1)[1], 16) for name in ('tcp', 'tcp6')
            for line in Path('/proc/net/' + name).read_text().splitlines()[1:]}


def settle(job):
    output = Path(job['out'])
    requested = json.loads((output / 'requested.json').read_text())
    finished = [rid for rid in requested if (output / 'done' / f'{rid}.json').is_file()
                and json.loads((output / 'done' / f'{rid}.json').read_text()).get('status') == 'finished']
    enough = len(finished) >= .9 * len(requested)
    job.update(state='DONE' if enough else 'FAILED', finished=len(finished), requested=len(requested))
    if enough:
        atomic(output / 'DONE', dict(cand=job['cand'], variant='k', seed=job['seed'], done=len(finished),
                                    requested=len(requested), finished=time.time(), owner='cx_gk_batch'))
    else:
        atomic(output / 'ERROR', dict(reason='More than 10% missing; original completion gate', done=len(finished), requested=len(requested)))
    with (sch.TABLE.parent / 'table.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = sch.load()
        for row in rows:
            if row['lane'] == job['lane']:
                row['status'] = 'done ' + job['state']
        sch.save(rows)
    event('SETTLED', job=job)



_WAIT_REASONS = {}


def wait_reason(key, reason, **detail):
    record = dict(reason=reason, **detail)
    if _WAIT_REASONS.get(key) != record:
        _WAIT_REASONS[key] = record
        event('WAIT', key=key, **record)
    atomic(OUT / 'admission.json', _WAIT_REASONS)


def pending_workers(job, carla_count):
    # Before its runner starts reserve all workers; later only actual route claims
    # can start/restart a server. Idle tail workers are not future reservations.
    if (Path(job['runtime']) / 'owned-runners').exists():
        expected = min(job['workers'], len(list((Path(job['out']) / 'claims').glob('*.lock'))))
    else:
        expected = job['workers']
    return max(0, expected - carla_count)


def choose_workers(probe, pending, device):
    for workers in range(6, 0, -1):
        if (probe['pids'] + 400 * (pending + workers) + 100 <= 16000
                and probe['cores_used'] + 2.5 * (pending + workers) + 2 <= 165
                and device['used_gb'] + 9 * workers + 2 <= 88):
            return workers
    return 0


def release_completed_b(rows):
    base = DATA / 'runs/nq3/b'
    if not (base / 'DONE').is_file() or (base / 'QUEUE').read_text().strip():
        return
    if not all((base / 'results' / name).is_file() and (base / 'results' / name).stat().st_size
               for name in ('arms.csv', 'per_route.csv', 'summary.md')):
        return
    for process in rows.values():
        argv = process['argv']
        if '--out' in argv and argv[argv.index('--out') + 1].startswith(str(base / 'arms') + '/'):
            return
        # A reparented status-only watcher does not retain ports; any actual chain
        # with a parent still prevents revocation, including a newly resumed chain.
        if 'experiments/night_queue_3/archive/nq3_b.sh' in argv and 'chain' in argv and process['ppid'] != 1:
            return
    with (sch.TABLE.parent / 'table.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = sch.load()
        row = next((r for r in table if r['lane'] == 'nq3-b'), None)
        if row and row['gpus'] != '-':
            previous = dict(row)
            row.update(gpus='-', workers='-', idx0='-', idx_span='-', cpus='-',
                       status='revoked completed B; no live output writer')
            sch.save(table)
            event('RELEASE_COMPLETED_B', previous=previous)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = claim(OUT / 'owner.lock')
    statefile = OUT / 'state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    (OUT / 'pid').write_text(str(os.getpid()))
    children = {}
    with lock:
        event('OWNER_START', pid=os.getpid(), candidates=CANDIDATES)
        while True:
            rows = process_snapshot()
            release_completed_b(rows)
            for key, job in state.items():
                if job['state'] == 'RUNNING':
                    if key in children:
                        children[key].poll()
                    record, snapshot = refresh(Path(job['runtime']) / 'owner.json')
                    if any(same(p, snapshot) for p in record['members']):
                        continue
                    settle(job)
            atomic(statefile, state)
            pending = [(c, s) for c in CANDIDATES if gate(c) for s in range(3) if f'{c}.s{s}' not in state]
            if time.time() < DEADLINE and pending:
                probe = sch.probe()
                active = [j for j in state.values() if j['state'] == 'RUNNING']
                # Reserve startup workers until their CARLA servers are visible, plus GPU1 pilot headroom.
                unmet = sum(pending_workers(j, next(g['carla'] for g in probe['gpus'] if g['gpu'] == j['gpu'])) for j in active)
                debug_pending = max(0, 2 - next(g['carla'] for g in probe['gpus'] if g['gpu'] == 1))
                for gpu in [0, 6, 5, 2, 3, 4]:
                    if not pending:
                        break
                    device = next(g for g in probe['gpus'] if g['gpu'] == gpu)
                    if device['used_gb'] > 1 or device['carla'] or any(j['gpu'] == gpu for j in active):
                        continue
                    # No live B runner or other GPU owner may be displaced by a grant.
                    if any('--gpu-rank' in r['argv'] and r['argv'][r['argv'].index('--gpu-rank')+1] == str(gpu) for r in rows.values()):
                        continue
                    workers = choose_workers(probe, unmet + debug_pending, device)
                    if not workers:
                        wait_reason(str(gpu), 'CAPACITY', pids=probe['pids'], pending=unmet, debug_pending=debug_pending, cpu=probe['cores_used'])
                        continue
                    c, seed = pending.pop(0); key = f'{c}.s{seed}'
                    lane = 'cx-gk-batch-' + str(gpu)
                    with (sch.TABLE.parent / 'table.lock').open('a') as tablelock:
                        import fcntl
                        fcntl.flock(tablelock, fcntl.LOCK_EX)
                        table = sch.load()
                        index = free_block(table, ports_bound())
                        if index is None:
                            wait_reason(str(gpu), 'NO_LEGAL_INDEX_BLOCK', registered_lanes=[r['lane'] for r in table if r['gpus'] != '-' and not r['status'].startswith(('done','revoked'))])
                            pending.insert(0, (c, seed)); continue
                        cpus = {0:'118-133', 6:'150-165', 5:'134-149', 2:'166-179', 3:'0-15', 4:'16-31'}[gpu]
                        table = [r for r in table if r['lane'] != lane]
                        table.append(dict(lane=lane, gpus=str(gpu), workers=str(workers), idx0=str(index), idx_span='8',
                                          cpus=cpus, status='batch verified pilot '+key, go='-'))
                        sch.save(table)
                    output = prepare(c, seed)
                    runtime = OUT / 'leases' / key; runtime.mkdir(parents=True, exist_ok=True)
                    command = ['bash', 'experiments/night_queue_4/archive/cx_gk_batch_worker.sh', c, str(seed), str(gpu), str(workers), str(index), '8', cpus, str(runtime)]
                    with (runtime / 'worker.log').open('a') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    record, _ = refresh(runtime / 'owner.json', child.pid)
                    state[key] = dict(state='RUNNING', cand=c, seed=seed, gpu=gpu, workers=workers, index=index, cpus=cpus,
                                      lane=lane, out=str(output), runtime=str(runtime), process=record['root'], command=command)
                    children[key] = child
                    _WAIT_REASONS.pop(str(gpu), None)
                    atomic(OUT / 'admission.json', _WAIT_REASONS)
                    atomic(statefile, state); event('LAUNCHED', key=key, job=state[key])
                    active.append(state[key]); unmet += workers
                    time.sleep(20)
            if not pending:
                wait_reason('queue', 'NO_PASSED_UNSTARTED_SEED', pending_pilots=[c for c in CANDIDATES if not gate(c)])
            else:
                _WAIT_REASONS.pop('queue', None)
            (OUT / 'STATUS.md').write_text('# K batch refill\n\n' + '\n'.join(f'- {key}: {j["state"]}, GPU{j["gpu"]}' for key,j in state.items()) + '\n')
            if time.time() >= DEADLINE and not any(j['state'] == 'RUNNING' for j in state.values()):
                event('DEADLINE_SETTLED'); return
            time.sleep(20)


def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    with claim(OUT / 'supervisor.lock'), (OUT / 'supervisor.log').open('a', buffering=1) as log:
        while True:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'run', '--out', str(G / 'arms_k')],
                                    stdout=log, stderr=subprocess.STDOUT)
            log.write(f'{time.time()} scheduler exit rc={result.returncode}\n')
            if result.returncode == 0:
                return
            time.sleep(30)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'gate', 'supervise'])
    ap.add_argument('candidate', nargs='?')
    ap.add_argument('--out', type=Path, default=G / 'arms_k')
    args = ap.parse_args()
    if args.out != G / 'arms_k':
        raise ValueError('Only the registered K batch output is supported')
    if args.mode == 'supervise':
        supervise()
    else:
        raise SystemExit((0 if gate(args.candidate) else 1) if args.mode == 'gate' else run())
