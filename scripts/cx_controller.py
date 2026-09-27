#!/usr/bin/env python3
"""Single box-local scheduling owner. No scientific verdicts and no process signals.

run holds the global grant lock and retired dispatcher's lock. inbox is durable,
not an LLM wake-up channel. Existing chains retain sole ownership of outputs.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import glob
import json
import shlex
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / 'scripts/cx_controller_registry.json'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.controller-')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: f.close(); raise RuntimeError(f'another owner holds {path}')
    return f


def process_snapshot():
    if not Path('/proc/self/stat').exists(): raise RuntimeError('Linux /proc unavailable; node is UNKNOWN')
    rows = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if s[0] == 'Z': continue
            rows[int(p.name)] = dict(pid=int(p.name), start_ticks=s[19], ppid=int(s[1]), pgid=int(s[2]),
                                    sid=int(s[3]), argv=(p / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1])
        except (FileNotFoundError, ProcessLookupError): continue
        # Permission and incomplete reads are UNKNOWN, never proof of exit.
    return rows


def identity(p):
    return {k: p[k] for k in ('pid', 'start_ticks', 'pgid', 'sid')}


def same(record, rows):
    return bool(record and record['pid'] in rows and rows[record['pid']]['start_ticks'] == record['start_ticks'])


def matches(p, patterns):
    tokens = [os.path.basename(t) if t.endswith('.sh') else t for t in p['argv']]
    return any(all(token in tokens for token in pattern) for pattern in patterns)


def members(job, previous, rows):
    seeds = {p['pid'] for p in rows.values() if matches(p, job.get('process_match', []))}
    seeds |= {p['pid'] for p in previous.get('processes', []) if same(p, rows)}
    root = job.get('output_root')
    if root:
        for p in rows.values():
            argv = p['argv']
            if '--out' in argv and argv[argv.index('--out') + 1].startswith(root.rstrip('/') + '/'):
                seeds.add(p['pid'])
    owned = set(seeds)
    while True:
        extra = {p['pid'] for p in rows.values() if p['ppid'] in owned}
        if extra <= owned: break
        owned |= extra
    return [identity(rows[p]) for p in sorted(owned)]


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
    gpus = {}
    for line in result.stdout.splitlines():
        index, uuid, used, total = [x.strip() for x in line.split(',')]
        gpus[int(index)] = dict(uuid=uuid, used_mb=int(used), total_mb=int(total), pids=[])
    if not gpus: raise RuntimeError('empty GPU inventory')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=8, check=True)
    for line in result.stdout.splitlines():
        uuid, pid = [x.strip() for x in line.split(',')]
        for gpu in gpus.values():
            if gpu['uuid'] == uuid: gpu['pids'].append(int(pid))
    return gpus


class Controller:
    def __init__(self, data, config, out=None):
        self.data, self.config = Path(data), config
        self.out = out or self.data / 'runs/sched/controller'
        self.out.mkdir(parents=True, exist_ok=True)
        p = self.out / 'state.json'
        self.state = json.loads(p.read_text()) if p.exists() else {'jobs': {}, 'alerts': {}}
        self.locks = []

    def path(self, value):
        return Path(value.replace('$DATA_DIR', str(self.data)).replace('$REPO', str(REPO)))

    def event(self, kind, **detail):
        row = dict(time=time.time(), utc=dt.datetime.now(dt.timezone.utc).isoformat(), kind=kind, **detail)
        for name in ('events.jsonl', 'log.txt'):
            with (self.out / name).open('a') as f:
                f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())

    def alert(self, key, detail):
        if self.state['alerts'].get(key) == detail: return
        self.state['alerts'][key] = detail
        self.event('ALERT', key=key, detail=detail)
        with (self.out / 'inbox.jsonl').open('a') as f:
            f.write(json.dumps(dict(id=f'{time.time_ns()}:{key}', time=time.time(), key=key, detail=detail)) + '\n')
            f.flush(); os.fsync(f.fileno())

    def save(self):
        atomic(self.out / 'state.json', self.state)

    def gate(self, rule):
        p = self.path(rule['path'])
        if not p.is_file() or p.stat().st_size == 0: return False
        if 'json' in rule:
            value = json.loads(p.read_text())
            return all(value.get(k) == v for k, v in rule['json'].items())
        return True

    def complete(self, job):
        def found(pattern):
            return any(Path(p).is_file() and Path(p).stat().st_size > 0 for p in glob.glob(str(self.path(pattern))))
        markers = job.get('done', [])
        if not markers or not all(self.path(p).is_file() for p in markers): return False
        if job.get('result'):
            result = json.loads(self.path(job['result']).read_text())
            if result.get('status') != 'DONE' or result.get('rc') != 0: return False
        elif not job.get('artifacts'): return False
        return all(found(p) for p in job.get('artifacts', [])) and all(
            any(found(p) for p in group) for group in job.get('artifact_any', []))

    def b_ready_queue(self):
        """Read the original queue and exact gate policy; never create a PASS."""
        if not self.config.get('b_handoff'): return []
        base = self.data / 'runs/nq3/b'
        queue = base / 'QUEUE'
        if not queue.exists(): return []
        approved = set((base / 'APPROVED').read_text().splitlines()) if (base / 'APPROVED').exists() else set()
        skipped = set((base / 'SKIP').read_text().splitlines()) if (base / 'SKIP').exists() else set()
        ready = []
        for line in queue.read_text().splitlines():
            fields = line.split()
            if len(fields) != 4: raise ValueError('malformed original B queue')
            arm, seed = fields[:2]
            if f'{arm} {seed}' in skipped or (base / f'arms/{arm}/s{seed}/DONE').exists(): continue
            verdict = self.data / f'runs/sched/pilot/b/arms/{arm}/s0/verdict.json'
            if arm in approved or (verdict.exists() and json.loads(verdict.read_text()).get('verdict') == 'PASS'):
                ready.append(f'B:{arm}:{seed}')
        return ready

    def observe(self, job, rows, gpus):
        name = job['id']; old = self.state['jobs'].get(name, {})
        expanded_job = dict(job)
        if job.get('output_root'): expanded_job['output_root'] = str(self.path(job['output_root']))
        procs = members(expanded_job, old, rows)
        pids = {p['pid'] for p in procs}
        held = [g for g, v in gpus.items() if pids.intersection(v['pids'])]
        error = next((str(self.path(p)) for p in job.get('errors', []) if self.path(p).is_file()), None)
        done = self.complete(job)
        if error and done:
            newest_done = max(self.path(p).stat().st_mtime for p in job['done'])
            if self.path(error).stat().st_mtime < newest_done:
                error = None  # Older ERROR is historical evidence, not a new failure.
        if error: done = False
        value = dict(old, processes=procs, actual_gpus=held, resources_released=not procs,
                     complete=done, error=error, observed_at=time.time())
        if error: status = 'QUARANTINED'
        elif done: status = 'COMPLETE_DRAINING' if procs else 'COMPLETE'
        elif procs: status = 'RUNNING'
        elif job.get('disabled'): status = 'BLOCKED_HUMAN'
        else: status = 'PENDING'
        value['status'] = status
        heartbeats = [self.path(p) for p in job.get('heartbeats', [])]
        stamps = [p.stat().st_mtime for p in heartbeats if p.is_file()]
        value['progress_age_s'] = round(time.time() - max(stamps), 1) if stamps else None
        if procs and stamps and value['progress_age_s'] > job.get('stall_s', 3600):
            self.alert(name + ':stale', 'Process alive but progress evidence stale; inspect logs; no restart or kill.')
        if error: self.alert(name + ':error', {'path': error, 'action': 'isolated; no unregistered repair'})
        if old.get('processes') and not procs:
            self.event('RESOURCE_RELEASE', job=name, previous_processes=old['processes'], complete=done)
            if not done and not error: self.alert(name + ':exit', 'Recorded identities exited without verified completion.')
        if old.get('status') != status: self.event('STATE', job=name, status=status, actual_gpus=held, complete=done)
        self.state['jobs'][name] = value
        return value

    def audit_b(self, rows, gpus):
        cfg = self.config.get('b_handoff')
        if not cfg: return
        base = self.data / 'runs/nq3/b'
        current = (base / 'CURRENT').read_text().strip() if (base / 'CURRENT').exists() else ''
        want = cfg['gpus']; workers = cfg['workers']
        runners = []
        fields = current.split()
        expected_out = str(base / 'arms' / fields[0] / ('s' + fields[1])) if len(fields) == 2 else None
        for p in rows.values():
            a = p['argv']
            if any(os.path.basename(x) == 'nq3_b_cl10.sh' for x in a):
                i = next(i for i,x in enumerate(a) if os.path.basename(x) == 'nq3_b_cl10.sh')
                if len(a) > i + 6 and a[i + 6].startswith(str(base / 'arms')):
                    runners.append(dict(identity(p), arm=a[i+1], gpu=int(a[i+2]), workers=int(a[i+3]), output=a[i+6]))
            elif any(os.path.basename(x) == 'b2d_run.py' for x in a) and '--out' in a and a[a.index('--out')+1].startswith(str(base / 'arms')):
                if '--gpu-rank' in a and '--workers' in a:
                    runners.append(dict(identity(p), arm=current.split()[0] if current else '', gpu=int(a[a.index('--gpu-rank')+1]),
                                        workers=int(a[a.index('--workers')+1]), output=a[a.index('--out')+1]))
        runners = [r for r in runners if r['output'] == expected_out and r['arm'] == fields[0]] if expected_out else []
        # Wrappers and nested runners can both appear; ack is per GPU.
        observed = sorted({r['gpu'] for r in runners if r['workers'] == workers})
        actual = sorted(g for g,v in gpus.items() if set(v['pids']) & {p['pid'] for p in self.state['jobs'].get('B', {}).get('processes', [])})
        previous = self.state.get('b_handoff', {})
        since = previous.get('since', time.time()) if previous.get('current') == current else time.time()
        ack = all(g in observed and g in actual for g in want)
        value = dict(current=current, target_gpus=want, workers=workers, runner_gpus=observed, gpu_process_ack=actual,
                     runners=runners, since=since, status='RUNNER_AND_GPU_ACK' if ack else 'WAIT_BOUNDARY_OR_LAUNCH')
        self.state['b_handoff'] = value
        initial = self.state.setdefault('b_initial_arm', current)
        boundary_seen = self.state.get('b_boundary_seen', False) or current != initial
        self.state['b_boundary_seen'] = boundary_seen
        if not boundary_seen and not ack: value['status'] = 'WAIT_EXISTING_ARM_BOUNDARY'
        if boundary_seen and not ack and time.time() - since > cfg.get('ack_timeout_s', 300):
            self.alert('B:grant_not_acknowledged:' + current, dict(current=current, target=want, actual_runners=observed,
                       action='Existing B owns output; inspect its logs/ERROR and boundary. No second B launched.'))
        if ack and previous.get('status') != 'RUNNER_AND_GPU_ACK': self.event('B_RUNNER_ACK', **value)

    def capacity(self, rows, gpus, target, workers):
        """Resident threads plus only the unrealized CARLA grant, not twice both."""
        current = int(Path('/sys/fs/cgroup/pids.current').read_text())
        maximum = int(Path('/sys/fs/cgroup/pids.max').read_text())
        owned = {p['pid'] for p in self.state['jobs'].get('B', {}).get('processes', [])}
        if any(g not in gpus for g in target): return False, 'target GPU missing'
        carla = sum(1 for g in target for pid in gpus[g]['pids']
                    if pid in owned and any('CarlaUE4' in a for a in rows.get(pid, {}).get('argv', [])))
        projected = current + max(0, len(target) * workers - carla) * 400
        if projected > min(16000, maximum - 1024): return False, f'projected threads {projected}'
        for g in target:
            if g not in gpus: return False, f'GPU {g} unavailable'
            if set(gpus[g]['pids']) - owned: return False, f'GPU {g} occupied by non-B process'
            resident = sum(1 for pid in gpus[g]['pids'] if pid in owned and
                           any('CarlaUE4' in a for a in rows.get(pid, {}).get('argv', [])))
            projected_mb = gpus[g]['used_mb'] + max(0, workers - resident) * 7680
            if projected_mb > min(88 * 1024, gpus[g]['total_mb'] - 8192): return False, f'GPU {g} projected memory {projected_mb}'
        raw = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        start = time.monotonic(); time.sleep(0.1)
        end = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
        used = (int(end['usage_usec']) - int(raw['usage_usec'])) / (time.monotonic() - start) / 1e6
        if used > 165: return False, f'CPU usage {used:.1f} above cap'
        return True, dict(resident_threads=current, projected_threads=projected, cpu_used=round(used,1))

    def b_grant(self, rows, gpus):
        cfg = self.config.get('b_handoff')
        if not cfg: return
        import sch_table as sch
        sch.TABLE = self.data / 'runs/sched/table.tsv'
        table = sch.load(); row = next((r for r in table if r['lane'] == 'nq3-b'), None)
        if row is None:
            self.alert('B:missing_registration', 'No existing nq3-b table row; cannot invent indices.'); return
        b = self.state['jobs']['B']
        if b.get('complete') or b.get('error') or b.get('status') == 'UNKNOWN': return
        go = self.data / 'runs/nq3/b/GO'
        text = go.read_text() if go.exists() else ''
        target = cfg['gpus']; workers = cfg['workers']
        target_text = 'GPUS="' + ' '.join(map(str,target)) + '"'
        parsed = {}
        for line in text.splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=',1)
                parts = shlex.split(value, comments=True)
                parsed[key] = ' '.join(parts)
        required = dict(GPUS=' '.join(map(str,target)), WORKERS=str(workers), B_CPUS=cfg['cpus'],
                        IDX_SPAN='30', B_IDX=cfg['idx0'].replace(',', ' '),
                        B_EXPAND_GPUS=' '.join(map(str,target)), B_EXPAND_WORKERS=str(workers), B_CPUS_WIDE=cfg['cpus'])
        complete_go = all(parsed.get(k) == v for k,v in required.items())
        agrees = complete_go and sch.gpus(row) == target and row['workers'] == str(workers) and row['idx0'].replace(' ', ',') == cfg['idx0'] and row['cpus'] == cfg['cpus'] and row['idx_span'] == '30'
        if agrees:
            if not self.state.get('b_grant_adopted'):
                self.event('B_GRANT_ADOPTED', gpus=target, workers=workers, note='grant is not runtime acknowledgement')
                self.state['b_grant_adopted'] = True
            return
        a = self.state['jobs'].get('A', {})
        if not a.get('resources_released') or a.get('status') == 'UNKNOWN': return
        # This exact index map and CPU pool were reviewed in the live inventory.
        if row['idx0'].replace(' ', ',') != cfg['idx0'] or row['cpus'] not in cfg.get('allowed_from_cpus', [cfg['cpus']]):
            self.alert('B:grant_registration_changed', 'CPU/index registration changed; require review.'); return
        needed_cpus = set()
        for part in cfg['cpus'].split(','):
            a, _, bcpu = part.partition('-'); needed_cpus.update(range(int(a), int(bcpu or a) + 1))
        if not needed_cpus <= os.sched_getaffinity(0):
            self.alert('B:cpu_affinity', 'Registered CPU mask unavailable'); return
        ok, detail = self.capacity(rows,gpus,target,workers)
        if not ok:
            self.alert('B:capacity', detail); return
        candidate = dict(row, gpus=','.join(map(str,target)), workers=str(workers), cpus=cfg['cpus'], status='batch controller B boundary grant')
        proposed = [candidate if r is row else r for r in table]
        conflicts = sch.conflicts(proposed)
        if conflicts:
            self.alert('B:ports', conflicts); return
        # No new port map is allocated here: only the reviewed B map is reused.
        # Retain every lane extra (B_IDX, expansion and policy settings).
        staging = go.with_suffix('.controller-staging')
        staging.write_text(text)
        sch.write_go(staging, candidate, 'B')
        lines = staging.read_text().splitlines()
        for key, value in required.items():
            replacement = key + '=' + shlex.quote(value)
            found = False
            for index, line in enumerate(lines):
                if line.startswith(key + '='):
                    lines[index] = replacement; found = True
            if not found: lines.append(replacement)
        staging.write_text('\n'.join(lines) + '\n')
        sch.save(proposed)
        staging.replace(go)  # Publish every required setting together at one arm boundary.
        self.state['b_grant_adopted'] = True
        self.event('B_GRANT', gpus=target, workers=workers, capacity=detail)

    def b_recover(self, rows, now):
        """Only the handoff's one registered owner-loss retry, never CL10 ERROR."""
        cfg = self.config.get('b_handoff')
        if not cfg: return
        b = self.state['jobs']['B']
        if b.get('processes'):
            b['ever_adopted'] = True
            b.pop('owner_absent_since', None)
            return
        if b.get('complete') or b.get('error') or not b.get('ever_adopted'): return
        if b.get('status') == 'UNKNOWN': return
        if now >= dt.datetime.fromisoformat(self.config['deadline']).timestamp(): return
        if b.get('owner_loss_retries', 0) >= 1:
            self.alert('B:owner_loss_retry_exhausted', 'Registered single owner-loss retry consumed; inspect before restart.')
            return
        # Wait two complete observations so an exec transition cannot duplicate B.
        first = b.setdefault('owner_absent_since', now)
        if now - first < 60: return
        if any('nq3_b.sh' in p['argv'] or 'scripts/nq3_b.sh' in p['argv'] for p in rows.values()): return
        base = self.data / 'runs/nq3/b'
        if not (base / 'GO').is_file() or not (base / 'cl0/DONE').is_file():
            self.alert('B:retry_missing_gate', 'GO/CL0 absent; no owner-loss retry.'); return
        if not self.state.get('b_grant_adopted'): return
        gpu_live = gpu_snapshot()
        ok, detail = self.capacity(rows, gpu_live, cfg['gpus'], cfg['workers'])
        if not ok:
            self.alert('B:retry_capacity', detail); return
        b['owner_loss_retries'] = 1
        self.save()  # Consume budget before launch; ambiguous launch never repeats.
        env = dict(os.environ, DATA_DIR=str(self.data), B_PILOT_DIR=str(self.data / 'runs/sched/pilot/b'))
        r = subprocess.run(['bash', 'scripts/tmux_run.sh', 'cx-b-owner-loss-retry', 'bash', 'scripts/nq3_b.sh', 'chain'],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=10)
        self.event('B_OWNER_LOSS_RETRY', rc=r.returncode, output=r.stdout, error=r.stderr)
        if r.returncode: self.alert('B:retry_launch_failed', r.stderr or r.stdout)

    def reclaim(self, gpus=None):
        import sch_table as sch
        sch.TABLE = self.data / 'runs/sched/table.tsv'
        rows = sch.load(); changed = False
        for job in self.config['jobs']:
            lane = job.get('schedule_lane'); value = self.state['jobs'].get(job['id'], {})
            if not lane or value.get('status') == 'UNKNOWN' or not value.get('resources_released'): continue
            # A previously observed owner or reviewed retired lane is needed.
            if not (job.get('retired') or value.get('complete') or value.get('error')): continue
            for row in rows:
                if row['lane'] == lane and gpus is not None and any(gpus.get(g, {}).get('pids') for g in sch.gpus(row)):
                    self.alert(lane + ':resident_without_owner', 'GPU residents remain without an attributable lane owner; no claim release.'); continue
                if row['lane'] == lane and not row['status'].startswith(('done', 'revoked')):
                    row['status'] = 'revoked controller: verified no resident owner'
                    changed = True; self.event('GRANT_RELEASE', lane=lane)
        if changed: sch.save(rows)

    def tick(self, rows=None, gpus=None, now=None):
        now = time.time() if now is None else now
        try:
            rows = process_snapshot() if rows is None else rows
            gpus = gpu_snapshot() if gpus is None else gpus
        except Exception as e:
            for value in self.state['jobs'].values(): value.pop('owner_absent_since', None)
            self.state['node'] = 'UNKNOWN'; self.alert('node:unknown', str(e)); self.save(); return
        self.state.update(node='ONLINE', heartbeat=now, identity=identity(rows[os.getpid()]) if os.getpid() in rows else None)
        for job in self.config['jobs']:
            try:
                value = self.observe(job, rows, gpus)
            except Exception as e:
                self.state['jobs'].setdefault(job['id'], {}).update(status='UNKNOWN', complete=False, resources_released=False)
                self.state['jobs'][job['id']].pop('owner_absent_since', None)
                self.alert(job['id'] + ':observation', str(e))
        self.audit_b(rows, gpus)
        self.reclaim(gpus)
        if now < dt.datetime.fromisoformat(self.config['deadline']).timestamp(): self.b_grant(rows, gpus)
        self.b_recover(rows, now)
        free = {g for g,v in gpus.items() if not v['pids'] and v['used_mb'] < 1024}
        deadline = dt.datetime.fromisoformat(self.config['deadline']).timestamp()
        pending_ready = self.b_ready_queue()
        self.state['queue'] = dict(status='DEADLINE_NO_NEW_CONTROLLER_LAUNCH' if now >= deadline else
                                  'READY_DELEGATED_TO_B' if pending_ready else 'NO_READY_WORK', ready=pending_ready,
                                  free_gpus=sorted(free), note='Existing B chain owns its fixed queue and arm boundaries')
        if now >= deadline: self.alert('deadline', 'Admission deadline reached; preserve active work and partial outputs; no killing.')
        self.save()
        lines = ['# Unified controller', '', 'Durable alerts: inbox.jsonl (no automatic LLM wake-up).', '', f'Queue: {self.state["queue"]["status"]}', '', '| Job | State | GPUs | Resources exited |', '|---|---|---|---|']
        lines += [f"| {name} | {v.get('status')} | {v.get('actual_gpus', [])} | {v.get('resources_released')} |" for name,v in self.state['jobs'].items()]
        (self.out / 'STATUS.md').write_text('\n'.join(lines) + '\n')
        atomic(self.out / 'heartbeat.json', dict(time=now, node='ONLINE', pid=os.getpid()))

    def acquire(self):
        self.locks.append(claim(self.out / 'controller.lock'))
        self.locks.append(claim(self.data / 'runs/sched/owner.lock'))
        self.locks.append(claim(self.data / 'runs/nq4/cx/orchestration/dispatcher.lock'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['run', 'once', 'inbox'])
    ap.add_argument('--data', type=Path, default=Path(os.environ.get('DATA_DIR', '/root/autodl-tmp/ujs')))
    ap.add_argument('--registry', type=Path, default=DEFAULT)
    a = ap.parse_args()
    if a.mode == 'inbox':
        p = a.data / 'runs/sched/controller/inbox.jsonl'
        print(p.read_text() if p.exists() else 'No durable alerts.'); return 0
    cfg = json.loads(a.registry.read_text())
    interval = cfg.get('poll_s', 30)
    if not 5 <= interval <= 60: raise ValueError('poll_s must be 5..60')
    c = Controller(a.data, cfg); c.acquire()
    c.event('OWNER_START', registry=str(a.registry), sha256=hashlib.sha256(a.registry.read_bytes()).hexdigest())
    while True:
        started = time.monotonic()
        try: c.tick()
        except Exception as e:
            c.alert('controller:tick', f'{type(e).__name__}: {e}'); c.save()
        if a.mode == 'once': return 0
        time.sleep(max(0.1, interval - (time.monotonic() - started)))


if __name__ == '__main__': sys.exit(main())
