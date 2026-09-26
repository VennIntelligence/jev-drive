#!/usr/bin/env python3
"""Isolated X repair validation: capacity wait, one route, rule 8, ten-route pilot.

No full batch launch and no shared model socket. Only recorded child process groups
are stopped. Run through scripts/tmux_run.sh, using the repository Python env.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sch_table as SCH
from jevdrive import nq4_g as G, nq4_x as X

REPO = Path(__file__).resolve().parents[1]
DATA = SCH.DATA
LANE = 'cx-x-fix'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Run:
    def __init__(self, number):
        self.out = DATA / 'runs/nq4/cx/x-fix' / f'round{number:02}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out.parent / 'lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.out / 'started').exists():
            raise RuntimeError('Round already started; inspect its PID/result and use a fresh round')
        (self.out / 'started').write_text(str(time.time()))
        self.children = []
        self.env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                        NUMBA_NUM_THREADS='2', B2D_PIDS_WAIT='16000', B2D_NQ4_TRACE='1', B2D_SENSOR_TICK='1',
                        PYTHONUNBUFFERED='1', HF_ENDPOINT='https://hf-mirror.com')
        self.cpus = '204-207'
        self.index = None
        (self.out / 'pid').write_text(str(os.getpid()))
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(str(self.out / 'tb'))

    def event(self, kind, **data):
        rec = dict(t=time.time(), kind=kind, **data)
        text = json.dumps(rec)
        with (self.out / 'events.jsonl').open('a') as f:
            f.write(text + '\n')
        with (self.out / 'log.txt').open('a') as f:
            f.write(text + '\n')
        print(text, flush=True)

    def start(self, command, name, env=None):
        self.event('command', name=name, argv=list(map(str, command)))
        log = (self.out / f'{name}.log').open('a')
        p = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env or self.env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.children.append(p)
        (self.out / f'{name}.pid').write_text(str(p.pid))
        return p

    def wait(self, p, timeout):
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.event('infrastructure_timeout', pid=p.pid, seconds=timeout)
            raise RuntimeError('Infrastructure timeout; this is not a model verdict')
        if code:
            raise RuntimeError(f'Process {p.pid} exited {code}')

    def capacity(self):
        """Reserve an unused block only when one worker fits debug and global limits."""
        while True:
            probe = SCH.probe()
            gpu = next(g for g in probe['gpus'] if g['gpu'] == 1)
            if (probe['pids'] + 600 > SCH.PIDS_CAP or probe['cores_used'] > SCH.CPU_CAP - 4 or
                    gpu['carla'] >= 4 or gpu['used_gb'] > 72):
                self.event('waiting_capacity', probe=probe)
                time.sleep(60)
                continue
            # Coordinate scheduler writers; root is told this lane owns this row.
            with (SCH.TABLE.parent / 'table.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                rows = [r for r in SCH.load() if r['lane'] != LANE]
                used_ports = set()
                ss = subprocess.check_output(['ss', '-H', '-ltn'], text=True)
                for line in ss.splitlines():
                    try:
                        used_ports.add(int(line.split()[3].rsplit(':', 1)[1]))
                    except (IndexError, ValueError):
                        pass
                for index in range(0, 493):
                    row = dict(lane=LANE, gpus='1', workers='1', idx0=str(index), idx_span='3',
                               cpus=self.cpus, status='debug X isolated validation', go='-')
                    ports = {p for i in range(index, index + 3) for p in
                             [2000 + 50*i, 2001 + 50*i, 2002 + 50*i, *range(8000 + 50*i, 8050 + 50*i)]}
                    if not SCH.conflicts(rows + [row]) and not used_ports & ports:
                        SCH.save(rows + [row])
                        self.index = index
                        break
            if self.index is None:
                self.event('waiting_ports')
                time.sleep(60)
                continue
            checked = subprocess.run([sys.executable, 'scripts/sch_table.py', 'check'], cwd=REPO,
                                     capture_output=True, text=True)
            self.event('reservation', index=self.index, span=3, gpu=1, cpus=self.cpus,
                       check_returncode=checked.returncode, check_output=checked.stdout, probe=probe)
            if checked.returncode:
                self.release()
                time.sleep(60)
                continue
            return

    def release(self):
        if self.index is None:
            return
        with (SCH.TABLE.parent / 'table.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            rows = SCH.load()
            for row in rows:
                if row['lane'] == LANE:
                    row['status'] = 'done isolated X validation'
            SCH.save(rows)
        self.index = None

    def cleanup(self):
        for p in reversed(self.children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        # b2d_run owns detached CARLA/route groups; verify their recorded identity.
        for pattern in ('*/servers/carla-*.pid', '*/attempts/*/*/route.pid'):
            for file in self.out.glob(pattern):
                pid = int(file.read_text().strip())
                proc = Path('/proc') / str(pid)
                if not proc.exists():
                    continue
                cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if str(self.out) not in cmd and not ('CarlaUE4' in cmd and
                       any(f'-carla-rpc-port={2000 + 50*i}' in cmd for i in range(self.index or 0, (self.index or 0)+3))):
                    self.event('cleanup_identity_refused', pid=pid, command=cmd)
                    continue
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
                    self.event('cleanup_group', pid=pid)
        self.release()
        self.tb.close()

    def prepare(self):
        root = G.root()
        q2 = root / 'heads_xfit/q2'
        if not (q2 / 'READY').exists():
            if not (DATA / 'runs/nq3/q2/closed_loop_head/READY').exists():
                raise RuntimeError('Formal Q2 source not READY')
            claim = root / 'heads_xfit/q2.lock'
            claim.mkdir()  # Do not interfere with a concurrent exporter.
            p = self.start(['taskset', '-c', self.cpus, sys.executable, '-m', 'jevdrive.nq4_x', 'export-q2'],
                           'export-q2', dict(self.env, CUDA_VISIBLE_DEVICES='1'))
            self.wait(p, 3600)
            if not (q2 / 'READY').exists():
                raise RuntimeError('Formal Q2 export did not write READY')
        cfg = dict(model='head', warmup_s=5., desire=True, head_cam_tick=0., arm='q2', x=True,
                   socket=str(self.out / 'head.sock'), controller='fixed', controller_preset='pursuit',
                   controller_config=str(REPO / 'todos/2026-09-23-tfv6-controller/controller-eval/P7.json'),
                   seed=0, dump_every=1, cruise_mps=8.,
                   folds=dict(split=str(DATA / 'runs/nq4/k/route_split.json'), key='q2_dir',
                              R1=str(q2 / 'R1'), R2=str(q2 / 'R2'), pick='unseen', rule='label'))
        (self.out / 'config.json').write_text(json.dumps(cfg, indent=2))
        inputs = [Path(cfg['controller_config']), Path(cfg['folds']['split']), root / 'g_routes.xml',
                  *q2.glob('R*/*.json'), *q2.glob('R*/*.npz'),
                  REPO / 'scripts/nq4_x_agent.py', REPO / 'scripts/nq4_x_controller.py', REPO / 'jevdrive/nq4_x.py']
        (self.out / 'inputs.json').write_text(json.dumps({str(p): sha(p) for p in inputs}, indent=2))
        p = self.start(['taskset', '-c', self.cpus, DATA / 'envs/openpilot/bin/python', 'scripts/nq3_cl_server.py',
                        '--pool', '1', '--socket', self.out / 'head.sock', '--ready-file', self.out / 'head.ready'],
                       'head', dict(self.env, CUDA_VISIBLE_DEVICES='1'))
        deadline = time.monotonic() + 900
        while not (self.out / 'head.ready').exists():
            if p.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Isolated head server not ready')
            time.sleep(5)

    def routes(self, stage, ids):
        out = self.out / stage
        out.mkdir(exist_ok=True)
        command = ['taskset', '-c', self.cpus, DATA / 'envs/carla/bin/python', 'scripts/b2d_run.py',
                   '--routes', G.root() / 'g_routes.xml', '--route-ids', ','.join(ids), '--out', out,
                   '--workers', '1', '--server-index', str(self.index), '--index-span', '3', '--gpu-rank', '1',
                   '--tm-seed', '0', '--no-spectator', '--no-reap', '--client-threads', '8', '--max-attempts', '3',
                   '--stall-s', '480', '--route-timeout-s', '3600', '--python', DATA / 'envs/scout-tfv6/bin/python',
                   '--agent', 'scripts/nq4_x_agent.py', '--agent-config', self.out / 'config.json',
                   '--fast-copy', '--cache-lights']
        # pilot_check reads this exact conventional log name.
        log = (out / 'runner-g1.log').open('a')
        self.event('route_command', stage=stage, argv=list(map(str, command)))
        p = subprocess.Popen(list(map(str, command)), cwd=REPO, env=self.env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        log.close(); self.children.append(p)
        (out / 'runner.pid').write_text(str(p.pid))
        # 10 min/route registered prior; allow 2x plus startup. Timeout is infrastructure only.
        started = time.monotonic()
        self.wait(p, 1800 + 1200 * len(ids))
        self.event('route_timing', stage=stage, routes=len(ids), wall_s=time.monotonic() - started)
        verdict = G.pilot_check('x', 'orig', out, ids)
        (out / 'pilot.json').write_text(json.dumps(verdict, indent=2))
        self.event('pilot_check', stage=stage, **verdict)
        self.tb.add_scalar(f'{stage}/moving', verdict['facts'].get('moving', 0), 0)
        return verdict

    def equivalence(self, stages):
        split = json.loads((DATA / 'runs/nq4/k/route_split.json').read_text())['routes']
        results = {}
        low = high = 0
        for stage, ids in stages:
            for rid in ids:
                adir = G.finished_attempt(self.out / stage, rid)
                if adir is None:
                    raise RuntimeError(f'Missing finished attempt {stage}/{rid}')
                path = X.check(adir)
                fold = json.loads((adir / 'fold.json').read_text())
                correct = fold['fold'] == X.fold_pick(split, X.base_of(rid), 'unseen')
                for line in (adir / 'x_dump.jsonl').read_text().splitlines():
                    rec = json.loads(line)
                    if rec['mode'] in (1, 4):
                        low += rec['speed_mps'] < 1.
                        high += rec['speed_mps'] >= 1.
                p = subprocess.run([str(DATA / 'envs/openpilot/bin/python'), '-m', 'jevdrive.nq4_x',
                                    'check-head', str(adir)], cwd=REPO, env=self.env, capture_output=True, text=True)
                (adir / 'head-check.txt').write_text(p.stdout + p.stderr)
                results[str(adir)] = dict(path=path, fold_correct=correct, head_returncode=p.returncode)
        good = bool(results) and all(v['path']['differing'] == 0 and v['path']['plans'] > 0 and
                                    v['fold_correct'] and v['head_returncode'] == 0 for v in results.values())
        result = dict(pass_=good, low_speed_stop_plans=low, high_speed_stop_plans=high, attempts=results)
        (self.out / 'rule8.json').write_text(json.dumps(result, indent=2))
        self.event('rule8', **result)
        return good

    def run(self):
        progress = tqdm(total=3, desc='X validation stages')
        self.capacity()
        self.prepare()
        # Prespecified rule-8 routes; no selection based on outcomes.
        one = [str(2534 * 100 + 90)]
        extra = [str(b * 100 + 90) for b in (2668, 1790)]
        first = self.routes('stage1', one)
        first_equivalent = self.equivalence([('stage1', one)])
        progress.update(1)
        if not first['pass'] or not first_equivalent:
            return dict(status='stage1_failed', model_verdict=True, pilot=first, rule8_pass=first_equivalent)
        self.routes('rule8', extra)
        if not self.equivalence([('stage1', one), ('rule8', extra)]):
            return dict(status='rule8_failed', model_verdict=False)
        progress.update(1)
        # Same deterministic first ten obstacle routes as registered G route order.
        table = G.route_table()
        ids = [str(int(b) * 100 + 90) for b in table.loc[table.obstacle, 'base'].astype(str).tolist()[:10]]
        if len(ids) != 10:
            raise RuntimeError('Expected ten registered obstacle pilot routes')
        pilot = self.routes('stage2', ids)
        good = self.equivalence([('stage1', one), ('rule8', extra), ('stage2', ids)])
        progress.update(1)
        progress.close()
        return dict(status='validated' if pilot['pass'] and good else 'stage2_failed',
                    pilot=pilot, rule8_pass=good, full_batch_started=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True, choices=range(1, 11))
    a = ap.parse_args()
    run = Run(a.round)
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        result = run.run()
    except BaseException as e:
        result = dict(status='infrastructure_error', error=repr(e), model_verdict=False)
        import traceback
        run.event('exception', traceback=traceback.format_exc())
    finally:
        run.cleanup()
    (run.out / 'result.json').write_text(json.dumps(result, indent=2))
    run.event('round_complete', **result)
    return 0 if result['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
