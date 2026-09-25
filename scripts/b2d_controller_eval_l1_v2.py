"""Task 10 v2 L1 driver: one validator (one fresh CARLA server) per route, N routes in parallel.

kind=probe    arm A, nominal spawn, 40 ticks: dumps each route's dense path and spawn pose
kind=ramp|profile|step
              arms x perturbations; ramp/profile feed the fixed reference trace, step feeds the
              route adapter's cruise-from-standstill trajectory (the v1 "oracle" interface)
Infrastructure failures (zero controlled ticks, RPC timeout or setup error) rerun only the
missing cases, at most three attempts per route; anything else stops the whole batch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'todos/2026-09-23-tfv6-controller/controller-eval'
LOCK = threading.Lock()
INTERFACES = ('short_2s', 'sparse_5s', 'stop_jitter', 'model_noise', 'stale_5hz', 'stale_2hz', 'stale_1hz', 'pose_plan')
# Plan refresh period in 20 Hz ticks; nominal plans refresh every tick.
DECIMATE = {'stale_5hz': 4, 'stale_2hz': 10, 'stale_1hz': 20}
# Town12/13 servers take 6-8 GB each; three at once fill a 24 GB card.
BIG = ('Town12', 'Town13')
BIG_SLOTS = threading.Semaphore(2)


def log(out, event, **data):
    row = dict(t=datetime.now(timezone.utc).isoformat(), kind=event, **data)
    with LOCK, (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(row['t'][11:19], event, data, flush=True)


def status(case):
    path = case / 'validation.json'
    if not path.exists():
        return 'missing'
    d = json.loads(path.read_text())
    if not d.get('exception') and not d.get('cleanup_errors') and not d.get('telemetry_parse_errors') \
            and d['gates']['telemetry_complete']:
        return 'valid'
    if d.get('ticks') == 0 and ('time-out' in str(d.get('exception')) or d.get('status') == 'setup_error'):
        return 'infrastructure'
    return 'bug'


def run_route(a, route, index):
    rid = route.get('id')
    home = a.out / f'route-{rid}'
    if (home / 'done.json').exists():
        return rid, 'skip'
    home.mkdir(parents=True, exist_ok=True)
    xml = home / 'route.xml'
    root = ET.Element('routes'); root.append(route)
    ET.ElementTree(root).write(xml, encoding='utf-8', xml_declaration=True)
    seeds = ['p00'] if a.kind == 'probe' else a.seeds.split(',')
    variants = json.loads(a.variants.read_text())
    arms = (['A'] if a.kind == 'probe' else a.arms.split(',') if ',' in a.arms or a.arms in variants
            else list(a.arms))
    PRESET = {arm: variants[arm]['presets'][0] for arm in arms}
    wanted = [(s, arm) for s in seeds for arm in arms]
    max_ticks = 40
    if a.kind in ('ramp', 'profile', 'crawl'):
        refs = json.loads((a.refs / f'{a.kind}-traces.json').read_text())
        duration = json.loads(Path(refs[rid]).read_text())['bounds']['duration_s']
        max_ticks = math.ceil(duration / .05) + 400
        (home / 'traces.json').write_text(json.dumps({rid: refs[rid]}))
    elif a.kind == 'step':
        max_ticks = 1800
    # Town13 servers hang after two or three consecutive cases in one process, so they get
    # two cases per process. Retries stop after three attempts in a row with no new valid case.
    chunk = 2 if route.get('town') == 'Town13' else len(wanted)
    stalls, before = 0, -1
    while True:
        attempt = 1 + len(list(home.glob('attempt-*/')))
        found = {}
        for s, arm in wanted:
            for prior in sorted(home.glob('attempt-*')):
                state = status(prior / s / rid / arm / PRESET[arm])
                if state == 'bug':
                    raise RuntimeError(f'{rid} {s}/{arm}: non-infrastructure failure in {prior}')
                if state == 'valid':
                    found[(s, arm)] = str(prior / s / rid / arm / PRESET[arm])
        stalls = stalls + 1 if len(found) == before else 0
        before = len(found)
        missing = [c for c in wanted if c not in found]
        if not missing:
            (home / 'done.json').write_text(json.dumps({f'{s}/{arm}': p for (s, arm), p in found.items()}, indent=1))
            return rid, f'valid after {attempt - 1} attempt(s)'
        if stalls >= 3:
            break
        missing = missing[:chunk]
        target = home / f'attempt-{attempt}'
        cases = home / f'cases-{attempt}.json'
        cases.write_text(json.dumps([dict(route=rid, perturbation_id=s, variant=arm, preset=PRESET[arm])
                                     for s, arm in missing]))
        command = ['/data/envs/tfv6/bin/python', str(ROOT / 'scripts/b2d_controller_validate.py'),
                   '--routes', str(xml), '--out', str(target), '--variants', str(a.variants),
                   '--route-cruises', str(INPUT / 'l1-cruises.json'),
                   '--perturbations', str(INPUT / 'perturbations.json'),
                   '--perturbation-ids', ','.join(sorted({s for s, _ in missing})), '--case-list', str(cases),
                   '--server-index', str(index), '--max-ticks', str(max_ticks),
                   '--rig', 'none', '--no-rendering', '--strict-invariants',
                   '--reference-interface', a.interface if a.interface in ('short_2s', 'sparse_5s', 'stop_jitter', 'model_noise') else 'nominal',
                   # Plans refresh every tick (20 Hz, as TFv6/TCP run); stale_* hold each plan longer.
                   '--decimate', str(DECIMATE.get(a.interface, 1))]
        if a.kind in ('ramp', 'profile', 'crawl'):
            command += ['--reference-traces', str(home / 'traces.json')]
            # pose_plan: the plan goes through the noisy estimated pose instead of the true ego frame.
            if a.interface != 'pose_plan':
                command += ['--plan-from-truth']
        env = dict(os.environ, DATA_DIR='/data', OPENBLAS_CORETYPE='Barcelona',
                   BENCH2DRIVE_ROOT='/data/runs/b2d/tfv6-repro/runtime/Bench2Drive')
        log(a.out, 'attempt_start', route=rid, attempt=attempt, cases=len(missing), server=index)
        with (home / f'attempt-{attempt}.log').open('w') as f:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        states = {f'{s}/{arm}': status(target / s / rid / arm / PRESET[arm]) for s, arm in missing}
        log(a.out, 'attempt_end', route=rid, attempt=attempt, rc=rc,
            counts={k: list(states.values()).count(k) for k in set(states.values())})
    raise RuntimeError(f'{rid}: three consecutive attempts without a new valid case')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('probe', 'ramp', 'profile', 'crawl', 'step'), required=True)
    p.add_argument('--routes', type=Path, default=INPUT / 'l1-v2-heldout.xml')
    p.add_argument('--refs', type=Path, help='directory with <kind>-traces.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', default='p01,p02,p03')
    p.add_argument('--arms', default='ABCD', help='letters, or comma-separated variant labels')
    p.add_argument('--variants', type=Path, default=INPUT / 'l1-variants.json')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--server-base', type=int, default=120)
    p.add_argument('--interface', default='nominal', choices=('nominal',) + INTERFACES)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    routes = ET.parse(a.routes).getroot().findall('route')
    # Alternate big and small maps so at most two big-map servers usually run at once.
    big = [r for r in routes if r.get('town') in BIG]
    small = [r for r in routes if r.get('town') not in BIG]
    routes = [r for pair in zip(big, small) for r in pair] + big[len(small):] + small[len(big):]
    log(a.out, 'start', reference=a.kind, routes=len(routes), seeds=a.seeds, arms=a.arms, workers=a.workers)
    slots = list(range(a.workers))
    with ThreadPoolExecutor(a.workers) as pool:
        def job(route):
            with LOCK:
                slot = slots.pop()
            try:
                if route.get('town') in BIG:
                    with BIG_SLOTS:
                        return run_route(a, route, a.server_base + 2 * slot)
                return run_route(a, route, a.server_base + 2 * slot)
            finally:
                with LOCK:
                    slots.append(slot)

        futures = [pool.submit(job, r) for r in routes]
        done = 0
        for future in as_completed(futures):
            try:
                rid, note = future.result()
            except BaseException as error:
                pool.shutdown(wait=False, cancel_futures=True)
                log(a.out, 'stopped', error=repr(error))
                raise
            done += 1
            log(a.out, 'route_done', route=rid, note=note, done=done, total=len(routes))
    log(a.out, 'end', done=done)


if __name__ == '__main__':
    main()
