#!/usr/bin/env python
"""Monitor route loop: one b2d_run Runner per route on one CARLA server that is kept across routes.

Order: the list is grouped by town (first-appearance order), so routes on one map run back to back; the route
processes get pyhook/ on PYTHONPATH, whose sitecustomize keeps the loaded world when the next route is on the same
map (VIZ_REUSE_WORLD=0: reload every route, as the leaderboard ships).
Requests (viewer picker): <live>/request.json {queue: [{route, mode: next|now, ts}], ...}, written only by the viewer;
this loop records what it took in <live>/consumed.json {ts: [...]} and never writes request.json. Pending entries run
before the list: the newest "now" first (it aborts the running route the way b2d_run does on Ctrl-C, SIGINT to the
route's group, and keeps CARLA), else the oldest entry on the current town, else the oldest entry.
<live>/loop.json {route, town, why, next, ids, queue, state, t} tells the viewer what is loading.
Monitor dir (live/, runs/, carla-logs/): env B2D_MON_DIR, default $DATA_DIR/runs/b2d_monitor.
usage (via loop.sh): viz_loop.py <comma ids> <agent config> [--agent experiments/b2d_monitor/lib/op_arb_viz_agent.py]
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
import b2d_run as R  # noqa: E402

V = Path(os.environ.get('B2D_MON_DIR') or R.DATA_DIR / 'runs/b2d_monitor')
XML = str(R.BENCH2DRIVE / 'leaderboard/data/bench2drive_0.0.4_val.xml')


class KeptServer(R.Server):
    """While `keep` is set (a user abort), the Runner's stop()/move() leave the server running."""
    keep = False

    def stop(self):
        if not self.keep:
            super().stop()

    def move(self):
        if not self.keep:
            super().move()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = path.with_name('.' + path.name)
    tmp.write_text(json.dumps(obj))
    os.replace(str(tmp), str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ids')
    p.add_argument('cfg')
    p.add_argument('--agent', default=str(REPO / 'experiments/b2d_monitor/lib/op_arb_viz_agent.py'))
    p.add_argument('--server-args', default='-WinX=40 -WinY=1300')
    p.add_argument('--server-index', type=int, default=0)
    p.add_argument('--gpu', type=int, default=int(os.environ.get('MON_GPU', '0')), help='card of CARLA (Tokyo: 0; card 1 is not ours)')
    p.add_argument('--live', default=str(V / 'live'))
    p.add_argument('--offscreen', action='store_true')
    p.add_argument('--max-ticks', type=int, default=0, help='per route (tests)')
    p.add_argument('--max-routes', type=int, default=0, help='stop after this many routes (tests); 0 = forever')
    p.add_argument('--prefix', default='loop')
    o = p.parse_args()

    def interrupt(*_):          # tmux kill-window (SIGHUP) / kill: stop like Ctrl-C, so the route and CARLA go too
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    live = Path(o.live)
    live.mkdir(parents=True, exist_ok=True)
    req_f, done_f, state_f = live / 'request.json', live / 'consumed.json', live / 'loop.json'
    os.environ['B2D_PREVIEW_DIR'] = str(live)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parent / 'pyhook')] + [x for x in [os.environ.get('PYTHONPATH')] if x])
    town = dict(R.select_routes(XML, 'all', '', 0))
    raw = [i for i in o.ids.split(',') if i in town]
    first = {}
    for rid in raw:
        first.setdefault(town[rid], len(first))
    ids = sorted(raw, key=lambda r: first[town[r]])     # stable: list order inside a town
    consumed = list(read_json(done_f, {}).get('ts', []))

    def pending():
        r = read_json(req_f)
        if not r:
            return []
        q = r.get('queue') if 'queue' in r else ([dict(route=r['route'], mode=r.get('mode', 'next'), ts=r.get('ts', 0))]
                                                  if r.get('route') else [])
        return [e for e in q if str(e.get('route')) in town and e.get('ts') not in consumed]

    def take(cur_town):
        q = pending()
        if not q:
            return None
        now = [e for e in q if e.get('mode') == 'now']
        same = [e for e in q if town[str(e['route'])] == cur_town]
        e = now[-1] if now else same[0] if same else q[0]
        consumed.append(e.get('ts'))
        del consumed[:-200]
        write_json(done_f, {'ts': consumed})
        return e

    args_common = ['--routes', XML, '--workers', '1', '--server-index', str(o.server_index), '--gpu-rank', str(o.gpu),
                   '--tm-seed', '0', '--no-reap', '--client-threads', '8', '--max-attempts', '2', '--stall-s', '480',
                   '--route-timeout-s', '2400', '--python', R.PYTHON, '--fast-copy',
                   '--cache-lights', '--fresh', '--server-args=' + o.server_args]
    args_common += ['--agent', o.agent, '--agent-config', o.cfg] if o.agent else []   # --agent '': b2d_run's stub (tests)
    args_common += ([] if o.offscreen else ['--windowed']) + (['--max-ticks', str(o.max_ticks)] if o.max_ticks else [])

    def args(out, rid):
        a = R.parse_args(args_common + ['--route-ids', rid, '--out', str(out)])
        if a.agent:
            a.agent, a.agent_config = str(Path(a.agent).resolve()), str(Path(a.agent_config).resolve())
        return a

    server = KeptServer(o.server_index, V / 'carla-logs', 'Epic', o.gpu, stride=1, windowed=not o.offscreen,
                        extra_args=o.server_args.split(), base=o.server_index, span=1)
    cur = {'runner': None, 'route': None, 't': 0.0, 'aborting': False}

    def watch():                # a "now" request newer than the running route aborts it; the main loop then takes it
        while True:
            time.sleep(0.5)
            run = cur['runner']
            if run is None or run.stop_flag:
                continue
            if any(e.get('mode') == 'now' and e.get('ts', 0) > cur['t'] for e in pending()):
                cur['aborting'] = server.keep = True
                run.stop_flag = True
    threading.Thread(target=watch, daemon=True).start()

    def log(out, msg):
        line = '%s %s' % (time.strftime('%H:%M:%S'), msg)
        print(line, flush=True)
        with open(str(out / 'runner.log'), 'a') as fh:
            fh.write(line + '\n')

    n, i, out, ran, last_town = 0, 0, None, 0, None
    try:
        while not o.max_routes or ran < o.max_routes:
            if out is None or i >= len(ids):    # a pass = the list once, one fresh output dir
                n, i = n + 1, 0
                out = V / 'runs' / ('%s-%s-p%d' % (o.prefix, time.strftime('%m%d-%H%M'), n))
                out.mkdir(parents=True, exist_ok=True)
                log(out, 'pass %d, list by town: %s' % (n, ' '.join('%s:%s' % (r, town[r]) for r in ids)))
            e = take(last_town)
            if e:
                rid, why = str(e['route']), 'request ' + e.get('mode', 'next')
            else:
                rid, why = ids[i], 'list %d/%d' % (i + 1, len(ids))
                i += 1
            write_json(state_f, dict(route=rid, town=town[rid], why=why, out=str(out), next=ids[i % len(ids)], ids=ids,
                                     queue=[x['route'] for x in pending()], state='loading', t=time.time()))
            log(out, 'route %s (%s, %s%s) -> %s' % (rid, town[rid], why, ', same map' if town[rid] == last_town else '',
                                                    out.name))
            run = R.Runner(args(out, rid), [(rid, town[rid])], servers=[server])
            cur.update(runner=run, route=rid, t=time.time())
            rc = run.run()
            cur['runner'], ran, last_town = None, ran + 1, town[rid]
            done = out / 'done' / (rid + '.json')
            log(out, 'route %s: %s' % (rid, 'aborted for a request' if cur['aborting'] else
                                       read_json(done, {}).get('status') if done.exists() else 'not finished (rc %s)' % rc))
            if rc == 130 and not cur['aborting']:
                break           # Ctrl-C in the loop window
            cur['aborting'] = server.keep = False
            if not server.alive():
                last_town = None
                log(out, 'CARLA server gone; the next route starts a new one')
    except KeyboardInterrupt:
        pass
    finally:
        write_json(state_f, dict(state='stopped', route=cur['route'], out=str(out), t=time.time()))
        server.keep = False
        server.stop()


if __name__ == '__main__':
    main()
