#!/usr/bin/env python
"""Six paired actual-TCP routes, one owned CARLA process, exclusive raw archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tqdm import tqdm
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from b2d_controller_archive import snapshot, digest
from b2d_controller_campaign import Tee


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--server-index', type=int, default=101)
    parser.add_argument('--routes')
    parser.add_argument('--checkpoint')
    parser.add_argument('--protocol', default='todos/2026-09-23-tcp-controller/protocol.md')
    a = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') and \
            len(subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()) > 1:
        parser.error('Several GPUs visible: pin one with CUDA_VISIBLE_DEVICES (Tokyo: 1, its GPU 0 is broken)')
    from b2d_run import B2D_ZOO, BENCH2DRIVE, DATA_DIR, Runner, Server, parse_args, select_routes
    a.routes = a.routes or str(BENCH2DRIVE/'leaderboard/data/bench2drive220.xml')
    a.checkpoint = a.checkpoint or str(DATA_DIR/'models/bench2drive/tcp/tcp_b2d.ckpt')
    repo = Path(__file__).resolve().parents[1]
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use a fresh output directory; evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(IS_BENCH2DRIVE='1', PLANNER_TYPE='only_traj', TORCH_HOME=str(DATA_DIR/'models/torch'),
               B2D_TCP_OPTIMIZE='1', B2D_TCP_PIPELINE='1', B2D_TCP_FAST_COLOR='1',
               B2D_TCP_DEBUG_VIEWS='1', B2D_TCP_EARLY_RGB='1', B2D_ASYNC_DISPLAY='1',
               B2D_CAPTURE_CRITERION_EVENTS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPATH=':'.join(map(str, (repo/'scripts', B2D_ZOO, B2D_ZOO/'TCP'))))
    os.environ.update(env)
    vendor = B2D_ZOO
    inputs = [Path(a.routes), Path(a.protocol)] + [vendor/p for p in (
        'team_code/tcp_b2d_agent.py', 'team_code/planner.py', 'TCP/model.py', 'TCP/config.py')]
    provenance = snapshot(out/'provenance', inputs)
    bindings = {repo/row['path']: row['sha256'] for row in provenance['sources']}
    bindings.update({Path(row['original']): row['sha256'] for row in provenance['inputs']})
    checkpoint_sha = digest(a.checkpoint)
    if checkpoint_sha != 'e6573ff1f8ea910b9a53eddfb68f69cac469bf5bfa253a516578f6126110b4fe':
        raise ValueError('Checkpoint differs from declared TCP model')
    manifest = dict(started=time.time(), config=vars(a), environment=env,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    checkpoint_sha256=checkpoint_sha, routes=['24211', '1711', '1773'],
                    arms=['native_common', 'pi_common'], tm_seed=0,
                    world_reset='evaluator load_world resets each route; one process until crash',
                    route_timeout_s=600, max_attempts=2)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    original = sys.stdout
    log = (out/'log.txt').open('x', buffering=1)
    events = (out/'events.jsonl').open('x', buffering=1)
    sys.stdout = Tee(original, log)
    tb = EventFileWriter(str(out/'tb'))
    progress = tqdm(total=6, desc='TCP paired cases', file=sys.stdout)
    server = Server(a.server_index, out/'servers', 'Epic', gpu_rank=0)
    active = [None]
    def event(kind, **fields):
        row = dict(t=time.time(), kind=kind, **fields)
        events.write(json.dumps(row)+'\n'); print(json.dumps(row), flush=True)
    def interrupt(signum, frame):
        if active[0] is not None:
            active[0].stop_flag = True
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    groups = []
    try:
        server.start()
        event('start', planned_groups=6, server_pid=server.proc.pid)
        for rid in manifest['routes']:
            for arm in manifest['arms']:
                for source, sha in bindings.items():
                    if digest(source) != sha:
                        raise RuntimeError('Frozen source/input changed: '+str(source))
                dest = out/(rid+'-'+arm)
                os.environ.update(B2D_TCP_CONTROL_ARM=arm, SAVE_PATH=str(dest/'vendor-save'),
                                  B2D_PREVIEW_DIR=str(dest/'live'))
                args = parse_args(['--routes', a.routes, '--route-ids', rid, '--towns', 'all',
                    '--workers', '1', '--out', str(dest), '--server-index', str(server.index),
                    '--gpu-rank', '0'] + (['--windowed'] if server.windowed else []) + ['--no-spectator', '--zero-copy',
                    '--agent', str(repo/'scripts/b2d_tcp_comparison_agent.py'), '--agent-config', a.checkpoint,
                    '--python', str(DATA_DIR/'envs/b2d-tcp/bin/python'), '--decimate', '1', '--tm-seed', '0',
                    '--route-timeout-s', '600', '--stall-s', '120', '--max-attempts', '2'])
                runner = Runner(args, select_routes(args.routes, args.towns, args.route_ids, args.limit), servers=[server])
                active[0] = runner
                runner.learn_maps(server)
                start = time.time(); event('group_start', route=rid, arm=arm, server_pid=server.proc.pid)
                code = runner.run(); runner.events.close(); active[0] = None
                group = dict(route=rid, arm=arm, out=str(dest), returncode=code, wall_s=time.time()-start)
                groups.append(group); (out/'groups.json').write_text(json.dumps(groups, indent=2))
                tb.add_event(Event(wall_time=time.time(), step=len(groups), summary=Summary(value=[
                    Summary.Value(tag=arm+'/attempt_group_wall_s', simple_value=group['wall_s'])])))
                tb.flush(); progress.update(1)
                event('group_end', **group)
                if code == 130:
                    event('end', status='interrupted'); return 130
                if not server.alive():
                    server.start()
        event('end', status='completed', groups=len(groups))
        return 1 if any(g['returncode'] for g in groups) else 0
    finally:
        if active[0] is not None:
            active[0].stop_flag = True
        server.stop()
        progress.close(); tb.close()
        sys.stdout = original; log.close(); events.close()


if __name__ == '__main__':
    raise SystemExit(main())
