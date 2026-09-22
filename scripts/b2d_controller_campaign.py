#!/usr/bin/env python
"""Run controller comparison groups sequentially on one owned CARLA process.

The existing runner still owns routes, retries and result records. This outer layer
owns the server across groups; evaluator load_world resets every official route.
Run inside tmux. Python 3.8.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from b2d_controller_archive import snapshot, assert_sources_unchanged


def load_runtime():
    # Configuration inspection must not require the evaluator's DATA_DIR env.
    from b2d_run import Runner, Server, parse_args, select_routes
    return Runner, Server, parse_args, select_routes


PRESETS = ('carla', 'tcp', 'pursuit')


def resolve_configs(default_path, presets, mapping_path=None):
    """Validate inputs before launching CARLA; retain exact validated hashes."""
    presets = [x.strip() for x in presets.split(',')]
    if not presets or any(x not in PRESETS for x in presets) or len(set(presets)) != len(presets):
        raise ValueError('presets must be unique members of carla,tcp,pursuit')
    default = Path(default_path).resolve()
    inputs, hashes = [], {}

    def read(path):
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('%s must contain a JSON object' % path)
        if path not in inputs:
            inputs.append(path)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return value

    read(default)
    overrides = {}
    if mapping_path is not None:
        mapping = read(Path(mapping_path).resolve())
        unknown = set(mapping) - set(PRESETS)
        if unknown:
            raise ValueError('unknown preset config keys: %s' % sorted(unknown))
        for preset, value in mapping.items():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError('preset config paths must be absolute: %s' % preset)
            path = Path(value).resolve()
            read(path)
            overrides[preset] = path
    return dict(presets=presets, default=default, inputs=inputs, hashes=hashes,
                selected={preset: overrides.get(preset, default) for preset in presets},
                overrides=overrides)


def archived_configs(resolved, provenance, provenance_dir):
    """Bind execution to archived bytes, rejecting any validation/snapshot race."""
    records = {row['original']: row for row in provenance['inputs']}
    bound = {}
    for source, expected in resolved['hashes'].items():
        record = records[source]
        archived = (Path(provenance_dir) / record['archived']).resolve()
        actual = hashlib.sha256(archived.read_bytes()).hexdigest()
        if actual != record['sha256'] or actual != expected:
            raise ValueError('input changed while snapshotting: %s' % source)
        bound[source] = dict(source_path=source, archived_path=str(archived), sha256=actual)
    selected = {preset: dict(bound[str(path)], fallback=preset not in resolved['overrides'])
                for preset, path in resolved['selected'].items()}
    return selected, bound[str(resolved['default'])]


class Tee:
    def __init__(self, *streams):
        self.streams, self.lock = streams, threading.Lock()

    def write(self, value):
        with self.lock:
            for stream in self.streams:
                stream.write(value); stream.flush()

    def flush(self):
        with self.lock:
            for stream in self.streams: stream.flush()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--routes', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--controller-config', required=True)
    p.add_argument('--preset-configs', help='JSON file mapping preset names to absolute controller config paths; omitted entries use --controller-config')
    p.add_argument('--presets', default='carla,tcp,pursuit')
    p.add_argument('--seeds', default='0')
    p.add_argument('--server-index', type=int, default=70)
    p.add_argument('--round-timeout-s', type=float, default=3600)
    p.add_argument('--cruise-mps', type=float, default=8)
    p.add_argument('--holdout-routes', help='Run the same presets on this XML after all primary groups')
    p.add_argument('--holdout-seed', type=int, default=0)
    p.add_argument('--slope-check', action='store_true', help='Measure stationary brake holding before stopping the owned server')
    a = p.parse_args()
    try:
        resolved=resolve_configs(a.controller_config,a.presets,a.preset_configs)
    except (OSError,ValueError) as exc:
        p.error(str(exc))
    Runner,Server,runner_args,select_routes=load_runtime()
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty; use a new run directory to preserve all attempts')
    out.mkdir(parents=True,exist_ok=True)
    route_path=Path(a.routes).resolve()
    inputs=[route_path]+resolved['inputs']
    if a.holdout_routes:inputs.append(Path(a.holdout_routes).resolve())
    provenance=snapshot(out/'provenance',list(dict.fromkeys(inputs)))
    bindings,default_binding=archived_configs(resolved,provenance,out/'provenance')
    config=Path(default_binding['archived_path'])
    log=(out/'log.txt').open('a');events=(out/'events.jsonl').open('a',buffering=1)
    original=sys.stdout;sys.stdout=Tee(original,log)
    active=[None];round_done=threading.Event()
    server=Server(a.server_index,out/'servers','Epic',gpu_rank=0,windowed=True)
    manifest=dict(config=vars(a),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  preset_controller_configs=bindings,default_controller_config=default_binding,
                  started=time.time(),server_reuse='one process across preset/seed groups; worlds reset by evaluator')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if a.holdout_routes:
        holdout_manifest=dict(manifest)
        holdout_manifest['config']=dict(manifest['config'],routes=str(Path(a.holdout_routes).resolve()),
                                        seeds=str(a.holdout_seed),out=str(out/'holdout'))
        holdout_manifest['parent_campaign']=str(out)
        (out/'holdout').mkdir()
        (out/'holdout'/'manifest.json').write_text(json.dumps(holdout_manifest,indent=2))

    def event(kind,**fields):
        row=dict(t=time.time(),kind=kind,**fields);events.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)

    def interrupt(signum,frame):
        if active[0] is not None:active[0].stop_flag=True
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM,interrupt)
    all_groups=[]
    try:
        server.start()
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)
        (out/'gpu.txt').write_text(gpu);event('gpu',processes=gpu)
        rounds=[(route_path,seed,out) for seed in [int(x) for x in a.seeds.split(',')]]
        if a.holdout_routes:
            rounds.append((Path(a.holdout_routes).resolve(),a.holdout_seed,out/'holdout'))
        for current_routes,seed,group_root in rounds:
            for preset in resolved['presets']:
                binding=bindings[preset]
                if hashlib.sha256(Path(binding['archived_path']).read_bytes()).hexdigest()!=binding['sha256']:
                    raise ValueError('archived controller config changed: %s'%preset)
                assert_sources_unchanged(provenance)
                dest=group_root/('%s-seed%d'%(preset,seed))
                args=runner_args(['--routes',str(current_routes),'--towns','all','--workers','1','--out',str(dest),
                                  '--server-index',str(server.index),'--gpu-rank','0','--windowed',
                                  '--rig','front3','--width','800','--height','450','--decimate','4','--overlap',
                                  '--no-spectator','--zero-copy','--policy','none','--drive','controller',
                                  '--controller-preset',preset,'--controller-config',binding['archived_path'],
                                  '--cruise-mps',str(a.cruise_mps),'--tm-seed',str(seed)])
                routes=select_routes(args.routes,args.towns,args.route_ids,args.limit)
                runner=Runner(args,routes,servers=[server]);active[0]=runner
                runner.learn_maps(server)
                round_done.clear();timed_out=[False]
                def timeout():
                    if not round_done.wait(a.round_timeout_s):
                        timed_out[0]=True;runner.stop_flag=True
                        event('round_timeout',preset=preset,seed=seed)
                watchdog=threading.Thread(target=timeout,daemon=True);watchdog.start()
                started=time.time();event('group_start',preset=preset,seed=seed,controller_config=binding,server_pid=server.proc.pid if server.proc else None)
                code=runner.run();round_done.set();watchdog.join();runner.events.close();active[0]=None
                group=dict(preset=preset,seed=seed,controller_config=binding,routes=str(current_routes),out=str(dest),returncode=code,
                           wall_s=time.time()-started,timed_out=timed_out[0])
                all_groups.append(group);event('group_end',**group)
                with (dest/'report.txt').open('w') as report:
                    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'b2d_report.py'),
                                    '--out',str(dest),'--routes',str(current_routes),'--csv',str(dest/'all-attempts.csv'),
                                    '--controller-json',str(dest/'controller-report.json')],stdout=report,stderr=subprocess.STDOUT,check=True)
                (out/'groups.json').write_text(json.dumps(all_groups,indent=2))
                if code==130 or timed_out[0]:
                    event('end',status='interrupted');return 130
                # Exhausted infrastructure failures stop the owned server. Its normal
                # restart path is retained for the following comparison group.
                if not server.alive():server.start()
        if a.slope_check:
            import carla
            from b2d_controller_slope import run as slope_run
            client=carla.Client('127.0.0.1',server.port);client.set_timeout(120.)
            event('slope_start',server_pid=server.proc.pid if server.proc else None)
            slope=slope_run(client,config,out/'slope')
            event('slope_end',status=slope['status'],gate_pass=slope['gate_pass'])
        event('end',status='completed',groups=len(all_groups))
        return 1 if any(x['returncode'] for x in all_groups) else 0
    finally:
        round_done.set()
        if active[0] is not None:active[0].stop_flag=True
        server.stop();sys.stdout=original;log.close();events.close()


if __name__=='__main__':
    raise SystemExit(main())
