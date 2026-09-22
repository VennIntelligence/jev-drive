#!/usr/bin/env python
"""Run controller comparison groups sequentially on one owned CARLA process.

The existing runner still owns routes, retries and result records. This outer layer
owns the server across groups; evaluator load_world resets every official route.
Run inside tmux. Python 3.8.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from b2d_run import Runner, Server, parse_args as runner_args, select_routes


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
    p.add_argument('--presets', default='carla,tcp,pursuit')
    p.add_argument('--seeds', default='0')
    p.add_argument('--server-index', type=int, default=70)
    p.add_argument('--round-timeout-s', type=float, default=3600)
    p.add_argument('--cruise-mps', type=float, default=8)
    a = p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    route_path=Path(a.routes).resolve();config=Path(a.controller_config).resolve()
    log=(out/'log.txt').open('a');events=(out/'events.jsonl').open('a',buffering=1)
    original=sys.stdout;sys.stdout=Tee(original,log)
    active=[None];round_done=threading.Event()
    server=Server(a.server_index,out/'servers','Epic',gpu_rank=0,windowed=True)
    manifest=dict(config=vars(a),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  started=time.time(),server_reuse='one process across preset/seed groups; worlds reset by evaluator')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

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
        for seed in [int(x) for x in a.seeds.split(',')]:
            for preset in a.presets.split(','):
                dest=out/('%s-seed%d'%(preset,seed))
                args=runner_args(['--routes',str(route_path),'--towns','all','--workers','1','--out',str(dest),
                                  '--server-index',str(server.index),'--gpu-rank','0','--windowed',
                                  '--rig','front3','--width','800','--height','450','--decimate','4','--overlap',
                                  '--no-spectator','--zero-copy','--policy','none','--drive','controller',
                                  '--controller-preset',preset,'--controller-config',str(config),
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
                started=time.time();event('group_start',preset=preset,seed=seed,server_pid=server.proc.pid if server.proc else None)
                code=runner.run();round_done.set();watchdog.join();runner.events.close();active[0]=None
                group=dict(preset=preset,seed=seed,returncode=code,wall_s=time.time()-started,timed_out=timed_out[0])
                all_groups.append(group);event('group_end',**group)
                with (dest/'report.txt').open('w') as report:
                    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'b2d_report.py'),
                                    '--out',str(dest),'--routes',str(route_path),'--csv',str(dest/'all-attempts.csv'),
                                    '--controller-json',str(dest/'controller-report.json')],stdout=report,stderr=subprocess.STDOUT,check=True)
                (out/'groups.json').write_text(json.dumps(all_groups,indent=2))
                if code==130 or timed_out[0]:
                    event('end',status='interrupted');return 130
                # Exhausted infrastructure failures stop the owned server. Its normal
                # restart path is retained for the following comparison group.
                if not server.alive():server.start()
        event('end',status='completed',groups=len(all_groups))
        return 1 if any(x['returncode'] for x in all_groups) else 0
    finally:
        round_done.set()
        if active[0] is not None:active[0].stop_flag=True
        server.stop();sys.stdout=original;log.close();events.close()


if __name__=='__main__':
    raise SystemExit(main())
