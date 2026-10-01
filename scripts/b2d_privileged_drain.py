"""Drain exact owned wrappers naturally, then terminate only the paused coordinator."""
import json
import os
from pathlib import Path
import signal
import sys
import time

from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevdrive.runlog import RunLog


def state(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    except FileNotFoundError:return None


def main():
    root=Path(os.environ['DATA_DIR'])/'runs/b2d_privileged_ceiling'
    request=json.loads((root/'QUEUE_DRAIN.json').read_text())
    log=RunLog('b2d_privileged_ceiling','queue-drain');completed=set()
    try:
        with tqdm(total=len(request['wrappers']),desc='Naturally draining old shards') as progress:
            while len(completed)<len(request['wrappers']):
                for wrapper in request['wrappers']:
                    pid=wrapper['pid']
                    if pid in completed:continue
                    fields=state(pid)
                    if fields is not None and fields[19]==wrapper['start'] and fields[0]!='Z':continue
                    adir=root/'arms'/wrapper['unit']
                    assert all((adir/'done'/(rid+'.json')).exists() for rid in wrapper['ids']),f'Incomplete drained unit: {wrapper}'
                    completed.add(pid);progress.update(1)
                    log.info('Naturally completed '+wrapper['unit']);log.event('drained_unit',**wrapper)
                pids=int(Path('/sys/fs/cgroup/pids.current').read_text())
                assert pids<17500,'PID hard checklist failed while draining'
                log.scalar('resources/pids',pids,int(time.time()))
                temporary=root/'STATUS-drain.tmp'
                temporary.write_text(time.strftime('%F %T')+f' draining q4 naturally: {len(completed)}/{len(request["wrappers"])} wrappers finished; control unchanged\n')
                temporary.replace(root/'STATUS')
                if len(completed)<len(request['wrappers']):time.sleep(5)
        pid=request['chain_pid'];fields=state(pid)
        assert fields is not None and fields[19]==request['chain_start'] and fields[0]=='T','Paused coordinator PID changed'
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        assert len(cmd)>1 and cmd[1].endswith(b'scripts/b2d_privileged_chain.py')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
        for _ in range(50):
            fields=state(pid)
            if fields is None or fields[0]=='Z':break
            time.sleep(.1)
        assert fields is None or fields[0]=='Z','Coordinator did not exit'
        result=dict(request=request,naturally_completed_wrappers=len(completed),terminated_coordinator=pid,
                    trajectories_interrupted=0,control_unchanged=True,time=time.time())
        (root/'QUEUE_DRAIN_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        log.info('Drain complete; all trajectories retained, only paused coordinator terminated')
        log.event('end',status='complete',**result)
    except BaseException as exc:
        (root/'ERROR').write_text('Queue drain failed: '+repr(exc)+'\n');raise
    finally:log.close()


if __name__=='__main__':main()
