"""Compare fixed Cinque input sequences in serial and four concurrent isolated sessions."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevdrive.runlog import RunLog
import zeroshot_wire as wire


def main():
    root=Path(os.environ['DATA_DIR'])/'runs/b2d_privileged_ceiling'
    unit=json.loads((root/'units/profile-v4-after-g0-k0-drive-s0.json').read_text())
    assert unit['gpu']==0 and unit['slot']==0 and unit['workers']==4
    assert not (root/'FULL_STARTED').exists(),'Only check an idle profile server before formal evaluation'
    log=RunLog('b2d_privileged_ceiling','pool-check')
    connections=[]
    try:
        endpoint=root/'card0s0/srv/op.sock'
        for _ in range(4):
            conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);conn.settimeout(30)
            conn.connect(str(endpoint));connections.append(conn)
        rng=np.random.default_rng(0)
        images={k:rng.integers(0,256,(1208,1928,4),dtype=np.uint8) for k in ('OP_ROAD','OP_WIDE')}
        for image in images.values():image[...,3]=255
        desires=(0,0,1,1,0,2)
        def sequence(conn,barrier=None):
            wire.send(conn,dict(cmd='reset'));reply,_=wire.recv(conn);assert reply['ok']
            if barrier is not None:barrier.wait(timeout=30)
            outputs=[]
            for i,desire in enumerate(desires):
                wire.send(conn,dict(cmd='plan',seed=0,dump='',speed=2.,t=i*.05,desire=desire,twin=False,intent=0),images)
                _,out=wire.recv(conn);outputs.append(out)
            return outputs
        start=time.perf_counter();reference=sequence(connections[0]);serial_s=time.perf_counter()-start
        barrier=threading.Barrier(4);start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(sequence,conn,barrier) for conn in connections]
            concurrent=[f.result() for f in tqdm(futures,desc='Fixed-input Cinque sessions')]
        parallel_s=time.perf_counter()-start;errors={}
        for outputs in concurrent:
            for actual,expected in zip(outputs,reference):
                assert actual.keys()==expected.keys()
                for key in actual:
                    assert np.isfinite(actual[key]).all()
                    errors[key]=max(errors.get(key,0.),float(np.max(np.abs(actual[key]-expected[key]))))
        assert max(errors.values())<=1e-5,errors
        result=dict(passed=True,input='Fixed seeded native road and wide BGRA arrays; six recurrent steps and desire changes',
                    sessions=4,steps_per_session=6,maxabs_by_head=errors,tolerance=1e-5,
                    serial_s=serial_s,parallel_four_sequences_s=parallel_s,
                    server=str(endpoint),server_pool=4,formal_data_used=False,
                    formal_started_during_check=(root/'FULL_STARTED').exists())
        assert not result['formal_started_during_check']
        (root/'pool_checks.json').write_text(json.dumps(result,indent=2)+'\n')
        (log.dir/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
        log.info('Fixed-input pool checks passed: '+json.dumps(result));log.event('end',status='complete')
    finally:
        for conn in connections:conn.close()
        log.close()


if __name__=='__main__':main()
