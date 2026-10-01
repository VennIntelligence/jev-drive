"""Check event parsing against constructed histories and official debug route records."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevdrive.runlog import RunLog
from b2d_privileged_report import event_rows, bootstrap_delta, read_unit


def lines(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def main():
    root=Path(os.environ['DATA_DIR'])/'runs/b2d_privileged_ceiling'
    log=RunLog('b2d_privileged_ceiling','readout-check')
    try:
        fixture=log.dir/'fixture';fixture.mkdir()
        (fixture/'route.json').write_text(json.dumps(dict(xy=[[0,0],[200,0]])))
        scenes=[]
        for t,s,groups in ((0,10,[[1],[2]]),(1,20,[[1],[2]]),(2,100,[[1,2]])):
            obstacles=[dict(key=ids[0],ids=ids,start_s=30,end_s=60) for ids in groups]
            scenes.append(dict(t=t,ego=dict(v=2,xyz=[s,0,0]),actors=[],
                          pc=dict(warm=False,ego_s=s,junctions=[],obstacles=obstacles,bypass=True,
                                  bypass_state=dict(ids=[1,2],borrow=False),gap_open=True)))
        lines(fixture/'privileged.jsonl',scenes);lines(fixture/'plans.jsonl',[])
        lines(fixture/'contacts.jsonl',[dict(t=1.1,id=2,impulse=2),dict(t=1.2,id=2,impulse=2)])
        meta=dict(route='fixture',arm='pbyp',seed=0,group='obstacle')
        ee,_=event_rows(fixture,meta)
        assert len(ee)==1 and json.loads(ee[0]['ids'])==[1,2]
        assert ee[0]['passed']==1 and ee[0]['returned']==1 and ee[0]['failed']==1 and ee[0]['contacts']==1
        lines(fixture/'contacts.jsonl',[])
        ee,_=event_rows(fixture,meta)
        assert len(ee)==1 and ee[0]['success']==1 and ee[0]['failed']==0
        rr=pd.DataFrame([dict(route=str(i),arm=a,seed=s,group='junction',DS=50+5*(a=='pjunc'))
                         for i in range(4) for s in (0,1) for a in ('drive','pjunc')])
        ev=pd.DataFrame([dict(route=str(i),arm=a,seed=s,group='junction',kind='junction',
                             failed=int(a=='drive'),opportunity=1)
                        for i in range(4) for s in (0,1) for a in ('drive','pjunc')])
        delta=bootstrap_delta(rr,ev,'pjunc','junction','junction')
        assert all(delta[k]==5 for k in ('dDS','DS_lo','DS_hi'))
        assert all(delta[k]==-1 for k in ('fail_delta','fail_lo','fail_hi'))
        assert delta['routes']==4 and delta['drive_opps']==8 and delta['arm_opps']==8
        all_kinds=bootstrap_delta(rr,ev,'pjunc','junction','all')
        assert all_kinds['fail_delta']==-1 and all_kinds['dDS']==5 and all_kinds['arm_opps']==8
        checked=[]
        for name in tqdm(('debug-v3-27787-drive-s0','debug-v3-26872-pjunc-s0','debug-v3-25169-pbyp-s0'),desc='Official debug readouts'):
            adir=root/'arms'/name;arm=name.split('-')[-2]
            started=time.perf_counter();read_unit(adir,'audit-debug',arm,0,log)
            rows=pd.read_csv(root/'readouts'/name/'routes.csv')
            assert len(rows)==1 and np.isfinite(rows[['DS','RC','v_mean']].to_numpy()).all()
            route=str(int(rows.iloc[0]['route']));done=json.loads((adir/'done'/(route+'.json')).read_text())
            attempt=adir/'attempts'/route/str(done['attempt'])
            official=json.loads((attempt/'results.json').read_text())['_checkpoint']['records'][0]
            assert rows.iloc[0]['DS']==official['scores']['score_composed']
            assert rows.iloc[0]['RC']==official['scores']['score_route']
            for column,field in (('coll_veh','collisions_vehicle'),('red_light','red_light')):
                assert rows.iloc[0][column]==len(official['infractions'][field])
            checked.append(dict(unit=name,parse_s=time.perf_counter()-started,
                                DS=float(rows.iloc[0]['DS']),RC=float(rows.iloc[0]['RC'])))
        result=dict(connected_groups_one_opportunity=True,related_contact_failure=True,
                    contact_free_return_success=True,same_actor_contacts_merged=True,
                    bootstrap_clusters_seeds_counts=True,bootstrap_2000_seed0=True,
                    official_debug_records=checked,formal_data_used=False)
        (log.dir/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'readout_checks.json').write_text(json.dumps(result,indent=2)+'\n')
        log.info('Readout checks passed: '+json.dumps(result));log.event('end',status='complete')
    finally:log.close()


if __name__=='__main__':main()
