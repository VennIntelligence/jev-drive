"""Audit official prefixed crash statuses and archive logging-crashed attempts.

Raw trajectories are never deleted. Only failed completion pointers and derived caches
are moved out of the formal estimator after the owned coordinator has drained.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from tqdm import tqdm

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from jevdrive.runlog import RunLog
from b2d_privileged_checks import is_crash

BASE_COMMIT='c1ab219'
OLD_FIELD='"rel": rel, "rb": self.resume_blocked'
NEW_FIELD='"rel": bool(rel) if isinstance(rel, np.bool_) else rel, "rb": self.resume_blocked'


def source_check():
    paths=('scripts/op_arb_agent.py','scripts/b2d_privileged_geometry.py')
    old=[subprocess.check_output(['git','show',BASE_COMMIT+':'+p],cwd=REPO) for p in paths]
    new=[(REPO/p).read_bytes() for p in paths]
    assert old[1]==new[1], 'Privileged geometry changed during logging repair'
    assert old[0].decode().count(OLD_FIELD)==1
    assert new[0]==old[0].decode().replace(OLD_FIELD,NEW_FIELD).encode(), 'Repair changed more than one logging field'
    tree=ast.parse(new[0])
    expressions=[value for node in ast.walk(tree) if isinstance(node,ast.Dict)
                 for key,value in zip(node.keys,node.values) if isinstance(key,ast.Constant) and key.value=='rel']
    assert len(expressions)==1
    expression=compile(ast.Expression(expressions[0]),'<actual plan-log field>','eval')
    for value in (None,False,True,np.bool_(False),np.bool_(True),'timeout','privileged_clear'):
        normalized=eval(expression,{'rel':value,'np':np})
        decoded=json.loads(json.dumps({'rel':normalized}))['rel']
        assert decoded==value and (not isinstance(value,np.bool_) or type(decoded) is bool)
    for status in ('Failed - Agent crashed','Failed - Simulation crashed',"Failed - Agent couldn't be set up",
                   "Failed - Agent's sensors were invalid",'Agent crashed','Failed'):
        assert is_crash(status),status
    for status in ('Perfect','Completed','Failed - TickRuntime','Failed - Agent got blocked'):
        assert not is_crash(status),status
    return dict(base_commit=BASE_COMMIT,old_control_sha256=hashlib.sha256(b''.join(old)).hexdigest(),
                new_control_sha256=hashlib.sha256(b''.join(new)).hexdigest(),
                geometry_identical=True,only_plan_log_rel_normalized=True,
                actual_log_expression_checked=True,prefixed_crash_classifier_checked=True,
                model_and_control_computation_unchanged=True)


def main():
    root=Path(os.environ['DATA_DIR'])/'runs/b2d_privileged_ceiling'
    log=RunLog('b2d_privileged_ceiling','crash-repair-prepare')
    try:
        result=source_check()
        drain=json.loads((root/'QUEUE_DRAIN_RESULT.json').read_text())
        assert drain['request']['chain_pid']==541722 and drain['reason'].startswith('Stopped after substantive official crash-status audit')
        assert not Path('/proc/541722').exists() or Path('/proc/541722/stat').read_text().split(') ',1)[1].startswith('Z')
        destination=root/'crash-audit/invalidated';destination.mkdir(parents=True,exist_ok=True)
        failures=[];valid=[]
        for pointer in tqdm(sorted((root/'arms').glob('eval-*/done/*.json')),desc='Independent official-status audit'):
            unit=pointer.parent.parent
            attempt=unit/'attempts'/pointer.stem/str(json.loads(pointer.read_text())['attempt'])
            record=json.loads((attempt/'results.json').read_text())['_checkpoint']['records'][0]
            if not is_crash(record['status']):
                valid.append(dict(unit=unit.name,route=pointer.stem,attempt=str(attempt),status=record['status']))
                continue
            text=(attempt/'route.log').read_text(errors='replace')
            assert record['status']=='Failed - Agent crashed' and 'Object of type bool_ is not JSON serializable' in text, (str(attempt),record['status'])
            failures.append(dict(unit=unit.name,route=pointer.stem,attempt=str(attempt),status=record['status'],
                                 reason='NumPy bool_ in existing latch-release plan log',DS=record['scores']['score_composed'],RC=record['scores']['score_route']))
        assert failures, 'Expected logging-crashed attempts were not present'
        result.update(invalidated_attempts=failures,retained_valid_attempts=len(valid),
                      retained_official_statuses={s:sum(row['status']==s for row in valid) for s in sorted({row['status'] for row in valid})},
                      drain=drain,raw_attempts_deleted=0,driving_failures_retried=0)
        (root/'crash_repair_source_checks.json').write_text(json.dumps(result,indent=2)+'\n')
        for name in ('lock.json','debug_checks.json'):
            target=root/('pre-logging-v5-'+name)
            assert not target.exists(), 'Preparation already ran; do not invalidate more attempts'
            target.write_bytes((root/name).read_bytes())
        affected={row['unit'] for row in failures}
        for row in failures:
            target=destination/row['unit']/'done'/ (row['route']+'.json');target.parent.mkdir(parents=True,exist_ok=True)
            (root/'arms'/row['unit']/'done'/(row['route']+'.json')).rename(target)
        for unit in affected:
            target=destination/unit
            for source,name in ((root/'readouts'/unit,'readouts'),(root/'units'/(unit+'.json'),'unit.json'),(root/'arms'/unit/'DONE','DONE')):
                if source.exists():source.rename(target/name)
        (root/'DONE-crash-prepare').write_text('Independent audit and logging-only source checks passed\n')
        log.info('Logging repair prepared: '+json.dumps(result));log.event('end',status='complete',**result)
    except BaseException as exc:
        (root/'ERROR').write_text('Crash-repair preparation failed: '+repr(exc)+'\n');raise
    finally:log.close()


if __name__=='__main__':main()
