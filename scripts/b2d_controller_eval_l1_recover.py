"""Resume missing Town13 L1 cases one CARLA server per case; never rerun valid driving."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data/runs/b2d/controller-eval')
INPUT=ROOT/'todos/2026-09-23-tfv6-controller/controller-eval'
PRESET={'A':'author_route','B':'author_waypoint','C':'pursuit','D':'pursuit'}


def case_dir(source,seed,arm):
    return source/seed/'3364'/arm/PRESET[arm]


def status(path):
    file=path/'validation.json'
    if not file.exists():return None
    d=json.loads(file.read_text())
    if not d.get('exception') and not d.get('cleanup_errors') and not d.get('telemetry_parse_errors') and d['gates']['telemetry_complete']:
        return 'valid'
    text=str(d.get('exception'))
    if d.get('ticks')==0 and ('time-out of 90000ms' in text or d.get('status')=='setup_error'):
        return 'infrastructure'
    return 'bug'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',choices=('oracle','expert'),required=True)
    p.add_argument('--sources',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--server-index',type=int,default=104)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sources=list(a.sources)
    for seed in ('p01','p02','p03','p04','p05'):
        for arm in 'ABCD':
            places=[(source,case_dir(source,seed,arm)) for source in sources]
            valid=[path for _,path in places if status(path)=='valid']
            if len(valid)>1:raise RuntimeError(f'Duplicate valid {seed}/{arm}: {valid}')
            if valid:
                print('skip valid',seed,arm,valid[0],flush=True);continue
            failures=[path for _,path in places if status(path)=='infrastructure']
            bugs=[path for _,path in places if status(path)=='bug']
            if bugs:raise RuntimeError(f'Non-infrastructure failure {seed}/{arm}: {bugs}')
            if len(failures)>=3:raise RuntimeError(f'Infrastructure retry limit {seed}/{arm}: {failures}')
            for attempt_no in range(len(failures)+1,4):
                target=a.out/f'{seed}-{arm}-attempt-{attempt_no}'
                if target.exists():raise RuntimeError(f'Refusing overwrite: {target}')
                one=target.parent/f'case-{seed}-{arm}-attempt-{attempt_no}.json'
                one.write_text(json.dumps([dict(route='3364',perturbation_id=seed,
                                                variant=arm,preset=PRESET[arm])]))
                command=[str(Path('/data/envs/tfv6/bin/python')),str(ROOT/'scripts/b2d_controller_validate.py'),
                    '--routes',str(BASE/'route-3364.xml'),'--out',str(target),
                    '--variants',str(INPUT/'l1-variants.json'),
                    '--route-cruises',str(INPUT/'l1-cruises.json'),
                    '--perturbations',str(INPUT/'perturbations.json'),
                    '--perturbation-ids',seed,'--case-list',str(one),
                    '--server-index',str(a.server_index),'--max-ticks','1800',
                    '--rig','none','--no-rendering','--strict-invariants']
                if a.reference=='expert':
                    # Only route 3364 is in this recovery process.
                    ref={"3364":json.loads((BASE/'references/reference-traces.json').read_text())['3364']}
                    ref_file=target.parent/'recovery-reference-3364.json'
                    ref_file.write_text(json.dumps(ref))
                    command.extend(['--reference-traces',str(ref_file)])
                environment=os.environ.copy()
                environment.update(DATA_DIR='/data',BENCH2DRIVE_ROOT='/data/runs/b2d/tfv6-repro/runtime/Bench2Drive',
                                   OPENBLAS_CORETYPE='Barcelona')
                log=target.parent/f'{target.name}.log'
                print('start',datetime.now(timezone.utc).isoformat(),seed,arm,attempt_no,flush=True)
                with log.open('w') as f:
                    rc=subprocess.run(command,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT).returncode
                path=case_dir(target,seed,arm)
                kind=status(path)
                print('end',datetime.now(timezone.utc).isoformat(),seed,arm,attempt_no,rc,kind,flush=True)
                sources.append(target)
                if kind=='valid':break
                if kind!='infrastructure':raise RuntimeError(f'Runner/agent bug {seed}/{arm}: {path}, rc={rc}')
                if attempt_no==3:raise RuntimeError(f'Infrastructure retry limit {seed}/{arm}')
    (a.out/'sources.json').write_text(json.dumps([str(x) for x in sources],indent=2)+'\n')
    print('complete all 20 Town13 cases',flush=True)


if __name__=='__main__':main()
