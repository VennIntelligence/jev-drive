"""Explicit historical metrics-only aliases; never eligible for live qualification."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

BASE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=Path('/data/runs/b2d/controller/turns-v1'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    summary={}
    for fixture in ('same','missing-frame','missing-pose-field'):
        for route in ('24240','26966','17563'):
            source=args.raw/route/'baseline-max/pursuit'
            for variant in ('baseline-zero','candidate-fixed-k'):
                target=args.out/fixture/route/variant/'pursuit';target.mkdir(parents=True)
                for name in ('control.jsonl','route_reference.json','agent_config.json','validation_trace.json','validation.json','motion.jsonl'):
                    if fixture=='missing-frame' and route=='26966' and variant=='candidate-fixed-k' and name=='control.jsonl':
                        data=source.joinpath(name).read_text().splitlines()
                        prior=list(csv.DictReader((args.out/'same-analysis/frames.csv').open()))
                        frame=next(int(r['frame']) for r in prior if r['route']=='26966' and r['variant']=='baseline-zero' and 24<=float(r['progress'])<=51)
                        target.joinpath(name).write_text('\n'.join(line for line in data if json.loads(line)['frame']!=frame)+'\n')
                    elif fixture=='missing-pose-field' and route=='26966' and variant=='candidate-fixed-k' and name=='control.jsonl':
                        data=[json.loads(line) for line in source.joinpath(name).read_text().splitlines()]
                        data[100]['pose_status'].pop('lateral_dt_s',None)
                        target.joinpath(name).write_text(''.join(json.dumps(r)+'\n' for r in data))
                    else:target.joinpath(name).symlink_to(source/name)
        cmd=[sys.executable,str(BASE/'analyze_pose.py'),'--run-root',str(args.out/fixture),'--out',str(args.out/(fixture+'-analysis')),
             '--protocol',str(BASE.parent/'protocol.md')]
        if fixture!='missing-pose-field':cmd.append('--legacy-fixture')
        r=subprocess.run(cmd,capture_output=True,text=True);(args.out/(fixture+'.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError(r.stderr)
        conditions=json.loads((args.out/(fixture+'-analysis/required-conditions.json')).read_text())
        bad=[r for r in conditions['required_conditions'] if r['status']!='pass']
        if fixture=='same':assert len(bad)==1 and bad[0]['condition']=='turn/26966/1/primary_cte_rms_15percent'
        elif fixture=='missing-frame':
            assert any(r['status']=='fail' and r['condition']=='case/26966/candidate-fixed-k/complete_frames_and_truth' for r in bad)
            assert any(r['status']=='fail' and r['condition']=='coverage/26966/candidate-fixed-k/1' for r in bad)
        else:assert any(r['status']=='fail' and r['condition']=='case/26966/candidate-fixed-k/pose_compensation_contract' for r in bad)
        summary[fixture]=dict(expected_failure_proved=True,nonpassing=bad,live_qualification_allowed=conditions['live_qualification_allowed'])
    (args.out/'verify-source.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:len(v['nonpassing']) for k,v in summary.items()}))


if __name__=='__main__':main()
