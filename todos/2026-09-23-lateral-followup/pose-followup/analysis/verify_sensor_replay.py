"""Read-only historical zero-coefficient full raw-sensor regression evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from pose_contract import replay_sensors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise SystemExit('Refusing existing output')
    inputs={};cases=[]
    def record(p):inputs[str(p.resolve())]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    for source in (Path(__file__),Path(__file__).with_name('pose_contract.py')):record(source)
    for route in ('26966','24240','17563'):
        for variant in ('baseline-max','short-max'):
            directory=args.raw/route/variant/'pursuit'
            for name in ('control.jsonl','motion.jsonl','route_reference.json'):record(directory/name)
            readlines=lambda name:[json.loads(x) for x in (directory/name).read_text().splitlines()]
            result=replay_sensors(readlines('control.jsonl'),readlines('motion.jsonl'),json.loads((directory/'route_reference.json').read_text()),0.)
            cases.append(dict(route=route,variant=variant,complete=result['complete'],frames=len(result['frames']),
                max_position_difference_m=max((r['position_difference_m'] for r in result['frames']),default=None),errors=result['errors']))
    payload=dict(complete=all(r['complete'] for r in cases),total_frames=sum(r['frames'] for r in cases),cases=cases,inputs=inputs,
        description='Independent raw GPS/compass/SPEED/gyro full-sequence zero-k pose reconstruction. No logged pose feedback and no truth input. Historical fixtures, not new physical runs.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(payload,indent=2)+'\n')
    if not payload['complete']:raise SystemExit('Sensor replay mismatch')
    print(json.dumps(dict(complete=payload['complete'],total_frames=payload['total_frames'])))


if __name__=='__main__':main()
