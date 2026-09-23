"""Descriptive repeatability audit for the exact D2 GO list."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from b2d_tfv6_d2 import GROUPS, ROOT

FORMAL = Path('/data/runs/b2d/tfv6-w2/formal')
OUT = Path(__file__).resolve().parents[1] / 'todos/2026-09-23-tfv6-controller/results/diagnosis/d2'
EXCLUDE = {'ROUTE_COMPLETION', 'MIN_SPEED_INFRACTION'}


def case_path(level, route, seed, arm, repeat):
    base = ROOT / f'repeat-{repeat}' if repeat else FORMAL / f'level{level}'
    return base / 'cases' / level / f'route-{route}' / f'seed-{seed}' / arm


def details(level, route, seed, arm, repeat):
    folder = case_path(level, route, seed, arm, repeat)
    done = json.loads((folder / 'done.json').read_text())
    run = Path(done['run_dir'])
    rec = json.loads((run / 'attempts' / route / '1' / 'results.json').read_text())['_checkpoint']['records'][0]
    events = json.loads((run.parent / 'infractions.json').read_text())['infractions']
    all_events = [(e['event_type'].split('.')[-1], int(e['step'])) for e in events
                  if e['event_type'].split('.')[-1] != 'ROUTE_COMPLETION']
    scored = [e for e in all_events if e[0] not in EXCLUDE]
    frames = [json.loads(line) for line in (run.parent / 'frames.jsonl').open()]
    status = rec['status']
    if 'deviated from the route' in status:
        scored.append(('ROUTE_DEVIATION', frames[-1]['step']))
        all_events.append(('ROUTE_DEVIATION', frames[-1]['step']))
    if 'got blocked' in status:
        scored.append(('VEHICLE_BLOCKED', frames[-1]['step']))
        all_events.append(('VEHICLE_BLOCKED', frames[-1]['step']))
    first = min(scored, key=lambda x: x[1]) if scored else (None, None)
    first_any = min(all_events, key=lambda x: x[1]) if all_events else (None, None)
    recorder_path = (next((run / 'attempts' / route / '1' / 'recorder').glob('*.log'))
                     if repeat else None)
    near = None
    if first[1] is not None:
        frame = min(frames, key=lambda f: abs(f['step'] - first[1]))
        nearby = frame.get('nearby_actors')
        if isinstance(nearby, list):
            near = nearby[:3]
    return dict(level=level, route=route, seed=seed, arm=arm, repeat=repeat,
                ds=float(rec['scores']['score_composed']), rc=float(rec['scores']['score_route']),
                status=status, first_type=first[0],
                first_time_s=round(first[1]*.05, 2) if first[1] is not None else None,
                first_step=first[1], nearest_at_first=near,
                first_any_type=first_any[0],
                first_any_time_s=round(first_any[1]*.05, 2) if first_any[1] is not None else None,
                recorder=str(recorder_path) if recorder_path else None,
                recorder_bytes=recorder_path.stat().st_size if recorder_path else None,
                frames=len(frames),
                actor_ticks=sum(isinstance(f.get('nearby_actors'), list) for f in frames),
                actor_errors=sum(isinstance(f.get('nearby_actors'), dict) for f in frames),
                run_dir=str(run))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for level, route, seed, arms in GROUPS:
        for arm in arms:
            original = details(level, route, seed, arm, 0)
            for repeat in (1, 2):
                observed = details(level, route, seed, arm, repeat)
                same_type = observed['first_type'] == original['first_type']
                if observed['first_time_s'] is None or original['first_time_s'] is None:
                    time_within_1s = observed['first_time_s'] == original['first_time_s']
                else:
                    time_within_1s = abs(observed['first_time_s'] - original['first_time_s']) <= 1.0
                if observed['first_any_time_s'] is None or original['first_any_time_s'] is None:
                    any_time_within_1s = observed['first_any_time_s'] == original['first_any_time_s']
                else:
                    any_time_within_1s = abs(observed['first_any_time_s'] - original['first_any_time_s']) <= 1.0
                row = dict(level=level, route=route, seed=seed, arm=arm, repeat=repeat,
                           original_ds=original['ds'], repeat_ds=observed['ds'],
                           delta_ds=round(observed['ds']-original['ds'], 6),
                           ds_exact=abs(observed['ds']-original['ds']) <= .01,
                           original_rc=original['rc'], repeat_rc=observed['rc'],
                           original_status=original['status'], repeat_status=observed['status'],
                           status_exact=original['status']==observed['status'],
                           original_first=original['first_type'] or '',
                           repeat_first=observed['first_type'] or '',
                           first_type_exact=same_type,
                           original_first_s=original['first_time_s'] if original['first_time_s'] is not None else '',
                           repeat_first_s=observed['first_time_s'] if observed['first_time_s'] is not None else '',
                           first_time_within_1s=time_within_1s,
                           first_outcome_reproduced=same_type and time_within_1s,
                           original_first_any=original['first_any_type'] or '',
                           repeat_first_any=observed['first_any_type'] or '',
                           first_any_type_exact=original['first_any_type']==observed['first_any_type'],
                           original_first_any_s=(original['first_any_time_s'] if original['first_any_time_s'] is not None else ''),
                           repeat_first_any_s=(observed['first_any_time_s'] if observed['first_any_time_s'] is not None else ''),
                           first_any_time_within_1s=any_time_within_1s,
                           nearest_at_first=json.dumps(observed['nearest_at_first']),
                           recorder=observed['recorder'], recorder_bytes=observed['recorder_bytes'],
                           frames=observed['frames'], actor_ticks=observed['actor_ticks'],
                           actor_errors=observed['actor_errors'],
                           run_dir=observed['run_dir'])
                rows.append(row)
    with (OUT/'repeatability.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'ds_exact':sum(x['ds_exact'] for x in rows),
                      'status_exact':sum(x['status_exact'] for x in rows),
                      'first_scored_reproduced':sum(x['first_outcome_reproduced'] for x in rows),
                      'first_any_reproduced':sum(x['first_any_type_exact'] and x['first_any_time_within_1s']
                                                 for x in rows)},indent=2))


if __name__ == '__main__':
    main()
