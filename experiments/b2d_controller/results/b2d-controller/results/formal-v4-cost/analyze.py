"""Read completed formal campaign costs; preserve every attempt and input hash."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    root = args.campaign.resolve()
    manifest = read(root / 'manifest.json')
    groups = read(root / 'groups.json')
    raw = (root / 'events.jsonl').read_bytes()
    sources[str(root / 'events.jsonl')] = hashlib.sha256(raw).hexdigest()
    events = [json.loads(line) for line in raw.splitlines()]
    ends = [row for row in events if row['kind'] == 'end']
    if len(groups) != 6 or not ends or ends[-1].get('status') != 'completed':
        raise ValueError('Require all six groups and campaign completion event')
    rows, summaries = [], []
    for group in groups:
        folder = Path(group['out'])
        selected, group_rows, latencies = set(), [], []
        paths = sorted(folder.glob('attempts/*/*/attempt.json'),
                       key=lambda p: (p.parent.parent.name, int(p.parent.name)))
        for path in paths:
            attempt = read(path)
            route_path = path.parent / 'route_result.json'
            route = read(route_path) if route_path.exists() else {}
            profile = route.get('profile') or {}
            rid = path.parent.parent.name
            pick = attempt.get('status') == 'finished' and rid not in selected
            completion = None
            if pick:
                selected.add(rid)
                result = read(path.parent / 'results.json')
                records = result['_checkpoint']['records']
                if len(records) != 1:
                    raise ValueError('Expected one official route record')
                completion = records[0]['scores']['score_route']
                control = path.parent / 'control.jsonl'
                data = control.read_bytes()
                sources[str(control)] = hashlib.sha256(data).hexdigest()
                latencies.extend(json.loads(line)['controller_step_ms'] for line in data.splitlines())
            ticks_used = profile.get('ticks_used', 0)
            loop = ticks_used * profile.get('total_ms_mean', 0.) / 1000.
            row = dict(group=folder.name, route_id=rid, attempt=int(path.parent.name),
                       status=attempt['status'], selected=pick, completion=completion,
                       wall_s=attempt['wall_s'], ticks=profile.get('ticks'),
                       profiled_ticks=ticks_used, profiled_loop_s=loop,
                       server_age_routes=attempt.get('server_age_routes'), source=str(path))
            rows.append(row)
            group_rows.append(row)
        if len(selected) != 10:
            raise ValueError('Missing finalized route in ' + folder.name)
        chosen = [r for r in group_rows if r['selected']]
        used = sum(r['profiled_ticks'] for r in chosen)
        loop = sum(r['profiled_loop_s'] for r in chosen)
        final_wall = sum(r['wall_s'] for r in chosen)
        ticks = lambda subset: ([min(r['ticks'] for r in subset),
                                float(np.median([r['ticks'] for r in subset])),
                                max(r['ticks'] for r in subset)] if subset else None)
        summaries.append(dict(group=folder.name, completed=sum(r['completion'] >= 100 for r in chosen),
            all_ticks_min_median_max=ticks(chosen),
            completed_ticks_min_median_max=ticks([r for r in chosen if r['completion'] >= 100]),
            failed_ticks_min_median_max=ticks([r for r in chosen if r['completion'] < 100]),
            selected_wall_s=final_wall, extra_attempt_wall_s=sum(r['wall_s'] for r in group_rows if not r['selected']),
            group_wall_s=group['wall_s'], profiled_ticks=used, profiled_loop_s=loop,
            weighted_profiled_ms_per_tick=loop * 1000. / used,
            selected_unseparated_remainder_s=final_wall-loop,
            controller_step_p99_ms=float(np.percentile(latencies, 99))))
    interval = ends[-1]['t'] - manifest['started']
    attempts_wall = sum(row['wall_s'] for row in rows)
    result = dict(groups=summaries, attempts=len(rows), selected_routes=sum(r['selected'] for r in rows),
        manifest_start_to_end_event_s=interval, all_attempt_wall_s=attempts_wall,
        campaign_non_attempt_remainder_s=interval-attempts_wall,
        definitions=['Profiled loop uses ticks_used times rounded total_ms_mean; warmup excluded.',
            'Selected remainder mixes setup, cleanup, unprofiled warmup and untimed work.',
            'Campaign remainder includes initial readiness, restarts, reports and slope check.',
            'Final server.stop is after the end event and excluded; no neural policy cost measured.'])
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.out / 'attempts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir()}
    (args.out / 'manifest.json').write_text(json.dumps(dict(inputs=sources, outputs=outputs), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
