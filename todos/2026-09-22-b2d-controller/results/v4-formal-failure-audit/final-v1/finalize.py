#!/usr/bin/env python3
"""Finalize all-attempt attribution evidence after the formal 60-case campaign."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edition', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.edition.resolve()
    summary = json.loads((source / 'summary.json').read_text())
    if not summary['campaign_ended']:
        parser.error('campaign has not ended; refuse a final table')
    contexts = json.loads((source / 'contexts.json').read_text())
    cases = list(csv.DictReader((source / 'all-attempt-cases.csv').open()))
    unique = {(r['group'], r['route_id']) for r in cases}
    if len(unique) != 60:
        parser.error('expected 60 distinct requested group/route cases; got %d; reconcile missing attempts first' % len(unique))
    if args.out.exists():
        parser.error('output exists; use a new version')
    args.out.mkdir(parents=True)
    keyed = {(c['group'], c['route_id'], c['attempt']): c for c in contexts}
    for row in cases:
        context = keyed[row['group'], row['route_id'], int(row['attempt'])]
        events = context['critical_events']
        collisions = [e for e in events if e['event'].get('type', '').startswith('COLLISION')]
        frames = [e['event']['frame'] for e in collisions]
        earliest = min(frames) if frames else None
        invalid_motion = [r for r in context['nontracking_samples'] if r.get('reason') == 'invalid_motion']
        stalled = float(row.get('low_speed_max_s') or 0) >= 5.
        failed = row['driving_completed'] != 'True'
        row.update(tickruntime='tickruntime' in (row['official_status'] or '').lower(),
                   physical_low_speed_stall=stalled,
                   stall_or_unfinished_cause='unknown' if stalled or failed else 'not_applicable',
                   collision_association=('contact_before_or_during_low_speed' if any(e['event']['frame'] <= run['last']['frame'] for e in collisions for run in context.get('low_speed_runs', [])) else 'low_speed_precedes_recorded_contact') if collisions and stalled else 'contact_recorded' if collisions else 'no_contact_recorded',
                   collision_counterpart_types=';'.join(re.search(r'type=([^ ]+)', e['event'].get('message', '')).group(1)
                                                       if re.search(r'type=([^ ]+)', e['event'].get('message', '')) else 'unknown' for e in collisions),
                   collision_frames=';'.join(map(str, frames)),
                   collision_event_indices=';'.join(str(e['event_index']) for e in collisions),
                   critical_event_types=';'.join(e['event'].get('type', '') for e in events),
                   critical_event_frames=';'.join(str(e['event'].get('frame')) for e in events),
                   criterion_event_path=str(Path(row['attempt_path']) / 'criterion_events.json'),
                   control_path=str(Path(row['attempt_path']) / 'control.jsonl'),
                   controller_invalid_motion_precontact=sum(r['frame'] < earliest for r in invalid_motion) if earliest is not None else len(invalid_motion),
                   pose_fault_precontact=sum(r['frame'] < earliest for r in context['invalid_pose_samples']) if earliest is not None else len(context['invalid_pose_samples']),
                   low_speed_start_frames=';'.join(str(r['first']['frame']) for r in context.get('low_speed_runs', [])),
                   low_speed_end_frames=';'.join(str(r['last']['frame']) for r in context.get('low_speed_runs', [])),
                   attribution_limitation='ego-only telemetry; counterpart motion/contact impulse unavailable; association is not fault assignment')
    with (args.out / 'all-attempt-attribution.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)
    selected = {}
    for row in sorted(cases, key=lambda r: (r['group'], r['route_id'], int(r['attempt']))):
        if row['harness_status'] == 'finished':
            selected.setdefault((row['group'], row['route_id']), row)
    # Keep every requested key visible even if no harness-finished attempt exists.
    for row in cases:
        if (row['group'], row['route_id']) not in selected:
            selected[row['group'], row['route_id']] = dict(row, selection_state='no_harness_finished_attempt')
    selected_rows = [dict(row, selection_state=row.get('selection_state', 'first_harness_finished')) for _, row in sorted(selected.items())]
    with (args.out / 'selected-60-attribution.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected_rows[0])); writer.writeheader(); writer.writerows(selected_rows)
    shutil.copyfile(str(Path(__file__)), str(args.out / 'finalize.py'))
    manifest = dict(source_edition=str(source), source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
                    all_attempts=len(cases), selected_route_cases=len(selected_rows),
                    labels='Attribution remains unknown; association labels preserve event order but do not assign fault')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(all_attempts=len(cases), selected_route_cases=len(selected_rows), output=str(args.out))))


if __name__ == '__main__':
    main()
