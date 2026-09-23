#!/usr/bin/env python3
"""Audit frozen three-preset G4 groups; no winner from incomplete comparisons.

Strict SR follows pinned B2D record semantics but uses the requested subset size,
not the official full220 script's hardcoded denominator. Inputs are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import b2d_controller_compare as compare

PRESETS = ('carla', 'tcp', 'pursuit')
SEEDS = (0, 1)
REFERENCE = 'carla'
CANDIDATE = 'pursuit'


def finite(value):
    return compare.report.finite(value)


def official_record(row):
    records = row.get('official_records')
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        return None
    return records[0]


def strict_success(record):
    """None is unavailable, False is a genuine unsuccessful official record."""
    if not record or not isinstance(record.get('infractions'), dict) or not record.get('status'):
        return None
    if any(not isinstance(events, list) for events in record['infractions'].values()):
        return None
    return record['status'] in ('Completed', 'Perfect') and all(
        not events for kind, events in record['infractions'].items() if kind != 'min_speed_infractions')


def selected(group):
    return [r for r in group['rows'] if r.get('selected_attempt')]


def group_result(group, route_ids):
    rows = selected(group)
    records = [official_record(r) for r in rows]
    ready = (bool(group.get('source_report')) and len(rows) == len(route_ids)
             and {r['route_id'] for r in rows} == set(route_ids)
             and len({r['route_id'] for r in rows}) == len(rows)
             and all(r.get('official_finalized') and not r.get('harness_capped')
                     and finite(r.get('completion')) and record is not None
                     for r, record in zip(rows, records)))
    successes = [strict_success(r) for r in records]
    n = len(route_ids)
    driving = sum(r.get('completion', -1) >= 100 for r in rows if finite(r.get('completion')))
    scores = [(r.get('scores') or {}).get('score_composed') for r in records if r]
    return dict(ready=ready, requested_routes=n, official_selected_records=len([r for r in records if r]),
                subset_diagnostic_success_count=sum(x is True for x in successes),
                subset_diagnostic_sr=sum(x is True for x in successes)/n if ready and all(x is not None for x in successes) else None,
                driving_completed_count=driving,
                driving_completed_fraction=driving/n if ready else None,
                completion_mean=compare.report.mean(r['completion'] for r in rows) if ready else None,
                subset_mean_official_driving_score=compare.report.mean(scores) if ready and len(scores)==n and all(finite(x) for x in scores) else None,
                source_report=group.get('source_report'), source_report_sha256=group.get('source_report_sha256'),
                source_commit=group.get('manifest', {}).get('git_commit'),
                attempts=group['summary']['attempts'], retries=group['summary']['retries'],
                failed_attempts=group['summary']['failed_attempts'],
                infractions=group['summary']['infractions'], metrics=group['summary']['metrics'])


def controlled_events(group, kind, adjudications):
    count = 0
    unknown = []
    for row in selected(group):
        events = row.get(kind)
        if not finite(events):
            unknown.append(row['route_id'])
        elif events:
            key = '%s-seed%d/%s' % (group['preset'], group['seed'], row['route_id'])
            label = adjudications.get(key, {}).get(kind, 'unknown')
            if label == 'controller':
                count += events
            elif label != 'external':
                unknown.append(row['route_id'])
    return dict(controller_count=count if not unknown else None, unresolved_routes=unknown,
                note='Manual evidence labels required for nonzero events; no automatic causal inference.')


def metric_value(group, key, scope='tracking'):
    value = group['summary']['metrics'][key][scope]
    n = len(selected(group))
    return value.get('route_mean_rms') if value.get('routes_with_samples') == n else None


def evaluate(groups, route_ids, adjudications=None):
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError('Expected unique nonempty route IDs')
    adjudications = adjudications or {}
    mapping = {(g['preset'], int(g['seed'])):g for g in groups}
    if len(mapping) != len(groups):
        raise ValueError('Duplicate preset/seed groups')
    summaries = {}
    for (preset, seed), group in mapping.items():
        if preset in PRESETS and seed in SEEDS:
            summaries['%s-seed%d' % (preset, seed)] = group_result(group, route_ids)
    seeds = []
    for seed in SEEDS:
        names = ['%s-seed%d' % (preset, seed) for preset in PRESETS]
        ready = all(summaries.get(name, {}).get('ready', False) for name in names)
        result = dict(seed=seed, three_groups_final=ready, conditions=None,
                      missing_or_incomplete_groups=[name for name in names if not summaries.get(name, {}).get('ready', False)])
        if ready:
            cand, ref = mapping[(CANDIDATE, seed)], mapping[(REFERENCE, seed)]
            cm, rm = summaries[names[2]], summaries[names[0]]
            lateral = [metric_value(g, 'truth_cross_track_m') for g in (cand, ref)]
            speed = [metric_value(g, 'reference_speed_error_mps') for g in (cand, ref)]
            conditions = dict(completion_not_lower=cm['completion_mean'] >= rm['completion_mean'],
                lateral_route_mean_rms_reduced_20pct=(lateral[0] <= .8*lateral[1]) if all(finite(x) for x in lateral) else None,
                trajectory_reference_speed_rms_increase_le_01=(speed[0] <= speed[1]+.1) if all(finite(x) for x in speed) else None)
            event_comparison = {}
            for kind in ('vehicle_blocked', 'route_dev'):
                c, r = [controlled_events(g, kind, adjudications) for g in (cand, ref)]
                event_comparison[kind] = dict(candidate=c, reference=r)
                conditions['controller_'+kind+'_not_increased'] = (c['controller_count'] <= r['controller_count']) if c['controller_count'] is not None and r['controller_count'] is not None else None
            result.update(conditions=conditions, event_attribution=event_comparison,
                          lateral_route_mean_rms_candidate_reference=lateral,
                          trajectory_reference_speed_route_mean_rms_candidate_reference=speed,
                          before_collision={key:[metric_value(g,key,'before_collision') for g in (cand,ref)]
                                            for key in ('truth_cross_track_m','reference_speed_error_mps')})
        seeds.append(result)
    complete = all(s['three_groups_final'] for s in seeds)
    aggregate_conditions = None
    aggregate_metrics = None
    if complete:
        lateral = [[s['lateral_route_mean_rms_candidate_reference'][i] for s in seeds] for i in (0,1)]
        speed = [[s['trajectory_reference_speed_route_mean_rms_candidate_reference'][i] for s in seeds] for i in (0,1)]
        avg = lambda values: compare.report.mean(values) if all(finite(x) for x in values) else None
        lat, spd = [avg(x) for x in lateral], [avg(x) for x in speed]
        aggregate_metrics = dict(lateral_route_mean_rms_candidate_reference=lat,
                                 trajectory_reference_speed_route_mean_rms_candidate_reference=spd)
        aggregate_conditions = dict(completion_not_lower_each_seed=all(s['conditions']['completion_not_lower'] for s in seeds),
            lateral_route_mean_rms_reduced_20pct=(lat[0] <= .8*lat[1]) if all(finite(x) for x in lat) else None,
            trajectory_reference_speed_rms_increase_le_01=(spd[0] <= spd[1]+.1) if all(finite(x) for x in spd) else None)
        for kind in ('vehicle_blocked','route_dev'):
            cs=[s['event_attribution'][kind]['candidate']['controller_count'] for s in seeds]
            rs=[s['event_attribution'][kind]['reference']['controller_count'] for s in seeds]
            aggregate_conditions['controller_'+kind+'_not_increased']=(sum(cs)<=sum(rs)) if all(x is not None for x in cs+rs) else None
    checks = list((aggregate_conditions or {}).values())
    status = ('incomplete_comparison' if not complete else
              'g4_condition_failed' if any(v is False for v in checks) else
              'requires_metric_or_failure_attribution' if any(v is None for v in checks) else
              'listed_g4_conditions_satisfied_pending_remaining_acceptance')
    return dict(schema_version=1, status=status, all_six_groups_final=complete,
                reference='carla: shared PI Kp=.5 Ki=.25, additive; fixed on G2, not selected on Dev10',
                candidate='pursuit: PI Kp=.5 Ki=.25, max lookahead',
                other_reference='tcp: vendor longitudinal', groups=summaries, seeds=seeds,
                aggregate_conditions=aggregate_conditions, aggregate_metrics=aggregate_metrics,
                aggregation='Completion compared separately in both seeds; other primary numeric conditions use equally weighted route/seed observations; per-seed conditions remain diagnostic.',
                new_default_qualified=False,
                limitations=['Subset diagnostic SR uses official record status/infraction semantics with requested-route denominator; not official full220 SR.',
                    'Driving completed (completion>=100), strict SR, G2 gates and harness finished are distinct.',
                    'Official TickRuntime failures remain denominator; user harness caps prevent readiness.',
                    'First harness-finished attempt selection inherited from report; all retries and failures retained, never best attempt.',
                    'Speed comparison is trajectory-derivative reference tracking, not an independent fixed-cruise or official comfort metric.',
                    'Full-route CTE is used for the numerical 20% condition; before-collision metrics shown separately and never substituted.',
                    'All-reference-G2-failed exemption does not apply: frozen CARLA+PI reference passed G2.',
                    'Passing listed G4 conditions alone does not replace G1-G3, repeat variability review, or planned confirmation.'])


def verify_configs(groups, expected_paths):
    records = []
    for group in groups:
        preset = group['preset']
        expected = compare.sha256(expected_paths.get(preset, 'missing'))
        identity = (group.get('manifest', {}).get('preset_controller_configs') or {}).get(preset) or {}
        archived = identity.get('archived_path')
        actual = compare.sha256(archived) if archived else None
        valid = expected is not None and actual == expected and identity.get('sha256') == expected
        records.append(dict(preset=preset, seed=group['seed'], valid=valid,
                            expected_sha256=expected, actual_sha256=actual, archived_path=archived))
    return records


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--preset-configs',required=True,help='Frozen preset-to-config JSON mapping; archived campaign bytes must match')
    p.add_argument('--routes',required=True,help='Expected frozen Dev10 XML, even when groups missing')
    p.add_argument('--adjudications',help='Evidence-reviewed controller/external/unknown labels by preset-seedN/route and event kind')
    p.add_argument('--out',required=True,help='New output directory; existing directory rejected')
    args=p.parse_args()
    ids=[r.get('id') for r in ET.parse(args.routes).getroot().findall('route')]
    groups=compare.load_campaign(args.campaign)
    configs=json.loads(Path(args.preset_configs).read_text())
    config_checks=verify_configs(groups,configs)
    result=evaluate(groups,ids,json.loads(Path(args.adjudications).read_text()) if args.adjudications else {})
    result['frozen_config_verification']=config_checks
    if any(not check['valid'] for check in config_checks):
        result['status']='frozen_configuration_mismatch_or_unavailable'
    result['inputs']={str(Path(x).resolve()):compare.sha256(x) for x in [args.routes,args.preset_configs,str(Path(__file__).resolve()),str(Path(compare.__file__).resolve()),str(Path(compare.report.__file__).resolve())]+list(configs.values())+([args.adjudications] if args.adjudications else [])}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'g4.json').write_text(json.dumps(result,indent=2))
    lines=['# Frozen G4 audit','',result['status'],'','| Group | Final | Completion mean | Driving completed | Subset diagnostic SR |','|---|---|---:|---:|---:|']
    for name,g in result['groups'].items():
        lines.append('| %s | %s | %s | %s/%s | %s |'%(name,g['ready'],g['completion_mean'],g['driving_completed_count'],g['requested_routes'],g['subset_diagnostic_sr']))
    lines+=['','Full seed conditions and all failed attempts are in g4.json. No new default is declared.','']+['- '+x for x in result['limitations']]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(result['status'])


if __name__=='__main__':
    main()
