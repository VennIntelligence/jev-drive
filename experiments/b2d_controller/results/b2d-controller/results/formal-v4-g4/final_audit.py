"""Read-only final wrapper for original-plan checks beyond the frozen G4 helper."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent/'v4-formal-failure-audit/final-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    with path.open('x') as f:json.dump(value,f,indent=2)
def write_csv(path, rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data=json.loads((OUT/'g4.json').read_text())
    assert data['all_six_groups_final'] and len(data['groups'])==6
    assert all(x['valid'] for x in data['frozen_config_verification'])
    rows=list(csv.DictReader((AUDIT/'selected-60-attribution.csv').open()))
    attempts=list(csv.DictReader((AUDIT/'all-attempt-attribution.csv').open()))
    assert len(rows)==60 and len({(r['group'],r['route_id']) for r in rows})==60
    bygroup={name:[r for r in rows if r['group']==name] for name in data['groups']}
    table=[]
    for seed in (0,1):
        for preset in ('carla','tcp','pursuit'):
            name='%s-seed%d'%(preset,seed);g=data['groups'][name];rs=bygroup[name]
            assert len(rs)==10
            assert sum(r['driving_completed']=='True' for r in rs)==g['driving_completed_count']
            assert sum(r['subset_diagnostic_sr']=='True' for r in rs)==g['subset_diagnostic_success_count']
            table.append(dict(group=name,requested=10,attempts=g['attempts'],driving_completed=g['driving_completed_count'],
                incomplete_routes=';'.join(r['route_id'] for r in rs if r['driving_completed']!='True'),
                completion_mean_pct=g['completion_mean'],subset_diagnostic_sr=g['subset_diagnostic_sr'],
                subset_mean_official_ds=g['subset_mean_official_driving_score'],
                full_route_mean_truth_cte_rms_m=g['metrics']['truth_cross_track_m']['tracking']['route_mean_rms'],
                precollision_route_mean_truth_cte_rms_m=g['metrics']['truth_cross_track_m']['before_collision']['route_mean_rms'],
                trajectory_reference_speed_route_mean_rms_mps=g['metrics']['reference_speed_error_mps']['tracking']['route_mean_rms'],
                official_tickruntime_routes=sum(r['tickruntime']=='True' for r in rs),
                low_speed_5s_route_count=sum(r['physical_low_speed_stall']=='True' for r in rs),
                unknown_stall_or_unfinished_routes=sum(r['stall_or_unfinished_cause']=='unknown' for r in rs)))
    additional=[]
    for seed in (0,1):
        cg=data['groups']['pursuit-seed%d'%seed]
        refs={p:data['groups']['%s-seed%d'%(p,seed)] for p in ('carla','tcp')}
        strongest=max(g['completion_mean'] for g in refs.values())
        best=[p for p,g in refs.items() if g['completion_mean']==strongest]
        cfail=10-cg['driving_completed_count']
        additional.append(dict(seed=seed,candidate_mean_completion_pct=cg['completion_mean'],
            strongest_reference_mean_completion_pct=strongest,strongest_completion_reference_presets=best,
            completion_not_below_either_reference=cg['completion_mean']>=strongest,
            candidate_minus_strongest_completion_percentage_points=cg['completion_mean']-strongest,
            candidate_incomplete_count=cfail,
            incomplete_by_reference={p:10-g['driving_completed_count'] for p,g in refs.items()},
            failure_count_not_increased_vs_each_reference={p:cfail<=10-g['driving_completed_count'] for p,g in refs.items()},
            failure_count_not_increased_vs_strongest_completion_reference=all(cfail<=10-refs[p]['driving_completed_count'] for p in best)))
    unknown=[dict(group=r['group'],route_id=r['route_id'],tickruntime=r['tickruntime']=='True',
                  driving_completed=r['driving_completed']=='True',low_speed_max_s=float(r['low_speed_max_s'] or 0),
                  cause=r['stall_or_unfinished_cause'],association=r['collision_association'],
                  source_control=r['control_path'],source_events=r['criterion_event_path'])
             for r in rows if r['stall_or_unfinished_cause']=='unknown']
    necessary=dict(completion_not_below_strongest_reference_each_seed=all(r['completion_not_below_either_reference'] for r in additional),
                   failure_count_not_increased_each_seed=all(r['failure_count_not_increased_vs_strongest_completion_reference'] for r in additional),
                   lateral_rms_reduced_20pct=data['aggregate_conditions']['lateral_route_mean_rms_reduced_20pct'],
                   trajectory_reference_speed_increase_le_01=data['aggregate_conditions']['trajectory_reference_speed_rms_increase_le_01'],
                   actual_controller_caused_stall_or_deviation_not_increased=None if unknown else True)
    status='original_g4_necessary_conditions_failed' if any(x is False for x in necessary.values()) else 'requires_causal_review' if any(x is None for x in necessary.values()) else 'necessary_conditions_met_pending_other_acceptance'
    result=dict(status=status,new_default_qualified=False,all_required_routes_present=True,requested_routes=60,
        all_attempts=len(attempts),infra_attempts=[dict(group=r['group'],route_id=r['route_id'],attempt=r['attempt'],harness_status=r['harness_status']) for r in attempts if r['harness_status']!='finished'],
        driving_completed=sum(r['driving_completed']=='True' for r in rows),strict_subset_success_count=sum(r['subset_diagnostic_sr']=='True' for r in rows),
        official_tickruntime_routes=sum(r['tickruntime']=='True' for r in rows),
        additional_per_seed_checks=additional,necessary_conditions=necessary,
        known_low_speed_or_unfinished_unknown_causes=unknown,
        helper_official_event_count_conditions=data['aggregate_conditions'],
        warning='Helper official blocked/deviation counts are zero. This does not establish no real stalls or no controller-caused failures; final manual causal condition remains unknown.',
        sources={str(p):sha(p) for p in [OUT/'g4.json',AUDIT/'selected-60-attribution.csv',AUDIT/'all-attempt-attribution.csv',AUDIT/'manifest.json',Path(__file__)]})
    write_json(OUT/'manual-qualification.json',result)
    write_csv(OUT/'six-group-table.csv',table)
    for name in ('selected-60-attribution.csv','all-attempt-attribution.csv'):
        with (OUT/name).open('xb') as f:f.write((AUDIT/name).read_bytes())
    cte=data['aggregate_metrics']['lateral_route_mean_rms_candidate_reference']
    speed=data['aggregate_metrics']['trajectory_reference_speed_route_mean_rms_candidate_reference']
    lines=['# Formal v4: final G4 audit','',
        '**Original G4 necessary conditions are not met. No replacement default is qualified by this comparison.**',
        '', 'All 60 requested routes have selected final official records; all 63 attempts are retained. There are 49 driving-completed routes, 10 successes under the strict subset diagnostic SR rule, and 11 official TickRuntime failures.',
        '', '| Group | Driving completed | Mean completion % | Strict subset SR | Official DS subset mean | Full CTE RMS m | Before-collision CTE RMS m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| %s | %d/10 | %.3f | %.0f%% | %.6f | %.6f | %.6f |'%(r['group'],r['driving_completed'],r['completion_mean_pct'],100*r['subset_diagnostic_sr'],r['subset_mean_official_ds'],r['full_route_mean_truth_cte_rms_m'],r['precollision_route_mean_truth_cte_rms_m']))
    lines+=['','## Original necessary conditions','',
        '- Seed0 candidate completion92.627% is below strongest reference CARLA94.726% by2.099percentage points. Incomplete routes increase from1 to2. Seed1 candidate92.627% exceeds CARLA90.790%; both have2 incomplete routes. TCP85.186% is also checked in each seed.',
        '- Equally weighted full-route CTE RMS is %.9fm for candidate versus %.9fm for frozen CARLA reference: %.3f%% higher, failing the required20%% reduction.'%(cte[0],cte[1],100*(cte[0]/cte[1]-1)),
        '- Trajectory-reference speed RMS difference is +%.9fm/s, within+.1. This is not independent fixed-cruise error or official comfort.'%(speed[0]-speed[1]),
        '- Official blocked/deviation counts are zero, but28 routes have at least one recorded low-speed interval of5s or longer, including11 unfinished TickRuntime routes. Causal dominance remains unknown; no zero-event shortcut establishes no controller-caused stalls.',
        '', '## Interpretation and provenance','',
        'The frozen lateral reference uses CARLA lateral PID with the same PI(.5,.25), additive lookahead. TCP uses vendor longitudinal control; the candidate is pursuit max with PI(.5,.25). All six archived configuration hashes match the declared mapping. Runtime source commit is recorded in g4.json.',
        '', 'Driving completed means completion>=100. Strict subset SR requires official status Completed/Perfect and no nonempty infraction list except min_speed_infractions; denominator10 per group, not the full220 script denominator. Reported DS values are means of official serialized score_composed. These route-oracle results do not establish planner/full220 or comfort improvements.',
        '', 'Full-route and before-collision metrics remain separate. Long stalls can make full-route CTE small. Every unfinished route has recorded contact before prolonged low speed, but ego-only logs do not identify collision responsibility or isolate controller, obstacle and contact dynamics. Same TrafficManager seed does not ensure identical counterpart realization; near-identical outputs across seeds are not independent random replications.',
        '', 'Three rc139 infrastructure attempts are retained separately from60 route results. Pose faults and invalid actuation are distinct from logged signed-reverse invalid_motion after contact; see the failure audit for exact event/frame context.',
        '', '[Exact six-group CSV](six-group-table.csv), [manual original-plan checks](manual-qualification.json), [frozen helper output](g4.json), [selected60 evidence](selected-60-attribution.csv), and [all63 attempts](all-attempt-attribution.csv). Existing helper README/output were preserved; this final wrapper adds strongest-reference completion, failure-count and real-stall scrutiny.',
        '', 'The candidate slope-hold check is separately archived by the campaign owner. It cannot overturn the failed Dev10 conditions.']
    with (OUT/'FINAL-AUDIT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=status,necessary_conditions=necessary,unknown_low_speed_or_unfinished=len(unknown),rows=len(rows),attempts=len(attempts))))


if __name__=='__main__':main()
