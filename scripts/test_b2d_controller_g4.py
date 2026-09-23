import copy
import unittest
import tempfile
from pathlib import Path
import hashlib

import b2d_controller_g4 as g4


def fixture(preset, seed, completion=100, status='Completed', infractions=None, cte=None):
    inf = {'vehicle_blocked':[], 'route_dev':[], 'min_speed_infractions':[]}
    inf.update(infractions or {})
    record = dict(status=status, scores=dict(score_route=completion,score_composed=completion), infractions=inf)
    row = dict(route_id='1', attempt=1, selected_attempt=True, completion=completion,
               official_finalized=True, harness_capped=False, official_records=[record],
               vehicle_blocked=len(inf['vehicle_blocked']), route_dev=len(inf['route_dev']))
    metrics={}
    for key, value in [('truth_cross_track_m', cte if cte is not None else (.3 if preset=='pursuit' else .5)),('reference_speed_error_mps',.2)]:
        metrics[key]={scope:dict(routes_with_samples=1,route_mean_rms=value) for scope in ('tracking','before_collision')}
    return dict(preset=preset,seed=seed,source_report='fixture.json',rows=[row],
                summary=dict(attempts=1,retries=0,failed_attempts=[],infractions={},metrics=metrics))


def complete():
    return [fixture(p,s) for s in g4.SEEDS for p in g4.PRESETS]


class G4Tests(unittest.TestCase):
    def test_full_completion_is_not_strict_success(self):
        g=fixture('pursuit',0,infractions={'collisions_vehicle':['collision']})
        r=g4.group_result(g,['1'])
        self.assertEqual(r['driving_completed_fraction'],1)
        self.assertEqual(r['subset_diagnostic_sr'],0)
        self.assertEqual(r['subset_mean_official_driving_score'],100)

    def test_min_speed_is_exempt_but_failed_status_is_not(self):
        g=fixture('carla',0,infractions={'min_speed_infractions':['42%']})
        self.assertEqual(g4.group_result(g,['1'])['subset_diagnostic_sr'],1)
        g['rows'][0]['official_records'][0]['status']='Failed - TickRuntime'
        self.assertEqual(g4.group_result(g,['1'])['subset_diagnostic_sr'],0)
        self.assertTrue(g4.group_result(g,['1'])['ready'])

    def test_three_groups_per_seed_and_both_seeds_required(self):
        r=g4.evaluate(complete()[:-1],['1'])
        self.assertEqual(r['status'],'incomplete_comparison')
        self.assertTrue(r['seeds'][0]['three_groups_final'])
        self.assertIsNone(r['seeds'][1]['conditions'])
        self.assertIsNone(r['aggregate_conditions'])
        self.assertFalse(r['new_default_qualified'])

    def test_missing_or_capped_data_stays_unavailable(self):
        groups=complete();groups[0]['rows'][0]['harness_capped']=True
        self.assertEqual(g4.evaluate(groups,['1'])['status'],'incomplete_comparison')
        g=fixture('carla',0);g['rows'][0]['official_records'][0].pop('infractions')
        self.assertIsNone(g4.group_result(g,['1'])['subset_diagnostic_sr'])

    def test_unknown_blocked_requires_evidence_not_inferred_from_counts(self):
        groups=complete();groups[-1]=fixture('pursuit',1,infractions={'vehicle_blocked':['stuck']})
        self.assertEqual(g4.evaluate(groups,['1'])['status'],'requires_metric_or_failure_attribution')
        external={'pursuit-seed1/1':{'vehicle_blocked':'external'}}
        self.assertEqual(g4.evaluate(groups,['1'],external)['status'],'listed_g4_conditions_satisfied_pending_remaining_acceptance')
        controller={'pursuit-seed1/1':{'vehicle_blocked':'controller'}}
        self.assertEqual(g4.evaluate(groups,['1'],controller)['status'],'g4_condition_failed')

    def test_only_selected_record_and_equal_seed_mean(self):
        groups=complete();extra=copy.deepcopy(groups[0]['rows'][0]);extra['selected_attempt']=False;extra['completion']=0
        groups[0]['rows'].append(extra)
        groups[2]['summary']['metrics']['truth_cross_track_m']['tracking']['route_mean_rms']=.45
        groups[5]['summary']['metrics']['truth_cross_track_m']['tracking']['route_mean_rms']=.3
        r=g4.evaluate(groups,['1'])
        self.assertFalse(r['seeds'][0]['conditions']['lateral_route_mean_rms_reduced_20pct'])
        self.assertTrue(r['aggregate_conditions']['lateral_route_mean_rms_reduced_20pct'])
        self.assertEqual(r['groups']['carla-seed0']['subset_diagnostic_sr'],1)

    def test_actual_archived_configuration_must_match_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'config.json';p.write_text('{"pi_kp":0.5}')
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            group=fixture('carla',0)
            group['manifest']={'preset_controller_configs':{'carla':{'archived_path':str(p),'sha256':h}}}
            self.assertTrue(g4.verify_configs([group],{'carla':str(p)})[0]['valid'])
            group['manifest']['preset_controller_configs']['carla']['sha256']='bad'
            self.assertFalse(g4.verify_configs([group],{'carla':str(p)})[0]['valid'])

    def test_expected_routes_missing_and_duplicate_groups_rejected(self):
        self.assertFalse(g4.group_result(fixture('carla',0),['1','2'])['ready'])
        self.assertRaises(ValueError,g4.evaluate,complete()+[fixture('carla',0)],['1'])


if __name__=='__main__':unittest.main()
