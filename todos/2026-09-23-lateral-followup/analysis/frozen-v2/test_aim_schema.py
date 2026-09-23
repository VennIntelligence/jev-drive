import unittest
from analyze_aim import aim_mode_evidence,plant_features,required_conditions,invalid_control


def row(mode='hermite',used='hermite',fallback=None,aim=True):
    return dict(aim_interpolation=mode,aim_interpolation_used=used,aim_interpolation_fallback=fallback,aim_xy=[3.,.2] if aim else None,reason='tracking' if aim else 'stationary_trajectory')

class SchemaTests(unittest.TestCase):
    def test_valid_modes_and_named_fallback(self):
        for expected,r in [('linear',row('linear','linear')),('hermite',row()),('hermite',row(used='linear',fallback='local_tangent_reversal'))]:self.assertFalse(any(aim_mode_evidence([r],expected).values()))
    def test_null_or_wrong_requested_is_failure(self):
        for mode in (None,'linear','unknown'):
            evidence=aim_mode_evidence([row(mode=mode)],'hermite');self.assertEqual(evidence['aim_mode_mismatch_ticks'],1)
            case=dict(route='26966',variant='candidate-hermite',**evidence)
            condition=next(c for c in required_conditions([], [case], {}, [])['required_conditions'] if c['condition'].endswith('/aim_mode_evidence'))
            self.assertEqual(condition['status'],'fail')
    def test_missing_field_is_insufficient(self):
        r=row();r.pop('aim_interpolation_fallback');e=aim_mode_evidence([r],'hermite');self.assertEqual(e['aim_mode_missing_ticks'],1)
    def test_evaluated_aim_requires_used_and_fallback_consistency(self):
        for expected,r in [('hermite',row(used=None)),('hermite',row(used='linear')),('hermite',row(fallback='unexpected')),('linear',row('linear','hermite'))]:self.assertEqual(aim_mode_evidence([r],expected)['aim_used_or_fallback_error_ticks'],1)
    def test_safe_without_aim_allows_null_used(self):self.assertFalse(any(aim_mode_evidence([row(used=None,aim=False)],'hermite').values()))
    def test_actual_time_fault_reasons_and_legal_stops(self):
        for reason in ('motion_gap','time_regression','future_trajectory','trajectory_outside_history','duplicate_tick'):
            self.assertTrue(invalid_control(dict(steer=0.,throttle=0.,brake=1.,reason=reason),{}))
        for reason in ('stationary_trajectory','trajectory_behind'):
            self.assertFalse(invalid_control(dict(steer=0.,throttle=0.,brake=1.,reason=reason),{}))
    def test_unknown_reason_evidence_is_insufficient(self):
        c=dict(route='26966',variant='candidate-hermite',unknown_reason_count=1)
        item=next(c for c in required_conditions([], [c], {}, [])['required_conditions'] if c['condition'].endswith('/reason_evidence'))
        self.assertEqual(item['status'],'insufficient')
    def test_world_derivative_before_body_projection(self):
        rows=[dict(frame=1,sim_time=.05,acceleration_mps2=[1.,0.,0.],right_vector=[0.,1.,0.],forward_vector=[1.,0.,0.]),dict(frame=2,sim_time=.1,acceleration_mps2=[1.,0.,0.],right_vector=[-1.,0.,0.],forward_vector=[0.,1.,0.])]
        self.assertEqual(plant_features(rows)[2]['lateral_jerk'],0.)

if __name__=='__main__':unittest.main()
