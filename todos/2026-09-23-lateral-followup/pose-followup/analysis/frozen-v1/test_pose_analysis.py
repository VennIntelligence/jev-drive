import copy
import math
import unittest
import numpy as np
from analyze_pose import required_conditions, plant_features, invalid_control, aim_mode_evidence, clocks_agree, common_config
from pose_contract import FIXED_K, KEY, FIELDS, audit_pose, pose_features, replay_sensors


def example(k=FIXED_K):
    rows=[];motion=[]
    prior=np.array([0.,0.]);yaw=.2
    for i,(t,speed,gyro) in enumerate(((.05,6.,.1),(.15,8.,.3))):
        frame=20+i
        status=dict(zip(FIELDS,(k,False,'no_interval',None,None,None,None,None)))
        raw=np.array([.6,.02]) if i else prior.copy()
        if i:
            dt=.1;v=7.;omega=.2;mid=yaw+.5*.3*dt
            delta=-k*v*v*omega*dt*np.array([-math.sin(mid),math.cos(mid)])
            prior=.95*(prior+v*dt*np.array([math.cos(mid),math.sin(mid)])+delta)+.05*raw
            status.update(dict(zip(FIELDS,(k,True,'applied' if k else 'disabled',dt,v,omega,mid,delta.tolist()))))
        rows.append(dict(frame=frame,sim_time=t,speed_mps=speed,yaw_rate_rps=-gyro,
            pose_status=status,reason='tracking',pose_xy=prior.tolist(),pose_yaw=yaw,raw_pose_xy=raw.tolist()))
        motion.append(dict(frame=frame,sim_time=t,sensors={
            'SPEED':dict(frame=frame,data={'speed':speed}),
            'IMU':dict(frame=frame,data=[0.,0.,0.,0.,0.,gyro,yaw+math.pi/2]),
            'GPS':dict(frame=frame,data=[0.,0.,0.])}))
    return rows,motion,dict(aim_interpolation='linear',**{KEY:k})


class PoseContractTests(unittest.TestCase):
    def test_distinct_time_origins_require_constant_offset(self):
        self.assertTrue(clocks_agree([dict(sim_time=.05),dict(sim_time=.1)],[dict(sim_time=8.05),dict(sim_time=8.1)]))
        self.assertFalse(clocks_agree([dict(sim_time=.05),dict(sim_time=.1)],[dict(sim_time=8.05),dict(sim_time=8.2)]))
    def test_common_configuration_cannot_tune_another_parameter(self):
        config=dict(lookahead='max',max_lookahead_time_s=.5,longitudinal_mode='pi',pi_kp=.5,pi_ki=.25,aim_interpolation='linear')
        self.assertTrue(common_config(config));config['max_lookahead_time_s']=.375
        self.assertFalse(common_config(config))
    def test_fixed_and_disabled_mean_motion_formula(self):
        for k,variant in ((0.,'baseline-zero'),(FIXED_K,'candidate-fixed-k')):
            rows,motion,config=example(k)
            self.assertTrue(audit_pose(rows,motion,config,variant)['complete'])
    def test_each_required_field_missing_fails(self):
        for field in FIELDS:
            rows,motion,config=example();rows[1]['pose_status'].pop(field)
            self.assertFalse(audit_pose(rows,motion,config,'candidate-fixed-k')['complete'],field)
    def test_wrong_coefficient_sign_and_mean_convention_fail(self):
        for field,value in (('lateral_coefficient_s2_per_m',0.),('lateral_mean_world_gyro_rps',.3),('lateral_dt_s',.05)):
            rows,motion,config=example();rows[1]['pose_status'][field]=value
            self.assertFalse(audit_pose(rows,motion,config,'candidate-fixed-k')['complete'])
    def test_delta_sign_and_post_gnss_injection_detected(self):
        rows,motion,config=example();delta=np.array(rows[1]['pose_status']['lateral_delta_xy_m'])
        rows[1]['pose_xy']=(np.array(rows[1]['pose_xy'])+.05*delta).tolist()
        self.assertTrue(any('fusion' in e['reason'] for e in audit_pose(rows,motion,config,'candidate-fixed-k')['errors']))
    def test_deleted_motion_frame_and_stale_sensor_timestamp_fail(self):
        rows,motion,config=example();self.assertFalse(audit_pose(rows,motion[1:],config,'candidate-fixed-k')['complete'])
        motion[1]['sensors']['IMU']['frame']-=1
        self.assertFalse(audit_pose(rows,motion,config,'candidate-fixed-k')['complete'])
    def test_missing_mode_or_config_coefficient_fail(self):
        for field in ('aim_interpolation',KEY):
            rows,motion,config=example();config.pop(field)
            self.assertFalse(audit_pose(rows,motion,config,'candidate-fixed-k')['complete'])
    def test_first_tick_cannot_claim_interval(self):
        rows,motion,config=example();rows[0]['pose_status']['lateral_valid']=True
        self.assertFalse(audit_pose(rows,motion,config,'candidate-fixed-k')['complete'])
    def test_truth_is_not_audit_input(self):
        rows,motion,config=example();before=audit_pose(rows,motion,config,'candidate-fixed-k')
        for row in rows:row.update(truth_xy=[1e6,-1e6],truth_yaw=900.)
        self.assertEqual(before,audit_pose(rows,motion,config,'candidate-fixed-k'))
    def test_signed_pose_error_left_positive(self):
        f=pose_features(dict(pose_xy=[0.,-1.],raw_pose_xy=[0.,-2.],truth_xy=[0.,0.],pose_yaw=0.,truth_yaw=0.))
        self.assertEqual(f['pose_left_error'],1.)
    def test_fault_reasons_and_safe_stop_distinguished(self):
        for reason in ('invalid_lateral_prediction','invalid_pose','motion_gap','duplicate_tick'):
            self.assertTrue(invalid_control(dict(steer=0.,throttle=0.,brake=1.,reason=reason),{}))
        self.assertFalse(invalid_control(dict(steer=0.,throttle=0.,brake=1.,reason='trajectory_behind'),{}))
    def test_world_jerk_not_derivative_of_rotating_projection(self):
        trace=[dict(frame=1,sim_time=.1,acceleration_mps2=[1,0,0],right_vector=[0,1,0],forward_vector=[1,0,0]),dict(frame=2,sim_time=.2,acceleration_mps2=[1,0,0],right_vector=[-1,0,0],forward_vector=[0,1,0])]
        self.assertEqual(plant_features(trace)[2]['lateral_jerk'],0.)
    def test_same_metrics_only_primary_right_benefit_fails(self):
        metrics=[];coverage={}
        for route,segments in [('26966',[1]),('24240',[1]),('17563',[1,2])]:
            for segment in segments:
                for variant in ('baseline-zero','candidate-fixed-k'):
                    coverage[(route,variant,segment)]={'complete':True}
                    for band in ('window','post10m'):
                        metrics.append(dict(route=route,variant=variant,segment=segment,band=band,subset='all',count=30,
                            cte_m_rms=.2,cte_m_p95_abs=.3,cte_m_max_abs=.4,heading_deg_p95_abs=1.,speed_error_mps_rms=.1,
                            actual_speed_mean_mps=6.,lateral_accel_mps2_p95_abs=1.,steer_rate_per_s_p95_abs=.2))
        bad=[c for c in required_conditions(metrics,[],coverage,[])['required_conditions'] if c['status']!='pass']
        self.assertEqual([c['condition'] for c in bad],['turn/26966/1/primary_cte_rms_15percent'])
    def test_linear_mode_required_when_aim_evaluated(self):
        r=dict(aim_interpolation='linear',aim_interpolation_used='hermite',aim_interpolation_fallback=None,aim_xy=[3.,0.])
        self.assertEqual(aim_mode_evidence([r],'linear')['aim_used_or_fallback_error_ticks'],1)


if __name__=='__main__':unittest.main()
