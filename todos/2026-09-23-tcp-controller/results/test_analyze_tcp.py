import unittest
import math
import tempfile
import json
from pathlib import Path
from analyze_tcp import frame_metrics,basis,recompute_desired,stats,analyze


def row(frame,t,yaw=0,accel=None):
    return dict(frame=frame,timestamp=t,arm='native_common',forward_count=1,native_pid_calls=1,raw_speed_mps=2.,selected_control=[.2,0.,0.],
                prediction=dict(raw_waypoints=[[10.,0.],[11.,0.],[12.,0.],[13.,0.]],metadata={'desired_speed':2.},native_control=[.2,0.,0.]),
                truth=dict(frame=frame,rotation_deg=[0.,0.,yaw],velocity=[2.,0.,0.],acceleration=accel or [1.,0.,0.],gear=1,applied_control=[.2,0.,0.]))


class AnalysisTests(unittest.TestCase):
    def test_world_jerk_before_rotating_body_projection(self):
        a,b=frame_metrics([row(1,.05,0),row(2,.1,90)])
        self.assertAlmostEqual(b['jerk_lon_mps3'],0)
        self.assertAlmostEqual(b['jerk_lat_mps3'],0)
        self.assertAlmostEqual(b['body_accel_lon_derivative_mps3'],-20)
    def test_world_acceleration_delta_projects_into_current_body(self):
        a,b=frame_metrics([row(1,.05,90,[0.,0.,0.]),row(2,.1,90,[0.,.1,0.])])
        self.assertAlmostEqual(b['jerk_lon_mps3'],2)
        self.assertAlmostEqual(b['jerk_lat_mps3'],0)
    def test_gap_is_not_bridged(self):
        a,b=frame_metrics([row(1,.05),row(3,.15,0,[3.,0.,0.])])
        self.assertIsNone(b['jerk_lon_mps3']);self.assertFalse(b['nominal_interval'])
    def test_speed_reference_excludes_origin_and_invalid_is_missing(self):
        self.assertEqual(recompute_desired([[10,0],[11,0],[12,0],[13,0]]),2)
        self.assertIsNone(recompute_desired([[10,0],[11,0],[12,0],['nan',0]]))
        r=row(1,.05);r['prediction']=None
        self.assertIsNone(frame_metrics([r])[0]['speed_error_mps'])
        self.assertIsNone(stats([])['rms'])
    def test_pitch_roll_axes_and_frame_mismatch(self):
        f,r=basis([20,30,40]);self.assertAlmostEqual(sum(x*x for x in f),1);self.assertAlmostEqual(sum(x*x for x in r),1);self.assertAlmostEqual(sum(x*y for x,y in zip(f,r)),0)
        x=row(1,.05);x['truth']['frame']=2
        self.assertFalse(frame_metrics([x])[0]['truth_aligned']);self.assertIsNone(frame_metrics([x])[0]['accel_lon_mps2'])
    def test_current_nested_native_call_schema(self):
        x=row(1,.05);x.pop('native_pid_calls');x['prediction']['native_pid_calls']=0
        self.assertEqual(frame_metrics([x])[0]['native_pid_calls'],0)

    def test_low_speed_requires_five_seconds_observed_span(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            for n,expected in [(100,0),(101,1)]:
                rs=[row(i+1,(i+1)*.05) for i in range(n)]
                for r in rs:r['truth']['velocity']=[0.,0.,0.]
                (path/'tcp-control.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
                result,_=analyze(path,'24211','native_common')
                self.assertEqual(len(result['low_speed_runs_ge_5s']),expected)
                if expected:self.assertAlmostEqual(result['low_speed_runs_ge_5s'][0]['duration_s'],5.)

    def test_sync_and_common_envelope_are_distinct(self):
        x=row(7,.35);x['sensor_frames']={k:7 for k in ('GPS','IMU','SPEED','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')};x['sensor_frames']['bev']=6
        x['raw_imu']=[0.]*7;x['prediction']['raw_target']=[1.,0.]
        x['official_tail_control']=[.1,1e-9,0.]
        y=frame_metrics([x])[0]
        self.assertTrue(y['sensor_frames_aligned']);self.assertTrue(y['raw_imu_finite'])
        self.assertAlmostEqual(y['official_tail_throttle_difference'],.1)
        self.assertAlmostEqual(y['official_tail_steer_difference'],-1e-9)
        x['sensor_frames']['CAM_FRONT']=6
        self.assertFalse(frame_metrics([x])[0]['sensor_frames_aligned'])

    def test_actual_pedal_matches_previous_command(self):
        rs=frame_metrics([row(1,.05),row(2,.1)]);self.assertEqual(rs[-1]['applied_pedals_previous_command_max_diff'],0)

if __name__=='__main__':unittest.main()
