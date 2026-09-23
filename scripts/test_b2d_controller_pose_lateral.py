"""Rear-axle propagation formula, interval ownership, opt-in and failure atomicity."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from b2d_controller_adapter import PoseFilter, controller_speed
from test_b2d_controller_pose_dropout import IdentityProjector, FINITE_GOLDENS
from test_b2d_controller_agent import gps_from_world, load_stub_agent

K=.010659832


class LateralPoseTests(unittest.TestCase):
    def make(self,k=K,**kw):
        return PoseFilter(IdentityProjector(),-1.4,lateral_coefficient_s2_per_m=k,**kw)

    def test_opt_in_validation_and_default_exact_golden(self):
        for k in (-1.,float('nan'),float('inf'),-float('inf')):
            with self.assertRaisesRegex(ValueError,'finite and nonnegative'):self.make(k)
        for explicit in (False,True):
            kw=dict(lateral_coefficient_s2_per_m=0.) if explicit else {}
            c=PoseFilter(IdentityProjector(),-.4,gnss_gain=.13,heading_gain=.17,**kw)
            for tick,expected in enumerate(FINITE_GOLDENS):
                xy,yaw=c.update([tick*.12+math.sin(tick)*.03,math.cos(tick/3.)*.1],
                    math.pi/2+.01*tick+.03*math.sin(tick/5.),
                    2.+.2*math.sin(tick/4.),.18+.02*math.cos(tick),tick*.05)
                self.assertEqual([float(xy[0]),float(xy[1]),float(yaw)],expected)
                self.assertEqual(c.diagnostics['lateral_reason'],'disabled' if tick else 'no_interval')

    def test_signed_interval_formula_and_gnss_attenuation(self):
        base,candidate=self.make(0.,gnss_gain=.2),self.make(gnss_gain=.2)
        for c in (base,candidate):c.update([0.,0.],math.pi/2,6.,.2,0.)
        b,yb=base.update([.8,0.],math.pi/2+.06,10.,.6,.1)
        p,yp=candidate.update([.8,0.],math.pi/2+.06,10.,.6,.1)
        # Positive yaw is a right turn; correction points left before GNSS.
        expected=-K*64*.4*.1*np.array([-math.sin(.03),math.cos(.03)])
        np.testing.assert_allclose(p-b,.8*expected,rtol=0,atol=2e-16)
        self.assertEqual(yp,yb)
        d=candidate.diagnostics
        self.assertTrue(d['lateral_valid']);self.assertEqual(d['lateral_reason'],'applied')
        self.assertEqual(d['lateral_mean_speed_mps'],8.)
        self.assertEqual(d['lateral_mean_world_gyro_rps'],.4)
        np.testing.assert_allclose(d['lateral_delta_xy_m'],expected,rtol=0,atol=1e-16)
        d['lateral_delta_xy_m'][0]=999
        self.assertNotEqual(candidate.diagnostics['lateral_delta_xy_m'][0],999)

    def test_rotation_translation_and_mirror(self):
        for reflect in (1.,-1.):
            theta=.73;rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
            transform=rot@np.diag([1.,reflect]);translation=np.array([10.,-20.])
            c,d=self.make(),self.make()
            for i in range(20):
                gps=np.array([i*.25,.07*math.sin(i*.3)])
                yaw=.04*i;speed=5.+.1*i;gyro=.2+.02*math.cos(i)
                x,h=c.update(gps,math.pi/2+yaw,speed,gyro,i*.05)
                xx,hh=d.update(transform@gps+translation,math.pi/2+theta+reflect*yaw,
                               speed,reflect*gyro,i*.05)
                np.testing.assert_allclose(xx,transform@x+translation,rtol=0,atol=2e-14)
                self.assertAlmostEqual(hh,theta+reflect*h,places=13)

    def test_faults_do_not_mutate_pose_or_gyro_history(self):
        for fault in ('gps','speed','gyro','time','duplicate','regression','overflow'):
            c=self.make();c.update([0.,0.],math.pi/2,8.,.3,0.)
            state=(c.xy.copy(),c.yaw,c.t,c.previous_speed,c.previous_world_gyro,c._last_compass_time)
            args=dict(gps=[.4,0.],compass=math.pi/2,speed=8.,world_yaw_rate=.3,timestamp=.05)
            if fault=='gps':args['gps']=[float('nan'),0.]
            elif fault in ('speed','gyro'):args['world_yaw_rate' if fault=='gyro' else fault]=float('nan')
            elif fault=='time':args['timestamp']=.25
            elif fault=='duplicate':args['timestamp']=0.
            elif fault=='regression':args['timestamp']=-.05
            else:args['speed']=1e308
            with self.assertRaises(ValueError):c.update(**args)
            np.testing.assert_array_equal(c.xy,state[0])
            self.assertEqual((c.yaw,c.t,c.previous_speed,c.previous_world_gyro,c._last_compass_time),state[1:])
            self.assertFalse(c.diagnostics['lateral_valid'])
            self.assertIsNone(c.diagnostics['lateral_delta_xy_m'])

    def test_reset_dropouts_and_stationary_interval(self):
        c=self.make();c.update([0.,0.],math.pi/2,2.,.4,0.)
        for t in (.05,.1,.15,.2):
            c.update([2*t,0.],float('nan'),2.,.4,t)
            self.assertEqual(c.diagnostics['lateral_reason'],'applied')
        state=c.xy.copy();gyro=c.previous_world_gyro
        with self.assertRaisesRegex(ValueError,'dropout exceeds'):
            c.update([.5,0.],float('nan'),2.,.4,.25)
        np.testing.assert_array_equal(c.xy,state);self.assertEqual(c.previous_world_gyro,gyro)
        c.reset();self.assertIsNone(c.previous_world_gyro)
        self.assertEqual(c.diagnostics['lateral_reason'],'no_interval')
        c.update([1.,2.],math.pi/2,0.,.4,10.)
        self.assertFalse(c.diagnostics['lateral_valid'])
        c.update([1.,2.],math.pi/2,0.,.4,10.05)
        self.assertEqual(c.diagnostics['lateral_delta_xy_m'],[0.,0.])
        self.assertEqual(c.diagnostics['lateral_reason'],'applied')
        self.assertEqual(controller_speed(-.0001),0.)
        self.assertEqual(controller_speed(-.1),-.1)

    def test_top_level_and_nested_agent_config_reach_pose_only(self):
        for nested in (False,True):
            module,_=load_stub_agent();agent=module.StubAgent()
            points=np.array([[0.,0.],[100.,0.]])
            gps=gps_from_world(points)
            agent.set_global_plan([({'lat':p[0],'lon':p[1]},None) for p in gps],
                [(SimpleNamespace(location=SimpleNamespace(x=x,y=y)),None) for x,y in points])
            with tempfile.TemporaryDirectory() as temp:
                parameters=dict(rear_axle_offset_m=-1.4,truth_logging=False)
                if nested:parameters['adapter']={'pose_lateral_coefficient_s2_per_m':K}
                else:parameters['pose_lateral_coefficient_s2_per_m']=K
                pp=Path(temp)/'controller.json';pp.write_text(json.dumps(parameters))
                config=dict(drive='controller',rig='none',policy='none',controller_preset='pursuit',
                            controller_config=str(pp),out=temp)
                path=Path(temp)/'agent.json';path.write_text(json.dumps(config))
                agent.setup(str(path))
                try:
                    self.assertEqual(agent._pose_filter.lateral_coefficient,K)
                    self.assertFalse(hasattr(agent._controller,'pose_lateral_coefficient_s2_per_m'))
                    archived=json.loads((Path(temp)/'route_reference.json').read_text())
                    self.assertEqual(archived['adapter']['pose_lateral_coefficient_s2_per_m'],K)
                finally:agent.destroy()


if __name__=='__main__':unittest.main()
