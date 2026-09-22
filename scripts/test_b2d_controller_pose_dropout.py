"""Bounded compass loss, independent lever-arm checks and finite-path goldens."""
import math
import unittest
import numpy as np
from b2d_controller_adapter import PoseFilter


class IdentityProjector:
    def project(self, gps):
        return np.asarray(gps, dtype=float)[:2].copy()


class CompassDropoutTests(unittest.TestCase):
    def make(self, **kwargs):
        return PoseFilter(IdentityProjector(), -.4, **kwargs)

    def test_gyro_prediction_and_gps_lever_arm_without_heading_observation(self):
        c = self.make(gnss_gain=.2, heading_gain=1.)
        c.update([0., 0.], math.pi/2, 2., .4, 0.)
        xy, yaw = c.update([4., -3.], float('nan'), 2., .4, .05)
        predicted = np.array([1., 0.]) + .1*np.array([math.cos(.01), math.sin(.01)])
        gps_rear = np.array([4., -3.]) + np.array([math.cos(.02), math.sin(.02)])
        np.testing.assert_allclose(xy, predicted+.2*(gps_rear-predicted), rtol=0., atol=1e-15)
        np.testing.assert_allclose(c.raw_xy, gps_rear, rtol=0., atol=1e-15)
        self.assertAlmostEqual(yaw, .02, places=14)
        self.assertEqual(c.diagnostics['reason'], 'compass_dropout_prediction')
        self.assertTrue(c.diagnostics['degraded'])
        d = c.diagnostics; d['reason'] = 'mutated'
        self.assertEqual(c.diagnostics['reason'], 'compass_dropout_prediction')

    def test_four_ticks_allowed_fifth_expires_without_pose_mutation(self):
        c = self.make();c.update([0., 0.], math.pi/2, 1., .2, 0.)
        for t in [.05, .1, .15, .2]:
            _, yaw = c.update([t, 0.], float('nan'), 1., .2, t)
            self.assertAlmostEqual(yaw, .2*t, places=14)
            self.assertAlmostEqual(c.diagnostics['compass_age_s'], t)
        saved = (c.xy.copy(), c.yaw, c.t, c.raw_xy.copy())
        with self.assertRaisesRegex(ValueError, 'dropout exceeds'):
            c.update([.25, 0.], float('nan'), 1., .2, .25)
        np.testing.assert_array_equal(c.xy, saved[0]);self.assertEqual((c.yaw,c.t),saved[1:3])
        np.testing.assert_array_equal(c.raw_xy,saved[3])
        self.assertEqual(c.diagnostics['reason'],'compass_dropout_timeout')

    def test_recovery_restores_compass_correction_and_refreshes_age(self):
        c = self.make();c.update([0.,0.],math.pi/2,0.,1.,0.)
        c.update([0.,0.],float('nan'),0.,1.,.05)
        _,yaw = c.update([0.,0.],math.pi/2+.3,0.,1.,.1)
        self.assertAlmostEqual(yaw,.1+.1*(.3-.1),places=14)
        self.assertFalse(c.diagnostics['degraded']);self.assertEqual(c.diagnostics['compass_age_s'],0.)
        c.update([0.,0.],float('nan'),0.,1.,.15)
        self.assertAlmostEqual(c.diagnostics['compass_age_s'],.05)

    def test_initial_missing_compass_and_reset_require_fresh_valid_pose(self):
        c=self.make()
        with self.assertRaisesRegex(ValueError,'initialization'):
            c.update([0.,0.],float('nan'),0.,0.,0.)
        self.assertIsNone(c.xy)
        c.update([2.,3.],math.pi/2,0.,0.,0.)
        c.update([2.,3.],float('nan'),0.,0.,.05)
        c.reset()
        self.assertIsNone(c.xy);self.assertIsNone(c.previous_speed)
        self.assertEqual(c.diagnostics['reason'],'no_pose')
        with self.assertRaisesRegex(ValueError,'initialization'):
            c.update([2.,3.],float('nan'),0.,0.,.1)
        np.testing.assert_array_equal(c.update([2.,3.],math.pi/2,0.,0.,.15)[0],[3.,3.])

    def test_noncompass_nonfinite_remains_strict(self):
        for field in ('gps','speed','gyro','timestamp'):
            c=self.make();c.update([0.,0.],math.pi/2,0.,0.,0.)
            original=c.xy.copy()
            args=dict(gps=[0.,0.],compass=math.pi/2,speed=0.,world_yaw_rate=0.,timestamp=.05)
            args[{'gyro':'world_yaw_rate'}.get(field,field)] = [float('nan'),0.] if field=='gps' else float('nan')
            with self.assertRaisesRegex(ValueError,'other than compass'):
                c.update(**args)
            np.testing.assert_array_equal(c.xy,original);self.assertEqual(c.t,0.)
            self.assertEqual(c.diagnostics['reason'],'invalid_motion')

    def test_finite_path_matches_frozen_goldens_exactly(self):
        c=self.make(gnss_gain=.13,heading_gain=.17)
        for tick,expected in enumerate(FINITE_GOLDENS):
            xy,yaw=c.update([tick*.12+math.sin(tick)*.03,math.cos(tick/3.)*.1],
                            math.pi/2+.01*tick+.03*math.sin(tick/5.),
                            2.+.2*math.sin(tick/4.),.18+.02*math.cos(tick),tick*.05)
            self.assertEqual([float(xy[0]),float(xy[1]),float(yaw)],expected,tick)
            self.assertFalse(c.diagnostics['degraded'])


# Exact pre-change PoseFilter outputs, generated once from captured frozen source
# SHA256 072ffa4f0dd7c1fdc7fc75433ffa8311e93992e404cfdbcfc57dc8b51cb48a76.
# Identity projector, 40 deterministic finite samples defined above; no old code vendored.
FINITE_GOLDENS = [
    [0.9999999999999999, 0.1, 0.0],
    [1.106940385288964, 0.10177929783737033, 0.010631664500925453],
    [1.2178708230481412, 0.10422805254730265, 0.02133491320720804],
    [1.3287758740729982, 0.10613450648912248, 0.03233596080411916],
    [1.4388006834390552, 0.10680518195629421, 0.04422483932568966],
    [1.5503242836301818, 0.10607876568132577, 0.057203558276777144],
    [1.666072483815561, 0.1042315020450043, 0.070699294046078],
    [1.7859479969109007, 0.10184311228844409, 0.0837019465522908],
    [1.90651391522744, 0.09969398127652226, 0.09551967598585209],
    [2.023624512500229, 0.09870255613374081, 0.10626171586837163],
    [2.1358025328711245, 0.0998507705363855, 0.11660821167842572],
    [2.2452734979081086, 0.10404816458217095, 0.12708182068190332],
    [2.355716022627395, 0.1119494321858477, 0.13749317217253854],
    [2.468799210093764, 0.12380062192700175, 0.1470715707281003],
    [2.5826859593150444, 0.13938627056421213, 0.15516133516123087],
    [2.6938670497986816, 0.15808723168511724, 0.16184307925725427],
    [2.8006240651928116, 0.1789959981821149, 0.16790719028337175],
    [2.9049176565882564, 0.20102573653692402, 0.1742013227442789],
    [3.010941064620322, 0.2229906512346702, 0.18094830648479032],
    [3.1216423099694595, 0.24368406893567635, 0.18765724397166306],
    [3.2363856455637006, 0.26199274872009964, 0.1937045278817151],
    [3.3518756157257745, 0.2770536312946896, 0.19904510641652795],
    [3.4654691372723563, 0.2884118540844074, 0.20439430026297423],
    [3.5778095280670414, 0.29611595712482025, 0.21070719369286106],
    [3.692345315270226, 0.3007082285093001, 0.21842859983610152],
    [3.8121674450751484, 0.303121972470346, 0.22719792239702885],
    [3.937070836843443, 0.3045430245295717, 0.2362755998828936],
    [4.06354530785776, 0.306292049206224, 0.24529517380501709],
    [4.187712908297742, 0.3097367661136891, 0.25464657153597514],
    [4.308527023125613, 0.3161937111117935, 0.26513629551020745],
    [4.428315059181494, 0.32677867743056477, 0.27723613493606436],
    [4.550193361297455, 0.34222010441026435, 0.2906114721995423],
    [4.674760138496433, 0.362709112883996, 0.30436432825957827],
    [4.799134169565987, 0.38786122620155555, 0.31777023370917323],
    [4.919247319884685, 0.4168002275823333, 0.33083495874134927],
    [5.033305549213831, 0.4483029784182936, 0.34416358287855964],
    [5.143255802333103, 0.4809312314666094, 0.3582672700324445],
    [5.2529229270350974, 0.5131287628262771, 0.3729505390786829],
    [5.364534019759641, 0.5433222386771817, 0.38734804888722696],
    [5.476822069344861, 0.5700707311929533, 0.4005827652716678],
]

if __name__=='__main__':unittest.main()
