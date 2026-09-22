"""PI integral/safety and independent-plant acceptance, without CARLA."""
import unittest
import numpy as np
from b2d_controller import ConditionalPI, Controller
from b2d_controller_pi_selftest import speed_steps
from b2d_controller_selftest import stop


def straight(speed):
    return np.column_stack((speed * np.arange(1, 21) * .25, np.zeros(20)))


class PITests(unittest.TestCase):
    def test_saturation_does_not_charge_and_reversed_error_unwinds(self):
        pi = ConditionalPI()
        for _ in range(200):
            self.assertEqual(pi.step(10., .05), .75)
        self.assertEqual(pi.integral, 0.)
        for _ in range(100):
            pi.step(.1, .05)
        prior = pi.integral
        self.assertGreater(prior, 0.)
        pi.step(-.1, .1)
        self.assertLess(pi.integral, prior)
        for _ in range(200):
            pi.step(-10., .05)
        self.assertGreaterEqual(pi.integral, -1.)
        pi.reset(); self.assertEqual(pi.integral, 0.)

    def test_actual_elapsed_and_all_safe_paths_clear_bias(self):
        ctrl = Controller(longitudinal_mode='pi')
        ctrl.update(straight(6), 0.)
        ctrl.step(0., 5.5, 0.)
        self.assertAlmostEqual(ctrl.longitudinal_pi.integral, .25 * .5 * .05)
        ctrl.step(.1, 5.5, 0.)
        self.assertAlmostEqual(ctrl.longitudinal_pi.integral, .25 * .5 * .15)
        ctrl.step(.1, 5.5, 0.)  # duplicate tick must not integrate
        self.assertAlmostEqual(ctrl.longitudinal_pi.integral, .25 * .5 * .15)
        ctrl.step(.4, 5.5, 0.)
        self.assertEqual(ctrl.diagnostics['reason'], 'motion_gap')
        self.assertEqual(ctrl.longitudinal_pi.integral, 0.)
        for reason in ('invalid_motion', 'stationary_trajectory', 'stale_trajectory'):
            ctrl.longitudinal_pi.integral = .3
            control = ctrl._safe(reason)
            self.assertEqual(control[0], 0.)
            self.assertEqual(control[2], 1.)
            self.assertEqual(ctrl.longitudinal_pi.integral, 0.)

    def test_zero_trajectory_holds_then_restarts_without_bias(self):
        ctrl = Controller(longitudinal_mode='pi')
        ctrl.longitudinal_pi.integral = .5
        ctrl.update(straight(0), 0.)
        self.assertEqual(ctrl.step(0., 0., 0.), (0., 0., 1.))
        self.assertEqual(ctrl.longitudinal_pi.integral, 0.)
        ctrl.update(straight(6), .05)
        self.assertGreater(ctrl.step(.05, 0., 0.)[0], 0.)
        ctrl.reset(); self.assertEqual(ctrl.longitudinal_pi.integral, 0.)

    def test_frozen_vendor_outputs_and_default_unchanged(self):
        expected = {'carla': [.75, .4149, -.6669, .75, -1., .75, .1044, -.07605],
                    'tcp': [.75, .354375, 0., .75, -1., .75, 0., .010625]}
        for preset in ('carla', 'tcp', 'pursuit'):
            for mode in ({}, {'longitudinal_mode': 'vendor'}):
                ctrl = Controller(preset, **mode)
                actual = []
                for tick, speed in enumerate([0., 5.9, 6.2, 5.3, 7., 0., 6., 6.05]):
                    if tick % 4 == 0:
                        ctrl.update(straight(6), tick * .05)
                    throttle, _, brake = ctrl.step(tick * .05, speed, 0.)
                    actual.append(throttle - brake)
                np.testing.assert_allclose(actual, expected['carla' if preset == 'pursuit' else preset], atol=1e-12)

    def test_independent_plant_steps_delay_hold_and_deceleration(self):
        for delay in (0., .1, .2):
            result = speed_steps(delay)
            self.assertTrue(result['main_pass'], result)
        result = stop('pursuit', longitudinal_mode='pi')
        self.assertTrue(result['main_pass'], result)
        self.assertLess(result['deceleration_speed_rms_mps'], .5)


if __name__ == '__main__':
    unittest.main()
