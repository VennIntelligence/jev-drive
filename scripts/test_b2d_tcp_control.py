"""Numerical contract tests for the real-TCP B/C helper, without CARLA/models."""
import json
import math
import unittest

from b2d_tcp_control import TCPControlComparison


class TCPControlComparisonTests(unittest.TestCase):
    def step(self, c, t, speed=2., desired=2.2, throttle=.3, steer=-.123, brake=0.):
        return c.step(t, speed, desired, throttle, steer, brake)

    def test_native_common_preserves_steering_and_continuous_brake(self):
        c = TCPControlComparison()
        r = self.step(c, 0., throttle=.9, brake=.03)
        self.assertEqual(r['native_control'], [.9, -.123, .03])
        self.assertEqual(r['native_common_control'], [0., -.123, .03])
        self.assertEqual(r['pi_common_control'][1], -.123)
        r = self.step(c, .05, throttle=1., brake=0.)
        self.assertEqual(r['native_common_control'], [.75, -.123, 0.])

    def test_actual_time_integral_and_duplicate_idempotence(self):
        a, b = TCPControlComparison(), TCPControlComparison()
        for t in (0., .05, .1, .15, .2):
            ra = self.step(a, t)
        for t in (0., .1, .2):
            rb = self.step(b, t)
        # Constant .2 m/s error over .25 s including first nominal tick.
        self.assertAlmostEqual(a.pi.integral, .25 * .2 * .25)
        self.assertAlmostEqual(a.pi.integral, b.pi.integral)
        self.assertEqual(ra['pi_common_control'], rb['pi_common_control'])
        before = a.pi.integral
        dup = self.step(a, .2, desired=8., steer=.9)
        self.assertEqual(dup['diagnostics']['reason'], 'duplicate_tick')
        self.assertEqual(dup['pi_common_control'], ra['pi_common_control'])
        self.assertEqual(a.pi.integral, before)
        dup['pi_common_control'][0] = 99.
        self.assertNotEqual(self.step(a, .2)['pi_common_control'][0], 99.)

    def test_long_saturation_has_no_latent_throttle_on_deceleration(self):
        c = TCPControlComparison()
        for i in range(300):
            r = self.step(c, i * .05, speed=0., desired=8.)
            self.assertEqual(r['pi_common_control'][0], .75)
        self.assertEqual(c.pi.integral, 0.)
        self.assertTrue(r['diagnostics']['integration_limited'])
        r = self.step(c, 15., speed=8., desired=0.)
        self.assertEqual(r['pi_common_control'], [0., -.123, 1.])
        self.assertEqual(c.pi.integral, 0.)

    def test_stop_hold_reset_and_restart(self):
        c = TCPControlComparison()
        self.step(c, 0.)
        self.assertGreater(c.pi.integral, 0.)
        r = self.step(c, .05, speed=.099, desired=.049)
        self.assertEqual(r['diagnostics']['reason'], 'stop_hold')
        self.assertEqual(r['pi_common_control'], [0., -.123, 1.])
        self.assertEqual(c.pi.integral, 0.)
        r = self.step(c, .1, speed=0., desired=2.)
        self.assertEqual(r['pi_common_control'], [.75, -.123, 0.])
        c.reset()
        self.assertIsNone(c._last_time)
        self.assertEqual(c.pi.integral, 0.)
        self.assertEqual(self.step(c, 123.)['diagnostics']['elapsed_s'], .05)

    def test_bad_input_is_safe_strict_json_and_recovers(self):
        good = [0., 2., 2.2, .3, -.123, 0.]
        for index in range(6):
            for bad in (float('nan'), float('inf'), None):
                with self.subTest(index=index, bad=bad):
                    c = TCPControlComparison()
                    self.step(c, -.05)
                    values = good.copy(); values[index] = bad
                    r = c.step(*values)
                    self.assertEqual(r['diagnostics']['reason'], 'nonfinite_input')
                    for key in ('native_common_control', 'pi_common_control'):
                        self.assertEqual(r[key][0], 0.)
                        self.assertEqual(r[key][2], 1.)
                        self.assertTrue(all(math.isfinite(x) for x in r[key]))
                    json.dumps(r, allow_nan=False)
                    self.assertEqual(c.pi.integral, 0.)
                    recovered = self.step(c, .05)
                    self.assertEqual(recovered['diagnostics']['reason'], 'tracking')
                    self.assertAlmostEqual(c.pi.integral, .25 * .2 * .05)

    def test_reverse_is_not_silently_clipped(self):
        c = TCPControlComparison()
        r = self.step(c, 0., speed=-.00001)
        self.assertEqual(r['diagnostics']['reason'], 'reverse_motion')
        self.assertEqual(r['diagnostics']['speed_mps'], -.00001)
        self.assertEqual(r['pi_common_control'], [0., -.123, 1.])

    def test_gap_boundary_and_regression_reset(self):
        c = TCPControlComparison()
        self.step(c, 0.)
        self.assertEqual(self.step(c, .2)['diagnostics']['reason'], 'tracking')
        r = self.step(c, .4001)
        self.assertEqual(r['diagnostics']['reason'], 'motion_gap')
        self.assertEqual(c.pi.integral, 0.)
        self.assertEqual(self.step(c, .4501)['diagnostics']['reason'], 'tracking')
        self.assertEqual(self.step(c, .1)['diagnostics']['reason'], 'time_regression')
        self.assertEqual(c.pi.integral, 0.)
        self.assertEqual(self.step(c, .15)['diagnostics']['reason'], 'tracking')

    def test_bounds_mutual_exclusion_across_variable_inputs(self):
        c = TCPControlComparison()
        for i in range(120):
            steer = math.sin(i)
            r = self.step(c, i * .05, speed=(i % 13) * .8, desired=(i % 7) * .9,
                          throttle=math.sin(i) * 2., brake=math.cos(i) * 2., steer=steer)
            for key in ('native_common_control', 'pi_common_control'):
                throttle, actual_steer, brake = r[key]
                self.assertTrue(0. <= throttle <= .75 and 0. <= brake <= 1.)
                self.assertEqual(throttle * brake, 0.)
                self.assertEqual(actual_steer, steer)


if __name__ == '__main__':
    unittest.main()
