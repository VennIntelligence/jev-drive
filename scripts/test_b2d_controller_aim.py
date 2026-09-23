"""Pure numerical contracts for the opt-in pursuit Hermite aim."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from b2d_controller import Controller


class HermiteAimTests(unittest.TestCase):
    @staticmethod
    def controller(mode='hermite'):
        return Controller('pursuit', lookahead='max', longitudinal_mode='pi',
                          pi_kp=.5, pi_ki=.25, aim_interpolation=mode)

    def evaluate(self, points, station, mode='hermite', arc=None):
        c = self.controller(mode)
        points = np.asarray(points, dtype=float)
        c._arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] if arc is None else arc.copy()
        return c._lateral_aim(points, station), c

    def test_default_and_explicit_linear_match_frozen_controls_exactly(self):
        fixture = json.loads((Path(__file__).parent / 'testdata/b2d_controller_turn_default_golden.json').read_text())
        for row in fixture['rows']:
            for extra in ({}, {'aim_interpolation': 'linear'}):
                c = Controller(row['preset'], lookahead=row['lookahead'],
                               longitudinal_mode=row['longitudinal_mode'], pi_kp=.5, pi_ki=.25, **extra)
                for i, expected in enumerate(row['controls']):
                    now, speed = i * .05, 6. + 2. * math.sin(i / 9.)
                    if i % 4 == 0:
                        x = speed * np.arange(1, 21) * .25
                        c.update(np.column_stack((x, .015 * math.cos(i / 7.) * x * x)), now)
                    self.assertEqual(list(c.step(now, speed, .025 * math.sin(i / 8.))), expected)
                    self.assertEqual(c.diagnostics['reason'], row['reasons'][i])

    def test_opt_in_validation_and_safe_diagnostics(self):
        for preset in ('carla', 'tcp'):
            with self.assertRaisesRegex(ValueError, 'requires pursuit'):
                Controller(preset, aim_interpolation='hermite')
        for mode in ('cubic', None, 1):
            with self.assertRaises(ValueError):
                Controller('pursuit', aim_interpolation=mode)
        c = self.controller()
        self.assertEqual(c.diagnostics['aim_interpolation'], 'hermite')
        self.assertIsNone(c.diagnostics['aim_interpolation_used'])
        self.assertIsNone(c.diagnostics['aim_interpolation_fallback'])

    def test_rigid_transform_and_mirror_equivariance(self):
        points = np.array([[0., 0.], [1., .08], [2.3, .4], [4., .9], [6., 1.7]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        for angle in (.0, .7, 2.4):
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            for mirror in (1., -1.):
                matrix = rotation @ np.diag([1., mirror])
                shift = np.array([3.7, -8.2])
                for station in (-1., .7, 1.5, 3.6, 5.2, arc[-1]+1):
                    original, _ = self.evaluate(points, station, arc=arc)
                    moved, c = self.evaluate(points @ matrix.T + shift, station, arc=arc)
                    np.testing.assert_allclose(moved, original @ matrix.T + shift, atol=2e-14)
                    self.assertIsNone(c.diagnostics['aim_interpolation_fallback'])

    def test_duplicate_knots_do_not_disable_moving_portion_or_change_arc(self):
        points = np.array([[0., 0.], [2., .1], [2., .1], [4., .4], [6., .9], [6., .9], [6., .9]])
        aim, c = self.evaluate(points, 2.7)
        before = c._arc.copy()
        deduplicated = points[[0, 1, 3, 4]]
        expected, _ = self.evaluate(deduplicated, 2.7)
        np.testing.assert_array_equal(aim, expected)
        self.assertEqual(c.diagnostics['aim_interpolation_used'], 'hermite')
        np.testing.assert_array_equal(c._arc, before)
        for station, endpoint in ((-10., points[0]), (100., points[-1])):
            value = c._lateral_aim(points, station)
            np.testing.assert_array_equal(value, endpoint)

    def test_two_points_stationary_and_local_cusp_fallback(self):
        for points in ([[0., 0.], [2., 1.]], [[0., 0.], [0., 0.]],
                       [[0., 0.], [1., 0.], [0., 0.], [-1., 0.]]):
            actual, c = self.evaluate(points, .5)
            expected, _ = self.evaluate(points, .5, mode='linear')
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(c.diagnostics['aim_interpolation_used'], 'linear')
            self.assertIsNotNone(c.diagnostics['aim_interpolation_fallback'])
        points = [[0., 0.], [1., .1], [2., .3], [3., .5], [2., .5]]
        _, c = self.evaluate(points, .5)
        self.assertEqual(c.diagnostics['aim_interpolation_used'], 'hermite')
        # A bad neighboring tangent falls back to a finite queried chord.
        points = np.array([[0., 0.], [1., .1], [np.nan, .3]])
        value, c = self.evaluate(points, .5, arc=np.array([0., 1., 2.]))
        self.assertTrue(np.isfinite(value).all())
        self.assertEqual(c.diagnostics['aim_interpolation_fallback'], 'nonfinite_local_knots')

    def test_speed_projection_and_pedals_unchanged_on_identical_inputs(self):
        a, b = self.controller('linear'), self.controller('hermite')
        for tick in range(120):
            now, speed = tick * .05, 6. + .2 * math.sin(tick / 5.)
            if tick % 4 == 0:
                x = speed * np.arange(1, 21) * .25
                points = np.column_stack((x, .006 * x*x))
                if 60 <= tick < 80:
                    points[8:] = points[8]
                if tick >= 100:
                    points[:] = 0.
                self.assertEqual(a.update(points, now), b.update(points, now))
            ca = a.step(now, speed, .01)
            cb = b.step(now, speed, .01)
            self.assertEqual((ca[0], ca[2]), (cb[0], cb[2]))
            self.assertEqual(a.diagnostics['target_speed_mps'], b.diagnostics['target_speed_mps'])
            np.testing.assert_array_equal(a._arc, b._arc)
            np.testing.assert_array_equal(a._times, b._times)
            np.testing.assert_array_equal(a._pose, b._pose)
        c = self.controller()
        c.update(np.tile([.1, 0.], (20, 1)), 0.)
        control = c.step(0., 0., 0.)
        self.assertTrue(np.isfinite(control).all())
        self.assertEqual(c.diagnostics['aim_interpolation_used'], 'linear')

    def test_ideal_circle_phase_reduces_ripple_and_analytic_error(self):
        for radius in (8., 20.):
            for speed in (6., 8.):
                for sign in (-1., 1.):
                    controllers = [self.controller(mode) for mode in ('linear', 'hermite')]
                    samples = [[], []]
                    future = np.arange(1, 21) * .25
                    theta = speed * future / radius
                    path = np.column_stack((radius*np.sin(theta), sign*radius*(1-np.cos(theta))))
                    for tick in range(120):
                        for i, c in enumerate(controllers):
                            if tick % 4 == 0:
                                c.update(path, tick*.05)
                            c.step(tick*.05, speed, sign*speed/radius)
                            samples[i].append(c.diagnostics['raw_steer'])
                            if i == 1:
                                self.assertEqual(c.diagnostics['aim_interpolation_used'], 'hermite')
                    c = controllers[0]
                    scale = np.interp(speed*3.6, c.steering_curve[:, 0], c.steering_curve[:, 1])
                    reference = -math.atan(c.wheelbase*sign/radius)/(c.max_steer_rad*scale)
                    linear, hermite = [np.array(x[20:]) for x in samples]
                    self.assertLess(np.ptp(hermite), .5*np.ptp(linear))
                    self.assertLess(np.sqrt(np.mean((hermite-reference)**2)),
                                    .5*np.sqrt(np.mean((linear-reference)**2)))
                    self.assertLess(abs(hermite.mean()-reference), abs(linear.mean()-reference))


if __name__ == '__main__':
    unittest.main()
