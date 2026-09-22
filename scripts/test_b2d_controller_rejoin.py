"""Independent geometric/timing checks for the v2 route recovery reference."""
import math
import unittest

import numpy as np

from b2d_controller import Controller
from b2d_controller_adapter import RouteAdapter


class RejoinTests(unittest.TestCase):
    def test_8mps_has_no_origin_bridge_spike(self):
        route = RouteAdapter([[0., 0.], [100., 0.]], cruise_mps=8.)
        ego = np.array([0., 4.4])
        route.project(ego, 0., 8., 0.)
        trajectory = route.trajectory(ego, 0.)
        original_first_speed = np.linalg.norm([2., -4.4]) / .25
        self.assertGreater(original_first_speed, 19.)
        timed_speeds = np.linalg.norm(np.diff(np.vstack(([0., 0.], trajectory)), axis=0), axis=1) / .25
        self.assertLessEqual(float(timed_speeds.max()), 8. + 1e-10)
        self.assertGreater(timed_speeds[0], 7.9)
        controller = Controller()
        controller.update(trajectory, 0.)
        controller.step(0., 8., 0.)
        self.assertLessEqual(controller.diagnostics['target_speed_mps'], 8.)
        self.assertGreater(trajectory[-1, 1], 4.3)  # retains the real route offset in ego coordinates
        self.assertLess(route.rejoin_diagnostics['max_rejoin_curvature_inv_m'], .2)
        self.assertTrue(route.rejoin_diagnostics['curvature_bound_satisfied'])

    def test_start_behind_route_does_not_inflate_speed(self):
        route = RouteAdapter([[0., 0.], [100., 0.]], cruise_mps=8.)
        ego = np.array([-1.38863322, 0.])
        route.project(ego, 0., 0., 0.)
        trajectory = route.trajectory(ego, 0.)
        np.testing.assert_allclose(trajectory[:, 0], np.arange(1, 21) * 2., atol=1e-10)
        np.testing.assert_allclose(trajectory[:, 1], 0.)

    def test_mirrored_left_right_positions_and_heading(self):
        upper = RouteAdapter([[0., 0.], [25., 0.], [50., 4.], [90., 4.]], cruise_mps=6.)
        lower = RouteAdapter([[0., 0.], [25., 0.], [50., -4.], [90., -4.]], cruise_mps=6.)
        left, right = np.array([0., 2.]), np.array([0., -2.])
        upper.project(left, .15, 6., 0.)
        lower.project(right, -.15, 6., 0.)
        a, b = upper.trajectory(left, .15), lower.trajectory(right, -.15)
        np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-10)
        np.testing.assert_allclose(a[:, 1], -b[:, 1], atol=1e-10)
        self.assertAlmostEqual(upper.rejoin_diagnostics['join_length_m'], lower.rejoin_diagnostics['join_length_m'])

    def test_rejoin_returns_to_immutable_route_with_forward_tangent(self):
        route = RouteAdapter([[0., 0.], [100., 0.]], cruise_mps=6.)
        original = route.points.copy()
        ego = np.array([0., 2.])
        yaw = .2
        route.project(ego, yaw, 6., 0.)
        path = route._rejoin_path(ego, yaw)
        np.testing.assert_allclose(path[0], ego)
        heading = math.atan2(*(path[1] - path[0])[::-1])
        self.assertLess(abs(heading - yaw), .01)
        self.assertLess(np.max(np.abs(path[path[:, 0] > 19., 1])), 1e-10)
        self.assertTrue(np.all(np.diff(path[:, 0]) > 0))
        self.assertLess(route.rejoin_diagnostics['max_rejoin_curvature_inv_m'], .2)
        np.testing.assert_array_equal(route.points, original)
        np.testing.assert_array_equal(route.official_points, original[:-1])
        # Observe successive points on the returned path using the public bounded
        # projection, proving recovery can advance instead of locking its origin.
        for tick, xy in enumerate(path[:90], 1):
            route.project(xy, 0., 6., tick * .05)
        self.assertGreater(route.progress, 10.)

    def test_terminal_same_extended_endpoint_without_early_stop(self):
        route = RouteAdapter([[0., 0.], [100., 0.]], cruise_mps=6.)
        route.progress = 99.
        ego = np.array([99., .2])
        route.project(ego, 0., 2., 1.)
        result = route.trajectory(ego, 0.)
        expected = route.points[-1] - ego
        expected[1] *= -1
        np.testing.assert_allclose(result[-1], expected, atol=1e-9)
        self.assertGreater(result[-1, 0], 3.)
        self.assertFalse(route.terminal_hold)
        self.assertGreater(np.linalg.norm(result[0]), 0.)
        self.assertLessEqual(np.linalg.norm(result[0]) / .25, 6.)

    def test_short_remainder_reports_curvature_not_false_zero(self):
        route = RouteAdapter([[0., 0.], [10., 0.]], cruise_mps=6., end_extension_m=0.)
        route.progress = 9.95
        result = route.trajectory(np.array([9.95, .2]), 0.)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(route.rejoin_diagnostics['max_rejoin_curvature_inv_m'], .2)
        self.assertFalse(route.rejoin_diagnostics['curvature_bound_satisfied'])
        np.testing.assert_allclose(result[-1], [.05, .2], atol=1e-9)

    def test_almost_sideways_terminal_connector_is_flagged(self):
        route = RouteAdapter([[0., 0.], [10., 0.]], end_extension_m=0.)
        route.progress = 10.
        trajectory = route.trajectory(np.array([9.999, .5]), 0.)
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertEqual(route.rejoin_diagnostics['reason'], 'terminal_curvature_concern')
        self.assertFalse(route.rejoin_diagnostics['curvature_bound_satisfied'])
        np.testing.assert_allclose(trajectory[-1], [.001, .5], atol=1e-9)

    def test_tight_original_curve_is_retained_with_diagnostics(self):
        theta = np.linspace(0, math.pi / 2, 40)
        points = np.column_stack((3. * np.sin(theta), 3. * (1 - np.cos(theta))))
        route = RouteAdapter(points, cruise_mps=6.)
        route.project(np.array([0., 0.]), 0., 6., 0.)
        result = route.trajectory(np.array([0., 0.]), 0.)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(route.rejoin_diagnostics['reference_curvature_inv_m'], .3)
        self.assertGreater(route.rejoin_diagnostics['curvature_limit_inv_m'], .3)

    def test_terminal_opposite_final_tangent_cusp_stops_explicitly(self):
        route = RouteAdapter([[10., 0.], [0., 0.]], end_extension_m=0.)
        route.progress = 10.
        result = route.trajectory(np.array([-.5, 0.]), 0.)
        np.testing.assert_array_equal(result, np.zeros((20, 2)))
        self.assertEqual(route.rejoin_diagnostics['reason'], 'terminal_connector_reversal')
        self.assertFalse(route.rejoin_diagnostics['curvature_bound_satisfied'])
        self.assertIsNone(route.rejoin_diagnostics['max_rejoin_curvature_inv_m'])

    def test_opposite_heading_cusp_is_rejected_even_when_collinear(self):
        for yaw in (math.pi, math.pi - .01, -math.pi + .01):
            route = RouteAdapter([[0., 0.], [10., 0.], [30., 0.]])
            with self.assertRaisesRegex(ValueError, 'forward route rejoin'):
                route.trajectory(np.array([0., 0.]), yaw)

    def test_wrong_progress_cannot_fold_backward_rejoin(self):
        route = RouteAdapter([[0., 0.], [100., 0.]])
        with self.assertRaisesRegex(ValueError, 'forward route rejoin'):
            route.trajectory(np.array([79., 0.]), 0.)


if __name__ == '__main__':
    unittest.main()
