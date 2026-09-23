"""Deterministic contract tests; no CARLA server or torch required."""
import ast
from collections import deque
import json
import math
import os
from pathlib import Path
import unittest
import numpy as np
from b2d_controller import Controller, WindowPID, advance_pose


TIMES = np.arange(1, 21) * .25


def straight(speed=6.):
    return np.column_stack((speed * TIMES, np.zeros(20)))


class ControllerTests(unittest.TestCase):
    def test_pid_vendor_scalar_sources(self):
        data = Path(os.environ.get('DATA_DIR', '/data'))
        carla = Path(os.environ.get('CARLA_ROOT', data / 'third_party/carla/CARLA_0.9.15'))
        zoo = Path(os.environ.get('B2D_ZOO_ROOT', data / 'third_party/Bench2DriveZoo'))
        sources = [
            (str(carla / 'PythonAPI/carla/agents/navigation/controller.py'),
             'PIDLongitudinalController', '_pid_control', 'carla'),
            (str(zoo / 'TCP/model.py'), 'PIDController', 'step', 'tcp')]
        for path, class_name, method_name, semantics in sources:
            if not Path(path).exists():
                self.skipTest('pinned local vendor source unavailable: ' + path)
            tree = ast.parse(Path(path).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
            module = ast.Module(body=[method], type_ignores=[])
            namespace = {'np': np}
            exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)
            reference = type('Reference', (), {})()
            reference._k_p = reference._K_P = .2
            reference._k_i = reference._K_I = .13
            reference._k_d = reference._K_D = .07
            reference._dt = .05
            reference._error_buffer = deque(maxlen=10)
            reference._window = deque([0.] * 40, maxlen=40)
            reference._max = reference._min = 0.
            ours = WindowPID(.2, .13, .07, 40 if semantics == 'tcp' else 10, semantics)
            for value in [0., .1, .1, .1, .4, -.3, -.3, 0.] * 8:
                expected = namespace[method_name](reference, value, 0.) if semantics == 'carla' else namespace[method_name](reference, value)
                self.assertAlmostEqual(ours.step(value), expected, places=12)

    def test_left_right_mirror_and_limits(self):
        outputs = []
        for sign in (1., -1.):
            ctrl = Controller(preset='pursuit')
            angle = 6. * TIMES / 20.
            path = np.column_stack((20. * np.sin(angle), sign * 20. * (1. - np.cos(angle))))
            ctrl.update(path, 0.)
            output = ctrl.step(0., 6., 0.)
            self.assertLess(sign * output[1], 0.)
            self.assertLessEqual(abs(output[1]), .1 + 1e-9)
            outputs.append(output)
        self.assertAlmostEqual(outputs[0][1], -outputs[1][1], places=12)

    def test_analytic_delayed_reprojection(self):
        for delay_ticks in (0, 1, 2, 6):
            for yaw_rate in (0., .3, -.3):
                ctrl = Controller(preset='pursuit')
                path = straight()
                for tick in range(delay_ticks + 1):
                    if tick == delay_ticks:
                        ctrl.update(path, 0.)
                    ctrl.step(tick * .05, 6., yaw_rate)
                t = delay_ticks * .05
                expected = np.array([6. * t, 0., 0.]) if yaw_rate == 0 else np.array([
                    6. / yaw_rate * math.sin(yaw_rate * t),
                    6. / yaw_rate * (1. - math.cos(yaw_rate * t)), yaw_rate * t])
                np.testing.assert_allclose(ctrl._pose, expected, atol=1e-12)
                np.testing.assert_allclose(ctrl._points[1:], path, atol=1e-12)
                reproj, _ = ctrl._geometry(t)
                c, s = math.cos(expected[2]), math.sin(expected[2])
                expected_local = (path - expected[:2]) @ np.array([[c, -s], [s, c]])
                np.testing.assert_allclose(reproj[1:], expected_local, atol=1e-12)

    def test_interpolated_source_pose(self):
        ctrl = Controller()
        ctrl.step(0., 6., .3)
        ctrl.step(.05, 6., .3)
        ctrl.update(straight(), .025)
        ctrl.step(.10, 6., .3)
        expected = advance_pose(np.zeros(3), 6., .3, .025)
        np.testing.assert_allclose(ctrl._points[0], expected[:2], atol=1e-12)

    def test_delayed_speed_window_uses_age(self):
        ctrl = Controller()
        ctrl.step(0., 0., 0.)
        ctrl.update(np.column_stack((8. * TIMES - TIMES ** 2 / 2., np.zeros(20))), 0.)
        ctrl.step(.30, 0., 0.)
        # Piecewise linear timed arc interpolant: [.30,.55] spans slopes 7.625 and 7.375.
        self.assertAlmostEqual(ctrl.diagnostics['target_speed_mps'], 7.575, places=9)
        self.assertAlmostEqual(ctrl.diagnostics['reference_speed_mps'], 7.625, places=9)

    def test_invalid_stale_order_reset(self):
        ctrl = Controller()
        self.assertEqual(ctrl.step(0., 0., 0.), (0., 0., 1.))
        self.assertEqual(ctrl.diagnostics['reason'], 'no_trajectory')
        self.assertTrue(ctrl.update(straight(), 0.))
        ctrl.step(.05, 0., 0.)
        self.assertFalse(ctrl.update(straight(), 0.))
        self.assertEqual(ctrl._rejection, 'duplicate_trajectory')
        self.assertFalse(ctrl.update(straight(), -.1))
        self.assertEqual(ctrl._rejection, 'out_of_order_trajectory')
        ctrl.step(.55, 0., 0.)
        self.assertEqual(ctrl.diagnostics['reason'], 'stale_trajectory')
        for invalid in ([], [[1., 2.]], np.full((20, 2), np.nan), np.full((20, 2), np.inf)):
            self.assertFalse(ctrl.update(invalid, .6))
            self.assertEqual(ctrl.step(.6, 0., 0.)[2], 1.)
            json.dumps(ctrl.diagnostics, allow_nan=False)
        ctrl.reset()
        self.assertIsNone(ctrl._source_time)
        self.assertEqual(len(ctrl._history), 0)
        ctrl.update(np.zeros((20, 2)), 1.)
        self.assertEqual(ctrl.step(1., 0., 0.), (0., 0., 1.))
        self.assertEqual(ctrl.diagnostics['reason'], 'stationary_trajectory')

    def test_future_history_and_clock(self):
        ctrl = Controller()
        ctrl.update(straight(), 1.)
        ctrl.step(0., 0., 0.)
        self.assertEqual(ctrl.diagnostics['reason'], 'future_trajectory')
        ctrl.update(straight(), -1.)
        ctrl.step(.05, 0., 0.)
        self.assertEqual(ctrl.diagnostics['reason'], 'trajectory_outside_history')
        ctrl.step(-1., 0., 0.)
        self.assertEqual(ctrl.diagnostics['reason'], 'time_regression')
        self.assertEqual(ctrl.step(0., float('nan'), 0.)[2], 1.)
        json.dumps(ctrl.diagnostics, allow_nan=False)
        ctrl.reset()
        ctrl.update(straight(), 0.)
        result = ctrl.step(0., 0., 0.)
        count = len(ctrl.longitudinal.errors)
        self.assertEqual(ctrl.step(0., 0., 0.), result)
        self.assertEqual(len(ctrl.longitudinal.errors), count)

    def test_repeated_points_and_behind(self):
        ctrl = Controller()
        ctrl.update(np.tile([2., 0.], (20, 1)), 0.)
        result = ctrl.step(0., 0., 0.)
        self.assertTrue(np.isfinite(result).all())
        ctrl.reset()
        ctrl.update(-straight(), 0.)
        self.assertEqual(ctrl.step(0., 0., 0.)[2], 1.)
        self.assertEqual(ctrl.diagnostics['reason'], 'trajectory_behind')

    def test_curve_uses_kmh(self):
        angle = 6. * TIMES / 20.
        path = np.column_stack((20. * np.sin(angle), 20. * (1. - np.cos(angle))))
        ctrl = Controller('pursuit', steering_curve=[(0., 1.), (36., .5)], steer_rate=100.)
        ctrl.update(path, 0.)
        ctrl.step(0., 10., 0.)
        aim = ctrl.diagnostics['aim_xy']
        expected = -math.atan(ctrl.wheelbase * 2. * aim[1] / sum(x * x for x in aim)) / (ctrl.max_steer_rad * .5)
        self.assertAlmostEqual(ctrl.diagnostics['raw_steer'], expected, places=12)

    def test_diagnostic_snapshot(self):
        ctrl = Controller()
        ctrl.update(straight(), 0.)
        ctrl.step(0., 6., 0.)
        snapshot = ctrl.diagnostics
        snapshot['reason'] = 'changed'
        snapshot['aim_xy'][0] = 999.
        self.assertEqual(ctrl.diagnostics['reason'], 'tracking')
        self.assertNotEqual(ctrl.diagnostics['aim_xy'][0], 999.)

    def test_stop_hold_latch_and_release(self):
        ctrl = Controller('pursuit')
        ctrl.update(np.tile([.04, 0.], (20, 1)), 0.)
        self.assertEqual(ctrl.step(0., 0., 0.)[2], 1.)
        self.assertEqual(ctrl.diagnostics['reason'], 'stop_hold')
        ctrl.update(np.tile([.04, 0.], (20, 1)), .2)
        self.assertEqual(ctrl.step(.2, 0., 0.)[2], 1.)
        ctrl.update(straight(), .4)
        self.assertGreater(ctrl.step(.4, 0., 0.)[0], 0.)
        self.assertFalse(ctrl._stop_latched)
        ctrl.reset()
        self.assertFalse(ctrl._stop_latched)

    def test_fuzz_output_invariants(self):
        rng = np.random.RandomState(23)
        for preset in ('carla', 'tcp', 'pursuit'):
            ctrl = Controller(preset)
            previous = 0.
            for tick in range(400):
                if tick % 4 == 0:
                    ctrl.update(np.cumsum(rng.normal([1., 0.], [1., .8], (20, 2)), axis=0), tick * .05)
                throttle, steer, brake = ctrl.step(tick * .05, float(rng.uniform(0., 20.)), float(rng.uniform(-1., 1.)))
                self.assertTrue(np.isfinite([throttle, steer, brake]).all())
                self.assertTrue(0 <= throttle <= .75 and 0 <= brake <= 1 and abs(steer) <= .8)
                self.assertEqual(throttle * brake, 0.)
                self.assertLessEqual(abs(steer - previous), .10000001)
                previous = steer
                json.dumps(ctrl.diagnostics, allow_nan=False)


class ValidationMetricTests(unittest.TestCase):
    def fixture(self):
        rows, records = [], []
        for tick in range(221):
            t = tick * .05
            moving = tick < 120
            rows.append(dict(frame=tick + 1, elapsed_s=t, sim_time=t,
                             speed=8. if moving else 0., reference_speed_mps=8. if moving else 0.,
                             cross_track_m=.1, projection_distance_m=.1,
                             endpoint_error_m=20. if moving else .1, pitch_deg=3.,
                             truth_xy=[8. * min(t, 6.), .1]))
            records.append(dict(frame=tick + 1, truth_frame=tick + 1,
                                speed_mps=8. if moving else 0., target_speed_mps=None,
                                pose_error_m=.1, raw_pose_error_m=.3,
                                pose_heading_error_rad=math.radians(.2)))
        return rows, records

    def test_independent_truth_projection(self):
        from b2d_controller_validate import TruthProjection
        projection = TruthProjection([[0., 0.], [10., 0.], [10., 10.]])
        result = projection.measure(np.array([3., .2]), 0., 6.)
        self.assertAlmostEqual(result['progress_m'], 3.)
        self.assertAlmostEqual(result['cross_track_m'], -.2)
        self.assertAlmostEqual(result['projection_distance_m'], .2)
        # Evaluation uses only its own true progress; no agent object exists.
        result = projection.measure(np.array([4., .3]), 0., 6.)
        self.assertAlmostEqual(result['progress_m'], 4.)

    def test_null_speed_and_empty_metrics(self):
        from b2d_controller_validate import summarize
        rows, records = self.fixture()
        result = summarize(rows, records, 8., 2., 'completed', [], 120)
        self.assertTrue(result['gate_pass'])
        self.assertIsNone(result['command_speed_all']['rms'])
        self.assertEqual(result['cruise_speed_after_5s']['count'], 20)
        empty = summarize([], [], 8., 2., 'error', [], None)
        self.assertFalse(empty['gate_pass'])
        json.dumps(empty, allow_nan=False)

    def test_hold_drift_missing_heading_and_wrong_frames_fail(self):
        from b2d_controller_validate import summarize
        rows, records = self.fixture()
        for tick, row in enumerate(rows[120:]):
            row['truth_xy'][0] += .0015 * tick
            row['speed'] = .03
        result = summarize(rows, records, 8., 2., 'completed', [], 120)
        self.assertFalse(result['gates']['hold_displacement'])
        self.assertAlmostEqual(result['stop_hold_displacement_m'], .15)
        rows, records = self.fixture()
        for record in records:
            record.pop('pose_heading_error_rad')
        self.assertFalse(summarize(rows, records, 8., 2., 'completed', [], 120)['gates']['heading'])
        rows, records = self.fixture()
        for record in records:
            record['frame'] += 1000
            record['truth_frame'] += 1000
        self.assertFalse(summarize(rows, records, 8., 2., 'completed', [], 120)['gates']['telemetry_complete'])


if __name__ == '__main__':
    unittest.main()
