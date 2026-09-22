"""Offline integration checks; no CARLA server, Torch, or privileged control input."""
import importlib.util
import io
import json
import math
from pathlib import Path
from queue import Queue
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from b2d_controller_adapter import (FrameRouter, GPSProjector, PoseFilter, RouteAdapter,
                                    TruthLogger, compass_to_yaw, controller_speed, world_to_local)


def gps_from_world(points, lat_ref=42., lon_ref=2.):
    points = np.asarray(points)
    radius = 6378137.
    scale = math.cos(math.radians(lat_ref))
    mx = scale * radius * math.radians(lon_ref) + points[:, 0]
    my = scale * radius * math.log(math.tan(math.radians(90 + lat_ref) / 2)) - points[:, 1]
    return np.column_stack((np.rad2deg(2 * np.arctan(np.exp(my / (radius * scale)))) - 90,
                            np.rad2deg(mx / (radius * scale))))


class AdapterTests(unittest.TestCase):
    def test_signed_standstill_jitter_preserves_real_reverse_fault(self):
        from b2d_controller import Controller
        for speed in (-.00002, .00002, 0.):
            self.assertEqual(controller_speed(speed), 0.)
        self.assertEqual(controller_speed(-.02), -.02)
        controller = Controller()
        controller.step(0, controller_speed(-.02), 0)
        self.assertEqual(controller.diagnostics['reason'], 'invalid_motion')

    def test_cardinal_heading_left_positive_and_rear_offset(self):
        world = np.array([[0, 0], [100, 10]], dtype=float)
        projector = GPSProjector(gps_from_world(world), world)
        for yaw in (0, math.pi / 2, math.pi, -math.pi / 2):
            self.assertAlmostEqual(math.cos(compass_to_yaw(yaw + math.pi / 2)), math.cos(yaw))
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            left = np.array([math.sin(yaw), -math.cos(yaw)])
            np.testing.assert_allclose(world_to_local([forward, left], [0, 0], yaw), np.eye(2), atol=1e-12)
            actor = np.array([10., 12.])
            gps = gps_from_world([actor - 1.4 * forward])[0]
            pose = PoseFilter(projector, -1.38863322)
            xy, _ = pose.update(gps, yaw + math.pi / 2, 0, 0, .05)
            np.testing.assert_allclose(xy, actor - 1.38863322 * forward, atol=1e-6)

    def test_gps_projection_multiple_latitudes_and_route_orientations(self):
        for latitude in (0., 42., -35., 65.):
            for points in (np.array([[123, 20], [500, 20]]),
                           np.array([[123, 20], [123, 500]]),
                           np.array([[123, 20], [500, 250], [200, -10]])):
                gps = gps_from_world(points, latitude, -72.)
                projector = GPSProjector(gps, points)
                for expected, measurement in zip(points, gps):
                    np.testing.assert_allclose(projector.project(measurement), expected, atol=1e-6)
        with self.assertRaises(ValueError):
            GPSProjector([[42, 2], [42, 2]], [[0, 0], [0, 0]])

    def test_pose_filter_reduces_seeded_noise_without_truth(self):
        route = np.array([[0., 0.], [100, 0.]])
        projector = GPSProjector(gps_from_world(route), route)
        pose = PoseFilter(projector, -1.4)
        rng = np.random.RandomState(17)
        raw_errors, errors = [], []
        for tick in range(400):
            expected = np.array([tick * .05 * 8, 0.])
            noisy = expected + rng.normal(0, .5, size=2)
            xy, _ = pose.update(gps_from_world([noisy])[0], math.pi / 2, 8, 0, tick * .05)
            if tick >= 40:
                raw_errors.append(np.linalg.norm(noisy - expected))
                errors.append(np.linalg.norm(xy - expected))
        self.assertLess(np.mean(errors), .3 * np.mean(raw_errors))

    def test_motion_exact_camera_coherent_and_late_packets_retained(self):
        interface = SimpleNamespace(_data_buffers=Queue(), _queue_timeout=.02)
        router = FrameRouter(('A', 'B'))
        def motion(frame):
            for tag in ('GPS', 'IMU', 'SPEED'):
                interface._data_buffers.put((tag, frame, tag))
        interface._data_buffers.put(('A', 10, 'camera_a'))
        interface._data_buffers.put(('B', 9, 'camera_b_old'))
        motion(10)
        first = router.read(interface, 10)
        self.assertEqual(set(first), {'GPS', 'IMU', 'SPEED'})
        interface._data_buffers.put(('B', 10, 'camera_b'))
        motion(11)
        second = router.read(interface, 11)
        self.assertEqual(second['A'][0], 10)
        self.assertEqual(second['B'][0], 10)
        self.assertTrue(all(second[k][0] == 11 for k in router.MOTION))
        motion(13)
        motion(12)
        router.read(interface, 12)
        self.assertTrue(all(router.read(interface, 13)[k][0] == 13 for k in router.MOTION))
        with self.assertRaisesRegex(RuntimeError, 'motion frame 14'):
            router.read(interface, 14)

    def test_route_projection_monotonic_bounded_and_endpoint(self):
        points = [[0, 0], [50, 0], [50, 2], [0, 2]]
        route = RouteAdapter(points, end_extension_m=3)
        route.project(np.array([1., 1.9]), 0., 8., .05)
        self.assertLess(route.progress, 5.)
        prior = route.progress
        route.project(np.array([0., 1.9]), 0., 8., .1)
        self.assertEqual(route.progress, prior)
        route.project(np.array([45., 0.]), 0., 8., .15)
        self.assertLessEqual(route.progress, prior + 2.)
        straight = RouteAdapter([[0, 0], [100, 0]], end_extension_m=3)
        straight.progress = 99
        trajectory = straight.trajectory(np.array([99., 0]), 0.)
        self.assertGreater(trajectory[-1, 0], 1.)
        self.assertAlmostEqual(trajectory[-1, 0], 4.)
        np.testing.assert_allclose(trajectory[:, 1], 0)

    def test_truth_logger_frame_alignment_and_nonmutation(self):
        route = RouteAdapter([[0, 0], [100, 0]])
        route.progress = 12.
        xy, raw = np.array([10., 0.]), np.array([10.1, .2])
        logger = TruthLogger(lambda: None, route, -1.4)
        self.assertEqual(logger.measure(20, xy, raw, 0)['truth_error'], 'hero_unavailable')
        self.assertEqual(route.progress, 12.)
        np.testing.assert_equal(xy, [10, 0])
        transform = SimpleNamespace(location=SimpleNamespace(x=11.4, y=0),
                                    rotation=SimpleNamespace(yaw=0))
        snapshot = SimpleNamespace(frame=20, find=lambda _: SimpleNamespace(get_transform=lambda: transform))
        hero = SimpleNamespace(id=7, get_world=lambda: SimpleNamespace(get_snapshot=lambda: snapshot))
        logger = TruthLogger(lambda: hero, route, -1.4)
        result = logger.measure(20, xy, raw, 0)
        self.assertAlmostEqual(result['pose_error_m'], 0)
        self.assertAlmostEqual(result['truth_cross_track_m'], 0)
        transform.rotation.pitch = 15.
        transform.location.x = 10. + 1.4 * math.cos(math.radians(15.))
        pitched = logger.measure(20, xy, raw, 0)
        self.assertAlmostEqual(pitched['pose_error_m'], 0)
        snapshot.frame = 21
        self.assertEqual(logger.measure(20, xy, raw, 0)['truth_error'], 'frame_mismatch')

    def test_terminal_parking_latch_rejects_off_path_and_survives_noise(self):
        route = RouteAdapter([[0, 0], [100, 0]], end_extension_m=3.)
        route.progress = 102.9
        route.project(np.array([102.9, 2.]), 0., 0., .05)
        self.assertFalse(route.terminal_hold)
        route.project(np.array([102.9, 0.]), 0., 1., .1)
        self.assertFalse(route.terminal_hold)
        route.project(np.array([102.9, 0.]), 0., 0., .15)
        self.assertTrue(route.terminal_hold)
        # Later GNSS error must not create a fake velocity from ego to endpoint.
        for tick in range(100):
            xy = np.array([102.9 + .3 * math.sin(tick), .2 * math.cos(tick)])
            route.project(xy, 0., 0., .2 + tick * .05)
            np.testing.assert_array_equal(route.trajectory(xy, 0.), np.zeros((20, 2)))


def load_stub_agent():
    # Isolate simulator imports without replacing adapter/controller code under test.
    names = ('carla', 'leaderboard', 'leaderboard.autoagents',
             'leaderboard.autoagents.autonomous_agent', 'srunner',
             'srunner.scenariomanager', 'srunner.scenariomanager.timer')
    modules = {name: ModuleType(name) for name in names}
    modules['carla'].VehicleControl = lambda **kwargs: SimpleNamespace(**kwargs)
    class BaseAgent:
        def set_global_plan(self, gps, world):
            self._global_plan, self._global_plan_world_coord = gps[::50], world[::50]
    modules[names[3]].AutonomousAgent = BaseAgent
    modules[names[3]].Track = SimpleNamespace(SENSORS='SENSORS')
    clock = SimpleNamespace(frame=0, timestamp=0.)
    clock.get_frame = lambda: clock.frame
    clock.get_time = lambda: clock.timestamp
    modules[names[-1]].GameTime = clock
    spec = importlib.util.spec_from_file_location('test_isolated_b2d_agent', Path(__file__).with_name('b2d_agent.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, clock


class AgentTests(unittest.TestCase):
    def test_compass_dropout_brakes_after_bound_and_replans_on_recovery(self):
        module, clock = load_stub_agent()
        agent = module.StubAgent()
        points = np.array([[0., 0.], [100., 0.]])
        gps = gps_from_world(points)
        agent.set_global_plan([({'lat': p[0], 'lon': p[1]}, None) for p in gps],
            [(SimpleNamespace(location=SimpleNamespace(x=x, y=y)), None) for x, y in points])
        with tempfile.TemporaryDirectory() as temp:
            parameters = Path(temp) / 'controller.json'
            parameters.write_text(json.dumps({'rear_axle_offset_m': -1.4,
                'truth_logging': False, 'longitudinal_mode': 'pi', 'pi_kp': .5, 'pi_ki': .25}))
            config = Path(temp) / 'agent.json'
            config.write_text(json.dumps({'drive': 'controller', 'rig': 'none', 'policy': 'none',
                'controller_preset': 'pursuit', 'controller_config': str(parameters),
                'decimate': 4, 'out': temp}))
            agent.setup(str(config))
            agent.sensor_interface = SimpleNamespace(_data_buffers=Queue(), _queue_timeout=.01)
            try:
                controls = []
                for tick in range(11):
                    clock.frame, clock.timestamp = 100 + tick, tick * .05
                    compass = float('nan') if 1 <= tick <= 6 else math.pi / 2
                    measurement = gps_from_world([[tick * .1, 0.]])[0]
                    speed = float('nan') if tick == 9 else 2.
                    if tick == 8:
                        measurement[0] = float('nan')
                    imu = np.array([0., 0., 0., 0., 0., 0., compass])
                    for tag, data in [('GPS', measurement), ('IMU', imu), ('SPEED', {'speed': speed})]:
                        agent.sensor_interface._data_buffers.put((tag, clock.frame, data))
                    ctrl = agent._controller_call()
                    controls.append((ctrl.throttle, ctrl.steer, ctrl.brake))
                    if tick in (5, 6, 8, 9):
                        self.assertEqual(controls[-1], (0., 0., 1.))
                        self.assertIsNone(agent._pose_filter.t)
                        self.assertIsNone(agent._controller.diagnostics['trajectory_time'])
                records = [json.loads(line) for line in (Path(temp) / 'control.jsonl').read_text().splitlines()]
                motions = [json.loads(line) for line in (Path(temp) / 'motion.jsonl').read_text().splitlines()]
                trajectories = [json.loads(line) for line in (Path(temp) / 'trajectories.jsonl').read_text().splitlines()]
                self.assertEqual(len(records), 11)
                self.assertEqual(len(motions), 11)
                self.assertTrue(all(records[i]['pose_status']['degraded'] for i in range(1, 5)))
                self.assertTrue(all(records[i]['reason'] == 'invalid_pose' for i in (5, 6, 8, 9)))
                self.assertEqual(motions[1]['sensors']['IMU']['data'][6], 'nan')
                self.assertEqual(records[9]['speed_mps'], 'nan')
                # Tick 7 is outside the ordinary 4-tick planning phase: it must
                # still create a fresh trajectory before any resumed actuation.
                self.assertEqual(records[7]['trajectory_frame'], 107)
                self.assertEqual(records[10]['trajectory_frame'], 110)
                self.assertIn(107, [row['frame'] for row in trajectories])
                self.assertTrue(np.isfinite(controls).all())
                self.assertEqual(len(agent.timings['agent_total']), 11)
            finally:
                agent.destroy()

    def test_vendor_route_before_setup_lifecycle(self):
        module, _ = load_stub_agent()
        agent = module.StubAgent()
        points = np.array([[0., 0], [50., 0], [100., 0]])
        gps = gps_from_world(points)
        gps_plan = [({'lat': point[0], 'lon': point[1]}, None) for point in gps]
        world_plan = [(SimpleNamespace(location=SimpleNamespace(x=x, y=y)), None) for x, y in points]
        agent.set_global_plan(gps_plan, world_plan)
        self.assertEqual(len(agent._drive_plan), 3)
        with tempfile.TemporaryDirectory() as temp:
            controller_path = Path(temp) / 'controller.json'
            controller_path.write_text(json.dumps({'rear_axle_offset_m': -1.38863322,
                                                   'truth_logging': False}))
            config_path = Path(temp) / 'config.json'
            config_path.write_text(json.dumps({'drive': 'controller', 'rig': 'none',
                'policy': 'none', 'decimate': 4, 'controller_config': str(controller_path), 'out': temp}))
            agent.setup(str(config_path) + '+route_test')
            try:
                self.assertIsNotNone(agent._pose_filter)
                self.assertEqual(len(agent._route_adapter.official_points), 3)
                self.assertTrue((Path(temp) / 'route_reference.json').exists())
                frequencies = {sensor['id']: sensor.get('sensor_tick') for sensor in agent.sensors()}
                self.assertEqual(frequencies['GPS'], .05)
                self.assertEqual(frequencies['IMU'], .05)
                # Reused agent route reset clears controller/pose, parking, frame
                # caches, held actuation, and decimation phase together.
                agent._route_adapter.terminal_hold = True
                agent._route_adapter.progress = 90.
                agent._tick = 13
                agent._trajectory_frame = 50
                agent._frame_router.frames[50] = {'CAM_FRONT': (50, 'old')}
                agent._controller.update(np.column_stack((np.arange(1, 21), np.zeros(20))), 4.)
                agent._controller.step(4., 3., 0.)
                agent.set_global_plan(gps_plan, world_plan)
                self.assertEqual(agent._tick, 0)
                self.assertIsNone(agent._trajectory_frame)
                self.assertEqual(agent._frame_router.frames, {})
                self.assertFalse(agent._route_adapter.terminal_hold)
                self.assertEqual(agent._route_adapter.progress, 0.)
                self.assertIsNone(agent._pose_filter.t)
                self.assertIsNone(agent._controller.diagnostics['trajectory_time'])
                self.assertEqual(agent._control.brake, 1.)
            finally:
                agent.destroy()

    def replay(self, truth):
        module, clock = load_stub_agent()
        agent = module.StubAgent()
        agent._tick, agent.decimate = 0, 4
        agent._trajectory_frame = None
        agent._frame_router = FrameRouter()
        agent.sensor_interface = SimpleNamespace(_data_buffers=Queue(), _queue_timeout=.01)
        points = np.array([[0., 0], [100., 0]])
        agent._pose_filter = PoseFilter(GPSProjector(gps_from_world(points), points), -1.4)
        agent._route_adapter = RouteAdapter(points)
        calls = {'updates': [], 'steps': []}
        from b2d_controller import Controller as RealController
        class Controller:
            def __init__(self):
                self.real = RealController(preset='pursuit')
            @property
            def diagnostics(self):
                return self.real.diagnostics
            def update(self, trajectory, timestamp):
                calls['updates'].append(timestamp)
                return self.real.update(trajectory, timestamp)
            def step(self, timestamp, speed, yaw_rate):
                calls['steps'].append((timestamp, speed, yaw_rate))
                return self.real.step(timestamp, speed, yaw_rate)
        agent._controller = Controller()
        agent._truth_logger = truth
        agent._telemetry = io.StringIO()
        agent._trajectory_log = io.StringIO()
        agent._motion_log = io.StringIO()
        agent.timings = {'policy_ticks': 0, 'sensor_wait': [], 'infer': [],
                         'controller_step_ms': [], 'agent_total': []}
        controls = []
        for tick in range(12):
            clock.frame, clock.timestamp = tick + 100, tick * .05
            imu = np.array([0, 0, 0, 0, 0, .2, math.pi / 2])
            gps = gps_from_world([[tick * .4, 0.]])[0]
            for tag, data in [('GPS', gps), ('IMU', imu), ('SPEED', {'speed': 8.})]:
                agent.sensor_interface._data_buffers.put((tag, clock.frame, data))
            ctrl = agent._controller_call()
            controls.append((ctrl.throttle, ctrl.steer, ctrl.brake))
        return calls, controls, agent._telemetry.getvalue()

    def test_every_tick_step_decimated_update_and_truth_noninterference(self):
        without = self.replay(None)
        evil_truth = SimpleNamespace(measure=lambda *args: {'truth_xy': [1e8, -1e8], 'pose_error_m': 1e8})
        with_truth = self.replay(evil_truth)
        self.assertEqual(without[:2], with_truth[:2])
        calls = without[0]
        self.assertEqual(len(calls['steps']), 12)
        np.testing.assert_allclose(calls['updates'], [0, .2, .4])
        self.assertTrue(all(rate == -.2 for _, _, rate in calls['steps']))
        self.assertEqual(len(without[2].splitlines()), 12)


if __name__ == '__main__':
    unittest.main()
