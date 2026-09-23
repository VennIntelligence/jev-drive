"""Lateral v2 opt-ins: rear-slip pursuit frame, Ackermann steer inverse, N-point trajectories,
and the diagnostic-only truth-pose ceiling. Off must stay bit-identical to the frozen controller."""
import json
import math
import os
from pathlib import Path
from queue import Queue
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from b2d_controller import (Controller, GRAVITY, ackermann_inner_angle, rear_slip_angle)
from test_b2d_controller_agent import gps_from_world, load_stub_agent
from test_b2d_controller_pi_review import vendor_replay_inputs

C_MKZ, TRACK_MKZ, L = 11.0, 1.5929, 2.8604714913890885
OFF = dict(pursuit_frame='body', rear_slip_c_per_rad=None, steer_inverse='nominal', track_width_m=None)
BOTH = dict(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ, steer_inverse='ackermann', track_width_m=TRACK_MKZ)
TIMES = np.arange(1, 21) * .25
# Keys the agent strips from a controller config before constructing Controller (b2d_agent.py).
ADAPTER_KEYS = ('adapter', 'rear_axle_offset_m', 'gnss_x_m', 'pose_gnss_gain', 'pose_heading_gain',
                'route_end_extension_m', 'truth_logging', 'route_stop_deceleration',
                'pose_lateral_coefficient_s2_per_m', 'diagnostic_truth_pose_ceiling', 'metadata', 'preset')
RECORDED = Path(os.environ.get('DATA_DIR', '/data')) / 'runs/b2d/controller/gpubox-port-v1'


def pursuit(**extra):
    return Controller('pursuit', lookahead='max', longitudinal_mode='pi', pi_kp=.5, pi_ki=.25, **extra)


def arc_path(speed, curvature, n=20, dt=.25):
    """Points on a circle of signed curvature (left positive) at constant speed, rear-axle frame."""
    s = speed * np.arange(1, n + 1) * dt
    if abs(curvature) < 1e-12:
        return np.column_stack((s, np.zeros(n)))
    return np.column_stack((np.sin(curvature * s) / curvature, (1 - np.cos(curvature * s)) / curvature))


def drive(controller, path_fn, yaw_rate, ticks=24, speed=8.):
    out = []
    for tick in range(ticks):
        now = tick * .05
        if tick % 4 == 0:
            controller.update(path_fn(tick), now)
        out.append(controller.step(now, speed, yaw_rate(tick)))
    return out


class OptInValidation(unittest.TestCase):
    def test_switch_and_parameter_must_come_together(self):
        for bad in (dict(pursuit_frame='rear_slip'), dict(rear_slip_c_per_rad=11.),
                    dict(steer_inverse='ackermann'), dict(track_width_m=1.6),
                    dict(pursuit_frame='rear_slip', rear_slip_c_per_rad=0.),
                    dict(pursuit_frame='rear_slip', rear_slip_c_per_rad=float('nan')),
                    dict(steer_inverse='ackermann', track_width_m=-1.),
                    dict(steer_inverse='ackermann', track_width_m=2 * L),
                    dict(pursuit_frame='slip'), dict(steer_inverse='inverse')):
            with self.assertRaises(ValueError, msg=bad):
                pursuit(**bad)
        for preset in ('carla', 'tcp'):
            for extra in (dict(pursuit_frame='rear_slip', rear_slip_c_per_rad=11.),
                          dict(steer_inverse='ackermann', track_width_m=1.6)):
                with self.assertRaisesRegex(ValueError, 'require pursuit'):
                    Controller(preset, **extra)
        pursuit(**BOTH)
        for preset in ('carla', 'tcp', 'pursuit'):
            Controller(preset, **OFF)


class BitIdenticalWhenOff(unittest.TestCase):
    def test_explicit_off_equals_default_on_vendor_replay(self):
        for preset in ('carla', 'tcp', 'pursuit'):
            default, explicit = Controller(preset), Controller(preset, **OFF)
            for tick, now, path, speed, yaw in vendor_replay_inputs():
                for c in (default, explicit):
                    if path is not None:
                        c.update(path, now)
                self.assertEqual(default.step(now, speed, yaw), explicit.step(now, speed, yaw), (preset, tick))
                self.assertEqual(default.diagnostics, explicit.diagnostics)

    def test_explicit_twenty_point_dt_equals_default_call(self):
        default, explicit = pursuit(), pursuit()
        for tick in range(80):
            now, speed = tick * .05, 6. + math.sin(tick / 9.)
            if tick % 4 == 0:
                path = arc_path(speed, .05 * math.sin(tick / 17.))
                default.update(path, now)
                explicit.update(path, now, trajectory_dt=.25)
            self.assertEqual(default.step(now, speed, .1), explicit.step(now, speed, .1))

    @unittest.skipUnless(RECORDED.is_dir(), 'recorded GPU-box pose-g2 runs not present')
    def test_recorded_closed_loop_replay_bit_identical(self):
        """Replay every logged update/step of the six pose-g2 cases; default and explicit-off must
        both reproduce the emitted controls exactly (run with OPENBLAS_CORETYPE=Barcelona)."""
        frames = 0
        for control_log in sorted(RECORDED.glob('*/*/pursuit/control.jsonl')):
            variant = control_log.parent.parent.name
            parameters = json.loads((RECORDED / 'inputs' / ('controller-%s.json' % variant)).read_text())
            for key in ADAPTER_KEYS:
                parameters.pop(key, None)
            updates = {row['frame']: row for row in map(json.loads, control_log.with_name('trajectories.jsonl')
                                                         .read_text().splitlines())}
            rows = [json.loads(line) for line in control_log.read_text().splitlines()]
            self.assertTrue(all(row['reason'] != 'invalid_pose' for row in rows))
            for extra in ({}, OFF):
                c = Controller('pursuit', **parameters, **extra)
                for row in rows:
                    update = updates.get(row['frame'])
                    if update is not None:
                        self.assertEqual(c.update(update['trajectory_xy'], update['sim_time']), update['accepted'])
                    output = c.step(row['sim_time'], row['speed_mps'], row['yaw_rate_rps'])
                    self.assertEqual(list(output), [row['throttle'], row['steer'], row['brake']],
                                     (str(control_log), row['frame'], extra))
            frames += len(rows)
        self.assertEqual(frames, 2758)


class RearSlipFrame(unittest.TestCase):
    def test_slip_angle_formula_sign_and_standstill(self):
        self.assertEqual(rear_slip_angle(8., .8, C_MKZ), math.atan(9. * .8 / (C_MKZ * GRAVITY)))
        self.assertGreater(rear_slip_angle(8., .5, C_MKZ), 0.)  # left turn: body points inside the course
        self.assertEqual(rear_slip_angle(8., -.5, C_MKZ), -rear_slip_angle(8., .5, C_MKZ))
        self.assertEqual(rear_slip_angle(0., 0., C_MKZ), 0.)
        self.assertLess(abs(rear_slip_angle(0., .001, C_MKZ)), 1e-5)
        # Equivalent to the frozen-k form beta = atan(k(v) v w), k(v) = (1 + 1/v)/(c g).
        v, w = 6., -.7
        self.assertAlmostEqual(rear_slip_angle(v, w, C_MKZ), math.atan((1 + 1 / v) / (C_MKZ * GRAVITY) * v * w), 15)

    def test_zero_yaw_rate_is_exactly_body_frame(self):
        body, slip = pursuit(), pursuit(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ)
        path = lambda tick: arc_path(8., .06)
        self.assertEqual(drive(body, path, lambda t: 0.), drive(slip, path, lambda t: 0.))

    def test_turning_adds_curvature_into_the_turn_and_mirrors(self):
        for curvature in (.06, -.06, .12):
            yaw = 8. * curvature
            body = drive(pursuit(), lambda t: arc_path(8., curvature), lambda t: yaw)
            slip_c = pursuit(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ)
            slip = drive(slip_c, lambda t: arc_path(8., curvature), lambda t: yaw)
            # CARLA steer is right positive: a left turn (curvature > 0) must steer further left.
            self.assertLess(math.copysign(1, curvature) * slip[-1][1], math.copysign(1, curvature) * body[-1][1])
            beta = slip_c.diagnostics['pursuit_rear_slip_rad']
            self.assertEqual(beta, rear_slip_angle(8., yaw, C_MKZ))
            mirror = drive(pursuit(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ),
                           lambda t: arc_path(8., -curvature), lambda t: -yaw)
            np.testing.assert_allclose([m[1] for m in mirror], [-s[1] for s in slip], atol=1e-12, rtol=0)
            np.testing.assert_allclose([m[0] for m in mirror], [s[0] for s in slip], atol=1e-12, rtol=0)

    def test_pursuit_arc_is_tangent_to_rotated_velocity(self):
        """The commanded curvature equals the circle through the aim point tangent to the rear-axle
        velocity (heading -beta in the body frame)."""
        c = pursuit(pursuit_frame='rear_slip', rear_slip_c_per_rad=C_MKZ, steer_inverse='ackermann',
                    track_width_m=TRACK_MKZ)
        drive(c, lambda t: arc_path(8., .09), lambda t: .72, ticks=1)
        aim, beta = np.array(c.diagnostics['aim_xy']), c.diagnostics['pursuit_rear_slip_rad']
        heading = -beta
        # Circle through origin tangent to (cos h, sin h) through aim: kappa = 2 (n . aim) / |aim|^2.
        normal = np.array([-math.sin(heading), math.cos(heading)])
        self.assertAlmostEqual(c.diagnostics['pursuit_curvature_inv_m'], 2 * normal @ aim / (aim @ aim), 12)


class AckermannInverse(unittest.TestCase):
    def test_round_trip_through_physx_ackermann_centre(self):
        for curvature in (1e-6, .01, .1, .145, .25, -.2):
            inner = ackermann_inner_angle(curvature, L, TRACK_MKZ)
            # PhysX accuracy-1 Ackermann: cot(centre) = cot(inner) + w / (2L).
            centre = math.copysign(math.atan(1 / (1 / math.tan(abs(inner)) + TRACK_MKZ / (2 * L))), inner)
            self.assertAlmostEqual(math.tan(centre) / L, curvature, 12)
            self.assertGreater(abs(inner), abs(math.atan(L * curvature)))
        self.assertEqual(ackermann_inner_angle(0., L, TRACK_MKZ), 0.)
        self.assertEqual(ackermann_inner_angle(-.1, L, TRACK_MKZ), -ackermann_inner_angle(.1, L, TRACK_MKZ))
        self.assertAlmostEqual(ackermann_inner_angle(1e-9, L, TRACK_MKZ) / (L * 1e-9), 1., 8)

    def test_controller_uses_inverse_and_matches_measured_gain_loss(self):
        nominal, ackermann = pursuit(), pursuit(steer_inverse='ackermann', track_width_m=TRACK_MKZ)
        path = lambda tick: arc_path(8., 1 / 6.9)
        a = drive(nominal, path, lambda t: 8. / 6.9, ticks=40)
        b = drive(ackermann, path, lambda t: 8. / 6.9, ticks=40)
        self.assertEqual([x[0] for x in a], [x[0] for x in b])  # longitudinal untouched
        kappa = ackermann.diagnostics['pursuit_curvature_inv_m']
        self.assertAlmostEqual(ackermann.diagnostics['pursuit_nominal_angle_rad'],
                               ackermann_inner_angle(kappa, L, TRACK_MKZ), 15)
        # On the R = 6.9 m circle the pursuit asks for about the path curvature, and the inverse
        # commands a larger nominal angle than atan(L kappa) by the Ackermann factor.
        self.assertAlmostEqual(kappa, 1 / 6.9, delta=.02)
        self.assertGreater(abs(ackermann.diagnostics['pursuit_nominal_angle_rad']), abs(math.atan(L * kappa)))


class NPointTrajectory(unittest.TestCase):
    def test_four_points_half_second_accepted_and_horizon_not_extrapolated(self):
        c = pursuit()
        tcp = arc_path(8., .02, n=4, dt=.5)
        self.assertTrue(c.update(tcp, 0., trajectory_dt=.5))
        throttle, steer, brake = c.step(0., 8., .16)
        self.assertEqual(c.diagnostics['reason'], 'tracking')
        self.assertAlmostEqual(c.diagnostics['target_speed_mps'], 8., 2)
        short = pursuit()
        self.assertTrue(short.update(arc_path(8., 0., n=2, dt=.25), 0., trajectory_dt=.25))
        short.step(0., 8., 0.)
        for tick in range(1, 7):
            short.step(tick * .05, 8., 0.)
        # Near window [age, age + .25] passes the .5 s horizon at age .3: no invented points.
        self.assertEqual(short.diagnostics['reason'], 'trajectory_horizon_exhausted')

    def test_linear_resampling_equivalence(self):
        """4 points at .5 s track the same polyline and timing as their midpoint resampling at .25 s."""
        coarse = arc_path(8., .03, n=4, dt=.5)
        fine = np.vstack([np.vstack(((a + b) / 2, b)) for a, b in zip(np.vstack((np.zeros(2), coarse[:-1])), coarse)])
        a, b = pursuit(), pursuit()
        a.update(coarse, 0., trajectory_dt=.5)
        b.update(fine, 0., trajectory_dt=.25)
        for tick in range(5):
            np.testing.assert_allclose(a.step(tick * .05, 8., .24), b.step(tick * .05, 8., .24), atol=1e-12, rtol=0)

    def test_invalid_shapes_and_steps_rejected(self):
        c = pursuit()
        for points, dt in ((np.zeros((0, 2)), .5), (np.ones((4, 3)), .5), (np.ones((4, 2)), 0.),
                           (np.ones((4, 2)), -.5), (np.ones((4, 2)), float('nan')), (np.ones((4, 2)), 'x')):
            self.assertFalse(c.update(points, 1., trajectory_dt=dt))
            self.assertEqual(c._rejection, 'invalid_trajectory')
        # The frozen default contract still demands exactly 20 points.
        self.assertFalse(c.update(np.ones((4, 2)), 2.))
        self.assertTrue(c.update(np.column_stack((8. * TIMES, np.zeros(20))), 3.))


class FakeHero:
    def __init__(self, clock, pose):
        self.id, self.clock, self.pose = 7, clock, pose

    def get_world(self):
        hero = self
        location = SimpleNamespace(x=hero.pose[0], y=hero.pose[1])
        rotation = SimpleNamespace(yaw=hero.pose[2], pitch=0.)
        actor = SimpleNamespace(get_transform=lambda: SimpleNamespace(location=location, rotation=rotation))
        snapshot = SimpleNamespace(frame=hero.clock.frame, find=lambda i: actor if i == hero.id else None)
        return SimpleNamespace(get_snapshot=lambda: snapshot)


class TruthPoseCeiling(unittest.TestCase):
    def setup_agent(self, temp, controller, agent_extra):
        module, clock = load_stub_agent()
        agent = module.StubAgent()
        points = np.array([[0., 0.], [100., 0.]])
        gps = gps_from_world(points)
        agent.set_global_plan([({'lat': p[0], 'lon': p[1]}, None) for p in gps],
                              [(SimpleNamespace(location=SimpleNamespace(x=x, y=y)), None) for x, y in points])
        parameters = Path(temp) / 'controller.json'
        parameters.write_text(json.dumps(dict({'rear_axle_offset_m': -1.4, 'truth_logging': False}, **controller)))
        config = Path(temp) / 'agent.json'
        config.write_text(json.dumps(dict({'drive': 'controller', 'rig': 'none', 'policy': 'none',
                                           'controller_preset': 'pursuit', 'controller_config': str(parameters),
                                           'decimate': 4, 'out': temp}, **agent_extra)))
        agent.setup(str(config))
        agent.sensor_interface = SimpleNamespace(_data_buffers=Queue(), _queue_timeout=.01)
        return agent, clock

    def test_requires_double_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            for controller, extra, message in (({'diagnostic_truth_pose_ceiling': True}, {}, 'allow_truth_pose'),
                                               ({'diagnostic_truth_pose_ceiling': True},
                                                {'allow_truth_pose_diagnostic': 'yes'}, 'allow_truth_pose'),
                                               ({'diagnostic_truth_pose_ceiling': 1},
                                                {'allow_truth_pose_diagnostic': True}, 'boolean')):
                with self.assertRaisesRegex(ValueError, message):
                    self.setup_agent(temp, controller, extra)
            agent, _ = self.setup_agent(temp, {}, {'allow_truth_pose_diagnostic': True})
            self.assertFalse(agent._truth_pose_ceiling)  # the agent flag alone enables nothing
            agent.destroy()

    def test_truth_rear_pose_drives_route_and_is_labelled(self):
        with tempfile.TemporaryDirectory() as temp:
            agent, clock = self.setup_agent(temp, {'diagnostic_truth_pose_ceiling': True},
                                            {'allow_truth_pose_diagnostic': True})
            truth = [10., 1.5, 0.]  # actor x, y, yaw deg; rear axle 1.4 m behind
            agent.hero_actor = FakeHero(clock, truth)
            try:
                for tick in range(5):
                    clock.frame, clock.timestamp = 200 + tick, tick * .05
                    truth[0] = 10. + .4 * tick
                    gps = gps_from_world([[truth[0] - 1.4, 0.]])[0]  # sensors say: on the route centre
                    imu = np.array([0., 0., 0., 0., 0., 0., math.pi / 2])
                    for tag, data in [('GPS', gps), ('IMU', imu), ('SPEED', {'speed': 8.})]:
                        agent.sensor_interface._data_buffers.put((tag, clock.frame, data))
                    agent._controller_call()
                records = [json.loads(x) for x in (Path(temp) / 'control.jsonl').read_text().splitlines()]
                trajectories = [json.loads(x) for x in (Path(temp) / 'trajectories.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['pose_source'] == 'truth_diagnostic_ceiling' for r in records))
                np.testing.assert_allclose(records[-1]['pose_xy'], [truth[0] - 1.4, 1.5], atol=1e-9)
                self.assertLess(abs(records[-1]['sensor_pose_xy'][1]), .05)
                np.testing.assert_allclose(trajectories[0]['pose_xy'], [8.6, 1.5], atol=1e-9)
                self.assertAlmostEqual(records[-1]['route_cross_track_m'], -1.5, 6)  # CARLA y right: 1.5 m right
                clock.frame, clock.timestamp = clock.frame + 1, clock.timestamp + .05
                agent.hero_actor.clock = SimpleNamespace(frame=-1)
                for tag, data in [('GPS', gps), ('IMU', imu), ('SPEED', {'speed': 8.})]:
                    agent.sensor_interface._data_buffers.put((tag, clock.frame, data))
                with self.assertRaisesRegex(RuntimeError, 'no truth at frame'):
                    agent._controller_call()
            finally:
                agent.destroy()


class RunPerturbation(unittest.TestCase):
    def test_table_validation_and_nominal_default(self):
        from b2d_controller_validate import load_perturbation
        self.assertIsNone(load_perturbation(None, None))
        with tempfile.TemporaryDirectory() as temp:
            table = Path(temp) / 'p.json'
            good = dict(gnss_noise_seed=3, imu_noise_seed=1003, spawn_lateral_m=.2, spawn_yaw_deg=-.5)
            table.write_text(json.dumps({'p03': good, 'p00': None, 'bad': dict(good, spawn_lateral_m=3.),
                                         'short': dict(gnss_noise_seed=1), 'flag': dict(good, gnss_noise_seed=True)}))
            self.assertEqual(load_perturbation(str(table), 'p03'), dict(good, id='p03'))
            self.assertIsNone(load_perturbation(str(table), 'p00'))  # registered nominal anchor
            for key in ('bad', 'short', 'flag', 'missing'):
                with self.assertRaises(ValueError):
                    load_perturbation(str(table), key)
            with self.assertRaises(ValueError):
                load_perturbation(None, 'p03')

    def test_spawn_offset_right_vector_and_identity(self):
        from b2d_controller_validate import perturbed_spawn
        class Location(SimpleNamespace):
            def __init__(self, x=0., y=0., z=0.):
                super().__init__(x=x, y=y, z=z)
            def __add__(self, o):
                return Location(self.x + o.x, self.y + o.y, self.z + o.z)
        carla = SimpleNamespace(Location=Location, Rotation=lambda **k: SimpleNamespace(**k),
                                Transform=lambda loc, rot: SimpleNamespace(location=loc, rotation=rot))
        start = SimpleNamespace(location=Location(10., 20., 1.), rotation=SimpleNamespace(pitch=0., yaw=90., roll=0.))
        nominal = perturbed_spawn(start, None, carla)
        self.assertEqual((nominal.location.x, nominal.location.y, nominal.location.z), (10., 20., 1.5))
        self.assertIs(nominal.rotation, start.rotation)
        moved = perturbed_spawn(start, dict(spawn_lateral_m=.3, spawn_yaw_deg=-1.), carla)
        # Heading +y (CARLA yaw 90 deg, facing south): its right is -x.
        self.assertAlmostEqual(moved.location.x, 9.7, 12)
        self.assertAlmostEqual(moved.location.y, 20., 12)
        self.assertEqual(moved.rotation.yaw, 89.)

    def test_agent_forwards_noise_seeds_only_when_given(self):
        module, _ = load_stub_agent()
        agent = module.StubAgent()
        agent.n_cam, agent.decimate, agent.drive = 0, 4, 'controller'
        agent.cfg = {}
        self.assertFalse(any('noise_seed' in spec for spec in agent.sensors()))
        agent.cfg = dict(gnss_noise_seed=7, imu_noise_seed=1007)
        seeds = {spec['id']: spec.get('noise_seed') for spec in agent.sensors()}
        self.assertEqual((seeds['GPS'], seeds['IMU'], seeds['SPEED']), (7, 1007, None))


if __name__ == '__main__':
    unittest.main()
