"""CPU-only wrapper checks against AST-extracted local official TCP code.

Run in the b2d-tcp environment. Network forward is a deterministic CPU stub;
official run_step, native PID and comparison methods execute their real source.
CUDA transfer requests in official run_step are redirected to CPU by a scoped
test patch. This does not test checkpoint inference or CARLA sensor integration.
"""
import ast
import copy
from collections import deque
import hashlib
import io
import json
import math
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from b2d_tcp_control import TCPControlComparison


HERE = Path(__file__).resolve().parent
ZOO = Path(os.environ.get('B2D_ZOO_ROOT', Path(os.environ.get('DATA_DIR', '/data')) / 'third_party/Bench2DriveZoo'))


def node(path, name, parent=None):
    tree = ast.parse(path.read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parent).body if parent else tree.body
    return copy.deepcopy(next(n for n in body if getattr(n, 'name', None) == name))


def execute(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<extracted TCP source>', 'exec'), namespace)


NATIVE = {'np': np, 'deque': deque, 'torch': torch, 'PLANNER_TYPE': 'only_traj', 'SAVE_PATH': None}
pid_method = node(ZOO / 'TCP/model.py', 'control_pid', 'TCP')
run_method = node(ZOO / 'team_code/tcp_b2d_agent.py', 'run_step', 'TCPAgent')
run_method.decorator_list = []
execute([node(ZOO / 'TCP/model.py', 'PIDController'), pid_method, run_method], NATIVE)


class Control:
    def __init__(self):
        self.throttle = self.steer = self.brake = 0.


NATIVE['carla'] = types.SimpleNamespace(VehicleControl=Control)


def config():
    return types.SimpleNamespace(seq_len=1, aim_dist=4., angle_thresh=.3,
                                 dist_thresh=10., brake_speed=.4, brake_ratio=1.1,
                                 clip_delta=.25, max_throttle=.75)


class NativeNet:
    control_pid = NATIVE['control_pid']

    def __init__(self):
        self.config = config()
        self.turn_controller = NATIVE['PIDController'](.75, .75, .3, 40)
        self.speed_controller = NATIVE['PIDController'](5., .5, 1., 40)
        self.forward_calls = 0
        self.prediction = torch.tensor([[[2., .1], [4., .25], [6., .4], [8., .5]]])

    def __call__(self, *args):
        self.forward_calls += 1
        self.agent._before_forward(self, args)
        return {'pred_wp': self.prediction.clone()}

    def process_action(self, *args):
        return 0., 0., 0., {}


class StubVisual:
    def _before_forward(self, *args):
        pass

    def tick(self, inputs):
        self.step += 1
        return dict(speed=inputs['SPEED'][1]['speed'], target_point=[12., .2],
                    next_command=4, compass=1., gps=np.zeros(2),
                    rgb=np.zeros((2, 2, 3), dtype=np.uint8))

    def run_step(self, inputs, timestamp):
        control = NATIVE['run_step'](self, inputs, timestamp)
        return self._postprocess_control(control, inputs, timestamp)


WRAPPER = dict(np=np, copy=copy, hashlib=hashlib, json=json, math=math, os=os,
               Path=Path, VisualTCPAgent=StubVisual, TCPControlComparison=TCPControlComparison,
               GameTime=types.SimpleNamespace(get_frame=lambda: 100))
execute([node(HERE / 'b2d_tcp_comparison_agent.py', name)
         for name in ('serial', 'vector', 'controls', 'ComparisonTCPAgent')], WRAPPER)
Agent = WRAPPER['ComparisonTCPAgent']


def agent(arm='native_common'):
    a = Agent()
    a._arm = arm
    a._comparison = TCPControlComparison()
    a._capture = None
    a._forward_count = 0
    a._telemetry = io.StringIO()
    a._truth = lambda frame: dict(frame=frame, test_stub=True)
    a._tcp_input = {}
    a.config = config()
    a.step = -1
    a.initialized = True
    a.metric_info = {}
    a.get_metric_info = lambda: {}
    a._im_transform = lambda rgb: torch.zeros((3, 2, 2))
    a.net = NativeNet()
    a.net.agent = a
    a._native_pid = a.net.control_pid
    a.native_pid_calls = 0
    def counted(*args):
        a.native_pid_calls += 1
        return a._capture_pid(*args)
    a.net.control_pid = counted
    return a


def inputs(speed=2.):
    return dict(SPEED=(100, {'speed': speed}), IMU=(100, np.zeros(7)), GPS=(100, np.zeros(3)))


ORIGINAL_TO = torch.Tensor.to


def cpu_to(tensor, *args, **kwargs):
    args = tuple('cpu' if isinstance(x, str) and x.startswith('cuda') else x for x in args)
    if isinstance(kwargs.get('device'), str) and kwargs['device'].startswith('cuda'):
        kwargs['device'] = 'cpu'
    return ORIGINAL_TO(tensor, *args, **kwargs)


class ComparisonAgentTests(unittest.TestCase):
    def test_native_pid_exact_outputs_and_pre_swap_copy(self):
        wrapped, native = agent(), NativeNet()
        rng = np.random.RandomState(9123)
        for tick in range(100):
            raw = np.cumsum(rng.uniform([.5, -.2], [2., .2], (1, 4, 2)), axis=1).astype('float32')
            target = rng.uniform([2., -.5], [15., .5], (1, 2)).astype('float32')
            velocity = torch.tensor([float(rng.uniform(0, 8))])
            expected = native.control_pid(torch.tensor(raw.copy()), velocity.clone(), torch.tensor(target.copy()))
            wp_tensor, target_tensor = torch.tensor(raw.copy()), torch.tensor(target.copy())
            wrapped._capture = None
            actual = wrapped._capture_pid(wp_tensor, velocity, target_tensor)
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(wrapped._capture['raw_waypoints'], raw[0])
            np.testing.assert_array_equal(wrapped._capture['raw_target'], target.reshape(2))
            np.testing.assert_array_equal(wp_tensor.numpy()[0], raw[0, :, ::-1])
            np.testing.assert_array_equal(target_tensor.numpy(), target[:, ::-1])
            self.assertAlmostEqual(wrapped._capture['desired_speed_recomputed_mps'], actual[3]['desired_speed'])
        self.assertEqual(list(wrapped.net.speed_controller._window), list(native.speed_controller._window))
        self.assertEqual(list(wrapped.net.turn_controller._window), list(native.turn_controller._window))

    def test_actual_vendor_dispatch_neutral_then_one_forward_and_pid(self):
        a = agent()
        with patch.object(torch.Tensor, 'to', cpu_to):
            neutral = a.run_step(inputs(), 0.)
            self.assertEqual(WRAPPER['controls'](neutral), [0., 0., 0.])
            self.assertEqual(a.net.forward_calls, 0)
            self.assertEqual(a.native_pid_calls, 0)
            self.assertIsNone(a._comparison._last_time)
            for i in range(1, 25):
                a.run_step(inputs(), i * .05)
                self.assertEqual(a.net.forward_calls, i)
                self.assertEqual(a.native_pid_calls, i)
                self.assertEqual(a._forward_count, 1)
        rows = [json.loads(x) for x in a._telemetry.getvalue().splitlines()]
        self.assertEqual(len(rows), 25)
        self.assertIsNone(rows[0]['comparison'])
        self.assertTrue(all(r['comparison'] is not None for r in rows[1:]))

    def test_two_arms_match_shadow_and_native_steering(self):
        b, c = agent('native_common'), agent('pi_common')
        with patch.object(torch.Tensor, 'to', cpu_to):
            for i in range(20):
                speed = 1. + (i % 8) * .3
                b.run_step(inputs(speed), i * .05)
                c.run_step(inputs(speed), i * .05)
        br = [json.loads(x) for x in b._telemetry.getvalue().splitlines()][1:]
        cr = [json.loads(x) for x in c._telemetry.getvalue().splitlines()][1:]
        for x, y in zip(br, cr):
            self.assertEqual(x['comparison'], y['comparison'])
            self.assertEqual(x['prediction'], y['prediction'])
            self.assertEqual(x['selected_control'], x['comparison']['native_common_control'])
            self.assertEqual(y['selected_control'], y['comparison']['pi_common_control'])
            steer = x['prediction']['native_control'][1]
            self.assertEqual(x['selected_control'][1], steer)
            self.assertEqual(y['selected_control'][1], steer)
        self.assertTrue(any(r['selected_control'][0] > r['official_tail_control'][0] for r in br))

    def test_duplicate_pid_and_missing_forward_rejected(self):
        a = agent()
        raw = a.net.prediction
        a._capture_pid(raw.clone(), torch.tensor([2.]), torch.tensor([[12., .2]]))
        with self.assertRaises(RuntimeError):
            a._capture_pid(raw.clone(), torch.tensor([2.]), torch.tensor([[12., .2]]))
        with self.assertRaises(RuntimeError):
            a._postprocess_control(Control(), inputs(), 0.)

    def test_nonfinite_each_input_skips_native_clears_histories_and_recovers(self):
        for arm in ('native_common', 'pi_common'):
            for field in ('waypoints', 'target', 'velocity'):
                with self.subTest(arm=arm, field=field):
                    a = agent(arm)
                    a._comparison.step(0., 2., 2.2, .3, .1, 0.)
                    for pid in (a.net.turn_controller, a.net.speed_controller):
                        pid.step(.2)
                    raw = a.net.prediction.clone()
                    target, velocity = torch.tensor([[12., .2]]), torch.tensor([2.])
                    {'waypoints': raw, 'target': target, 'velocity': velocity}[field].view(-1)[0] = float('nan')
                    with patch.object(a, '_native_pid', wraps=a._native_pid) as native:
                        result = a._capture_pid(raw, velocity, target)
                        native.assert_not_called()
                    self.assertEqual(result[:3], (0., 0., 1.))
                    self.assertEqual(a._capture['native_pid_calls'], 0)
                    self.assertFalse(a._capture['finite_prediction'])
                    for pid in (a.net.turn_controller, a.net.speed_controller):
                        self.assertEqual(list(pid._window), [0.] * 40)
                        self.assertEqual((pid._min, pid._max), (0., 0.))
                    a._forward_count = 1
                    fault_input = inputs(float('nan') if field == 'velocity' else 2.)
                    control = a._postprocess_control(Control(), fault_input, .05)
                    self.assertEqual(WRAPPER['controls'](control), [0., 0., 1.])
                    self.assertEqual(a._comparison.pi.integral, 0.)
                    row = json.loads(a._telemetry.getvalue().splitlines()[-1])
                    self.assertEqual(row['comparison']['diagnostics']['reason'], 'nonfinite_input')
                    json.dumps(row, allow_nan=False)
                    a._capture = None
                    native_clean = NativeNet()
                    args = (a.net.prediction.clone(), torch.tensor([2.]), torch.tensor([[12., .2]]))
                    expected = native_clean.control_pid(*(x.clone() for x in args))
                    recovered = a._capture_pid(*args)
                    self.assertEqual(recovered, expected)
                    control = a._postprocess_control(Control(), inputs(), .1)
                    self.assertTrue(all(math.isfinite(v) for v in WRAPPER['controls'](control)))
                    self.assertEqual(a._capture['native_pid_calls'], 1)

    def test_fault_during_actual_vendor_dispatch_brakes_both_arms(self):
        for arm in ('native_common', 'pi_common'):
            a = agent(arm)
            with patch.object(torch.Tensor, 'to', cpu_to):
                a.run_step(inputs(), 0.)
                a.run_step(inputs(), .05)
                saved = a.net.prediction.clone()
                a.net.prediction[0, 2, 0] = float('nan')
                control = a.run_step(inputs(), .1)
                self.assertEqual(WRAPPER['controls'](control), [0., 0., 1.])
                self.assertEqual(a._capture['native_pid_calls'], 0)
                a.net.prediction = saved
                a.run_step(inputs(), .15)
                self.assertEqual(a._capture['native_pid_calls'], 1)
                self.assertEqual(a.net.forward_calls, 3)

    def test_horizon_change_and_finite_metric_mismatch_are_loud(self):
        a = agent()
        with self.assertRaisesRegex(ValueError, 'horizon/shape'):
            a._capture_pid(torch.zeros((1, 5, 2)), torch.tensor([2.]), torch.tensor([[12., .2]]))
        self.assertIsNone(a._capture)
        a._capture_pid(a.net.prediction.clone(), torch.tensor([2.]), torch.tensor([[12., .2]]))
        a._capture['desired_speed_recomputed_mps'] += 1.
        a._forward_count = 1
        with self.assertRaisesRegex(ValueError, 'desired speed'):
            a._postprocess_control(Control(), inputs(), 0.)

    def test_visual_default_hook_identity_and_order(self):
        cls = node(HERE / 'b2d_tcp_visual_agent.py', 'VisualTCPAgent')
        hook = next(n for n in cls.body if getattr(n, 'name', None) == '_postprocess_control')
        namespace = {}
        execute([hook], namespace)
        control = Control()
        self.assertIs(namespace['_postprocess_control'](object(), control, {}, 0.), control)
        run = next(n for n in cls.body if getattr(n, 'name', None) == 'run_step')
        calls = [(n.lineno, n.func.attr) for n in ast.walk(run)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in ('run_step', '_postprocess_control', 'publish')]
        self.assertEqual([name for _, name in sorted(calls)], ['run_step', '_postprocess_control', 'publish'])


if __name__ == '__main__':
    unittest.main()
