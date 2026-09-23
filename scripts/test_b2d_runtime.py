"""CARLA-client-only regressions: DATA_DIR=/data envs/carla/bin/python scripts/test_b2d_runtime.py.

No simulator is started; imports use Bench2Drive 0.0.4 and the client venv.
"""
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import b2d_route

b2d_route.add_bench2drive_to_path(os.environ.get(
    "BENCH2DRIVE_ROOT", str(Path(os.environ["DATA_DIR"]) / "third_party/Bench2Drive")))
import carla
from b2d_agent import StubAgent
import b2d_run


class SteeringTests(unittest.TestCase):
    def agent(self, xy, x, y=0, yaw=0):
        agent = StubAgent.__new__(StubAgent)
        agent.hero_actor = SimpleNamespace(get_transform=lambda: carla.Transform(
            carla.Location(x=x, y=y), carla.Rotation(yaw=yaw)))
        agent._global_plan_world_coord = [(carla.Transform(carla.Location(x=px, y=py)), None)
                                         for px, py in xy]
        return agent

    def test_passed_origin_is_not_a_steering_target(self):
        agent = self.agent([(0, 0), (10, 0), (20, 0)], 8)
        self.assertAlmostEqual(agent._steer_to_route(), 0)
        self.assertEqual(agent._route_index, 1)

    def test_progress_never_goes_backwards(self):
        agent = self.agent([(x, 0) for x in range(21)], 8)
        agent._steer_to_route()
        agent.hero_actor.get_transform = lambda: carla.Transform(carla.Location(x=6))
        self.assertAlmostEqual(agent._steer_to_route(), 0)
        self.assertEqual(agent._route_index, 8)

    def test_dense_plan_survives_base_downsampling(self):
        from leaderboard.autoagents.autonomous_agent import AutonomousAgent
        agent = self.agent([(x, 0) for x in range(21)], 8)
        route = agent._global_plan_world_coord
        with patch.object(AutonomousAgent, "set_global_plan"):
            agent.set_global_plan([], route)
        self.assertEqual(len(agent._drive_plan), 21)
        self.assertAlmostEqual(agent._steer_to_route(), 0)

    def test_forward_turn_has_correct_sign(self):
        agent = self.agent([(0, 0), (5, 0), (10, 5), (10, 10)], 4)
        self.assertGreater(agent._steer_to_route(), 0)


class CancellationTests(unittest.TestCase):
    def args(self, out):
        with patch('sys.argv', ['b2d_run.py', '--out', out, '--workers', '1', '--route-ids', '1711']):
            return b2d_run.parse_args()

    def test_evaluator_signal_is_recorded_and_delegated(self):
        class Evaluator:
            def _signal_handler(self, signum, frame):
                self.stopped = True
        b2d_route._patch_signal_handler(Evaluator)
        evaluator = Evaluator()
        evaluator._signal_handler(signal.SIGINT, None)
        self.assertEqual(evaluator._jev_interrupt_signal, signal.SIGINT)
        self.assertTrue(evaluator.stopped)

    def test_cancel_overrides_child_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = b2d_run.Runner(self.args(tmp), [('1711', 'Town12')])
            result = Path(tmp) / 'attempts/1711/1/route_result.json'
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({'status': 'finished', 'profile': {'ticks': 12}}))
            server = SimpleNamespace(tm_port=8000, port=2000, index=0, log='mock.log',
                                     routes_served=0, started_at=time.time())
            proc = SimpleNamespace(pid=99999999, returncode=130)
            with patch.object(runner, 'supervise', return_value='cancelled_by_user'), \
                    patch.object(b2d_run.subprocess, 'Popen', return_value=proc), \
                    patch.object(b2d_run, 'port_free', return_value=True):
                ok, record = runner.run_once(0, server, '1711', 1)
            self.assertFalse(ok)
            self.assertEqual(record['status'], 'cancelled_by_user')
            runner.events.close()

    def test_interrupt_waits_for_worker_cleanup_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = b2d_run.Runner(self.args(tmp), [('1711', 'Town12')])
            cleaned = threading.Event()
            def worker(wi):
                while not runner.stop_flag:
                    cleaned.wait(0.01)
                cleaned.set()
            with patch.object(runner, 'worker', side_effect=worker), \
                    patch.object(b2d_run.time, 'sleep', side_effect=KeyboardInterrupt):
                self.assertEqual(runner.run(), 130)
            self.assertTrue(cleaned.is_set())
            self.assertEqual(json.loads((Path(tmp) / 'summary.json').read_text())[
                'routes_never_finished'], ['1711'])
            runner.events.close()

    def test_worker_error_releases_claim_and_fails_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = b2d_run.Runner(self.args(tmp), [('1711', 'Town12')])
            with patch.object(b2d_run.Server, 'start', side_effect=RuntimeError('test startup failure')):
                self.assertEqual(runner.run(), 1)
            self.assertFalse(list((Path(tmp) / 'claims').glob('*')))
            runner.events.close()


class ExternalAgentTests(unittest.TestCase):
    def test_external_agent_and_config_reach_leaderboard(self):
        with patch('sys.argv', ['b2d_route.py', '--routes', '/routes.xml', '--route-id', '1711',
                               '--out', '/tmp/test-output', '--agent', '/official/npc_agent.py',
                               '--agent-config', '/weights/model.pth']):
            args = b2d_route.parse_args()
        result = b2d_route._leaderboard_args(args, Path('/stub.json'), Path('/tmp/test-output'))
        self.assertEqual(result.agent, '/official/npc_agent.py')
        self.assertEqual(result.agent_config, '/weights/model.pth')


if __name__ == '__main__':
    unittest.main()
