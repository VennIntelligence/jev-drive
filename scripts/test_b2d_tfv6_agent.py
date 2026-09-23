"""W2 arm routing and author postprocessor contract."""

from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

from b2d_controller import Controller
from b2d_tfv6_controller_agent import (
    _apply_postprocessors, _changed_processor_keys, _clone_processor,
    _configure_author_arm, _normalize_brake, _processor_snapshot, _select_arm_control,
    controller_speed,
)


class AgentPathTests(unittest.TestCase):
    def test_a_retains_author_route_and_target_speed(self):
        config = SimpleNamespace(steer_modality="route", throttle_modality="target_speed",
                                 brake_modality="target_speed")
        _configure_author_arm(config, "A")
        self.assertEqual((config.steer_modality, config.throttle_modality,
                          config.brake_modality), ("route", "target_speed", "target_speed"))

    def test_b_uses_author_waypoint_pid_for_all_actions(self):
        config = SimpleNamespace(steer_modality="route", throttle_modality="target_speed",
                                 brake_modality="target_speed")
        _configure_author_arm(config, "B")
        self.assertEqual((config.steer_modality, config.throttle_modality,
                          config.brake_modality), ("waypoint", "waypoint", "waypoint"))

    def test_c_d_select_distinct_registered_controllers(self):
        raw = {arm: {"steer": index, "throttle": .2, "brake": 0.}
               for index, arm in enumerate("ABCD")}
        self.assertEqual(_select_arm_control("C", raw)["steer"], 2)
        self.assertEqual(_select_arm_control("D", raw)["steer"], 3)
        with self.assertRaises(ValueError):
            _select_arm_control("C", {"C": raw["C"]})

    def test_emergency_then_stop_sign_order_on_both_controller_arms(self):
        calls = []

        class Emergency:
            def adjust(self, speed, throttle, brake):
                calls.append("emergency")
                return .4, 0.

        class StopSign:
            def adjust(self, speed, throttle, brake):
                calls.append("stop")
                return 0., 1.

        for arm in "CD":
            final = _apply_postprocessors({"steer": .1, "throttle": .8, "brake": 0.},
                                          2.0, Emergency(), StopSign())
            self.assertEqual(final, {"steer": .1, "throttle": 0., "brake": 1.})
        self.assertEqual(calls, ["emergency", "stop", "emergency", "stop"])

    def test_author_brake_interlock_precedes_postprocessors(self):
        self.assertEqual(_normalize_brake({"steer": .3, "throttle": .5, "brake": 1.}, 0.),
                         {"steer": 0., "throttle": 0., "brake": 1.})

    def test_shadow_replays_last_creep_tick_without_advancing_live_processor(self):
        class ForceMove:
            def __init__(self):
                self.force_move = 1

            def adjust(self, speed, throttle, brake):
                if self.force_move:
                    self.force_move -= 1
                    return max(.4, throttle), 0.
                return throttle, brake

        class StopSign:
            def __init__(self):
                self.stop_sign_buffer = deque([SimpleNamespace(x=1.0)], maxlen=1)
                self.calls = 0

            def adjust(self, speed, throttle, brake):
                self.calls += 1
                return throttle, brake

        force, stop = ForceMove(), StopSign()
        force_adjust, stop_adjust = force.adjust, stop.adjust
        force.adjust = lambda *args: force_adjust(*args)
        stop.adjust = lambda *args: stop_adjust(*args)
        force_before, stop_before = _processor_snapshot(force), _processor_snapshot(stop)
        shadow_force, shadow_stop = _clone_processor(force), _clone_processor(stop)
        raw = {"steer": .0102737, "throttle": .3013387, "brake": 0.}
        actual = _apply_postprocessors(raw, .2072897, force, stop)
        replay = _apply_postprocessors(raw, .2072897, shadow_force, shadow_stop)
        self.assertEqual(actual, replay)
        self.assertEqual(actual["throttle"], .4)
        self.assertEqual(force.force_move, 0)
        self.assertEqual(shadow_force.force_move, 0)
        self.assertEqual(stop.calls, 1)
        self.assertEqual(shadow_stop.calls, 1)
        self.assertIsNot(stop.stop_sign_buffer, shadow_stop.stop_sign_buffer)
        self.assertEqual(_changed_processor_keys(force_before, force), ["force_move"])
        self.assertEqual(_changed_processor_keys(stop_before, stop), ["calls"])
        after_actual_force, after_actual_stop = _processor_snapshot(force), _processor_snapshot(stop)
        _apply_postprocessors(raw, .2072897, _clone_processor(force), _clone_processor(stop))
        self.assertEqual(_changed_processor_keys(after_actual_force, force), [])
        self.assertEqual(_changed_processor_keys(after_actual_stop, stop), [])

    def test_guard_catches_shared_mutable_state_in_shadow(self):
        class SharedList:
            def __init__(self):
                self.events = []

            def adjust(self, speed, throttle, brake):
                self.events.append("shadow")
                return throttle, brake

        live = SharedList()
        before = _processor_snapshot(live)
        shadow = _clone_processor(live)
        shadow.adjust(1.0, .3, 0.)
        self.assertEqual(_changed_processor_keys(before, live), ["events"])

    def test_production_speed_contract_on_forward_tfv6_plan(self):
        plan = np.column_stack((np.arange(1, 9, dtype=float) * .25, np.zeros(8)))
        def control(raw_speed):
            controller = Controller(preset="pursuit", longitudinal_mode="pi", lookahead="max",
                                    pi_kp=.5, pi_ki=.25, max_lookahead_time_s=.5)
            self.assertTrue(controller.update(plan, 0., trajectory_dt=.25))
            result = controller.step(0., controller_speed(raw_speed), 0.)
            return result, controller.diagnostics["reason"]

        (throttle, steer, brake), reason = control(-4e-6)
        self.assertGreater(throttle, 0.)
        self.assertEqual(brake, 0.)
        self.assertEqual(reason, "tracking")
        (throttle, steer, brake), reason = control(-.5)
        self.assertEqual((throttle, steer, brake), (0., 0., 1.))
        self.assertEqual(reason, "invalid_motion")


if __name__ == "__main__":
    unittest.main()
