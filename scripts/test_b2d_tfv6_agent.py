"""W2 arm routing and author postprocessor contract."""

from types import SimpleNamespace
import unittest

from b2d_tfv6_controller_agent import (
    _apply_postprocessors, _configure_author_arm, _normalize_brake, _select_arm_control,
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


if __name__ == "__main__":
    unittest.main()
