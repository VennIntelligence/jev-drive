"""PI gain validation and exact default replay against frozen v3 input/output."""
import json
from pathlib import Path
import unittest
import numpy as np
from b2d_controller import ConditionalPI, Controller
from test_b2d_controller_pi_review import vendor_replay_inputs
from b2d_controller_pi_selftest import speed_steps
from b2d_controller_selftest import circle, stop, s_curve


class PIGainTests(unittest.TestCase):
    def test_default_outputs_exactly_match_frozen_v3(self):
        golden = json.loads((Path(__file__).parent / 'testdata' / 'b2d_controller_pi_default_golden.json').read_text())
        for preset in ('carla', 'tcp', 'pursuit'):
            for gains in ({}, {'pi_kp': 1., 'pi_ki': .25}):
                controller = Controller(preset, longitudinal_mode='pi', **gains)
                actual = []
                for _, now, path, speed, yaw in vendor_replay_inputs():
                    if path is not None:
                        controller.update(path, now)
                    actual.append(list(controller.step(now, speed, yaw)) + [controller.longitudinal_pi.integral])
                np.testing.assert_array_equal(actual, golden['outputs'][preset])

    def test_invalid_gains_rejected_for_both_modes(self):
        for key in ('pi_kp', 'pi_ki'):
            for value in (0., -1., float('inf'), float('-inf'), float('nan')):
                for mode in ('vendor', 'pi'):
                    with self.subTest(key=key, value=value, mode=mode), self.assertRaises(ValueError):
                        Controller(longitudinal_mode=mode, **{key: value})

    def test_gains_are_instance_local_and_preserve_units(self):
        low = ConditionalPI(kp=.5, ki=.25)
        default = ConditionalPI()
        self.assertAlmostEqual(low.step(.2, .1), .105)
        self.assertAlmostEqual(default.step(.2, .1), .205)
        self.assertAlmostEqual(low.integral, .005)
        ctrl = Controller(longitudinal_mode='pi', pi_kp=.5)
        ctrl.update(np.column_stack((np.arange(1, 21) * 1.5, np.zeros(20))), 0.)
        ctrl.step(0., 5.8, 0.)
        self.assertEqual(ctrl.diagnostics['longitudinal_kp'], .5)
        self.assertEqual(ctrl.diagnostics['longitudinal_ki'], .25)

    def test_helpers_accept_explicit_gains(self):
        # Each helper exposes gain parameters through its trace identity as well
        # as passing them into the controller, without mutable global settings.
        results = [stop('pursuit', longitudinal_mode='pi', pi_kp=.5, pi_ki=.25),
                   circle('pursuit', longitudinal_mode='pi', pi_kp=.5, pi_ki=.25),
                   s_curve('pursuit', longitudinal_mode='pi', pi_kp=.5, pi_ki=.25),
                   speed_steps(pi_kp=.5, pi_ki=.25)]
        for result in results:
            self.assertTrue(result.get('main_pass', result.get('pass_accuracy', False)), result)


if __name__ == '__main__':
    unittest.main()
