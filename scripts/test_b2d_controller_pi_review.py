"""Independent PI review: compare vendor outputs to the frozen pre-PI source."""
import json
import math
from pathlib import Path
import unittest

import numpy as np
from b2d_controller import Controller

GOLDEN = Path(__file__).parent / 'testdata' / 'b2d_controller_vendor_golden.json'
TIMES = np.arange(1, 21) * .25


def vendor_replay_inputs():
    """Protocol vendor-irregular-v1, shared with one-time frozen-source extraction."""
    rng = np.random.RandomState(8231)
    now = 0.
    for tick in range(600):
        now += float(rng.uniform(.035, .065))
        desired = 5. + 2.*math.sin(tick / 35.)
        path = None
        if tick % 4 == 0:
            x = desired * TIMES
            path = np.column_stack((x, .008*math.sin(tick / 40.)*x*x))
            if 240 <= tick <= 268:
                path[:] = 0.
        speed = -.1 if tick == 120 else max(0., desired + math.sin(tick/15.))
        yaw = .02*math.sin(tick/25.)
        yield tick, now, path, speed, yaw


class IndependentPIReview(unittest.TestCase):
    @staticmethod
    def tick(controller, now, desired, measured):
        controller.update(np.column_stack((desired * TIMES, np.zeros(20))), now)
        return controller.step(now, measured, 0.)

    def test_pi_integral_depends_on_elapsed_time_not_tick_partition(self):
        controllers = []
        for times in ([0., .1, .2], [0., .05, .1, .15, .2]):
            c = Controller(preset='pursuit', longitudinal_mode='pi')
            for now in times:
                output = self.tick(c, now, 1., .8)
            # Error .2 m/s integrated over nominal first .05s + observed .2s.
            self.assertAlmostEqual(c.longitudinal_pi.integral, .25 * .2 * .25, places=12)
            previous_integral = c.longitudinal_pi.integral
            self.assertEqual(c.step(times[-1], .8, 0.), output)
            self.assertEqual(c.longitudinal_pi.integral, previous_integral)
            controllers.append(c)
        self.assertAlmostEqual(controllers[0]._last_control[0], controllers[1]._last_control[0], places=12)

    def test_saturated_launch_has_no_latent_throttle_and_respects_limits(self):
        c = Controller(preset='tcp', longitudinal_mode='pi', max_throttle=.4, max_brake=.3)
        for tick in range(300):
            throttle, _, brake = self.tick(c, tick*.05, 8., 0.)
            self.assertEqual((throttle, brake), (.4, 0.))
        throttle, _, brake = self.tick(c, 15., 8., 8.)
        self.assertAlmostEqual(throttle, 0., places=12)
        self.assertAlmostEqual(brake, 0., places=12)
        throttle, _, brake = self.tick(c, 15.05, 8., 12.)
        self.assertEqual((throttle, brake), (0., .3))

    def test_integral_is_cleared_by_each_safety_boundary(self):
        for fault in ('invalid', 'stationary', 'regression', 'gap', 'reset'):
            c = Controller(preset='carla', longitudinal_mode='pi')
            for tick in range(10):
                self.tick(c, tick*.05, 1., .8)
            self.assertGreater(c.longitudinal_pi.integral, 0.)
            if fault == 'invalid':
                output = c.step(.5, float('nan'), 0.)
            elif fault == 'stationary':
                output = self.tick(c, .5, 0., 0.)
            elif fault == 'regression':
                output = c.step(.4, .8, 0.)
            elif fault == 'gap':
                output = self.tick(c, .7, 1., .8)
                self.assertEqual(c.diagnostics['reason'], 'motion_gap')
            else:
                c.reset()
                output = c._last_control
            self.assertEqual(c.longitudinal_pi.integral, 0., fault)
            self.assertEqual((output[0], output[2]), (0., 1.), fault)

    def test_vendor_outputs_equal_frozen_replay(self):
        fixture = json.loads(GOLDEN.read_text())
        self.assertEqual(fixture['protocol']['seed'], 8231)
        self.assertEqual(fixture['protocol']['ticks_per_preset'], 600)
        for preset in ('carla', 'tcp', 'pursuit'):
            controller = Controller(preset=preset)
            expected = fixture['expected'][preset]
            self.assertEqual(len(expected['controls']), 600)
            self.assertEqual(len(expected['reasons']), 600)
            for tick, now, path, speed, yaw in vendor_replay_inputs():
                if path is not None:
                    controller.update(path, now)
                output = controller.step(now, speed, yaw)
                self.assertEqual(list(output), expected['controls'][tick], (preset, tick))
                self.assertEqual(controller.diagnostics['reason'], expected['reasons'][tick], (preset, tick))
                if tick % 23 == 0:
                    self.assertEqual(list(controller.step(now, speed, yaw)), expected['controls'][tick])
                if tick == 400:
                    controller.reset()


if __name__ == '__main__':
    unittest.main()
