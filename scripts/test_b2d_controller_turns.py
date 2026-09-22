"""Frozen-default equivalence and bounded max-lookahead candidate contracts."""
from functools import partial
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from b2d_controller import Controller
import b2d_controller_selftest as analytic


class TurnLookaheadTests(unittest.TestCase):
    def test_default_and_explicit_half_second_match_frozen_source_exactly(self):
        fixture = json.loads((Path(__file__).parent / 'testdata/b2d_controller_turn_default_golden.json').read_text())
        self.assertEqual(len(fixture['rows']), 18)
        for row in fixture['rows']:
            for extra in ({}, {'max_lookahead_time_s': .5}):
                c = Controller(row['preset'], lookahead=row['lookahead'],
                               longitudinal_mode=row['longitudinal_mode'], pi_kp=.5, pi_ki=.25, **extra)
                for i, expected in enumerate(row['controls']):
                    now = i * .05
                    speed = 6. + 2. * math.sin(i / 9.)
                    if i % 4 == 0:
                        x = speed * np.arange(1, 21) * .25
                        c.update(np.column_stack((x, .015 * math.cos(i / 7.) * x * x)), now)
                    self.assertEqual(list(c.step(now, speed, .025 * math.sin(i / 8.))), expected)
                    self.assertEqual(c.diagnostics['reason'], row['reasons'][i])

    def test_parameter_rejects_nonfinite_nonpositive_and_nonscalar(self):
        for value in (0., -.1, float('nan'), float('inf'), float('-inf'), None, 'bad', [1., 2.]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Controller(max_lookahead_time_s=value)

    @staticmethod
    def tick(speed, factor, rule='max'):
        c = Controller('pursuit', lookahead=rule, max_lookahead_time_s=factor,
                       longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
        x = speed * np.arange(1, 21) * .25
        c.update(np.column_stack((x, .001 * x ** 3)), 0.)
        control = c.step(0., speed, 0.)
        return c, control

    def test_candidate_changes_eight_mps_geometry_but_preserves_floor(self):
        baseline, _ = self.tick(8., .5)
        candidate, _ = self.tick(8., .375)
        self.assertEqual(baseline.diagnostics['lookahead_m'], 4.)
        self.assertEqual(candidate.diagnostics['lookahead_m'], 3.)
        self.assertNotEqual(baseline.diagnostics['aim_xy'], candidate.diagnostics['aim_xy'])
        self.assertNotEqual(baseline.diagnostics['raw_steer'], candidate.diagnostics['raw_steer'])
        for speed in (2., 5.9, 6.):
            a, ac = self.tick(speed, .5)
            b, bc = self.tick(speed, .375)
            self.assertEqual(a.diagnostics['lookahead_m'], 3.)
            self.assertEqual(b.diagnostics['lookahead_m'], 3.)
            self.assertEqual(ac, bc)
        for rule in ('additive', 'fixed4'):
            a, ac = self.tick(8., .5, rule)
            b, bc = self.tick(8., .375, rule)
            self.assertEqual(ac, bc)
            self.assertEqual(a.diagnostics, b.diagnostics)

    def test_candidate_mirrored_circle_and_delayed_history(self):
        gains = dict(longitudinal_mode='pi', pi_kp=.5, pi_ki=.25)
        with patch.object(analytic, 'Controller', partial(Controller, max_lookahead_time_s=.375)):
            for delay in (0., .3):
                left = analytic.circle('pursuit', speed=8., sign=1., delay=delay, lookahead='max', **gains)
                right = analytic.circle('pursuit', speed=8., sign=-1., delay=delay, lookahead='max', **gains)
                self.assertEqual(left['lateral_rms_m'], right['lateral_rms_m'])
                self.assertEqual(left['reference_speed_rms_mps'], right['reference_speed_rms_mps'])
                self.assertTrue(left['main_pass'])
                self.assertTrue(right['main_pass'])
                self.assertEqual(left['reason_counts'].get('trajectory_behind', 0), 0)


if __name__ == '__main__':
    unittest.main()
