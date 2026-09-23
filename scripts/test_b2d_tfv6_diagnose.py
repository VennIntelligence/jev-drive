"""Focused invariants for the frozen W2 log diagnosis."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('diagnose', Path(__file__).with_name('b2d_tfv6_diagnose.py'))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


class DiagnosisTest(unittest.TestCase):
    def test_log_decomposition_adds_to_official_ds_difference(self):
        # RC gains and a vehicle collision penalty cancel partly.
        base = {'ds': 80.0, 'logs': diagnose.score_logs(100, 80, {'stop_infraction': ['x']})}
        arm = {'ds': 90.0, 'logs': diagnose.score_logs(150, 90, {'collisions_vehicle': ['x']})}
        parts = diagnose.decompose_ds_pair(arm, base)
        self.assertAlmostEqual(sum(parts.values()), 10.0)
        self.assertGreater(parts['rc'], 0)
        self.assertLess(parts['collisions_vehicle'], 0)
        self.assertGreater(parts['stop_infraction'], 0)
        self.assertIsNone(diagnose.score_logs(0, 0, {}))
        self.assertIsNone(diagnose.decompose_ds_pair({'ds': 0, 'logs': None}, base))

    def test_first_divergence_uses_first_strict_threshold_on_common_steps(self):
        a_steps = np.array([0, 1, 2, 3, 4])
        b_steps = np.array([1, 2, 3, 4])
        a_xy = np.array([[0, 0], [0, 0], [1, 0], [2.01, 0], [4, 0]], float)
        b_xy = np.array([[0, 0], [0, 0], [1, 0], [2, 0]], float)
        a_speed = np.array([0, 0, 0, 1, 2], float)
        b_speed = np.array([0, 0, 1, 0], float)
        result = diagnose.first_divergence(a_steps, a_xy, a_speed, b_steps, b_xy, b_speed)
        self.assertEqual(result['step'], 3)
        self.assertEqual(result['index_a'], 3)
        self.assertEqual(result['index_b'], 2)
        self.assertIsNone(diagnose.first_divergence(a_steps[:2], a_xy[:2], a_speed[:2],
                                                    b_steps[-2:], b_xy[-2:], b_speed[-2:]))


if __name__ == '__main__':
    unittest.main()
