"""lib/parity_hugsim.ego_inputs: the `zero_acc` probe option zeroes only ax / ay of the ego features and is off by default.
python -m unittest tests.test_parity_hugsim -v"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "lib"), str(REPO / "scripts")]
import parity_hugsim as PH  # noqa: E402


class ZeroAcc(unittest.TestCase):
    def setUp(self):
        t = np.arange(12) * 0.25
        pos = np.stack([np.zeros(12), 5.0 * t], -1)                   # straight along +z at 5 m/s
        self.hist = SimpleNamespace(t=list(t), pos=list(pos), th=[0.0] * 12)
        self.info = {"ego_velo": [5.0], "accelerate": [1.3], "command": [2]}

    def test_default_unchanged_and_zero_only_ax(self):
        e0, p0 = PH.ego_inputs(self.hist, self.info, 1.0, 1.25)
        e1, p1 = PH.ego_inputs(self.hist, self.info, 1.0, 1.25, zero_acc=True)
        self.assertAlmostEqual(float(e0[6]), 1.3 * 1.25 ** 2 / 3.0, places=5)
        self.assertEqual(float(e1[6]), 0.0)
        self.assertEqual(float(e1[7]), 0.0)
        keep = np.ones(e0.shape, bool)
        keep[6:8] = False
        np.testing.assert_array_equal(e0[keep], e1[keep])             # speed, command, pose history identical
        np.testing.assert_array_equal(p0, p1)
        e2, _ = PH.ego_inputs(self.hist, self.info, 1.0, 1.25, False)
        np.testing.assert_array_equal(e0, e2)


class StatelessInputs(unittest.TestCase):
    """core_inputs on the real clock (agent opt `stateless`, frame_ab arm `b`): key count through the cold start, undilated speed and poses."""

    def test_keys_and_clock(self):
        t = np.arange(12) * 0.25
        st = np.stack([t, np.zeros(12), 5.0 * t, np.zeros(12), np.full(12, 5.0), np.full(12, 0.4), np.full(12, 2.0)], -1)
        img = {k: np.full((2, 6, 4, 4), k, np.uint8) for k in range(12)}
        for n, m in ((0, 1), (1, 1), (2, 2), (3, 2), (4, 3), (5, 3), (6, 4), (11, 4)):
            kw, _ = PH.core_inputs(st, img, n, PH.M2_BACK, 1.0, 1.0)
            self.assertEqual(len(kw["keys"]), m)
            self.assertEqual([int(k[0, 0, 0, 0]) for k in kw["keys"]], [n - b for b in PH.M2_BACK if n - b >= 0])
        np.testing.assert_allclose(kw["pose"][:, 0], [-7.5, -5.0, -2.5, 0.0], atol=1e-9)      # 5 m/s, keys 0.5 s apart
        np.testing.assert_allclose(kw["vel"][:, 0], 5.0)
        np.testing.assert_allclose(kw["acc"], [0.4, 0.0])
        kd, _ = PH.core_inputs(st, img, 11, PH.M1_BACK, 1.25, 1.0)                              # the dilated bridge arm: speed x 1.25
        np.testing.assert_allclose(kd["vel"][:, 0], 6.25)


if __name__ == "__main__":
    unittest.main()
