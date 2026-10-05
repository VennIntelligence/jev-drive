"""rft.act_target / act_target_path: modeld's action convention (decision 130): action[0] = -kappa_inst(t + 0.275 s) * max(1, v_model)^2, gain 1,
right-positive. CPU, needs numpy + torch (op-train env or .venv).  python -m unittest tests.test_rft_act_target -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/op_route_ft/scripts"))
import rft as F  # noqa: E402

T16 = 0.25 * np.arange(1, 17)


def arc_hum(R, v, left=True, straight_m=0.0):
    """Timed target at T16: straight for straight_m, then a circle of radius R, at constant speed v (rear axle, y left)."""
    s = v * T16
    out = []
    for si in s:
        if si <= straight_m:
            out.append((si, 0.0))
        else:
            a = (si - straight_m) / R
            out.append((straight_m + R * np.sin(a), R * (1 - np.cos(a)) * (1 if left else -1)))
    P = np.array(out)
    return np.concatenate([P, np.full((16, 1), v)], 1)


def arc_path(R, left=True, straight_m=0.0, n=80):
    s = np.arange(n, dtype=float)
    a = np.clip(s - straight_m, 0, None) / R
    x = np.where(s <= straight_m, s, straight_m + R * np.sin(a))
    y = np.where(s <= straight_m, 0.0, R * (1 - np.cos(a))) * (1 if left else -1)
    return np.stack([x, y], -1).astype(np.float32), np.ones(n, bool)


class ActTarget(unittest.TestCase):
    def test_arc_gain_one(self):
        for R, v in ((20.0, 5.0), (50.0, 10.0), (8.0, 3.0)):
            a, w = F.act_target(arc_hum(R, v, left=True), v)
            self.assertEqual(w, 1.0)
            self.assertAlmostEqual(a, -v * v / R, delta=0.03 * v * v / R)          # left turn -> negative (right +)
            a, _ = F.act_target(arc_hum(R, v, left=False), v)
            self.assertAlmostEqual(a, v * v / R, delta=0.03 * v * v / R)

    def test_straight_and_slow(self):
        self.assertAlmostEqual(F.act_target(arc_hum(1e9, 6.0), 6.0)[0], 0.0, places=4)
        self.assertEqual(F.act_target(arc_hum(10.0, 0.8), 0.8), (0.0, 0.0))        # below 1 m/s: unsupervised
        self.assertEqual(F.act_target(np.zeros((16, 3)), 3.0), (0.0, 0.0))         # not moving

    def test_v_model_scale(self):
        a, _ = F.act_target(arc_hum(20.0, 8.0), 8.0, v_model=8.0 * 0.87)
        self.assertAlmostEqual(a, -(8.0 * 0.87) ** 2 / 20.0, delta=0.05)

    def test_clip(self):
        a, _ = F.act_target(arc_hum(3.0, 2.0), 2.0)
        self.assertAlmostEqual(a, -F.K_CLIP * 4.0, places=6)

    def test_path_inst(self):
        tp, tm = arc_path(10.0, left=False)
        a, w = F.act_target_path(tp, tm, 2.0)
        self.assertAlmostEqual(a, 4.0 / 10.0, delta=0.02)
        tp, tm = arc_path(10.0, straight_m=6.0)
        self.assertAlmostEqual(F.act_target_path(tp, tm, 2.0)[0], 0.0, places=6)  # 6 m before the arc: still straight at t + 0.275
        self.assertEqual(F.act_target_path(tp, tm, 0.5), (0.0, 0.0))
        a, w = F.act_target_path(*arc_path(10.0), 0.5, vmin=0.3)
        self.assertAlmostEqual(a, -0.1, delta=0.005)                               # max(1, v)^2 = 1 below 1 m/s


if __name__ == "__main__":
    unittest.main()
