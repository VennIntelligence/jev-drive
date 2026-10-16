"""lib/route_desire.py on synthetic route memories: python -m unittest tests.test_route_desire -v (numpy only)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import route_desire as RD  # noqa: E402


def feats(fn, yaw=0.0, dy=0.0):
    """Route path fn(s) -> (X, Y, H) in a road frame, seen from an ego at lateral dy with yaw -> (200,) rmem features."""
    X, Y, H = fn(RD.OFFS)
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y = c * X - s * (Y - dy), s * X + c * (Y - dy)
    F = np.zeros((40, 5), np.float32)
    F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4] = 1, x / 20, np.clip(y / 20, -4, 4), np.cos(H - yaw), np.sin(H - yaw)
    return F.reshape(-1)


def lc(s0, s1, L):
    def f(s):
        u = np.clip((s - s0) / (s1 - s0), 0, 1)
        Y = L * (3 * u ** 2 - 2 * u ** 3)
        return s.astype(float), Y, np.arctan(np.gradient(Y, s))
    return f


def curve(R):
    return lambda s: (R * np.sin(s / R), R * (1 - np.cos(s / R)), s / R)


def junction(d, L, s0, s1, side=1, R=10.0):
    def f(s):
        X, Y, H = lc(s0, s1, L)(s)
        m = s > d
        th = np.minimum((s[m] - d) / R, np.pi / 2)
        X[m], Y[m], H[m] = d + R * np.sin(th), L + side * R * (1 - np.cos(th)), side * th
        return X, Y, H
    return f


class TestIntent(unittest.TestCase):
    def side(self, fn, **k):
        return int(RD.intent(feats(fn, **k))["side"])

    def test_lane_changes(self):
        self.assertEqual(self.side(lc(15, 35, 3.5)), 1)
        self.assertEqual(self.side(lc(15, 35, -3.5)), -1)
        self.assertEqual(self.side(lc(15, 35, 3.5), yaw=np.radians(3)), 1)

    def test_no_lane_change(self):
        for fn in (lc(0, 1, 0), curve(200), curve(100), junction(40, 0.0, 10, 30)):
            self.assertEqual(self.side(fn), 0)
        self.assertEqual(self.side(lc(0, 1, 0), yaw=np.radians(3)), 0)       # the ego's yaw is no lane change
        self.assertEqual(self.side(lc(0, 1, 0), dy=0.8), 0)

    def test_turn_lane_entry(self):
        r = RD.intent(feats(junction(50, 3.5, 15, 35)))                         # route moves a lane left, then turns 50 m ahead
        self.assertEqual(int(r["side"]), 1)
        self.assertTrue(np.isfinite(r["d_turn"]) and r["a"] < r["d_turn"])
        self.assertEqual(self.side(junction(30, 3.5, 5, 20)), 1)
        r = RD.intent(feats(junction(8, 3.5, -6, 2)))                           # inside the mouth: blocked
        self.assertEqual(int(r["side"]), 0)
        self.assertTrue(bool(r["blocked"]))

    def test_state_machine_and_queue(self):
        rd, seen = RD.RouteDesire(v_min=5.0), []
        for k in range(120):
            t = 0.1 * k
            dy = float(np.clip((t - 1.0) / 5 * 3.5, 0, 3.5))
            d = rd.step(t, 10.0, feats(lc(15 - 10 * t, 35 - 10 * t, 3.5) if t < 2 else lc(-100, -99, 3.5), dy=dy))
            seen.append((d, rd.state))
        self.assertEqual(seen[0], (3, "starting"))
        self.assertIn("finishing", [s for _, s in seen])
        self.assertEqual(seen[-1], (0, "off"))
        self.assertEqual(rd.why, "done")
        self.assertEqual(RD.RouteDesire(v_min=8.9).step(0.0, 5.0, feats(lc(15, 35, 3.5))), 0)   # the speed gate
        q = RD.desire_queue(4, 1.0, 1.45)
        self.assertEqual(np.argwhere(q).tolist(), [[0, 30, 4]])
        self.assertFalse(RD.desire_queue(4, 1.0, 7.7).any())
        self.assertFalse(RD.desire_queue(0, 1.0, 1.0).any())
        self.assertEqual([RD.block_of(0.0, a) for a in (0.0, 0.4, 1.0, 2.0)], [32, 30, 27, 22])   # dc.py's open-loop ages

    def test_memory_gap(self):
        def gapped(f):
            f = f.reshape(40, 5).copy()
            f[RD.OFFS < 42] = 0
            return f.reshape(-1)
        self.assertEqual(int(RD.intent(gapped(feats(lc(10, 30, 3.5))))["side"]), 1)
        self.assertEqual(int(RD.intent(gapped(feats(lc(0, 1, 0), yaw=np.radians(3))))["side"]), 0)
        self.assertEqual(int(RD.intent(gapped(feats(lc(10, 30, 3.5), yaw=np.radians(8))))["why"]), 2)    # ego off the road direction

    def test_batch_matches_rows(self):
        F = np.stack([feats(lc(15, 35, 3.5)), feats(curve(150)), feats(junction(50, -3.5, 15, 35, side=-1))])
        r = RD.intent(F)
        self.assertEqual(r["side"].tolist(), [int(RD.intent(f)["side"]) for f in F])
        self.assertEqual(r["side"].tolist(), [1, 0, -1])


if __name__ == "__main__":
    unittest.main()
