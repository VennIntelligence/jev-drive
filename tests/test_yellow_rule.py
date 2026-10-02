"""Go / stop decision of vred3 on a grid of (distance, speed, yellow age): no CARLA.

    python -m unittest tests.test_yellow_rule -v
"""
import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import yellow_rule as Y  # noqa: E402

VC = 8.0                                                  # normal cruise speed of the agent, m/s
SPEEDS = [0.5, 2.0, 3.5, 4.5, 6.0, 8.0]
D_STOP = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0]
GAP = 4.0                                                 # stop line to junction entrance, m (3.2 - 6.6 m on the batch routes)
AGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def dec(d_stop, v, age, gap=GAP):
    return Y.decide(d_stop, d_stop + gap + 0.5, v, Y.remaining_yellow(age), VC)    # d_stop here: bumper to stop line - 0.5


class Pieces(unittest.TestCase):
    def test_remaining(self):
        self.assertAlmostEqual(Y.remaining_yellow(0.5), 2.0)
        self.assertAlmostEqual(Y.remaining_yellow(2.0), 0.5)
        self.assertEqual(Y.remaining_yellow(2.5), -1.0)           # last green answer too old
        self.assertEqual(Y.remaining_yellow(None), -1.0)

    def test_tail(self):
        self.assertAlmostEqual(Y.TAIL_M, 5.9016, places=3)

    def test_stop_distance(self):
        self.assertEqual(Y.stop_distance(0.0), 0.0)
        self.assertAlmostEqual(Y.stop_distance(6.0), 6.0 * 0.5 + 36 / 6.0)
        d = [Y.stop_distance(v) for v in SPEEDS]
        self.assertEqual(d, sorted(d))

    def test_clear_time(self):
        self.assertEqual(Y.clear_time(0.0, 5.0, VC), 0.0)
        self.assertAlmostEqual(Y.clear_time(1.5, 5.0, VC), 0.3)                       # inside the latency
        self.assertAlmostEqual(Y.clear_time(40.0, 8.0, VC), 0.3 + (40 - 2.4) / 8.0)   # at cruise: constant speed
        a = Y.clear_time(10.0, 4.0, VC)
        self.assertGreater(a, 10.0 / VC)
        self.assertLess(a, 10.0 / 4.0)                                                # accelerating beats the constant speed
        t = [Y.clear_time(d, 4.0, VC) for d in (2, 5, 10, 20, 40)]
        self.assertEqual(t, sorted(t))
        s = [Y.clear_time(15.0, v, VC) for v in SPEEDS]
        self.assertEqual(s, sorted(s, reverse=True))                                  # faster start, shorter time


class Grid(unittest.TestCase):
    def test_comfortable_stop_always_wins(self):
        for v, d, age in itertools.product(SPEEDS, D_STOP, AGES):
            if d >= Y.stop_distance(v):
                self.assertEqual(dec(d, v, age), "stop", (v, d, age))

    def test_go_needs_time_and_distance(self):
        for v, d, age in itertools.product(SPEEDS, D_STOP, AGES):
            r = dec(d, v, age)
            if r == "go":
                rem = Y.remaining_yellow(age)
                self.assertGreater(rem, 0)
                self.assertLess(d, Y.stop_distance(v))
                self.assertLessEqual(Y.clear_time(d + GAP + 0.5 + Y.TAIL_M + 0.5, v, VC), rem)
            if r == "stop_hard":
                rem = Y.remaining_yellow(age)
                self.assertTrue(rem <= 0 or Y.clear_time(d + GAP + 0.5 + Y.TAIL_M + 0.5, v, VC) > rem)

    def test_monotone_in_age_and_distance(self):
        order = {"stop_hard": 0, "go": 1, "stop": 1}                                  # go and stop are both "resolved"; stop_hard is the failure
        for v, d in itertools.product(SPEEDS, D_STOP):
            rs = [dec(d, v, a) for a in AGES]
            ok = [r != "stop_hard" for r in rs]
            self.assertEqual(ok, sorted(ok, reverse=True), (v, d, rs))                # a later look at the same state never resolves more
        for v, age in itertools.product(SPEEDS, AGES):
            ok = [dec(d, v, age) != "stop_hard" for d in D_STOP]
            # more room to stop never turns a resolved case into stop_hard beyond the point where the stop becomes comfortable
            first_stop = next((i for i, d in enumerate(D_STOP) if d >= Y.stop_distance(v)), len(D_STOP))
            self.assertTrue(all(ok[first_stop:]), (v, age))
        self.assertEqual(order["go"], order["stop"])

    def test_known_cases(self):
        # 5 m / s, yellow just seen (age 0.5 s), 4.5 m to the stop target: not comfortably stoppable (6.7 m), 13.1 m to clear in 2.0 s remaining: no
        self.assertEqual(dec(4.5, 5.0, 0.5), "stop_hard")
        # slow approach under the R1 cap, 2 m to the target: stop needs 3.4 m at 4.5 m/s? 4.5 * 0.5 + 20.25 / 6 = 5.6 m: not stoppable; clear 2 + 4.5 + 5.9 + .5 = 12.9 m
        self.assertEqual(dec(2.0, 4.5, 0.5), "stop_hard")
        # 8 m to the target at 4.5 m/s: comfortable stop (5.6 m)
        self.assertEqual(dec(8.0, 4.5, 0.5), "stop")
        # a car 0.4 m before the line at 8 m/s with 1 line gap: nothing works in 2 s remaining? 0.4+1+0.5+5.9+0.5 = 8.3 m in 0.3 + ... = ~1.2 s: go
        self.assertEqual(Y.decide(0.4, 1.9, 8.0, 2.0, VC), "go")
        # a stopped car (v = 0) at the line: stop (distance 0 <= d_stop)
        self.assertEqual(Y.decide(0.0, 4.5, 0.0, 2.0, VC), "stop")
        # already red (remaining <= 0): never go
        self.assertEqual(Y.decide(0.4, 1.9, 8.0, 0.0, VC), "stop_hard")


if __name__ == "__main__":
    unittest.main()
