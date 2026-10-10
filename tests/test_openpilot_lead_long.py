"""jevdrive.openpilot.lead_long: openpilot's lead path (moved from experiments/alpasim/lib/serve_fix.py) and PlanLead, the HUGSIM form.
python -m unittest tests.test_openpilot_lead_long -v   (needs scipy)"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/alpasim/lib")]
from jevdrive.openpilot import interface as IF  # noqa: E402
from jevdrive.openpilot import lead_long as LD  # noqa: E402

T = np.r_[0.0, 0.5 * np.arange(1, 7)] / 1.25               # HUGSIM plan times in the model's clock


def straight(v):
    return np.stack([np.zeros_like(T), v * T], -1)          # HUGSIM axes: x right, y forward


def lead(x, v, p):
    mu = np.zeros((3, 6, 4))
    mu[:, 0, 0], mu[:, 0, 2] = x, v
    return mu, np.full(3, p)


def drive(pl, v, x, vl, p, steps=40, dt=0.2):
    """Closed loop on a straight: the ego takes the limited plan's first-point speed; the lead stands or moves at vl. -> gaps, speeds."""
    gaps, vs = [], []
    for k in range(steps):
        xy, info = pl(straight(max(v, 6.0)), T, v, k * dt, *lead(x + 1.5, vl, p), v)
        v = float(xy[1, 1] / T[1])
        x += (vl - v) * dt
        gaps.append(x), vs.append(v)
    return np.array(gaps), np.array(vs)


class Moved(unittest.TestCase):
    def test_serve_fix_reexports_the_same_objects(self):
        import serve_fix as F
        for k in ("LeadPlanner", "LongMpc", "T_MPC", "safe_distance", "obstacle", "get_accel_from_plan", "ACCEL_MIN", "STOP_DISTANCE", "DT_MDL"):
            self.assertIs(getattr(F, k), getattr(LD, k), k)
        self.assertTrue(all(hasattr(F, k) for k in ("Serve", "path", "place", "join", "T_TRACK", "TG", "K5", "V_MOVING")))


class Plan(unittest.TestCase):
    def test_no_lead_returns_the_plan_itself(self):
        pl, xy = LD.PlanLead(1.5), straight(8.0)
        for k in range(10):
            out, info = pl(xy, T, 8.0, 0.2 * k, *lead(20.0, 0.0, 0.1), 8.0)
            self.assertIs(out, xy)
            self.assertFalse(info["changed"])

    def test_only_slows_and_keeps_the_path(self):
        pl = LD.PlanLead(1.5)
        th = np.linspace(0, 0.6, len(T))
        xy = np.stack([20 * (1 - np.cos(th)), 20 * np.sin(th)], -1)       # a right bend
        s = lambda a: np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))]  # noqa: E731
        n = 0
        for k in range(8):                                    # the unconverged 8-iteration solve differs in the last digits between BLAS builds:
            out, info = pl(xy, T, 8.0, 0.2 * k, *lead(12.0, 0.0, 0.95), 8.0)   # assert the invariants, not the numbers
            n += info["changed"]
            self.assertTrue(np.all(s(out) <= s(xy) + 1e-9))
            self.assertTrue(np.all(np.abs(np.hypot(out[:, 0] - 20, out[:, 1]) - 20) < 0.03))   # on the chords of the 20 m arc (sagitta 0.025 m)
        self.assertGreaterEqual(n, 3)

    def test_stops_behind_a_standing_lead(self):
        gaps, vs = drive(LD.PlanLead(1.5), 8.0, 40.0, 0.0, 0.95, steps=120)
        self.assertGreater(gaps.min(), 2.0)
        self.assertLess(vs[-1], 0.3)

    def test_lifts_when_the_lead_goes(self):
        pl = LD.PlanLead(1.5)
        for k in range(10):
            pl(straight(8.0), T, 2.0, 0.2 * k, *lead(9.0, 0.0, 0.95), 2.0)
        for k in range(10, 40):
            out, info = pl(straight(8.0), T, 8.0, 0.2 * k, *lead(60.0, 20.0, 0.02), 8.0)
        self.assertFalse(info["changed"])

    def test_interface_trick(self):
        on = IF.resolve_hugsim({"op_ctrl": True, "op_lead": {}}, "opctrl")
        off = IF.resolve_hugsim({"op_ctrl": True}, "opctrl")
        self.assertIn("op_lead", on["tricks"])
        self.assertNotIn("op_lead", off["tricks"])


if __name__ == "__main__":
    unittest.main()
