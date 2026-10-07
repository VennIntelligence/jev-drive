"""jevdrive.openpilot.lead_margin: the shared lead standstill margin, and its interface resolution on HUGSIM and NAVSIM.
python -m unittest tests.test_openpilot_lead_margin -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jevdrive.openpilot import interface as IF  # noqa: E402
from jevdrive.openpilot import lead_margin as LM  # noqa: E402

T = np.r_[0.0, 0.5 * np.arange(1, 7)]


def straight(v):
    return np.stack([np.zeros_like(T), v * T], -1)          # HUGSIM axes: x right, y forward


class Rule(unittest.TestCase):
    def test_bias_table(self):
        self.assertEqual(LM.bias(3.0), 2.0)
        self.assertAlmostEqual(LM.bias(8.0), 1.0)
        self.assertEqual(LM.bias(12.0), 0.0)

    def test_untriggered_is_identity(self):
        xy = straight(5.0)
        for lp, lx in ((0.4, 5.0), (None, 5.0), (0.9, None)):
            out, _, info = LM.apply(xy, T, lp, lx, 0.0)
            self.assertIs(out, xy)
            self.assertFalse(info["changed"])

    def test_far_lead_does_not_change(self):
        out, _, info = LM.apply(straight(2.0), T, 0.9, 30.0, 0.0)  # s(3) = 6 m < 30 - 2.5
        self.assertTrue(info["trig"])
        self.assertFalse(info["changed"])

    def test_stopped_lead_caps_and_keeps_path(self):
        xy = straight(4.0)                                      # s(3) = 12 m
        out, _, info = LM.apply(xy, T, 0.9, 8.0, 0.0)           # g = 8 - 1 = 7, s_max = 4.5 m
        s = LM.arclen(out)
        self.assertAlmostEqual(s[-1], 4.5)
        np.testing.assert_allclose(s, np.minimum(LM.arclen(xy), 4.5))
        np.testing.assert_allclose(out[:, 0], 0.0)              # shape kept
        self.assertTrue(np.all(np.diff(s) >= -1e-9))
        self.assertAlmostEqual(info["ds_max"], 7.5)

    def test_close_lead_stops_and_moving_lead_relaxes(self):
        out, _, _ = LM.apply(straight(3.0), T, 0.9, 4.0, 0.0)    # g = 2 < d_min: stand
        np.testing.assert_allclose(out, 0.0)
        out, _, _ = LM.apply(straight(3.0), T, 0.9, 4.0, 2.0)    # s_max(t) = max(0, 2t - 0.5)
        np.testing.assert_allclose(LM.arclen(out), np.minimum(3.0 * T, np.maximum(0.0, 2.0 * T - 0.5)))

    def test_never_adds_motion_and_curved_path(self):
        th = 0.2 * T
        xy = np.stack([10 * (1 - np.cos(th)), 10 * np.sin(th)], -1)
        yaw = th.copy()
        out, (yw,), info = LM.apply(xy, T, 0.8, 6.0, 0.0, extra=(yaw,))
        self.assertTrue(np.all(LM.arclen(out) <= LM.arclen(xy) + 1e-9))
        self.assertTrue(np.all(np.diff(yw) >= -1e-9))
        self.assertTrue(info["changed"])

    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            LM.LeadMargin({"dmin": 1.0})


class Interface(unittest.TestCase):
    def test_hugsim_trick(self):
        base = IF.resolve_hugsim({"op_ctrl": True, "op_ctrl_src": "plan_smooth"}, "opctrl")
        on = IF.resolve_hugsim({"op_ctrl": True, "op_ctrl_src": "plan_smooth", "lead_margin": {}}, "opctrl")
        self.assertNotIn("lead_margin", base["tricks"])
        self.assertIn("lead_margin", on["tricks"])
        IF.record("hugsim", "spec_plan_smooth", on)

    def test_navsim_trick(self):
        self.assertNotIn("lead_margin", IF.resolve_navsim()["tricks"])
        v = IF.resolve_navsim(lead_margin=True)
        self.assertIn("lead_margin", v["tricks"])
        IF.record("navsim", "spec", v)


class Bench(unittest.TestCase):
    def test_lm_option(self):
        from jevdrive.bench import models, navsim
        m = models.resolve("P2H10-F-s0:lm")
        self.assertEqual((m.opt, m.key("navtest"), m.key("hugsim")), ("lm", "P2H10-F-s0@warp_lm", "P2H10-F-s0"))
        self.assertEqual(navsim.adapter(m), "lm")
        self.assertTrue(str(navsim.pred_file(m, "navtest")).endswith("P2H10-F-s0-warp_lm__lm.npz"))
        self.assertEqual(navsim.adapter(models.resolve("P2H10-F-s0")), "base")


if __name__ == "__main__":
    unittest.main()
