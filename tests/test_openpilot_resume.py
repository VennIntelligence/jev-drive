"""jevdrive.openpilot.resume: the shared emulated driver resume, and its interface resolution on HUGSIM and B2D.
python -m unittest tests.test_openpilot_resume -v"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jevdrive.openpilot import interface as IF  # noqa: E402
from jevdrive.openpilot import resume as R  # noqa: E402

DT = 0.25


def run(rule, n, v=0.0, lp=0.1, lx=40.0, s1=0.0, t0=0.0):
    """n steps at constant inputs -> list of (active, why)."""
    return [rule.step(t0 + DT * k, v, lp, lx, s1) for k in range(n)]


class Rule(unittest.TestCase):
    def test_launch_dist(self):
        self.assertAlmostEqual(R.launch_dist(0.0, 1.2, 2.5, 1.0), 0.6)
        tc = 2.5 / 1.2
        self.assertAlmostEqual(R.launch_dist(0.0, 1.2, 2.5, 3.0), 0.5 * 1.2 * tc * tc + 2.5 * (3.0 - tc))
        self.assertAlmostEqual(R.launch_dist(3.0, 1.2, 2.5, 2.0), 6.0)

    def test_fires_after_t_stand_when_clear(self):
        r = R.ResumeRule()
        out = run(r, 41)                                  # 0 .. 10.0 s standing
        self.assertTrue(all(not a for a, _ in out[:40]))
        self.assertEqual(out[40], (True, "fire"))
        self.assertEqual(r.n, 1)

    def test_moving_resets_and_lead_blocks(self):
        r = R.ResumeRule()
        run(r, 30)
        self.assertEqual(r.step(30 * DT, 0.5, 0.1, 40.0, 0.0), (False, "moving"))
        out = run(r, 41, t0=31 * DT)                      # the standstill restarted at 7.75 s: fires at 17.75 s, not before
        self.assertEqual([k for k, (a, _) in enumerate(out) if a], [40])
        r2 = R.ResumeRule()
        out = run(r2, 80, lp=0.9, lx=8.0)                 # a close lead the whole time: never fires
        self.assertFalse(any(a for a, _ in out))
        self.assertEqual(out[-1][1], "lead")
        self.assertTrue(r2.clear(0.9, 20.0) and r2.clear(0.4, 5.0) and r2.clear(None, None))

    def test_lead_must_be_clear_for_t_clear(self):
        r = R.ResumeRule()
        run(r, 44, lp=0.9, lx=8.0)                        # blocked through 10.75 s
        out = run(r, 5, t0=44 * DT)                       # last blocked sample 10.75 s: fires at 11.75 s
        self.assertEqual([a for a, _ in out], [False, False, False, True, True])

    def test_handbacks_and_cooldown(self):
        r = R.ResumeRule()
        run(r, 41)
        self.assertEqual(r.step(10.25, 1.0, 0.1, 40.0, 0.0), (True, "launch"))
        self.assertEqual(r.step(10.5, 2.6, 0.1, 40.0, 0.0), (False, "speed"))
        self.assertEqual(r.step(10.75, 0.0, 0.1, 40.0, 0.0), (False, "cool"))
        out = run(r, 80, t0=11.0)                         # cooldown to 20.5 s, then 10 s of standstill: fires at 30.5 s
        k = [i for i, (a, _) in enumerate(out) if a][0]
        self.assertAlmostEqual(11.0 + DT * k, 30.5)
        for kw, why in (({"lp": 0.9, "lx": 5.0}, "lead_abort"), ({"s1": 5.0}, "plan")):
            r = R.ResumeRule()
            run(r, 41)
            a = dict(lp=0.1, lx=40.0, s1=0.0)
            a.update(kw)
            self.assertEqual(r.step(10.25, 0.5, a["lp"], a["lx"], a["s1"]), (False, why))
        r = R.ResumeRule()
        run(r, 41)
        out = run(r, 30, v=0.05, t0=10.25)                # blocked car: hand back after t_max
        self.assertIn((False, "timeout"), out)

    def test_params_and_plan_s1(self):
        with self.assertRaises(ValueError):
            R.ResumeRule(t_wait=3)
        self.assertEqual(R.ResumeRule(t_stand=3).p["t_stand"], 3)
        self.assertAlmostEqual(R.plan_s1_from_points([(0, 1.0), (0, 3.0), (0, 6.0)], 0.5), 3.0)
        self.assertAlmostEqual(R.plan_s1_from_points([(0, 1.0), (0, 3.0), (0, 6.0)], 0.4), 4.5)
        self.assertIsNone(IF.resume_rule(None))
        self.assertEqual(IF.resume_rule({}).p, R.DEFAULT)


class Interface(unittest.TestCase):
    def test_hugsim(self):
        p = IF.HUGSIM_PRESETS["spec"]
        base = IF.resolve_hugsim(p["opts"], p["controller"], "nuscenes", p["env"]["OP_CTRL"])
        on = IF.resolve_hugsim(dict(p["opts"], resume={}), p["controller"], "nuscenes", p["env"]["OP_CTRL"])
        self.assertEqual((base["lon.resume"], on["lon.resume"]), ("none", "rule"))
        dev = {d.key: d for d in IF.deviations("hugsim", on)}
        self.assertEqual(dev["lon.resume"].privilege, "real-car")
        self.assertEqual(IF.deviations("navsim", IF.resolve_navsim())[0].key != "lon.resume", True)

    def test_b2d(self):
        sys.path.insert(0, str(REPO / "tests"))
        from test_openpilot_interface import OLD_TOP, op_arb
        A = op_arb()
        _, cfg = A.resolve_config(dict(OLD_TOP, arb={"preset": "spec"}))
        self.assertEqual(IF.b2d_values(cfg, env={})["lon.resume"], "timer")
        _, cfg = A.resolve_config(dict(OLD_TOP, arb={"preset": "spec", "resume_rule": {}}))
        v = IF.b2d_values(cfg, env={})
        self.assertEqual(v["lon.resume"], "timer+rule")
        self.assertIn("lon.resume", {d.key for d in IF.deviations("b2d", v)})
        _, cfg = A.resolve_config(dict(OLD_TOP, arb={"mode": "native"}))
        self.assertEqual(IF.b2d_values(cfg, env={})["lon.resume"], "none")


if __name__ == "__main__":
    unittest.main()
