"""jevdrive.openpilot.interface: the spec, per-board resolution, declared deviations, and the B2D / HUGSIM presets.
python -m unittest tests.test_openpilot_interface -v   (no GPU, no CARLA: carla / leaderboard / srunner are stubbed)"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jevdrive.openpilot import interface as IF  # noqa: E402

# the `drive` arb string of experiments/op_closed_loop/archive/op_arb.sh before 2026-10-05 (LAT_EXEC, RESUME_S unset)
OLD_DRIVE_ARB = {"mode": "drive", "lat": "op", "lat_exec": "curv", "lon": "op", "hold": "intent", "release": "planx",
                 "release_th": 2.0, "release_s": 1.0, "latch_max_s": 5, "coast_v": 2.5}
OLD_TOP = {"model": "cinque", "socket": "s", "plan_every": 1, "ctl_every": 4, "op_camera_tick": 0.05, "plan_origin": "rear",
           "warmup_s": 5.0, "desire": True, "controller": "fixed", "controller_preset": "pursuit", "controller_config": "p",
           "seed": 0, "dump_every": 0}


def op_arb():
    """lib/op_arb_agent.py with the CARLA-side modules stubbed."""
    for name in ("carla", "leaderboard", "leaderboard.autoagents", "leaderboard.autoagents.autonomous_agent", "srunner",
                 "srunner.scenariomanager", "srunner.scenariomanager.timer"):
        sys.modules.setdefault(name, types.ModuleType(name))
    aa = sys.modules["leaderboard.autoagents.autonomous_agent"]
    aa.AutonomousAgent = getattr(aa, "AutonomousAgent", type("AutonomousAgent", (), {}))
    aa.Track = getattr(aa, "Track", types.SimpleNamespace(SENSORS="SENSORS"))
    sys.modules["srunner.scenariomanager.timer"].GameTime = object
    sys.path.insert(0, str(REPO / "lib"))
    import op_arb_agent
    return op_arb_agent


class Spec(unittest.TestCase):
    def test_spec_has_no_deviation(self):
        for board in IF.DECLARED:
            self.assertEqual(IF.deviations(board, {}), [])

    def test_unknown_key_and_undeclared(self):
        with self.assertRaises(IF.InterfaceError):
            IF.deviations("wod", {"rig.height": 1.2})
        with self.assertRaises(IF.InterfaceError):                    # WOD declares no route geometry
            IF.deviations("wod", {"command.route_geometry": "dense-zones"})

    def test_tolerances(self):
        self.assertEqual(IF.deviations("b2d", {"rig.height_m": 1.24, "lateral.delay_s": 0.2}), [])
        self.assertEqual([d.key for d in IF.deviations("b2d", {"rig.height_m": 1.433})], ["rig.height_m"])

    def test_record_write(self):
        rec = IF.record("navsim", "spec", IF.resolve_navsim(), config={"x": 1})
        with tempfile.TemporaryDirectory() as d:
            back = json.loads(Path(IF.write(d, rec)).read_text())
        self.assertEqual(back["board"], "navsim")
        self.assertIn("rig.height_m", [x["key"] for x in back["deviations"]])
        self.assertEqual(back["resolved"]["rig.height_m"], 1.87)
        self.assertNotIn("rig.height_m", [d.key for d in IF.deviations("navsim", IF.resolve_navsim(vcam=1.22))])
        IF.deviations("wod", IF.resolve_wod(long_scale=1.06))


class Hugsim(unittest.TestCase):
    def test_spec_preset_lateral_is_on_spec(self):
        p = IF.HUGSIM_PRESETS["spec"]
        v = IF.resolve_hugsim(p["opts"], p["controller"], "nuscenes", p["env"]["OP_CTRL"])
        keys = {d.key for d in IF.deviations("hugsim", v)}
        self.assertFalse(keys & {"lateral.source", "lateral.exec", "lateral.delay_s", "rig.height_m"})   # 0.25 s / 1.25 = 0.2
        self.assertEqual(v["history.warmup"], "static")
        c = IF.HUGSIM_PRESETS["spec_cold"]
        self.assertEqual(IF.resolve_hugsim(c["opts"], c["controller"])["history.warmup"], "cold")
        self.assertIn("lon.source", keys)
        h = IF.HUGSIM_PRESETS["spec_hold"]
        vh = IF.resolve_hugsim(h["opts"], h["controller"], "waymo", h["env"]["OP_CTRL"])
        self.assertEqual((vh["history.clock"], vh["lateral.delay_s"]), ("hold", 0.2))

    def test_exam_preset_is_legacy(self):
        v = IF.resolve_hugsim({}, "fixed", "pandaset")
        self.assertEqual((v["lateral.source"], v["lateral.exec"], v["history.warmup"]), ("plan", "ilqr", "static"))
        IF.deviations("hugsim", v)
        with self.assertRaises(IF.InterfaceError):                     # the curvature row needs the opctrl tree
            IF.resolve_hugsim({"op_ctrl": True}, "fixed")


class B2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = op_arb()

    def test_drive_preset_is_the_old_shell_string(self):
        name, cfg = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "drive"}))
        self.assertEqual(name, "drive")
        self.assertEqual(cfg, dict(OLD_TOP, arb=OLD_DRIVE_ARB))
        _, cfg2 = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "drive", "lat": "route", "resume": "timer"}))
        self.assertEqual(cfg2["arb"], dict(OLD_DRIVE_ARB, lat="route", resume="timer"))

    def test_explicit_mode_is_untouched_and_default_is_spec(self):
        raw = dict(OLD_TOP, arb={"mode": "native", "twin": True})
        self.assertEqual(self.A.resolve_config(raw), ("explicit", raw))
        name, cfg = self.A.resolve_config(dict(OLD_TOP))
        self.assertEqual((name, cfg["op_ctrl"], cfg["arb"]["resume"]), ("spec", IF.B2D_SPEC_OP_CTRL, "timer"))

    def test_spec_deviations(self):
        _, cfg = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "spec"}))
        v = IF.b2d_values(cfg, env={})
        self.assertEqual((v["lateral.source"], v["lateral.exec"], v["lateral.delay_s"]), ("action", "op-path", 0.2))
        keys = {d.key for d in IF.deviations("b2d", v)}
        self.assertFalse(keys & {"lateral.source", "lateral.exec", "lateral.delay_s", "light.source", "history.rate_hz"})
        self.assertIn("command.route_geometry", keys)
        self.assertEqual(IF.B2D_SPEC_MOUNT, (1.59, 0.0, 1.86))
        self.assertIn("rig.height_m", keys)                             # declared: open-loop-aligned viewpoint, not openpilot's 1.22 m
        self.assertEqual(self.A.PRESETS["spec_bumper122"]["op_mount"], [3.8, 0.0, 1.22])
        self.assertEqual(self.A.PRESETS["spec_windshield143"]["op_mount"][2], 1.433)
        _, nz = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "spec", "zones": False, "div_m": 1e9}))
        self.assertEqual(IF.b2d_values(nz, env={})["command.route_geometry"], "none")

    def test_legacy_drive_values(self):
        _, cfg = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "drive"}))
        v = IF.b2d_values(cfg, env={})
        self.assertEqual((v["lateral.exec"], v["lateral.delay_s"]), ("bicycle", "carla"))
        self.assertEqual(IF.b2d_values(cfg, env={"OP_CTRL": '{"delay": 0.2}'})["lateral.exec"], "op-path")
        self.assertEqual(IF.b2d_values(cfg, env={"VLM_ARM": "vred"})["light.source"], "vlm")
        IF.deviations("b2d", v)

    def test_nored_and_slow_tick_refused(self):
        _, cfg = self.A.resolve_config(dict(OLD_TOP, arb={"preset": "drive", "resume": "nored"}))
        with self.assertRaises(IF.InterfaceError):
            IF.deviations("b2d", IF.b2d_values(cfg, env={}))
        _, cfg = self.A.resolve_config(dict(OLD_TOP, op_camera_tick=0.2, arb={"preset": "drive"}))
        with self.assertRaises(IF.InterfaceError):                     # 5 Hz frames: undeclared on B2D
            IF.deviations("b2d", IF.b2d_values(cfg, env={}))


if __name__ == "__main__":
    unittest.main()
