"""Safety tests for the fixed, authorized mutation boundaries."""
import importlib.util
import multiprocessing
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("cx", Path(__file__).with_name("cx_maintenance.py"))
cx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cx)


def writer(path, entries):
    cx.append_unique(Path(path), entries)


class MaintenanceSafety(unittest.TestCase):
    def test_exact_flag_sets(self):
        def v(arm, reasons, fail=None): return {"arm": arm, "verdict": "FLAG", "flag": reasons, "fail": fail or []}
        self.assertEqual(cx.flag_action(v("cl2", [cx.FLAG_BLOCKED, cx.FLAG_MOVES])), "SKIP")
        self.assertEqual(cx.flag_action(v("cl7", [cx.FLAG_MOVES])), "SKIP")
        self.assertEqual(cx.flag_action(v("cl4", [cx.FLAG_BLOCKED, cx.FLAG_DS])), "APPROVED")
        for arm, reasons in (("cl2", [cx.FLAG_DS]), ("cl4", [cx.FLAG_MOVES]),
                             ("cl4", [cx.FLAG_BLOCKED, "DS is 0 on every route"]), ("cl4", [])):
            self.assertNotIn(cx.flag_action(v(arm, reasons)), ("SKIP", "APPROVED"))
        self.assertIsNone(cx.flag_action(v("cl4", [cx.FLAG_DS], ["crash rate > 0.25"])))

    def test_concurrent_skip_preserves_other_entries(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "SKIP"
            p.write_text("other 1\nother 1\n")
            entries = [f"cl2 {s}" for s in range(3)]
            ctx = multiprocessing.get_context("fork")
            ps = [ctx.Process(target=writer, args=(str(p), entries + [f"cl7 {i}"])) for i in range(3)]
            for child in ps: child.start()
            for child in ps: child.join(); self.assertEqual(child.exitcode, 0)
            self.assertEqual(set(p.read_text().splitlines()), {"other 1", *entries, "cl7 0", "cl7 1", "cl7 2"})
            self.assertEqual(len(p.read_text().splitlines()), 7)
            self.assertFalse(cx.append_unique(p, entries))

    def test_t9_never_consolidates_with_live_pid(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            w = cx.Watch(td)
            with patch.object(cx, "alive", return_value=True), patch.object(w, "_consolidate") as command:
                w.alpamayo()
                command.assert_not_called()

    def test_t11_unknown_stage_does_not_delete(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            p = data / "runs/sched/pull_forward/q4b/provenance.txt"
            p.parent.mkdir(parents=True); p.write_text("bad provenance\n")
            cache = data / "processed/carla_p5v1_ba/nq3_q4b_navsim_protocol/cinque.npz"
            cache.parent.mkdir(parents=True); cache.write_bytes(b"keep")
            cx.Watch(data).provenance()
            self.assertEqual(cache.read_bytes(), b"keep")

    def test_t11_active_mismatch_waits_preserves_cache(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            d = data / "runs/nq3/d"; d.mkdir(parents=True); (d / "current").write_text("q4a-fit")
            p = data / "runs/sched/pull_forward/q4b/provenance.txt"
            p.parent.mkdir(parents=True); p.write_text("0" * 32 + "  scripts/nq3_d/q4b.sh\n")
            cache = data / "processed/carla_p5v1_ba/nq3_q4b_navsim_protocol/cinque.npz"
            cache.parent.mkdir(parents=True); cache.write_bytes(b"keep")
            w = cx.Watch(data); w.provenance()
            self.assertEqual(cache.read_bytes(), b"keep")
            self.assertIn("WAIT", w.state["T11:q4b"]["action"])

    def test_stale_error_cannot_override_live_writer(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            data = Path(td)
            producer = data / "runs/sched/pull_forward/q4b"
            producer.mkdir(parents=True); (producer / "DONE").touch(); (producer / "pgid").write_text("42")
            step = data / "runs/nq3/d/q4a-fit"; step.mkdir(parents=True); (step / "pid").write_text("55")
            (step.parent / "ERROR").touch()
            with patch.object(cx, "executing_processes", return_value=[(43, 42, [b"python", b"producer.py"])]):
                self.assertFalse(cx.cache_writers_stopped(data, "q4b", "q4a-fit"))
            with patch.object(cx, "executing_processes", return_value=[(999, 999, [b"bash", b"scripts/nq3_d.sh"])]):
                self.assertFalse(cx.cache_writers_stopped(data, "q4b", "q4a-fit"))
            with patch.object(cx, "executing_processes", return_value=[]):
                self.assertTrue(cx.cache_writers_stopped(data, "q4b", "q4a-fit"))


if __name__ == "__main__": unittest.main()
