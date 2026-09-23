"""Comparison denominator, missing evidence, pairing and cap semantics; no simulator."""
import json
from pathlib import Path
import tempfile
import unittest

import b2d_controller_compare as compare
from test_b2d_controller_report import fixture, write


class CompareTests(unittest.TestCase):
    def campaign(self, root, presets="carla,tcp,pursuit", seeds="0"):
        route_path = root / "routes.xml"
        root.mkdir(parents=True, exist_ok=True)
        route_path.write_text('<routes><route id="1" town="Town01"/><route id="2" town="Town02"/></routes>')
        write(root / "manifest.json", {"git_commit": "fixture-commit", "config": {
            "routes": str(route_path), "presets": presets, "seeds": seeds}})
        return root

    def test_pending_presets_keep_all_requested_routes_and_no_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.campaign(Path(tmp))
            fixture(root / "carla-seed0", rid="1", completion=100)
            groups = compare.load_campaign(root)
            self.assertEqual(len(groups), 3)
            self.assertEqual([g["summary"]["requested_routes"] for g in groups], [2, 2, 2])
            self.assertEqual(groups[0]["summary"]["completion_mean_observed"], 100)
            self.assertIsNone(groups[0]["summary"]["completion_mean_all_routes"])
            self.assertEqual(groups[1]["summary"]["attempts"], 0)
            self.assertFalse(compare.strongest_references(groups)[0]["ready"])
            self.assertEqual(groups[0]["summary"]["infractions"]["collisions_vehicle"]["missing_routes"], 2)

    def test_vendor_cap_is_terminal_failure_and_best_retry_never_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.campaign(Path(tmp), presets="carla,tcp")
            for preset in ("carla", "tcp"):
                for rid in ("1", "2"):
                    fixture(root / (preset + "-seed0"), rid=rid, completion=100)
            adir = fixture(root / "carla-seed0", rid="1", completion=42)
            raw = json.loads((adir / "results.json").read_text())
            raw["_checkpoint"]["records"][0]["status"] = "Failed - TickRuntime"
            write(adir / "results.json", raw)
            fixture(root / "carla-seed0", rid="1", attempt=2, completion=100)
            groups = compare.load_campaign(root)
            self.assertTrue(groups[0]["summary"]["data_complete"])
            self.assertEqual(groups[0]["summary"]["completion_mean_all_routes"], 71)
            self.assertEqual(groups[0]["summary"]["official_tick_runtime_route_ids"], ["1"])
            self.assertEqual(groups[0]["summary"]["true_completed"], 1)
            self.assertEqual(groups[0]["summary"]["retries"], 1)
            pair = compare.pair_groups(*groups)
            self.assertTrue(pair["complete_pair"])
            self.assertEqual(pair["completion_delta_right_minus_left"], 29)
            reference = compare.strongest_references(groups)[0]
            self.assertTrue(reference["ready"])
            self.assertTrue(reference["strongest_observed_reference"].endswith("tcp-seed0"))

    def test_harness_cap_prevents_complete_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.campaign(Path(tmp), presets="carla,tcp")
            for preset in ("carla", "tcp"):
                for rid in ("1", "2"):
                    fixture(root / (preset + "-seed0"), rid=rid, completion=100, capped=preset == "carla")
            groups = compare.load_campaign(root)
            self.assertFalse(groups[0]["summary"]["data_complete"])
            self.assertFalse(compare.pair_groups(*groups)["complete_pair"])
            self.assertFalse(compare.strongest_references(groups)[0]["ready"])

    def test_g2_preserves_each_source_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / "failed.json", Path(tmp) / "fixed.json"]
            for path, passed in zip(paths, (False, True)):
                write(path, [{"preset": "pursuit", "route_id": "1", "gates": {"endpoint": passed}, "gate_pass": passed}])
            evidence = compare.read_g2(paths)
            self.assertEqual(len(evidence), 2)
            self.assertFalse(evidence[0]["by_preset"]["pursuit"]["all_recorded_gates_pass"])
            self.assertTrue(evidence[1]["by_preset"]["pursuit"]["all_recorded_gates_pass"])

    def test_snapshot_outputs_finite_json_and_all_attempts_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.campaign(Path(tmp) / "input")
            fixture(root / "carla-seed0", rid="1", attempt=1, completion=0, status="crashed")
            fixture(root / "carla-seed0", rid="1", attempt=2, completion=50)
            groups = compare.load_campaign(root)
            out = Path(tmp) / "output"
            compare.write_outputs(out, groups, [], [], compare.strongest_references(groups))
            data = json.loads((out / "comparison.json").read_text())
            self.assertEqual(len(data["groups"]), 3)
            self.assertEqual(len((out / "routes.csv").read_text().splitlines()), 7)
            self.assertEqual(len((out / "attempts.csv").read_text().splitlines()), 8)
            self.assertIn("origin-to-first-point", (out / "comparison.md").read_text())


if __name__ == "__main__":
    unittest.main()
