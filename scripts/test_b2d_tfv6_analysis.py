"""Route-level resampling and canonical case discovery."""

import json
from pathlib import Path
import tempfile
import unittest

from b2d_tfv6_analyze import bootstrap, collect_done_items


class ClusterBootstrapTests(unittest.TestCase):
    def test_route_clusters_preserve_seed_replicates(self):
        paired = [{"comparison": "C-B", "route": route, "ds_diff": value}
                  for route, value in (("1", -10.), ("2", 10.))
                  for _ in range(3)]
        result = bootstrap(paired, "C-B")
        self.assertEqual(result["n_routes"], 2)
        self.assertEqual(result["n_pairs"], 6)
        self.assertEqual(result["mean"], 0.)
        self.assertEqual((result["ci_low"], result["ci_high"]), (-10., 10.))


class CaseDiscoveryTests(unittest.TestCase):
    def test_voided_done_with_stale_live_pointer_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "level1"
            case = root / "cases/1/route-2091/seed-0/C"
            case.mkdir(parents=True)
            live = dict(level="1", route="2091", seed=0, arm="C", attempt=1,
                        run_dir=str(case / "attempt-1/run"))
            (case / "done.json").write_text(json.dumps(live))
            voided = root / "aborted/signed-speed-bug/cases/1/route-2091/seed-0/C"
            voided.mkdir(parents=True)
            (voided / "done.json").write_text(json.dumps(live))
            self.assertEqual(collect_done_items([root]), [live])

    def test_selected_attempt_mismatch_and_duplicate_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "level1"
            case = root / "cases/1/route-2091/seed-0/C"
            case.mkdir(parents=True)
            path = case / "done.json"
            live = dict(level="1", route="2091", seed=0, arm="C", attempt=1,
                        run_dir=str(case / "attempt-1/run"))
            path.write_text(json.dumps({**live, "run_dir": str(case / "attempt-2/run")}))
            with self.assertRaisesRegex(ValueError, "selected attempt"):
                collect_done_items([root])
            path.write_text(json.dumps(live))
            with self.assertRaisesRegex(ValueError, "Duplicate case key"):
                collect_done_items([root], [path])


if __name__ == "__main__":
    unittest.main()
