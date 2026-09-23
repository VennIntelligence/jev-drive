"""Route-level resampling must keep all seeds of one route together."""

import unittest

from b2d_tfv6_analyze import bootstrap


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


if __name__ == "__main__":
    unittest.main()
