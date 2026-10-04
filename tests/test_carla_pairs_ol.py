"""carla_pairs_render_ol.build_idx / pack equal the closed-loop harness (OpenpilotModel.pack, frames.get_warp_matrix)."""
import importlib.util, sys, types, unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "experiments/op_route_cmd/scripts")]


class T(unittest.TestCase):
    def test_pack_matches_harness(self):
        try:
            import zeroshot_policy_server as ZP
            from jevdrive.openpilot import frames as opf
        except ImportError as e:
            self.skipTest(str(e))
        import carla_pairs_render_ol as OL
        idx = OL.build_idx()
        w, h = OL.SENSOR_WH
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        stub = types.SimpleNamespace(opf=opf, idx={k: (v[0], v[1]) for k, v in idx.items()})
        for name in ("road", "wide"):
            ref = ZP.OpenpilotModel.pack(stub, img, name)
            self.assertTrue(np.array_equal(ref, OL.pack(img, idx[name])), name)
            M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, OL.FOCAL[name]), name == "wide")
            self.assertTrue(np.array_equal(opf._nn_index(M, (512, 256), (w, h)), idx[name][0]))


if __name__ == "__main__":
    unittest.main()
