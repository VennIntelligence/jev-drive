"""experiments/b2d_collect/lib/b2dc_frames.py: the collector's packing is bit-identical to the closed-loop policy server's
(scripts/zeroshot_policy_server.OpenpilotModel.pack), and the yuv420p storage round trip is exact.

    python -m unittest tests.test_b2dc_frames -v          (NumPy + scipy; the lossless H.264 check runs when ffmpeg is present)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / "scripts"), str(R / "experiments/b2d_collect/lib")]
import b2dc_frames as F  # noqa: E402


class TestFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.img = {n: rng.integers(0, 256, (1208, 1928, 4), dtype=np.uint8) for n in ("road", "wide")}
        cls.P = F.Packer()

    def test_matches_policy_server(self):
        import zeroshot_policy_server as S
        from jevdrive.openpilot import frames as opf
        fake = SimpleNamespace(opf=opf, idx={n: F.indices(n) for n in ("road", "wide")})
        for n in ("road", "wide"):
            ref = S.OpenpilotModel.pack(fake, self.img[n], n)
            np.testing.assert_array_equal(self.P(self.img[n], n), ref)

    def test_index_build_matches_server(self):
        """The server's own index construction (copied loop in OpenpilotModel.__init__) = F.indices."""
        from jevdrive.openpilot import frames as opf
        w, h = F.CAM_WH
        for name, f in F.FOCAL.items():
            M = opf.get_warp_matrix(np.zeros(3), opf.intrinsics(w, h, f), name == "wide",
                                    model_K=opf.MEDMODEL_K if name == "road" else opf.wide_K(None))
            y = opf._nn_index(M, (opf.MODEL_W, opf.MODEL_H), (w, h))
            np.testing.assert_array_equal(y, F.indices(name)[0])

    def test_yuv_roundtrip(self):
        img2 = np.stack([self.P(self.img[n], n) for n in ("road", "wide")])
        buf = np.frombuffer(F.pair_to_yuv(img2), np.uint8)[None]
        np.testing.assert_array_equal(F.yuv_to_pairs(buf)[0], img2)

    @unittest.skipUnless(os.path.exists(F.FFMPEG), "no ffmpeg")
    def test_lossless_h264(self):
        rng = np.random.default_rng(1)
        pairs = []
        with tempfile.TemporaryDirectory() as d:
            e = F.Encoder(Path(d) / "x.mp4", crf=0)
            for k in range(5):
                img = {n: np.clip(self.img[n].astype(int) + rng.integers(-3, 4, (1, 1, 4)), 0, 255).astype(np.uint8) for n in self.img}
                p = np.stack([self.P(img[n], n) for n in ("road", "wide")])
                pairs.append(p)
                e.write(F.pair_to_yuv(p))
            e.close()
            back = F.read_pairs(Path(d) / "x.mp4")
        np.testing.assert_array_equal(back, np.stack(pairs))


if __name__ == "__main__":
    unittest.main()
