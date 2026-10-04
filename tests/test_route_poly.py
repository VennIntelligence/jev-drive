"""lib/route_poly.py: hindsight label, navigation noise (option C bit-identity, statistics), sidecar attach. No GPU, no DATA_DIR.

    python -m unittest tests.test_route_poly -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))
import route_poly as R  # noqa: E402


def legacy_noisy_road(xy, sigma, rng, decim=10.0, corr=30.0):
    """Frozen copy of turn_calibration_sparse.noisy_road before it moved to lib/ (the reference for option C)."""
    P, g = R.poly_resample(xy, 0.25)
    keep = np.arange(0, len(P), int(decim / 0.25))
    keep = np.r_[keep, len(P) - 1] if keep[-1] != len(P) - 1 else keep
    Q = P[keep]
    t = np.gradient(Q, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-t[:, 1], t[:, 0]], -1)
    a = np.exp(-decim / corr)
    e = np.zeros(len(Q))
    e[0] = rng.normal()
    for i in range(1, len(Q)):
        e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * rng.normal()
    return Q + sigma * e[:, None] * nrm


def left_turn(radius=12.0, pre=40, post=70, step=1.0):
    th = np.linspace(0, np.pi / 2, 60)
    return np.r_[[[0.0, 0.0]], np.c_[np.arange(step, pre + step, step), np.zeros(int(pre / step))],
                 np.c_[pre + radius * np.sin(th), radius - radius * np.cos(th)],
                 np.c_[np.full(int(post / step), pre + radius), radius + np.arange(step, post + step, step)]]


class TestHindsight(unittest.TestCase):
    def test_straight(self):
        h = R.hindsight(np.c_[np.arange(0, 200.0, 2.0), np.zeros(100)])
        self.assertEqual(h["pmask"].sum(), 16)
        self.assertAlmostEqual(h["plen"], 150.0)
        np.testing.assert_allclose(h["poly"][:, 0], 10.0 * np.arange(16), atol=0.3)
        self.assertTrue(np.isnan(h["turn_deg"]))
        self.assertEqual(h["n_turn"], 0)

    def test_left_and_right_turn_sign(self):
        h = R.hindsight(left_turn())
        self.assertGreater(h["turn_deg"], 80)                  # left = positive
        self.assertLess(h["turn_deg"], 100)
        self.assertAlmostEqual(h["turn_s"], 39.0, delta=4.0)
        self.assertAlmostEqual(h["turn_rmin"], 12.0, delta=3.0)
        xy = left_turn()
        xy[:, 1] *= -1
        h2 = R.hindsight(xy)
        self.assertAlmostEqual(h2["turn_deg"], -h["turn_deg"], delta=1.0)
        self.assertEqual(R.turn_bin([h["turn_deg"]])[0], 2)

    def test_short_future_is_masked_and_incomplete_turn_not_counted(self):
        h = R.hindsight(left_turn(post=0, step=1.0)[:60])      # path ends inside the turn
        self.assertEqual(h["n_turn"], 0)
        self.assertLess(h["pmask"].sum(), 16)
        self.assertFalse(h["pmask"][int(h["pmask"].sum()):].any())
        self.assertTrue((h["poly"][~h["pmask"]] == 0).all())

    def test_stationary(self):
        h = R.hindsight(np.zeros((10, 2)))
        self.assertEqual(h["pmask"].sum(), 1)
        self.assertEqual(h["plen"], 0.0)

    def test_first_vertex_is_origin(self):
        np.testing.assert_array_equal(R.hindsight(left_turn())["poly"][0], [0, 0])

    def test_sparse_2hz_poses_still_find_the_turn(self):
        xy = left_turn()
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        pts = np.stack([np.interp(np.arange(0, s[-1], 5.0), s, xy[:, k]) for k in range(2)], -1)   # 5 m between poses
        self.assertGreater(R.hindsight(pts)["turn_deg"], 75)


class TestNoise(unittest.TestCase):
    def test_option_c_bit_identical_to_the_original(self):
        xy = left_turn()
        for seed in range(5):
            a = legacy_noisy_road(xy, 1.0, np.random.default_rng(seed))
            np.testing.assert_array_equal(a, R.noisy_road(xy, 1.0, np.random.default_rng(seed)))
            np.testing.assert_array_equal(a, R._vertices(*R.poly_resample(xy, 0.25), R.C_OPTION, np.random.default_rng(seed)))
        a = legacy_noisy_road(xy, 2.0, np.random.default_rng(9), decim=5.0)
        np.testing.assert_array_equal(a, R.noisy_road(xy, 2.0, np.random.default_rng(9), decim=5.0))

    def test_closed_loop_scripts_use_the_shared_function(self):
        sys.path.insert(0, str(REPO / "experiments" / "op_closed_loop" / "scripts"))
        with mock.patch.dict("os.environ", {"DATA_DIR": tempfile.gettempdir()}):
            import turn_calibration_sparse as S
        self.assertIs(S.noisy_road, R.noisy_road)
        self.assertIs(S.poly_resample, R.poly_resample)

    def test_lateral_statistics(self):
        h = R.hindsight(np.c_[np.arange(0, 400.0, 1.0), np.zeros(400)])
        cfg = R.NavNoise(along=0.0, p_drop=0.0)
        rng = np.random.default_rng(0)
        y = np.array([R.noise_polyline(h["poly"], h["pmask"], rng, cfg)[0][:, 1] for _ in range(4000)])
        self.assertAlmostEqual(y.std(), 1.0, delta=0.05)           # sd of the lateral offset
        self.assertAlmostEqual(np.corrcoef(y[:, 4], y[:, 5])[0, 1], np.exp(-10 / 30), delta=0.05)   # AR(1) over 10 m vertices

    def test_along_jitter_and_drop(self):
        h = R.hindsight(np.c_[np.arange(0, 400.0, 1.0), np.zeros(400)])
        rng = np.random.default_rng(1)
        cfg = R.NavNoise(sigma=0.0, along=1.5, p_drop=0.0)
        x = np.array([R.noise_polyline(h["poly"], h["pmask"], rng, cfg)[0][1:-1, 0] for _ in range(2000)])
        keep = np.arange(1, 15)
        self.assertAlmostEqual((x[:, :14] - 10.0 * keep).std(), 1.5, delta=0.15)
        n = [R.noise_polyline(h["poly"], h["pmask"], rng, R.NavNoise(sigma=0.0, along=0.0, p_drop=0.1))[1].sum() for _ in range(3000)]
        self.assertAlmostEqual(16 - np.mean(n), 0.1 * 14, delta=0.1)   # only the 14 interior vertices can drop

    def test_output_shape_mask_and_determinism(self):
        h = R.hindsight(left_turn())
        a1, m1 = R.noise_polyline(h["poly"], h["pmask"], np.random.default_rng(3))
        a2, m2 = R.noise_polyline(h["poly"], h["pmask"], np.random.default_rng(3))
        np.testing.assert_array_equal(a1, a2)
        self.assertEqual(a1.shape, (16, 2))
        self.assertFalse(a1[~m1].any())                            # padding is zero
        self.assertTrue(m1[0])
        self.assertLessEqual(m1.sum(), h["pmask"].sum() + 1)
        pb, mb = R.noise_batch(np.stack([h["poly"]] * 3), np.stack([h["pmask"]] * 3), np.random.default_rng(0))
        self.assertEqual(pb.shape, (3, 16, 2))

    def test_zero_noise_is_identity_up_to_resampling(self):
        h = R.hindsight(left_turn())
        a, m = R.noise_polyline(h["poly"], h["pmask"], np.random.default_rng(0), R.NavNoise(sigma=0.0, along=0.0, p_drop=0.0))
        n = int(h["pmask"].sum())
        self.assertAlmostEqual(float(np.abs(a[:n - 1] - h["poly"][:n - 1]).max()), 0.0, delta=0.6)   # chord vs arc spacing on the curve

    def test_degenerate_inputs(self):
        p, m = np.zeros((16, 2)), np.zeros(16, bool)
        a, om = R.noise_polyline(p, m, np.random.default_rng(0))
        self.assertFalse(om.any())
        m[0] = True
        a, om = R.noise_polyline(p, m, np.random.default_rng(0))
        self.assertEqual(om.sum(), 1)


class TestAttach(unittest.TestCase):
    def test_align_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "route.npz"
            np.savez(f, id=np.array(["a", "b"]), poly=np.ones((2, 16, 2), np.float32), pmask=np.ones((2, 16), bool),
                     turn_deg=np.array([30.0, np.nan], np.float32), n_exit=np.array([3, 2], np.int16), cmd=np.array(["left", "straight"]))
            o = R.attach(f, ["b", "zz", "a"])
            np.testing.assert_array_equal(o["has_route"], [True, False, True])
            self.assertFalse(o["pmask"][1].any())
            self.assertTrue(np.isnan(o["turn_deg"][1]))
            self.assertEqual(o["n_exit"].tolist(), [2, -1, 3])
            self.assertEqual(o["cmd"].tolist(), ["straight", "", "left"])
            self.assertTrue((o["poly"][1] == 0).all())


if __name__ == "__main__":
    unittest.main()
