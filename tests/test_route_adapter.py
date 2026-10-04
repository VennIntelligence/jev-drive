"""lib/route_adapter.py: maneuver features, zero-init identity, numpy == torch."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import route_adapter as RA  # noqa: E402
import route_poly as RP  # noqa: E402


def l_turn(d=40.0, R=10.0, sign=1.0):
    s = np.arange(0, d, 0.5)
    a = np.stack([s, 0 * s], -1)
    th = np.linspace(0, np.pi / 2, 60)
    arc = np.stack([d + R * np.sin(th), sign * R * (1 - np.cos(th))], -1)
    tail = np.stack([np.full(200, d + R), sign * (R + np.arange(200) * 0.5)], -1)
    return np.vstack([a, arc, tail])


class T(unittest.TestCase):
    def test_maneuver(self):
        for sign in (1, -1):
            h = RP.hindsight(l_turn(sign=sign))
            has, d, b = RA.maneuver(h["poly"], h["pmask"])
            self.assertEqual(has, 1.0)
            self.assertLess(abs(np.degrees(b) - sign * 90), 15)
            self.assertLess(abs(d - 40), 12)
        s = np.stack([np.arange(0, 160, 0.5), np.zeros(320)], -1)
        h = RP.hindsight(s)
        self.assertEqual(RA.maneuver(h["poly"], h["pmask"])[0], 0.0)

    def test_noise_keeps_side(self):
        h = RP.hindsight(l_turn(sign=-1))
        rng = np.random.default_rng(0)
        b = [RA.features("bear", h["poly"], h["pmask"], rng)[5] for _ in range(50)]
        self.assertGreater(np.mean(np.array(b) < -0.5), 0.9)

    def test_identity_and_numpy(self):
        import torch
        for enc in ("bear", "poly"):
            ad = RA.RouteAdapter(enc)
            h = RP.hindsight(l_turn())
            f = torch.from_numpy(RA.features(enc, h["poly"], h["pmask"]))[None]
            self.assertEqual(float(ad(f).abs().max()), 0.0)
            torch.nn.init.normal_(ad.out.weight, std=0.01)
            f0 = f.clone()
            f0[:, 0] = 0
            self.assertEqual(float(ad(f0).abs().max()), 0.0)
            with tempfile.TemporaryDirectory() as d:
                ad.to_npz(Path(d) / "a.npz")
                nb = RA.NumpyAdapter(Path(d) / "a.npz").bias(f[0].numpy())
            np.testing.assert_allclose(nb.astype(np.float32), ad(f).detach().numpy().astype(np.float16).astype(np.float32), atol=2e-3)

    def test_route_from_path(self):
        p, m = RA.route_poly_from_path(l_turn()[1:])
        self.assertTrue(m.all())
        np.testing.assert_allclose(p[0], 0)
        np.testing.assert_allclose(p[2], [20, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
