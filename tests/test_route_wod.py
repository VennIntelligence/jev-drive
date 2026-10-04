"""experiments/op_route_cmd/scripts/route_wod.py: the 2.5 s chaining of WOD-E2E frames on a synthetic drive. No GPU, no DATA_DIR.

    python -m unittest tests.test_route_wod -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments" / "op_route_cmd" / "scripts"), str(REPO / "lib")]
import route_wod as W  # noqa: E402


def drive(n_frames=200, v=8.0, radius=15.0, turn_at=8.0):
    """Global ego poses at 10 Hz: straight, a 90 deg left turn, straight."""
    dt, x, y, yaw, out = 0.1, 0.0, 0.0, 0.3, []
    for i in range(n_frames):
        t = i * dt
        out.append((x, y, yaw))
        k = (1.0 / radius) if turn_at <= t < turn_at + (np.pi / 2) * radius / v else 0.0
        yaw += v * k * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
    return np.array(out)


def ego_logs(G):
    """past (n, 16, 6) / future (n, 20, 3) in each frame's own ego frame at 4 Hz, like WOD-E2E."""
    n = len(G)
    past, fut = np.zeros((n, 16, 6)), np.zeros((n, 20, 3))
    ts = np.arange(n) * 0.1

    def pose_at(t):
        return np.stack([np.interp(t, ts, G[:, k]) for k in range(2)], -1)
    for i in range(n):
        c, s = np.cos(G[i, 2]), np.sin(G[i, 2])
        for arr, off in ((past, -0.25 * np.arange(15, -1, -1)), (fut, 0.25 * np.arange(1, 21))):
            tt = ts[i] + off
            ok = (tt >= 0) & (tt <= ts[-1])
            d = pose_at(np.clip(tt, 0, ts[-1])) - G[i, :2]
            arr[i, :, 0] = c * d[:, 0] + s * d[:, 1]
            arr[i, :, 1] = -s * d[:, 0] + c * d[:, 1]
            if arr is fut and (~ok).any():
                arr[i, ~ok] = np.nan
    return past, fut


class TestChain(unittest.TestCase):
    def test_chain_recovers_the_driven_path(self):
        G = drive()
        past, fut = ego_logs(G)
        n = len(G)
        nxt = np.where(np.arange(n) + 75 < n, np.arange(n) + 25, -1)       # frames near the end lack a full future
        th, t, ok, rms = W.fit_links(past, fut, nxt >= 0, nxt)
        self.assertTrue(ok[nxt >= 0].all())
        self.assertLess(np.nanmax(rms), 0.05)
        i = 10
        (path, dur), = W.chain_paths(fut, nxt, ok, th, t, [i])
        c, s = np.cos(G[i, 2]), np.sin(G[i, 2])
        tf = np.arange(i * 0.1, (len(G) - 1) * 0.1, 0.005)
        d = np.stack([np.interp(tf, np.arange(len(G)) * 0.1, G[:, k]) for k in range(2)], -1) - G[i, :2]
        truth = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
        # every chained point lies on the true path (distance to the dense truth)
        dist = np.sqrt(((path[:, None] - truth[None]) ** 2).sum(2)).min(1)
        self.assertLess(dist.max(), 0.15)
        self.assertGreater(dur, 10.0)
        self.assertGreater(np.hypot(*np.diff(path, axis=0).T).sum(), 100.0)

    def test_hindsight_on_the_chain_sees_the_turn(self):
        import route_poly as RP
        G = drive()
        past, fut = ego_logs(G)
        n = len(G)
        nxt = np.where(np.arange(n) + 75 < n, np.arange(n) + 25, -1)
        th, t, ok, _ = W.fit_links(past, fut, nxt >= 0, nxt)
        (path, _), = W.chain_paths(fut, nxt, ok, th, t, [5])
        h = RP.hindsight(path)
        self.assertGreater(h["turn_deg"], 80)
        self.assertLess(h["turn_deg"], 100)

    def test_missing_partner_ends_the_chain_with_5s(self):
        G = drive(60)
        past, fut = ego_logs(G)
        nxt = -np.ones(len(G), int)
        (path, dur), = W.chain_paths(fut, nxt, np.zeros(len(G), bool), np.zeros(len(G)), np.zeros((len(G), 2)), [0])
        self.assertEqual(dur, 5.0)
        self.assertEqual(len(path), 21)

    def test_stationary_link_is_pure_translation(self):
        n = 100
        G = np.tile([3.0, 4.0, 1.0], (n, 1))
        past, fut = ego_logs(G)
        nxt = np.where(np.arange(n) + 75 < n, np.arange(n) + 25, -1)
        th, t, ok, _ = W.fit_links(past, fut, nxt >= 0, nxt)
        self.assertTrue(ok[nxt >= 0].all())
        np.testing.assert_allclose(th[nxt >= 0], 0.0)


if __name__ == "__main__":
    unittest.main()
