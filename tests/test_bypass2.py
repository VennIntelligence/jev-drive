"""pbyp2 geometry on synthetic scenes (plan experiments/vlm_arb/plans/2026-10-02-pbyp2-vred2.md): no CARLA, no GPU.

    python -m unittest tests.test_bypass2 -v

Covers the route projection fix (`project_ext`), the same-direction gap check (`same_direction_gap`) and, through
`Privileged.geometry` with a fake agent, the activation rule: blockers outside the route never count, a blocker must
be static for 5 s, no activation within 5 s of a red / yellow ego light, and the gap check holds the pull-out
until the ego is half-way across.
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import b2d_privileged_geometry as G  # noqa: E402

ROUTE = np.stack([np.arange(0., 200.1, 2.), np.zeros(101)], -1)     # straight road along +x, 200 m, 2 m spacing
OFFSET = -3.5                                                       # target lane: 3.5 m to the right (negative normal side)


def car(i, x, y, v=0., yaw=0., half=(2.4, .9), stationary=0.):
    return dict(id=i, type="vehicle.audi.tt", xyz=[x, y, 0.], yaw=yaw, velocity=[v * math.cos(yaw), v * math.sin(yaw)],
                extent=list(half), stationary_s=stationary)


class Projection(unittest.TestCase):
    def test_legacy_clamps(self):
        s, d, _ = G.project([[-10., .5], [210., -2.]], ROUTE)
        np.testing.assert_allclose(s, [0., 200.])                   # the bug: both look like route end points

    def test_extended(self):
        s, d, u = G.project_ext([[-10., .5], [210., -2.], [50., 1.]], ROUTE)
        np.testing.assert_allclose(s, [-10., 210., 50.])
        np.testing.assert_allclose(d, [.5, -2., 1.])
        np.testing.assert_allclose(u, [[1, 0]] * 3)

    def test_bent_route_end(self):
        path = np.array([[0., 0.], [10., 0.], [10., 10.]])           # last segment points +y
        s, d, _ = G.project_ext([[10.5, 14.]], path)
        np.testing.assert_allclose([s[0], d[0]], [24., -.5])
        s, d, _ = G.project_ext([[4., 0.5]], path)                   # inside: same as project
        np.testing.assert_allclose(s, G.project([[4., 0.5]], path)[0])


class SameDirectionGap(unittest.TestCase):
    def check(self, vehicles, ego_s=30., speed=1., **kw):
        return G.same_direction_gap(vehicles, ROUTE, ego_s, speed, OFFSET, **kw)

    def test_empty_and_other_lane(self):
        self.assertEqual(self.check([]), (True, None))
        self.assertTrue(self.check([car(1, 30., 0., v=8.)])[0])                    # own lane, alongside: not this check
        self.assertTrue(self.check([car(2, 30., -7., v=8.)])[0])                   # two lanes over

    def test_alongside_and_just_ahead(self):
        self.assertEqual(self.check([car(1, 30., OFFSET)]), (False, 1))
        self.assertEqual(self.check([car(2, 30. + 7., OFFSET)]), (False, 2))       # rear edge 4.6 m ahead of the rear axle: inside front bumper + 5 m
        self.assertTrue(self.check([car(3, 30. + 14., OFFSET)])[0])                # rear edge 11.6 m ahead of the rear axle, front bumper + 5 m = 8.8: clear

    def test_behind_static_and_moving(self):
        self.assertTrue(self.check([car(1, 30. - 20., OFFSET)])[0])                # static, 11.6 m clear of the rear bumper
        self.assertFalse(self.check([car(2, 30. - 6., OFFSET)])[0])                # static but 2.6 m from the rear bumper
        self.assertFalse(self.check([car(3, 30. - 15., OFFSET, v=8.)])[0])         # 8 m/s, closing 7: 5 m + 1 s x 7 m/s = 12 m needed, 9.6 m clear
        self.assertTrue(self.check([car(4, 30. - 40., OFFSET, v=8.)])[0])          # far enough
        self.assertTrue(self.check([car(5, 30. - 40., OFFSET, v=1.)], speed=1.)[0])  # not closing: static rule only

    def test_oncoming_and_excluded(self):
        self.assertTrue(self.check([car(1, 30., OFFSET, v=8., yaw=math.pi)])[0])  # oncoming lane is the other check
        self.assertTrue(self.check([car(1, 30., OFFSET)], exclude=(1,))[0])        # the obstacle being passed

    def test_behind_route_start(self):
        v = [car(1, -10., OFFSET, v=8.)]                                           # 13 m behind the ego at s = 3 and closing
        self.assertFalse(G.same_direction_gap(v, ROUTE, 3., 1., OFFSET)[0])
        v = [car(1, -30., OFFSET, v=0.)]
        self.assertTrue(G.same_direction_gap(v, ROUTE, 3., 1., OFFSET)[0])


class FakeAgent:
    def __init__(self):
        self.route = SimpleNamespace(xy=ROUTE, s=np.r_[0., np.cumsum(np.linalg.norm(np.diff(ROUTE, axis=0), axis=1))])
        self.ctx = {}

    def _ctx(self):
        return self.ctx


def scene(arm, actors, ego_s=10., speed=1., ego_y=0., t=20., ctx=None, priv=None):
    """One `geometry` call on a synthetic scene; returns (privileged, path, meta)."""
    if priv is None:
        agent = FakeAgent()
        priv = G.Privileged.__new__(G.Privileged)
        priv.agent, priv.arm = agent, arm
        priv.junction, priv.bypass, priv.gap_check = False, True, arm in ("pbypgap", "pbyp2")
        priv.v2, priv.red, priv.last_red = arm == "pbyp2", False, -1e9
        priv.last, priv.static_since, priv.bypass_state, priv.prepared, priv.snapshot_ms = 1e9, {}, None, True, 0.
        priv.flags, priv.jids = np.zeros(len(ROUTE), bool), -np.ones(len(ROUTE), int)
        priv.hero_row = dict(extent=[2.4508, .9])
        priv.adjacent = lambda state: dict(state, offset=OFFSET, borrow=False)
    priv.agent.ctx = ctx or {}
    priv.actors, priv.last = actors, t
    xy = np.array([ego_s, ego_y])
    world = np.r_[xy[None], ROUTE[ROUTE[:, 0] > ego_s + .5][:40]]
    path = priv.geometry(speed, t, xy, 0., world, False)
    return priv, path, priv.meta


class Activation(unittest.TestCase):
    OBST = lambda self, **kw: car(7, 45., 0., stationary=kw.pop("stationary", 6.), **kw)

    def test_real_obstacle_activates_in_both(self):
        for arm in ("pbyp", "pbyp2"):
            _, _, m = scene(arm, [self.OBST()])
            self.assertTrue(m["bypass"], arm)

    def test_blocker_behind_route_start(self):
        a = [car(3, -10., 0., stationary=6.)]
        self.assertTrue(scene("pbyp", a, ego_s=0.)[2]["bypass"])                   # legacy: clamped projection passes
        _, _, m = scene("pbyp2", a, ego_s=0.)
        self.assertFalse(m["bypass"])
        self.assertEqual(m["obstacles"], [])

    def test_blocker_beyond_route_end(self):
        a = [car(3, 203., 0., stationary=6.)]
        self.assertTrue(scene("pbyp", a, ego_s=160.)[2]["bypass"])
        self.assertEqual(scene("pbyp2", a, ego_s=160.)[2]["obstacles"], [])

    def test_static_time(self):
        self.assertTrue(scene("pbyp", [self.OBST(stationary=3.)])[2]["bypass"])
        self.assertEqual(scene("pbyp2", [self.OBST(stationary=3.)])[2]["obstacles"], [])
        self.assertTrue(scene("pbyp2", [self.OBST(stationary=5.1)])[2]["bypass"])

    def test_red_light_memory(self):
        priv, _, m = scene("pbyp2", [], t=20., ctx={"tl": 2, "tl_dist": 20.})       # ego light red, no obstacle in view yet
        priv, _, m = scene("pbyp2", [self.OBST()], t=22., ctx={"tl": 0, "tl_dist": 18.}, priv=priv)
        self.assertFalse(m["bypass"])                                              # green now, red 2 s ago
        self.assertEqual(m["suppressed"], "red_memory")
        priv, _, m = scene("pbyp2", [self.OBST()], t=25.1, ctx={"tl": 0, "tl_dist": 5.}, priv=priv)
        self.assertTrue(m["bypass"])                                               # 5.1 s after the last red
        _, _, m = scene("pbyp2", [self.OBST()], t=20., ctx={"tl": 2, "tl_dist": 60.})     # red beyond 50 m: no memory
        self.assertTrue(m["bypass"])

    def test_gap_hold_then_open_then_commit(self):
        passer = car(9, 16., OFFSET, v=8.)                                         # approaching from behind in the target lane
        priv, path, m = scene("pbyp2", [self.OBST(), passer], ego_s=30., speed=0.)
        self.assertFalse(m["bypass"])
        self.assertTrue(m["gap_hold"])
        self.assertEqual(m["gap_blocker"], 9)
        far = car(9, 30. - 80., OFFSET, v=8.)
        priv, path, m = scene("pbyp2", [self.OBST(), far], ego_s=30., speed=0., t=20.2, priv=priv)
        self.assertTrue(m["bypass"])                                               # gap open: the shifted path is used
        self.assertLess(path[:, 1].min(), -1.)                                     # shifted towards the target lane (y < 0)
        # ego 55% across: a vehicle arriving now no longer holds the manoeuvre
        priv, path, m = scene("pbyp2", [self.OBST(), passer], ego_s=30., speed=2., ego_y=OFFSET * .55, t=20.4, priv=priv)
        self.assertTrue(m["committed"])
        self.assertTrue(m["bypass"])

    def test_legacy_ignores_same_direction_traffic(self):
        passer = car(9, 0., OFFSET, v=8.)
        self.assertTrue(scene("pbypgap", [self.OBST(), passer], ego_s=30.)[2]["bypass"])


if __name__ == "__main__":
    unittest.main()
