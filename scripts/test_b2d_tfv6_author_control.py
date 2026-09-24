"""Golden outputs from LEAD route and waypoint PID on one synthetic plan."""
import numpy as np

from b2d_tfv6_author_control import AuthorController


def test_author_route_matches_vendor_golden():
    t = np.arange(1, 21) * .25
    points = np.column_stack((6*t, -.02*(6*t)**2))
    ctl = AuthorController('route')
    assert ctl.update(points, 0.)
    throttle, steer, brake = ctl.step(0., 6., 0.)
    assert abs(throttle - .8163400508504441) < 1e-9
    assert steer == .072
    assert brake == 0.


def test_author_waypoint_matches_vendor_golden():
    t = np.arange(1, 21) * .25
    points = np.column_stack((6*t, -.02*(6*t)**2))
    ctl = AuthorController('waypoint')
    assert ctl.update(points, 0.)
    throttle, steer, brake = ctl.step(0., 6., 0.)
    assert abs(throttle - .36641520261764526) < 1e-6
    assert abs(steer - .06056542628341251) < 1e-6
    assert brake == 0.
