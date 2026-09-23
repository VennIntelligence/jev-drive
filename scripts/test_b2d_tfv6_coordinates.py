"""Checks for the preregistered TFv6 actor-to-rear conversion."""

import unittest

import numpy as np

from b2d_tfv6_coordinates import rear_waypoints


class RearWaypointTests(unittest.TestCase):
    def test_straight_is_identity_after_y_flip(self):
        actor = np.column_stack((np.arange(1, 9, dtype=float), np.zeros(8)))
        np.testing.assert_allclose(rear_waypoints(actor), actor, atol=1e-12)

    def test_rotated_future_actor_places_rear_behind_future_heading(self):
        actor = np.column_stack((np.zeros(8), -np.arange(1, 9, dtype=float)))
        result = rear_waypoints(actor)
        np.testing.assert_allclose(result[:, 0], np.full(8, 1.389), atol=1e-12)
        np.testing.assert_allclose(result[:, 1], np.arange(1, 9) - 1.389, atol=1e-12)

    def test_all_coincident_uses_forward_tangent(self):
        actor = np.zeros((8, 2))
        np.testing.assert_array_equal(rear_waypoints(actor), actor)

    def test_short_segment_reuses_previous_tangent(self):
        actor = np.column_stack((np.array([1, 2, 2.01, 2.015, 2.02, 2.025, 2.03, 2.035]),
                                 np.zeros(8)))
        np.testing.assert_allclose(rear_waypoints(actor), actor)


if __name__ == "__main__":
    unittest.main()
