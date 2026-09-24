"""Checks for the preregistered TFv6 actor-to-rear conversion."""

import unittest

import numpy as np

from b2d_tfv6_coordinates import REAR_OFFSET_M, rear_waypoints


def old_rear_waypoints(actor):
    points = np.asarray(actor, dtype=float) * np.array([1.0, -1.0])
    tangent = np.empty_like(points)
    previous = np.array([1.0, 0.0])
    for i in range(8):
        delta = (points[0] if i == 0 else points[7] - points[6] if i == 7
                 else points[i + 1] - points[i - 1])
        norm = np.linalg.norm(delta)
        if norm >= 0.05:
            previous = delta / norm
        tangent[i] = previous
    return points - REAR_OFFSET_M * tangent + np.array([REAR_OFFSET_M, 0.0])


class RearWaypointTests(unittest.TestCase):
    def test_straight_is_identity_after_y_flip(self):
        actor = np.column_stack((np.arange(1, 9, dtype=float), np.zeros(8)))
        np.testing.assert_allclose(rear_waypoints(actor), actor, atol=1e-12)

    def test_rotated_future_actor_places_rear_behind_future_heading(self):
        actor = np.column_stack((np.zeros(8), -np.arange(1, 9, dtype=float)))
        result = rear_waypoints(actor)
        np.testing.assert_array_equal(result[0], np.array([0., 1.]))
        np.testing.assert_allclose(result[1:, 0], np.full(7, REAR_OFFSET_M), atol=1e-12)
        np.testing.assert_allclose(result[1:, 1], np.arange(2, 9) - REAR_OFFSET_M, atol=1e-12)

    def test_3514_standstill_jitter_does_not_create_phantom_target(self):
        actor = np.tile([0., -0.08], (8, 1))
        np.testing.assert_array_equal(rear_waypoints(actor)[0], [0., 0.08])

    def test_past_rear_offset_is_bit_identical_to_frozen_rule(self):
        actor = np.array([[.05, -.08], [.31, -.24], [.63, -.46], [1.02, -.61],
                          [1.50, -.70], [1.91, -.79], [2.32, -.87], [2.76, -.96]])
        s = np.cumsum(np.linalg.norm(np.diff(np.vstack((np.zeros(2),
                            actor * [1., -1.])), axis=0), axis=1))
        np.testing.assert_array_equal(rear_waypoints(actor)[s >= REAR_OFFSET_M],
                                      old_rear_waypoints(actor)[s >= REAR_OFFSET_M])

    def test_all_coincident_uses_forward_tangent(self):
        actor = np.zeros((8, 2))
        np.testing.assert_array_equal(rear_waypoints(actor), actor)

    def test_short_segment_reuses_previous_tangent(self):
        actor = np.column_stack((np.array([1, 2, 2.01, 2.015, 2.02, 2.025, 2.03, 2.035]),
                                 np.zeros(8)))
        np.testing.assert_allclose(rear_waypoints(actor), actor)


if __name__ == "__main__":
    unittest.main()
