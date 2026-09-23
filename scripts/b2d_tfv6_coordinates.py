"""TFv6 actor-origin waypoints in the current rear-axle, y-left frame."""

import numpy as np


REAR_OFFSET_M = 1.389


def rear_waypoints(actor_waypoints, rear_offset_m=REAR_OFFSET_M):
    points = np.asarray(actor_waypoints, dtype=float)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError("TFv6 waypoints must be finite (8, 2)")
    points = points * np.array([1.0, -1.0])
    tangent = np.empty_like(points)
    previous = np.array([1.0, 0.0])
    for index in range(len(points)):
        if index == 0:
            delta = points[0]
        elif index == len(points) - 1:
            delta = points[index] - points[index - 1]
        else:
            delta = points[index + 1] - points[index - 1]
        norm = np.linalg.norm(delta)
        if norm >= 0.05:
            previous = delta / norm
        tangent[index] = previous
    return points - rear_offset_m * tangent + np.array([rear_offset_m, 0.0])
