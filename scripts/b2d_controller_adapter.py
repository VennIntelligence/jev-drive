"""Sensor-only pose and dense-route diagnostic adapter. Python 3.8, NumPy only.

World coordinates follow CARLA (x east, y south); trajectories use x forward,
y left at the rear axle. GPS/world route pairs are diagnostic oracle inputs.
"""
import math
from queue import Empty
import time

import numpy as np


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compass_to_yaw(compass):
    return wrap(float(compass) - math.pi / 2)


def controller_speed(raw_speed):
    """Suppress signed standstill numerical jitter, retaining meaningful reverse."""
    raw_speed = float(raw_speed)
    return 0. if abs(raw_speed) < .01 else raw_speed


def world_to_local(points, xy, yaw):
    d = np.asarray(points, dtype=float) - xy
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack((d[:, 0] * c + d[:, 1] * s,
                            d[:, 0] * s - d[:, 1] * c))


class GPSProjector:
    """Invert the locked vendor Mercator transform using paired route positions.

    Recover its scale from the best-conditioned route axis, then its translation.
    No world/map/hero access, and no assumed map latitude.
    """
    RADIUS = 6378137.0

    @classmethod
    def mercator(cls, gps):
        gps = np.asarray(gps, dtype=float)
        lat, lon = np.deg2rad(gps[..., 0]), np.deg2rad(gps[..., 1])
        return np.stack((cls.RADIUS * lon,
                         -cls.RADIUS * np.log(np.tan(math.pi / 4 + lat / 2))), axis=-1)

    def __init__(self, gps, world):
        merc = self.mercator(gps)
        world = np.asarray(world, dtype=float)
        if merc.shape != world.shape or len(merc) < 2:
            raise ValueError('Need matched dense GPS/world route pairs')
        centered = merc - merc.mean(axis=0)
        target = world - world.mean(axis=0)
        denominator = float(np.sum(centered ** 2))
        if denominator < 1e-6:
            raise ValueError('Route too short to recover GPS projection')
        self.scale = float(np.sum(centered * target) / denominator)
        self.offset = np.mean(world - self.scale * merc, axis=0)
        residual = np.linalg.norm(self.scale * merc + self.offset - world, axis=1)
        self.max_residual_m = float(np.max(residual))
        if not 0 < self.scale <= 1.000001 or self.max_residual_m > .001:
            raise ValueError('GPS/world route projection mismatch: %.6f m' % self.max_residual_m)

    def project(self, gps):
        return self.scale * self.mercator(np.asarray(gps)[:2]) + self.offset


class PoseFilter:
    def __init__(self, projector, rear_axle_offset_m, gnss_x_m=-1.4,
                 gnss_gain=.05, heading_gain=.1):
        self.projector = projector
        self.rear_offset = float(rear_axle_offset_m)
        self.gnss_x = float(gnss_x_m)
        self.gain, self.heading_gain = float(gnss_gain), float(heading_gain)
        if (not np.all(np.isfinite([self.rear_offset, self.gnss_x, self.gain, self.heading_gain]))
                or not 0 < self.gain <= 1 or not 0 < self.heading_gain <= 1):
            raise ValueError('Pose filter gains must be in (0,1]')
        self.xy = self.yaw = self.t = self.raw_xy = None
        self.previous_speed = None

    def update(self, gps, compass, speed, world_yaw_rate, timestamp):
        values = np.r_[np.asarray(gps)[:2], compass, speed, world_yaw_rate, timestamp]
        if not np.all(np.isfinite(values)):
            raise ValueError('Nonfinite motion sensor')
        observed_yaw = compass_to_yaw(compass)
        raw = self.projector.project(gps)
        raw += (self.rear_offset - self.gnss_x) * np.array([
            math.cos(observed_yaw), math.sin(observed_yaw)])
        if self.t is None:
            self.xy, self.yaw = raw.copy(), observed_yaw
        else:
            dt = float(timestamp) - self.t
            if dt <= 0 or dt > .2:
                raise ValueError('Motion sensor timestamp discontinuity')
            turn = float(world_yaw_rate) * dt
            middle = self.yaw + .5 * turn
            distance = .5 * (float(speed) + self.previous_speed) * dt
            self.xy += distance * np.array([math.cos(middle), math.sin(middle)])
            self.yaw = wrap(self.yaw + turn)
            self.yaw = wrap(self.yaw + self.heading_gain * wrap(observed_yaw - self.yaw))
            self.xy += self.gain * (raw - self.xy)
        self.raw_xy, self.t, self.previous_speed = raw, float(timestamp), float(speed)
        return self.xy.copy(), self.yaw


class FrameRouter:
    """Exact motion barrier with retained, coherent optional camera frames.

    Camera rendering phase is independent of agent tick count. With policy none,
    cameras are optional: never wait on a frame that sensor_tick will not produce.
    """
    MOTION = frozenset(('GPS', 'IMU', 'SPEED'))

    def __init__(self, camera_tags=(), retain_frames=16):
        self.cameras = frozenset(camera_tags)
        self.frames = {}
        self.retain = int(retain_frames)

    def _put(self, item):
        tag, frame, data = item
        self.frames.setdefault(int(frame), {})[tag] = (int(frame), data)

    def read(self, interface, frame):
        deadline = time.monotonic() + interface._queue_timeout
        queue = interface._data_buffers
        while not self.MOTION.issubset(self.frames.get(frame, {})):
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError('Timed out waiting for motion frame %s' % frame)
            try:
                self._put(queue.get(True, left))
            except Empty:
                raise RuntimeError('Timed out waiting for motion frame %s' % frame)
        # Preserve late camera packets, including incomplete sets, across calls.
        while True:
            try:
                self._put(queue.get(False))
            except Empty:
                break
        result = {tag: self.frames[frame][tag] for tag in self.MOTION}
        complete = [f for f, data in self.frames.items()
                    if f <= frame and self.cameras and self.cameras.issubset(data)]
        if complete:
            latest = self.frames[max(complete)]
            result.update({tag: latest[tag] for tag in self.cameras})
        self.frames = {f: data for f, data in self.frames.items() if f >= frame - self.retain}
        return result


class RouteAdapter:
    def __init__(self, points, cruise_mps=8, end_extension_m=3., stop_deceleration=2.):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError('Route must be an N by 2 array with at least two positions')
        if not np.all(np.isfinite([cruise_mps, end_extension_m, stop_deceleration])) or end_extension_m < 0:
            raise ValueError('Route parameters must be finite with nonnegative end extension')
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5]
        points = points[keep]
        if len(points) < 2 or not np.all(np.isfinite(points)):
            raise ValueError('Route needs at least two finite distinct positions')
        self.official_points = points.copy()
        self.official_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if end_extension_m > 0:
            tangent = points[-1] - points[-2]
            points = np.vstack((points, points[-1] + end_extension_m * tangent / np.linalg.norm(tangent)))
        self.points = points
        self.segments = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.arc = np.r_[0, np.cumsum(self.lengths)]
        self.progress = 0.
        self.cruise = float(cruise_mps)
        self.deceleration = float(stop_deceleration)
        if self.cruise <= 0 or self.deceleration <= 0:
            raise ValueError('Cruise speed and stop deceleration must be positive')
        self.last_time = None
        self.terminal_hold = False
        self.endpoint_distance_m = None

    def nearest(self, xy, yaw=None, low=0., high=None):
        high = self.arc[-1] if high is None else high
        indices = np.where((self.arc[:-1] <= high) & (self.arc[1:] >= low))[0]
        if len(indices) == 0:
            raise ValueError('Empty route projection interval')
        starts, segments = self.points[indices], self.segments[indices]
        fraction = np.sum((xy - starts) * segments, axis=1) / self.lengths[indices] ** 2
        minimum = np.maximum(0., (low - self.arc[indices]) / self.lengths[indices])
        maximum = np.minimum(1., (high - self.arc[indices]) / self.lengths[indices])
        fraction = np.clip(fraction, minimum, maximum)
        projection = starts + fraction[:, None] * segments
        distance = np.sum((xy - projection) ** 2, axis=1)
        if yaw is not None:
            heading = np.arctan2(segments[:, 1], segments[:, 0])
            # Heading helps disambiguate nearby opposing branches; bounded progress
            # is the primary protection against jumping across a route crossing.
            distance += 4. * (1. - np.cos(heading - yaw))
        j = int(np.argmin(distance))
        idx = indices[j]
        tangent = self.segments[idx] / self.lengths[idx]
        error = xy - projection[j]
        cross_track = tangent[1] * error[0] - tangent[0] * error[1]
        return float(self.arc[idx] + fraction[j] * self.lengths[idx]), float(cross_track)

    def project(self, xy, yaw, speed, timestamp):
        dt = .05 if self.last_time is None else timestamp - self.last_time
        forward = 20. if self.last_time is None else max(2., 2. * abs(speed) * dt)
        position, cross_track = self.nearest(xy, yaw, max(0., self.progress - 2.),
                                            min(self.arc[-1], self.progress + forward))
        self.progress = max(self.progress, position)
        self.last_time = timestamp
        self.endpoint_distance_m = float(np.linalg.norm(np.asarray(xy) - self.points[-1]))
        # Route progress alone is insufficient after an off-path excursion. The
        # parking policy also requires being close to the physical endpoint and
        # nearly stopped. Once parked, GNSS jitter must not command reacceleration.
        if (self.arc[-1] - self.progress <= .2 and abs(speed) < .1
                and self.endpoint_distance_m <= .5):
            self.terminal_hold = True
        return cross_track

    def trajectory(self, xy, yaw):
        if self.terminal_hold:
            return np.zeros((20, 2))
        # Exact constant-deceleration terminal profile, reaching beyond the official
        # endpoint so rear-axle stopping cannot prevent actor-based completion.
        remaining = max(0., self.arc[-1] - self.progress)
        speed = min(self.cruise, math.sqrt(2. * self.deceleration * remaining))
        times = np.arange(1, 21) * .25
        braking_distance = speed * speed / (2. * self.deceleration)
        cruise_time = max(0., (remaining - braking_distance) / speed) if speed > 0 else 0.
        cruise_part = np.minimum(times, cruise_time) * speed
        braking_time = np.minimum(np.maximum(times - cruise_time, 0), speed / self.deceleration)
        distance = cruise_part + speed * braking_time - .5 * self.deceleration * braking_time ** 2
        arc = np.minimum(self.progress + distance, self.arc[-1])
        world = np.column_stack([np.interp(arc, self.arc, self.points[:, i]) for i in (0, 1)])
        return world_to_local(world, xy, yaw)


class TruthLogger:
    """Read-only evaluation path. Its results are never passed into control state."""
    def __init__(self, hero_provider, route, rear_offset):
        self.hero_provider, self.route, self.rear_offset = hero_provider, route, rear_offset

    def measure(self, frame, pose, raw_xy, yaw):
        try:
            hero = self.hero_provider()
            if hero is None:
                return {'truth_error': 'hero_unavailable'}
            world = hero.get_world()
            snapshot = world.get_snapshot()
            if snapshot.frame != frame:
                return {'truth_frame': snapshot.frame, 'truth_error': 'frame_mismatch'}
            actor = snapshot.find(hero.id)
            if actor is None:
                return {'truth_frame': frame, 'truth_error': 'actor_unavailable'}
            tf = actor.get_transform()
            truth_yaw = math.radians(tf.rotation.yaw)
            truth_pitch = math.radians(getattr(tf.rotation, 'pitch', 0.))
            truth = np.array([tf.location.x, tf.location.y]) + self.rear_offset * math.cos(truth_pitch) * np.array([
                math.cos(truth_yaw), math.sin(truth_yaw)])
            _, cross = self.route.nearest(truth, truth_yaw, max(0, self.route.progress - 10),
                                          min(self.route.arc[-1], self.route.progress + 10))
            return {'truth_frame': frame, 'truth_xy': truth.tolist(), 'truth_yaw': truth_yaw,
                    'truth_cross_track_m': cross, 'pose_error_m': float(np.linalg.norm(pose - truth)),
                    'raw_pose_error_m': float(np.linalg.norm(raw_xy - truth)),
                    'pose_heading_error_rad': wrap(yaw - truth_yaw)}
        except Exception as exc:
            return {'truth_error': '%s: %s' % (type(exc).__name__, exc)}
