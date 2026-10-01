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
                 gnss_gain=.05, heading_gain=.1, lateral_coefficient_s2_per_m=0.):
        self.projector = projector
        self.rear_offset = float(rear_axle_offset_m)
        self.gnss_x = float(gnss_x_m)
        self.gain, self.heading_gain = float(gnss_gain), float(heading_gain)
        self.lateral_coefficient = float(lateral_coefficient_s2_per_m)
        if not math.isfinite(self.lateral_coefficient) or self.lateral_coefficient < 0:
            raise ValueError('Lateral coefficient must be finite and nonnegative')
        if (not np.all(np.isfinite([self.rear_offset, self.gnss_x, self.gain, self.heading_gain]))
                or not 0 < self.gain <= 1 or not 0 < self.heading_gain <= 1):
            raise ValueError('Pose filter gains must be in (0,1]')
        self.reset()

    def reset(self):
        """Forget localization history; recovery requires a fully valid pose."""
        self.xy = self.yaw = self.t = self.raw_xy = None
        self.previous_speed = None
        self.previous_world_gyro = None
        self._last_compass_time = None
        self._diagnostics = dict(reason='no_pose', degraded=False,
                                 compass_valid=False, compass_age_s=None)
        self._diagnostics.update(self._lateral_diagnostics())

    def _lateral_diagnostics(self):
        return dict(lateral_coefficient_s2_per_m=self.lateral_coefficient,
                    lateral_valid=False, lateral_reason='no_interval', lateral_dt_s=None,
                    lateral_mean_speed_mps=None, lateral_mean_world_gyro_rps=None,
                    lateral_midpoint_yaw_rad=None, lateral_delta_xy_m=None)

    @property
    def diagnostics(self):
        result = dict(self._diagnostics)
        if result.get('lateral_delta_xy_m') is not None:
            result['lateral_delta_xy_m'] = list(result['lateral_delta_xy_m'])
        return result

    def _fail(self, reason, message):
        self._diagnostics['reason'] = reason
        self._diagnostics.update(self._lateral_diagnostics())
        self._diagnostics['lateral_reason'] = reason
        raise ValueError(message)

    def update(self, gps, compass, speed, world_yaw_rate, timestamp):
        # Captured smoke input contained an isolated nonfinite compass; its
        # simulator-side cause is unproven. Only heading has a bounded fallback. GPS,
        # speed, gyro and time remain mandatory, and truth is never consulted.
        compass_valid = bool(np.isfinite(compass))
        age = (float(timestamp) - self._last_compass_time
               if self._last_compass_time is not None and np.isfinite(timestamp) else None)
        self._diagnostics = dict(reason='tracking', degraded=False,
                                 compass_valid=compass_valid,
                                 compass_age_s=0. if compass_valid else age)
        self._diagnostics.update(self._lateral_diagnostics())
        values = np.r_[np.asarray(gps)[:2], speed, world_yaw_rate, timestamp]
        if not np.all(np.isfinite(values)):
            self._fail('invalid_motion', 'Nonfinite motion sensor other than compass')
        if self.t is not None:
            dt = float(timestamp) - self.t
            if dt <= 0 or dt > .2:
                self._fail('timestamp_discontinuity', 'Motion sensor timestamp discontinuity')
        if not compass_valid:
            if self.t is None or self._last_compass_time is None:
                self._fail('compass_uninitialized', 'Missing compass before pose initialization')
            if age > .2 + 1e-8:
                self._fail('compass_dropout_timeout', 'Compass dropout exceeds 0.2 seconds')
            self._diagnostics.update(reason='compass_dropout_prediction', degraded=True)
            # Gyro prediction also supplies the GNSS lever-arm heading. There
            # is no compass correction on this tick, nor a fabricated compass.
            observed_yaw = wrap(self.yaw + float(world_yaw_rate) * dt)
        else:
            observed_yaw = compass_to_yaw(compass)
        raw = self.projector.project(gps)
        raw += (self.rear_offset - self.gnss_x) * np.array([
            math.cos(observed_yaw), math.sin(observed_yaw)])
        if self.t is None:
            self.xy, self.yaw = raw.copy(), observed_yaw
        else:
            turn = float(world_yaw_rate) * dt
            middle = self.yaw + .5 * turn
            distance = .5 * (float(speed) + self.previous_speed) * dt
            mean_speed = .5 * (float(speed) + self.previous_speed)
            mean_gyro = .5 * (float(world_yaw_rate) + self.previous_world_gyro)
            lateral_delta = np.zeros(2)
            if self.lateral_coefficient != 0.:
                # Empirical rear-axle motion term, opt-in for measured MKZ tests.
                # Same prior/current interval convention as the frozen replay;
                # heading and the subsequent GNSS correction stay unchanged.
                with np.errstate(over='ignore', invalid='ignore'):
                    lateral_delta = (-self.lateral_coefficient * mean_speed * mean_speed
                                     * mean_gyro * dt * np.array([-np.sin(middle), np.cos(middle)]))
                if not np.isfinite(lateral_delta).all():
                    self._fail('invalid_lateral_prediction', 'Nonfinite lateral propagation')
                # All input/interval/dropout checks and prediction validation have
                # completed before modifying any pose or motion-history state.
                self.xy += lateral_delta
            self._diagnostics.update(lateral_valid=True,
                lateral_reason='applied' if self.lateral_coefficient != 0. else 'disabled',
                lateral_dt_s=dt, lateral_mean_speed_mps=mean_speed,
                lateral_mean_world_gyro_rps=mean_gyro, lateral_midpoint_yaw_rad=middle,
                lateral_delta_xy_m=lateral_delta.tolist())
            self.xy += distance * np.array([math.cos(middle), math.sin(middle)])
            self.yaw = wrap(self.yaw + turn)
            if compass_valid:
                self.yaw = wrap(self.yaw + self.heading_gain * wrap(observed_yaw - self.yaw))
            self.xy += self.gain * (raw - self.xy)
        self.raw_xy, self.t, self.previous_speed = raw, float(timestamp), float(speed)
        self.previous_world_gyro = float(world_yaw_rate)
        if compass_valid:
            self._last_compass_time = float(timestamp)
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
        # Smooth the given route through its existing points, retaining those
        # points unchanged for independent reference/error evaluation.
        self._route_slopes = np.empty_like(self.points)
        self._route_slopes[0] = self.segments[0] / self.lengths[0]
        self._route_slopes[-1] = self.segments[-1] / self.lengths[-1]
        if len(self.points) > 2:
            self._route_slopes[1:-1] = ((self.points[2:] - self.points[:-2]) /
                                      (self.arc[2:] - self.arc[:-2])[:, None])
        self.rejoin_diagnostics = {}
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

    def _reference_curve(self, stations):
        """C1 Hermite interpolation through the immutable dense reference points."""
        stations = np.clip(np.asarray(stations, dtype=float), 0., self.arc[-1])
        indices = np.clip(np.searchsorted(self.arc, stations, side='right') - 1,
                          0, len(self.lengths) - 1)
        length = self.lengths[indices, None]
        u = ((stations - self.arc[indices]) / self.lengths[indices])[:, None]
        a, b = self.points[indices], self.points[indices + 1]
        da, db = self._route_slopes[indices], self._route_slopes[indices + 1]
        positions = ((2*u**3 - 3*u**2 + 1) * a + (u**3 - 2*u**2 + u) * length * da
                     + (-2*u**3 + 3*u**2) * b + (u**3 - u**2) * length * db)
        derivative = ((6*u**2 - 6*u) * a / length + (3*u**2 - 4*u + 1) * da
                      + (-6*u**2 + 6*u) * b / length + (3*u**2 - 2*u) * db)
        return positions, derivative

    @staticmethod
    def _sampled_curvature(points):
        """Three-point circumcircle curvature, measured on the constructed path."""
        if len(points) < 3:
            return np.zeros(0)
        a, b = np.diff(points, axis=0)[:-1], np.diff(points, axis=0)[1:]
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(a + b, axis=1)
        return 2. * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.maximum(denominator, 1e-12)

    def _rejoin_path(self, xy, yaw):
        """Join actual ego position/heading to the route without a timed teleport.

        Quintic position/tangent corrections decay along the route, rather than
        connecting a chord across its turns. A finite 6--18 m search limits added
        curvature where feasible. Curvature is a geometric diagnostic, not a
        proof of feasible tire forces at the configured cruise speed.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(np.r_[xy, yaw]).all():
            raise ValueError('Nonfinite route rejoin pose')
        remaining = max(0., float(self.arc[-1] - self.progress))
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        if remaining < 1e-6:
            endpoint = self.points[-1]
            delta = endpoint - xy
            if np.linalg.norm(delta) < 1e-6:
                self.rejoin_diagnostics = {'reason': 'at_endpoint', 'join_length_m': 0.,
                    'max_rejoin_curvature_inv_m': 0., 'reference_curvature_inv_m': 0.,
                    'curvature_bound_satisfied': True, 'path_length_m': 0.}
                return np.array([xy, endpoint])
            # No arc remains to build a forward join. Returning a stationary
            # reference is an explicit stop request, not a fictitious timed jump
            # to a target beside or behind the axle.
            if np.dot(delta, forward) <= 0:
                self.rejoin_diagnostics = {'reason': 'terminal_target_not_forward',
                    'join_length_m': 0., 'max_rejoin_curvature_inv_m': None,
                    'reference_curvature_inv_m': None, 'curvature_bound_satisfied': False,
                    'path_length_m': 0., 'endpoint_distance_m': float(np.linalg.norm(delta))}
                return np.array([xy, xy])
            distance = float(np.linalg.norm(delta))
            u = np.linspace(0., 1., 33)[:, None]
            final_tangent = self.segments[-1] / self.lengths[-1]
            connector = ((2*u**3 - 3*u**2 + 1) * xy + (u**3 - 2*u**2 + u) * distance * forward
                         + (-2*u**3 + 3*u**2) * endpoint + (u**3 - u**2) * distance * final_tangent)
            increments = np.diff(connector, axis=0)
            # Circumcircle curvature is zero at a perfectly collinear reversal.
            # Such a cusp is not a usable forward path, even with finite points.
            if (np.any(np.sum(increments[:-1] * increments[1:], axis=1) <= 0.)
                    or np.any(np.linalg.norm(increments, axis=1) < 1e-10)):
                self.rejoin_diagnostics = {'reason': 'terminal_connector_reversal',
                    'join_length_m': distance, 'max_rejoin_curvature_inv_m': None,
                    'reference_curvature_inv_m': 0., 'curvature_bound_satisfied': False,
                    'path_length_m': 0., 'endpoint_distance_m': distance}
                return np.array([xy, xy])
            curvature = float(np.max(self._sampled_curvature(connector)))
            self.rejoin_diagnostics = {'reason': 'terminal_forward_connector' if curvature <= .2 else 'terminal_curvature_concern',
                'join_length_m': distance, 'max_rejoin_curvature_inv_m': curvature,
                'reference_curvature_inv_m': 0., 'curvature_limit_inv_m': .2,
                'curvature_bound_satisfied': curvature <= .2,
                'path_length_m': float(np.sum(np.linalg.norm(np.diff(connector, axis=0), axis=1)))}
            return connector
        origin, origin_derivative = self._reference_curve(np.array([self.progress]))
        offset = xy - origin[0]
        derivative_correction = forward - origin_derivative[0]
        minimum = min(remaining, max(6., min(12., self.cruise), 2. * np.linalg.norm(offset)))
        maximum = min(remaining, 18.)
        minimum = min(minimum, maximum)
        joins = np.unique(np.r_[minimum, np.linspace(minimum, maximum, 4)])
        # <=0.2 m sampling resolves the smooth blend; all original dense knots
        # remain available separately for route projection and truth metrics.
        offsets = np.linspace(0., remaining, max(33, int(math.ceil(remaining / .2)) + 1))
        base, derivatives = self._reference_curve(self.progress + offsets)
        reference_curvature = self._sampled_curvature(base)
        path = None
        for join_length in joins:
            u = np.clip(offsets / join_length, 0., 1.)
            weight = 1. - 10*u**3 + 15*u**4 - 6*u**5
            tangent_weight = join_length * (u - 6*u**3 + 8*u**4 - 3*u**5)
            candidate = base + weight[:, None] * offset + tangent_weight[:, None] * derivative_correction
            candidate[0], candidate[-1] = xy, self.points[-1]
            lengths = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
            increments = np.diff(candidate, axis=0)
            reference_tangents = derivatives[:-1] + derivatives[1:]
            if (not np.isfinite(candidate).all() or float(np.sum(lengths)) < 1e-6
                    or np.dot(candidate[1] - candidate[0], forward) <= 0
                    or np.any(np.sum(increments * reference_tangents, axis=1) < -1e-5)):
                continue
            local = offsets[1:-1] <= join_length + .2
            curvature = self._sampled_curvature(candidate)
            maximum_curvature = float(np.max(curvature[local])) if local.any() else 0.
            reference_maximum = float(np.max(reference_curvature[local])) if local.any() else 0.
            curvature_limit = max(.2, reference_maximum + .08)
            satisfied = maximum_curvature <= curvature_limit
            path = candidate
            self.rejoin_diagnostics = {'reason': 'rejoin' if satisfied else 'rejoin_curvature_concern',
                'join_length_m': float(join_length), 'max_rejoin_curvature_inv_m': maximum_curvature,
                'reference_curvature_inv_m': reference_maximum, 'curvature_limit_inv_m': curvature_limit,
                'curvature_bound_satisfied': bool(satisfied), 'path_length_m': float(np.sum(lengths)),
                'initial_offset_m': float(np.linalg.norm(offset))}
            if satisfied:
                break
        if path is None:
            raise ValueError('No finite nondegenerate forward route rejoin could be constructed')
        return path

    def trajectory(self, xy, yaw):
        if self.terminal_hold:
            self.rejoin_diagnostics = {'reason': 'terminal_hold', 'join_length_m': 0.,
                'max_rejoin_curvature_inv_m': 0., 'reference_curvature_inv_m': 0.,
                'curvature_bound_satisfied': True, 'path_length_m': 0.}
            return np.zeros((20, 2))
        path = self._rejoin_path(xy, yaw)
        path_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        remaining = float(path_arc[-1])
        # Keep the configured cruise/deceleration profile, but parameterize it
        # by actual rejoin-path distance from ego to the unchanged endpoint.
        speed = min(self.cruise, math.sqrt(2. * self.deceleration * remaining))
        times = np.arange(1, 21) * .25
        braking_distance = speed * speed / (2. * self.deceleration)
        cruise_time = max(0., (remaining - braking_distance) / speed) if speed > 0 else 0.
        cruise_part = np.minimum(times, cruise_time) * speed
        braking_time = np.minimum(np.maximum(times - cruise_time, 0), speed / self.deceleration)
        distance = cruise_part + speed * braking_time - .5 * self.deceleration * braking_time ** 2
        arc = np.minimum(distance, remaining)
        world = np.column_stack([np.interp(arc, path_arc, path[:, i]) for i in (0, 1)])
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
