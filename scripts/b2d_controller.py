"""NumPy-only trajectory control, Python 3.8 compatible.

Input: 20 future rear-axle points, x forward / y left, sampled at .25 s; or, with an explicit
per-update trajectory_dt, N points at that spacing (horizon N*dt, never extrapolated).
Output: throttle, CARLA steer (right positive), brake. Yaw rate is left positive.

Opt-in pursuit plant corrections (defaults off; off is bit-identical to the frozen controller):
  pursuit_frame='rear_slip'  pursue along the rear-axle velocity, estimated from sensors only as
                             beta = atan((v + 1 m/s) * yaw_rate / (c g)), the PhysX linear-tyre
                             rear sideslip (todos/2026-09-23-controller-next/lateral-physics.md);
  steer_inverse='ackermann'  PhysX (Ackermann accuracy 1) applies the nominal angle to the inner
                             wheel; command cot(inner) = cot(centre) - w/(2L), centre = atan(L * kappa).
Default geometry: stock Lincoln measured 2026-09-22, Town10HD, CARLA 0.9.15.
Source: /data/runs/b2d/controller/calibration/controller_config.json.
Geometry does not imply an exact bicycle model of real steering dynamics.
CARLA/TCP presets adapt vendor scalar PID semantics to a shared trajectory API;
they are not complete reproductions of the vendors' agents.
"""
from collections import deque
import argparse
import json
import math
import numpy as np


class WindowPID:
    """Exact scalar buffer semantics of the pinned CARLA and TCP sources."""
    def __init__(self, kp, ki, kd, window, semantics='carla', dt=.05):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.window, self.semantics, self.dt = window, semantics, dt
        self.reset()

    def reset(self):
        self.errors = deque([0.] * self.window if self.semantics == 'tcp' else [],
                            maxlen=self.window)

    def step(self, error):
        self.errors.append(float(error))
        integral = derivative = 0.
        if len(self.errors) >= 2:
            if self.semantics == 'tcp':
                integral = sum(self.errors) / len(self.errors)
                derivative = self.errors[-1] - self.errors[-2]
            else:
                integral = sum(self.errors) * self.dt
                derivative = (self.errors[-1] - self.errors[-2]) / self.dt
        value = self.kp * error + self.ki * integral + self.kd * derivative
        return float(np.clip(value, -1., 1.)) if self.semantics == 'carla' else float(value)



class ConditionalPI:
    """SI-unit PI candidate, independent of both vendor buffer algorithms.

    Error is m/s, integral stores actuator effort. Conditional integration stops
    charging into saturation and permits unwinding when error changes direction.
    """
    def __init__(self, lower=-1., upper=.75, kp=1., ki=.25):
        if not np.isfinite([kp, ki]).all() or kp <= 0 or ki <= 0:
            raise ValueError('PI gains must be positive and finite')
        self.kp, self.ki = float(kp), float(ki)
        self.lower, self.upper = float(lower), float(upper)
        self.reset()

    def reset(self):
        self.integral = 0.
        self.raw_effort = 0.
        self.integration_limited = False

    def step(self, error, elapsed):
        if not np.isfinite([error, elapsed]).all() or elapsed <= 0:
            raise ValueError('PI requires finite error and positive elapsed time')
        candidate = self.integral + self.ki * error * elapsed
        raw = self.kp * error + candidate
        self.integration_limited = bool((raw > self.upper and error > 0)
                                        or (raw < self.lower and error < 0))
        if not self.integration_limited:
            self.integral = float(np.clip(candidate, self.lower, self.upper))
        self.raw_effort = float(self.kp * error + self.integral)
        return float(np.clip(self.raw_effort, self.lower, self.upper))


GRAVITY = 9.81
# PhysX gMinLatSpeedForTireModel is 1 tolerance length per second: 1 m/s in UE's centimetre world.
PHYSX_MIN_LATERAL_SPEED = 1.


def rear_slip_angle(speed, yaw_rate, c_per_rad):
    """Rear-axle sideslip body-minus-course (left positive with yaw rate) of the PhysX linear tyre.

    Steady cornering gives v_y,rear = -(1 + v0/v) v^2 w / (c g), so beta = atan((v + v0) w / (c g)),
    independent of mass and CoM. Finite at standstill; inputs are measured speed and gyro only.
    """
    return math.atan((speed + PHYSX_MIN_LATERAL_SPEED) * yaw_rate / (c_per_rad * GRAVITY))


def ackermann_inner_angle(curvature, wheelbase, track_width):
    """Nominal (inner-wheel) angle whose Ackermann bicycle-centre angle yields `curvature` (signed).

    cot(inner) = 1/(L|k|) - w/2L. Continuous through 0; above 90 deg only for R < w/2."""
    magnitude = abs(curvature)
    return math.copysign(math.atan2(wheelbase * magnitude, 1. - .5 * track_width * magnitude), curvature)


def pchip(x, y, query):
    """Fritsch-Carlson monotone cubic Hermite interpolation; clamps outside [x0, xn]."""
    x, y, query = np.asarray(x, float), np.asarray(y, float), np.clip(np.asarray(query, float), x[0], x[-1])
    if len(x) < 3:
        return np.interp(query, x, y)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    m[0], m[-1] = d[0], d[-1]
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    same = d[:-1] * d[1:] > 0
    m[1:-1][same] = (w1[same] + w2[same]) / (w1[same] / d[:-1][same] + w2[same] / d[1:][same])
    i = np.clip(np.searchsorted(x, query, side='right') - 1, 0, len(x) - 2)
    t = (query - x[i]) / h[i]
    return ((2*t**3 - 3*t**2 + 1) * y[i] + (t**3 - 2*t**2 + t) * h[i] * m[i]
            + (-2*t**3 + 3*t**2) * y[i+1] + (t**3 - t**2) * h[i] * m[i+1])


def advance_pose(pose, speed, yaw_rate, dt):
    """Exact constant-twist SE(2) integration, including the zero-turn limit."""
    x, y, yaw = pose
    angle = yaw_rate * dt
    distance = speed * dt
    mid = yaw + angle / 2.
    scale = float(np.sinc(angle / (2. * math.pi)))
    return np.array([x + distance * scale * math.cos(mid),
                     y + distance * scale * math.sin(mid), yaw + angle])


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Controller:
    def __init__(self, preset='carla', wheelbase=2.8604714913890885, max_steer_deg=69.99999237060547,
                 steering_curve=None, lookahead=None, speed_window='near', dt=.05,
                 trajectory_dt=.25, stale_timeout=.5, history_seconds=2.,
                 max_throttle=.75, max_brake=1., max_steer=.8, steer_rate=2.,
                 longitudinal_mode='vendor', pi_kp=1., pi_ki=.25,
                 max_lookahead_time_s=.5, aim_interpolation='linear',
                 pursuit_frame='body', rear_slip_c_per_rad=None,
                 steer_inverse='nominal', track_width_m=None,
                 accel_kp=1., accel_ki=.3, accel_integral_limit=1.5,
                 jerk_limit=4., jerk_limit_brake=8., brake_hysteresis=.3,
                 throttle_map=None, brake_map=None, low_speed_brake_mps=0.,
                 plan_interp='linear', feedforward_tau_s=0., adaptive_stale=False, stale_factor=2.5,
                 feedforward_limit=None, terminal_approach=False):
        if longitudinal_mode not in ('vendor', 'pi', 'accel'):
            raise ValueError('longitudinal_mode must be vendor, pi or accel')
        # 'accel': acceleration command = plan feedforward + PI on speed, jerk-limited, then the
        # measured MKZ throttle/brake -> acceleration maps are inverted, so decelerations the
        # engine drag can give (coast is about -2.8 m/s^2) never touch the brake, whose least
        # application already gives about -3.2 m/s^2.
        self.throttle_map = np.array(throttle_map if throttle_map is not None else
            [(0., -2.8), (.125, -2.0), (.2, -1.65), (.3, -.7), (.4, 0.), (.5, .6),
             (.6, 1.5), (.7, 2.3), (.75, 3.)], dtype=float)
        self.brake_map = np.array(brake_map if brake_map is not None else
            [(0., -2.8), (.05, -3.2), (.25, -3.9), (.45, -4.5), (.75, -6.3), (1., -8.)], dtype=float)
        for table in (self.throttle_map, self.brake_map):
            if table.ndim != 2 or table.shape[1] != 2 or not np.isfinite(table).all() or (np.diff(table[:, 0]) <= 0).any():
                raise ValueError('actuator maps need increasing command and finite acceleration')
        if (np.diff(self.throttle_map[:, 1]) <= 0).any() or (np.diff(self.brake_map[:, 1]) >= 0).any():
            raise ValueError('throttle map must increase and brake map decrease in acceleration')
        accel_values = [accel_kp, accel_ki, accel_integral_limit, jerk_limit, jerk_limit_brake, brake_hysteresis]
        if not np.all(np.isfinite(accel_values)) or min(accel_values) < 0 or jerk_limit <= 0 or jerk_limit_brake <= 0:
            raise ValueError('accel-mode gains and limits must be finite, jerk limits positive')
        self.accel_kp, self.accel_ki, self.accel_integral_limit = float(accel_kp), float(accel_ki), float(accel_integral_limit)
        self.jerk_limit, self.jerk_limit_brake, self.brake_hysteresis = float(jerk_limit), float(jerk_limit_brake), float(brake_hysteresis)
        # The maps were measured at 2-9 m/s. Near standstill there is no engine drag and any
        # throttle creeps forward, so below this speed a deceleration command always brakes.
        # Low-rate / sparse plans (e.g. VLA planners at 1-2 Hz): 'pchip' reads plan timing through a
        # monotone cubic of station over time, so sparse points still give continuous speed and
        # acceleration; feedforward_tau_s low-passes the plan feedforward across plan switches;
        # adaptive_stale scales the stale timeout with the measured plan period.
        if plan_interp not in ('linear', 'pchip'):
            raise ValueError('plan_interp must be linear or pchip')
        self.plan_interp, self.adaptive_stale = plan_interp, bool(adaptive_stale)
        self.feedforward_tau_s, self.stale_factor = float(feedforward_tau_s), float(stale_factor)
        if not np.isfinite([self.feedforward_tau_s, self.stale_factor]).all() or self.feedforward_tau_s < 0 or self.stale_factor < 1:
            raise ValueError('feedforward_tau_s must be >= 0 and stale_factor >= 1')
        # Plan feedforward beyond what the vehicle can do (e.g. a lagging car handed a catch-up jump
        # in the first plan segment) carries no information; clip it to (low, high) m/s^2.
        self.feedforward_limit = None if feedforward_limit is None else tuple(map(float, feedforward_limit))
        if self.feedforward_limit is not None and not (len(self.feedforward_limit) == 2
                                                       and self.feedforward_limit[0] < 0 < self.feedforward_limit[1]):
            raise ValueError('feedforward_limit must be (negative, positive)')
        # Low-rate plans: once the time-indexed remainder of a plan is parked but its endpoint is still
        # ahead, approach the endpoint by position (projection on the whole plan, sqrt-profile speed)
        # instead of treating the plan as stationary and holding short of it.
        self.terminal_approach = bool(terminal_approach)
        self.low_speed_brake_mps = float(low_speed_brake_mps)
        if not math.isfinite(self.low_speed_brake_mps) or self.low_speed_brake_mps < 0:
            raise ValueError('low_speed_brake_mps must be finite and nonnegative')
        self.longitudinal_mode = longitudinal_mode
        if preset not in ('carla', 'tcp', 'pursuit'):
            raise ValueError('unknown controller preset')
        if aim_interpolation not in ('linear', 'hermite'):
            raise ValueError('aim_interpolation must be linear or hermite')
        if aim_interpolation == 'hermite' and preset != 'pursuit':
            raise ValueError('hermite aim_interpolation requires pursuit preset')
        self.aim_interpolation = aim_interpolation
        if pursuit_frame not in ('body', 'rear_slip'):
            raise ValueError('pursuit_frame must be body or rear_slip')
        if steer_inverse not in ('nominal', 'ackermann'):
            raise ValueError('steer_inverse must be nominal or ackermann')
        if (pursuit_frame != 'body' or steer_inverse != 'nominal') and preset != 'pursuit':
            raise ValueError('rear_slip pursuit_frame and ackermann steer_inverse require pursuit preset')
        # A parameter without its switch (or a switch without its parameter) is a config error,
        # never a silent no-op.
        if (pursuit_frame == 'rear_slip') != (rear_slip_c_per_rad is not None):
            raise ValueError('rear_slip_c_per_rad is required by, and only by, pursuit_frame=rear_slip')
        if (steer_inverse == 'ackermann') != (track_width_m is not None):
            raise ValueError('track_width_m is required by, and only by, steer_inverse=ackermann')
        self.pursuit_frame, self.steer_inverse = pursuit_frame, steer_inverse
        self.rear_slip_c_per_rad = self.track_width_m = None
        if rear_slip_c_per_rad is not None:
            self.rear_slip_c_per_rad = float(rear_slip_c_per_rad)
            if not math.isfinite(self.rear_slip_c_per_rad) or self.rear_slip_c_per_rad <= 0:
                raise ValueError('rear_slip_c_per_rad must be positive and finite')
        if track_width_m is not None:
            self.track_width_m = float(track_width_m)
            if not math.isfinite(self.track_width_m) or not 0 < self.track_width_m < 2 * wheelbase:
                raise ValueError('track_width_m must be positive, finite and below twice the wheelbase')
        if speed_window not in ('near', 'reference'):
            raise ValueError('speed_window must be near or reference')
        self.lookahead = lookahead or ('fixed4' if preset == 'tcp' else 'additive')
        if self.lookahead not in ('additive', 'max', 'fixed4'):
            raise ValueError('lookahead must be additive, max or fixed4')
        try:
            self.max_lookahead_time_s = float(max_lookahead_time_s)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('max_lookahead_time_s must be positive and finite')
        if not math.isfinite(self.max_lookahead_time_s) or self.max_lookahead_time_s <= 0.:
            raise ValueError('max_lookahead_time_s must be positive and finite')
        values = [wheelbase, max_steer_deg, dt, trajectory_dt, stale_timeout,
                  history_seconds, max_throttle, max_brake, max_steer, steer_rate]
        if not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError('controller dimensions and limits must be positive finite values')
        if max_steer_deg >= 90 or max(max_throttle, max_brake, max_steer) > 1:
            raise ValueError('invalid steering angle or actuator limit')
        if history_seconds < stale_timeout:
            raise ValueError('history must cover the stale timeout')
        self.preset, self.wheelbase = preset, float(wheelbase)
        self.max_steer_rad = math.radians(max_steer_deg)
        # CARLA steering_curve x is speed in km/h, y is steering scale.
        curve = np.array(steering_curve if steering_curve is not None else
                         [(0., 1.), (20., .9), (60., .8), (120., .7)], dtype=float)
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 1 or
                not np.isfinite(curve).all() or (curve[:, 1] <= 0).any() or
                (np.diff(curve[:, 0]) <= 0).any()):
            raise ValueError('steering_curve must contain increasing speed and positive scale')
        self.steering_curve = curve
        self.speed_window, self.dt, self.trajectory_dt = speed_window, dt, trajectory_dt
        self.stale_timeout, self.history_seconds = stale_timeout, history_seconds
        self.max_throttle, self.max_brake = max_throttle, max_brake
        self.max_steer, self.steer_rate = max_steer, steer_rate
        tcp = preset == 'tcp'
        self.lateral = WindowPID(.75, .75, .3, 40, 'tcp', dt) if tcp else WindowPID(1.95, .05, .2, 10, dt=dt)
        self.longitudinal = WindowPID(5., .5, 1., 40, 'tcp', dt) if tcp else WindowPID(1., .05, 0., 10, dt=dt)
        self.longitudinal_pi = ConditionalPI(-self.max_brake, self.max_throttle, kp=pi_kp, ki=pi_ki)
        self.reset()

    @property
    def diagnostics(self):
        """Return a JSON-compatible snapshot; callers cannot mutate state."""
        result = dict(self._diagnostics)
        if result.get('aim_xy') is not None:
            result['aim_xy'] = list(result['aim_xy'])
        return result

    def reset(self):
        self.lateral.reset()
        self.longitudinal.reset()
        self.longitudinal_pi.reset()
        self._accel_integral = 0.
        self._feedforward = None
        self._plan_period = None
        self._accel_command = None
        self._braking = False
        self._history = deque()
        self._pose = np.zeros(3)
        self._last_time = None
        self._source_time = None
        self._pending = None
        self._points = None
        self._arc = None
        self._point_dt = self.trajectory_dt
        self._last_steer = 0.
        self._stop_latched = False
        self._last_control = (0., 0., self.max_brake)
        self._rejection = None
        self._diagnostics = self._empty_diagnostics('no_trajectory')

    def _empty_diagnostics(self, reason):
        return dict(trajectory_time=self._source_time, trajectory_age_s=None,
                    target_speed_mps=None, reference_speed_mps=None, aim_xy=None,
                    cross_track_m=None, heading_error_rad=None, reason=reason,
                    update_rejection=self._rejection,
                    aim_interpolation=self.aim_interpolation,
                    aim_interpolation_used=None, aim_interpolation_fallback=None,
                    longitudinal_mode=self.longitudinal_mode,
                    longitudinal_integral_effort=self.longitudinal_pi.integral if self.longitudinal_mode == 'pi' else None,
                    longitudinal_effort=None)

    def update(self, traj_xy, t_frame, trajectory_dt=None):
        """Queue future points at +dt, +2dt, ... after t_frame.

        Default: exactly 20 points at the constructor trajectory_dt (the frozen contract).
        With an explicit trajectory_dt: any N >= 1 points; the horizon is N*dt and is never
        extrapolated (e.g. TCP: 4 points at .5 s).
        """
        try:
            points = np.asarray(traj_xy, dtype=float)
            stamp = float(t_frame)
            dt = self.trajectory_dt if trajectory_dt is None else float(trajectory_dt)
        except (TypeError, ValueError, OverflowError):
            return self._reject('invalid_trajectory')
        shape_ok = (points.shape == (20, 2) if trajectory_dt is None else
                    points.ndim == 2 and points.shape[1] == 2 and len(points) >= 1)
        if (not shape_ok or not np.isfinite(points).all() or not math.isfinite(stamp)
                or not math.isfinite(dt) or dt <= 0):
            return self._reject('invalid_trajectory')
        newest = self._pending[1] if self._pending is not None else self._source_time
        if newest is not None and stamp <= newest + 1e-9:
            self._rejection = 'duplicate_trajectory' if abs(stamp - newest) <= 1e-9 else 'out_of_order_trajectory'
            return False
        # Resolve the source pose only after this tick's motion was integrated.
        self._pending = (points.copy(), stamp, dt)
        return True

    def _reject(self, reason):
        self._pending = self._points = None
        self._rejection = reason
        return False

    def _pose_at(self, stamp):
        if not self._history or stamp < self._history[0][0] - 1e-8:
            return None
        for index in range(len(self._history) - 1, -1, -1):
            t, pose, _, _ = self._history[index]
            if stamp >= t - 1e-8:
                if index == len(self._history) - 1 or abs(stamp - t) < 1e-8:
                    return pose.copy()
                _, _, v, w = self._history[index + 1]
                return advance_pose(pose, v, w, stamp - t)
        return None

    def _accept_pending(self, now):
        if self._pending is None:
            return
        points, stamp, dt = self._pending
        self._pending = None
        if stamp > now + 1e-8:
            self._reject('future_trajectory')
            return
        source_pose = self._pose_at(stamp)
        if source_pose is None:
            self._reject('trajectory_outside_history')
            return
        points = np.vstack((np.zeros(2), points))
        self._points = points @ _rotation(source_pose[2]).T + source_pose[:2]
        self._arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        self._times = np.arange(len(points)) * dt
        self._point_dt = dt
        self._stationary_tail = bool(np.linalg.norm(points[-1] - points[-2]) < 1e-6)
        if self._source_time is not None and stamp > self._source_time:
            period = stamp - self._source_time
            self._plan_period = period if self._plan_period is None else .7 * self._plan_period + .3 * period
        self._source_time = stamp
        self._rejection = None

    def _speed_at(self, age, start, end):
        a, b = age + start, age + end
        if b > self._times[-1] + 1e-8 and not self._stationary_tail:
            return None
        if self.plan_interp == 'pchip':
            return float(np.diff(pchip(self._times, self._arc, np.array([a, b])))[0] / (b - a))
        return float((np.interp(b, self._times, self._arc) -
                      np.interp(a, self._times, self._arc)) / (b - a))

    def _lateral_aim(self, points, station):
        """Aim-only C1 chord-length Hermite; projection/timing stay piecewise linear.

        Secant derivatives are convex weighted vector averages, so the geometry
        commutes with rotations/translations. Repeated arc knots are removed only
        in this interpolation view. Guards inspect the queried segment's local
        stencil, never reject a moving path merely for a distant stationary tail.
        """
        linear = np.array([np.interp(station, self._arc, points[:, axis]) for axis in (0, 1)])
        self._diagnostics.update(aim_interpolation=self.aim_interpolation,
                                 aim_interpolation_used='linear', aim_interpolation_fallback=None)
        if self.aim_interpolation == 'linear':
            return linear

        def fallback(reason):
            self._diagnostics['aim_interpolation_fallback'] = reason
            return linear

        if not math.isfinite(station) or not np.isfinite(self._arc).all():
            return fallback('nonfinite_station_or_arc')
        if np.any(np.diff(self._arc) < 0.):
            return fallback('nonmonotonic_arc')
        keep = np.r_[True, np.diff(self._arc) > 1e-8]
        arc, knots = self._arc[keep], points[keep]
        if len(arc) <= 2:
            return fallback('two_or_fewer_distinct_knots')
        query = float(np.clip(station, arc[0], arc[-1]))
        segment = min(max(0, int(np.searchsorted(arc, query, side='right')) - 1), len(arc) - 2)
        begin, end = max(0, segment - 1), min(len(arc), segment + 3)
        if not np.isfinite(knots[begin:end]).all():
            return fallback('nonfinite_local_knots')
        lengths = np.diff(arc[begin:end])
        slopes = np.diff(knots[begin:end], axis=0) / lengths[:, None]
        if not np.isfinite(slopes).all():
            return fallback('nonfinite_local_tangent')
        # A local reversal/cusp is not repaired by rounding it with a spline.
        if len(slopes) > 1 and np.any(np.sum(slopes[:-1] * slopes[1:], axis=1) <= 0.):
            return fallback('local_tangent_reversal')
        local = segment - begin
        secant = slopes[local]
        left = secant if segment == 0 else (
            slopes[local-1] * lengths[local] + secant * lengths[local-1]
        ) / (lengths[local-1] + lengths[local])
        right = secant if segment + 1 == len(arc) - 1 else (
            secant * lengths[local+1] + slopes[local+1] * lengths[local]
        ) / (lengths[local] + lengths[local+1])
        if (not np.isfinite(np.r_[left, right]).all()
                or min(float(left @ secant), float(right @ secant)) <= 0.):
            return fallback('invalid_local_tangent')
        h = arc[segment+1] - arc[segment]
        u = (query - arc[segment]) / h
        aim = ((2*u**3 - 3*u**2 + 1) * knots[segment] + (u**3 - 2*u**2 + u) * h * left
               + (-2*u**3 + 3*u**2) * knots[segment+1] + (u**3 - u**2) * h * right)
        # The endpoint slopes have nonnegative projection onto their chord.
        # Keep a numerical guard on the evaluated point against backward/forward
        # overshoot; lateral bowing is intentional (e.g. an actual circular arc).
        chord = knots[segment+1] - knots[segment]
        chord_squared = float(chord @ chord)
        along = float((aim - knots[segment]) @ chord)
        tolerance = 1e-10 * max(1., chord_squared)
        if not np.isfinite(aim).all() or along < -tolerance or along > chord_squared + tolerance:
            return fallback('invalid_local_interpolation')
        self._diagnostics['aim_interpolation_used'] = 'hermite'
        return aim

    def _accel_actuation(self, age, desired, speed, elapsed):
        early, late = self._speed_at(age, 0., .5), self._speed_at(age, .5, 1.)
        feedforward = 0. if early is None or late is None else (late - early) / .5
        if self.feedforward_limit is not None:
            feedforward = float(np.clip(feedforward, *self.feedforward_limit))
        if self.feedforward_tau_s > 0.:
            if self._feedforward is not None:
                feedforward = self._feedforward + (feedforward - self._feedforward) * min(1., elapsed / self.feedforward_tau_s)
            self._feedforward = feedforward
        error = desired - speed
        raw = feedforward + self.accel_kp * error + self._accel_integral
        low, high = float(self.brake_map[-1, 1]), float(self.throttle_map[-1, 1])
        # Conditional integration: no charging further into actuator saturation.
        if not ((raw >= high and error > 0) or (raw <= low and error < 0)):
            self._accel_integral = float(np.clip(self._accel_integral + self.accel_ki * error * elapsed,
                                                 -self.accel_integral_limit, self.accel_integral_limit))
        command = float(np.clip(feedforward + self.accel_kp * error + self._accel_integral, low, high))
        if self._accel_command is not None:
            command = float(np.clip(command, self._accel_command - self.jerk_limit_brake * elapsed,
                                    self._accel_command + self.jerk_limit * elapsed))
        self._accel_command = command
        coast = float(self.throttle_map[0, 1])
        if speed < self.low_speed_brake_mps and command < 0.:
            # Crawl speed: brake in proportion to the command over the full brake range.
            self._braking = True
            brake = float(np.clip(-command / -float(self.brake_map[-1, 1]), 0., 1.))
            self._diagnostics.update(accel_feedforward_mps2=feedforward, accel_command_mps2=command,
                                     accel_integral_mps2=self._accel_integral, accel_low_speed_brake=True)
            return 0., min(brake, self.max_brake)
        # Brake only below coast with hysteresis, so the actuator does not chatter across the gap.
        self._braking = command < coast - (0. if self._braking else self.brake_hysteresis)
        if self._braking:
            brake = float(np.interp(-command, -self.brake_map[:, 1], self.brake_map[:, 0]))
            throttle = 0.
        else:
            throttle = float(np.interp(command, self.throttle_map[:, 1], self.throttle_map[:, 0]))
            brake = 0.
        self._diagnostics.update(accel_feedforward_mps2=feedforward, accel_command_mps2=command,
                                 accel_integral_mps2=self._accel_integral)
        return min(throttle, self.max_throttle), min(brake, self.max_brake)

    def _geometry(self, age):
        points = (self._points - self._pose[:2]) @ _rotation(self._pose[2])
        # Keep the segment straddling the current reference time. The origin is
        # an auxiliary timed point, never a claimed measured path-error metric.
        first = min(int(max(age, 0.) / self._point_dt), len(points) - 2)
        segments = points[first + 1:] - points[first:-1]
        lengths = np.linalg.norm(segments, axis=1)
        usable = lengths > 1e-8
        if not usable.any():
            return points, None
        indices = np.where(usable)[0]
        starts, vectors = points[first + indices], segments[indices]
        fractions = np.clip(-np.sum(starts * vectors, axis=1) / lengths[indices] ** 2, 0., 1.)
        projections = starts + vectors * fractions[:, None]
        best = int(np.argmin(np.sum(projections ** 2, axis=1)))
        segment = first + indices[best]
        tangent = vectors[best] / lengths[indices[best]]
        projection = projections[best]
        station = self._arc[segment] + fractions[best] * lengths[indices[best]]
        cross_track = float(tangent[0] * -projection[1] - tangent[1] * -projection[0])
        heading = math.atan2(tangent[1], tangent[0])
        return points, (station, cross_track, heading)

    def _safe(self, reason, elapsed=None):
        self._accel_integral, self._accel_command, self._braking = 0., None, True
        self._feedforward = None
        if self.longitudinal_mode == 'pi':
            self.longitudinal_pi.reset()
            self._diagnostics['longitudinal_integral_effort'] = 0.
        self._diagnostics['longitudinal_effort'] = -float(self.max_brake)
        self._diagnostics['reason'] = reason
        amount = self.steer_rate * (self.dt if elapsed is None else elapsed)
        self._last_steer = float(np.clip(0., self._last_steer - amount, self._last_steer + amount))
        self._last_control = (0., self._last_steer, float(self.max_brake))
        return self._last_control

    def step(self, t_now, speed_mps, yaw_rate_rps):
        try:
            now, speed, yaw_rate = float(t_now), float(speed_mps), float(yaw_rate_rps)
        except (TypeError, ValueError, OverflowError):
            self._diagnostics = self._empty_diagnostics('invalid_motion')
            return self._safe('invalid_motion')
        if not np.isfinite([now, speed, yaw_rate]).all() or speed < 0:
            self._diagnostics = self._empty_diagnostics('invalid_motion')
            return self._safe('invalid_motion')
        elapsed = self.dt if self._last_time is None else now - self._last_time
        if self._last_time is not None and elapsed < -1e-8:
            self.reset()
            self._diagnostics['reason'] = 'time_regression'
            return self._safe('time_regression')
        if self._last_time is not None and elapsed <= 1e-8:
            self._diagnostics['reason'] = 'duplicate_tick'
            return self._last_control
        if self.longitudinal_mode == 'pi' and self._last_time is not None and elapsed > .2 + 1e-8:
            # A missing motion interval is not an opportunity to integrate a
            # large unobserved PI correction or accept an invented pose history.
            self.reset()
            self._last_time = now
            self._history.append((now, self._pose.copy(), speed, yaw_rate))
            return self._safe('motion_gap')
        if self._last_time is not None:
            self._pose = advance_pose(self._pose, speed, yaw_rate, elapsed)
        self._history.append((now, self._pose.copy(), speed, yaw_rate))
        self._last_time = now
        while len(self._history) > 2 and self._history[1][0] < now - self.history_seconds:
            self._history.popleft()
        self._accept_pending(now)
        self._diagnostics = self._empty_diagnostics('tracking')
        if self._points is None:
            return self._safe(self._rejection or 'no_trajectory', elapsed)
        age = max(0., now - self._source_time)
        self._diagnostics['trajectory_age_s'] = age
        stale = self.stale_timeout
        if self.adaptive_stale and self._plan_period is not None:
            # Keep at least 1 s of plan ahead for the speed and feedforward windows.
            stale = max(stale, min(self.stale_factor * self._plan_period, self._times[-1] - 1.))
            self._diagnostics['stale_timeout_s'] = stale
        if age > stale + 1e-8:
            return self._safe('stale_trajectory', elapsed)
        start, end = (0., .25) if self.speed_window == 'near' else (.25, 1.)
        desired = self._speed_at(age, start, end)
        reference = self._speed_at(age, 0., min(.001, self._point_dt))
        if desired is None or reference is None:
            return self._safe('trajectory_horizon_exhausted', elapsed)
        self._diagnostics.update(target_speed_mps=desired, reference_speed_mps=reference)
        points, geometry = self._geometry(age)
        if self.terminal_approach and self._stationary_tail:
            if geometry is None:
                points, geometry = self._geometry(0.)
            if geometry is not None:
                remaining = float(self._arc[-1] - geometry[0])
                if remaining > .3 and points[-1, 0] > 0. and desired < .4:
                    desired = max(desired, min(1.5, math.sqrt(1.6 * (remaining - .15))))
                    self._diagnostics.update(terminal_approach_remaining_m=remaining, target_speed_mps=desired)
        if geometry is None:
            return self._safe('stationary_trajectory', elapsed)
        if not np.any(points[1:, 0] > 1e-6) and desired > .05:
            return self._safe('trajectory_behind', elapsed)
        station, cross_track, heading = geometry
        distance = {'additive': 3. + .5 * speed,
                    'max': max(3., self.max_lookahead_time_s * speed), 'fixed4': 4.}[self.lookahead]
        aim_station = min(station + distance, self._arc[-1])
        aim = self._lateral_aim(points, aim_station)
        bearing = math.atan2(aim[1], aim[0])
        if self.preset == 'pursuit':
            target = aim
            if self.pursuit_frame == 'rear_slip':
                # Express the aim point in the rear-axle velocity frame, which lies -beta from the
                # body axis: the pursuit arc must be tangent to the actual axle motion.
                beta = rear_slip_angle(speed, yaw_rate, self.rear_slip_c_per_rad)
                cb, sb = math.cos(beta), math.sin(beta)
                target = np.array([cb * aim[0] - sb * aim[1], sb * aim[0] + cb * aim[1]])
                self._diagnostics['pursuit_rear_slip_rad'] = beta
            curvature = 2. * target[1] / max(float(target @ target), 1e-8)
            if self.steer_inverse == 'ackermann':
                angle = ackermann_inner_angle(curvature, self.wheelbase, self.track_width_m)
            else:
                angle = math.atan(self.wheelbase * curvature)
            if self.pursuit_frame != 'body' or self.steer_inverse != 'nominal':
                self._diagnostics.update(pursuit_curvature_inv_m=curvature, pursuit_nominal_angle_rad=angle)
            scale = float(np.interp(speed * 3.6, self.steering_curve[:, 0], self.steering_curve[:, 1]))
            raw_steer = -angle / (self.max_steer_rad * scale)
        else:
            raw_steer = -self.lateral.step(bearing / (math.pi / 2.) if self.preset == 'tcp' else bearing)
        steer = float(np.clip(raw_steer, self._last_steer - self.steer_rate * elapsed,
                              self._last_steer + self.steer_rate * elapsed))
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        if self.longitudinal_mode == 'pi':
            effort = self.longitudinal_pi.step(desired - speed, elapsed)
            throttle, brake = max(effort, 0.), max(-effort, 0.)
            self._diagnostics.update(longitudinal_integral_effort=self.longitudinal_pi.integral,
                                     longitudinal_unsaturated_effort=self.longitudinal_pi.raw_effort,
                                     longitudinal_integration_limited=self.longitudinal_pi.integration_limited,
                                     longitudinal_kp=self.longitudinal_pi.kp, longitudinal_ki=self.longitudinal_pi.ki)
        elif self.longitudinal_mode == 'accel':
            throttle, brake = self._accel_actuation(age, desired, speed, elapsed)
        elif self.preset == 'tcp':
            brake = float(desired < .4 or speed > desired * 1.1)
            throttle = float(np.clip(self.longitudinal.step(np.clip(desired - speed, 0., .25)), 0., self.max_throttle))
            brake = min(brake, self.max_brake)
            if brake:
                throttle = 0.
        else:
            acceleration = self.longitudinal.step((desired - speed) * 3.6)
            throttle = min(max(acceleration, 0.), self.max_throttle)
            brake = min(max(-acceleration, 0.), self.max_brake)
        reason = 'tracking'
        endpoint_distance = float(np.linalg.norm(points[-1]))
        # A stationary endpoint close to the axle is a parking target. Latch
        # brake hold to avoid repeated tiny origin-to-endpoint speed commands
        # reaccelerating the vehicle after it first stops. A new moving plan or
        # a meaningfully displaced endpoint releases the latch.
        if desired > .4 or endpoint_distance > .25 or not self._stationary_tail:
            self._stop_latched = False
        if self._stationary_tail and desired < .4 and speed < .1 and endpoint_distance < .2:
            self._stop_latched = True
        if self._stop_latched or (desired < .05 and speed < .1):
            throttle, brake, reason = 0., self.max_brake, 'stop_hold'
            self._diagnostics['target_speed_mps'] = 0.
            if self.longitudinal_mode == 'pi':
                self.longitudinal_pi.reset()
                self._diagnostics['longitudinal_integral_effort'] = 0.
            self._accel_integral, self._accel_command, self._braking = 0., None, True
        self._diagnostics['longitudinal_effort'] = float(throttle - brake)
        self._diagnostics.update(aim_xy=aim.tolist(), cross_track_m=cross_track,
                                heading_error_rad=heading, reason=reason,
                                raw_steer=float(raw_steer), steer_limited=abs(steer - raw_steer) > 1e-8,
                                lookahead_m=distance, speed_window=self.speed_window)
        self._last_steer = steer
        self._last_control = (float(throttle), steer, float(brake))
        return self._last_control


ADAPTER_KEYS = ('rear_axle_offset_m', 'pose_lateral_coefficient_s2_per_m')


def pursuit_from_config(path):
    """A pursuit Controller from an L1 JSON controller config; pose-adapter keys are dropped."""
    with open(path) as stream:
        config = json.load(stream)
    return Controller(preset='pursuit', **{k: v for k, v in config.items() if k not in ADAPTER_KEYS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.selftest:
        parser.error('use --selftest')
    from b2d_controller_selftest import run_suite
    result = run_suite()
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(output + '\n')
    print(output)
    return 0 if result['candidate_main_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
