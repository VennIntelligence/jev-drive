"""NumPy-only trajectory control, Python 3.8 compatible.

Input: 20 future rear-axle points, x forward / y left, sampled at .25 s.
Output: throttle, CARLA steer (right positive), brake. Yaw rate is left positive.
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
                 longitudinal_mode='vendor', pi_kp=1., pi_ki=.25):
        if longitudinal_mode not in ('vendor', 'pi'):
            raise ValueError('longitudinal_mode must be vendor or pi')
        self.longitudinal_mode = longitudinal_mode
        if preset not in ('carla', 'tcp', 'pursuit'):
            raise ValueError('unknown controller preset')
        if speed_window not in ('near', 'reference'):
            raise ValueError('speed_window must be near or reference')
        self.lookahead = lookahead or ('fixed4' if preset == 'tcp' else 'additive')
        if self.lookahead not in ('additive', 'max', 'fixed4'):
            raise ValueError('lookahead must be additive, max or fixed4')
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
        self._history = deque()
        self._pose = np.zeros(3)
        self._last_time = None
        self._source_time = None
        self._pending = None
        self._points = None
        self._arc = None
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
                    longitudinal_mode=self.longitudinal_mode,
                    longitudinal_integral_effort=self.longitudinal_pi.integral if self.longitudinal_mode == 'pi' else None,
                    longitudinal_effort=None)

    def update(self, traj_xy, t_frame):
        try:
            points = np.asarray(traj_xy, dtype=float)
            stamp = float(t_frame)
        except (TypeError, ValueError, OverflowError):
            return self._reject('invalid_trajectory')
        if points.shape != (20, 2) or not np.isfinite(points).all() or not math.isfinite(stamp):
            return self._reject('invalid_trajectory')
        newest = self._pending[1] if self._pending is not None else self._source_time
        if newest is not None and stamp <= newest + 1e-9:
            self._rejection = 'duplicate_trajectory' if abs(stamp - newest) <= 1e-9 else 'out_of_order_trajectory'
            return False
        # Resolve the source pose only after this tick's motion was integrated.
        self._pending = (points.copy(), stamp)
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
        points, stamp = self._pending
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
        self._times = np.arange(21) * self.trajectory_dt
        self._stationary_tail = bool(np.linalg.norm(points[-1] - points[-2]) < 1e-6)
        self._source_time = stamp
        self._rejection = None

    def _speed_at(self, age, start, end):
        a, b = age + start, age + end
        if b > self._times[-1] + 1e-8 and not self._stationary_tail:
            return None
        return float((np.interp(b, self._times, self._arc) -
                      np.interp(a, self._times, self._arc)) / (b - a))

    def _geometry(self, age):
        points = (self._points - self._pose[:2]) @ _rotation(self._pose[2])
        # Keep the segment straddling the current reference time. The origin is
        # an auxiliary timed point, never a claimed measured path-error metric.
        first = min(int(max(age, 0.) / self.trajectory_dt), len(points) - 2)
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
        if age > self.stale_timeout + 1e-8:
            return self._safe('stale_trajectory', elapsed)
        start, end = (0., .25) if self.speed_window == 'near' else (.25, 1.)
        desired = self._speed_at(age, start, end)
        reference = self._speed_at(age, 0., min(.001, self.trajectory_dt))
        if desired is None or reference is None:
            return self._safe('trajectory_horizon_exhausted', elapsed)
        self._diagnostics.update(target_speed_mps=desired, reference_speed_mps=reference)
        points, geometry = self._geometry(age)
        if geometry is None:
            return self._safe('stationary_trajectory', elapsed)
        if not np.any(points[1:, 0] > 1e-6) and desired > .05:
            return self._safe('trajectory_behind', elapsed)
        station, cross_track, heading = geometry
        distance = {'additive': 3. + .5 * speed, 'max': max(3., .5 * speed), 'fixed4': 4.}[self.lookahead]
        aim_station = min(station + distance, self._arc[-1])
        aim = np.array([np.interp(aim_station, self._arc, points[:, axis]) for axis in (0, 1)])
        bearing = math.atan2(aim[1], aim[0])
        if self.preset == 'pursuit':
            curvature = 2. * aim[1] / max(float(aim @ aim), 1e-8)
            angle = math.atan(self.wheelbase * curvature)
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
        self._diagnostics['longitudinal_effort'] = float(throttle - brake)
        self._diagnostics.update(aim_xy=aim.tolist(), cross_track_m=cross_track,
                                heading_error_rad=heading, reason=reason,
                                raw_steer=float(raw_steer), steer_limited=abs(steer - raw_steer) > 1e-8,
                                lookahead_m=distance, speed_window=self.speed_window)
        self._last_steer = steer
        self._last_control = (float(throttle), steer, float(brake))
        return self._last_control


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
