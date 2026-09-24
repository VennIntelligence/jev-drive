"""TFv6 author execution formulas on a shared, time-parameterized L1 plan.

The vendor controls live in LEAD. This adapter changes only the plan interface:
20 points at 0.25 s become the original eight waypoint points for B; A receives
eight approximately 1 m route checkpoints and the same p2-to-p4 speed used by B.
No model or route planner is used here. Local input axes are forward/left.
"""
import math

import numpy as np


PARAMS = np.array([1.1990342347353184, -0.8057602384167799,
                   1.710818710950062, 0.921890257450335,
                   1.556497522998393, -0.7013479734904027,
                   1.031266635497984])


class WindowPID:
    def __init__(self, kp, ki, kd, size, prefill=True):
        self.kp, self.ki, self.kd, self.size, self.prefill = kp, ki, kd, size, prefill
        self.reset()

    def reset(self):
        self.history = [0.] * self.size if self.prefill else []

    def step(self, error):
        self.history.append(float(error))
        self.history = self.history[-self.size:]
        derivative = 0. if len(self.history) == 1 else self.history[-1] - self.history[-2]
        return self.kp * error + self.ki * float(np.mean(self.history)) + self.kd * derivative


def _spatial_route(points, n=8):
    points = np.asarray(points, float)
    path = np.vstack((np.zeros(2), points))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(arc) > 1e-4]
    path, arc = path[keep], arc[keep]
    if len(path) < 2:
        return np.zeros((n, 2))
    locations = np.minimum(np.arange(1, n+1, dtype=float), arc[-1])
    return np.column_stack((np.interp(locations, arc, path[:, 0]),
                            np.interp(locations, arc, path[:, 1])))


def _author_throttle(target, speed):
    if target < 1e-5:
        return 0., 1.
    target = max(target, 1./3.6)
    current_kph, target_kph = speed * 3.6, target * 3.6
    error = target_kph - current_kph
    if error > 1.89:
        return 1., 0.
    if current_kph / target_kph > PARAMS[-1]:
        return 0., 1.
    clipped = max(error, 0.) / 100.
    current = current_kph / 100.
    features = np.array([current, current**2, 100*clipped, clipped**2,
                         current*clipped, current**2*clipped])
    return float(np.clip(features @ PARAMS[:-1], 0, 1)), 0.


class AuthorController:
    accepts_route = True

    def __init__(self, mode):
        if mode not in ('route', 'waypoint'):
            raise ValueError('mode must be route or waypoint')
        self.mode = mode
        self.turn = WindowPID(1.25, .75, .3, 20)
        self.speed = WindowPID(1.75, 1., 2., 20)
        self.route_turn = WindowPID(3.118357247806046, .6406067986034124,
                                    1.3782508892109167, 6, prefill=False)
        self.reset()

    def reset(self):
        for controller in (self.turn, self.speed, self.route_turn): controller.reset()
        self.points = self.route_points = None
        self.diagnostics = {'reason': 'no_plan'}

    def update(self, traj_xy, t_frame, trajectory_dt=None, route_xy=None):
        # route_xy: optional spatial route checkpoints (local forward/left), the route PID's
        # native input; without it the route is resampled from the time trajectory.
        points = np.asarray(traj_xy, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
            self.points = None
            self.diagnostics = {'reason': 'invalid_plan'}
            return False
        self.points = points
        self.route_points = None if route_xy is None else np.asarray(route_xy, float)
        self.t_frame = float(t_frame)
        self.plan_dt = float(trajectory_dt or .25)
        self.diagnostics = {'reason': 'tracking'}
        return True

    def step(self, t_now, speed_mps, yaw_rate_rps):
        if self.points is None or not np.isfinite(speed_mps) or t_now - self.t_frame > .5:
            self.diagnostics = {'reason': 'stale_or_invalid'}
            return 0., 0., 1.
        speed = max(float(speed_mps), 0.)
        # The L1 adapter samples the same p2/p4 physical interval used by TFv6 B.
        sample_t = np.arange(1, 9) * .25
        raw_t = np.arange(len(self.points)+1) * self.plan_dt
        path = np.vstack((np.zeros(2), self.points))
        wp = np.column_stack((np.interp(sample_t, raw_t, path[:, 0]),
                              np.interp(sample_t, raw_t, path[:, 1])))
        desired = float(np.linalg.norm(wp[3] - wp[1]) * 2.)
        if self.mode == 'waypoint':
            brake = desired < .4 or speed / max(desired, 1e-6) > 1.1
            throttle = float(self.speed.step(np.clip(desired - speed, 0., .99)))
            if brake: throttle = 0.
            aim_distance = 2.25 if desired < 5.5 else 3.
            aim = wp[-1]
            for point in wp:
                if np.linalg.norm(point) >= aim_distance:
                    aim = point
                    break
            angle = -math.degrees(math.atan2(aim[1], aim[0])) / 90.
            if speed < .01 or brake: angle = 0.
            steer = float(np.clip(self.turn.step(angle), -1., 1.))
            brake = float(brake)
        else:
            route = _spatial_route(wp if self.route_points is None else self.route_points)
            # LEAD's route PID operates in CARLA forward/right. It chooses a
            # point index from speed and the eight 1 m checkpoints.
            index = int(min(np.clip(.9755321901954155 * speed*3.6 +
                                    1.9152884533402488, 24, 105)/10 - 2, 7))
            curvature = float(np.mean(np.abs(np.arctan2(route[:, 1], route[:, 0]))))
            index = min(index + int(np.clip(int(curvature * 3.5), 0, 2)), 7)
            aim = route[index]
            angle = -math.degrees(math.atan2(aim[1], aim[0])) / 90.
            steer = round(float(np.clip(self.route_turn.step(angle), -1., 1.)), 3)
            throttle, brake = _author_throttle(desired, speed)
        if brake > 0: throttle = 0.
        self.diagnostics = {'reason': 'tracking', 'target_speed_mps': desired,
                            'mode': self.mode, 'plan_age_s': t_now - self.t_frame}
        return float(throttle), float(steer), float(brake)
