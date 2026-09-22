"""Independent analytic-path validation with explicitly synthetic plant dynamics.

Geometry uses measured stock Lincoln dimensions; longitudinal force, instantaneous
bicycle steering and perturbations below are synthetic, not CARLA measurements.
No controller internals produce evaluation error or advance the vehicle state.
"""
import math
import time
from collections import deque
import numpy as np
from b2d_controller import Controller, WindowPID

DT = .05
WHEELBASE = 2.8604714913890885
MAX_ANGLE = math.radians(69.99999237060547)
CURVE = np.array([[0., 1.], [20., .9], [60., .8], [120., .7]])


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def _longitudinal(v, throttle, brake, dt, slope=0.):
    """Coulomb brake friction with static hold, signed velocity and exact stop.

A flat-road stop uses crossing-time integration. A slope exceeding available
brake force accelerates backwards, so hold is not made true by clipping v >= 0.
"""
    drive = 3. * throttle - .08 * v + slope
    friction = 8. * brake
    if abs(v) < 1e-10:
        a = math.copysign(max(abs(drive) - friction, 0.), drive)
        return a * dt, .5 * a * dt * dt
    a = drive - math.copysign(friction, v)
    end = v + a * dt
    if v * end >= 0:
        return end, (v + end) * .5 * dt
    crossing = -v / a
    remaining = dt - crossing
    restart_a = math.copysign(max(abs(drive) - friction, 0.), drive)
    return restart_a * remaining, .5 * v * crossing + .5 * restart_a * remaining ** 2


def _plant(state, control, wheelbase_gain=1., steering_gain=1., slope=0.):
    x, y, yaw, speed = state
    throttle, steer, brake = control
    new_speed, distance = _longitudinal(speed, throttle, brake, DT, slope)
    scale = np.interp(abs(speed) * 3.6, CURVE[:, 0], CURVE[:, 1])
    angle = -steer * MAX_ANGLE * scale * steering_gain
    curvature = math.tan(angle) / (WHEELBASE * wheelbase_gain)
    turn = distance * curvature
    if abs(curvature) > 1e-10:
        x += (math.sin(yaw + turn) - math.sin(yaw)) / curvature
        y += (-math.cos(yaw + turn) + math.cos(yaw)) / curvature
    else:
        x += distance * math.cos(yaw)
        y += distance * math.sin(yaw)
    # Motion is the interval-average observation; this isolates reprojection
    # mathematics from unspecified sensor endpoint sampling conventions.
    return np.array([x, y, yaw + turn, new_speed]), distance / DT, turn / DT


def _local(world, state):
    yaw = state[2]
    c, s = math.cos(yaw), math.sin(yaw)
    return (world - state[:2]) @ np.array([[c, -s], [s, c]])



class _BearingPIDExperiment(Controller):
    """Offline-only exact handoff lateral sum, under identical common plumbing.

The full CARLA bearing PID is added to pursuit before the shared slew/steer
limits. This class is never available as a production preset.
"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_bearing_pid = WindowPID(1.95, .05, .2, 10, dt=self.dt)

    def step(self, now, speed, yaw_rate):
        previous_steer = self._last_steer
        previous_time = self._last_time
        throttle, steer, brake = super().step(now, speed, yaw_rate)
        diagnostics = self.diagnostics
        if diagnostics['reason'] != 'tracking' or diagnostics.get('aim_xy') is None:
            return throttle, steer, brake
        x, y = diagnostics['aim_xy']
        pid = -self.extra_bearing_pid.step(math.atan2(y, x))
        combined = diagnostics['raw_steer'] + pid
        elapsed = self.dt if previous_time is None else now - previous_time
        steer = float(np.clip(combined, previous_steer - self.steer_rate * elapsed,
                              previous_steer + self.steer_rate * elapsed))
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self._last_steer = steer
        self._last_control = (throttle, steer, brake)
        self._diagnostics.update(experiment='original_handoff_full_bearing_pid_sum',
                                 pursuit_steer=diagnostics['raw_steer'], bearing_pid_steer=pid,
                                 raw_steer=combined)
        return self._last_control


def circle(preset, speed=6., radius=20., sign=1., delay=.0,
           lateral_offset=0., heading_offset=0., wheelbase_gain=1.,
           steering_gain=1., steering_delay=0., speed_window='near',
           duration=14., lookahead=None, bearing_pid=False):
    controller_class = _BearingPIDExperiment if bearing_pid else Controller
    ctrl = controller_class(preset, speed_window=speed_window, lookahead=lookahead)
    state = np.array([0., lateral_offset, heading_offset, speed])
    queued, controls = deque(), deque()
    lateral, speeds, command_errors, timings = [], [], [], []
    motion_speed, motion_yaw = speed, 0.
    reason_counts = {}
    delay_ticks = int(round(delay / DT))
    steering_ticks = int(round(steering_delay / DT))
    # History exists before delayed plans arrive; start stationary command during
    # that interval, preserving the real startup penalty in full-period metrics.
    for tick in range(int(duration / DT)):
        now = tick * DT
        if tick % 4 == 0:
            future = now + np.arange(1, 21) * .25
            theta = speed * future / radius
            world = np.column_stack((radius * np.sin(theta), sign * radius * (1. - np.cos(theta))))
            queued.append((tick + delay_ticks, _local(world, state), now))
        while queued and queued[0][0] <= tick:
            _, path, source_time = queued.popleft()
            ctrl.update(path, source_time)
        start = time.perf_counter_ns()
        command = ctrl.step(now, max(motion_speed, 0.), motion_yaw)
        timings.append((time.perf_counter_ns() - start) / 1e6)
        reason = ctrl.diagnostics['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        target = ctrl.diagnostics['target_speed_mps']
        command_errors.append(state[3] - target if target is not None else 0.)
        lateral.append(math.hypot(state[0], state[1] - sign * radius) - radius)
        speeds.append(state[3] - speed)
        controls.append(command[1])
        applied_steer = controls.popleft() if len(controls) > steering_ticks else 0.
        state, motion_speed, motion_yaw = _plant(state, (command[0], applied_steer, command[2]), wheelbase_gain, steering_gain)
    warmup = int(2. / DT)
    result = dict(kind='circle', preset=preset, speed_mps=speed, radius_m=radius,
                  sign=sign, delay_s=delay, lateral_offset_m=lateral_offset,
                  heading_offset_deg=math.degrees(heading_offset), wheelbase_gain=wheelbase_gain,
                  steering_gain=steering_gain, steering_delay_s=steering_delay,
                  speed_window=speed_window, lookahead=ctrl.lookahead,
                  experiment='original_full_bearing_pid_sum' if bearing_pid else 'production_controller',
                  lateral_rms_full_m=_rms(lateral), lateral_rms_m=_rms(lateral[warmup:]),
                  lateral_p95_m=float(np.percentile(np.abs(lateral[warmup:]), 95)),
                  reference_speed_rms_mps=_rms(speeds[warmup:]),
                  command_speed_rms_mps=_rms(command_errors[warmup:]),
                  controller_p99_ms=float(np.percentile(timings, 99)), reason_counts=reason_counts,
                  physics_lateral_acceleration_mps2=speed ** 2 / radius)
    result['main_pass'] = result['lateral_rms_m'] < .2 and result['reference_speed_rms_mps'] < .5
    return result


def _stop_reference(t):
    t = np.asarray(t)
    elapsed = np.clip(t - 3., 0., 3.)
    position = 8. * np.minimum(t, 3.) + 8. * elapsed - (4. / 3.) * elapsed ** 2
    speed = np.where(t < 3., 8., np.maximum(8. - 8. / 3. * (t - 3.), 0.))
    return position, speed


def stop(preset, speed_window='near'):
    ctrl = Controller(preset, speed_window=speed_window)
    state = np.array([0., 0., 0., 8.])
    motion_speed, motion_yaw = 8., 0.
    errors, command_errors, positions, speeds, decel_errors = [], [], [], [], []
    for tick in range(int(12. / DT) + 1):
        now = tick * DT
        if tick % 4 == 0:
            positions_future, _ = _stop_reference(now + np.arange(1, 21) * .25)
            ctrl.update(_local(np.column_stack((positions_future, np.zeros(20))), state), now)
        command = ctrl.step(now, max(motion_speed, 0.), motion_yaw)
        _, reference = _stop_reference(now)
        if 3. <= now <= 6.:
            decel_errors.append(float(state[3] - reference))
        if now >= 3.:
            errors.append(float(state[3] - reference))
            target = ctrl.diagnostics['target_speed_mps']
            if target is not None:
                command_errors.append(state[3] - target)
        positions.append(state[0])
        speeds.append(state[3])
        state, motion_speed, motion_yaw = _plant(state, command)
    planned_index = int(6. / DT)
    hold_index = next((i for i in range(planned_index, len(speeds)) if abs(speeds[i]) < 1e-6), len(speeds) - 1)
    hold_end = min(hold_index + int(5. / DT) + 1, len(speeds))
    result = dict(kind='stop', preset=preset, speed_window=speed_window,
                  reference_speed_rms_mps=_rms(errors), deceleration_speed_rms_mps=_rms(decel_errors),
                  command_speed_rms_mps=_rms(command_errors),
                  stop_position_error_m=float(positions[-1] - 36.),
                  hold_displacement_m=float(max(positions[hold_index:hold_end]) - min(positions[hold_index:hold_end])),
                  hold_max_speed_mps=float(max(abs(v) for v in speeds[hold_index:hold_end])),
                  hold_duration_s=(hold_end - hold_index - 1) * DT,
                  actual_stop_time_s=hold_index * DT,
                  speed_at_planned_stop_mps=float(speeds[planned_index]),
                  deceleration_start_s=3., planned_stop_time_s=6.)
    result['main_pass'] = (result['deceleration_speed_rms_mps'] < .5 and
                           abs(result['stop_position_error_m']) <= 1. and
                           result['hold_displacement_m'] <= .1 and result['hold_max_speed_mps'] < .1 and
                           result['hold_duration_s'] >= 5.)
    return result



def s_curve(preset='pursuit', delay=0.):
    """Sine centerline: independent Newton projection on y=2 sin(x/12)."""
    ctrl = Controller(preset)
    state = np.array([0., 0., math.atan(2. / 12.), 6. * math.sqrt(1. + (2. / 12.) ** 2)])
    motion_speed, motion_yaw = state[3], 0.
    queue = deque()
    errors, speed_errors = [], []
    for tick in range(400):
        now = tick * DT
        if tick % 4 == 0:
            x = 6. * (now + np.arange(1, 21) * .25)
            queue.append((tick + int(round(delay / DT)), _local(np.column_stack((x, 2. * np.sin(x / 12.))), state), now))
        while queue and queue[0][0] <= tick:
            _, path, stamp = queue.popleft()
            ctrl.update(path, stamp)
        command = ctrl.step(now, max(motion_speed, 0.), motion_yaw)
        projection_x = state[0]
        for _ in range(8):
            y = 2. * math.sin(projection_x / 12.)
            dy = 2. / 12. * math.cos(projection_x / 12.)
            ddy = -2. / 144. * math.sin(projection_x / 12.)
            projection_x -= ((projection_x - state[0]) + (y - state[1]) * dy) / (1. + dy ** 2 + (y - state[1]) * ddy)
        errors.append(math.hypot(projection_x - state[0], 2. * math.sin(projection_x / 12.) - state[1]))
        speed_errors.append(state[3] - 6. * math.sqrt(1. + (2. / 12. * math.cos(6. * now / 12.)) ** 2))
        state, motion_speed, motion_yaw = _plant(state, command)
    return dict(kind='s_curve', preset=preset, delay_s=delay,
                lateral_rms_full_m=_rms(errors), lateral_rms_m=_rms(errors[40:]),
                reference_speed_rms_mps=_rms(speed_errors[40:]),
                pass_accuracy=_rms(errors[40:]) <= .3 and _rms(speed_errors[40:]) < .5)


def run_suite():
    main = [circle(preset, sign=sign) for preset in ('carla', 'tcp', 'pursuit') for sign in (1., -1.)]
    stops = [stop(preset, window) for preset in ('carla', 'tcp', 'pursuit') for window in ('near', 'reference')]
    robust = []
    # Predeclared feasible grid, v^2/R <= 5 m/s^2; 12/14m/s at R20 are
    # saturation cases separately. These bounds are a synthetic test protocol.
    for speed in (2., 6., 8., 12., 14.):
        for radius in (20., 40.):
            if speed ** 2 / radius > 5.:
                continue
            zero = circle('pursuit', speed, radius)
            delayed = circle('pursuit', speed, radius, delay=.30)
            delayed['delay_rms_increment_m'] = delayed['lateral_rms_m'] - zero['lateral_rms_m']
            delayed['delay_pass'] = delayed['lateral_rms_m'] <= .3 and delayed['delay_rms_increment_m'] <= .1
            robust.extend((zero, delayed))
    for offset in (-.5, .5):
        for heading in (-5., 5.):
            robust.append(circle('pursuit', lateral_offset=offset, heading_offset=math.radians(heading)))
    for gain in (.9, 1.1):
        robust.append(circle('pursuit', wheelbase_gain=gain))
        robust.append(circle('pursuit', steering_gain=gain))
    robust.append(circle('pursuit', steering_delay=.15))
    bearing_pid_ablation = [circle('pursuit', sign=sign, lookahead='max', speed_window='reference', bearing_pid=enabled)
                            for sign in (1., -1.) for enabled in (False, True)]
    lookahead_comparison = [circle('pursuit', speed=speed, lookahead=rule)
                            for speed in (6., 8.) for rule in ('additive', 'max')]
    s_curves = [s_curve(delay=delay) for delay in (0., .3)]
    saturation = [circle('pursuit', speed=v, radius=20.) for v in (12., 14.)]
    # Prove static hold is physically modeled and reverse motion remains possible.
    hold_v, hold_dx = _longitudinal(0., 0., 1., 5., slope=-1.)
    reverse_v, _ = _longitudinal(0., 0., 0., .1, slope=-1.)
    candidate = [row for row in main + stops if row['preset'] == 'pursuit' and row['speed_window'] == 'near']
    return dict(schema_version=1, plant='synthetic_kinematic_bicycle_with_brake_static_friction',
                geometry_source='/data/runs/b2d/controller/calibration/controller_config.json',
                parameters=dict(dt_s=DT, throttle_acceleration_mps2=3., brake_deceleration_mps2=8.,
                                viscous_drag_per_s=.08, wheelbase_m=WHEELBASE,
                                max_steer_deg=math.degrees(MAX_ANGLE), steering_curve=CURVE.tolist()),
                limitations=['No tire slip or measured actuator model.',
                             'Motion inputs are exact interval averages; real sensor endpoint sampling requires separate validation.',
                             'GNSS noise not injected: measured noise/fusion belongs to agent tests.',
                             'Static friction unit exercise is not a measured CARLA slope test.'],
                main=main, speed_window_comparison=stops, robustness=robust, s_curves=s_curves, saturation=saturation,
                lookahead_comparison=lookahead_comparison,
                original_bearing_pid_ablation=dict(rows=bearing_pid_ablation,
                    formula='CARLA_steer = -atan(L*2*y/(x*x+y*y))/(max_angle*curve(v)) - CARLA_PID(atan2(y,x))',
                    pid=dict(kp=1.95, ki=.05, kd=.2, window=10, dt_s=.05),
                    held_identical='max(3,.5v) lookahead, reference speed window, measured geometry, pose/history, projection and shared output limits',
                    scope='Exact original lateral sum isolated; shared corrected time/projection implementation, not a claim of reproducing unspecified original waypoint indexing.'),
                synthetic_static_hold=dict(speed_mps=hold_v, displacement_m=hold_dx, unbraked_reverse_mps=reverse_v),
                candidate_main_pass=all(row['main_pass'] for row in candidate),
                candidate_delay_pass=all(row.get('delay_pass', True) for row in robust))
