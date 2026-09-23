"""Pure numeric native-TCP longitudinal comparison; no planner or CARLA imports.

Both arms consume the same native desired speed and already-clipped native
steering, before the official low-speed throttle cap / binary-brake tail.
The caller selects a common-envelope control and resets this object per route.
This does not replace TCP's four-point planner, target arbitration or lateral PID.
"""
import copy
import math

from b2d_controller import ConditionalPI


class TCPControlComparison:
    """Frozen PI .5/.25 versus native pedals, with matched actuator envelopes.

    Call once per simulation tick. First valid tick uses .05 s; subsequent ticks
    use actual elapsed simulation time. Equal timestamps (within 1e-8 s) are
    idempotent. Regression or gaps above .2 s brake/reset, anchoring the next tick
    at the rejected finite timestamp. Invalid motion/controls also brake/reset;
    a nonfinite timestamp clears the clock. Numerical resting speeds with
    abs(speed)<.01 m/s are zeroed; raw speed remains in diagnostics. Larger
    reverse speed is still a fault, matching the existing route-agent boundary.
    Valid native steering is passed through exactly, including during safe brake.
    """
    nominal_dt = .05
    max_gap = .2

    def __init__(self):
        self.pi = ConditionalPI(lower=-1., upper=.75, kp=.5, ki=.25)
        self.reset()

    def reset(self):
        self.pi.reset()
        self._last_time = None
        self._last_result = None

    @staticmethod
    def _envelope(throttle, steer, brake):
        brake = min(max(brake, 0.), 1.)
        throttle = 0. if brake > 0. else min(max(throttle, 0.), .75)
        return [float(throttle), float(steer), float(brake)]

    @staticmethod
    def _number(value):
        try:
            return float(value)
        except (ValueError, TypeError, OverflowError):
            return float('nan')

    @staticmethod
    def _logged(value):
        # Preserve nonfinite input identity without non-standard JSON NaN tokens.
        return value if math.isfinite(value) else str(value)

    def _result(self, native, common, candidate, reason, elapsed, desired, speed):
        return dict(native_control=[self._logged(v) for v in native],
                    native_common_control=common, pi_common_control=candidate,
                    diagnostics=dict(reason=reason, elapsed_s=elapsed,
                                     desired_speed_mps=self._logged(desired),
                                     speed_mps=self._logged(speed),
                                     longitudinal_mode='pi', pi_kp=.5, pi_ki=.25,
                                     integral_effort=self.pi.integral,
                                     unsaturated_effort=self.pi.raw_effort,
                                     integration_limited=self.pi.integration_limited,
                                     pi_effort=candidate[0] - candidate[2]))

    def step(self, timestamp, speed_mps, desired_speed_mps,
             native_throttle, native_steer, native_brake):
        now, speed, desired, throttle, steer, brake = map(self._number, (
            timestamp, speed_mps, desired_speed_mps,
            native_throttle, native_steer, native_brake))
        raw_speed = speed
        if math.isfinite(speed) and abs(speed) < .01:
            speed = 0.
        native = [throttle, steer, brake]
        elapsed = self.nominal_dt if self._last_time is None else now - self._last_time
        reason = None
        if not all(math.isfinite(v) for v in (now, speed, desired, throttle, steer, brake)):
            reason = 'nonfinite_input'
        elif speed < 0.:
            reason = 'reverse_motion'
        elif desired < 0. or abs(steer) > 1.:
            reason = 'invalid_native_input'
        elif elapsed < -1e-8:
            reason = 'time_regression'
        elif self._last_time is not None and elapsed <= 1e-8:
            result = copy.deepcopy(self._last_result)
            result['diagnostics']['reason'] = 'duplicate_tick'
            return result
        elif elapsed > self.max_gap + 1e-8:
            reason = 'motion_gap'

        if reason is not None:
            self.reset()
            self._last_time = now if math.isfinite(now) else None
            safe_steer = steer if math.isfinite(steer) and abs(steer) <= 1. else 0.
            safe = [0., safe_steer, 1.]
            result = self._result(native, safe.copy(), safe.copy(), reason,
                                  elapsed if math.isfinite(elapsed) else None, desired, speed)
        else:
            effort = self.pi.step(desired - speed, elapsed)
            candidate = self._envelope(max(effort, 0.), steer, max(-effort, 0.))
            reason = 'tracking'
            # Same zero-speed hold threshold as the frozen controller; no
            # endpoint or stationary-tail assumptions about native predictions.
            if desired < .05 and speed < .1:
                self.pi.reset()
                candidate = [0., steer, 1.]
                reason = 'stop_hold'
            common = self._envelope(throttle, steer, brake)
            self._last_time = now
            result = self._result(native, common, candidate, reason, elapsed, desired, speed)
        result['diagnostics'].update(raw_speed_mps=self._logged(raw_speed),
                                     standstill_deadband_applied=math.isfinite(raw_speed) and raw_speed != speed)
        self._last_result = copy.deepcopy(result)
        return result
