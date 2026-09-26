"""X-only independent longitudinal target; frozen P7 geometry and actuation stay intact.

A None target is exactly the existing timed-trajectory contract. An explicit target
is constant over the accepted plan, including its acceleration feedforward. Safety
checks (invalid/missing geometry, stale plans and horizon) still belong to P7.
"""
import math

from b2d_controller import Controller


class XController(Controller):
    def reset(self):
        self._x_target = self._x_pending_target = None
        super().reset()

    def update(self, traj_xy, t_frame, trajectory_dt=None, *, target_speed_mps=None):
        if target_speed_mps is not None:
            target_speed_mps = float(target_speed_mps)
            if not math.isfinite(target_speed_mps) or target_speed_mps < 0:
                return self._reject('invalid_speed_target')
        accepted = super().update(traj_xy, t_frame, trajectory_dt)
        if accepted:
            self._x_pending_target = target_speed_mps
        return accepted

    def _accept_pending(self, now):
        stamp = self._pending[1] if self._pending is not None else None
        target = self._x_pending_target
        super()._accept_pending(now)
        if stamp is not None and self._source_time == stamp and self._points is not None:
            self._x_target = target

    def _speed_at(self, age, start, end):
        learned = super()._speed_at(age, start, end)
        return learned if learned is None or self._x_target is None else self._x_target

    def _accel_actuation(self, age, desired, speed, elapsed):
        if self._x_target is not None:
            # A constant target has zero feedforward, including the filter state.
            self._feedforward = None
        return super()._accel_actuation(age, desired, speed, elapsed)

    def step(self, t_now, speed_mps, yaw_rate_rps):
        terminal = self.terminal_approach
        # _accept_pending runs inside super.step; inspect the queued target too.
        target = self._x_pending_target if self._pending is not None else self._x_target
        if target is not None:
            self.terminal_approach = False
        try:
            control = super().step(t_now, speed_mps, yaw_rate_rps)
            self._diagnostics['x_target_speed_mps'] = self._x_target
            return control
        finally:
            self.terminal_approach = terminal
