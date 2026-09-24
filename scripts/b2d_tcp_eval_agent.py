"""Task 10 TCP: shared 4×0.5 s model plan, native plus A/B/C/D execution.

Network code/checkpoint and its once-per-frame native PID are untouched. All
candidate shadows advance on each prediction, including when not selected.
"""
import json
import os

import numpy as np

from b2d_controller import Controller
from b2d_tcp_comparison_agent import ComparisonTCPAgent, controls, serial
from b2d_tfv6_author_control import AuthorController
from srunner.scenariomanager.timer import GameTime


def get_entry_point():
    return 'Task10TCPAgent'


def envelope(triple, speed):
    throttle, steer, brake = map(float, triple)
    if not np.isfinite([throttle, steer, brake]).all():
        raise ValueError('Nonfinite candidate TCP control')
    brake = float(np.clip(brake, 0., 1.))
    throttle = 0. if brake > 0 else float(np.clip(throttle, 0., 1.))
    steer = 0. if brake > 0 and speed < .01 else float(np.clip(steer, -1., 1.))
    return [throttle, steer, brake]


class Task10TCPAgent(ComparisonTCPAgent):
    def setup(self, path_to_conf_file):
        requested = os.environ['B2D_TCP_CONTROL_ARM']
        if requested not in 'NABCD' or len(requested) != 1:
            raise ValueError('Task 10 TCP arm must be one of N/A/B/C/D')
        os.environ['B2D_TCP_CONTROL_ARM'] = 'native_common'
        try:
            super().setup(path_to_conf_file)
        finally:
            os.environ['B2D_TCP_CONTROL_ARM'] = requested
        self._arm = requested
        common = dict(longitudinal_mode='pi', lookahead='max', pi_kp=.5,
                      pi_ki=.25, max_lookahead_time_s=.5)
        self._eval = {
            'A': AuthorController('route'),
            'B': AuthorController('waypoint'),
            'C': Controller(preset='pursuit', **common),
            'D': Controller(preset='pursuit', **common, pursuit_frame='rear_slip',
                            rear_slip_c_per_rad=11., steer_inverse='ackermann',
                            track_width_m=1.5929),
        }
        self._eval_log = (self._out / 'controller-eval.jsonl').open('x', buffering=1)
        (self._out / 'tcp-eval-setup.json').write_text(json.dumps({
            'arm': requested, 'plan_axes': 'TCP forward/right -> controller forward/left',
            'plan_shape': [4, 2], 'plan_dt_s': .5,
            'plan_origin': 'GNSS sensor x=-1.4 m, 0.0114 m from rear axle',
            'native_mode': 'official only_traj including official tail',
            'A_B_adapter': 'same author formulas; 4x0.5s linearly interpolated to 8x0.25s',
            'C_D_config': common, 'chosen_arm': requested}, indent=2))

    def _postprocess_control(self, control, input_data, timestamp):
        official = controls(control)
        result = None
        selected = official
        shadows = {}
        if self._capture is not None:
            if self._forward_count != 1 or self._capture['native_pid_calls'] != 1:
                raise RuntimeError('TCP prediction/native PID count mismatch')
            c = self._capture
            if not c['finite_prediction']:
                raise RuntimeError('Nonfinite TCP prediction is an invariant failure')
            desired = float(c['metadata']['desired_speed'])
            if not np.isclose(desired, c['desired_speed_recomputed_mps'], rtol=1e-6, atol=1e-6):
                raise ValueError('Native TCP desired speed differs from waypoint segments')
            speed = float(input_data['SPEED'][1]['speed'])
            yaw_rate = -float(input_data['IMU'][1][5])
            path = np.asarray(c['raw_waypoints'], float) * np.array([1., -1.])
            for arm, candidate in self._eval.items():
                if not candidate.update(path, float(timestamp), trajectory_dt=.5):
                    raise RuntimeError('TCP candidate rejected finite prediction')
                raw = candidate.step(float(timestamp), speed, yaw_rate)
                shadows[arm] = envelope(raw, speed)
            result = self._comparison.step(timestamp, speed, desired, *c['native_control'])
            selected = official if self._arm == 'N' else shadows[self._arm]
            control.throttle, control.steer, control.brake = selected
        elif self._forward_count or self.step >= self.config.seq_len:
            raise RuntimeError('Missing native PID output after initialization')
        observed = controls(control)
        if max(abs(a-b) for a,b in zip(observed, selected)) > 1e-6:
            raise RuntimeError('TCP selected control differs from executed control')
        frame = GameTime.get_frame()
        row = dict(frame=frame, timestamp=float(timestamp), step=self.step, arm=self._arm,
                   forward_count=self._forward_count,
                   sensor_frames={k: int(v[0]) for k, v in input_data.items()},
                   raw_speed_mps=float(input_data['SPEED'][1]['speed']),
                   model_input=self._tcp_input, prediction=self._capture,
                   comparison=result, official_tail_control=official,
                   shadow_control=shadows, selected_control=observed,
                   truth=self._truth(frame))
        self._eval_log.write(json.dumps(serial(row), allow_nan=False) + '\n')
        # Keep the frozen comparison-agent telemetry contract for independent
        # frame/inference/invariant checks, with the actually selected control.
        self._telemetry.write(json.dumps(serial(row), allow_nan=False) + '\n')
        return control

    def destroy(self):
        if hasattr(self, '_eval_log'):
            self._eval_log.close()
        super().destroy()
