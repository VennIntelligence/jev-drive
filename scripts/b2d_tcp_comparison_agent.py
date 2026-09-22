"""Actual TCP inference with native lateral control and a paired longitudinal test.

Both common-envelope arms replace the official low-speed throttle tail. Native
PID still advances once per frame in both arms, and PI advances once as a shadow.
Simulator truth is read only after selecting controls and is never fed to TCP.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from TCP.model import TCP
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from team_code import tcp_b2d_agent as vendor
from b2d_tcp_visual_agent import VisualTCPAgent
from b2d_tcp_control import TCPControlComparison


def get_entry_point():
    return 'ComparisonTCPAgent'


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def vector(v):
    return [float(v.x), float(v.y), float(v.z)]


def controls(c):
    return [float(c.throttle), float(c.steer), float(c.brake)]


class ComparisonTCPAgent(VisualTCPAgent):
    def setup(self, path_to_conf_file):
        self._arm = os.environ['B2D_TCP_CONTROL_ARM']
        if self._arm not in ('native_common', 'pi_common') or vendor.PLANNER_TYPE != 'only_traj':
            raise ValueError('Comparison requires native_common/pi_common and only_traj')
        self._out = Path(os.environ['B2D_ATTEMPT_OUT'])
        self._telemetry = (self._out / 'tcp-control.jsonl').open('x', buffering=1)
        self._comparison = TCPControlComparison()
        self._capture = None
        self._forward_count = 0
        self._npc_ids = []
        original_load = TCP.load_state_dict
        compatibility = {}

        def checked_load(net, state, strict=False):
            result = original_load(net, state, strict=True)
            compatibility.update(missing_keys=list(result.missing_keys),
                                 unexpected_keys=list(result.unexpected_keys), strict=True,
                                 parameters=sum(p.numel() for p in net.parameters()))
            return result

        TCP.load_state_dict = checked_load
        try:
            super().setup(path_to_conf_file)
        finally:
            TCP.load_state_dict = original_load
        self._native_pid = self.net.control_pid
        self.net.control_pid = self._capture_pid
        (self._out / 'tcp-setup.json').write_text(json.dumps(dict(
            arm=self._arm, planner_type=vendor.PLANNER_TYPE, checkpoint=self.config_path,
            compatibility=compatibility, prediction_shape=[4, 2], prediction_dt_s=.5,
            prediction_axes='forward/right; physical origin unconfirmed',
            controller_dt_s=.05, kp=.5, ki=.25,
            envelope='throttle[0,.75], brake[0,1], mutually exclusive; native steer',
            environment={k: os.environ.get(k) for k in (
                'B2D_TCP_OPTIMIZE', 'B2D_TCP_PIPELINE', 'B2D_TCP_FAST_COLOR',
                'B2D_TCP_DEBUG_VIEWS', 'B2D_TCP_EARLY_RGB', 'B2D_ASYNC_DISPLAY',
                'CUDA_VISIBLE_DEVICES', 'PLANNER_TYPE', 'PYTHONPATH')}), indent=2))

    def _before_forward(self, model, inputs):
        self._forward_count += 1
        super()._before_forward(model, inputs)

    def _capture_pid(self, waypoints, velocity, target):
        if self._capture is not None:
            raise RuntimeError('More than one native control_pid call in a tick')
        # The native method swaps axes in a CPU numpy view. Save before it runs.
        raw = waypoints.detach().cpu().numpy().copy()
        raw_target = target.detach().cpu().numpy().copy()
        if raw.shape != (1, 4, 2):
            raise ValueError('Native TCP horizon/shape changed')
        finite = bool(np.isfinite(raw).all() and np.isfinite(raw_target).all()
                      and np.isfinite(velocity.detach().cpu().numpy()).all())
        if not finite:
            # Common fault policy: do not poison either native PID history.
            for pid in (self.net.turn_controller, self.net.speed_controller):
                pid._window.clear()
                pid._window.extend([0.] * pid._window.maxlen)
                pid._min = pid._max = 0.
            metadata = dict(desired_speed=float('nan'), steer=0., throttle=0., brake=1.)
            self._capture = dict(raw_waypoints=raw[0], raw_target=raw_target.reshape(2),
                                 metadata=metadata.copy(), desired_speed_recomputed_mps=float('nan'),
                                 native_control=[0., 0., 1.], finite_prediction=False,
                                 native_pid_calls=0, fault='nonfinite_prediction_or_motion')
            return 0., 0., 1., metadata
        result = self._native_pid(waypoints, velocity, target)
        steer, throttle, brake, metadata = result
        desired = sum(float(np.linalg.norm(raw[0, i+1] - raw[0, i])) * 2 / 3
                      for i in range(3)) if raw.shape == (1, 4, 2) else float('nan')
        if raw.shape != (1, 4, 2):
            raise ValueError('Native TCP horizon/shape changed')
        self._capture = dict(raw_waypoints=raw[0], raw_target=raw_target.reshape(2),
                             metadata=copy.deepcopy(metadata),
                             desired_speed_recomputed_mps=desired,
                             native_control=[float(throttle), float(steer), float(brake)],
                             finite_prediction=True, native_pid_calls=1)
        return result

    def tick(self, input_data):
        data = super().tick(input_data)
        self._tcp_input = dict(target_point=list(data['target_point']),
                               command=int(data['next_command']), compass_used=float(data['compass']),
                               gps_projected=data['gps'].tolist(),
                               rgb_sha256=hashlib.sha256(data['rgb'].tobytes()).hexdigest())
        return data

    def run_step(self, input_data, timestamp):
        self._capture = None
        self._forward_count = 0
        return super().run_step(input_data, timestamp)

    def _postprocess_control(self, control, input_data, timestamp):
        official = controls(control)
        result = None
        if self._capture is not None:
            if self._forward_count != 1:
                raise RuntimeError('Expected exactly one network forward per control step')
            c = self._capture
            desired = c['metadata']['desired_speed']
            if c['finite_prediction'] and not np.isclose(
                    desired, c['desired_speed_recomputed_mps'], rtol=1e-6, atol=1e-6):
                raise ValueError('Native desired speed does not match its three waypoint segments')
            if not c['finite_prediction']:
                desired = float('nan')
            result = self._comparison.step(timestamp, float(input_data['SPEED'][1]['speed']),
                                           desired, *c['native_control'])
            selected = result[self._arm + '_control']
            control.throttle, control.steer, control.brake = selected
        elif self._forward_count or self.step >= self.config.seq_len:
            raise RuntimeError('Missing native PID output after initialization')
        frame = GameTime.get_frame()
        row = dict(frame=frame, timestamp=float(timestamp), step=self.step, arm=self._arm,
                   forward_count=self._forward_count,
                   sensor_frames={k: int(v[0]) for k, v in input_data.items()},
                   raw_speed_mps=float(input_data['SPEED'][1]['speed']),
                   raw_imu=np.asarray(input_data['IMU'][1]), raw_gps=np.asarray(input_data['GPS'][1]),
                   model_input=self._tcp_input, prediction=self._capture,
                   comparison=result, official_tail_control=official, selected_control=controls(control),
                   truth=self._truth(frame))
        self._telemetry.write(json.dumps(serial(row), allow_nan=False) + '\n')
        return control

    def _truth(self, frame):
        try:
            hero = CarlaDataProvider.get_hero_actor()
            world = hero.get_world()
            snap = world.get_snapshot()
            if snap.frame != frame:
                return dict(error='frame_mismatch', frame=snap.frame)
            actor = snap.find(hero.id)
            tf = actor.get_transform()
            result = dict(frame=snap.frame, actor_id=hero.id, actor_type=hero.type_id,
                          xyz=vector(tf.location), rotation_deg=[tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
                          velocity=vector(actor.get_velocity()), acceleration=vector(actor.get_acceleration()),
                          angular_velocity_deg=vector(actor.get_angular_velocity()))
            applied = hero.get_control()
            result.update(applied_control=controls(applied), gear=applied.gear,
                          reverse=applied.reverse, hand_brake=applied.hand_brake)
            if self.step % 20 == 0:
                self._npc_ids = [(a.id, a.type_id) for a in world.get_actors()
                                 if a.id != hero.id and a.type_id.startswith(('vehicle.', 'walker.'))]
            if self.step % 4 == 0:
                nearby = []
                for actor_id, actor_type in self._npc_ids:
                    other = snap.find(actor_id)
                    if other is None:
                        continue
                    transform = other.get_transform()
                    if transform.location.distance(tf.location) < 50:
                        nearby.append(dict(id=actor_id, type=actor_type, xyz=vector(transform.location),
                                           yaw=transform.rotation.yaw, velocity=vector(other.get_velocity())))
                result['nearby_actors'] = nearby
            if not (self._out / 'tcp-route-reference.json').exists():
                path = [[w[0].location.x, w[0].location.y, w[0].location.z, int(w[1].value)]
                        for w in self._global_plan_world_coord]
                (self._out / 'tcp-route-reference.json').write_text(json.dumps(dict(
                    world_route=path, interpretation='navigation reference only, never fed into replacement control',
                    rear_axle_offset_m=-1.3886332201957703)))
            return result
        except Exception as exc:
            return dict(error=type(exc).__name__ + ': ' + str(exc))

    def destroy(self):
        if hasattr(self, '_telemetry'):
            self._telemetry.close()
        super().destroy()
