"""Four-arm TFv6 controller experiment; model and author postprocessors stay upstream."""

from collections import deque
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np

from lead.inference.sensor_agent import SensorAgent
from b2d_controller import Controller
from b2d_controller_adapter import GPSProjector, PoseFilter, controller_speed
from b2d_tfv6_coordinates import rear_waypoints


ARMS = "ABCD"
DIAGNOSTIC_ARMS = "EFK"


def get_entry_point():
    return "TFv6ControllerAgent"


def _triplet(steer, throttle, brake):
    return {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)}


def _logged_number(value):
    number = float(value)
    return number if np.isfinite(number) else str(number)


def _normalize_brake(raw, speed):
    result = dict(raw)
    if result["brake"] > 0:
        result["throttle"] = 0.0
        if speed < 0.01:
            result["steer"] = 0.0
    return result


def _select_arm_control(arm, candidates):
    if arm not in ARMS or set(candidates) != set(ARMS):
        raise ValueError("Expected exactly four registered TFv6 controls")
    return dict(candidates[arm])


def _compose_d3_arm(arm, raw, speed):
    """Build a diagnostic arm after B/C raw controls, before author postprocessors."""
    if arm == 'E':
        mixed = _triplet(raw['C']['steer'], raw['B']['throttle'], raw['B']['brake'])
    elif arm == 'F':
        mixed = _triplet(raw['B']['steer'], raw['C']['throttle'], raw['C']['brake'])
    elif arm == 'K':
        mixed = dict(raw['C'])
    else:
        raise ValueError(f'Unknown D3 diagnostic arm {arm}')
    return _normalize_brake(mixed, speed)


def _kalman_input_only(original_control, shadow_b):
    """Construct the filter-only B input; never mutate the actual C control."""
    if shadow_b is None:
        return original_control
    return type(original_control)(steer=float(shadow_b['steer']),
                                  throttle=float(shadow_b['throttle']),
                                  brake=float(shadow_b['brake']))


def _configure_author_arm(config, arm):
    if arm == "A":
        return
    if arm == "B":
        config.steer_modality = "waypoint"
        config.throttle_modality = "waypoint"
        config.brake_modality = "waypoint"
        return
    if arm not in "CD":
        raise ValueError("Unknown TFv6 arm")


def _clone_processor(processor):
    result = copy(processor)
    # The live instance has an adjust wrapper installed for tracing. A shallow
    # copy retains that closure, which still calls the live processor and thus
    # advances its state a second time during shadow replay.
    result.__dict__.pop("adjust", None)
    if hasattr(processor, "stop_sign_buffer"):
        result.stop_sign_buffer = deque(processor.stop_sign_buffer, maxlen=processor.stop_sign_buffer.maxlen)
    return result


def _apply_postprocessors(raw, speed, force, stop):
    throttle, brake = force.adjust(float(speed), raw["throttle"], raw["brake"])
    throttle, brake = stop.adjust(float(speed), throttle, brake)
    return _triplet(raw["steer"], throttle, brake)


def _processor_snapshot(processor):
    return deepcopy({key: value for key, value in vars(processor).items() if key != "adjust"})


def _state_equal(left, right, seen=None):
    if type(left) is not type(right):
        return False
    if seen is None:
        seen = set()
    pair = (id(left), id(right))
    if pair in seen:
        return True
    seen.add(pair)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _state_equal(left[key], right[key], seen) for key in left)
    if isinstance(left, (list, tuple, deque)):
        return len(left) == len(right) and all(
            _state_equal(a, b, seen) for a, b in zip(left, right))
    if hasattr(left, "__dict__"):
        return _state_equal(vars(left), vars(right), seen)
    if hasattr(left, "__slots__"):
        return all(_state_equal(getattr(left, key), getattr(right, key), seen)
                   for key in left.__slots__)
    result = left == right
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def _changed_processor_keys(snapshot, processor):
    current = {key: value for key, value in vars(processor).items() if key != "adjust"}
    return sorted(set(snapshot) ^ set(current)) + [
        key for key in snapshot.keys() & current.keys()
        if not _state_equal(snapshot[key], current[key])]


def _assert_no_phantom(actor_waypoints, rear, frame_log, step, sim_time, arm):
    p0 = np.asarray(actor_waypoints[0], dtype=float) * np.array([1.0, -1.0])
    r0 = np.asarray(rear[0], dtype=float)
    if np.linalg.norm(p0) < 0.3 and np.linalg.norm(r0 - p0) > 0.3:
        frame_log.write(json.dumps({"diagnostic": "I1_phantom", "step": int(step),
            "sim_time": float(sim_time), "arm": arm, "p0": p0.tolist(),
            "r0": r0.tolist(), "rear_minus_actor_m": float(np.linalg.norm(r0 - p0))}) + "\n")
        raise AssertionError("I1 phantom rear waypoint")


class TFv6ControllerAgent(SensorAgent):
    def setup(self, path_to_conf_file, *args, **kwargs):
        self._d3 = os.environ.get('B2D_D3_DIAGNOSTIC') == '1'
        parts = path_to_conf_file.split("+")
        permitted = ARMS + (DIAGNOSTIC_ARMS if self._d3 else '')
        self.arm = (parts[1] if len(parts) > 1 and parts[1] in permitted
                    else os.environ.get("B2D_W2_ARM", "")).upper()
        if self.arm not in permitted:
            raise ValueError(f"TFv6 agent requires an arm in {permitted} via model_dir+ARM or B2D_W2_ARM")
        model_path = parts[0]
        output = os.environ.get("B2D_W2_LOG_DIR")
        if not output:
            raise RuntimeError("B2D_W2_LOG_DIR is required")
        Path(output).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SAVE_PATH", output)
        super().setup(model_path, *args, **kwargs)
        if self._d3:
            dense = [{'xyz': [float(transform.location.x),float(transform.location.y),
                              float(transform.location.z)], 'command': int(command)}
                     for transform,command in self._global_plan_world_coord]
            if len(dense)<2:
                raise RuntimeError('D3 dense evaluator global plan is empty')
            (Path(output)/'d3_global_plan.json').write_text(json.dumps(dense)+'\n')
            self._d3_dense_xy=np.asarray([item['xyz'][:2] for item in dense])
        _configure_author_arm(self.config_closed_loop, self.arm if self.arm in ARMS else 'C')
        # The W1 production pursuit configuration is fixed for both controller arms.
        common = dict(longitudinal_mode="pi", lookahead="max", pi_kp=0.5, pi_ki=0.25,
                      max_lookahead_time_s=0.5)
        self._controllers = {
            "C": Controller(preset="pursuit", **common),
            "D": Controller(preset="pursuit", **common, pursuit_frame="rear_slip",
                            rear_slip_c_per_rad=11.0, steer_inverse="ackermann",
                            track_width_m=1.5929),
        }
        gps_plan = np.asarray([[point["lat"], point["lon"]] for point, _ in self._global_plan])
        world_plan = np.asarray([[transform.location.x, transform.location.y]
                                 for transform, _ in self._global_plan_world_coord])
        self._pose_filter = PoseFilter(GPSProjector(gps_plan, world_plan),
                                       rear_axle_offset_m=-1.389, gnss_x_m=0.0,
                                       gnss_gain=0.05, heading_gain=0.1,
                                       lateral_coefficient_s2_per_m=0.010659832)
        self._sensor_pose = None
        self._raw = self._rear = self._prediction = None
        self._controller_reason = {}
        self._pre_force = self._pre_stop = None
        self._tick_data = None
        self._forward_ms = None
        self._frame_log = open(Path(output) / "frames.jsonl", "w", buffering=1)
        self._d2_actor_log = os.environ.get("B2D_D2_ACTORS") == "1"
        self._d2_actor_types = []
        self._d3_planners_installed = False
        self._d3_nav = None
        self._d3_pop_events = []
        self._d3_selected_pop_distance = None
        self._d3_prev_b = None
        self._d3_kalman_trace = None
        if self._d3 and self.arm == 'K':
            original_kalman = self.kalman_filter.step

            def diagnostic_kalman(*arguments, **keywords):
                actual = keywords['control']
                replacement = _kalman_input_only(actual, self._d3_prev_b)
                self._d3_kalman_trace = {'actual_input': _triplet(actual.steer, actual.throttle, actual.brake),
                                         'filter_input': _triplet(replacement.steer, replacement.throttle, replacement.brake),
                                         'used_b_shadow': self._d3_prev_b is not None}
                keywords['control'] = replacement
                return original_kalman(*arguments, **keywords)

            self.kalman_filter.step = diagnostic_kalman

        original_forward = self.closed_loop_inference.forward
        original_waypoints = self.closed_loop_inference.execute_waypoints
        self._author_waypoint_control = None

        def capture_waypoints(*arguments, **keywords):
            self._author_waypoint_control = original_waypoints(*arguments, **keywords)
            return self._author_waypoint_control

        self.closed_loop_inference.execute_waypoints = capture_waypoints

        def forward(*forward_args, **forward_kwargs):
            start = time.perf_counter()
            self._author_waypoint_control = None
            prediction = original_forward(*forward_args, **forward_kwargs)
            self._forward_ms = (time.perf_counter() - start) * 1000.0
            self._prediction = prediction
            try:
                actor_waypoints = prediction.pred_future_waypoints[0].detach().float().cpu().numpy()
                self._rear = rear_waypoints(actor_waypoints)
            except ValueError:
                self._rear = None
            if self._rear is not None:
                _assert_no_phantom(actor_waypoints, self._rear, self._frame_log,
                                   self.step, self._sim_time, self.arm)
            self._raw = {
                "A": _triplet(prediction.route_steer, prediction.target_speed_throttle,
                              prediction.target_speed_brake),
                "B": _triplet(*self._author_waypoint_control),
            }
            speed = self._controller_speed
            yaw_rate = -float(self._motion["imu"][1][5])
            for arm, controller in self._controllers.items():
                controller.update(self._rear if self._rear is not None else np.full((8, 2), np.nan),
                                  self._sim_time, trajectory_dt=0.25)
                throttle, steer, brake = controller.step(self._sim_time, speed, yaw_rate)
                self._raw[arm] = _triplet(steer, throttle, brake)
                self._controller_reason[arm] = controller.diagnostics["reason"]
            author_speed = float(self._tick_data["speed"].item())
            self._raw = {arm: _normalize_brake(raw, author_speed) for arm, raw in self._raw.items()}
            if self._d3 and self.arm in DIAGNOSTIC_ARMS:
                self._raw[self.arm] = _compose_d3_arm(self.arm, self._raw, author_speed)
            if self.arm in "CD":
                chosen = _select_arm_control(self.arm, self._raw)
                prediction.steer = chosen["steer"]
                prediction.throttle = chosen["throttle"]
                prediction.brake = chosen["brake"]
            elif self.arm in DIAGNOSTIC_ARMS:
                chosen = self._raw[self.arm]
                prediction.steer = chosen['steer']
                prediction.throttle = chosen['throttle']
                prediction.brake = chosen['brake']
            return prediction

        self.closed_loop_inference.forward = forward
        force_adjust = self.force_move_post_processor.adjust
        stop_adjust = self.stop_sign_post_processor.adjust

        def record_force(*arguments):
            self._pre_force = _clone_processor(self.force_move_post_processor)
            return force_adjust(*arguments)

        def record_stop(*arguments):
            self._pre_stop = _clone_processor(self.stop_sign_post_processor)
            return stop_adjust(*arguments)

        self.force_move_post_processor.adjust = record_force
        self.stop_sign_post_processor.adjust = record_stop

    def tick(self, input_data):
        if self._d3:
            self._install_d3_planner_trace()
        result = super().tick(input_data)
        self._tick_data = result
        if self._d3:
            planners = {}
            for distance, planner in self.gps_waypoint_planners_dict.items():
                route = list(planner.route)
                active = distance in (self.config_closed_loop.route_planner_min_distance, 4.0)
                selected = route if active else route[:3]
                target=route[1][0] if len(route)>1 else route[0][0]
                delta=self._d3_dense_xy-np.asarray(target[:2])
                global_index=int(np.argmin(np.sum(delta*delta,axis=1)))
                planners[str(distance)] = {'remaining': len(route),
                    'points': [{'xyz': np.asarray(point).tolist(), 'command': int(command)}
                               for point, command in selected], 'full_route': active,
                    'current_target_global_index':global_index,
                    'current_target_global_distance_m':float(np.linalg.norm(delta[global_index]))}
            chosen=planners.get(str(self._d3_selected_pop_distance))
            self._d3_nav = {
                'target_point_previous': np.asarray(result.get('target_point_previous')).tolist(),
                'target_point': np.asarray(result.get('target_point')).tolist(),
                'target_point_next': np.asarray(result.get('target_point_next')).tolist(),
                'command': np.asarray(result.get('command')).tolist(),
                'next_command': np.asarray(result.get('next_command')).tolist(),
                'filtered_state': np.asarray(result.get('filtered_state')).tolist(),
                'noisy_state': np.asarray(result.get('noisy_state')).tolist(),
                'selected_pop_distance': self._d3_selected_pop_distance,
                'selected_target_global_index':(chosen or {}).get('current_target_global_index'),
                'planners': planners, 'pop_events': self._d3_pop_events}
        return result

    def _install_d3_planner_trace(self):
        if self._d3_planners_installed or not getattr(self, 'gps_waypoint_planners_dict', None):
            return
        for distance, planner in self.gps_waypoint_planners_dict.items():
            original = planner.run_step
            def traced(gps, _original=original, _planner=planner, _distance=distance):
                previous = len(_planner.previous_target_points)
                value = _original(gps)
                popped = len(_planner.previous_target_points)-previous
                if popped > 0:
                    self._d3_pop_events.append({'tp_distance': _distance,
                        'count': popped,
                        'points': [{'xyz': np.asarray(p).tolist(), 'command': int(c)}
                                   for p,c in zip(_planner.previous_target_points[previous:],
                                                  _planner.previous_commands[previous:])]})
                return value
            planner.run_step = traced
        original_target = self.set_target_points
        def traced_target(input_data, pop_distance):
            value=original_target(input_data,pop_distance)
            self._d3_selected_pop_distance=pop_distance
            return value
        self.set_target_points=traced_target
        self._d3_planners_installed=True

    def _nearby_actors(self, world, snapshot, ego):
        """Read-only D2 scene context; no actor state is changed."""
        if self.step <= 1 or self.step % 10 == 0:
            self._d2_actor_types = [(actor.id, actor.type_id) for actor in world.get_actors()
                                    if actor.id != self._vehicle.id and
                                    actor.type_id.startswith(("vehicle.", "walker.", "static.", "traffic."))]
        transform = ego.get_transform()
        velocity = ego.get_velocity()
        import math
        yaw = math.radians(transform.rotation.yaw)
        co, si = math.cos(yaw), math.sin(yaw)
        result = []
        for actor_id, actor_type in self._d2_actor_types:
            other = snapshot.find(actor_id)
            if other is None:
                continue
            other_transform = other.get_transform()
            other_velocity = other.get_velocity()
            dx = other_transform.location.x - transform.location.x
            dy = other_transform.location.y - transform.location.y
            distance = math.hypot(dx, dy)
            if distance > 60:
                continue
            dvx = other_velocity.x - velocity.x
            dvy = other_velocity.y - velocity.y
            result.append({"id": int(actor_id), "type": actor_type,
                           "relative_xy_m": [co * dx + si * dy, -si * dx + co * dy],
                           "relative_velocity_xy_mps": [co * dvx + si * dvy,
                                                         -si * dvx + co * dvy],
                           "distance_m": distance})
        return sorted(result, key=lambda item: item["distance_m"])[:8]

    def run_step(self, input_data, timestamp, *args, **kwargs):
        if self._d3:
            self._d3_pop_events=[]
            self._d3_nav=None
            self._d3_selected_pop_distance=None
            self._d3_kalman_trace=None
        self._motion = input_data
        self._sim_time = float(timestamp)
        self._raw_signed_speed = float(input_data["speed"][1]["speed"])
        self._controller_speed = controller_speed(self._raw_signed_speed)
        try:
            imu = input_data["imu"][1]
            xy, yaw = self._pose_filter.update(input_data["gps"][1], float(imu[6]),
                                                self._controller_speed, float(imu[5]), self._sim_time)
            self._sensor_pose = {"xy": xy.tolist(), "yaw": yaw,
                                 "status": self._pose_filter.diagnostics}
        except ValueError as error:
            self._sensor_pose = {"xy": None, "yaw": None,
                                 "error": str(error), "status": self._pose_filter.diagnostics}
            self._pose_filter.reset()
        self._prediction = self._raw = self._rear = None
        self._controller_reason = {}
        self._pre_force = self._pre_stop = None
        started = time.perf_counter()
        control = super().run_step(input_data, timestamp, *args, **kwargs)
        truth = None
        nearby_actors = None
        traffic_light = None
        if getattr(self, "_vehicle", None) is not None:
            world = self._vehicle.get_world()
            snapshot = world.get_snapshot()
            actor = snapshot.find(self._vehicle.id)
            if actor is not None:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                angular = actor.get_angular_velocity()
                truth = {
                    "frame": int(snapshot.frame),
                    "location": [transform.location.x, transform.location.y, transform.location.z],
                    "rotation": [transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw],
                    "velocity": [velocity.x, velocity.y, velocity.z],
                    "speed_mps": float(np.linalg.norm([velocity.x, velocity.y, velocity.z])),
                    "forward_speed_mps": float(velocity.x * np.cos(np.deg2rad(transform.rotation.yaw)) +
                                               velocity.y * np.sin(np.deg2rad(transform.rotation.yaw))),
                    "angular_velocity": [angular.x, angular.y, angular.z],
                }
                if self._d2_actor_log:
                    try:
                        nearby_actors = self._nearby_actors(world, snapshot, actor)
                    except Exception as error:
                        nearby_actors = {"telemetry_error": str(error)}
                if self._d3:
                    try:
                        light = self._vehicle.get_traffic_light()
                        traffic_light = {'at_light': bool(self._vehicle.is_at_traffic_light()),
                                         'state': str(self._vehicle.get_traffic_light_state()),
                                         'light_id': int(light.id) if light is not None else None}
                    except Exception as error:
                        traffic_light = {'telemetry_error': str(error)}
        final = None
        guard_start = time.perf_counter()
        force_before = _processor_snapshot(self.force_move_post_processor)
        stop_before = _processor_snapshot(self.stop_sign_post_processor)
        guard_ms = (time.perf_counter() - guard_start) * 1000.0
        if self._raw is not None and self._pre_force is not None and self._pre_stop is not None:
            speed = float(self._tick_data["speed"].item())
            final = {arm: _apply_postprocessors(raw, speed,
                     _clone_processor(self._pre_force), _clone_processor(self._pre_stop))
                     for arm, raw in self._raw.items()}
            if self.step < self.training_config.inital_frames_delay:
                final = {arm: _triplet(0.0, 0.0, 1.0) for arm in self._raw}
        guard_start = time.perf_counter()
        changed = {"force": _changed_processor_keys(force_before, self.force_move_post_processor),
                   "stop": _changed_processor_keys(stop_before, self.stop_sign_post_processor)}
        guard_ms += (time.perf_counter() - guard_start) * 1000.0
        observed = _triplet(control.steer, control.throttle, control.brake)
        mismatch = final is not None and any(
            abs(observed[key] - final[self.arm][key]) > 1e-5 for key in observed)
        if any(changed.values()) or mismatch:
            diagnostic = "live_processor_state_changed" if any(changed.values()) else "shadow_mismatch"
            self._frame_log.write(json.dumps({"diagnostic": diagnostic, "step": int(self.step),
                "sim_time": self._sim_time, "arm": self.arm, "raw_control": self._raw,
                "final_control": final, "executed_control": observed,
                "speed": float(self._tick_data["speed"].item()) if self._tick_data is not None else None,
                "raw_signed_speed_mps": _logged_number(self._raw_signed_speed),
                "controller_speed_mps": _logged_number(self._controller_speed),
                "controller_reason": self._controller_reason,
                "changed_processor_keys": changed, "shadow_guard_ms": guard_ms,
                "force_state": {"stuck_detector": self.force_move_post_processor.stuck_detector,
                "force_move": self.force_move_post_processor.force_move},
                "stop_state": {"slower_stop_sign_count": self.stop_sign_post_processor.slower_stop_sign_count,
                "clear_stop_sign_cool_down": self.stop_sign_post_processor.clear_stop_sign_cool_down,
                "slower_for_stop_sign_cool_down": self.stop_sign_post_processor.slower_for_stop_sign_cool_down}},
                allow_nan=False) + "\n")
            raise RuntimeError("Shadow replay changed live postprocessor state" if any(changed.values())
                               else "Shadow postprocessing does not match executed control")
        prediction = self._prediction
        row = {
            "step": int(self.step), "sim_time": self._sim_time, "arm": self.arm,
            "route": os.environ.get("BENCHMARK_ROUTE_ID"),
            "waypoint": prediction.pred_future_waypoints[0].detach().float().cpu().tolist() if prediction else None,
            "route_prediction": prediction.pred_route[0].detach().float().cpu().tolist() if prediction else None,
            "target_speed": float(prediction.pred_target_speed_scalar[0, 0].item()) if prediction else None,
            "rear_waypoint": self._rear.tolist() if self._rear is not None else None,
            "raw_control": self._raw, "final_control": final,
            "executed_control": _triplet(control.steer, control.throttle, control.brake),
            "raw_signed_speed_mps": _logged_number(self._raw_signed_speed),
            "controller_speed_mps": _logged_number(self._controller_speed),
            "controller_reason": self._controller_reason,
            "truth": truth, "forward_ms": self._forward_ms,
            "nearby_actors": nearby_actors,
            "sensor_rear_pose": self._sensor_pose,
            "agent_ms": (time.perf_counter() - started) * 1000.0,
            "shadow_guard_ms": guard_ms,
            "heuristic": {"creeping_enabled": bool(self.config_closed_loop.sensor_agent_creeping),
                          "stop_sign_enabled": bool(self.config_closed_loop.slower_for_stop_sign),
                          "creep_active": bool(self.force_move_post_processor.force_move),
                          "stop_sign_active": bool(self.stop_sign_post_processor.stop_sign_buffer)},
        }
        if self._d3:
            row.update(d3_nav=self._d3_nav,d3_kalman=self._d3_kalman_trace,
                       d3_traffic_light=traffic_light)
        self._frame_log.write(json.dumps(row, allow_nan=False) + "\n")
        if self._d3 and final is not None:
            self._d3_prev_b = final['B']
        return control

    def destroy(self, results=None):
        if hasattr(self, "_frame_log"):
            self._frame_log.close()
        super().destroy(results)
