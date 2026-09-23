"""Four-arm TFv6 controller experiment; model and author postprocessors stay upstream."""

from collections import deque
from copy import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from lead.inference.sensor_agent import SensorAgent
from b2d_controller import Controller
from b2d_controller_adapter import GPSProjector, PoseFilter
from b2d_tfv6_coordinates import rear_waypoints


ARMS = "ABCD"


def get_entry_point():
    return "TFv6ControllerAgent"


def _triplet(steer, throttle, brake):
    return {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)}


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
    if hasattr(processor, "stop_sign_buffer"):
        result.stop_sign_buffer = deque(processor.stop_sign_buffer, maxlen=processor.stop_sign_buffer.maxlen)
    return result


def _apply_postprocessors(raw, speed, force, stop):
    throttle, brake = force.adjust(float(speed), raw["throttle"], raw["brake"])
    throttle, brake = stop.adjust(float(speed), throttle, brake)
    return _triplet(raw["steer"], throttle, brake)


class TFv6ControllerAgent(SensorAgent):
    def setup(self, path_to_conf_file, *args, **kwargs):
        parts = path_to_conf_file.split("+")
        self.arm = (parts[1] if len(parts) > 1 and parts[1] in ARMS
                    else os.environ.get("B2D_W2_ARM", "")).upper()
        if self.arm not in ARMS:
            raise ValueError("TFv6 agent requires --arm {A,B,C,D} via model_dir+ARM or B2D_W2_ARM")
        model_path = parts[0]
        output = os.environ.get("B2D_W2_LOG_DIR")
        if not output:
            raise RuntimeError("B2D_W2_LOG_DIR is required")
        Path(output).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SAVE_PATH", output)
        super().setup(model_path, *args, **kwargs)
        _configure_author_arm(self.config_closed_loop, self.arm)
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
        self._pre_force = self._pre_stop = None
        self._tick_data = None
        self._forward_ms = None
        self._frame_log = open(Path(output) / "frames.jsonl", "w", buffering=1)

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
                self._rear = rear_waypoints(prediction.pred_future_waypoints[0].detach().float().cpu().numpy())
            except ValueError:
                self._rear = None
            self._raw = {
                "A": _triplet(prediction.route_steer, prediction.target_speed_throttle,
                              prediction.target_speed_brake),
                "B": _triplet(*self._author_waypoint_control),
            }
            speed = float(self._tick_data["speed"].item())
            yaw_rate = -float(self._motion["imu"][1][5])
            for arm, controller in self._controllers.items():
                controller.update(self._rear if self._rear is not None else np.full((8, 2), np.nan),
                                  self._sim_time, trajectory_dt=0.25)
                throttle, steer, brake = controller.step(self._sim_time, speed, yaw_rate)
                self._raw[arm] = _triplet(steer, throttle, brake)
            self._raw = {arm: _normalize_brake(raw, speed) for arm, raw in self._raw.items()}
            if self.arm in "CD":
                chosen = _select_arm_control(self.arm, self._raw)
                prediction.steer = chosen["steer"]
                prediction.throttle = chosen["throttle"]
                prediction.brake = chosen["brake"]
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
        result = super().tick(input_data)
        self._tick_data = result
        return result

    def run_step(self, input_data, timestamp, *args, **kwargs):
        self._motion = input_data
        self._sim_time = float(timestamp)
        try:
            speed = float(input_data["speed"][1]["speed"])
            imu = input_data["imu"][1]
            xy, yaw = self._pose_filter.update(input_data["gps"][1], float(imu[6]),
                                                speed, float(imu[5]), self._sim_time)
            self._sensor_pose = {"xy": xy.tolist(), "yaw": yaw,
                                 "status": self._pose_filter.diagnostics}
        except ValueError as error:
            self._sensor_pose = {"xy": None, "yaw": None,
                                 "error": str(error), "status": self._pose_filter.diagnostics}
            self._pose_filter.reset()
        self._prediction = self._raw = self._rear = None
        self._pre_force = self._pre_stop = None
        started = time.perf_counter()
        control = super().run_step(input_data, timestamp, *args, **kwargs)
        truth = None
        if getattr(self, "_vehicle", None) is not None:
            snapshot = self._vehicle.get_world().get_snapshot()
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
        final = None
        if self._raw is not None and self._pre_force is not None and self._pre_stop is not None:
            speed = float(self._tick_data["speed"].item())
            final = {arm: _apply_postprocessors(raw, speed,
                     _clone_processor(self._pre_force), _clone_processor(self._pre_stop))
                     for arm, raw in self._raw.items()}
            if self.step < self.training_config.inital_frames_delay:
                final = {arm: _triplet(0.0, 0.0, 1.0) for arm in ARMS}
            observed = _triplet(control.steer, control.throttle, control.brake)
            if any(abs(observed[key] - final[self.arm][key]) > 1e-5 for key in observed):
                raise RuntimeError("Shadow postprocessing does not match executed control")
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
            "truth": truth, "forward_ms": self._forward_ms,
            "sensor_rear_pose": self._sensor_pose,
            "agent_ms": (time.perf_counter() - started) * 1000.0,
            "heuristic": {"creeping_enabled": bool(self.config_closed_loop.sensor_agent_creeping),
                          "stop_sign_enabled": bool(self.config_closed_loop.slower_for_stop_sign),
                          "creep_active": bool(self.force_move_post_processor.force_move),
                          "stop_sign_active": bool(self.stop_sign_post_processor.stop_sign_buffer)},
        }
        self._frame_log.write(json.dumps(row, allow_nan=False) + "\n")
        return control

    def destroy(self, results=None):
        if hasattr(self, "_frame_log"):
            self._frame_log.close()
        super().destroy(results)
