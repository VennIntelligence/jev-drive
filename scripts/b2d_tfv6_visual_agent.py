"""LEAD TFv6 official agent with a read-only live preview for the Tokyo monitor."""
import os
from pathlib import Path
import time

import cv2
import numpy as np

from lead.inference.sensor_agent import SensorAgent
from drive_runtime.preview import LivePreview


def get_entry_point():
    return "VisualTFv6Agent"


def _points(value):
    if value is None:
        return []
    return value[0].detach().float().cpu().tolist()


class VisualTFv6Agent(SensorAgent):
    """Publish model inputs and official predictions without changing control behavior."""

    def setup(self, path_to_conf_file, *args, **kwargs):
        super().setup(path_to_conf_file, *args, **kwargs)
        preview_dir = os.environ.get("B2D_PREVIEW_DIR")
        if not preview_dir:
            raise RuntimeError("B2D_PREVIEW_DIR is required for the TFv6 visual agent")
        self._preview = LivePreview(preview_dir)
        self._preview_input = None
        self._preview_tick_data = None
        self._preview_prediction = None
        self._preview_inference_ms = None

        forward = self.closed_loop_inference.forward

        def capture_forward(*forward_args, **forward_kwargs):
            start = time.perf_counter()
            prediction = forward(*forward_args, **forward_kwargs)
            self._preview_inference_ms = (time.perf_counter() - start) * 1000
            self._preview_prediction = prediction
            return prediction

        self.closed_loop_inference.forward = capture_forward

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        bgr = tick_data.get("original_rgb")
        if bgr is not None:
            self._preview_input = np.ascontiguousarray(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
        self._preview_tick_data = tick_data
        return tick_data

    def run_step(self, input_data, timestamp, *args, **kwargs):
        control = super().run_step(input_data, timestamp, *args, **kwargs)
        if self._preview_input is None or self._preview_tick_data is None:
            return control

        tick_data = self._preview_tick_data
        prediction = self._preview_prediction
        route = _points(prediction.pred_route) if prediction is not None else []
        waypoints = _points(prediction.pred_future_waypoints) if prediction is not None else []
        target_speed = None
        if prediction is not None and prediction.pred_target_speed_scalar is not None:
            target_speed = float(prediction.pred_target_speed_scalar[0, 0].item())
        self._preview.publish(
            {
                "agent": "TFv6",
                "t": time.time(),
                "running": True,
                "route": os.environ.get("BENCHMARK_ROUTE_ID", "Bench2Drive"),
                "step": int(self.step),
                # SensorAgent.tick has already fused/decoded the raw leaderboard inputs by the
                # time run_step returns. Use its own monotonic tick count for the preview frame.
                "frame": int(self.step),
                "speed_mps": float(tick_data["speed"].item()),
                "target_speed_mps": target_speed,
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steer": float(control.steer),
                "route_local": route,
                "pred_wp": waypoints,
                "prediction_origin": [0.0, 0.0],
                "prediction_frame": "CARLA actor frame (x forward, y right)",
                "inference_ms": self._preview_inference_ms,
                "ensemble_size": len(self.closed_loop_inference.nets),
            },
            {"input": self._preview_input},
        )
        return control

    def destroy(self, results=None):
        if hasattr(self, "_preview"):
            self._preview.close()
        super().destroy(results)
