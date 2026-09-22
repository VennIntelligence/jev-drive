"""Official TCP policy with a shared-memory, read-only preview and phase timings.

Policy cameras/preprocessing/control are unchanged. The existing debug BEV camera
is repositioned behind the car; it is not an input to the network. No extra sensors.
Set B2D_TCP_OPTIMIZE=1 for same-frame preprocessing overlap and copy elimination.
Individual B2D_TCP_* switches override the umbrella flag for controlled comparisons.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import math
import os
from pathlib import Path
from queue import Empty
import time

import carla
import cv2
import numpy as np
import torch
from team_code.tcp_b2d_agent import TCPAgent
from srunner.scenariomanager.timer import GameTime
from drive_runtime.preview import LivePreview
from b2d_tcp_preprocess import model_rgb, jpeg_camera, combine_rgb, CAMERAS
from drive_runtime.sensors import TimedSensorQueue, collect_frame
from leaderboard.envs.sensor_interface import SensorReceivedNoData

CHASE = dict(x=-6.0, y=0.0, z=3.5, pitch=-15.0, yaw=0.0, roll=0.0,
             width=640, height=360, fov=90.0)


def get_entry_point():
    return 'VisualTCPAgent'


class VisualTCPAgent(TCPAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        for key in ('PIPELINE', 'FAST_COLOR', 'DEBUG_VIEWS', 'EARLY_RGB'):
            os.environ.setdefault('B2D_TCP_' + key, os.environ.get('B2D_TCP_OPTIMIZE', '0'))
        self._image_pool = ThreadPoolExecutor(max_workers=3)
        pipeline = os.environ['B2D_TCP_PIPELINE'] == '1'
        early = os.environ['B2D_TCP_EARLY_RGB'] == '1'
        self._async_display = os.environ.get('B2D_ASYNC_DISPLAY', '0') == '1'
        self._sensor_queue = TimedSensorQueue(self._image_pool,
            transforms={tag: jpeg_camera for tag in CAMERAS} if pipeline else {},
            combine=combine_rgb if early and pipeline else None,
            optional_tags=('bev',) if self._async_display else ())
        self._preview_history = {}
        # setup precedes sensor registration; never replace a populated queue.
        if self.sensor_interface._sensors_objects or not self.sensor_interface._data_buffers.empty():
            raise RuntimeError('Sensor profiling must be installed before sensor registration')
        self.sensor_interface._data_buffers = self._sensor_queue
        self._sensor_traces = []
        self._dir = Path(os.environ['B2D_PREVIEW_DIR'])
        self._writer = LivePreview(self._dir)
        self._state, self._images = {}, {}
        self._recent = deque(maxlen=40)
        self._samples = []
        self._events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self._hooks = [self.net.register_forward_pre_hook(self._before_forward),
                       self.net.register_forward_hook(self._after_forward)]
        camera = carla.Transform(carla.Location(x=CHASE['x'], z=CHASE['z']),
                                 carla.Rotation(pitch=CHASE['pitch']))
        self._camera_inverse = camera.get_inverse_matrix()

    def sensors(self):
        sensors = super().sensors()
        for sensor in sensors:
            if sensor['id'] == 'bev':
                sensor.update(CHASE)
        return sensors

    def _before_forward(self, model, inputs):
        self._events[0].record()
        self._inferred = True

    def _after_forward(self, model, inputs, output):
        self._events[1].record()

    def tick(self, input_data):
        start = time.perf_counter()
        # Same official operations/pixels, independent JPEG cameras processed concurrently.
        self.step += 1
        compass = input_data['IMU'][1][-1]
        if math.isnan(compass):
            compass = 0.0
        pos = self.gps_to_location(input_data['GPS'][1][:2])
        next_wp, next_cmd = self._route_planner.run_step(pos)
        theta = compass - np.pi / 2
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        queue = getattr(self, '_sensor_queue', None)
        futures = queue.futures if queue is not None and queue.pipeline else None
        debug_views = os.environ.get('B2D_TCP_DEBUG_VIEWS', '0') == '1'
        rgb_future = queue.combined_future if queue is not None else None
        data = dict(rgb=(rgb_future.result() if rgb_future is not None else
                         model_rgb(input_data, self._image_pool, futures)),
                    rgb_front=(input_data['CAM_FRONT'][1][:, :, 2::-1] if debug_views else
                               cv2.cvtColor(input_data['CAM_FRONT'][1][:, :, :3], cv2.COLOR_BGR2RGB)),
                    bev=(input_data['bev'][1][:, :, 2::-1] if debug_views else
                         cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)),
                    gps=pos, speed=input_data['SPEED'][1]['speed'], compass=compass,
                    next_command=next_cmd.value, target_point=tuple(rotation.dot(next_wp - pos)))
        self._preprocess_ms = (time.perf_counter() - start) * 1000
        self._rgb = data['rgb']  # Exact existing policy input, no preview preprocessing.
        # Reuse planner points and compass already consumed by the policy; no map RPC.
        theta = data['compass'] - np.pi / 2
        rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        self._route_local = [(rotation @ (p[0] - data['gps'])).tolist()
                             for p in islice(self._route_planner.route, 24)]
        return data

    def __call__(self):
        start = time.perf_counter()
        self._sensor_queue.begin(GameTime.get_frame())
        if self._async_display:
            required = self.sensor_interface._sensors_objects.keys() - {'bev'}
            try:
                inputs = collect_frame(self._sensor_queue, required, GameTime.get_frame(),
                                       self.sensor_interface._queue_timeout)
            except Empty:
                raise SensorReceivedNoData('A required model sensor took too long to send data')
            # Only the display path consumes bev; it never enters official TCP inference.
            display = self._sensor_queue.optional_frame('bev', GameTime.get_frame())
            inputs['bev'] = display or (-1, np.zeros((CHASE['height'], CHASE['width'], 4), dtype=np.uint8))
        else:
            inputs = self.sensor_interface.get_data(GameTime.get_frame())
        self._sensor_ms = (time.perf_counter() - start) * 1000
        self._sensor_traces.append(dict(frame=GameTime.get_frame(), sensors=dict(self._sensor_queue.trace)))
        control = self.run_step(inputs, GameTime.get_time())
        control.manual_gear_shift = False
        return control

    def run_step(self, input_data, timestamp):
        start = time.perf_counter()
        self._inferred = False
        control = super().run_step(input_data, timestamp)
        control = self._postprocess_control(control, input_data, timestamp)
        core_ms = (time.perf_counter() - start) * 1000
        gpu_ms = 0.0
        if self._inferred:
            self._events[1].synchronize()
            gpu_ms = self._events[0].elapsed_time(self._events[1])
        begin_preview = time.perf_counter()
        metadata = getattr(self, 'pid_metadata', {})
        wp = [metadata['wp_%d' % i] for i in range(1, 5) if 'wp_%d' % i in metadata]
        phases = {k: float(np.mean([s[k] for s in self._recent]))
                  for k in self._recent[0]} if self._recent else {}
        self._state = dict(t=time.time(), running=True, route=self.save_name, step=self.step,
                           frame=GameTime.get_frame(), sim_s=float(timestamp),
                           speed_mps=float(input_data['SPEED'][1]['speed']),
                           throttle=float(control.throttle), brake=float(control.brake),
                           steer=float(control.steer), pred_wp=[[float(p[1]), float(p[0])] for p in wp],
                           prediction_origin=[-1.4, 0.0],
                           route_local=[[p[0] - 1.4, p[1]] for p in self._route_local],
                           sensor_trace=self._sensor_queue.trace,
                           optimization=dict(pipeline=self._sensor_queue.pipeline,
                               fast_color=os.environ.get('B2D_TCP_FAST_COLOR', '0'),
                               debug_views=os.environ.get('B2D_TCP_DEBUG_VIEWS', '0'),
                               early_rgb=bool(self._sensor_queue.combine), async_display=self._async_display),
                           camera_inverse=self._camera_inverse, camera_fov=CHASE['fov'], phases=phases)
        self._preview_history[self._state['frame']] = {k: self._state[k] for k in
            ('pred_wp', 'prediction_origin', 'camera_inverse', 'camera_fov', 'frame')}
        for frame in list(self._preview_history):
            if frame < self._state['frame'] - 8:
                del self._preview_history[frame]
        self._state['chase_frame'] = int(input_data['bev'][0])
        self._state['chase_state'] = self._preview_history.get(self._state['chase_frame'])
        self._images = {'input': self._rgb, 'chase': input_data['bev'][1][:, :, 2::-1]}
        self._writer.publish(self._state, self._images)
        sample = dict(sensor_wait_ms=self._sensor_ms, preprocess_ms=self._preprocess_ms,
                      gpu_ms=gpu_ms, policy_ms=core_ms,
                      preview_ms=(time.perf_counter() - begin_preview) * 1000)
        self._samples.append(sample)
        self._recent.append(sample)
        return control

    def _postprocess_control(self, control, input_data, timestamp):
        """Explicit extension point; the visual-only agent preserves native control."""
        return control

    def save(self, tick_data):
        # Keep SAVE_PATH's official sensor-radius validation switch, not PNG dumping.
        pass

    def destroy(self):
        if hasattr(self, '_image_pool'):
            self._image_pool.shutdown(wait=True)
        if hasattr(self, '_writer'):
            for hook in self._hooks:
                hook.remove()
            self._writer.close()
            samples = self._samples[20:] or self._samples
            report = {k: dict(mean=float(np.mean([s[k] for s in samples])),
                              p95=float(np.percentile([s[k] for s in samples], 95)))
                      for k in samples[0]} if samples else {}
            (self._dir / ('performance-' + self.save_name + '.json')).write_text(
                json.dumps(dict(samples=len(samples), phases=report), indent=2))
            (self._dir / ('sensor-trace-' + self.save_name + '.json')).write_text(
                json.dumps(self._sensor_traces))
        super().destroy()
