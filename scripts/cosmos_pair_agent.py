#!/usr/bin/env python
"""Cosmos pilot re-render (todos/2026-09-28-cosmos-pilot.md): the P5 recorder (scripts/p5_pair_agent.py, BehaviorAgent,
unchanged) drives one P5 v1 world again, and inside a tick window it also records an openpilot-style front camera at
20 Hz as RGB, depth and instance segmentation. Same XML variant + same TM seed + same expert = the same drive tick by
tick (P5 v1 checked this bit for bit); jevdrive/cosmos_pilot.py `controls` checks it again against the original
pose.jsonl before a frame is used.

Config keys on top of p5_pair_agent's (same JSON file, --agent-config):
  cosmos_windows  {route id: [k0, k1]}: record ticks k0 <= k < k1, k = round(t / 0.05) as in p5_pairs.load_world
  cosmos_cam      {"x", "y", "z", "w", "h", "fov"}: camera in the hero's actor frame (CARLA axes), pixels, horizontal fov

Outputs, next to the P5 files in B2D_ATTEMPT_OUT:
  op/rgb/<k>.png    RGB (BGR PNG, lossless)
  op/depth/<k>.png  CARLA depth encoding (R + G*256 + B*65536) / (256^3 - 1) * 1000 m, BGR PNG
  op/inst/<k>.png   CARLA instance segmentation (R = semantic tag, G + 256 B = instance id), BGR PNG
  op/frames.jsonl   per recorded tick: k, frame, t, camera world matrix (4 x 4, CARLA axes)
The cameras exist only from k0 - 3 to k1, so the rest of the drive renders exactly what P5 rendered.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import carla
import cv2
import numpy as np
from srunner.scenariomanager.timer import GameTime

import p5_pair_agent as P5

KINDS = ("rgb", "depth", "inst")
BP = {"rgb": "sensor.camera.rgb", "depth": "sensor.camera.depth", "inst": "sensor.camera.instance_segmentation"}
# P4 / P5 front camera position (Waymo roof, p4_carla_agent.WAYMO_CAMS[0] in the actor frame): a camera behind the
# windshield sees the cabin in CARLA (the instance view there is all "car"), so the rig sits on the roof like P5's.
CAM = {"x": 1.519 - 1.388633220, "y": -0.026, "z": 1.806, "w": 1280, "h": 704, "fov": 64.0}


def get_entry_point():
    return "CosmosPairAgent"


class CosmosPairAgent(P5.P5PairAgent):
    def setup(self, path_to_conf_file, _=None, __=None):
        super().setup(path_to_conf_file)
        rid = os.environ.get("BENCHMARK_ROUTE_ID", "")
        self._win = tuple(self.cfg.get("cosmos_windows", {}).get(rid, (-1, -1)))
        self._cam = dict(CAM, **self.cfg.get("cosmos_cam", {}))
        self._op = self.out / "op"
        for k in KINDS:
            (self._op / k).mkdir(parents=True, exist_ok=True)
        self._op_frames = open(self._op / "frames.jsonl", "w")
        self._op_sensors, self._op_buf = {}, {k: {} for k in KINDS}
        self._op_pool = ThreadPoolExecutor(6)
        self._op_jobs = []

    def _spawn(self):
        c, lib = self._cam, self._world.get_blueprint_library()
        tf = carla.Transform(carla.Location(x=c["x"], y=c["y"], z=c["z"]))
        for k in KINDS:
            bp = lib.find(BP[k])
            for a, v in (("image_size_x", c["w"]), ("image_size_y", c["h"]), ("fov", c["fov"])):
                bp.set_attribute(a, str(v))
            s = self._world.spawn_actor(bp, tf, attach_to=self._hero)
            s.listen(lambda img, k=k: self._op_buf[k].__setitem__(img.frame, bytes(img.raw_data)))
            self._op_sensors[k] = s

    def _despawn(self):
        for s in self._op_sensors.values():
            s.stop()
            s.destroy()
        self._op_sensors = {}

    def _record(self, k, frame, t):
        deadline = time.time() + 10.0
        while any(frame not in self._op_buf[x] for x in KINDS) and time.time() < deadline:
            time.sleep(0.002)
        c = self._cam
        imgs = {x: self._op_buf[x].pop(frame, None) for x in KINDS}
        for x in KINDS:
            for f in [f for f in self._op_buf[x] if f < frame]:
                del self._op_buf[x][f]
        if any(v is None for v in imgs.values()):
            print("cosmos: frame %d (k %d) missing %s" % (frame, k, [x for x in KINDS if imgs[x] is None]), flush=True)
            return
        M = np.array(self._op_sensors["rgb"].get_transform().get_matrix())
        self._op_frames.write(json.dumps({"k": k, "frame": frame, "t": round(t, 4), "cam": M.round(5).tolist()}) + "\n")

        def write(x, raw):
            img = np.frombuffer(raw, np.uint8).reshape(c["h"], c["w"], 4)[:, :, :3]
            cv2.imwrite(str(self._op / x / ("%05d.png" % k)), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self._op_jobs += [self._op_pool.submit(write, x, imgs[x]) for x in KINDS]

    def __call__(self):
        control = super().__call__()
        k0, k1 = self._win
        if k0 < 0:
            return control
        t = GameTime.get_time()
        k = int(round(t / 0.05))
        # spawned during tick k0 - 3, so the first image the cameras deliver is of a later frame (P5: a sensor spawned
        # during a tick has no image of that tick)
        if k0 - 3 <= k < k1 and not self._op_sensors:
            self._spawn()
        elif k0 <= k < k1:
            self._record(k, GameTime.get_frame(), t)
        elif k >= k1 and self._op_sensors:
            self._despawn()
        return control

    def destroy(self, results=None):
        if hasattr(self, "_op_frames"):
            for j in self._op_jobs:
                j.result()
            self._op_frames.close()
            if self._op_sensors:
                self._despawn()
        super().destroy(results)
