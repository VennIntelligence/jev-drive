#!/usr/bin/env python
"""P4 data generator: a privileged CARLA BehaviorAgent drives a Bench2Drive route while three cameras that
mimic Waymo's FRONT / FRONT_LEFT / FRONT_RIGHT record at 5 Hz (todos/2026-09-23-p4-carla-feature-gap.md).

Loaded by the leaderboard through `scripts/b2d_run.py --agent scripts/p4_carla_agent.py --agent-config <json>`
(pass `--decimate 4` so b2d_hooks lets a camera spec carry `sensor_tick`). Everything goes to the attempt
directory b2d_route.py exports as B2D_ATTEMPT_OUT:

  meta.json          rig, route, town, weather, vehicle geometry
  route.json         the dense route plan: x, y, z, yaw (CARLA world, left-handed) and RoadOption per point
  pose.jsonl         one line per tick (20 Hz): simulator truth of the hero -- pose, velocity, acceleration
  frames.jsonl       one line per camera frame: image frame number, tick it was read at, JPEG paths
  cams/<cam>/<frame>.jpg

The camera model reproduces Waymo's calibration (read from WOD-E2E `context.camera_calibrations`): a
pinhole image with a centred principal point is rendered larger than needed, then remapped through Waymo's
radial distortion (k1, k2) onto a 972 x 1079 image whose principal point sits where Waymo's does. The driver
is privileged and never looks at the cameras.

The route ends early when enough has been recorded or the car has stood still too long: a module-level flag
is checked after every scenario tick (the same hook b2d_route.py uses for --max-ticks).

Python 3.8: runs in envs/carla.
"""
import json
import math
import os
import queue
import time
from pathlib import Path

import carla
import cv2
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.scenarios.scenario_manager import ScenarioManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

DELTA = 0.05
CAM_TICKS = 4                       # sensor_tick 0.2 s at 20 Hz
REAR_AXLE_X = -1.388633220          # actor-relative rear axle of vehicle.lincoln.mkz_2020 (controller_config.json)
# Waymo FRONT / FRONT_LEFT / FRONT_RIGHT in Waymo's vehicle frame (origin: rear axle on the ground, +y left),
# medians over WOD-E2E val. CARLA is left-handed (+y right, yaw clockwise), so y and yaw flip sign below.
WAYMO_CAMS = (("front", 1.519, 0.026, 1.806, 0.0), ("front_left", 1.445, 0.153, 1.806, 45.0),
              ("front_right", 1.482, -0.116, 1.806, -45.0))
DEFAULT = {"out_w": 972, "out_h": 1079, "f": 1113.5, "cu": 488.1, "cv": 719.2, "k1": -0.0736, "k2": -0.0366,
           "render_w": 1088, "render_h": 1560, "jpeg_q": 95, "sensor_tick": 0.2, "max_sim_s": 150.0,
           "stuck_s": 30.0, "behavior": "normal", "origin_z": 0.0}
STOP = {"flag": False, "why": ""}


def _patch_scenario_manager():
    if getattr(ScenarioManager, "_p4_patched", False):
        return
    inner = ScenarioManager._tick_scenario

    def tick(self):
        inner(self)
        if STOP["flag"]:
            self._running = False

    ScenarioManager._tick_scenario = tick
    ScenarioManager._p4_patched = True


_patch_scenario_manager()


def get_entry_point():
    return "P4Agent"


def distortion_maps(c):
    """cv2.remap maps from the Waymo-distorted output image to the centred pinhole render.

    For each output pixel the distorted normalised coordinate is (u - cu) / f; the undistorted one solves
    x_d = x_u (1 + k1 r_u^2 + k2 r_u^4) by fixed-point iteration (the distortion is mild, 10 steps reach
    float32 precision), and lands at f x_u + centre in the render."""
    u, v = np.meshgrid(np.arange(c["out_w"], dtype=np.float64), np.arange(c["out_h"], dtype=np.float64))
    xd, yd = (u - c["cu"]) / c["f"], (v - c["cv"]) / c["f"]
    xu, yu = xd.copy(), yd.copy()
    for _ in range(10):
        r2 = xu * xu + yu * yu
        s = 1.0 + c["k1"] * r2 + c["k2"] * r2 * r2
        xu, yu = xd / s, yd / s
    mx = c["f"] * xu + (c["render_w"] - 1) / 2.0
    my = c["f"] * yu + (c["render_h"] - 1) / 2.0
    assert mx.min() >= 0 and my.min() >= 0 and mx.max() <= c["render_w"] - 1 and my.max() <= c["render_h"] - 1, \
        "render too small for the distortion remap"
    return mx.astype(np.float32), my.astype(np.float32)


class P4Agent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        path = (path_to_conf_file or "").split("+")[0]      # Bench2Drive appends "+<route name>"
        cfg = dict(DEFAULT)
        with open(path) as fh:
            cfg.update(json.load(fh))
        self.cfg = cfg
        self.out = Path(os.environ["B2D_ATTEMPT_OUT"])
        for name, *_ in WAYMO_CAMS:
            (self.out / "cams" / name).mkdir(parents=True, exist_ok=True)
        self.mx, self.my = distortion_maps(cfg)
        self.fov = math.degrees(2 * math.atan(cfg["render_w"] / 2.0 / cfg["f"]))
        self._tick, self._ba, self._still_since, self._n_frames, self._buf = 0, None, None, 0, {}
        self._last_cam, self._last_saved, self._missed, self._arrivals = None, None, 0, []
        self._pose = open(self.out / "pose.jsonl", "w")
        self._frames = open(self.out / "frames.jsonl", "w")
        self._t_img = 0.0

    def sensors(self):
        c = self.cfg
        return [{"type": "sensor.camera.rgb", "id": name, "x": x + REAR_AXLE_X, "y": -y, "z": z - c["origin_z"],
                 "roll": 0.0, "pitch": 0.0, "yaw": -yaw, "width": c["render_w"], "height": c["render_h"],
                 "fov": self.fov, **({"sensor_tick": c["sensor_tick"] - 0.01} if c["sensor_tick"] > 0 else {})}
                for name, x, y, z, yaw in WAYMO_CAMS]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense = list(global_plan_world_coord)      # the leaderboard calls this before setup()

    def _init_driver(self):
        from agents.navigation.behavior_agent import BehaviorAgent
        hero = CarlaDataProvider.get_hero_actor()
        cmap = CarlaDataProvider.get_map()
        # grp_inst: the plan is handed over below, so the agent never routes; a placeholder saves building a
        # GlobalRoutePlanner over a Large Map, which is minutes on Town12.
        self._ba = BehaviorAgent(hero, behavior=self.cfg["behavior"], map_inst=cmap, grp_inst=object())
        plan = [(cmap.get_waypoint(t.location), opt) for t, opt in self._dense]
        self._ba.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        rows = [{"x": t.location.x, "y": t.location.y, "z": t.location.z, "yaw": t.rotation.yaw,
                 "option": int(opt.value), "option_name": opt.name} for t, opt in self._dense]
        (self.out / "route.json").write_text(json.dumps(rows))
        bb = hero.bounding_box
        w = CarlaDataProvider.get_world().get_weather()
        meta = {"town": cmap.name, "vehicle": hero.type_id, "bbox_location": [bb.location.x, bb.location.y, bb.location.z],
                "bbox_extent": [bb.extent.x, bb.extent.y, bb.extent.z], "rear_axle_x": REAR_AXLE_X,
                "weather": {k: getattr(w, k) for k in ("cloudiness", "precipitation", "precipitation_deposits",
                                                       "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                                                       "fog_density", "wetness")},
                "cams": self.sensors(), "config": self.cfg, "fov_render": self.fov,
                "driver": "carla.agents.navigation.BehaviorAgent(%s)" % self.cfg["behavior"],
                "route_points": len(self._dense)}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1))
        self._hero = hero

    def _log_pose(self, frame, t, control):
        tf = self._hero.get_transform()
        v, a, w = self._hero.get_velocity(), self._hero.get_acceleration(), self._hero.get_angular_velocity()
        self._pose.write(json.dumps({"frame": frame, "t": round(t, 4), "x": tf.location.x, "y": tf.location.y,
                                     "z": tf.location.z, "yaw": tf.rotation.yaw, "pitch": tf.rotation.pitch,
                                     "roll": tf.rotation.roll, "vx": v.x, "vy": v.y, "vz": v.z, "ax": a.x, "ay": a.y,
                                     "az": a.z, "wz": w.z, "throttle": control.throttle, "steer": control.steer,
                                     "brake": control.brake}) + "\n")
        return math.hypot(v.x, v.y)

    def _drain_cameras(self, frame):
        """Collect camera data keyed by the image's own frame number; return the frames all three cameras are in.

        `sensor_tick` is set a little under 4 ticks, so the cameras fire on every 4th tick exactly. Once a
        firing is due (4 ticks after the last complete set), wait for it: without the wait the simulation
        runs ahead of a slow renderer (a Large Map on a shared card) and the server drops frames -- the first
        Town12 smoke kept 4 of ~100. Labels are joined to images by the image frame number, so a set that
        lands a tick late costs nothing."""
        si = self.sensor_interface
        if self.cfg["sensor_tick"] > 0:
            due = self._last_cam is not None and frame >= self._last_cam + CAM_TICKS
            due = due or (self._last_cam is None and self._tick > 2 * CAM_TICKS)
        else:                                       # cameras on every tick: wait for this tick's set
            due = self._last_cam is None or frame > self._last_cam
        deadline = time.time() + 20.0
        out = []
        while True:
            try:
                if due and not out:
                    tag, f, data = si._data_buffers.get(True, max(0.01, deadline - time.time()))
                else:
                    tag, f, data = si._data_buffers.get_nowait()
            except queue.Empty:
                if due and not out:
                    self._missed += 1
                    self._last_cam = frame          # a lost firing: expect the next one, do not wait again
                break
            self._buf.setdefault(f, {})[tag] = data
            if len(self._arrivals) < 4000:
                self._arrivals.append((self._tick, frame, tag, f, round(time.time(), 3)))
            if len(self._buf[f]) == len(WAYMO_CAMS):
                out.append((f, self._buf.pop(f)))
                self._last_cam = f
                for old in [g for g in self._buf if g < f]:
                    del self._buf[old]          # an incomplete set older than a complete one never completes
        return out

    def _save(self, frame, got):
        t0 = time.perf_counter()
        files, std = {}, []
        for name, *_ in WAYMO_CAMS:
            bgr = np.ascontiguousarray(got[name][:, :, :3])
            img = cv2.remap(bgr, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rel = "cams/%s/%07d.jpg" % (name, frame)
            cv2.imwrite(str(self.out / rel), img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_q"])])
            files[name] = rel
            std.append(round(float(img[::8, ::8].std()), 2))
        self._frames.write(json.dumps({"frame": frame, "tick_frame": GameTime.get_frame(), "std": std,
                                       "files": files}) + "\n")
        self._n_frames += 1
        self._t_img += time.perf_counter() - t0

    def __call__(self):
        self._tick += 1
        if self._ba is None:
            self._init_driver()
        frame, t = GameTime.get_frame(), GameTime.get_time()
        for f, got in self._drain_cameras(frame):
            if self._last_saved is None or f - self._last_saved >= CAM_TICKS:
                self._save(f, got)
                self._last_saved = f
        control = self.run_step(None, t)
        speed = self._log_pose(frame, t, control)
        # Stop conditions: enough simulated time, or standing still for too long (a blocked lane, a jam).
        self._still_since = (self._still_since if self._still_since is not None else t) if speed < 0.2 else None
        if t > self.cfg["max_sim_s"]:
            STOP.update(flag=True, why="max_sim_s")
        elif self._still_since is not None and t - self._still_since > self.cfg["stuck_s"] and t > 10:
            STOP.update(flag=True, why="stuck")
        return control

    def run_step(self, input_data, timestamp):
        if self._ba.done():
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        control = self._ba.run_step()
        control.manual_gear_shift = False
        return control

    def destroy(self):
        if not hasattr(self, "_pose"):
            return
        for fh in (self._pose, self._frames):
            fh.close()
        with open(self.out / "arrivals.jsonl", "w") as fh:
            for a in self._arrivals:
                fh.write(json.dumps(dict(zip(("tick", "tick_frame", "tag", "frame", "wall"), a))) + "\n")
        (self.out / "p4_summary.json").write_text(json.dumps(
            {"ticks": self._tick, "camera_frames": self._n_frames, "missed_waits": self._missed, "stop": STOP["why"] or "route_end",
             "image_ms_per_frame": 1e3 * self._t_img / max(self._n_frames, 1)}))
