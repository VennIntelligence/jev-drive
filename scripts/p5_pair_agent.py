#!/usr/bin/env python
"""P5 pair recorder: a privileged BehaviorAgent drives one world of a counterfactual pair while we record
everything the exam needs (todos/2026-09-24-p5-carla-pairs-v0.md).

Loaded by the leaderboard through `scripts/b2d_run.py --agent scripts/p5_pair_agent.py --agent-config <json>
--python $DATA_DIR/envs/scout-tfv6/bin/python`. It IS TFv6's `SensorAgent` (LEAD cvpr2026), so the leaderboard
attaches TFv6's own rig; every tick the author's `run_step` runs on that rig's data - GPS filter, LiDAR
accumulation, JPEG round trip, ensemble forward - and its control is thrown away (shadow inference). The car is
driven by a privileged expert, config key "driver":

  behavior   CARLA's BehaviorAgent(normal), exactly as in P4 and P5 v0 (the default)
  pdm_lite   PDM-Lite (P5 v1, todos/2026-09-25-reactivity-program/i1-p5v1.md): SimLingo's Bench2Drive copy,
             leaderboard/team_code/autopilot.py, run as shipped. It needs that tree (BENCH2DRIVE_ROOT, for its
             CarlaDataProvider.active_scenarios bookkeeping) and IS_BENCH2DRIVE=1; its steering noise draws from its
             own numpy stream seeded by the TM seed, so both worlds of a pair draw the same noise.

Outputs, in the attempt directory B2D_ATTEMPT_OUT:

  meta.json       rig, route, town, weather, vehicle geometry, driver, TFv6 model
  route.json      dense route plan (CARLA world, left-handed) with RoadOptions
  lights.json     every traffic light: id, location, light boxes, stop waypoints
  pose.jsonl      per tick (20 Hz): hero truth pose / velocity / acceleration, expert control
  actors.npz      per tick: every vehicle and walker within 100 m (frame, id, x, y, z, yaw, vx, vy)
  actor_kinds.json  id -> type_id, role_name, bounding box
  frames.jsonl    per camera frame (5 Hz): JPEG paths, scenario trigger flags, light states within 100 m,
                  and the pixel count of every actor (vehicle, walker, traffic light) in an instance-segmentation
                  view of the Waymo front image, i.e. what the camera actually sees of it
  tfv6.jsonl      per tick: TFv6 target-speed distribution and scalar, 8 waypoints, 10 route points
  cams/<cam>/<frame>.jpg   the three Waymo-calibrated cameras of P4, unchanged

Python 3.10: runs in envs/scout-tfv6 (carla 0.9.15 cp310, torch 2.8 cu128).
"""
import json
import math
import os
import time
from pathlib import Path
from unittest import mock

import carla
import cv2
import numpy as np
import py_trees

from leaderboard.autoagents.autonomous_agent import Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

import p4_carla_agent as p4                      # Waymo camera model, early-stop flag (patches ScenarioManager)
from lead.common.base_agent import BaseAgent
from lead.inference.sensor_agent import SensorAgent

TFV6_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0)
DEFAULT = dict(p4.DEFAULT, sensor_tick=0.0, max_sim_s=50.0, stuck_s=30.0, after_trigger_s=20.0, tfv6_model_dir="",
               actor_radius=100.0, vis_radius=60.0, driver="behavior", save_threads=0)
FRONT = p4.WAYMO_CAMS[0]                         # ("front", x, y, z, yaw) in Waymo's rear-axle frame


def get_entry_point():
    return "P5PairAgent"


def _xyz(loc):
    return [round(loc.x, 3), round(loc.y, 3), round(loc.z, 3)]


class P5PairAgent(SensorAgent):
    def setup(self, path_to_conf_file, _=None, __=None):
        cfg = dict(DEFAULT)
        with open(path_to_conf_file.split("+")[0]) as fh:
            cfg.update(json.load(fh))
        self.cfg = cfg
        self.out = Path(os.environ["B2D_ATTEMPT_OUT"])
        self.tfv6 = bool(cfg["tfv6_model_dir"])
        if self.tfv6:
            # The author's setup refuses to start without ffmpeg (it compresses videos we never produce).
            with mock.patch("shutil.which", return_value="/bin/true"):
                super().setup(cfg["tfv6_model_dir"])
            forward = self.closed_loop_inference.forward

            def capture(*args, **kwargs):
                t0 = time.perf_counter()
                self._pred = forward(*args, **kwargs)
                self._t["tfv6_forward"].append(time.perf_counter() - t0)
                return self._pred

            self.closed_loop_inference.forward = capture
        self.track = Track.SENSORS
        for name, *_ in p4.WAYMO_CAMS:
            (self.out / "cams" / name).mkdir(parents=True, exist_ok=True)
        self.mx, self.my = p4.distortion_maps(cfg)
        self.fov = math.degrees(2 * math.atan(cfg["render_w"] / 2.0 / cfg["f"]))
        self._tick, self._ba, self._pdm, self._still_since, self._pred, self._t_trig = 0, None, None, None, None, None
        self._inited = False
        self._kinds, self._actor_rows = {}, []
        self._t = {k: [] for k in ("tick", "tfv6", "tfv6_base", "tfv6_forward", "expert", "snapshot", "save", "visibility")}
        self._pose = open(self.out / "pose.jsonl", "w")
        self._frames = open(self.out / "frames.jsonl", "w")
        self._tf = open(self.out / "tfv6.jsonl", "w")
        self._tfv6_errors = 0

    # The Bench2Drive agent base dumps a growing metric_info.json every tick when this exists; hide it.
    @property
    def get_metric_info(self):
        raise AttributeError("disabled for the recorder")

    def sensors(self):
        c = self.cfg
        own = [{"type": "sensor.camera.rgb", "id": name, "x": x + p4.REAR_AXLE_X, "y": -y, "z": z - c["origin_z"],
                "roll": 0.0, "pitch": 0.0, "yaw": -yaw, "width": c["render_w"], "height": c["render_h"], "fov": self.fov}
               for name, x, y, z, yaw in p4.WAYMO_CAMS]
        extra = super().sensors() if self.tfv6 else []
        if self.cfg["driver"] == "pdm_lite" and not any(x["id"] == "imu" for x in extra):
            extra = extra + [{"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0,
                              "yaw": 0.0, "sensor_tick": 0.05, "id": "imu"}]      # PDM-Lite's own (compass)
        return own + extra

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense = list(global_plan_world_coord)      # the leaderboard calls this before setup()
        self._plan_gps = list(global_plan_gps)

    # ------------------------------------------------------------------ one-off state

    def _pdm_lite(self):
        """PDM-Lite as shipped, as an inner agent that gets the same plan and the same input_data (it reads only the
        IMU compass; everything else it queries from the simulator)."""
        import sys
        team_code = os.path.join(os.environ["BENCH2DRIVE_ROOT"], "leaderboard", "team_code")
        if team_code not in sys.path:
            sys.path.insert(0, team_code)      # autopilot.py imports its siblings (config, nav_planner, ...) top-level
        os.environ["IS_BENCH2DRIVE"] = "1"
        # PDM-Lite was written for numpy < 1.24 (np.float etc.); these aliases are the builtins, restoring them
        # changes nothing numerically. envs/scout-tfv6 has a newer numpy.
        for k, v in (("float", float), ("int", int), ("bool", bool), ("object", object), ("complex", complex)):
            if not hasattr(np, k):
                setattr(np, k, v)
        # autopilot.py reads SAVE_PATH at import and in setup() and would then write its own dataset; LEAD wants it.
        save = os.environ.pop("SAVE_PATH", None)
        try:
            from autopilot import AutoPilot
            pdm = AutoPilot("127.0.0.1", 0, False)
            pdm.setup("p5+pdm_lite", None, None)
        finally:
            if save is not None:
                os.environ["SAVE_PATH"] = save
        pdm.set_global_plan(self._plan_gps, self._dense)
        # PDM-Lite adds 1e-3 * randn() to every steer from the global numpy stream. Give it a stream of its own, seeded
        # by the TM seed (the variant id's last digit, jevdrive/p5_pairs.py), so x+, x- and the null draw identical
        # noise whatever else in the process uses numpy.
        self._pdm_rng = np.random.RandomState(int(os.environ.get("BENCHMARK_ROUTE_ID", "0")) % 10).get_state()
        return pdm

    def _pdm_step(self, input_data, t):
        outer = np.random.get_state()
        np.random.set_state(self._pdm_rng)
        try:
            return self._pdm.run_step(input_data, t)
        finally:
            self._pdm_rng = np.random.get_state()
            np.random.set_state(outer)

    def _init_world(self):
        from agents.navigation.behavior_agent import BehaviorAgent
        hero = CarlaDataProvider.get_hero_actor()
        world, cmap = CarlaDataProvider.get_world(), CarlaDataProvider.get_map()
        self._hero, self._world = hero, world
        if self.cfg["driver"] == "pdm_lite":
            self._pdm, self._ba = self._pdm_lite(), None
            self._driver = "PDM-Lite (SimLingo Bench2Drive team_code/autopilot.py)"
        else:
            # grp_inst: the plan is handed over, so the agent never routes (P4: saves minutes on Town12)
            self._pdm = None
            self._ba = BehaviorAgent(hero, behavior=self.cfg["behavior"], map_inst=cmap, grp_inst=object())
            self._ba.set_global_plan([(cmap.get_waypoint(t.location), o) for t, o in self._dense],
                                     stop_waypoint_creation=True, clean_queue=True)
            self._driver = "carla.agents.navigation.BehaviorAgent(%s)" % self.cfg["behavior"]
        (self.out / "route.json").write_text(json.dumps(
            [{"x": t.location.x, "y": t.location.y, "z": t.location.z, "yaw": t.rotation.yaw, "option": int(o.value),
              "option_name": o.name} for t, o in self._dense]))
        # Traffic lights: locations now; boxes and stop waypoints the first time a light is within actor_radius,
        # because on a Large Map a far light is dormant and the server refuses to describe it.
        self._lights = list(world.get_actors().filter("traffic.traffic_light"))
        self._light_meta = [{"id": tl.id, "loc": _xyz(tl.get_location()), "boxes": None, "stops": None}
                            for tl in self._lights]
        self._light_xyz = np.array([l["loc"] for l in self._light_meta], np.float64).reshape(-1, 3)
        self._light_boxes = [np.zeros((0, 6))] * len(self._lights)
        # Visibility camera: instance segmentation at the Waymo front camera's pose and field of view, at half the
        # render resolution. Spawned here, not through the leaderboard (its sensor whitelist has no segmentation);
        # the G and B channels carry the actor id, so a factor actor is visible iff its pixels are in the image.
        c, (_, fx, fy, fz, _) = self.cfg, FRONT
        bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        for k, v in (("image_size_x", c["render_w"] // 2), ("image_size_y", c["render_h"] // 2), ("fov", self.fov)):
            bp.set_attribute(k, str(v))
        self._seg_buf = {}
        self._seg = world.spawn_actor(bp, carla.Transform(carla.Location(x=fx + p4.REAR_AXLE_X, y=-fy, z=fz - c["origin_z"])),
                                      attach_to=hero)
        self._seg.listen(lambda img: self._seg_buf.__setitem__(img.frame, bytes(img.raw_data)))
        # the part of the half-resolution render that the 972 x 1079 Waymo image crops (distortion ignored)
        x0, y0 = (c["render_w"] - 1) / 2 - c["cu"], (c["render_h"] - 1) / 2 - c["cv"]
        self._seg_crop = (slice(max(0, int(y0 / 2)), int((y0 + c["out_h"]) / 2)),
                          slice(max(0, int(x0 / 2)), int((x0 + c["out_w"]) / 2)))
        bb = hero.bounding_box
        self._ego_ext = np.array([bb.extent.x, bb.extent.y, bb.extent.z])
        self._ego_loc = np.array([bb.location.x, bb.location.y, bb.location.z])
        w = world.get_weather()
        meta = {"town": cmap.name, "vehicle": hero.type_id, "hero_id": hero.id, "bbox_location": _xyz(bb.location),
                "bbox_extent": self._ego_ext.tolist(), "rear_axle_x": p4.REAR_AXLE_X,
                "weather": {k: getattr(w, k) for k in ("cloudiness", "precipitation", "precipitation_deposits",
                                                       "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                                                       "fog_density", "wetness")},
                "cams": self.sensors(), "config": self.cfg, "fov_render": self.fov, "route_id": os.environ.get(
                    "BENCHMARK_ROUTE_ID"), "driver": self._driver,
                "bench2drive_root": os.environ.get("BENCH2DRIVE_ROOT", ""),
                "tfv6": {"model_dir": self.cfg["tfv6_model_dir"],
                         "models": sorted(os.listdir(self.cfg["tfv6_model_dir"])) if self.tfv6 else []},
                "route_points": len(self._dense)}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1))

    def _kind(self, aid):
        """(type_id, role, bbox) for vehicles and walkers, None for everything else; one RPC per new id."""
        k = self._kinds.get(aid, 0)
        if k == 0:
            a = self._world.get_actor(aid)
            k = None
            if a is not None and (a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")) \
                    and a.id != self._hero.id:
                b = a.bounding_box
                k = (a.type_id, a.attributes.get("role_name", ""), _xyz(b.location) + [b.extent.x, b.extent.y, b.extent.z])
            self._kinds[aid] = k
        return k

    # ------------------------------------------------------------------ per tick

    def _snapshot(self, frame):
        """Every vehicle and walker within actor_radius of the hero, from one world snapshot."""
        snap = self._world.get_snapshot()
        hl = self._hero.get_location()
        r2 = self.cfg["actor_radius"] ** 2
        rows = []
        for s in snap:
            if self._kind(s.id) is None:
                continue
            tf = s.get_transform()
            if (tf.location.x - hl.x) ** 2 + (tf.location.y - hl.y) ** 2 > r2:
                continue
            v = s.get_velocity()
            rows.append((frame, s.id, tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw, v.x, v.y))
        self._actor_rows.extend(rows)
        return rows

    def _visibility(self, frame, rows):
        """What the Waymo front camera actually sees of each vehicle, walker and traffic light within vis_radius.

        CARLA's instance ids are not actor ids, so each actor's 3-D box is projected into the instance-segmentation
        view (same pose and field of view as the front camera, half resolution, cropped to the Waymo image) and the
        visible pixels are those inside the projected box that carry the actor's semantic class (walkers 12,
        vehicles and two-wheelers 13-19, traffic lights 7) and belong to the box's dominant instance: an actor
        hidden behind a car contributes only the car's pixels, which belong to the car's own box. Returns
        {actor id: pixels, "L<light id>": pixels} for the actors that project into the image at all."""
        deadline = time.time() + 5.0
        while frame not in self._seg_buf and time.time() < deadline:
            time.sleep(0.002)
        raw = self._seg_buf.pop(frame, None)
        for f in [f for f in self._seg_buf if f < frame]:
            del self._seg_buf[f]
        if raw is None:
            return None
        c = self.cfg
        H, W = c["render_h"] // 2, c["render_w"] // 2
        img = np.frombuffer(raw, np.uint8).reshape(H, W, 4)
        tag = img[..., 2]
        inst = img[..., 1].astype(np.int32) + 256 * img[..., 0].astype(np.int32)
        (ys, xs) = self._seg_crop
        M = np.array(self._seg.get_transform().get_inverse_matrix())
        fs = W / 2.0 / math.tan(math.radians(self.fov) / 2.0)
        cam = self._seg.get_location()
        R2 = c["vis_radius"] ** 2

        def pixels(corners, classes):
            p = (M @ np.c_[corners, np.ones(len(corners))].T)[:3]
            if (p[0] <= 0.3).all():
                return 0
            p = p[:, p[0] > 0.3]
            u, v = W / 2.0 + fs * p[1] / p[0], H / 2.0 - fs * p[2] / p[0]
            u0, u1 = max(xs.start, int(u.min())), min(xs.stop, int(math.ceil(u.max())) + 1)
            v0, v1 = max(ys.start, int(v.min())), min(ys.stop, int(math.ceil(v.max())) + 1)
            if u0 >= u1 or v0 >= v1:
                return 0
            m = np.isin(tag[v0:v1, u0:u1], classes)
            if not m.any():
                return 0
            ids, n = np.unique(inst[v0:v1, u0:u1][m], return_counts=True)
            return int(n.max())

        out = {}
        for _, aid, x, y, z, yaw, _, _ in rows:
            if (x - cam.x) ** 2 + (y - cam.y) ** 2 > R2:
                continue
            tid, _, bb = self._kind(aid)
            tf = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw))
            box = carla.BoundingBox(carla.Location(*bb[:3]), carla.Vector3D(*bb[3:]))
            corners = np.array([[q.x, q.y, q.z] for q in box.get_world_vertices(tf)])
            n = pixels(corners, (12,) if tid.startswith("walker.") else (13, 14, 15, 16, 17, 18, 19))
            if n:
                out[str(aid)] = n
        near = np.flatnonzero(((self._light_xyz[:, :2] - [cam.x, cam.y]) ** 2).sum(1) <= R2) if len(self._light_xyz) else []
        for i in near:
            n = 0
            for b in self._light_boxes[i]:
                box = carla.BoundingBox(carla.Location(*b[:3]), carla.Vector3D(*b[3:]))
                n += pixels(np.array([[q.x, q.y, q.z] for q in box.get_world_vertices(carla.Transform())]), (7,))
            if n:
                out["L%d" % self._lights[i].id] = n
        return out

    def _light_states(self):
        hl = self._hero.get_location()
        near = np.flatnonzero(((self._light_xyz[:, :2] - [hl.x, hl.y]) ** 2).sum(1) <= self.cfg["actor_radius"] ** 2) \
            if len(self._light_xyz) else []
        for i in near:
            m = self._light_meta[i]
            if m["boxes"] is None:
                try:
                    tl = self._lights[i]
                    boxes = tl.get_light_boxes()
                    m["stops"] = [_xyz(w.transform.location) + [w.transform.rotation.yaw, w.road_id, w.lane_id]
                                  for w in tl.get_stop_waypoints()]
                    # On a Large Map the boxes come back shifted by the rebased world origin, a whole multiple of
                    # 1000 m (measured: Town12/13 offsets of (-1000, 5000), (-3000, 5000), (1000, 3000) m plus the
                    # mast arm, which is under 6 m); the rounding removes exactly that.
                    loc = np.array(m["loc"][:2])
                    shift = [0.0, 0.0]
                    if boxes:
                        c = np.mean([[b.location.x, b.location.y] for b in boxes], 0)
                        shift = list(np.round((loc - c) / 1000.0) * 1000.0)
                    m["box_shift"] = shift
                    m["boxes"] = [[round(b.location.x + shift[0], 3), round(b.location.y + shift[1], 3), round(b.location.z, 3),
                                   b.extent.x, b.extent.y, b.extent.z] for b in boxes]
                    self._light_boxes[i] = np.array(m["boxes"], np.float64).reshape(-1, 6)
                except RuntimeError:              # still dormant: ask again next frame
                    pass
        return {str(self._lights[i].id): str(self._lights[i].get_state()) for i in near}

    def _save_one(self, name, frame, input_data):
        bgr = np.ascontiguousarray(input_data[name][1][:, :, :3])
        img = cv2.remap(bgr, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rel = "cams/%s/%07d.jpg" % (name, frame)
        cv2.imwrite(str(self.out / rel), img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_q"])])
        return rel, round(float(img[::8, ::8].std()), 2)

    def _save(self, frame, input_data):
        """The three Waymo images: undistort-remap and JPEG. cv2 releases the GIL, so with save_threads > 0 the
        cameras run in parallel (same bytes on disk, P5 v1 profiling)."""
        names = [name for name, *_ in p4.WAYMO_CAMS]
        if self.cfg["save_threads"]:
            if getattr(self, "_pool", None) is None:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(int(self.cfg["save_threads"]))
            res = list(self._pool.map(lambda n: self._save_one(n, frame, input_data), names))
        else:
            res = [self._save_one(n, frame, input_data) for n in names]
        return {n: r[0] for n, r in zip(names, res)}, [r[1] for r in res]

    def _shadow(self, input_data, t, frame, cam_tick):
        """TFv6's own run_step on its own sensors; the control it returns is discarded.

        The exam reads TFv6 at the 5 Hz camera frames only, so the full `run_step` (the author's image / LiDAR /
        radar preprocessing and the ensemble forward, ~0.4 s a tick on this box) runs on camera ticks, and on the
        three ticks in between only the author's `BaseAgent.tick`, which is what carries state from tick to tick:
        the GPS filter, the ego pose history and the LiDAR / radar sweep queues. The model therefore sees exactly
        the inputs it would see in closed loop; what is skipped only feeds its control (PID state, stop-sign and
        creeping post-processors, the initial-frames brake), which the recorder discards anyway."""
        self._pred = None
        try:
            if not cam_tick and self.initialized:
                BaseAgent.tick(self, dict(input_data), use_kalman_filter=self.training_config.use_kalman_filter_for_gps)
                return
            SensorAgent.run_step(self, dict(input_data), t)
        except Exception as e:                    # recorded, never allowed to stop the expert's drive
            self._tfv6_errors += 1
            if self._tfv6_errors <= 3:
                print("tfv6 shadow failed at frame %d: %r" % (frame, e), flush=True)
            return
        p = self._pred
        if p is None:                              # the author's first tick only initialises
            return
        rec = {"frame": frame}
        for k in ("pred_target_speed_distribution", "pred_target_speed_scalar", "pred_future_waypoints", "pred_route"):
            v = getattr(p, k, None)
            if v is not None:
                rec[k] = np.round(v[0].detach().float().cpu().numpy(), 4).tolist()
        if "pred_target_speed_distribution" in rec:
            rec["v_expect"] = round(float(np.dot(rec["pred_target_speed_distribution"], TFV6_SPEEDS)), 4)
        self._tf.write(json.dumps(rec) + "\n")

    def __call__(self):
        t_tick = time.perf_counter()
        self._tick += 1
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        frame, t = GameTime.get_frame(), GameTime.get_time()
        if not self._inited:
            self._init_world()
            self._inited = True
        cam_tick = (self._tick - 1) % p4.CAM_TICKS == 0
        if self.tfv6:
            t0 = time.perf_counter()
            self._shadow(input_data, t, frame, cam_tick)
            self._t["tfv6" if cam_tick else "tfv6_base"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        rows = self._snapshot(frame)
        self._t["snapshot"].append(time.perf_counter() - t0)
        if cam_tick:
            t0 = time.perf_counter()
            files, std = self._save(frame, input_data)
            self._t["save"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            px = self._visibility(frame, rows)
            self._t["visibility"].append(time.perf_counter() - t0)
            bb = py_trees.blackboard.Blackboard()
            trig = [bool(bb.get("ScenarioRouteNumber%d" % i)) for i in range(2)]
            if trig[0] and self._t_trig is None:
                self._t_trig = t
            self._frames.write(json.dumps({"frame": frame, "tick": self._tick, "t": round(t, 4), "files": files,
                                           "std": std, "trig": trig, "lights": self._light_states(),
                                           "px": px}) + "\n")
        t0 = time.perf_counter()
        if self._pdm is not None:
            control = self._pdm_step(input_data, t)
        else:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) if self._ba.done() else self._ba.run_step()
        control.manual_gear_shift = False
        self._t["expert"].append(time.perf_counter() - t0)
        tf = self._hero.get_transform()
        v, a, w = self._hero.get_velocity(), self._hero.get_acceleration(), self._hero.get_angular_velocity()
        self._pose.write(json.dumps({"frame": frame, "t": round(t, 4), "x": tf.location.x, "y": tf.location.y,
                                     "z": tf.location.z, "yaw": tf.rotation.yaw, "pitch": tf.rotation.pitch,
                                     "roll": tf.rotation.roll, "vx": v.x, "vy": v.y, "vz": v.z, "ax": a.x, "ay": a.y,
                                     "az": a.z, "wz": w.z, "throttle": control.throttle, "steer": control.steer,
                                     "brake": control.brake}) + "\n")
        speed = math.hypot(v.x, v.y)
        self._still_since = (self._still_since if self._still_since is not None else t) if speed < 0.2 else None
        if t > self.cfg["max_sim_s"]:
            p4.STOP.update(flag=True, why="max_sim_s")
        elif self._t_trig is not None and t > self._t_trig + self.cfg["after_trigger_s"]:
            p4.STOP.update(flag=True, why="after_trigger")
        elif self._still_since is not None and t - self._still_since > self.cfg["stuck_s"] and t > 10:
            p4.STOP.update(flag=True, why="stuck")
        self._t["tick"].append(time.perf_counter() - t_tick)
        return control

    def destroy(self, results=None):
        # The author's destroy() compresses videos with ffmpeg; nothing of that exists here.
        if not hasattr(self, "_pose"):
            return
        for fh in (self._pose, self._frames, self._tf):
            fh.close()
        if getattr(self, "_seg", None) is not None:
            self._seg.stop()
            self._seg.destroy()
        a = np.array(self._actor_rows, np.float64).reshape(-1, 8)
        np.savez_compressed(self.out / "actors.npz", frame=a[:, 0].astype(np.int64), id=a[:, 1].astype(np.int64),
                            xyz=a[:, 2:5].astype(np.float32), yaw=a[:, 5].astype(np.float32),
                            v=a[:, 6:8].astype(np.float32))
        if hasattr(self, "_light_meta"):
            (self.out / "lights.json").write_text(json.dumps(
                [dict(m, boxes=m["boxes"] or [], stops=m["stops"] or []) for m in self._light_meta]))
        (self.out / "actor_kinds.json").write_text(json.dumps({str(k): v for k, v in self._kinds.items() if v}))
        ms = {k: round(1e3 * float(np.mean(v)), 2) for k, v in self._t.items() if v}
        (self.out / "p5_summary.json").write_text(json.dumps(
            {"ticks": self._tick, "stop": p4.STOP["why"] or "route_end", "t_trigger": self._t_trig,
             "driver": self.cfg["driver"],
             "tfv6_errors": self._tfv6_errors,
             "ms_mean": ms, "ms_p95": {k: round(1e3 * float(np.percentile(v, 95)), 2) for k, v in self._t.items() if v}}))
