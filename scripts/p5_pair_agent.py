#!/usr/bin/env python
"""P5 pair recorder: a privileged BehaviorAgent drives one world of a counterfactual pair while we record
everything the exam needs (todos/2026-09-24-p5-carla-pairs-v0.md).

Loaded by the leaderboard through `scripts/b2d_run.py --agent scripts/p5_pair_agent.py --agent-config <json>
--python $DATA_DIR/envs/scout-tfv6/bin/python`. It IS TFv6's `SensorAgent` (LEAD cvpr2026), so the leaderboard
attaches TFv6's own rig; every tick the author's `run_step` runs on that rig's data - GPS filter, LiDAR
accumulation, JPEG round trip, ensemble forward - and its control is thrown away (shadow inference). The car is
driven by CARLA's BehaviorAgent, exactly as in P4. Outputs, in the attempt directory B2D_ATTEMPT_OUT:

  meta.json       rig, route, town, weather, vehicle geometry, driver, TFv6 model
  route.json      dense route plan (CARLA world, left-handed) with RoadOptions
  lights.json     every traffic light: id, location, light boxes, stop waypoints
  pose.jsonl      per tick (20 Hz): hero truth pose / velocity / acceleration, expert control
  actors.npz      per tick: every vehicle and walker within 100 m (frame, id, x, y, z, yaw, vx, vy)
  actor_kinds.json  id -> type_id, role_name, bounding box
  frames.jsonl    per camera frame (5 Hz): JPEG paths, scenario trigger flags, light states within 100 m,
                  and for every vehicle / walker / light within 60 m in the Waymo front camera's frustum:
                  whether a ray from the camera reaches it unoccluded
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
               actor_radius=100.0, vis_radius=60.0)
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
        self._tick, self._ba, self._still_since, self._pred, self._t_trig = 0, None, None, None, None
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
        return own + (super().sensors() if self.tfv6 else [])

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense = list(global_plan_world_coord)      # the leaderboard calls this before setup()

    # ------------------------------------------------------------------ one-off state

    def _init_world(self):
        from agents.navigation.behavior_agent import BehaviorAgent
        hero = CarlaDataProvider.get_hero_actor()
        world, cmap = CarlaDataProvider.get_world(), CarlaDataProvider.get_map()
        self._hero, self._world = hero, world
        # grp_inst: the plan is handed over, so the agent never routes (P4: saves minutes on Town12)
        self._ba = BehaviorAgent(hero, behavior=self.cfg["behavior"], map_inst=cmap, grp_inst=object())
        self._ba.set_global_plan([(cmap.get_waypoint(t.location), o) for t, o in self._dense],
                                 stop_waypoint_creation=True, clean_queue=True)
        (self.out / "route.json").write_text(json.dumps(
            [{"x": t.location.x, "y": t.location.y, "z": t.location.z, "yaw": t.rotation.yaw, "option": int(o.value),
              "option_name": o.name} for t, o in self._dense]))
        self._lights = list(world.get_actors().filter("traffic.traffic_light"))
        lights = []
        for tl in self._lights:
            boxes = tl.get_light_boxes()
            stops = tl.get_stop_waypoints()
            lights.append({"id": tl.id, "loc": _xyz(tl.get_location()),
                           "boxes": [_xyz(b.location) + [b.extent.x, b.extent.y, b.extent.z] for b in boxes],
                           "stops": [_xyz(w.transform.location) + [w.transform.rotation.yaw, w.road_id, w.lane_id]
                                     for w in stops]})
        (self.out / "lights.json").write_text(json.dumps(lights))
        self._light_xyz = np.array([l["loc"] for l in lights], np.float64).reshape(-1, 3)
        self._light_box = [np.array([b[:3] for b in l["boxes"]] or [l["loc"]], np.float64) for l in lights]
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
                    "BENCHMARK_ROUTE_ID"), "driver": "carla.agents.navigation.BehaviorAgent(%s)" % self.cfg["behavior"],
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

    def _front_cam(self):
        """World transform of the Waymo front camera and its world->camera matrix."""
        _, x, y, z, _ = FRONT
        tf = self._hero.get_transform()
        loc = tf.transform(carla.Location(x=x + p4.REAR_AXLE_X, y=-y, z=z - self.cfg["origin_z"]))
        cam = carla.Transform(loc, tf.rotation)
        return loc, np.array(cam.get_inverse_matrix()), np.array(tf.get_inverse_matrix())

    def _in_frustum(self, M, pts):
        """Any of the world points projects inside the 972 x 1079 Waymo front image (pinhole, distortion ignored)."""
        c = self.cfg
        p = (M @ np.c_[pts, np.ones(len(pts))].T)[:3]          # camera frame: x forward, y right, z up
        ok = p[0] > 0.5
        u = c["cu"] + c["f"] * p[1] / np.where(ok, p[0], 1)
        v = c["cv"] - c["f"] * p[2] / np.where(ok, p[0], 1)
        return bool((ok & (u >= 0) & (u < c["out_w"]) & (v >= 0) & (v < c["out_h"])).any())

    def _ray_clear(self, cam, target, inside, Mego):
        """A ray from the camera to `target` meets nothing before it except the target itself and the hero."""
        d = cam.distance(target)
        for hit in self._world.cast_ray(cam, target):
            p = hit.location
            if cam.distance(p) >= d - 0.3 or inside(p):
                continue
            q = Mego @ np.array([p.x, p.y, p.z, 1.0])
            if (np.abs(q[:3] - self._ego_loc) <= self._ego_ext + 0.2).all():
                continue
            return False, "%s@%.1f" % (hit.label, cam.distance(p))
        return True, ""

    def _visibility(self, rows):
        """Frustum and occlusion for every vehicle / walker / light within vis_radius of the front camera."""
        cam, M, Mego = self._front_cam()
        R = self.cfg["vis_radius"]
        out = []
        for _, aid, x, y, z, yaw, _, _ in rows:
            if (x - cam.x) ** 2 + (y - cam.y) ** 2 > R * R:
                continue
            _, _, bb = self._kind(aid)
            tf = carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=yaw))
            ext = carla.Vector3D(bb[3], bb[4], bb[5])
            box = carla.BoundingBox(carla.Location(*bb[:3]), ext)
            corners = np.array([[v.x, v.y, v.z] for v in box.get_world_vertices(tf)])
            centre = tf.transform(carla.Location(*bb[:3]))
            if not self._in_frustum(M, np.r_[corners, [[centre.x, centre.y, centre.z]]]):
                continue
            Mi = np.array(tf.get_inverse_matrix())

            def inside(p, Mi=Mi, bb=bb):
                q = Mi @ np.array([p.x, p.y, p.z, 1.0])
                return (np.abs(q[:3] - np.array(bb[:3])) <= np.array(bb[3:]) + 0.3).all()
            top = centre + carla.Location(z=0.6 * bb[5])
            res = [self._ray_clear(cam, t, inside, Mego) for t in (centre, top)]
            out.append({"id": aid, "vis": any(r[0] for r in res), "blk": res[0][1] or res[1][1],
                        "d": round(cam.distance(centre), 2)})
        lights = []
        if len(self._light_xyz):
            near = np.flatnonzero(((self._light_xyz[:, :2] - [cam.x, cam.y]) ** 2).sum(1) <= R * R)
            for i in near:
                boxes = self._light_box[i]
                if not self._in_frustum(M, boxes):
                    continue
                vis = False
                for b in boxes:
                    tgt = carla.Location(*b)
                    vis = self._ray_clear(cam, tgt, lambda p, tgt=tgt: p.distance(tgt) < 1.0, Mego)[0]
                    if vis:
                        break
                lights.append({"id": self._lights[i].id, "vis": vis, "d": round(cam.distance(carla.Location(*boxes[0])), 2)})
        return out, lights

    def _light_states(self):
        hl = self._hero.get_location()
        near = np.flatnonzero(((self._light_xyz[:, :2] - [hl.x, hl.y]) ** 2).sum(1) <= self.cfg["actor_radius"] ** 2) \
            if len(self._light_xyz) else []
        return {str(self._lights[i].id): str(self._lights[i].get_state()) for i in near}

    def _save(self, frame, input_data):
        files, std = {}, []
        for name, *_ in p4.WAYMO_CAMS:
            bgr = np.ascontiguousarray(input_data[name][1][:, :, :3])
            img = cv2.remap(bgr, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rel = "cams/%s/%07d.jpg" % (name, frame)
            cv2.imwrite(str(self.out / rel), img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg["jpeg_q"])])
            files[name] = rel
            std.append(round(float(img[::8, ::8].std()), 2))
        return files, std

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
        if self._ba is None:
            self._init_world()
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
            vis, lvis = self._visibility(rows)
            self._t["visibility"].append(time.perf_counter() - t0)
            bb = py_trees.blackboard.Blackboard()
            trig = [bool(bb.get("ScenarioRouteNumber%d" % i)) for i in range(2)]
            if trig[0] and self._t_trig is None:
                self._t_trig = t
            self._frames.write(json.dumps({"frame": frame, "tick": self._tick, "t": round(t, 4), "files": files,
                                           "std": std, "trig": trig, "lights": self._light_states(),
                                           "vis": vis, "lvis": lvis}) + "\n")
        t0 = time.perf_counter()
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
        a = np.array(self._actor_rows, np.float64).reshape(-1, 8)
        np.savez_compressed(self.out / "actors.npz", frame=a[:, 0].astype(np.int64), id=a[:, 1].astype(np.int64),
                            xyz=a[:, 2:5].astype(np.float32), yaw=a[:, 5].astype(np.float32),
                            v=a[:, 6:8].astype(np.float32))
        (self.out / "actor_kinds.json").write_text(json.dumps({str(k): v for k, v in self._kinds.items() if v}))
        ms = {k: round(1e3 * float(np.mean(v)), 2) for k, v in self._t.items() if v}
        (self.out / "p5_summary.json").write_text(json.dumps(
            {"ticks": self._tick, "stop": p4.STOP["why"] or "route_end", "t_trigger": self._t_trig,
             "tfv6_errors": self._tfv6_errors,
             "ms_mean": ms, "ms_p95": {k: round(1e3 * float(np.percentile(v, 95)), 2) for k, v in self._t.items() if v}}))
