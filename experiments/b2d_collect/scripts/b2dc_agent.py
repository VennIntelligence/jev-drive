#!/usr/bin/env python
"""B2D collector agent: PDM-Lite drives (SimLingo's Bench2Drive copy, leaderboard/team_code/autopilot.py, run as shipped, as in
scripts/b2d_expert_agent.py) while the openpilot road + wide cameras on the open-loop-aligned rig are recorded at 20 Hz as the model frames
Cinque consumes, together with ego state, the expert's controls and plan, the route, other agents and traffic lights.

Loaded by the leaderboard through scripts/b2d_run.py --agent experiments/b2d_collect/scripts/b2dc_agent.py --agent-config <json>[+tag]
--python $DATA_DIR/envs/simlingo/bin/python (BENCH2DRIVE_ROOT=$DATA_DIR/third_party/simlingo/Bench2Drive, IS_BENCH2DRIVE=1, SAVE_PATH unset).

Rig: scripts/zeroshot_rigs.openpilot_sensor_specs(tick 0.05, mount) with mount = jevdrive.openpilot.interface.B2D_SPEC_MOUNT (x 1.59 m ahead
of the rear axle, z 1.86 m, level): the `spec` B2D camera of decision 127. Frames are packed exactly as the closed-loop policy server packs
them (lib/b2dc_frames.Packer) and stored, one 512 x 512 yuv420p picture per tick, in clip/frames.mp4 (libx264, crf 0 = lossless by default).

Outputs in clip/ under the attempt directory ($B2D_ATTEMPT_OUT):
  frames.mp4    packed (road, wide) model frames, one per tick (ego.npz `vid` = picture index; -1 = sensor frame missing, never so far)
  chase.mp4     optional third-person view (cfg chase; a plain CARLA camera on the hero, not a leaderboard sensor), about every
                `chase_every` ticks (ego.npz `chase` = picture index or -1; the newest image delivered by that tick)
  ego.npz       per tick (20 Hz), CARLA world frame (left-handed: x, y, yaw clockwise): t, frame, sensor_frame (road, wide: the frame
                the leaderboard's camera data came from; must equal frame), loc (x, y, z of the actor origin), rot
                (pitch, yaw, roll deg), vel / acc / angvel (world), speed, wheel steer angle (deg, front left), ctl_applied / ctl_expert
                (steer, throttle, brake), driver (0 expert, 1 policy), PDM-Lite internals (target speed, junction, stop sign / walker flags,
                its remaining route ahead: 64 points at 1 m, the lane-shifted path it steers along), traffic light affecting the ego
                (id, state), speed limit
  route.npz     dense route of the leaderboard (x, y, z, yaw, RoadOption int), the input any GPS-navigation agent gets
  actors.npz    rows (tick, id, x, y, z, yaw, vx, vy, vz) of every vehicle / walker within actor_radius each tick, static props every
                prop_every ticks; kinds.json id -> type_id, role, bbox extent / offset
  lights.json   traffic lights near the route (id, location, yaw, trigger volume, stop waypoints) and stop signs; lights.npz per tick
                (tick, id, state) of the lights within light_radius
  scen.jsonl    PDM-Lite's CarlaDataProvider.active_scenarios whenever it changes (tick, entries with actors as ids)
  meta.json     route id, town, weather, rig (sensor specs, intrinsics, model K, mount, rear axle), vehicle extent, config, timings
  DONE          written last by destroy(); a clip without it is incomplete

Early end: ego below 0.1 m/s for cfg stuck_s (45 s) or cfg max_sim_s (240 s) ends the route (timing.stop in meta.json / DONE).

DAgger: cfg driver "policy" (not built) would apply a learned policy's control while PDM-Lite still runs every tick on the true state, so
ctl_expert keeps labelling; the hook is `_drive()`.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(HERE.parent / "lib")]
TEAM_CODE = os.path.join(os.environ.get("BENCH2DRIVE_ROOT", ""), "leaderboard", "team_code")
if TEAM_CODE not in sys.path:
    sys.path.insert(0, TEAM_CODE)
os.environ.setdefault("IS_BENCH2DRIVE", "1")
os.environ.pop("SAVE_PATH", None)

import carla  # noqa: E402
from autopilot import AutoPilot  # noqa: E402
from leaderboard.scenarios.scenario_manager import ScenarioManager  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: E402
from srunner.scenariomanager.timer import GameTime  # noqa: E402

import b2dc_frames as F  # noqa: E402
import zeroshot_rigs as rigs  # noqa: E402

B2D_SPEC_MOUNT = (1.59, 0.0, 1.86)                 # jevdrive.openpilot.interface.B2D_SPEC_MOUNT (not imported: keeps the agent env lean)
DEFAULT = dict(mount=list(B2D_SPEC_MOUNT), tick=0.05, crf=0, preset="veryfast", enc_threads=2,
               chase=True, chase_w=480, chase_h=270, chase_every=4, chase_crf=26,
               actor_radius=80.0, light_radius=120.0, prop_every=20, route_ahead=64, driver="expert",
               stuck_s=45.0, max_sim_s=240.0)
ROUTE_OPT = {"VOID": -1, "LEFT": 1, "RIGHT": 2, "STRAIGHT": 3, "LANEFOLLOW": 4, "CHANGELANELEFT": 5, "CHANGELANERIGHT": 6}


STOP = {"flag": False, "why": ""}


def _patch_scenario_manager():
    """End the route after the current tick when STOP is set (scripts/p4_carla_agent.py's hook, as b2d_route.py --max-ticks)."""
    if getattr(ScenarioManager, "_b2dc_patched", False):
        return
    inner = ScenarioManager._tick_scenario

    def tick(self):
        inner(self)
        if STOP["flag"]:
            self._running = False

    ScenarioManager._tick_scenario = tick
    ScenarioManager._b2dc_patched = True


_patch_scenario_manager()


def get_entry_point():
    return "B2DCollectAgent"


def _xyz(v):
    return [v.x, v.y, v.z]


class B2DCollectAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        cfg = dict(DEFAULT)
        p = (path_to_conf_file or "").split("+")[0]
        if p and os.path.isfile(p):
            with open(p) as fh:
                cfg.update(json.load(fh))
        self.cfg = cfg
        self.out = Path(os.environ.get("B2D_ATTEMPT_OUT", ".")) / "clip"
        self.out.mkdir(parents=True, exist_ok=True)
        self.cam_specs = rigs.openpilot_sensor_specs(tick=cfg["tick"], mount=tuple(cfg["mount"]))
        self.packer = F.Packer(rigs.OP_CAMERA_WH)
        self.enc = F.Encoder(self.out / "frames.mp4", crf=cfg["crf"], preset=cfg["preset"], threads=cfg["enc_threads"])
        self.chase_enc = F.Encoder(self.out / "chase.mp4", w=cfg["chase_w"], h=cfg["chase_h"], fmt="bgra", crf=cfg["chase_crf"],
                                   fps=20.0 / cfg["chase_every"], threads=1) if cfg["chase"] else None
        self.rows = {k: [] for k in ("t", "frame", "loc", "rot", "vel", "acc", "angvel", "speed", "wheel", "ctl_applied", "ctl_expert",
                                     "driver", "target_speed", "flags", "ahead", "tl", "speed_limit", "vid", "chase", "sensor_frame")}
        self.actor_rows, self.light_rows, self.kinds = [], [], {}
        self.scen = open(self.out / "scen.jsonl", "w")
        self._last_scen = None
        self._k = 0
        self._t = {"pack": 0.0, "enc": 0.0, "log": 0.0, "expert": 0.0, "wall0": time.time()}
        self._static_done = False
        self._chase = None
        self._still = 0
        STOP.update(flag=False, why="")

    # The leaderboard reads sensors() after setup(); PDM-Lite's own IMU stays first (it reads input_data["imu"]). The chase view is not a
    # leaderboard sensor (it allows a 3 m mounting radius): _chase_start() spawns it as a plain CARLA camera on the hero, for people only.
    def sensors(self):
        return super().sensors() + [dict(s) for s in self.cam_specs]

    def _chase_start(self, world, hero):
        import queue
        c = self.cfg
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", c["chase_w"]), ("image_size_y", c["chase_h"]), ("fov", 90.0), ("sensor_tick", c["tick"] * c["chase_every"])):
            bp.set_attribute(k, str(v))
        tf = carla.Transform(carla.Location(x=-7.5, z=3.4), carla.Rotation(pitch=-13.0))
        self._chase_q = queue.Queue()
        self._chase = world.spawn_actor(bp, tf, attach_to=hero)
        self._chase.listen(lambda im: self._chase_q.put((im.frame, np.frombuffer(im.raw_data, np.uint8).reshape(im.height, im.width, 4).copy())))

    def _chase_take(self):
        """Newest chase image delivered so far (frame, BGRA) or None; older ones are dropped."""
        got = None
        while True:
            try:
                got = self._chase_q.get_nowait()
            except Exception:
                return got

    # ------------------------------------------------------------------ per tick
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        t0 = time.perf_counter()
        control = super().run_step(input_data, timestamp, sensors, plant)    # PDM-Lite on the true state: the label, always
        self._t["expert"] += time.perf_counter() - t0
        applied = self._drive(control, input_data)
        self._record(input_data, control, applied)
        self._check_stop()
        return applied

    def _check_stop(self):
        """Stuck (ego below 0.1 m/s for stuck_s: PDM-Lite waiting on a scenario that never clears, seen at 350-500 s in the 10-route
        stage) or max_sim_s: end the route; the clip keeps what was recorded and meta.json says why."""
        sp = self.rows["speed"]
        self._still = self._still + 1 if sp[-1] < 0.1 else 0
        if self._still * 0.05 >= self.cfg["stuck_s"]:
            STOP.update(flag=True, why="stuck")
        elif self._k * 0.05 >= self.cfg["max_sim_s"]:
            STOP.update(flag=True, why="max_sim")

    def _drive(self, expert_control, input_data):
        """The control the car gets. DAgger hook: a learned policy would act here (cfg driver 'policy'); PDM-Lite keeps labelling."""
        if self.cfg["driver"] != "expert":
            raise NotImplementedError("driver %r: only the expert drives in this collector version" % self.cfg["driver"])
        return expert_control

    def _record(self, input_data, ctl, applied):
        hero = self._vehicle
        world = self._world
        k = self._k
        if not self._static_done:
            self._write_static(hero, world)
        t1 = time.perf_counter()
        road, wide = input_data.get("OP_ROAD"), input_data.get("OP_WIDE")
        frame = int(GameTime.get_frame())
        self.rows["sensor_frame"].append([road[0] if road is not None else -1, wide[0] if wide is not None else -1])
        if road is not None and wide is not None:
            img2 = np.stack([self.packer(road[1], "road"), self.packer(wide[1], "wide")])
            t2 = time.perf_counter()
            self.enc.write(F.pair_to_yuv(img2))
            self.rows["vid"].append(self.enc.n - 1)
            self._t["pack"] += t2 - t1
            self._t["enc"] += time.perf_counter() - t2
        else:
            self.rows["vid"].append(-1)
        ch = self._chase_take() if self._chase is not None else None
        if ch is not None:
            self.chase_enc.write(ch[1])
            self.rows["chase"].append(self.chase_enc.n - 1)
        else:
            self.rows["chase"].append(-1)
        t3 = time.perf_counter()
        tf, v, a, w = hero.get_transform(), hero.get_velocity(), hero.get_acceleration(), hero.get_angular_velocity()
        R = self.rows
        R["t"].append(float(GameTime.get_time()))
        R["frame"].append(frame)
        R["loc"].append(_xyz(tf.location))
        R["rot"].append([tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll])
        R["vel"].append(_xyz(v))
        R["acc"].append(_xyz(a))
        R["angvel"].append(_xyz(w))
        R["speed"].append(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))
        try:
            R["wheel"].append(hero.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel))
        except Exception:
            R["wheel"].append(float("nan"))
        R["ctl_applied"].append([applied.steer, applied.throttle, applied.brake])
        R["ctl_expert"].append([ctl.steer, ctl.throttle, ctl.brake])
        R["driver"].append(0 if self.cfg["driver"] == "expert" else 1)
        R["target_speed"].append(float(getattr(self, "target_speed", float("nan"))))
        R["flags"].append([bool(getattr(self, "junction", False)), bool(getattr(self, "stop_sign_close", False)),
                           bool(getattr(self, "walker_close", False))])
        ahead = np.full((self.cfg["route_ahead"], 2), np.nan, np.float32)
        rr = getattr(self, "remaining_route", None)
        if rr is not None and len(rr):
            rr = np.asarray(rr)[: self.cfg["route_ahead"], :2]
            ahead[: len(rr)] = rr
        R["ahead"].append(ahead)
        tl = hero.get_traffic_light()
        R["tl"].append([tl.id if tl is not None else -1, int(hero.get_traffic_light_state()) if tl is not None else -1])
        R["speed_limit"].append(float(hero.get_speed_limit()))
        self._log_actors(world, tf.location, k)
        self._log_scen(k)
        self._t["log"] += time.perf_counter() - t3
        self._k += 1

    def _log_actors(self, world, ego_loc, k):
        r2 = self.cfg["actor_radius"] ** 2
        acts = world.get_actors()
        groups = [acts.filter("vehicle.*"), acts.filter("walker.pedestrian.*")]
        if k % self.cfg["prop_every"] == 0:
            groups.append(acts.filter("static.prop.*"))
        for g in groups:
            for x in g:
                if x.id == self._vehicle.id:
                    continue
                tf = x.get_transform()
                d = tf.location - ego_loc
                if d.x * d.x + d.y * d.y > r2:
                    continue
                v = x.get_velocity()
                self.actor_rows.append((k, x.id, tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw, v.x, v.y, v.z))
                if x.id not in self.kinds:
                    bb = x.bounding_box
                    self.kinds[x.id] = {"type_id": x.type_id, "role": x.attributes.get("role_name", ""),
                                        "extent": _xyz(bb.extent), "offset": _xyz(bb.location)}
        lr2 = self.cfg["light_radius"] ** 2
        for tid, loc in self._light_locs.items():
            d = loc - ego_loc
            if d.x * d.x + d.y * d.y <= lr2:
                self.light_rows.append((k, tid, int(self._light_actors[tid].get_state())))

    def _log_scen(self, k):
        reg = getattr(CarlaDataProvider, "active_scenarios", None)
        if reg is None:
            return
        def enc(x):
            if hasattr(x, "id") and hasattr(x, "type_id"):
                return int(x.id)
            if isinstance(x, (bool, str)) or x is None:
                return x
            try:
                return round(float(x), 3)
            except (TypeError, ValueError):
                return str(x)
        cur = [[typ] + [enc(x) for x in data] for typ, data in reg]
        if cur != self._last_scen:
            self.scen.write(json.dumps({"tick": k, "active": cur}) + "\n")
            self._last_scen = cur

    def _write_static(self, hero, world):
        self._static_done = True
        if self.chase_enc is not None:
            self._chase_start(world, hero)
        plan = getattr(self, "org_dense_route_world_coord", None) or []
        pts = np.array([[tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw] for tf, _ in plan], np.float64).reshape(-1, 4)
        opt = np.array([ROUTE_OPT.get(getattr(o, "name", str(o)).upper(), int(getattr(o, "value", -1))) for _, o in plan], np.int8)
        np.savez(self.out / "route.npz", xyzyaw=pts, option=opt)
        lo, hi = (pts[:, :2].min(0) - 150, pts[:, :2].max(0) + 150) if len(pts) else (np.full(2, -1e9), np.full(2, 1e9))
        acts = world.get_actors()
        lights, self._light_locs, self._light_actors = [], {}, {}
        for tl in acts.filter("traffic.traffic_light*"):
            loc = tl.get_transform().location
            if not (lo[0] <= loc.x <= hi[0] and lo[1] <= loc.y <= hi[1]):
                continue
            tv = tl.trigger_volume
            tvw = tl.get_transform().transform(tv.location)
            lights.append({"id": tl.id, "loc": _xyz(loc), "yaw": tl.get_transform().rotation.yaw, "trigger": _xyz(tvw),
                           "trigger_extent": _xyz(tv.extent),
                           "stop_wps": [_xyz(w.transform.location) + [w.transform.rotation.yaw] for w in tl.get_stop_waypoints()],
                           "group": [x.id for x in tl.get_group_traffic_lights()]})
            self._light_locs[tl.id], self._light_actors[tl.id] = loc, tl
        stops = []
        for s in acts.filter("traffic.stop"):
            tfm = s.get_transform()
            if lo[0] <= tfm.location.x <= hi[0] and lo[1] <= tfm.location.y <= hi[1]:
                stops.append({"id": s.id, "loc": _xyz(tfm.location), "yaw": tfm.rotation.yaw,
                              "trigger": _xyz(tfm.transform(s.trigger_volume.location)), "trigger_extent": _xyz(s.trigger_volume.extent)})
        (self.out / "lights.json").write_text(json.dumps({"lights": lights, "stops": stops}))
        bb = hero.bounding_box
        wmap = world.get_map()
        w = world.get_weather()
        weather = {k: getattr(w, k) for k in ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "sun_azimuth_angle",
                                              "sun_altitude_angle", "fog_density", "fog_distance", "wetness")}
        meta = {"town": wmap.name.split("/")[-1], "route_id": Path(os.environ.get("B2D_ATTEMPT_OUT", "x/x")).resolve().parent.name, "weather": weather,
                "hero": {"type_id": hero.type_id, "extent": _xyz(bb.extent), "offset": _xyz(bb.location), "rear_axle_x": rigs.REAR_AXLE_X},
                "rig": {"mount_rear_axle": list(self.cfg["mount"]), "sensors": self.cam_specs, "cam_wh": list(rigs.OP_CAMERA_WH),
                        "focal": rigs.OP_FOCAL, "model_K": {n: F.MODEL_K[n].tolist() for n in F.MODEL_K},
                        "calib_rpy": [0.0, 0.0, 0.0], "packing": "zeroshot_policy_server.OpenpilotModel.pack (BT.601 limited, NN warp)",
                        "video": {"layout": "512x512 yuv420p: Y rows 0-255 road, 256-511 wide", "crf": self.cfg["crf"]}},
                "cfg": self.cfg, "delta_s": 0.05, "start_frame": int(GameTime.get_frame())}
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    # ------------------------------------------------------------------ end of route
    def destroy(self, results=None):
        try:
            self._finish(results)
        finally:
            super().destroy(results)

    def _finish(self, results):
        if not hasattr(self, "rows") or getattr(self, "_finished", False):
            return
        self._finished = True
        if self._chase is not None:                      # stop before destroy (docs/carla.md: a callback on a destroyed sensor aborts)
            try:
                self._chase.stop()
                self._chase.destroy()
            except Exception:
                pass
            self._chase = None
        n_vid = self.enc.close()
        n_chase = self.chase_enc.close() if self.chase_enc is not None else 0
        self.scen.close()
        R = self.rows
        ego = {k: np.asarray(v, np.float32 if k not in ("t", "loc", "frame", "vid", "chase", "tl", "driver", "sensor_frame") else None) for k, v in R.items()}
        ego["loc"], ego["t"] = np.asarray(R["loc"], np.float64), np.asarray(R["t"], np.float64)
        ego["sensor_frame"] = np.asarray(R["sensor_frame"], np.int64).reshape(-1, 2)
        for k in ("frame", "vid", "chase", "driver"):
            ego[k] = np.asarray(R[k], np.int64)
        ego["tl"] = np.asarray(R["tl"], np.int64).reshape(-1, 2)
        np.savez_compressed(self.out / "ego.npz", **ego)
        a = np.asarray(self.actor_rows, np.float64).reshape(-1, 9)
        np.savez_compressed(self.out / "actors.npz", tick=a[:, 0].astype(np.int32), id=a[:, 1].astype(np.int64), xyz=a[:, 2:5].astype(np.float32),
                            yaw=a[:, 5].astype(np.float32), vel=a[:, 6:9].astype(np.float32))
        lr = np.asarray(self.light_rows, np.int64).reshape(-1, 3)
        np.savez_compressed(self.out / "lights.npz", tick=lr[:, 0].astype(np.int32), id=lr[:, 1], state=lr[:, 2].astype(np.int8))
        (self.out / "kinds.json").write_text(json.dumps({str(k): v for k, v in self.kinds.items()}))
        n = max(1, self._k)
        wall = time.time() - self._t["wall0"]
        timing = {"ticks": self._k, "stop": STOP["why"] or "route_end", "frames": n_vid, "chase_frames": n_chase, "wall_s": round(wall, 1),
                  "sim_s": round(self._k * 0.05, 2), "rtf": round(self._k * 0.05 / max(wall, 1e-6), 3),
                  **{f"{k}_ms_per_tick": round(1e3 * v / n, 2) for k, v in self._t.items() if k != "wall0"},
                  "video_mb": round((self.out / "frames.mp4").stat().st_size / 1e6, 2)}
        mp = self.out / "meta.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        meta["timing"] = timing
        mp.write_text(json.dumps(meta, indent=1, default=str))
        if self._k > 0:                                  # an attempt that never ticked is not a clip
            (self.out / "DONE").write_text(json.dumps(timing))
