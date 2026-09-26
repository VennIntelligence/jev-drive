#!/usr/bin/env python
"""Top-10 T3 re-recorder (todos/2026-09-26-top10-intersection.md, [T3]): the P5 v1 BA worlds again, with BridgeDrive in
shadow and BLUE's own camera recorded for offline inference.

Loaded by the leaderboard through `scripts/b2d_run.py --agent scripts/top10_t3_agent.py --agent-config <json>
--python $DATA_DIR/envs/bridgedrive/bin/python` with PYTHONPATH / LEAD_PROJECT_ROOT at BridgeDrive's pinned lead
(a41d116). It IS BridgeDrive's `SensorAgent` (lead/inference/sensor_agent_bridgedrive.py), so the leaderboard attaches
the author's rig, which is sensor for sensor the TFv6 rig the P5 recorder carried (3 x 384^2 FOV 60 cameras at yaw
0 / +-54.5, 2 LiDAR, 4 radar, IMU, GNSS, speedometer). The car is driven by CARLA's BehaviorAgent(normal), set up
exactly as scripts/p5_pair_agent.py does, so the expert trajectory repeats the original recording tick for tick
(checked against it, `jevdrive.top10_t3 check-det`).

BridgeDrive shadow, as the P5 recorder runs TFv6: the author's full `run_step` on camera ticks (every 4th tick, the
5 Hz the exam reads), the author's `BaseAgent.tick` on the ticks between (GPS / Kalman filter, route planner, LiDAR and
radar queues), control discarded. Two adapter choices: the Kalman filter is fed the control that was actually
applied (the expert's, one tick earlier) instead of the model's own, and torch is reseeded before every forward so
both worlds of a pair draw the same sampling noise. Config "shadow": "ref20" runs the full `run_step` on every tick
instead (the equivalence reference).

BLUE (SimLingo + gate) runs offline (scripts/top10_t3_blue.py, envs/blue): this recorder adds BLUE's camera spec
verbatim (rgb_0, 1024 x 512, FOV 110, x = -1.5, z = 2.0; its IMU / GNSS / speedometer specs equal the author rig's)
and writes what BLUE's `tick` reads: the raw image losslessly on camera ticks, GNSS / IMU / speed every tick, and the
route plan as the leaderboard hands it over.

Outputs in B2D_ATTEMPT_OUT:
  meta.json          rig, route, config, model
  plan.json          global plan: GPS (lat, lon, z) and world transforms, with RoadOptions
  pose.jsonl         per tick: hero truth pose / velocity / acceleration, expert control (as the P5 recorder)
  frames.jsonl       per camera tick: tick, t, scenario trigger flags, BLUE image path
  bridgedrive.jsonl  per camera tick: target-speed distribution and scalar, 8 waypoints, route, control, creep state
  blue_inputs.jsonl  per tick: GNSS, IMU, speedometer
  blue/<k>.png       BLUE's camera (BGR) on camera ticks
  t3_summary.json    ticks, stop reason, per-part timings

Config "need": JSON {route id: last tick index k the exam references}; the route stops two ticks after it.
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
import torch

from leaderboard.autoagents.autonomous_agent import Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

import p4_carla_agent as p4                      # early-stop flag (patches ScenarioManager), Waymo camera model
import lead.inference.config_closed_loop as _cc
import lead.training.config_training as _ct
from lead.common.base_agent import BaseAgent
from lead.inference.sensor_agent_bridgedrive import SensorAgent

DEFAULT = dict(p4.DEFAULT, max_sim_s=50.0, stuck_s=30.0, after_trigger_s=20.0, model_dir="", need="", rig="lean",
               shadow="shadow5", blue_png=1, save_threads=2, seed=0)
BLUE_CAM = {"type": "sensor.camera.rgb", "x": -1.5, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "width": 1024, "height": 512, "fov": 110, "id": "rgb_0"}      # team_code/config_simlingo.py camera 0


def _author_shims():
    """The two runtime shims of the BridgeDrive smoke (scripts/top10_smoke/bridgedrive_smoke.py), no source edits:
    TrainingConfig.gpu_name raises outside the author's GPU list (it only feeds training-time switches), and the
    released ClosedLoopConfig reads self.debug_mode without defining it."""
    _cc.ClosedLoopConfig.debug_mode = False
    get = _ct.TrainingConfig.gpu_name.fget

    def safe(self):
        try:
            return get(self)
        except Exception:
            return ""
    _ct.TrainingConfig.gpu_name = property(safe)


_author_shims()


def get_entry_point():
    return "T3Agent"


def _np(v):
    return np.round(v[0].detach().float().cpu().numpy(), 4).tolist()


class T3Agent(SensorAgent):
    def setup(self, path_to_conf_file, _=None, __=None):
        cfg = dict(DEFAULT)
        with open(path_to_conf_file.split("+")[0]) as fh:
            cfg.update(json.load(fh))
        self.cfg = cfg
        self.out = Path(os.environ["B2D_ATTEMPT_OUT"])
        self.rid = os.environ.get("BENCHMARK_ROUTE_ID", "")
        need = json.loads(Path(cfg["need"]).read_text()) if cfg["need"] else {}
        self.need_k = need.get(self.rid)
        with mock.patch("shutil.which", return_value="/bin/true"):     # setup demands ffmpeg for videos we never make
            super().setup(cfg["model_dir"])
        forward = self.closed_loop_inference.forward

        def capture(*args, **kwargs):
            torch.manual_seed(cfg["seed"])
            t0 = time.perf_counter()
            self._pred = forward(*args, **kwargs)
            self._t["bd_forward"].append(time.perf_counter() - t0)
            return self._pred

        self.closed_loop_inference.forward = capture
        self.track = Track.SENSORS
        (self.out / "blue").mkdir(parents=True, exist_ok=True)
        self._tick, self._ba, self._still_since, self._pred, self._t_trig = 0, None, None, None, None
        self._inited, self._prev_ctrl, self._pool = False, None, None
        self._t = {k: [] for k in ("tick", "bd", "bd_base", "bd_forward", "expert", "blue_save")}
        self._pose = open(self.out / "pose.jsonl", "w")
        self._frames = open(self.out / "frames.jsonl", "w")
        self._bd = open(self.out / "bridgedrive.jsonl", "w")
        self._blue = open(self.out / "blue_inputs.jsonl", "w")
        self._errors = 0

    # The Bench2Drive agent base dumps a growing metric_info.json every tick when this exists; hide it.
    @property
    def get_metric_info(self):
        raise AttributeError("disabled for the recorder")

    def sensors(self):
        rig = super().sensors() + [dict(BLUE_CAM)]
        if self.cfg["rig"] == "p5":                   # the P5 recorder's three Waymo cameras in front, not saved
            c = self.cfg
            fov = math.degrees(2 * math.atan(c["render_w"] / 2.0 / c["f"]))
            rig = [{"type": "sensor.camera.rgb", "id": name, "x": x + p4.REAR_AXLE_X, "y": -y, "z": z - c["origin_z"],
                    "roll": 0.0, "pitch": 0.0, "yaw": -yaw, "width": c["render_w"], "height": c["render_h"], "fov": fov}
                   for name, x, y, z, yaw in p4.WAYMO_CAMS] + rig
        return rig

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._dense = list(global_plan_world_coord)
        self._plan = {"gps": [[g["lat"], g["lon"], g["z"], int(o.value)] for g, o in global_plan_gps],
                      "world": [[t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw,
                                 t.rotation.roll, int(o.value)] for t, o in global_plan_world_coord]}

    def _init_world(self):
        """The expert exactly as scripts/p5_pair_agent.py builds it."""
        from agents.navigation.behavior_agent import BehaviorAgent
        hero = CarlaDataProvider.get_hero_actor()
        cmap = CarlaDataProvider.get_map()
        self._hero = hero
        self._ba = BehaviorAgent(hero, behavior=self.cfg["behavior"], map_inst=cmap, grp_inst=object())
        self._ba.set_global_plan([(cmap.get_waypoint(t.location), o) for t, o in self._dense],
                                 stop_waypoint_creation=True, clean_queue=True)
        self._seg = None
        if self.cfg["rig"] == "p5":                   # and its visibility camera, spawned on the first tick, unused
            c, (_, fx, fy, fz, _) = self.cfg, p4.WAYMO_CAMS[0]
            bp = CarlaDataProvider.get_world().get_blueprint_library().find("sensor.camera.instance_segmentation")
            fov = math.degrees(2 * math.atan(c["render_w"] / 2.0 / c["f"]))
            for key, val in (("image_size_x", c["render_w"] // 2), ("image_size_y", c["render_h"] // 2), ("fov", fov)):
                bp.set_attribute(key, str(val))
            self._seg = CarlaDataProvider.get_world().spawn_actor(
                bp, carla.Transform(carla.Location(x=fx + p4.REAR_AXLE_X, y=-fy, z=fz - c["origin_z"])), attach_to=hero)
        (self.out / "plan.json").write_text(json.dumps(self._plan))
        (self.out / "meta.json").write_text(json.dumps(
            {"town": cmap.name, "vehicle": hero.type_id, "hero_id": hero.id, "route_id": self.rid,
             "need_k": self.need_k, "cams": self.sensors(), "config": self.cfg,
             "driver": "carla.agents.navigation.BehaviorAgent(%s)" % self.cfg["behavior"],
             "bridgedrive": {"model_dir": self.cfg["model_dir"], "lead": os.environ.get("LEAD_PROJECT_ROOT", ""),
                             "closed_loop_config": os.environ.get("LEAD_CLOSED_LOOP_CONFIG", ""),
                             "training_config": os.environ.get("LEAD_TRAINING_CONFIG", ""),
                             "target_speed_classes": list(self.training_config.target_speed_classes)},
             "bench2drive_root": os.environ.get("BENCH2DRIVE_ROOT", "")}, indent=1))

    # ------------------------------------------------------------------ per tick

    def _shadow(self, input_data, t, frame, k, full):
        """BridgeDrive on its own sensors; the control it returns is discarded. The Kalman filter sees the control
        the expert applied on the previous tick."""
        self._pred = None
        if self._prev_ctrl is not None:
            self.control = self._prev_ctrl
        try:
            if not full and self.initialized:
                BaseAgent.tick(self, dict(input_data), use_kalman_filter=self.training_config.use_kalman_filter_for_gps)
                return
            ctrl = SensorAgent.run_step(self, dict(input_data), t)
        except Exception as e:                    # recorded, never allowed to stop the expert's drive
            self._errors += 1
            if self._errors <= 3:
                print("bridgedrive shadow failed at frame %d: %r" % (frame, e), flush=True)
            return
        p = self._pred
        if p is None:                              # the author's first tick only initialises
            return
        fm = self.force_move_post_processor
        rec = {"frame": frame, "k": k, "tick": self._tick, "step": self.step,
               "steer": float(ctrl.steer), "throttle": float(ctrl.throttle), "brake": float(ctrl.brake),
               "stuck_detector": int(fm.stuck_detector), "force_move": int(fm.force_move)}
        for key in ("pred_target_speed_distribution", "pred_target_speed_scalar", "pred_future_waypoints", "pred_route"):
            v = getattr(p, key, None)
            if v is not None:
                rec[key] = _np(v)
        self._bd.write(json.dumps(rec) + "\n")

    def _save_blue(self, k, input_data):
        rel = "blue/%05d.png" % k
        img = np.ascontiguousarray(input_data["rgb_0"][1][:, :, :3])
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(int(self.cfg["save_threads"]))
        self._pool.submit(cv2.imwrite, str(self.out / rel), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return rel

    def __call__(self):
        t_tick = time.perf_counter()
        self._tick += 1
        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        frame, t = GameTime.get_frame(), GameTime.get_time()
        k = int(round(t / p4.DELTA))
        if not self._inited:
            self._init_world()
            self._inited = True
        cam_tick = (self._tick - 1) % p4.CAM_TICKS == 0
        t0 = time.perf_counter()
        self._shadow(input_data, t, frame, k, cam_tick or self.cfg["shadow"] == "ref20")
        self._t["bd" if cam_tick else "bd_base"].append(time.perf_counter() - t0)
        imu, gps = input_data["imu"][1], input_data["gps"][1]
        self._blue.write(json.dumps({"k": k, "tick": self._tick, "frame": frame, "t": round(t, 4),
                                     "gps": [float(x) for x in gps], "imu": [float(x) for x in imu],
                                     "speed": float(input_data["speed"][1]["speed"])}) + "\n")
        if cam_tick:
            t0 = time.perf_counter()
            img = self._save_blue(k, input_data) if self.cfg["blue_png"] else None
            self._t["blue_save"].append(time.perf_counter() - t0)
            bb = py_trees.blackboard.Blackboard()
            trig = [bool(bb.get("ScenarioRouteNumber%d" % i)) for i in range(2)]
            if trig[0] and self._t_trig is None:
                self._t_trig = t
            self._frames.write(json.dumps({"frame": frame, "tick": self._tick, "k": k, "t": round(t, 4),
                                           "trig": trig, "blue": img}) + "\n")
        t0 = time.perf_counter()
        control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) if self._ba.done() else self._ba.run_step()
        control.manual_gear_shift = False
        self._prev_ctrl = carla.VehicleControl(throttle=control.throttle, steer=control.steer, brake=control.brake)
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
        if self.need_k is not None and k >= self.need_k + 2:
            p4.STOP.update(flag=True, why="need_k")
        elif t > self.cfg["max_sim_s"]:
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
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for fh in (self._pose, self._frames, self._bd, self._blue):
            fh.close()
        if getattr(self, "_seg", None) is not None:
            self._seg.destroy()
        ms = {k: round(1e3 * float(np.mean(v)), 2) for k, v in self._t.items() if v}
        (self.out / "t3_summary.json").write_text(json.dumps(
            {"ticks": self._tick, "stop": p4.STOP["why"] or "route_end", "t_trigger": self._t_trig,
             "need_k": self.need_k, "bd_errors": self._errors, "shadow": self.cfg["shadow"], "ms_mean": ms,
             "ms_p95": {k: round(1e3 * float(np.percentile(v, 95)), 2) for k, v in self._t.items() if v}}))


def smoke_check(capture: str, model_dir: str):
    """The recorder's model path (the author's SensorAgent.setup under this file's env config and shims, forward with
    the reseed) on the BridgeDrive smoke's captured network inputs, against the predictions the author's agent made
    live on those ticks (scripts/top10_smoke/bridgedrive_smoke.py)."""
    agent = SensorAgent.__new__(SensorAgent)
    with mock.patch("shutil.which", return_value="/bin/true"):
        agent.setup(model_dir)
    out = []
    for f in sorted(Path(capture).glob("frame_*.pth")):
        rec = torch.load(f, weights_only=False)
        data = {k: (v.to(agent.device, torch.float32) if isinstance(v, torch.Tensor) else v) for k, v in rec["data"].items()}
        torch.manual_seed(0)
        with torch.inference_mode():
            p = agent.closed_loop_inference.forward(data)
        out.append({"frame": f.name, **{k: float((getattr(p, k).detach().float().cpu() - rec["live"][k].float()).abs().max())
                                         for k in ("pred_future_waypoints", "pred_route", "pred_target_speed_scalar",
                                                   "pred_target_speed_distribution")}})
    print(json.dumps({"frames": len(out), "max_abs": {k: max(o[k] for o in out) for k in out[0] if k != "frame"}}, indent=1))


if __name__ == "__main__":
    import sys
    smoke_check(sys.argv[1], sys.argv[2])
