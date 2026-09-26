#!/usr/bin/env python
"""Night queue 3, lane A recorder (todos/2026-09-26-night-queue-3.md, Q1 v0 re-record and Q3 P6 v1; [A] entries).

PDM-Lite drives a P6 world exactly as scripts/p5_pair_agent.py (P6 v0) drove it: same inner agent, same numpy stream,
same hooks (scripts/b2d_hooks.py p6_world). What changes is who watches:

  shadows   TFv6 and / or BridgeDrive, each in a process of its own (scripts/nq3_shadow.py, its own venv and LEAD
            tree), fed every tick's author-rig data down a pipe. The drive never waits for a model; the readouts are
            the in-process recorders' (P5 / P6 v0 for TFv6, top-10 T3 for BridgeDrive), tick for tick.
  blue      BLUE / SimLingo's camera (rgb_0, 1024 x 512, FOV 110, x = -1.5, z = 2.0), lossless PNG on camera ticks, plus
            GNSS / IMU / speed every tick and the global plan: what scripts/top10_t3_blue.py replays offline.
  rig p5    the three Waymo-calibrated cameras, the visibility segmentation, the actor / light / registry record of
            P6 v0 (openpilot and the NAVSIM-family examinees read these). rig "lean" (the v0 re-record) keeps those
            actors in the world, spawned in the same order, but they never render and nothing is saved.

Config (JSON, --agent-config): p5_pair_agent's keys (driver pdm_lite, record_props, pass_stop_s, ...) plus
  rig          "p5" | "lean"
  cam_period   the Waymo / BLUE cameras' sensor_tick in ticks (4 = render at 5 Hz, only on camera ticks; 1 = 20 Hz)
  shadows      [{model, python, lead, model_dir, env}]
  blue         1 = record BLUE's inputs
  need         JSON {world id: last tick k referenced}; the world stops two ticks after it (v0 re-record)
The shadow processes start when the leaderboard imports this module, before the world loads, so their imports overlap
the map load; they get the plan at setup().

Outputs in B2D_ATTEMPT_OUT: p5_pair_agent's (rig p5), tfv6.jsonl / bridgedrive.jsonl, plan.json, blue_inputs.jsonl,
blue/<k>.png, frames.jsonl (+ "blue", "k"), nq3_summary.json (timings, shadow summaries, camera misses), shadow_*.log.
Python 3.10, envs/p5v1-pdm (PDM-Lite needs numpy 1.23), PYTHONPATH at LEAD cvpr2026 (the base class).
"""
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import py_trees

from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

import p4_carla_agent as p4
import p5_pair_agent as p5

HERE = Path(__file__).resolve().parent
BLUE_CAM = {"type": "sensor.camera.rgb", "x": -1.5, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "width": 1024, "height": 512, "fov": 110, "id": "rgb_0"}      # team_code/config_simlingo.py camera 0
DEFAULT = dict(p5.DEFAULT, rig="p5", cam_period=1, shadows=[], blue=1, need="", blue_threads=2, drain_s=1500.0)
NEVER = 1.0e4                                   # sensor_tick of a camera that is spawned but never needed


def get_entry_point():
    return "NQ3Recorder"


# ---------------------------------------------------------------- shadow processes (started at import)

def _send(p, msg):
    b = pickle.dumps(msg, protocol=5)
    p.stdin.write(struct.pack("<Q", len(b)) + b)
    p.stdin.flush()


def _recv(p):
    head = p.stdout.read(8)
    if len(head) < 8:
        raise EOFError("shadow process closed its pipe")
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(p.stdout.read(n))


def _config_path():
    a = sys.argv
    return a[a.index("--agent-config") + 1] if "--agent-config" in a else None


def _start_shadows():
    cfgp = _config_path()
    if not cfgp:
        return []
    cfg = json.loads(Path(cfgp.split("+")[0]).read_text())
    out = Path(os.environ["B2D_ATTEMPT_OUT"])
    # the Bench2Drive / CARLA PythonAPI entries b2d_route.py put on sys.path, without this process's LEAD tree
    b2d = [p for p in sys.path if p.endswith(("scenario_runner", "leaderboard", os.path.join("PythonAPI", "carla")))]
    procs = []
    for s in cfg.get("shadows", []):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LEAD_")}
        env.update(s.get("env", {}), LEAD_PROJECT_ROOT=s["lead"], PYTHONPATH=":".join([s["lead"]] + b2d + [str(HERE)]),
                   PYTHONUNBUFFERED="1")
        log = open(out / ("shadow_%s.log" % s["model"]), "wb")
        p = subprocess.Popen([s["python"], str(HERE / "nq3_shadow.py"), "--model", s["model"]], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, env=env, cwd=os.getcwd())
        procs.append({"spec": s, "proc": p, "alive": True, "t_spawn": time.time()})
    return procs


try:
    SHADOWS = _start_shadows()
except Exception as e:                         # never break the route import; setup() reports it
    SHADOWS, SHADOW_ERR = [], repr(e)
else:
    SHADOW_ERR = None
T_IMPORT = time.time()


class NQ3Recorder(p5.P5PairAgent):
    def setup(self, path_to_conf_file, _=None, __=None):
        t0 = time.time()
        cfg = dict(DEFAULT)
        with open(path_to_conf_file.split("+")[0]) as fh:
            cfg.update(json.load(fh))
        cfg["tfv6_model_dir"] = ""                 # TFv6 never runs in this process
        with open(os.environ["B2D_ATTEMPT_OUT"] + "/nq3_agent.json", "w") as fh:
            json.dump(cfg, fh)
        p5.P5PairAgent.setup(self, os.environ["B2D_ATTEMPT_OUT"] + "/nq3_agent.json")
        self.cfg = cfg
        self._tf.close()                           # p5's in-process TFv6 file; the TFv6 shadow writes its own
        os.remove(self.out / "tfv6.jsonl")
        self._tf = open(os.devnull, "w")
        self.rid = os.environ.get("BENCHMARK_ROUTE_ID", "")
        need = json.loads(Path(cfg["need"]).read_text()) if cfg["need"] else {}
        self.need_k = need.get(self.rid)
        self.track = Track.SENSORS
        if SHADOW_ERR:
            raise RuntimeError("shadow start failed: %s" % SHADOW_ERR)
        # the town name the shadows' model input carries; the map is loaded by now
        town = CarlaDataProvider.get_map().name
        init = {"plan_gps": [[g["lat"], g["lon"], g["z"], int(o.value)] for g, o in self._plan_gps],
                "plan_world": [[t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw,
                                t.rotation.roll, int(o.value)] for t, o in self._dense],
                "town": town, "out": str(self.out), "seed": 0}
        (self.out / "plan.json").write_text(json.dumps({"gps": init["plan_gps"], "world": init["plan_world"]}))
        for s in SHADOWS:
            _send(s["proc"], ("init", dict(init, model_dir=s["spec"]["model_dir"])))
        rigs = []
        for s in SHADOWS:
            tag, info = _recv(s["proc"])
            assert tag == "ready", tag
            s["ready"] = info
            rigs.append(info["sensors"])
        self._rig = rigs[0] if rigs else []
        for r in rigs[1:]:
            if [x["id"] for x in r] != [x["id"] for x in self._rig]:
                raise RuntimeError("the shadow models ask for different rigs: %s" % [[x["id"] for x in r] for r in rigs])
        self._rig_ids = [x["id"] for x in self._rig]
        self._t.update({k: [] for k in ("send", "blue_save", "read")})
        self._blue_fh = open(self.out / "blue_inputs.jsonl", "w") if cfg["blue"] else None
        if cfg["blue"]:
            (self.out / "blue").mkdir(exist_ok=True)
        self._blue_pool, self._prev_ctrl, self._cam_miss = None, None, {}
        self._setup_s = {"import_to_setup_s": round(t0 - T_IMPORT, 1), "setup_s": round(time.time() - t0, 1),
                         "shadows": [s.get("ready") for s in SHADOWS]}

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        from leaderboard.autoagents.autonomous_agent import AutonomousAgent
        AutonomousAgent.set_global_plan(self, global_plan_gps, global_plan_world_coord)   # no LEAD model here
        self._dense = list(global_plan_world_coord)      # the leaderboard calls this before setup()
        self._plan_gps = list(global_plan_gps)

    # ------------------------------------------------------------------ sensors

    def sensors(self):
        c = self.cfg
        period = int(c["cam_period"])
        # A shade under period x 0.05 s: at exactly 0.2 s the float sum of four 0.05 s steps sometimes falls short and the
        # camera slips a tick (smoke: 2 of 42 camera ticks); 0.1999 s fires on every 4th tick for > 100 s of simulation.
        tick = p4.DELTA * period - 1e-4 if period > 1 else 0.0
        lean = c["rig"] == "lean"
        own = [{"type": "sensor.camera.rgb", "id": name, "x": x + p4.REAR_AXLE_X, "y": -y, "z": z - c["origin_z"],
                "roll": 0.0, "pitch": 0.0, "yaw": -yaw, "width": c["render_w"], "height": c["render_h"], "fov": self.fov,
                "sensor_tick": NEVER if lean else tick}
               for name, x, y, z, yaw in p4.WAYMO_CAMS]
        # The author rig as the leaderboard spawned it in P6 v0 / T3: its whitelist dropped every sensor_tick, and with
        # sensor_tick passed through now (B2D_SENSOR_TICK, for the decimated cameras) it must stay dropped here.
        rig = [{k: v for k, v in x.items() if k != "sensor_tick"} for x in self._rig]
        if not any(x["id"] == "imu" for x in rig):   # PDM-Lite's compass
            rig.append({"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0,
                        "yaw": 0.0, "id": "imu"})
        blue = [dict(BLUE_CAM, sensor_tick=tick)] if c["blue"] else []
        return own + rig + blue

    def _periods(self):
        """Ticks between deliveries per sensor id: 1 every tick, cam_period for the decimated cameras, 0 never waited."""
        if not hasattr(self, "_per"):
            per = {s["id"]: 1 for s in self.sensors()}
            for name, *_ in p4.WAYMO_CAMS:
                per[name] = 0 if self.cfg["rig"] == "lean" else int(self.cfg["cam_period"])
            if self.cfg["blue"]:
                per["rgb_0"] = int(self.cfg["cam_period"])
            self._per = per
        return self._per

    def _read(self, frame, cam_tick):
        """This frame's data from the sensors due on this tick. A decimated camera that has not delivered this frame
        within 3 s on a camera tick is counted as a miss (its image is absent) instead of failing the route."""
        si = self.sensor_interface
        per = self._periods()
        want = {k for k, p in per.items() if p == 1 or (p > 1 and cam_tick)}
        got, t_end = {}, None
        while not want <= got.keys():
            late = {k for k in want - got.keys() if per[k] > 1}
            timeout = si._queue_timeout
            if late and want - got.keys() == late:
                t_end = t_end or time.time() + 3.0
                timeout = max(0.01, t_end - time.time())
            try:
                tag, f, data = si._data_buffers.get(True, timeout)
            except Exception:
                if late and want - got.keys() == late:
                    for k in late:
                        self._cam_miss[k] = self._cam_miss.get(k, 0) + 1
                    break
                raise SensorReceivedNoData("A sensor took too long to send their data")
            if f == frame:
                got[tag] = (f, data)
        return got

    # ------------------------------------------------------------------ one-off state

    def _init_world(self):
        """p5_pair_agent's, with the visibility camera's rendering off in rig lean (still spawned here, so every actor
        spawned later gets the id it had in P6 v0)."""
        if self.cfg["rig"] != "lean":
            return p5.P5PairAgent._init_world(self)
        lib = CarlaDataProvider.get_world().get_blueprint_library()
        find = type(lib).find

        class Lib:                                   # the segmentation blueprint with sensor_tick = NEVER
            def find(self_, key):
                bp = find(lib, key)
                if key == "sensor.camera.instance_segmentation":
                    bp.set_attribute("sensor_tick", str(NEVER))
                return bp

        world = CarlaDataProvider.get_world()
        orig = type(world).get_blueprint_library
        try:
            type(world).get_blueprint_library = lambda self_: Lib()
            p5.P5PairAgent._init_world(self)
        finally:
            type(world).get_blueprint_library = orig
        self._seg.stop()

    # ------------------------------------------------------------------ per tick

    def _to_shadows(self, input_data, t, frame, k, cam_tick):
        if not SHADOWS:
            return
        data = {i: input_data[i] for i in self._rig_ids if i in input_data}
        msg = ("tick", {"tick": self._tick, "frame": frame, "k": k, "t": t, "cam": cam_tick, "data": data,
                        "prev_ctrl": self._prev_ctrl})
        b = pickle.dumps(msg, protocol=5)
        head = struct.pack("<Q", len(b))
        for s in SHADOWS:
            if not s["alive"]:
                continue
            try:
                s["proc"].stdin.write(head + b)
                s["proc"].stdin.flush()
            except (BrokenPipeError, OSError) as e:
                s["alive"] = False
                print("shadow %s died at frame %d: %r" % (s["spec"]["model"], frame, e), flush=True)

    def _save_blue(self, k, input_data):
        if "rgb_0" not in input_data:
            return None
        rel = "blue/%05d.png" % k
        img = np.ascontiguousarray(input_data["rgb_0"][1][:, :, :3])
        if self._blue_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._blue_pool = ThreadPoolExecutor(int(self.cfg["blue_threads"]))
        self._blue_pool.submit(cv2.imwrite, str(self.out / rel), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return rel

    def __call__(self):
        t_tick = time.perf_counter()
        self._tick += 1
        frame, t = GameTime.get_frame(), GameTime.get_time()
        k = int(round(t / p4.DELTA))
        cam_tick = (self._tick - 1) % p4.CAM_TICKS == 0
        t0 = time.perf_counter()
        input_data = self._read(frame, cam_tick)
        self._t["read"].append(time.perf_counter() - t0)
        if not self._inited:
            self._init_world()
            self._inited = True
        t0 = time.perf_counter()
        self._to_shadows(input_data, t, frame, k, cam_tick)
        self._t["send"].append(time.perf_counter() - t0)
        lean = self.cfg["rig"] == "lean"
        if not lean:
            t0 = time.perf_counter()
            rows = self._snapshot(frame)
            self._t["snapshot"].append(time.perf_counter() - t0)
        if cam_tick:
            rec = {"frame": frame, "tick": self._tick, "k": k, "t": round(t, 4)}
            if not lean:
                t0 = time.perf_counter()
                if all(n in input_data for n, *_ in p4.WAYMO_CAMS):
                    files, std = self._save(frame, input_data)
                else:
                    files, std = {}, []
                self._t["save"].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                px = self._visibility(frame, rows)
                self._t["visibility"].append(time.perf_counter() - t0)
                rec.update(files=files, std=std, lights=self._light_states(), px=px)
            if self.cfg["blue"]:
                t0 = time.perf_counter()
                rec["blue"] = self._save_blue(k, input_data)
                self._t["blue_save"].append(time.perf_counter() - t0)
            bb = py_trees.blackboard.Blackboard()
            rec["trig"] = [bool(bb.get("ScenarioRouteNumber%d" % i)) for i in range(2)]
            if rec["trig"][0] and self._t_trig is None:
                self._t_trig = t
            if self._pdm is not None:
                rec["reg"] = self._registry()
            if self.cfg["pass_stop_s"] > 0 and self._passed(t):
                p4.STOP.update(flag=True, why="passed")
            self._frames.write(json.dumps(rec) + "\n")
        if self._blue_fh is not None:
            self._blue_fh.write(json.dumps({"k": k, "tick": self._tick, "frame": frame, "t": round(t, 4),
                                            "gps": [float(x) for x in input_data["gps"][1]],
                                            "imu": [float(x) for x in input_data["imu"][1]],
                                            "speed": float(input_data["speed"][1]["speed"])}) + "\n")
        t0 = time.perf_counter()
        if self._pdm is not None:
            control = self._pdm_step(input_data, t)
        else:
            control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0) if self._ba.done() else self._ba.run_step()
        control.manual_gear_shift = False
        self._prev_ctrl = (float(control.throttle), float(control.steer), float(control.brake))
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

    # ------------------------------------------------------------------ end

    def _drain(self):
        """Tell every shadow the route is over and wait for it to catch up; keep b2d_run's watchdog fed meanwhile."""
        out, beat = [], self.out / "heartbeat.json"
        for s in SHADOWS:
            if s["alive"]:
                try:
                    _send(s["proc"], ("end",))
                except (BrokenPipeError, OSError):
                    s["alive"] = False
        t0 = time.time()
        for s in SHADOWS:
            p, res = s["proc"], {"model": s["spec"]["model"], "alive": s["alive"]}
            if s["alive"]:
                import threading
                box = {}
                th = threading.Thread(target=lambda: box.update(msg=_recv(p)), daemon=True)
                th.start()
                i = 0
                while th.is_alive() and time.time() - t0 < self.cfg["drain_s"]:
                    th.join(10.0)
                    i += 1
                    tmp = str(beat) + ".tmp"
                    Path(tmp).write_text(json.dumps({"t": time.time(), "ticks": self._tick + i}))
                    os.replace(tmp, beat)
                if "msg" in box:
                    res.update(box["msg"][1])
                else:
                    res["error"] = "drain timeout" if th.is_alive() else "no summary"
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            res.update(rc=p.returncode, drain_s=round(time.time() - t0, 1))
            out.append(res)
        return out

    def destroy(self, results=None):
        if not hasattr(self, "_pose"):
            return
        shadows = self._drain() if hasattr(self, "_rig") else []
        if self._blue_pool is not None:
            self._blue_pool.shutdown(wait=True)
        if self._blue_fh is not None:
            self._blue_fh.close()
        p5.P5PairAgent.destroy(self, results)
        ms = {k: round(1e3 * float(np.mean(v)), 2) for k, v in self._t.items() if v}
        (self.out / "nq3_summary.json").write_text(json.dumps(
            {"ticks": self._tick, "stop": p4.STOP["why"] or "route_end", "t_trigger": self._t_trig,
             "need_k": self.need_k, "rig": self.cfg["rig"], "cam_period": self.cfg["cam_period"],
             "cam_miss": self._cam_miss, "setup": getattr(self, "_setup_s", {}), "shadows": shadows, "ms_mean": ms,
             "ms_p95": {k: round(1e3 * float(np.percentile(v, 95)), 2) for k, v in self._t.items() if v}}))
