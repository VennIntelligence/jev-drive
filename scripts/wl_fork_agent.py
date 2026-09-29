#!/usr/bin/env python
"""WL fork recorder (todos/2026-09-28-wm-loop.md): the P5 / P6 recorder (scripts/p5_pair_agent.py, unchanged outputs)
with the ego handed from the expert to P7 for scripted action windows.

Loaded like the P5 / P6 recorder: `scripts/b2d_run.py --agent scripts/wl_fork_agent.py --agent-config <json>
--tm-seed-from-id`. The agent config is the source set's recorder config plus
  "wl_jobs": path to a JSON {route id: job}, written by `jevdrive.wl forks` / `jevdrive.wl d2`, where a job is
      fork_tick     agent tick (P5 / P6 `k`) at which the fork action starts; null for a D2 run (random windows only)
      action        one of jevdrive.wl_traj.ACTIONS
      op_plan       openpilot Cinque's (20, 2) ego-frame plan at fork_tick from the source run's stored stream, or null
      branch_s      length of the fork window (3 s)
      cont_s        expert driving after the fork window, with random windows (20 s); the run stops after it
      iv_s, iv_gap  random window length (2 s) and gap range between windows ([6, 10] s)
      iv_from_s     D2: sim time of the first random window's earliest start (10 s)
      seed          the random windows' stream
      pre_cams      camera frames saved before each window (default 11: 8 history + 3 for V-JEPA's 4-frame clip)
      post_cams     camera frames saved after each window start (default 15 for the fork, 10 for random windows)
      src           the source attempt directory (bookkeeping only)
      rewind        opt-in (default off; or agent-config key "wl_rewind"): one of scripts/carla_rewind.METHODS. The run
                    then plays every action of `actions` (default: all 7) at the same fork in one route: prefix once,
                    snapshot at fork_tick - 1, a 3 s branch, rewind, the next branch; no continuation after a branch
                    (cont_s is ignored) and no random windows. todos/2026-09-29-carla-rewind.md has where it is exact.
Inside a window the ego follows the window's candidate trajectory (fixed in world coordinates at the window start,
re-expressed in the current ego frame at every camera tick and handed to P7, stepped every tick); the expert keeps
running on the same inputs and its control is discarded, so it picks up from the real state when the window ends.
Camera JPEGs are written from pre_cams camera frames before the first window to the end of the run (disk: ~250 KB per
image; the pipeline prunes them once the features are extracted). post_cams is kept in the job for bookkeeping only.

Extra outputs: wl.json (job, window schedule, every window's candidates in ego coordinates, the fork pose),
wl_ticks.jsonl (per tick inside a window: applied control, controller diagnostics, pose), collisions.jsonl (every
collision event of the ego: frame, other actor id / type, impulse). A rewind run adds rewind.jsonl (per branch: action,
first frame, restore cost) and a "branch" field in wl_ticks.jsonl / wl.json; its ticks restart at fork_tick for every
branch (frames stay unique), so a branch is cut out of the run by frame (scripts/rewind_eval.py).
Python 3.10: envs/scout-tfv6 (BehaviorAgent) or envs/p5v1-pdm (PDM-Lite), like the recorder.
"""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import carla
import numpy as np
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import p4_carla_agent as p4  # noqa: E402
from b2d_controller import pursuit_from_config  # noqa: E402
from p5_pair_agent import P5PairAgent  # noqa: E402
from jevdrive import wl_traj as WT  # noqa: E402

P7 = HERE.parent / "todos/2026-09-23-tfv6-controller/controller-eval/P7.json"
TICK = 0.05


def get_entry_point():
    return "WLForkAgent"


class WLForkAgent(P5PairAgent):
    def setup(self, path_to_conf_file, _=None, __=None):
        super().setup(path_to_conf_file)
        rid = os.environ["BENCHMARK_ROUTE_ID"]
        self.job = json.loads(Path(self.cfg["wl_jobs"]).read_text())[rid]
        j = self.job
        rw = j.get("rewind") or self.cfg.get("wl_rewind")
        self.rw_method = rw if rw and j.get("fork_tick") is not None else None
        self.rw, self.rw_pending, self.rw_idx, self._t_off = None, False, 0, 0.0
        if self.rw_method:
            import carla_rewind as CR
            CR.install()
            self.rw_actions = list(j.get("actions") or WT.ACTIONS)
            j = self.job = dict(j, action=self.rw_actions[0], cont_s=0.0)
            self._rw_log = open(self.out / "rewind.jsonl", "w", buffering=1)
        rng = np.random.RandomState(int(j.get("seed", 0)))
        gap, iv = j.get("iv_gap", [6.0, 10.0]), float(j.get("iv_s", 2.0))
        acts = list(WT.ACTIONS)
        wins = []
        if j.get("fork_tick") is not None:
            k0 = int(j["fork_tick"])
            wins.append({"tick": k0, "len_s": float(j.get("branch_s", 3.0)), "action": j["action"], "kind": "fork",
                         "post": int(j.get("post_cams", 15))})
            t = k0 * TICK + float(j.get("branch_s", 3.0))
            end = t + float(j.get("cont_s", 20.0))
        else:
            t, end = float(j.get("iv_from_s", 10.0)) - gap[0], float(j.get("max_s", self.cfg["max_sim_s"]))
        while True:
            t += rng.uniform(*gap)
            if t + iv > end:
                break
            wins.append({"tick": int(round(t / TICK / p4.CAM_TICKS)) * p4.CAM_TICKS + 1, "len_s": iv,
                         "action": acts[rng.randint(len(acts))], "kind": "random", "post": int(j.get("post_cams_iv", 10))})
            t += iv
        self.wins, self.end_tick = wins, int(round(end / TICK))
        if self.rw_method:
            wins[1:] = []
            self.fork_tick, self.end_tick = wins[0]["tick"], 10 ** 9     # the last branch's end stops the route
        pre = int(j.get("pre_cams", self.cfg.get("wl_pre_cams", 11)))
        # one contiguous range from pre_cams before the first window to the end: openpilot's recurrent `temporal` needs an
        # unbroken stream (the source run's frames supply everything before it); JPEGs are pruned after feature extraction
        self.save_ranges = [(wins[0]["tick"] - pre * p4.CAM_TICKS, self.end_tick)] if wins else []
        self.ctl = pursuit_from_config(str(P7))
        self.rear = float(json.loads(P7.read_text())["rear_axle_offset_m"])
        self.active, self.wl_log = None, {"job": j, "windows": wins, "end_tick": self.end_tick, "cands": []}
        # optional no-rendering prefix: rendering off from the first tick until 2 camera frames before the first save
        self.norender_until = (min(a for a, _ in self.save_ranges) - 2 * p4.CAM_TICKS) if self.cfg.get("wl_norender") and wins else 0
        self.norender, self._unpatch_at = False, None
        self._ticks = open(self.out / "wl_ticks.jsonl", "w")
        self._coll = open(self.out / "collisions.jsonl", "w", buffering=1)

    # ------------------------------------------------------------------ helpers

    def _init_world(self):
        super()._init_world()
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._csensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self._hero)
        self._csensor.listen(lambda e: self._coll.write(json.dumps(
            {"frame": e.frame, "other_id": e.other_actor.id, "other_type": e.other_actor.type_id,
             "impulse": [e.normal_impulse.x, e.normal_impulse.y, e.normal_impulse.z]}) + "\n"))
        self._route_xy = np.array([[t.location.x, t.location.y] for t, _ in self._dense], float)
        if self.norender_until > self._tick:
            self._set_render(False)
        if self.rw_method:
            import carla_rewind as CR
            self.rw = CR.Rewinder(self.rw_method, self._world, self._hero, CarlaDataProvider.get_traffic_manager_port())

    def _set_render(self, on):
        """Rendering on / off for the no-rendering prefix. While off, the leaderboard's sensor wait takes only the
        non-camera sensors (cameras deliver nothing), and the visibility count is skipped."""
        st = self._world.get_settings()
        st.no_rendering_mode = not on
        self._world.apply_settings(st)
        si = self.sensor_interface
        if on:
            self._unpatch_at = self._tick + 1        # this tick was simulated without rendering: keep the patch once more
        else:
            self.norender = True
            from queue import Empty
            cams = {x["id"] for x in self.sensors() if x["type"].startswith("sensor.camera")}
            need = [k for k in si._sensors_objects if k not in cams]

            def get_data(frame):
                out = {}
                while any(k not in out for k in need):
                    try:
                        d = si._data_buffers.get(True, si._queue_timeout)
                    except Empty:
                        from leaderboard.envs.sensor_interface import SensorReceivedNoData
                        raise SensorReceivedNoData("A sensor took too long to send their data")
                    if d[1] == frame and d[0] not in cams:
                        out[d[0]] = (d[1], d[2])
                return out
            si.get_data = get_data

    def _visibility(self, frame, rows):
        if self.norender:
            self._seg_first = True                     # the first frame after rendering resumes has no image yet
            return None
        return super()._visibility(frame, rows)

    def _rear_pose(self):
        tf = self._hero.get_transform()
        f = tf.get_forward_vector()
        return np.array([tf.location.x + self.rear * f.x, tf.location.y + self.rear * f.y]), tf.rotation.yaw

    def _save(self, frame, input_data):
        if any(a <= self._tick <= b for a, b in self.save_ranges):
            return super()._save(frame, input_data)
        return {}, [None, None, None]

    def _start(self, w, now):
        xy, yaw = self._rear_pose()
        v = self._hero.get_velocity()
        v0 = math.hypot(v.x, v.y)
        i = int(np.argmin(np.hypot(*(self._route_xy - xy).T)))
        route_ego = WT.world_to_ego(self._route_xy[max(0, i - 2): i + 400], xy, yaw)
        op = self.job.get("op_plan") if w["kind"] == "fork" else None
        cands = WT.candidates(op, route_ego, v0)
        traj = cands[w["action"]]
        self.active = {"w": w, "t0": now, "until": now + w["len_s"] - 1e-6,
                       "world": np.vstack([xy, WT.ego_to_world(traj, xy, yaw)])}   # t = 0, T
        if self.rw is not None:
            self._rw_log.write(json.dumps({"kind": "start", "branch": self.rw_idx, "action": w["action"], "tick": self._tick,
                                           "frame": GameTime.get_frame(), "t_game": GameTime.get_time(), "t_agent": now,
                                           "wall": time.time()}) + "\n")
        self.wl_log["cands"].append({"tick": self._tick, "t": now, "kind": w["kind"], "action": w["action"],
                                     **({"branch": self.rw_idx} if self.rw is not None else {}),
                                     "rear_xy": xy.tolist(), "yaw": yaw, "v0": v0, "op_plan": op,
                                     "route_ego": np.round(route_ego[:120], 3).tolist(),
                                     "cands": {k: np.round(c, 3).tolist() for k, c in cands.items()}})

    def _plan(self, now):
        a = self.active
        t_rel = now - a["t0"]
        q = t_rel + WT.DT * np.arange(1, 25)
        q = q[q <= WT.HORIZON + 1e-9]
        tt, wp = np.r_[0.0, WT.T], a["world"]
        pts = np.column_stack((np.interp(q, tt, wp[:, 0]), np.interp(q, tt, wp[:, 1])))
        xy, yaw = self._rear_pose()
        return self.ctl.update(WT.world_to_ego(pts, xy, yaw), now, trajectory_dt=WT.DT)

    # ------------------------------------------------------------------ per tick

    def _rewind_tick(self):
        """The dead tick: the world and the scenario go back to fork_tick - 1 and the expert's control of that tick is
        replayed, so the next tick is the fork tick of the next branch."""
        t0 = time.perf_counter()
        st = self.rw.restore()
        snap = self.rw.snap
        self.ctl = copy.deepcopy(snap["extra"]["ctl"])
        # tree methods put GameTime back; the others keep it running and shift the agent's clock instead
        self._t_off = 0.0 if self.rw_method in ("tree", "respawn") else GameTime.get_time() - snap["t"]
        self._tick = self.fork_tick - 1
        self.rw_idx += 1
        self.wins = [{"tick": self.fork_tick, "len_s": float(self.job.get("branch_s", 3.0)),
                      "action": self.rw_actions[self.rw_idx], "kind": "fork", "post": int(self.job.get("post_cams", 15))}]
        self.active, self.rw_pending = None, False
        self._rw_log.write(json.dumps({"kind": "rewind", "branch": self.rw_idx, "action": self.rw_actions[self.rw_idx],
                                       "dead_frame": GameTime.get_frame(), "dead_tick_ms": 1e3 * (time.perf_counter() - t0),
                                       "wall": time.time(), **st}) + "\n")
        c = carla.VehicleControl(snap["control"].throttle, snap["control"].steer, snap["control"].brake)
        c.manual_gear_shift = False
        return c

    def __call__(self):
        if self.rw_pending:
            return self._rewind_tick()
        if self.rw is not None:
            self.rw.unpatch()                         # the dead tick's snapshot reads end with it
        if self.norender and getattr(self, "_unpatch_at", None) is not None and self._tick + 1 >= self._unpatch_at:
            self.sensor_interface.__dict__.pop("get_data", None)
            self.norender, self._unpatch_at = False, None
        elif self.norender and getattr(self, "_unpatch_at", None) is None and self._tick + 1 >= self.norender_until:
            self._set_render(True)                    # the next world tick renders again
        control = super().__call__()                  # recorder + expert (its control is kept outside windows)
        now = GameTime.get_time() - self._t_off
        if self._tick < self.end_tick and p4.STOP["flag"] and p4.STOP["why"] in ("after_trigger", "passed", "stuck",
                                                                                  "max_sim_s"):
            p4.STOP.update(flag=False, why="")       # the source run's stop rules must not cut a fork short
        if self._tick >= self.end_tick:
            p4.STOP.update(flag=True, why="wl_end")
        v, w = self._hero.get_velocity(), self._hero.get_angular_velocity()
        speed, yaw_rate = math.hypot(v.x, v.y), -math.radians(w.z)
        if self.active is None and self.wins and self._tick >= self.wins[0]["tick"]:
            self._start(self.wins.pop(0), now)
        accepted = None
        if self.active is not None and (self._tick - 1) % p4.CAM_TICKS == 0:
            accepted = self._plan(now)
        thr, steer, brk = self.ctl.step(now, speed if speed >= 0.01 else 0.0, yaw_rate)   # every tick: odometry
        if self.active is None:
            if self.rw is not None and self.rw.snap is None and self._tick == self.fork_tick - 1:
                info = self.rw.snapshot(GameTime.get_time(), control, extra={"ctl": copy.deepcopy(self.ctl), "t_agent": now})
                self._rw_log.write(json.dumps({"kind": "snapshot", "tick": self._tick, "frame": GameTime.get_frame(),
                                               "wall": time.time(), **info}) + "\n")
            return control
        c = carla.VehicleControl(throttle=float(thr), steer=float(steer), brake=float(brk))
        c.manual_gear_shift = False
        xy, yaw = self._rear_pose()
        rwf = {"branch": self.rw_idx, "frame": GameTime.get_frame()} if self.rw is not None else {}
        self._ticks.write(json.dumps({"tick": self._tick, **rwf, "t": round(now, 4), "kind": self.active["w"]["kind"],
                                      "action": self.active["w"]["action"], "throttle": c.throttle, "steer": c.steer,
                                      "brake": c.brake, "accepted": accepted, "speed": speed, "rear_xy": xy.tolist(),
                                      "yaw": yaw, "diag": self.ctl.diagnostics if accepted is not None else None}, default=str) + "\n")
        if now >= self.active["until"]:
            self.active = None
            if self.rw is not None:
                self._rw_log.write(json.dumps({"kind": "end", "branch": self.rw_idx, "frame": GameTime.get_frame(),
                                               "wall": time.time()}) + "\n")
                if self.rw_idx + 1 < len(self.rw_actions):
                    self.rw_pending = True            # the next agent call is the dead tick
                else:
                    p4.STOP.update(flag=True, why="wl_end")
        return c

    def destroy(self, results=None):
        if getattr(self, "_csensor", None) is not None:
            self._csensor.stop()
            self._csensor.destroy()
        if getattr(self, "rw", None) is not None:
            self.rw.unpatch()
        for fh in ("_ticks", "_coll", "_rw_log"):
            if getattr(self, fh, None) is not None:
                getattr(self, fh).close()
        if hasattr(self, "wl_log"):
            (self.out / "wl.json").write_text(json.dumps(self.wl_log))
        super().destroy(results)
