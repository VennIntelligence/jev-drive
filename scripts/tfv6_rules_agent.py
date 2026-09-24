"""TFv6 rules x interface agent (todos/2026-09-25-tfv6-rules-interface/README.md).

The author's LEAD `SensorAgent` driving on its own, as shipped. The four arms differ only in two config
switches the author exposes:
  interface  A = route + target speed (the author's default modalities), B = waypoints (all three modalities
             set to "waypoint", the same switch W2 used in scripts/b2d_tfv6_controller_agent.py)
  rules      on = README reproduction config (Kalman GPS filter, stop-sign and creeping heuristics),
             off = LEAD's ClosedLoopConfig defaults (all three off); set by the runner through the author's
             own LEAD_CLOSED_LOOP_CONFIG environment variable, never here.
Nothing in the model, the preprocessing, the PIDs or the post-processors is changed. The agent only observes:
per tick it logs both channels (the route/target-speed PID and the waypoint PID are both run by the author's
ensemble() every tick), the control chosen before the post-processors, what each post-processor did to it, the
executed control, ego truth and the positions of the scenario's hazard actors (hidden.json, written by
b2d_hooks.track_hazards when the route carries p5_suppress).

Config string (leaderboard --agent-config): "<model_dir>+<A|B>". Environment:
  B2D_ATTEMPT_OUT        attempt dir (set by b2d_route.py); frames.jsonl, rules_summary.json go there
  TFV6_AFTER_TRIGGER_S   optional: end a pair-world route (id >= 100000) this many sim seconds after the
                         scenario triggers (x- and null only need the scenario window)
"""
import json
import math
import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import py_trees

from leaderboard.scenarios.scenario_manager import ScenarioManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from lead.inference.sensor_agent import SensorAgent

TFV6_SPEEDS = np.array([0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0])
STOP = {"flag": False, "why": ""}


def _patch_scenario_manager():
    """Early end of a truncated pair world: stop the scenario loop from inside, as p4_carla_agent does."""
    if getattr(ScenarioManager, "_tfv6_rules_patched", False):
        return
    inner = ScenarioManager._tick_scenario

    def tick(self):
        inner(self)
        if STOP["flag"]:
            self._running = False

    ScenarioManager._tick_scenario = tick
    ScenarioManager._tfv6_rules_patched = True


_patch_scenario_manager()


def get_entry_point():
    return "TFv6RulesAgent"


def _r(x, n=4):
    return None if x is None else round(float(x), n)


class TFv6RulesAgent(SensorAgent):
    def setup(self, path_to_conf_file, *args, **kwargs):
        model_dir, _, arm = path_to_conf_file.partition("+")
        self.arm = arm.upper() or "A"
        if self.arm not in ("A", "B"):
            raise ValueError("interface arm must be A or B, got %r" % arm)
        self.out = Path(os.environ["B2D_ATTEMPT_OUT"])
        os.environ.setdefault("SAVE_PATH", str(self.out / "lead_save"))
        # The author's setup refuses to start without ffmpeg (it compresses videos we never produce).
        with mock.patch("shutil.which", return_value="/bin/true"):
            super().setup(model_dir, *args, **kwargs)
        c = self.config_closed_loop
        if self.arm == "B":
            c.steer_modality = c.throttle_modality = c.brake_modality = "waypoint"
        # Only pair worlds are cut (their ids are base * 100 + world * 10 + seed, jevdrive/p5_pairs.py);
        # Bench2Drive routes (ids < 100000) always run to the end so their official score stands.
        rid = os.environ.get("BENCHMARK_ROUTE_ID", "0")
        self._after_trigger = (float(os.environ.get("TFV6_AFTER_TRIGGER_S", "0") or 0)
                               if rid.isdigit() and int(rid) >= 100000 else 0.0)
        self._rules = {"use_kalman_filter": bool(c.use_kalman_filter),
                       "sensor_agent_creeping": bool(c.sensor_agent_creeping),
                       "slower_for_stop_sign": bool(c.slower_for_stop_sign)}
        self._log = open(self.out / "frames.jsonl", "w", buffering=1 << 16)
        self._pred = self._pre = self._wp_ctrl = None
        self._post = {}
        self._t_trig = None
        self._hazard_ids = None
        self._ms = {"agent": [], "forward": []}
        self._n_rule = {"force": 0, "stop": 0}

        inference = self.closed_loop_inference
        forward, waypoints = inference.forward, inference.execute_waypoints

        def capture_waypoints(*a, **k):
            self._wp_ctrl = waypoints(*a, **k)
            return self._wp_ctrl

        def capture_forward(*a, **k):
            t0 = time.perf_counter()
            self._wp_ctrl = None
            p = forward(*a, **k)
            self._ms["forward"].append(time.perf_counter() - t0)
            self._pred = p
            self._pre = (float(p.steer), float(p.throttle), float(p.brake))   # before the post-processors
            return p

        inference.execute_waypoints = capture_waypoints
        inference.forward = capture_forward
        for name, proc in (("force", self.force_move_post_processor), ("stop", self.stop_sign_post_processor)):
            inner = proc.adjust

            def adjust(speed, throttle, brake, _inner=inner, _name=name, _proc=proc):
                out = _inner(speed, throttle, brake)
                changed = abs(float(out[0]) - float(throttle)) > 1e-6 or abs(float(out[1]) - float(brake)) > 1e-6
                self._post[_name] = [_r(throttle), _r(brake), _r(out[0]), _r(out[1])] if changed else None
                if changed:
                    self._n_rule[_name] += 1
                return out

            proc.adjust = adjust
        if self._after_trigger:
            STOP.update(flag=False, why="")

    # The Bench2Drive agent base dumps a growing metric_info.json every tick when this exists; hide it.
    @property
    def get_metric_info(self):
        raise AttributeError("disabled")

    def _hazards(self, world, snap):
        if self._hazard_ids is None:
            f = self.out / "hidden.json"
            self._hazard_ids = [int(h["id"]) for h in json.loads(f.read_text())] if f.exists() else []
        rows = []
        for aid in self._hazard_ids:
            s = snap.find(aid)
            if s is not None:
                tf, v = s.get_transform(), s.get_velocity()
                rows.append([aid, _r(tf.location.x, 3), _r(tf.location.y, 3), _r(tf.location.z, 2),
                             _r(math.hypot(v.x, v.y), 3)])
        return rows

    def run_step(self, input_data, timestamp, *args, **kwargs):
        t0 = time.perf_counter()
        self._pred = self._pre = self._wp_ctrl = None
        self._post = {}
        control = super().run_step(input_data, timestamp, *args, **kwargs)
        t = float(timestamp)
        if self._t_trig is None and py_trees.blackboard.Blackboard().get("ScenarioRouteNumber0"):
            self._t_trig = t
        hero = CarlaDataProvider.get_hero_actor()
        world = CarlaDataProvider.get_world()
        snap = world.get_snapshot()
        a = snap.find(hero.id)
        tf, v = a.get_transform(), a.get_velocity()
        row = {"step": int(self.step), "t": round(t, 3), "frame": int(snap.frame),
               "x": _r(tf.location.x, 4), "y": _r(tf.location.y, 4), "yaw": _r(tf.rotation.yaw, 4),
               "v": _r(math.hypot(v.x, v.y), 4),
               "exec": [_r(control.steer), _r(control.throttle), _r(control.brake)], "trig": self._t_trig is not None}
        p = self._pred
        if p is not None:
            dist = p.pred_target_speed_distribution
            row["ts"] = _r(p.pred_target_speed_scalar[0, 0].item())
            if dist is not None:                      # already softmaxed by the author's ensemble
                prob = dist.detach().float().reshape(-1).cpu().numpy()
                if prob.shape[0] == TFV6_SPEEDS.shape[0]:
                    row["ts_exp"] = _r(float(prob @ TFV6_SPEEDS))
                    row["ts_p0"] = _r(prob[0])
            wp = p.pred_future_waypoints[0].detach().float().cpu().numpy()
            row["wp"] = np.round(wp, 3).tolist()
            row["wp_v2"] = _r(np.linalg.norm(wp[7] - wp[6]) / 0.25)        # waypoint-implied speed at 2 s
            row["pre"] = [_r(x) for x in self._pre]
            row["chA"] = [_r(p.route_steer), _r(p.target_speed_throttle), _r(p.target_speed_brake)]
            if self._wp_ctrl is not None:
                row["chB"] = [_r(x) for x in self._wp_ctrl]
            if any(self._post.values()):
                row["rule"] = {k: val for k, val in self._post.items() if val}
            fm = self.force_move_post_processor
            row["stuck"] = int(fm.stuck_detector)
            if fm.force_move:
                row["force_move"] = int(fm.force_move)
        haz = self._hazards(world, snap)
        if haz:
            row["haz"] = haz
        self._log.write(json.dumps(row) + "\n")
        if self._after_trigger and self._t_trig is not None and t > self._t_trig + self._after_trigger:
            STOP.update(flag=True, why="after_trigger")
        self._ms["agent"].append(time.perf_counter() - t0)
        return control

    def destroy(self, results=None):
        # The author's destroy() compresses videos with ffmpeg; nothing of that exists here.
        if not hasattr(self, "_log"):
            return
        self._log.close()
        ms = {k: round(1e3 * float(np.mean(v)), 2) for k, v in self._ms.items() if v}
        (self.out / "rules_summary.json").write_text(json.dumps(
            {"arm": self.arm, "rules": self._rules, "ticks": int(self.step), "t_trigger": self._t_trig,
             "stop": STOP["why"] or "route_end", "rule_ticks": self._n_rule, "ms_mean": ms,
             "after_trigger_s": self._after_trigger}))
