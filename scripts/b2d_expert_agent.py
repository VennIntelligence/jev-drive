#!/usr/bin/env python
"""PDM-Lite expert (SimLingo's Bench2Drive copy, leaderboard/team_code/autopilot.py, run as shipped) with a trajectory log,
for the controller acceptance test (todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers.md).

The expert drives exactly as it does in SimLingo's data collection and in its published Bench2Drive result
(pdm_lite_b2d_traj, DS 97.0); this subclass only writes, every tick, the hero's simulator pose and speed and the control
the expert applied to `$B2D_ATTEMPT_OUT/expert.jsonl`:

    {"t": sim time (s), "frame": int, "x", "y": vehicle centre (CARLA world, m), "yaw": deg, "v": m/s,
     "throttle", "steer", "brake"}

The log is the known-good plan that scripts/b2d_zeroshot_agent.py ("replay": <this file>) feeds to each controller.

Requirements: the SimLingo Bench2Drive tree (BENCH2DRIVE_ROOT=$DATA_DIR/third_party/simlingo/Bench2Drive; PDM-Lite reads
CarlaDataProvider.active_scenarios, which only that tree records), its python ($DATA_DIR/envs/simlingo), and
IS_BENCH2DRIVE=1 (the shipped setup() reads an undefined name without it). SAVE_PATH must be unset (no data collection).
"""
import json
import os
import sys

TEAM_CODE = os.path.join(os.environ.get("BENCH2DRIVE_ROOT", ""), "leaderboard", "team_code")
if TEAM_CODE not in sys.path:
    sys.path.insert(0, TEAM_CODE)       # autopilot.py imports its siblings (nav_planner, config, ...) top-level
os.environ.setdefault("IS_BENCH2DRIVE", "1")
os.environ.pop("SAVE_PATH", None)

from autopilot import AutoPilot  # noqa: E402
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: E402
from srunner.scenariomanager.timer import GameTime  # noqa: E402


def get_entry_point():
    return "ExpertLogAgent"


class ExpertLogAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        out = os.environ.get("B2D_ATTEMPT_OUT", ".")
        self._log = open(os.path.join(out, "expert.jsonl"), "w", buffering=1)

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        control = super().run_step(input_data, timestamp, sensors, plant)
        hero = CarlaDataProvider.get_hero_actor()
        tf, vel = hero.get_transform(), hero.get_velocity()
        self._log.write(json.dumps({
            "t": round(float(GameTime.get_time()), 4), "frame": int(GameTime.get_frame()),
            "x": round(tf.location.x, 4), "y": round(tf.location.y, 4), "yaw": round(tf.rotation.yaw, 4),
            "v": round((vel.x ** 2 + vel.y ** 2) ** .5, 4),
            "throttle": round(float(control.throttle), 4), "steer": round(float(control.steer), 4),
            "brake": round(float(control.brake), 4)}) + "\n")
        return control

    def destroy(self, results=None):
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()
        super().destroy(results)
