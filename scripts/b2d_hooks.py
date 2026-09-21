#!/usr/bin/env python
"""Instrumentation and optional optimisations for the Bench2Drive closed-loop.

Everything here is a monkeypatch applied to an unmodified Bench2Drive checkout, so the checkout
stays the reference implementation and every optimisation is a flag (research/carla-efficiency.md:
"每个优化留开关"). Two groups:

Measurement (always on)
  * `_tick_scenario` is replaced by a line-for-line copy carrying a `perf_counter` around each
    phase. The copy is guarded by the md5 of the original source: if Bench2Drive changes the loop,
    the run aborts instead of silently profiling something that no longer matches.
  * the image callback is wrapped to time `frombuffer` and the copy separately, in the client
    worker thread where they actually run.

Optimisations (each behind its own flag, all off by default)
  * no_spectator  - drop the two per-tick RPCs that move a camera nobody is watching. Headless runs
                    render no spectator view, so this changes nothing observable.
  * fast_copy     - `np.frombuffer(...).copy()` instead of `copy.deepcopy`, same bytes.
  * zero_copy     - no copy at all: keep the `carla.Image` alive and hand out a view of its buffer.
                    Correct only because the leaderboard consumes each frame before the next one
                    arrives; the reference is dropped when the next frame for that tag lands.
  * sensor_tick   - let a camera spec carry `sensor_tick`. The leaderboard builds each sensor's
                    blueprint attributes from a hard-coded whitelist per sensor type
                    (agent_wrapper.py `_preprocess_sensor_spec`) which has no `sensor_tick` for any
                    sensor, so an agent cannot ask for its cameras at less than the tick rate: the
                    attribute is silently dropped and every camera renders 20 times a second
                    whatever the agent asked for. Without this patch, decimation saves nothing -
                    measured, the frames still arrive and the wait simply moves from the sensor
                    queue into the next `world.tick()`.

Python 3.8: runs in envs/carla.
"""
import copy
import hashlib
import inspect
import json
import os
import time

import carla
import numpy as np
import py_trees

from leaderboard.autoagents.agent_wrapper import AgentError, TickRuntimeError
from leaderboard.envs.sensor_interface import CallBack, SensorReceivedNoData
from leaderboard.scenarios.scenario_manager import ScenarioManager
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

# md5 of the Bench2Drive 0.0.4 `_tick_scenario` the instrumented copy below was derived from.
REFERENCE_TICK_MD5 = "266b1ddfcbdbe2e9cb4f2bd8d58655a2"

PHASES = ("world_tick", "snapshot", "provider", "agent", "control", "tree", "spectator", "total")


class TickProfile(object):
    def __init__(self, heartbeat_path=None):
        self.phase = dict((p, []) for p in PHASES)
        self.copy_ms = []       # per tick, summed over cameras, measured in the worker threads
        self.frombuffer_ms = []
        self.bytes_in = 0
        self.ticks = 0
        self._pending_copy = 0.0
        self._pending_frombuffer = 0.0
        self.heartbeat_path = heartbeat_path
        self._last_beat = 0.0
        self.t_start = time.time()

    def add_copy(self, frombuffer_s, copy_s, nbytes):
        # Called from CARLA client worker threads. The GIL makes += safe enough for a float sum.
        self._pending_frombuffer += frombuffer_s
        self._pending_copy += copy_s
        self.bytes_in += nbytes

    def close_tick(self, phases):
        for k, v in phases.items():
            self.phase[k].append(v)
        self.copy_ms.append(1e3 * self._pending_copy)
        self.frombuffer_ms.append(1e3 * self._pending_frombuffer)
        self._pending_copy = self._pending_frombuffer = 0.0
        self.ticks += 1
        self._beat()

    def _beat(self):
        """The watchdog in b2d_run.py reads this file. It asks whether ticks advance, not whether
        the process is alive: a hung CARLA leaves a perfectly healthy-looking process behind."""
        if not self.heartbeat_path:
            return
        now = time.time()
        if now - self._last_beat < 2.0:
            return
        self._last_beat = now
        tmp = self.heartbeat_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"t": now, "ticks": self.ticks}, fh)
        os.replace(tmp, self.heartbeat_path)

    def summary(self, drop=20):
        """Per-tick milliseconds. The first `drop` ticks are thrown away: the world is still
        streaming tiles and the scenario tree is still being built."""
        out = {"ticks": self.ticks, "ticks_used": max(0, self.ticks - drop)}
        for name, series in list(self.phase.items()) + [("copy", None), ("frombuffer", None)]:
            if series is None:
                series = [x / 1e3 for x in (self.copy_ms if name == "copy" else self.frombuffer_ms)]
            xs = sorted(series[drop:])
            if not xs:
                continue
            out[name + "_ms_mean"] = round(1e3 * sum(xs) / len(xs), 3)
            out[name + "_ms_median"] = round(1e3 * xs[len(xs) // 2], 3)
            out[name + "_ms_p95"] = round(1e3 * xs[int(0.95 * (len(xs) - 1))], 3)
        out["mib_in"] = round(self.bytes_in / 2 ** 20, 1)
        return out


def install(profile, no_spectator=False, fast_copy=False, zero_copy=False, sensor_tick=False):
    _patch_tick(profile, no_spectator)
    _patch_callback(profile, fast_copy, zero_copy)
    if sensor_tick:
        _patch_sensor_tick()


def _patch_sensor_tick():
    from leaderboard.autoagents.agent_wrapper import AgentWrapper

    original = AgentWrapper._preprocess_sensor_spec

    def preprocess(self, sensor_spec):
        type_, id_, transform, attributes = original(self, sensor_spec)
        if "sensor_tick" in sensor_spec:
            attributes["sensor_tick"] = str(sensor_spec["sensor_tick"])
        return type_, id_, transform, attributes

    AgentWrapper._preprocess_sensor_spec = preprocess


def _patch_callback(profile, fast_copy, zero_copy):
    keep = {}

    def parse_image(self, image, tag):
        t0 = time.perf_counter()
        raw = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        t1 = time.perf_counter()
        if zero_copy:
            # The view aliases the image's buffer, so the image has to outlive it. One slot per
            # tag: the previous frame is released when the next one arrives, by which time the
            # leaderboard has handed it to the agent and the agent is done with it.
            keep[tag] = image
            array = raw.reshape((image.height, image.width, 4))
        elif fast_copy:
            array = raw.copy().reshape((image.height, image.width, 4))
        else:
            array = copy.deepcopy(raw)
            array = np.reshape(array, (image.height, image.width, 4))
        t2 = time.perf_counter()
        profile.add_copy(t1 - t0, t2 - t1, raw.nbytes)
        self._data_provider.update_sensor(tag, array, image.frame)

    CallBack._parse_image_cb = parse_image


def _patch_tick(profile, no_spectator):
    original = inspect.getsource(ScenarioManager._tick_scenario)
    digest = hashlib.md5(original.encode()).hexdigest()
    if digest != REFERENCE_TICK_MD5:
        raise RuntimeError(
            "Bench2Drive's _tick_scenario has changed (md5 %s, expected %s). The instrumented copy "
            "in scripts/b2d_hooks.py no longer mirrors it; re-derive it before trusting any "
            "timing." % (digest, REFERENCE_TICK_MD5))

    def tick(self):
        """Line-for-line copy of Bench2Drive 0.0.4 ScenarioManager._tick_scenario with timers.
        Debug branches (`self._debug_mode > 1`) are dropped: we always run with debug 0."""
        t = {}
        t_tick0 = time.perf_counter()
        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)
        t["world_tick"] = time.perf_counter() - t_tick0

        t0 = time.perf_counter()
        timestamp = CarlaDataProvider.get_world().get_snapshot().timestamp
        t["snapshot"] = time.perf_counter() - t0

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            t0 = time.perf_counter()
            self._watchdog.update()
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            self.tick_count += 1
            self._watchdog.pause()
            t["provider"] = time.perf_counter() - t0

            if self.tick_count > 4000:
                raise TickRuntimeError("RuntimeError, tick_count > 4000")

            t0 = time.perf_counter()
            try:
                self._agent_watchdog.resume()
                self._agent_watchdog.update()
                ego_action = self._agent_wrapper()
                self._agent_watchdog.pause()
            except SensorReceivedNoData as e:
                raise RuntimeError(e)
            except Exception as e:
                raise AgentError(e)
            t["agent"] = time.perf_counter() - t0

            self._watchdog.resume()
            t0 = time.perf_counter()
            self.ego_vehicles[0].apply_control(ego_action)
            t["control"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            py_trees.blackboard.Blackboard().set("AV_control", ego_action, overwrite=True)
            self.scenario_tree.tick_once()
            t["tree"] = time.perf_counter() - t0

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            t0 = time.perf_counter()
            if not no_spectator:
                ego_trans = self.ego_vehicles[0].get_transform()
                self._spectator.set_transform(carla.Transform(
                    ego_trans.location + carla.Location(z=70), carla.Rotation(pitch=-90)))
            t["spectator"] = time.perf_counter() - t0

            t["total"] = time.perf_counter() - t_tick0
            profile.close_tick(t)

    ScenarioManager._tick_scenario = tick
