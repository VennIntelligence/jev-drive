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
  * cache_lights  - the street-light half of `RouteLightsBehavior` without the per-tick RPCs. On a
                    night route this one behaviour is the largest Python item in the whole run
                    (measured: 17.8 ms of a 71 ms tick, 14.5% of total wall clock), because it
                    re-fetches every street light in the map over RPC on every tick, measures each
                    one's distance in a Python loop, and re-sends `turn_on`/`turn_off` for lights
                    that are already in that state. Street lights do not move, so the list and the
                    positions are fetched once; the first update sends the whole state, later ones
                    only lights whose state changes. The first version trusted `is_on` at cache
                    time and left ~1600 far street lights on at night (not equivalent; fixed
                    2026-09-25, checked with $B2D_LIGHTS_CHECK).
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
# Known variants of that source -> the tick cap they enforce. SimLingo's vendored Bench2Drive (RenzKa/simlingo
# 743b243, based on B2D 0.0.3) is the same loop with the `tick_count > 4000` raise commented out, nothing else.
KNOWN_TICK_SOURCES = {REFERENCE_TICK_MD5: 4000, "73adabfd3ff26160c72e1f7d1f136d5b": None}

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


def install(profile, no_spectator=False, fast_copy=False, zero_copy=False, sensor_tick=False,
            cache_lights=False):
    _patch_tick(profile, no_spectator)
    _patch_callback(profile, fast_copy, zero_copy)
    if sensor_tick:
        _patch_sensor_tick()
    if cache_lights:
        _patch_lights()
    if os.environ.get("B2D_LIGHTS_CHECK") == "1":
        _check_lights()


def reseed_after_build(tm_seed):
    """Counterfactual pairs (scripts/p5_pair_agent.py): make the background independent of the scenarios.

    Background vehicles draw their blueprints, colours and spawn points from `CarlaDataProvider._rng`, the
    same stream the scenario actors draw from, and the scenarios are built first (RouteScenario builds every
    scenario within 500 m at construction, which on Bench2Drive's short routes is all of them). Removing one
    scenario from the XML therefore changes the whole background fleet. Re-seeding the stream in place right
    after RouteScenario.__init__ makes both worlds draw the background from the same state. In place, because
    BackgroundActivity keeps a reference to the RandomState object, created before the scenarios are built."""
    from leaderboard.scenarios.route_scenario import RouteScenario
    inner = RouteScenario.__init__

    def init(self, *args, **kwargs):
        inner(self, *args, **kwargs)
        CarlaDataProvider._rng.seed(2000 + int(tm_seed))

    RouteScenario.__init__ = init


def _hazards(scenario):
    """The actor(s) a counterfactual x- world must not contain: the one that cuts in or runs the light, or the
    walkers / two-wheelers that cross. Parked blockers, containers and props stay: they are context, not factor."""
    for name in ("_adversary_actor", "_parked_actor", "_cut_in_vehicle"):
        actor = getattr(scenario, name, None)
        if actor is not None:
            return [actor]
    actors = [a for a in scenario.other_actors if a is not None]
    crossing = [a for a in actors if a.type_id.startswith("walker.") or a.attributes.get("number_of_wheels") == "2"]
    if crossing:
        return crossing
    if type(scenario).__name__ == "OppositeVehicleRunningRedLight":
        return [a for a in actors if a.type_id.startswith("vehicle.")]
    return []


def track_hazards(out_dir, hide):
    """P5 pairs (scripts/p5_pair_agent.py): write the scenario's hazard actors to hidden.json (every world, so x+
    and x- name the same actors) and, in x- (`hide`), suppress them. x- runs the scenario unchanged - same actors spawned from the same
    random stream, same trigger, same commands to the background traffic (clear the junction, leave space) - but
    its hazard actors are kept 500 m under the road after every scenario tick, so nothing sees or meets them.
    Deleting the scenario from the XML instead also deletes those background commands, and then the background
    differs as soon as the scenario would have triggered. 500 m down is the scenarios' own hiding place, and every
    privileged check (BehaviorAgent, the traffic manager) measures distance in 3-D."""
    from leaderboard.scenarios.route_scenario import RouteScenario
    if hide:
        # HardBreakRoute has no hazard actor: its factor is the background's hard brake. x- keeps the scenario (and
        # everything it does to the background at build time) and only turns that brake command into a no-op.
        import srunner.scenarios.hard_break as hard_break

        class NoStopFrontVehicles(py_trees.behaviours.Success):
            def __init__(self, *args, **kwargs):
                super().__init__(name="NoStopFrontVehicles")

        hard_break.StopFrontVehicles = NoStopFrontVehicles
    hidden, log = [], []
    inner_build = RouteScenario.build_scenarios

    def build(self, ego_vehicle, debug=False):
        n0 = len(self.list_scenarios)
        inner_build(self, ego_vehicle, debug=debug)
        for sc in self.list_scenarios[n0:]:
            for a in _hazards(sc):
                if hide:
                    a.set_simulate_physics(False)
                    hidden.append([a, None])
                log.append({"scenario": type(sc).__name__, "id": a.id, "type_id": a.type_id,
                            "role": a.attributes.get("role_name", ""), "hidden": bool(hide)})
        with open(os.path.join(out_dir, "hidden.json"), "w") as fh:
            json.dump(log, fh)

    RouteScenario.build_scenarios = build
    inner_tick = ScenarioManager._tick_scenario

    def tick(self):
        inner_tick(self)
        for h in hidden:
            a = h[0]
            if not a.is_alive:
                continue
            loc = a.get_location()
            if h[1] is None:
                # 500 m under the hero's ground, not under the actor: HighwayCutIn and OppositeVehicleRunningRedLight
                # already park their actor 500 m down at spawn, and pushing it to -1000 m left the Large Map's bounds
                # and segfaulted the server (Town12 routes 2286, 3072, 3074, 2844, 2847).
                h[1] = CarlaDataProvider.get_hero_actor().get_location().z - 500.0
            # The scenario's own behaviours switch physics back on when they place the actor (HighwayCutIn's
            # ActorTransformSetter); a simulated car underground would fall, so physics goes off again every tick.
            a.set_simulate_physics(False)
            if loc.z > h[1] + 1.0:
                a.set_location(carla.Location(loc.x, loc.y, h[1]))

    ScenarioManager._tick_scenario = tick


def _patch_lights():
    from srunner.scenariomanager.lights_sim import RouteLightsBehavior

    def turn_close_lights_on(self, location):
        if getattr(self, "_cached_lights", None) is None:
            lights = list(self._light_manager.get_all_lights())
            self._cached_lights = lights
            self._light_xyz = np.array([[l.location.x, l.location.y, l.location.z]
                                        for l in lights], dtype=np.float64)
            # Not the lights' current is_on: at night the server switches street lights on by itself around the
            # time the route starts, after this cache is built, and a cache that believed them off never switched
            # ~1600 of them off again (measured with $B2D_LIGHTS_CHECK on Town12, 2026-09-25). The first update sends
            # the whole state; later ones only the changes.
            self._light_on = None
            self._vehicle_state = {}
        radius = max(self._radius,
                     self._radius_increase * CarlaDataProvider.get_velocity(self._ego_vehicle))

        here = np.array([location.x, location.y, location.z])
        want = np.linalg.norm(self._light_xyz - here, axis=1) <= radius
        was = ~want if self._light_on is None else self._light_on
        turn_on = [l for l, w, o in zip(self._cached_lights, want, was) if w and not o]
        turn_off = [l for l, w, o in zip(self._cached_lights, want, was) if o and not w]
        if turn_on:
            self._light_manager.turn_on(turn_on)
        if turn_off:
            self._light_manager.turn_off(turn_off)
        self._light_on = want

        # Vehicles: CarlaDataProvider already holds this tick's transforms, so asking the server
        # for each actor's location again is a round trip for data we have. The light state is only
        # written when it changes.
        for vehicle in CarlaDataProvider.get_all_actors().filter("*vehicle.*"):
            if vehicle.attributes.get("role_name") != "scenario":
                continue
            loc = CarlaDataProvider.get_location(vehicle)
            if loc is None:
                continue
            near = loc.distance(location) <= radius
            if self._vehicle_state.get(vehicle.id) == near:
                continue
            try:
                lights = vehicle.get_light_state()
                lights = (lights | self._vehicle_lights) if near else (lights & ~self._vehicle_lights)
                vehicle.set_light_state(carla.VehicleLightState(lights))
                self._vehicle_state[vehicle.id] = near
            except RuntimeError:
                pass

        lights = self._ego_vehicle.get_light_state()
        self._ego_vehicle.set_light_state(carla.VehicleLightState(lights | self._vehicle_lights))

    RouteLightsBehavior._turn_close_lights_on = turn_close_lights_on


def _check_lights(every=20):
    """Equivalence check for cache_lights, on when $B2D_LIGHTS_CHECK=1 (either implementation): every `every`-th
    street-light update, read every light back from the server and count lights whose on/off state differs from the
    rule RouteLightsBehavior implements (on iff within the radius of the ego). Appends to <attempt>/lights_check.jsonl
    [call, lights, on, mismatches]."""
    from srunner.scenariomanager.lights_sim import RouteLightsBehavior
    inner = RouteLightsBehavior._turn_close_lights_on
    fh = open(os.path.join(os.environ["B2D_ATTEMPT_OUT"], "lights_check.jsonl"), "a", buffering=1)
    calls = [0]

    def checked(self, location):
        inner(self, location)
        calls[0] += 1
        if calls[0] % every:
            return
        radius = max(self._radius, self._radius_increase * CarlaDataProvider.get_velocity(self._ego_vehicle))
        lights = self._light_manager.get_all_lights()
        want = [l.location.distance(location) <= radius for l in lights]
        bad = sum(w != bool(l.is_on) for w, l in zip(want, lights))
        fh.write(json.dumps([calls[0], len(lights), sum(want), bad]) + "\n")

    RouteLightsBehavior._turn_close_lights_on = checked


def _patch_sensor_tick():
    from leaderboard.autoagents.agent_wrapper import AgentWrapper

    original = AgentWrapper._preprocess_sensor_spec

    def preprocess(self, sensor_spec):
        type_, id_, transform, attributes = original(self, sensor_spec)
        # The same whitelist also drops `noise_seed`, so every GNSS/IMU draws the identical noise
        # sequence (seed 0) in every run; a controller experiment passes it to perturb noise.
        for key in ("sensor_tick", "noise_seed"):
            if key in sensor_spec:
                attributes[key] = str(sensor_spec[key])
        return type_, id_, transform, attributes

    AgentWrapper._preprocess_sensor_spec = preprocess


def _patch_callback(profile, fast_copy, zero_copy):
    keep = {}
    # Behaviour-equivalence check (docs/bench2drive-cost.md, 2026-09-25): with $B2D_FRAME_HASH=1 every camera frame's
    # md5 is appended to <attempt>/frame_hash.jsonl, so two runs can be compared image by image.
    hashes = None
    if os.environ.get("B2D_FRAME_HASH") == "1":
        import threading
        hashes = (open(os.path.join(os.environ["B2D_ATTEMPT_OUT"], "frame_hash.jsonl"), "a", buffering=1),
                  threading.Lock())

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
        if hashes is not None:
            # md5 for bitwise identity, plus a 16x9 block-mean grey thumbnail: CARLA's renderer is not bitwise
            # reproducible run to run (eye adaptation runs on wall time), so equivalence is judged by distance
            h, w = image.height, image.width
            grey = raw.reshape(h, w, 4)[:9 * (h // 9), :16 * (w // 16), :3]
            thumb = grey.reshape(9, h // 9, 16, w // 16, 3).mean(axis=(1, 3, 4)).round(1).ravel().tolist()
            line = json.dumps([tag, image.frame, hashlib.md5(raw.tobytes()).hexdigest(), thumb]) + "\n"
            with hashes[1]:
                hashes[0].write(line)
        self._data_provider.update_sensor(tag, array, image.frame)

    CallBack._parse_image_cb = parse_image


def _rc_tracer(every=20):
    """Read-only observer for the catalogue experiment (todos/2026-09-25-simlingo-catalogue), on when
    $B2D_RC_TRACE=1, into $B2D_ATTEMPT_OUT/rc_trace.jsonl: every `every` ticks append [tick, frame, route completion %,
    infraction events so far] from the route's own criteria, so a score under a different tick cap can be recomputed
    from the same trajectory. The tick on which RC first reads 100 is always written, with a
    fifth field the route's own completion percentage at that tick before RouteCompletionTest overwrote it with 100
    (`_route_accum_perc[_index]`), so a completion granted by the percentage threshold (90 vs 99) can be told apart.
    JSON lines, flushed, so a killed route keeps its trace."""
    if os.environ.get("B2D_RC_TRACE") != "1":
        return None
    fh = open(os.path.join(os.environ["B2D_ATTEMPT_OUT"], "rc_trace.jsonl"), "a")
    st = {"crit": None, "prev": 0.0}  # prev: RC as read on the previous tick

    def trace(manager, frame):
        if st["crit"] is None:  # the criteria tree is fixed once the route is built
            crit = manager.scenario.get_criteria()
            st["rc"] = next(c for c in crit if type(c).__name__ == "RouteCompletionTest")
            st["crit"] = [c for c in crit if type(c).__name__ not in ("RouteCompletionTest", "MinimumSpeedRouteTest")]
        n, rc = manager.tick_count, st["rc"].actual_value
        done = rc >= 100 and st["prev"] < 100
        if done or n % every == 0 or n == 1:
            row = [n, frame, rc, sum(len(c.events) for c in st["crit"])]
            if done:
                row.append(round(st["rc"]._route_accum_perc[st["rc"]._index], 2))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        st["prev"] = rc

    return trace


def _patch_tick(profile, no_spectator):
    original = inspect.getsource(ScenarioManager._tick_scenario)
    digest = hashlib.md5(original.encode()).hexdigest()
    if digest not in KNOWN_TICK_SOURCES:
        raise RuntimeError(
            "Bench2Drive's _tick_scenario has changed (md5 %s, expected %s). The instrumented copy "
            "in scripts/b2d_hooks.py no longer mirrors it; re-derive it before trusting any "
            "timing." % (digest, REFERENCE_TICK_MD5))
    tick_cap = KNOWN_TICK_SOURCES[digest]
    rc_trace = _rc_tracer()

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

            if tick_cap is not None and self.tick_count > tick_cap:
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
            if rc_trace is not None:
                rc_trace(self, timestamp.frame)

    ScenarioManager._tick_scenario = tick
