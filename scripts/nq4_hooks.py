#!/usr/bin/env python
"""Night queue 4, G and K: world variants and the per-tick trace of every closed-loop run
(todos/2026-09-26-night-queue-4.md, section G and the [F] entries under it, written before any G number).

scripts/b2d_route.py calls install() when the route carries an `nq4_world` attribute or $B2D_NQ4_TRACE is 1; with
neither, nothing here is imported. Route attributes (written by jevdrive/nq4_g.py):

  nq4_world  "orig"   the route as shipped
             "ghost"  the scenario deleted with the P5 x- / P6 x00 mechanism (rule 11): scripts/b2d_hooks.py
                      p6_world(obstacle="hide") keeps every actor the scenario spawns 500 m under the road, physics off,
                      after every scenario tick (made unconditional and applied right after the build, _ghost_strict:
                      the original lets a newly placed actor show for one frame), and deletes the scenario's entry in
                      CarlaDataProvider.active_scenarios
                      (the registry PDM-Lite plans on). Only SimLingo's Bench2Drive copy has that registry; the official
                      tree gets an empty list so the same code runs in both.
             "shift"  the scenario's trigger point moved along the route by `nq4_shift_m` (+ = later), on the dense
                      route, before the scenarios are built: everything the scenario places relative to its trigger
                      (obstacles, walkers, cut-in vehicles, their registry entry) moves with it.
             "swap"   an actor of the same class: obstacle and pedestrian <-> bicycle swaps are scenario-type swaps done in
                      the XML (no hook); `nq4_swap="van"` swaps the cut-in vehicle's class (base_type car -> van, special
                      type none) for StaticCutIn / ParkingCutIn / HighwayCutIn, blockers and parked cars unchanged.
  nq4_stop_m  G runs only: end the route (ScenarioManager._running = False, the evaluator still writes its record) once the
             hero's actor centre has passed this arc length of the dense route (every G readout lives before it), or
  nq4_stall_s  once it has not advanced 1 m along the route for this many simulated seconds. K runs carry neither.
  nq4_vis    "1"      also run the visibility camera (instance segmentation at the P5 recorder's front-camera pose and
                      field of view, half its render resolution, 5 Hz; P5's pixel rule, jevdrive/p5_pairs.PX_ACTOR)
                      for the scenario's actors: when the hazard is first visible, the reference for readout 2.

Every run writes <attempt>/nq4_trace.npz (flushed every 400 ticks and at exit):
  ego   (n, 7)  tick, sim t, x, y, z, yaw (deg), speed (m/s) of the hero, actor centre, CARLA world frame
  act   (m, 7)  tick, id, x, y, yaw, speed, kind (0 vehicle, 1 walker, 2 static prop) of every other actor within
                60 m of the hero (2-D) and not parked underground (z more than 50 m below the hero: hidden ghost actors,
                scenario actors before their placement), from one world snapshot per tick
  vis   (p, 3)  tick, scenario actor id, visible pixels (nq4_vis only; ticks with no visible scenario actor omitted,
                ticks with a camera frame listed in `vis_ticks`)
and <attempt>/nq4_meta.json: the dense route (x, y), the scenarios as built (type, trigger before / after the shift,
their actor ids and blueprints), the variant, kinds {id: [type_id, role_name]}, hero id.
Python 3.8 (envs/carla) and 3.10 (the model envs) alike.
"""
import atexit
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import carla
import numpy as np

import b2d_hooks

KIND = {"vehicle.": 0, "walker.": 1, "static.prop.": 2}
ACTOR_RADIUS_M = 60.0
VIS_RADIUS_M = 60.0
PX_ACTOR = 20                     # jevdrive/p5_pairs.PX_ACTOR: visible pixels in the half-resolution segmentation view
FLUSH_TICKS = 400
# P4 / P5 front camera (scripts/p4_carla_agent.py WAYMO_CAMS[0], DEFAULT): rig x, y, z; render and crop geometry
FRONT = (1.519, 0.026, 1.806)
REAR_AXLE_X = -1.388633220
RENDER_W, RENDER_H, F_PX, CU, CV, OUT_W, OUT_H = 1088, 1560, 1113.5, 488.1, 719.2, 972, 1079


def _route_attrs(routes, route_id):
    for r in ET.parse(routes).getroot().iter("route"):
        if r.get("id") == str(route_id):
            return dict(r.attrib)
    return {}


def install(out_dir, routes, route_id, tm_seed):
    attrs = _route_attrs(routes, route_id)
    world = attrs.get("nq4_world", "orig")
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    if world == "ghost":
        if not hasattr(CarlaDataProvider, "active_scenarios"):
            CarlaDataProvider.active_scenarios = []          # official tree: no registry; p6_world then finds nothing
        b2d_hooks.p6_world(out_dir, "hide", None, tm_seed)
        _ghost_strict()
    elif world == "shift":
        _shift(float(attrs["nq4_shift_m"]))
    elif world == "swap" and attrs.get("nq4_swap") == "van":
        _swap_cut_in()
    elif world not in ("orig", "swap"):
        raise ValueError("unknown nq4_world %r" % world)
    _setup_timing(out_dir)
    tr = Trace(out_dir, world, attrs, vis=attrs.get("nq4_vis") == "1")
    tr.stop_m = float(attrs["nq4_stop_m"]) if "nq4_stop_m" in attrs else None
    tr.stall_s = float(attrs["nq4_stall_s"]) if "nq4_stall_s" in attrs else None
    tr.install()


def _ghost_strict():
    """p6_world's hide moves an actor down only when its (cached, one tick old) location is above ground, so an actor
    shows for one frame when the scenario places it (smoke 18:41: a DynamicObjectCrossing walker at 57 m for one tick,
    its container on the first tick). Here every scenario actor goes 500 m under the hero right after the build, and is
    set there again after every scenario tick, unconditionally, after the tree's own transform commands of that tick."""
    from leaderboard.scenarios.route_scenario import RouteScenario
    from leaderboard.scenarios.scenario_manager import ScenarioManager
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    held = []                                        # [actor, x, y, z]
    inner_build = RouteScenario.build_scenarios

    def build(self, ego_vehicle, debug=False):
        n0 = len(self.list_scenarios)
        inner_build(self, ego_vehicle, debug=debug)
        z = ego_vehicle.get_location().z - 500.0
        for sc in self.list_scenarios[n0:]:
            for a in sc.other_actors:
                if a is None or any(h[0].id == a.id for h in held):
                    continue
                loc = a.get_location()
                a.set_simulate_physics(False)
                a.set_location(carla.Location(loc.x, loc.y, z))
                held.append([a, loc.x, loc.y, z])

    RouteScenario.build_scenarios = build
    inner_tick = ScenarioManager._tick_scenario

    def tick(self):
        inner_tick(self)
        for h in held:
            a = h[0]
            if not a.is_alive:
                continue
            loc = a.get_location()
            if loc.z > h[3] + 1.0:                   # moved by the scenario since the last tick: keep its new x, y
                h[1], h[2] = loc.x, loc.y
            a.set_simulate_physics(False)
            a.set_location(carla.Location(h[1], h[2], h[3]))

    ScenarioManager._tick_scenario = tick


# ---------------------------------------------------------------- setup timing and (optional) world reuse

TIMES = {}


def _setup_timing(out_dir):
    """Wall time of the route set-up phases -> <attempt>/nq4_setup.json (load_world, RouteScenario build, agent setup).
    With $B2D_NQ4_REUSE_WORLD=1: when this server still holds the world this worker loaded for an earlier route of the
    same town (same episode id, recorded in $B2D_NQ4_WORLD_CACHE/<port>.json), skip load_world: destroy every vehicle,
    walker, controller and sensor and every prop that the fresh load did not have, then continue exactly as
    _load_and_wait_for_world does (Large Map settings, traffic lights reset, provider, TM seed, one tick)."""
    import time
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.scenarios.route_scenario import RouteScenario
    from leaderboard.autoagents.agent_wrapper import AgentWrapper
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    inner_load, inner_rs, inner_setup = LeaderboardEvaluator._load_and_wait_for_world, RouteScenario.__init__, AgentWrapper.setup_sensors
    reuse = os.environ.get("B2D_NQ4_REUSE_WORLD") == "1"
    cache_dir = os.environ.get("B2D_NQ4_WORLD_CACHE", "")

    def dump():
        with open(os.path.join(out_dir, "nq4_setup.json"), "w") as fh:
            json.dump(TIMES, fh)

    def load(self, args, town):
        t0 = time.time()
        cf = os.path.join(cache_dir, "%d.json" % args.port) if cache_dir else ""
        w = self.client.get_world() if reuse and cf else None
        prev = json.load(open(cf)) if cf and os.path.exists(cf) else None
        if w is not None and prev and prev["town"] == town and prev["episode"] == w.id \
                and w.get_map().name.split("/")[-1] == town:
            keep = set(prev["props"])
            dead = [a.id for a in w.get_actors() if a.type_id.startswith(("vehicle.", "walker.", "controller.", "sensor."))
                    or (a.type_id.startswith("static.prop.") and a.id not in keep)]
            if dead:
                self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in dead])
            self.world = w
            settings = self.world.get_settings()
            settings.tile_stream_distance = 650
            settings.actor_active_distance = 650
            self.world.apply_settings(settings)
            self.world.reset_all_traffic_lights()
            CarlaDataProvider.set_client(self.client)
            CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
            CarlaDataProvider.set_world(self.world)
            self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
            self.world.tick()
            TIMES.update(world="reused", destroyed=len(dead))
        else:
            inner_load(self, args, town)
            TIMES["world"] = "loaded"
            if cf:
                w = self.client.get_world()
                with open(cf + ".tmp", "w") as fh:
                    json.dump({"town": town, "episode": w.id,
                               "props": [a.id for a in w.get_actors() if a.type_id.startswith("static.prop.")]}, fh)
                os.replace(cf + ".tmp", cf)
        TIMES["load_world_s"] = round(time.time() - t0, 2)
        dump()

    def rs(self, *args, **kwargs):
        t0 = time.time()
        inner_rs(self, *args, **kwargs)
        TIMES["route_scenario_s"] = round(time.time() - t0, 2)
        dump()

    def setup_sensors(self, *args, **kwargs):
        t0 = time.time()
        r = inner_setup(self, *args, **kwargs)
        TIMES["sensors_s"] = round(time.time() - t0, 2)
        dump()
        return r

    LeaderboardEvaluator._load_and_wait_for_world = load
    RouteScenario.__init__ = rs
    AgentWrapper.setup_sensors = setup_sensors


# ---------------------------------------------------------------- shift

SHIFTED = []


def _shift(shift_m):
    from leaderboard.scenarios.route_scenario import RouteScenario
    inner = RouteScenario._filter_scenarios

    def filt(self, configs):
        P = np.array([[t.location.x, t.location.y] for t, _ in self.route])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        for c in configs:
            tp = c.trigger_points[0]
            i = int(np.argmin(np.hypot(P[:, 0] - tp.location.x, P[:, 1] - tp.location.y)))
            j = int(np.clip(np.searchsorted(s, s[i] + shift_m), 0, len(P) - 1))
            t = self.route[j][0]
            before = [tp.location.x, tp.location.y, tp.location.z, tp.rotation.yaw]
            c.trigger_points[0] = carla.Transform(carla.Location(t.location.x, t.location.y, tp.location.z),
                                                  carla.Rotation(yaw=t.rotation.yaw))
            SHIFTED.append({"name": c.name, "type": c.type, "before": before,
                            "after": [t.location.x, t.location.y, tp.location.z, t.rotation.yaw],
                            "s_before": float(s[i]), "s_after": float(s[j]), "shift_m": shift_m})
        return inner(self, configs)

    RouteScenario._filter_scenarios = filt


# ---------------------------------------------------------------- swap (cut-in vehicle class)

SWAPPED = []


def _swap_cut_in():
    """During scenario construction, the cut-in vehicle's blueprint filter base_type car -> van (special_type ''), the
    rest of the filter unchanged. Which request is the cut-in vehicle: HighwayCutIn and ParkingCutIn make exactly one
    request with role 'scenario' (ParkingCutIn's blocker is 'scenario no lights'); StaticCutIn makes _back_vehicles
    blocker requests and then the cut-in vehicle's, all with the same filter, so it is its (_back_vehicles + 1)-th.
    The scenario is the caller's `self` (scenario modules are loaded twice by RouteScenario, so the class objects cannot
    be patched; the caller's frame is the same whichever copy runs)."""
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    inner = CarlaDataProvider.request_new_actor
    count = {}

    def request(model, spawn_point, rolename='scenario', autopilot=False, random_location=False, color=None,
                actor_category="car", attribute_filter=None, tick=True):
        sc = sys._getframe(1).f_locals.get("self")
        name = type(sc).__name__
        swap = False
        if rolename == "scenario" and (attribute_filter or {}).get("base_type") == "car":
            if name in ("HighwayCutIn", "ParkingCutIn"):
                swap = True
            elif name == "StaticCutIn":
                k = count[id(sc)] = count.get(id(sc), 0) + 1
                swap = k == int(sc._back_vehicles) + 1
        if swap:
            attribute_filter = dict(attribute_filter, base_type="van", special_type="")
        a = inner(model, spawn_point, rolename, autopilot, random_location, color, actor_category, attribute_filter,
                  tick)
        if swap:
            SWAPPED.append({"scenario": name, "id": a.id if a is not None else None,
                            "type_id": a.type_id if a is not None else None, "filter": attribute_filter})
        return a

    CarlaDataProvider.request_new_actor = staticmethod(request)


# ---------------------------------------------------------------- trace

class Trace(object):
    def __init__(self, out_dir, world, attrs, vis):
        self.out, self.world_name, self.attrs, self.vis = out_dir, world, attrs, vis
        self.ego, self.act, self.vrows, self.vis_ticks = [], [], [], []
        self.kinds, self.scen, self.route, self.meta_extra = {}, [], None, {}
        self.cam = self.seg_buf = None
        self.pending = {}                     # frame -> (tick, rows of scenario actors) awaiting its segmentation image
        self.last_flush = 0
        self.hero = None
        self.stop_m = self.stall_s = self.route_s = None
        self.stopped = None

    def install(self):
        from leaderboard.scenarios.route_scenario import RouteScenario
        from leaderboard.scenarios.scenario_manager import ScenarioManager
        trace = self
        inner_build = RouteScenario.build_scenarios

        def build(self, ego_vehicle, debug=False):
            n0 = len(self.list_scenarios)
            inner_build(self, ego_vehicle, debug=debug)
            if trace.route is None:
                trace.route = [[round(t.location.x, 3), round(t.location.y, 3)] for t, _ in self.route]
                trace.hero = ego_vehicle.id
                trace.configs = [{"name": c.name, "type": c.type,
                                  "trigger": [c.trigger_points[0].location.x, c.trigger_points[0].location.y,
                                              c.trigger_points[0].location.z, c.trigger_points[0].rotation.yaw]}
                                 for c in self.scenario_configurations]
                if trace.vis:
                    trace._spawn_camera(ego_vehicle)
            for sc in self.list_scenarios[n0:]:
                acts = [a for a in sc.other_actors if a is not None]
                trace.scen.append({"type": type(sc).__name__, "ids": [a.id for a in acts],
                                   "type_ids": [a.type_id for a in acts]})
            trace._write_meta()

        RouteScenario.build_scenarios = build
        inner_tick = ScenarioManager._tick_scenario

        def tick(self):
            n = self.tick_count
            inner_tick(self)
            if self.tick_count != n:
                trace._record(self.tick_count)
                why = trace._stop_reason()
                if why and self._running:
                    self._running = False
                    trace.stopped = why
                    trace.flush()

        ScenarioManager._tick_scenario = tick
        atexit.register(self.flush)

    def _stop_reason(self):
        """G early stop: progress along the dense route (a moving pointer, forward-only window of 30 points)."""
        if (self.stop_m is None and self.stall_s is None) or not self.ego or self.route is None:
            return None
        if self.route_s is None:
            P = np.asarray(self.route, float)
            self.route_P, self.route_s = P, np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
            self.ptr, self.best_s, self.best_t = 0, 0.0, self.ego[-1][1]
        x, y, t = self.ego[-1][2], self.ego[-1][3], self.ego[-1][1]
        j = self.ptr + int(np.argmin(np.hypot(self.route_P[self.ptr:self.ptr + 30, 0] - x,
                                              self.route_P[self.ptr:self.ptr + 30, 1] - y)))
        self.ptr = j
        s = float(self.route_s[j])
        if s > self.best_s + 1.0:
            self.best_s, self.best_t = s, t
        if self.stop_m is not None and s >= self.stop_m:
            return "passed_stop_m"
        if self.stall_s is not None and t - self.best_t > self.stall_s:
            return "stalled"
        return None

    def _spawn_camera(self, hero):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        world = CarlaDataProvider.get_world()
        self.fov = math.degrees(2 * math.atan(RENDER_W / 2.0 / F_PX))
        bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
        for k, v in (("image_size_x", RENDER_W // 2), ("image_size_y", RENDER_H // 2), ("fov", self.fov),
                     ("sensor_tick", 0.2)):
            bp.set_attribute(k, str(v))
        self.seg_buf = {}
        x, y, z = FRONT
        self.cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=x + REAR_AXLE_X, y=-y, z=z)), attach_to=hero)
        self.cam.listen(lambda img: self.seg_buf.__setitem__(img.frame, bytes(img.raw_data)))
        x0, y0 = (RENDER_W - 1) / 2.0 - CU, (RENDER_H - 1) / 2.0 - CV
        self.crop = (slice(max(0, int(y0 / 2)), int((y0 + OUT_H) / 2)), slice(max(0, int(x0 / 2)), int((x0 + OUT_W) / 2)))

    def _kind(self, world, aid):
        k = self.kinds.get(aid, 0)
        if k == 0:
            a = world.get_actor(aid)
            k = None
            if a is not None and aid != self.hero:
                code = next((c for p, c in KIND.items() if a.type_id.startswith(p)), None)
                if code is not None:
                    b = a.bounding_box
                    k = (code, a.type_id, a.attributes.get("role_name", ""),
                         [b.location.x, b.location.y, b.location.z, b.extent.x, b.extent.y, b.extent.z])
            self.kinds[aid] = k
        return k

    def _record(self, k):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from srunner.scenariomanager.timer import GameTime
        world = CarlaDataProvider.get_world()
        snap = world.get_snapshot()
        hero = CarlaDataProvider.get_hero_actor()
        if hero is None:
            return
        self.hero = hero.id
        hs = snap.find(hero.id)
        if hs is None:
            return
        ht, hv = hs.get_transform(), hs.get_velocity()
        hx, hy = ht.location.x, ht.location.y
        self.ego.append((k, GameTime.get_time(), hx, hy, ht.location.z, ht.rotation.yaw, math.hypot(hv.x, hv.y)))
        r2 = ACTOR_RADIUS_M ** 2
        scen = {i for s in self.scen for i in s["ids"]}
        vis_rows = []
        for s in snap:
            if s.id == hero.id:
                continue
            kd = self._kind(world, s.id)
            if kd is None:
                continue
            tf = s.get_transform()
            if (tf.location.x - hx) ** 2 + (tf.location.y - hy) ** 2 > r2 or tf.location.z < ht.location.z - 50.0:
                continue
            v = s.get_velocity()
            self.act.append((k, s.id, tf.location.x, tf.location.y, tf.rotation.yaw, math.hypot(v.x, v.y), kd[0]))
            if self.cam is not None and s.id in scen:
                vis_rows.append((s.id, tf))
        if self.cam is not None:
            self.pending[snap.frame] = (k, vis_rows)
            self._visibility()
        if k - self.last_flush >= FLUSH_TICKS:
            self.flush()

    def _visibility(self):
        """P5 recorder's rule (scripts/p5_pair_agent.py _visibility), scenario actors only: project the actor's box into
        the segmentation view, count the pixels of its semantic class that belong to the box's dominant instance."""
        for f in sorted(self.pending):
            raw = self.seg_buf.pop(f, None)
            if raw is None:
                if f < max(self.seg_buf, default=-1) or len(self.pending) > 40:
                    self.pending.pop(f)          # no image for that frame (camera ticks at 5 Hz)
                continue
            k, rows = self.pending.pop(f)
            self.vis_ticks.append(k)
            W, H = RENDER_W // 2, RENDER_H // 2
            img = np.frombuffer(raw, np.uint8).reshape(H, W, 4)
            tag = img[..., 2]
            inst = img[..., 1].astype(np.int32) + 256 * img[..., 0].astype(np.int32)
            ys, xs = self.crop
            M = np.array(self.cam.get_transform().get_inverse_matrix())
            fs = W / 2.0 / math.tan(math.radians(self.fov) / 2.0)
            cam = self.cam.get_location()
            for aid, tf in rows:
                if (tf.location.x - cam.x) ** 2 + (tf.location.y - cam.y) ** 2 > VIS_RADIUS_M ** 2:
                    continue
                code, tid, _, bb = self.kinds[aid]
                box = carla.BoundingBox(carla.Location(*bb[:3]), carla.Vector3D(*bb[3:]))
                corners = np.array([[q.x, q.y, q.z] for q in box.get_world_vertices(tf)])
                p = (M @ np.c_[corners, np.ones(len(corners))].T)[:3]
                if (p[0] <= 0.3).all():
                    continue
                p = p[:, p[0] > 0.3]
                u, v = W / 2.0 + fs * p[1] / p[0], H / 2.0 - fs * p[2] / p[0]
                u0, u1 = max(xs.start, int(u.min())), min(xs.stop, int(math.ceil(u.max())) + 1)
                v0, v1 = max(ys.start, int(v.min())), min(ys.stop, int(math.ceil(v.max())) + 1)
                if u0 >= u1 or v0 >= v1:
                    continue
                classes = (12,) if code == 1 else (20, 21, 22) if code == 2 else (13, 14, 15, 16, 17, 18, 19)
                m = np.isin(tag[v0:v1, u0:u1], classes)
                if m.any():
                    _, n = np.unique(inst[v0:v1, u0:u1][m], return_counts=True)
                    self.vrows.append((k, aid, int(n.max())))

    def _write_meta(self):
        meta = {"variant": self.world_name, "attrs": self.attrs, "hero": self.hero, "route_xy": self.route,
                "configs": getattr(self, "configs", []), "built": self.scen, "shifted": SHIFTED, "swapped": SWAPPED,
                "vis": self.vis, "stopped": self.stopped, "stop_m": self.stop_m, "stall_s": self.stall_s, "kinds": {str(i): list(v[1:3]) for i, v in self.kinds.items() if v},
                "extents": {str(i): v[3] for i, v in self.kinds.items() if v}}
        tmp = os.path.join(self.out, "nq4_meta.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp, os.path.join(self.out, "nq4_meta.json"))

    def flush(self):
        if not self.ego:
            return
        self.last_flush = self.ego[-1][0]
        tmp = os.path.join(self.out, "nq4_trace.tmp.npz")
        np.savez_compressed(tmp, ego=np.asarray(self.ego, np.float64).reshape(-1, 7),
                 act=np.asarray(self.act, np.float32).reshape(-1, 7) if self.act else np.zeros((0, 7), np.float32),
                 vis=np.asarray(self.vrows, np.int64).reshape(-1, 3), vis_ticks=np.asarray(self.vis_ticks, np.int64))
        os.replace(tmp, os.path.join(self.out, "nq4_trace.npz"))
        self._write_meta()
