"""In-place rewind of a running Bench2Drive route (todos/2026-09-29-carla-rewind.md).

A WL fork runs its prefix once, snapshots the world at tick k-1, runs one 3 s branch, puts the world back and runs the
next branch in the same route. The snapshot is taken at the end of the agent call of tick k-1; the rewind happens in
the agent call after a branch ends (the "dead tick"): the world, and for the tree methods the scenario's Python state,
are set to the snapshot, the agent returns the snapshot's expert control, and the leaderboard then ticks the scenario
tree and the world as it did at tick k-1, so the next tick is the branch's tick k.

Methods (what is put back; every method also restores the ego and the agent's own controller):
  poc       the external proposal (research/carla-rewind-branching.md): ego + vehicles within 50 m, transform and
            linear / angular velocity. Nothing else.
  teleport  every vehicle, walker and static prop (transform, velocities), walker controls, traffic-light states.
  tree      teleport + the scenario tree (every node's state, BackgroundActivity's tables), the py_trees blackboard,
            CarlaDataProvider's caches and random stream, GameTime; actors spawned during the branch are destroyed and
            actors destroyed during it (a hazard walker that finished its walk) are respawned and re-bound. During the
            dead tick, Python reads of actor transforms / velocities return the snapshot (the client's cache still
            holds the branch's last state until the next world tick).
  respawn   tree, but every background vehicle is destroyed and respawned from the snapshot and handed back to the
            traffic manager (its per-vehicle TM settings are not restored).
Not restorable through the Python API (known gaps): traffic-light phase timers, the traffic manager's internal state
(path buffers, PID integrators, random stream), wheel spin / gear / suspension, animation state.
"""
import enum
import time

import carla
import numpy as np
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

METHODS = ("poc", "teleport", "tree", "respawn")
POC_RADIUS_M = 50.0
MOVING = ("vehicle.", "walker.pedestrian", "static.prop.")
_MANAGER = {}


def install():
    """Capture the leaderboard's ScenarioManager at load_scenario (the scenario tree's owner)."""
    from leaderboard.scenarios.scenario_manager import ScenarioManager
    if getattr(ScenarioManager, "_rewind_patched", False):
        return
    inner = ScenarioManager.load_scenario

    def load(self, *args, **kwargs):
        _MANAGER["m"] = self
        return inner(self, *args, **kwargs)

    ScenarioManager.load_scenario = load
    ScenarioManager._rewind_patched = True


# ---------------------------------------------------------------- Python-state freeze / thaw (in place)

_VALUE = (int, float, bool, str, bytes, complex, type(None), enum.Enum, np.generic)
_COPY = {
    carla.Location: lambda v: carla.Location(v.x, v.y, v.z),
    carla.Vector3D: lambda v: carla.Vector3D(v.x, v.y, v.z),
    carla.Rotation: lambda v: carla.Rotation(v.pitch, v.yaw, v.roll),
    carla.Transform: lambda v: carla.Transform(carla.Location(v.location.x, v.location.y, v.location.z),
                                               carla.Rotation(v.rotation.pitch, v.rotation.yaw, v.rotation.roll)),
    carla.VehicleControl: lambda v: carla.VehicleControl(v.throttle, v.steer, v.brake, v.hand_brake, v.reverse,
                                                         v.manual_gear_shift, v.gear),
    carla.WalkerControl: lambda v: carla.WalkerControl(carla.Vector3D(v.direction.x, v.direction.y, v.direction.z),
                                                       v.speed, v.jump),
}
_PREFIX = ("srunner.", "leaderboard.scenarios", "leaderboard.utils", "agents.navigation", "py_trees.")
_KEEP_CLASSES = {"ScenarioManager", "AgentWrapper", "SensorInterface", "GlobalRoutePlanner", "Watchdog",
                 "StatisticsManager", "LeaderboardEvaluator", "RouteIndexer", "CallBack"}
_SKIP_KEYS = {"iterator", "logger", "_logger"}


class _C:          # a carla value, copied
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


class _Seq:        # list / deque / set / dict restored in place (identity kept for shared references)
    __slots__ = ("obj", "items")

    def __init__(self, obj, items):
        self.obj, self.items = obj, items


class _Tup:
    __slots__ = ("items", "cls")

    def __init__(self, items, cls):
        self.items, self.cls = items, cls


class _Obj:        # a Python object whose __dict__ is restored in place
    __slots__ = ("obj", "d")

    def __init__(self, obj, d):
        self.obj, self.d = obj, d


class _Rng:
    __slots__ = ("obj", "state")

    def __init__(self, obj):
        self.obj, self.state = obj, obj.get_state()


class _Ref:        # an object already frozen elsewhere in this snapshot: restored once, referenced here
    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


def _ours(v) -> bool:
    c = type(v)
    return hasattr(v, "__dict__") and c.__module__.startswith(_PREFIX) and c.__name__ not in _KEEP_CLASSES


def freeze(v, seen: dict):
    if isinstance(v, _VALUE):
        return v
    f = _COPY.get(type(v))
    if f is not None:
        return _C(f(v))
    i = id(v)
    if i in seen:
        return _Ref(v)
    if isinstance(v, np.ndarray):
        return _C(v.copy())
    if isinstance(v, np.random.RandomState):
        seen[i] = True
        return _Rng(v)
    if isinstance(v, (list, set)) or type(v).__name__ == "deque" or isinstance(v, dict):
        seen[i] = True
        items = ([(freeze(k, seen), freeze(x, seen)) for k, x in v.items()] if isinstance(v, dict)
                 else [freeze(x, seen) for x in v])
        return _Seq(v, items)
    if isinstance(v, tuple):
        return _Tup([freeze(x, seen) for x in v], type(v))
    if _ours(v):
        seen[i] = True
        return _Obj(v, {k: freeze(x, seen) for k, x in vars(v).items() if k not in _SKIP_KEYS})
    return v          # carla actors / world / map, generators, locks, foreign objects: the reference itself


def thaw(f, remap: dict, done: set):
    t = type(f)
    if t is _C:
        v = f.v
        return v.copy() if isinstance(v, np.ndarray) else _COPY[type(v)](v)
    if t is _Ref:
        return f.obj
    if t is _Seq:
        o = f.obj
        if id(o) not in done:
            done.add(id(o))
            if isinstance(o, dict):
                items = [(thaw(k, remap, done), thaw(x, remap, done)) for k, x in f.items]
                o.clear()
                o.update(items)
            else:
                items = [thaw(x, remap, done) for x in f.items]
                o.clear()
                (o.update if isinstance(o, set) else o.extend)(items)
        return o
    if t is _Tup:
        items = [thaw(x, remap, done) for x in f.items]
        return f.cls(*items) if hasattr(f.cls, "_fields") else f.cls(items)
    if t is _Obj:
        o = f.obj
        if id(o) not in done:
            done.add(id(o))
            d = vars(o)
            for k in [k for k in d if k not in f.d and k not in _SKIP_KEYS]:
                del d[k]
            for k, x in f.d.items():
                d[k] = thaw(x, remap, done)
        return o
    if t is _Rng:
        f.obj.set_state(f.state)
        return f.obj
    if isinstance(f, carla.Actor) and f.id in remap:
        return remap[f.id]
    return f


# ---------------------------------------------------------------- snapshot / restore

_READS = ("get_transform", "get_location", "get_velocity", "get_angular_velocity", "get_acceleration")
_ORIG = {k: getattr(carla.Actor, k) for k in _READS}


class Rewinder:
    def __init__(self, method: str, world, hero, tm_port: int):
        assert method in METHODS, method
        self.method, self.world, self.hero, self.tm_port = method, world, hero, tm_port
        self.snap, self.remap, self.patched = None, {}, False
        self.stats = []

    # -- world state
    def _actor_state(self, a):
        tf, v, w = a.get_transform(), a.get_velocity(), a.get_angular_velocity()
        s = {"actor": a, "id": a.id, "type": a.type_id, "attrs": dict(a.attributes), "tf": _COPY[carla.Transform](tf),
             "v": carla.Vector3D(v.x, v.y, v.z), "w": carla.Vector3D(w.x, w.y, w.z),
             "loc": carla.Location(tf.location.x, tf.location.y, tf.location.z),
             "acc": _COPY[carla.Vector3D](a.get_acceleration())}
        if a.type_id.startswith("walker.pedestrian"):
            s["ctl"] = _COPY[carla.WalkerControl](a.get_control())
        return s

    def snapshot(self, t_game: float, control, extra=None):
        t0 = time.perf_counter()
        hl = self.hero.get_location()
        acts = [a for a in self.world.get_actors() if a.type_id.startswith(MOVING) and a.id != self.hero.id]
        if self.method == "poc":
            acts = [a for a in acts if a.type_id.startswith("vehicle.") and a.get_location().distance(hl) <= POC_RADIUS_M]
        snap = {"t": t_game, "control": _COPY[carla.VehicleControl](control), "hero": self._actor_state(self.hero),
                "actors": {a.id: self._actor_state(a) for a in acts}, "extra": extra}
        if self.method != "poc":
            snap["lights"] = [(tl, tl.get_state()) for tl in self.world.get_actors().filter("traffic.traffic_light")]
        if self.method in ("tree", "respawn"):
            seen = {}
            m = _MANAGER.get("m")
            snap["tree"] = freeze(m.scenario_tree, seen) if m is not None else None
            snap["bb"] = freeze(py_trees.blackboard.Blackboard._Blackboard__shared_state, seen)
            cdp = {k: getattr(CarlaDataProvider, k) for k in ("_actor_velocity_map", "_actor_location_map",
                                                               "_actor_transform_map", "_carla_actor_pool", "_rng",
                                                               "active_scenarios") if hasattr(CarlaDataProvider, k)}
            snap["cdp"] = freeze(cdp, seen)
            snap["spawn_index"] = CarlaDataProvider._spawn_index
        self.snap = snap
        return {"snapshot_ms": 1e3 * (time.perf_counter() - t0), "n_actors": len(snap["actors"])}

    def _set(self, a, s, physics=True):
        a.set_transform(s["tf"])
        if physics:
            a.set_target_velocity(s["v"])
            a.set_target_angular_velocity(s["w"])
        if "ctl" in s:
            a.apply_control(s["ctl"])

    def _spawn(self, s):
        lib = self.world.get_blueprint_library()
        bp = lib.find(s["type"])
        for k, v in s["attrs"].items():
            if bp.has_attribute(k) and (k == "role_name" or bp.get_attribute(k).is_modifiable):
                bp.set_attribute(k, v)
        tf = s["tf"]
        up = carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + 200.0), tf.rotation)
        a = self.world.spawn_actor(bp, up)
        self._set(a, s)
        if s["type"].startswith("vehicle.") and s["attrs"].get("role_name") == "background":
            a.set_autopilot(True, self.tm_port)
        return a

    def restore(self) -> dict:
        """Called in the dead tick's agent call. Returns stats; the caller returns snap['control']."""
        t0 = time.perf_counter()
        s = self.snap
        live = {a.id: a for a in self.world.get_actors() if a.type_id.startswith(MOVING) and a.id != self.hero.id}
        n_new = n_resp = 0
        self._set(self.hero, s["hero"])
        full = self.method in ("tree", "respawn")
        if full:
            for aid, a in live.items():            # spawned during the branch: not part of the fork state
                if aid not in s["actors"] and aid not in {x.id for x in self.remap.values()}:
                    a.destroy()
                    n_new += 1
        for aid, st in s["actors"].items():
            cur = self.remap.get(aid)
            a = cur if cur is not None and cur.is_alive else live.get(aid)
            bg = st["type"].startswith("vehicle.") and st["attrs"].get("role_name") == "background"
            if full and a is not None and self.method == "respawn" and bg:
                a.destroy()
                a = None
            if a is None or not a.is_alive:
                if not full:
                    continue
                a = self._spawn(st)
                self.remap[aid] = a
                n_resp += 1
            else:
                self._set(a, st)
        if self.method != "poc":
            for tl, state in s["lights"]:
                tl.set_state(state)
        if full:
            done = set()
            if s["tree"] is not None:
                thaw(s["tree"], self.remap, done)
            thaw(s["bb"], self.remap, done)
            thaw(s["cdp"], self.remap, done)
            pool = CarlaDataProvider._carla_actor_pool
            for old, new in self.remap.items():
                pool.pop(old, None)
                pool[new.id] = new
            CarlaDataProvider._spawn_index = s["spawn_index"]
            CarlaDataProvider._all_actors = None
            GameTime._current_game_time = s["t"]
            self._patch_reads()
        st = {"restore_ms": 1e3 * (time.perf_counter() - t0), "destroyed_new": n_new, "respawned": n_resp,
              "remapped": len(self.remap)}
        self.stats.append(st)
        return st

    # -- dead-tick reads: the client cache holds the branch's last state until the next world tick
    def _patch_reads(self):
        by_id = {st["id"]: st for st in self.snap["actors"].values()}
        by_id[self.hero.id] = self.snap["hero"]
        for old, new in self.remap.items():
            if old in by_id:
                by_id[new.id] = by_id[old]
        key = {"get_transform": "tf", "get_location": "loc", "get_velocity": "v", "get_angular_velocity": "w",
               "get_acceleration": "acc"}

        def make(name):
            orig, k = _ORIG[name], key[name]

            def read(actor):
                s = by_id.get(actor.id)
                return orig(actor) if s is None else _COPY[type(s[k])](s[k])
            return read
        for name in _READS:
            setattr(carla.Actor, name, make(name))
        self.patched = True

    def unpatch(self):
        if self.patched:
            for name, f in _ORIG.items():
                setattr(carla.Actor, name, f)
            self.patched = False
