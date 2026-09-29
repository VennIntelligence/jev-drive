"""Route setup phases and same-map world reuse for scripts/b2d_route.py (todos/2026-09-29-carla-rewind.md).

Both off by default; b2d_route installs this when either env var is set.
  B2D_PHASES=1     write <attempt>/phases.json: seconds from process start to the evaluator's world load, the world
                   load (or in-place reset), the RouteScenario build, and whether the map was reused
  B2D_REUSE_MAP=1  when the server already runs the route's town, skip client.load_world: destroy what the previous
                   route left (vehicles, walkers, walker controllers, sensors, spawned static props), tick once, and
                   continue with the evaluator's own steps (Large Map stream distances, reset_all_traffic_lights,
                   CarlaDataProvider, TM seed, tick, map check). TM, GameTime and CarlaDataProvider are rebuilt by
                   every route's new process anyway; the scenario sets the weather.
"""
import json
import os
import time
from pathlib import Path

LEFTOVER = ("vehicle.", "walker.", "controller.", "sensor.", "static.prop.")


def _proc_start() -> float:
    """Wall time this process started (Linux: /proc/self/stat field 22, clock ticks since boot)."""
    try:
        ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        boot = next(float(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return float("nan")


def install(out_dir: str, reuse: bool):
    import carla
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.scenarios.route_scenario import RouteScenario
    ph = {"proc_start": _proc_start(), "reuse_enabled": reuse}
    path = Path(out_dir) / "phases.json"

    def dump():
        path.write_text(json.dumps(ph, indent=1))

    inner_load = LeaderboardEvaluator._load_and_wait_for_world

    def load(self, args, town):
        t0 = time.time()
        ph["t_load_call"] = t0
        world = self.client.get_world()
        cur = world.get_map().name.split("/")[-1]
        ph.update(server_map=cur, town=town, reused=bool(reuse and cur == town))
        if ph["reused"]:
            left = [a for a in world.get_actors() if a.type_id.startswith(LEFTOVER)]
            ph["leftover"] = {k: sum(a.type_id.startswith(k) for a in left) for k in LEFTOVER}
            for a in left:
                if a.type_id.startswith("sensor."):
                    try:
                        a.stop()
                    except RuntimeError:
                        pass
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in left], True)
            world.tick()
            client = self.client

            class _Same:            # the evaluator's load_world call returns the world already running
                def load_world(self, *_a, **_k):
                    return world

                def __getattr__(self, k):
                    return getattr(client, k)

            self.client = _Same()
            try:
                inner_load(self, args, town)
            finally:
                self.client = client
        else:
            inner_load(self, args, town)
        ph["load_s"] = time.time() - t0
        dump()

    LeaderboardEvaluator._load_and_wait_for_world = load
    inner_init = RouteScenario.__init__

    def init(self, *a, **k):
        t0 = time.time()
        inner_init(self, *a, **k)
        ph["scenario_build_s"] = time.time() - t0
        ph["t_scenario_done"] = time.time()
        dump()

    RouteScenario.__init__ = init
