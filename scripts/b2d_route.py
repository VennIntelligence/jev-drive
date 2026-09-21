#!/usr/bin/env python
"""Run exactly one Bench2Drive route against an already-running CARLA server, with a per-phase
profile. See research/carla-efficiency.md.

One route per process is the unit of isolation: a route that segfaults the server takes down this
process and nothing else (R1), and the result file it leaves behind is what makes resume work (R2).
b2d_run.py drives this; running it by hand is how you profile a single route.

    $DATA_DIR/envs/carla/bin/python scripts/b2d_route.py \
        --routes $B2D/leaderboard/data/bench2drive220.xml --route-id 24240 \
        --port 2000 --tm-port 8000 --out $DATA_DIR/runs/b2d/probe \
        --rig front3 --policy sleep --infer-ms 129

Exit codes: 0 route finished (whatever its score), 1 route failed, 2 harness/setup failure.
Python 3.8: runs in envs/carla.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def add_bench2drive_to_path(root):
    carla_root = os.environ.get(
        "CARLA_ROOT", str(Path(os.environ["DATA_DIR"]) / "third_party/carla/CARLA_0.9.15"))
    os.environ["CARLA_ROOT"] = carla_root
    # scenario_runner imports `agents.navigation...`, which ships inside CARLA's PythonAPI, not in
    # the pip wheel. The wheel still provides the `carla` module itself.
    for p in (str(Path(carla_root) / "PythonAPI" / "carla"),
              str(Path(root) / "scenario_runner"), str(Path(root) / "leaderboard"),
              str(Path(__file__).resolve().parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("SCENARIO_RUNNER_ROOT", str(Path(root) / "scenario_runner"))
    os.environ.setdefault("LEADERBOARD_ROOT", str(Path(root) / "leaderboard"))
    # The evaluator reads 'leaderboard/data/weather.xml' by a relative path, so it only runs from
    # the Bench2Drive root. Every path we hand it is made absolute first.
    os.chdir(root)


def parse_args():
    p = argparse.ArgumentParser()
    b2d_default = str(Path(os.environ.get("DATA_DIR", "")) / "third_party/Bench2Drive")
    p.add_argument("--bench2drive", default=os.environ.get("BENCH2DRIVE_ROOT", b2d_default))
    p.add_argument("--routes", required=True)
    p.add_argument("--route-id", required=True)
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--tm-seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", required=True, help="directory for this attempt's outputs")
    # agent / policy
    p.add_argument("--rig", default="front3", choices=["none", "front1", "front3", "b2d6"])
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--policy", default="none", choices=["none", "sleep", "gpu"])
    p.add_argument("--infer-ms", type=float, default=0.0)
    p.add_argument("--policy-socket", default="")
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--drive", default="route", choices=["straight", "route"])
    # optimisation flags, all off by default so the unoptimised path stays the reference
    p.add_argument("--no-spectator", action="store_true")
    p.add_argument("--fast-copy", action="store_true")
    p.add_argument("--zero-copy", action="store_true")
    p.add_argument("--cache-lights", action="store_true")
    # profiling
    p.add_argument("--max-ticks", type=int, default=0, help="stop the route early, for profiling")
    p.add_argument("--drop-ticks", type=int, default=20, help="warmup ticks excluded from the profile")
    p.add_argument("--cprofile", action="store_true",
                   help="also write a cProfile of the whole run. Use it to attribute the Python "
                        "phases to functions, never for absolute timings: the profiler itself "
                        "roughly doubles per-tick Python cost.")
    return p.parse_args()


def main():
    a = parse_args()
    a.routes, a.out = str(Path(a.routes).resolve()), str(Path(a.out).resolve())
    add_bench2drive_to_path(a.bench2drive)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import carla  # noqa: F401  (import after the path is set up, so the wheel is the one used)
    import b2d_hooks
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.utils.statistics_manager import StatisticsManager
    from leaderboard.scenarios.scenario_manager import ScenarioManager

    cfg = {"rig": a.rig, "width": a.width, "height": a.height, "policy": a.policy,
           "infer_ms": a.infer_ms, "decimate": a.decimate, "overlap": a.overlap,
           "drive": a.drive, "policy_socket": a.policy_socket}
    cfg_path = out / "agent_config.json"
    cfg_path.write_text(json.dumps(cfg))

    profile = b2d_hooks.TickProfile(heartbeat_path=str(out / "heartbeat.json"))
    b2d_hooks.install(profile, no_spectator=a.no_spectator, fast_copy=a.fast_copy,
                      zero_copy=a.zero_copy, sensor_tick=a.decimate > 1,
                      cache_lights=a.cache_lights)
    _patch_setup_simulation(LeaderboardEvaluator, a)
    if a.max_ticks:
        _patch_tick_limit(ScenarioManager, a.max_ticks)

    args = _leaderboard_args(a, cfg_path, out)
    stats = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    t0 = time.time()
    record = {"route_id": a.route_id, "status": "harness_error", "wall_s": 0.0}
    rc = 2
    prof = None
    if a.cprofile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    try:
        evaluator = LeaderboardEvaluator(args, stats)
        crashed = evaluator.run(args)
        record["status"] = "crashed" if crashed else "finished"
        rc = 1 if crashed else 0
    except SystemExit as e:  # the evaluator's own early exits
        record["status"] = "exit_%s" % e.code
        rc = 1
    except Exception:
        record["traceback"] = traceback.format_exc()
        print(record["traceback"], flush=True)
        rc = 2
    finally:
        if prof is not None:
            import pstats
            prof.disable()
            with open(str(out / "cprofile.txt"), "w") as fh:
                pstats.Stats(prof, stream=fh).sort_stats("tottime").print_stats(45)
        record["wall_s"] = round(time.time() - t0, 1)
        record["profile"] = profile.summary(drop=a.drop_ticks)
        record["config"] = vars(a)
        import b2d_agent
        agent = b2d_agent.LAST_AGENT
        if agent is not None:
            record["agent"] = _agent_summary(agent, a.drop_ticks)
        (out / "route_result.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"route_id": a.route_id, "status": record["status"],
                          "wall_s": record["wall_s"], "ticks": profile.ticks}), flush=True)
    return rc


def _agent_summary(agent, drop):
    out = {"policy_ticks": agent.timings["policy_ticks"]}
    for key in ("sensor_wait", "infer", "agent_total"):
        xs = sorted(agent.timings[key][drop:])
        if xs:
            out[key + "_ms_mean"] = round(1e3 * sum(xs) / len(xs), 3)
            out[key + "_ms_median"] = round(1e3 * xs[len(xs) // 2], 3)
    xs = agent.timings.get("server_infer_ms") or []
    if xs:
        out["server_infer_ms_mean"] = round(sum(xs) / len(xs), 2)
    return out


def _leaderboard_args(a, cfg_path, out):
    ns = argparse.Namespace(
        host="127.0.0.1", port=a.port, traffic_manager_port=a.tm_port, traffic_manager_seed=a.tm_seed,
        debug=0, record="", timeout=a.timeout, routes=a.routes, routes_subset=a.route_id,
        repetitions=1, agent=str(Path(__file__).resolve().parent / "b2d_agent.py"),
        agent_config=str(cfg_path), track="SENSORS", resume=False,
        checkpoint=str(out / "results.json"), debug_checkpoint=str(out / "live_results.txt"),
        gpu_rank=0)
    return ns


def _patch_setup_simulation(LeaderboardEvaluator, a):
    """Bench2Drive's evaluator spawns its own CARLA server and sleeps 30 s. We want the server's
    lifetime to be owned by b2d_run.py, so that one server can serve several routes and so that a
    crashed server is restarted deliberately rather than by a process that has just died. Same
    world settings, same traffic manager settings - only the ownership of the process changes."""
    import carla

    def setup(self, args):
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        client.get_world().apply_settings(carla.WorldSettings(
            synchronous_mode=True, fixed_delta_seconds=1.0 / LeaderboardEvaluator.frame_rate,
            deterministic_ragdolls=True, spectator_as_ego=False))
        tm = client.get_trafficmanager(args.traffic_manager_port)
        tm.set_synchronous_mode(True)
        tm.set_hybrid_physics_mode(True)
        return client, args.timeout, tm

    LeaderboardEvaluator._setup_simulation = setup


def _patch_tick_limit(ScenarioManager, max_ticks):
    """Profiling runs do not need a whole route; stop after N ticks and report."""
    inner = ScenarioManager._tick_scenario

    def tick(self):
        inner(self)
        if self.tick_count >= max_ticks:
            self._running = False

    ScenarioManager._tick_scenario = tick


if __name__ == "__main__":
    sys.exit(main())
