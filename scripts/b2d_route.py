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

Exit codes: 0 route finished (whatever its score), 1 route failed, 2 harness/setup failure,
128 + signal number if interrupted.
Python 3.8: runs in envs/carla.
"""
import sys

if sys.version_info >= (3, 9) and "xml.etree.ElementTree" not in sys.modules:
    # Bench2Drive 0.0.4's route parser calls Element.getchildren(), which Python 3.9 removed. The C Element
    # cannot take a new method, so model venvs on 3.10 (TFv6) use the pure-Python ElementTree with it restored,
    # and with items() returning a list as the C Element's does (the evaluator indexes it: case.items()[0][1]).
    sys.modules["_elementtree"] = None
    import xml.etree.ElementTree as _ET
    _ET.Element.getchildren = lambda self: list(self)
    _ET.Element.items = lambda self: list(self.attrib.items())

import argparse
import json
import math
import os
import signal
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


def parse_args(argv=None):
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
    p.add_argument("--agent", default="", help="external leaderboard agent .py (default: cost stub)")
    p.add_argument("--agent-config", default="", help="external agent config/checkpoint path")
    p.add_argument("--record-dir", default="", help="optional CARLA recorder directory")
    p.add_argument("--rig", default="front3", choices=["none", "front1", "front3", "b2d6"])
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--policy", default="none", choices=["none", "sleep", "gpu"])
    p.add_argument("--infer-ms", type=float, default=0.0)
    p.add_argument("--policy-socket", default="")
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--controller-preset", default="carla", choices=["carla", "tcp", "pursuit"])
    p.add_argument("--cruise-mps", type=float, default=8.0)
    p.add_argument("--controller-config", default="", help="controller parameter JSON path")
    p.add_argument("--drive", default="route", choices=["straight", "route", "controller"])
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
    a = p.parse_args(argv)
    if not math.isfinite(a.cruise_mps) or a.cruise_mps <= 0:
        p.error("--cruise-mps must be finite and positive")
    if a.decimate < 1:
        p.error("--decimate must be positive")
    if a.drive == "controller" and (a.policy != "none" or a.agent):
        p.error("--drive controller currently requires --policy none and the built-in agent")
    if a.controller_config:
        path = Path(a.controller_config).resolve()
        try:
            params = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            p.error("invalid --controller-config: %s" % exc)
        if not isinstance(params, dict):
            p.error("--controller-config must contain a JSON object")
        a.controller_config = str(path)
    return a


def main():
    a = parse_args()
    a.routes, a.out = str(Path(a.routes).resolve()), str(Path(a.out).resolve())
    if a.agent:
        a.agent = str(Path(a.agent).resolve())
        if a.agent_config:
            a.agent_config = str(Path(a.agent_config).resolve())
    add_bench2drive_to_path(a.bench2drive)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # External experiment agents can keep telemetry with this exact attempt.
    os.environ['B2D_ATTEMPT_OUT'] = str(out)
    os.environ['BENCHMARK_ROUTE_ID'] = str(a.route_id)

    import carla  # noqa: F401  (import after the path is set up, so the wheel is the one used)
    import b2d_hooks
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
    from leaderboard.utils.statistics_manager import StatisticsManager
    from leaderboard.scenarios.scenario_manager import ScenarioManager

    cfg = {"rig": a.rig, "width": a.width, "height": a.height, "policy": a.policy,
           "infer_ms": a.infer_ms, "decimate": a.decimate, "overlap": a.overlap,
           "drive": a.drive, "policy_socket": a.policy_socket,
           "controller_preset": a.controller_preset, "cruise_mps": a.cruise_mps,
           "controller_config": a.controller_config, "tm_seed": a.tm_seed, "out": a.out}
    cfg_path = out / "agent_config.json"
    cfg_path.write_text(json.dumps(cfg))

    profile = b2d_hooks.TickProfile(heartbeat_path=str(out / "heartbeat.json"))
    b2d_hooks.install(profile, no_spectator=a.no_spectator, fast_copy=a.fast_copy,
                      zero_copy=a.zero_copy, sensor_tick=a.decimate > 1,
                      cache_lights=a.cache_lights)
    if os.environ.get("B2D_RESEED_AFTER_BUILD") == "1":
        b2d_hooks.reseed_after_build(a.tm_seed)
    suppress = _route_attr(a.routes, a.route_id, "p5_suppress")
    if suppress is not None:
        b2d_hooks.track_hazards(a.out, hide=suppress == "1")
    _patch_setup_simulation(LeaderboardEvaluator, a)
    _patch_signal_handler(LeaderboardEvaluator)
    if a.max_ticks:
        _patch_tick_limit(ScenarioManager, a.max_ticks)

    args = _leaderboard_args(a, cfg_path, out)
    stats = StatisticsManager(args.checkpoint, args.debug_checkpoint)
    if a.drive == "controller" or os.environ.get('B2D_CAPTURE_CRITERION_EVENTS') == '1':
        _capture_criterion_events(stats, out)
    t0 = time.time()
    record = {"route_id": a.route_id, "status": "harness_error", "wall_s": 0.0}
    rc = 2
    prof = None
    evaluator = None
    if a.cprofile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    try:
        evaluator = LeaderboardEvaluator(args, stats)
        signal.signal(signal.SIGTERM, evaluator._signal_handler)
        crashed = evaluator.run(args)
        interrupted = getattr(evaluator, "_jev_interrupt_signal", None)
        if interrupted is not None:
            record["status"] = "cancelled_by_user"
            record["interrupt_signal"] = interrupted
            rc = 128 + interrupted
        else:
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
        record["tick_cap_configured"] = a.max_ticks
        record["capped"] = bool(a.max_ticks and profile.ticks >= a.max_ticks)
        if evaluator is not None:
            record["sensors"] = evaluator.sensors
        if not a.agent:
            import b2d_agent
            agent = b2d_agent.LAST_AGENT
            if agent is not None:
                record["agent"] = _agent_summary(agent, a.drop_ticks)
        (out / "route_result.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"route_id": a.route_id, "status": record["status"],
                          "wall_s": record["wall_s"], "ticks": profile.ticks}), flush=True)
    return rc


def _route_attr(routes, route_id, key):
    import xml.etree.ElementTree as ET
    for r in ET.parse(routes).getroot().iter("route"):
        if r.get("id") == route_id:
            return r.get(key)
    return None


def _capture_criterion_events(stats, out):
    """Snapshot existing evaluator events with frames; never alter criteria or control."""
    original = stats.compute_route_statistics

    def compute(*args, **kwargs):
        try:
            events = []
            scenario = getattr(stats, "_scenario", None)
            for node in scenario.get_criteria() if scenario is not None else []:
                for event in node.events:
                    data = event.get_dict() or {}
                    events.append({"type": event.get_type().name, "frame": event.get_frame(),
                                   "message": event.get_message(),
                                   "percentage": data.get("percentage")})
            (out / "criterion_events.json").write_text(json.dumps({"state": "ok", "events": events}))
        except Exception as exc:
            # Diagnostic serialization must never turn a driving result into a runtime failure.
            print("criterion event snapshot failed: %s" % exc, flush=True)
        return original(*args, **kwargs)

    stats.compute_route_statistics = compute


def _patch_signal_handler(evaluator_class):
    # Bench2Drive's handler stops the scenario normally and run() then returns False,
    # just as it does at a natural end. Preserve the signal so it cannot mean "finished".
    original = evaluator_class._signal_handler

    def interrupted(self, signum, frame):
        self._jev_interrupt_signal = signum
        return original(self, signum, frame)

    evaluator_class._signal_handler = interrupted


def _agent_summary(agent, drop):
    out = {"policy_ticks": agent.timings["policy_ticks"]}
    for key in ("sensor_wait", "infer", "agent_total"):
        xs = sorted(agent.timings[key][drop:])
        if xs:
            out[key + "_ms_mean"] = round(1e3 * sum(xs) / len(xs), 3)
            out[key + "_ms_median"] = round(1e3 * xs[len(xs) // 2], 3)
    xs = agent.timings.get("server_infer_ms") or []
    # policy_ticks counts submissions; this counts inferences that actually came back. They differ
    # when the policy is slower than its decimation period and a frame is skipped, and a zero here
    # against a non-zero policy_ticks means the policy never ran at all.
    out["policy_done"] = len(xs)
    if xs:
        out["server_infer_ms_mean"] = round(sum(xs) / len(xs), 2)
    return out


def _leaderboard_args(a, cfg_path, out):
    ns = argparse.Namespace(
        host="127.0.0.1", port=a.port, traffic_manager_port=a.tm_port, traffic_manager_seed=a.tm_seed,
        debug=0, record=a.record_dir, timeout=a.timeout, routes=a.routes, routes_subset=a.route_id,
        repetitions=1, agent=a.agent or str(Path(__file__).resolve().parent / "b2d_agent.py"),
        agent_config=a.agent_config if a.agent else str(cfg_path), track="SENSORS", resume=False,
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
