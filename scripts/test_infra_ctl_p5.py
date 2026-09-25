"""Checks for the P5 re-acceptance wiring (todos/2026-09-25-closed-loop-infra-acceptance/b2d-controllers-p5.md).

1. The exam agent (b2d_zeroshot_agent.py, controller_preset "pursuit") builds P5 exactly as the L1 / L2 harnesses do
   (b2d_controller.pursuit_from_config; pose_lateral_coefficient_s2_per_m to the PoseFilter as b2d_agent.py):
   bit-identical controls on a synthetic 2 Hz / 1 Hz plan stream with a stop and a restart.
2. replay_plan "time": on time, lagging, ahead of schedule, during the expert's wait and past the log's end the plan is
   the expert's schedule with the lag closed at REPLAY_CATCHUP_S, never behind the car, non-decreasing.

    python scripts/test_infra_ctl_p5.py        (no CARLA needed: carla / leaderboard / srunner are stubbed)
"""
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
P5 = HERE.parent / "todos/2026-09-23-tfv6-controller/controller-eval/P5.json"


def stub_modules(hero):
    class Anything(types.ModuleType):
        def __getattr__(self, name):
            return type(name, (), {"__init__": lambda self, *a, **k: None})
    for name in ("carla", "leaderboard", "leaderboard.autoagents", "leaderboard.autoagents.autonomous_agent",
                 "srunner", "srunner.scenariomanager", "srunner.scenariomanager.timer",
                 "srunner.scenariomanager.carla_data_provider"):
        sys.modules[name] = Anything(name)
    sys.modules["leaderboard.autoagents.autonomous_agent"].AutonomousAgent = object
    sys.modules["leaderboard.autoagents.autonomous_agent"].Track = types.SimpleNamespace(SENSORS=0)
    sys.modules["srunner.scenariomanager.carla_data_provider"].CarlaDataProvider = types.SimpleNamespace(
        get_hero_actor=lambda: hero)


def check_controller():
    from b2d_controller import pursuit_from_config, Controller
    params = json.loads(P5.read_text())
    params.pop("rear_axle_offset_m")                       # as ZeroShotAgent.setup
    lateral = float(params.pop("pose_lateral_coefficient_s2_per_m", 0.0))
    agent_ctl = Controller(preset="pursuit", **params)
    ref_ctl = pursuit_from_config(P5)
    assert lateral == 0.010659832
    for period in (10, 20):                                 # 2 Hz and 1 Hz plans, 20 Hz control
        agent_ctl.reset(), ref_ctl.reset()
        rng = np.random.default_rng(period)
        v, x, out_a, out_r = 0.0, 0.0, [], []
        for k in range(1200):
            t = k * 0.05
            if k % period == 0:
                target = 0.0 if 25 < t < 32 else 6.0
                tt = np.arange(1, 21) * 0.25
                plan = np.c_[np.minimum(target, v + 2 * tt) * tt, 0.02 * tt ** 2 + rng.normal(0, .01, 20)]
                assert agent_ctl.update(plan, t) == ref_ctl.update(plan, t)
            a = agent_ctl.step(t, v, 0.01)
            b = ref_ctl.step(t, v, 0.01)
            out_a.append(a)
            out_r.append(b)
            v = max(0.0, v + 0.05 * (4 * a[0] - 8 * a[2] - 0.5))
            x += v * 0.05
        assert out_a == out_r, "controls differ"
        assert max(o[0] for o in out_a) > 0.3 and max(o[2] for o in out_a) > 0.1
    print("controller: P5 via the exam agent == pursuit_from_config (bit-identical, 2 Hz and 1 Hz)")


def check_replay():
    hero = types.SimpleNamespace()
    stub_modules(hero)
    import b2d_zeroshot_agent as za
    agent = za.ZeroShotAgent.__new__(za.ZeroShotAgent)
    agent.replay, agent.replay_plan, agent.rear_offset, agent.replay_t0 = "log", "time", 0.0, 0.0
    # Expert: straight along x at 5 m/s from t=0, stops at x=50 for t in [10, 16], then 5 m/s again, log ends at t=30.
    t = np.arange(0, 30.0001, 0.05)
    xe = np.where(t < 10, 5 * t, np.where(t < 16, 50.0, 50 + 5 * (t - 16)))
    rows = [{"t": float(a), "x": float(b), "y": 0.0, "yaw": 0.0} for a, b in zip(t, xe)]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
    agent._load_replay(fh.name)
    os.unlink(fh.name)
    times = np.arange(1, 21) * 0.25

    def plan(now, car_x):
        hero.get_transform = lambda: types.SimpleNamespace(location=types.SimpleNamespace(x=car_x, y=0.0),
                                                            rotation=types.SimpleNamespace(yaw=0.0))
        agent.replay_i = int(np.searchsorted(agent.replay_log[1], car_x))   # as if tracked from the start
        p = agent._replay_path(now, times)
        assert np.all(np.diff(p[:, 0]) >= -1e-9) and p[0, 0] >= -1e-9 and np.abs(p[:, 1]).max() < 1e-9
        return p[:, 0]

    te, xl = agent.replay_log[0], agent.replay_log[1]      # the log as loaded: + 10 s straight on past its end

    def expect(now, car_x):
        lag = np.interp(now, te, xl) - car_x
        return np.maximum.accumulate(np.maximum(np.interp(now + times, te, xl) - lag * np.exp(-times / 2), car_x)) - car_x

    cases = {"on time": (4.0, 20.0), "lagging 10 m": (4.0, 10.0), "ahead 3 m": (4.0, 23.0),
             "expert waiting, car 8 m short": (12.0, 42.0), "expert waiting, car at stop": (12.0, 50.0),
             "lagging past the log end": (31.0, 90.0), "start": (0.0, 0.0)}
    for name, (now, car) in cases.items():
        got = plan(now, car)
        assert np.allclose(got, expect(now, car), atol=1e-6), (name, got[:4], expect(now, car)[:4])
    first = plan(4.0, 10.0)[0]
    assert abs(first - (1.25 + 10 * (1 - math.exp(-0.125)))) < 1e-6          # v_e*0.25 + 0.1175 * lag, no jump
    assert plan(12.0, 42.0)[-1] > 7.9                                          # creeps up to the expert's stop point
    assert np.all(plan(12.0, 50.0)[:16] < 1e-6)                                # waits there until the expert moves on (t=16)
    assert plan(4.0, 23.0)[0] < 1.25                                           # ahead: slower than the expert
    print("replay_plan time: 7 cases match s(t) = s_e(tau+t) - lag*exp(-t/2), clipped; no freeze, no jump")


if __name__ == "__main__":
    check_controller()
    check_replay()
