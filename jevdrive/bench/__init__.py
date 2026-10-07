"""jevdrive.bench: run any registered model on any of our benchmarks with one command, and report it the standard way.
Start at docs/bench.md.

  models    registry: shipped openpilot (cinque / lebowski / small), op_guard ONNX candidates, op_parity arms (P0, P*-init, every
            checkpoint in runs/op_parity/runs), WA-JEPA; how each is loaded / served, its inputs, frame protocol, weights
  sets      unit sets: HUGSIM scenario lists (all64, spin10, spec29, small11, turn23), NAVSIM tokens and log shards
  navsim    navtest EPDMS (v2 devkit, sharded by log) and navhard two-stage (devkit aggregation, per-group harness)
  hugsim    HUGSIM 64 under the exam / spec / spec_plan presets: per-card worker jobs with own servers, shared scenario queue,
            stall watchdog + server health checks, end classes / spins / launch stalls
  b2d       CARLA / Bench2Drive: generic agent wrapper over scripts/b2d_run.py and the pool's CARLA placement
  poses     per-token devkit pdm_score + DAC diagnostics of arbitrary (N, 8, 3) pose arrays (score-poses): token-chunk claim queue
            pulled by concurrent CPU pool jobs sized from the box
  runner    stages -> GPU-pool jobs (jevdrive.cl), resumable run dirs with STATUS / DONE / ERROR
  tables    per-arm tables, paired cluster-bootstrap contrasts, Shapley EPDMS attribution, strata

CLI: python -m jevdrive.bench {models, run, status, report, sets, score-poses}. Python:

    from jevdrive import bench
    d = bench.run("P2-F-s0", "hugsim", preset="spec", scenarios="all64")         # submits; returns the run dir
    bench.wait([d])
    bench.report("hugsim", ["P2=P2-F-s0+P2-F-s1"], vs=["WA-JEPA"], preset="exam")
    d = bench.score_poses("f.npz", "o.csv", keys=["a"]); bench.wait([d])        # pose scoring on the pool
"""
from __future__ import annotations

from pathlib import Path


def run_dir(model: str, bench: str, preset: str = "exam", subset: str = "", **kw):
    from .models import resolve
    from .runner import bench_root
    m = resolve(model)
    if bench == "hugsim":
        from .hugsim import run_key
        return bench_root("hugsim", run_key(m, preset, **kw))
    if bench == "b2d":
        return bench_root("b2d", m.key("b2d"))
    return bench_root(bench, m.key(bench) + (f"_{subset.replace('/', '-')}" if subset else ""))


def plan(model: str, bench: str, preset: str = "exam", scenarios: str = "all64", subset: str = "", shards: int = 0, workers: int = 6,
         jobs: int = 0, **kw):
    """(run dir, stages) of one model on one bench, without submitting."""
    from .models import resolve
    m = resolve(model, check=True)
    if bench not in m.benches:
        raise ValueError(f"{model} is registered for {m.benches}, not {bench}")
    identity = {k: v for k, v in kw.items() if k in ("opts", "controller", "controller_env", "repeat", "onnx")}
    d = run_dir(model, bench, preset, subset, **identity)
    if bench in ("navtest", "navhard"):
        from . import navsim
        return d, navsim.stages(m, bench, d, shards=shards, subset=subset)
    if bench == "hugsim":
        from . import hugsim
        from .sets import hugsim_scenarios
        return d, hugsim.stages(m, preset, d, hugsim_scenarios(scenarios), workers=workers, jobs=jobs, **kw)
    raise ValueError(f"unknown bench {bench!r} (navtest, navhard, hugsim; CARLA: jevdrive.bench.b2d)")


def run(model: str, bench: str, dry: bool = False, priority: float = 0.0, gpus=None, **kw):
    """Submit one model x bench run to the GPU pool; returns its run dir (idempotent: finished stages are skipped)."""
    from . import runner
    d, st = plan(model, bench, **kw)
    name = f"bn-{bench}-{d.name}"
    runner.submit(d, name, st, dry=dry, priority=priority, gpus=gpus)
    if not dry:
        runner.status(d, f"submitted: {', '.join(s.name for s in st if not Path(s.done).exists()) or 'nothing left'}")
    return d


def wait(dirs, poll_s: float = 30.0, timeout_s: float = 0) -> bool:
    from .runner import wait as _w
    return _w(list(dirs), poll_s, timeout_s=timeout_s)


def report(bench: str, arms, vs=(), preset: str = "exam", out: str = "", **kw):
    from .tables import report as _r
    return _r(bench, list(arms), list(vs), preset=preset, out=out, **kw)


def score_poses(poses, out, keys=(), tokens=None, **kw):
    """Submit pose scoring (jevdrive.bench.poses) to the pool; returns its run dir (wait with bench.wait)."""
    from .poses import submit
    return submit(poses, out, keys, tokens, **kw)
