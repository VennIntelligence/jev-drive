# Benchmarks: run any model on any benchmark with `jevdrive.bench`

**Rule.** Do not write a new runner, sharding script, server launcher or scoring chain for NAVSIM navtest, navhard, HUGSIM or
Bench2Drive. Use `jevdrive.bench`. If a model or a readout is missing, add it there (a registry entry or a stage), with a test,
so the next lane gets it for free. Old lane scripts keep working but are not the way to start a new test.

```bash
cd ~/data/jev-drive; B=".venv/bin/python -m jevdrive.bench"
$B models                                   # what can run: shipped openpilot, op_guard ONNX candidates, op_parity arms, WA-JEPA
$B run --model P2-F-s0 P2-F-s1 --bench navtest navhard          # open loop: plans -> export -> devkit scoring, all on the pool
$B run --model P2-F-s0 --bench hugsim --preset exam spec --scenarios all64     # closed loop, one worker job per card
$B status                                   # live runs: stage states + one STATUS line each (--wait blocks)
$B report --bench navtest --arms P0 P2=P2-F-s0+P2-F-s1 --vs WA-JEPA --out experiments/<topic>/results/x
$B report --bench hugsim --preset spec --arms P2=P2-F-s0+P2-F-s1 --vs P0 WA-JEPA --scenarios turn23
```

`run` submits every stage of every requested (model, bench, preset) to the GPU pool at once, chained with `after`, and returns;
the pool places them (no card or port picked by hand). Re-running the same command resumes: finished stages are skipped, live
jobs are reused. Python: `from jevdrive import bench; d = bench.run("P2-F-s0", "hugsim", preset="spec"); bench.wait([d]);
bench.report("hugsim", ["P2-F-s0"], vs=["WA-JEPA"], preset="spec")`.

## Models (`jevdrive/bench/models.py`)

Name syntax `<name>[@<frames>][:<opt>]`, e.g. `P0@gimm`, `P3-F-s0:noside`.

| Family | Names | navtest / navhard plans | HUGSIM serving |
|---|---|---|---|
| onnx | `cinque`, `lebowski`, `small` (shipped), op_guard candidates without a command adapter (`fw-S3`, `it_dw3-s0`, `dg1` ...) | `scripts/op_lb.py run` (envs/openpilot, TensorRT; an existing op_lb plan file of the same stem is reused) | `hugsim_zs_server.py <base> [--onnx X]` |
| parity | `P0` (shipped weights through the parity path), `P1-init`..`P3-init` (bias exactly 0), every checkpoint under `$DATA_DIR/runs/op_parity/runs/<tag>/ckpt-final.pt` (P1/P2/P3-F-s*, P2H10, HP, T*P, PX, PC, UF-*) | torch port on the pp_prep token cache (`pp_train.PModel`, fp16, batch 128); UF-* arms from pixels (`pp_unfreeze.py plans`) | the arm's ONNX (`pp_hugsim.py onnx`) + its bias server (`pp_hugsim.py serve`); P0 serves the untrained P3 adapter, as the lane did |
| wajepa | `WA-JEPA` | stored references (its navtest CSV, navhard harness dir) | its shipped client (`zs_run --agent wajepa`), exam preset only |

Frame protocols (navsim only; closed loop renders its own frames): `gimm` (G, GIMM-synthesised 0.2 s pairs; shipped models and
`P0@gimm`), `warp` (W, CPU ego-motion warp; the default of every full-run parity checkpoint and of `P0`), `keys` (N, 2 Hz keys),
`vh140` (1.40 m virtual camera, UF-V). A parity tag defaults to the protocol it was trained on. A missing token cache for a
protocol adds a `prep` stage (`pp_prep.py`).

## Benchmarks

| Bench | Unit / cluster | What runs | Output (`$DATA_DIR/runs/bench/<bench>/<key>/`) |
|---|---|---|---|
| `navtest` | token / log | plans, `op_interp nav-export` (adapter `base`), the v2 devkit (`navsim main @0a380a9`, the harness that reproduced WA-JEPA's 91.71, `OPENBLAS_CORETYPE=Haswell`) in K shards of whole logs (one per ~12 cores) | `units.csv` (token, log, score, NC DAC DDC TLC EP TTC LK HC EC), `summary.json` |
| `navhard` | scene-mapping group / log of its stage-1 token | plans, export, `experiments/op_guard/scripts/nav_harness.py` (per-token devkit score + the devkit's two-stage aggregation; mean over the 225 groups = the official number) | `units.csv` (group, orig, combined, stage1, stage2, log) |
| `hugsim` | scenario / scenario | `experiments/hugsim/archive/zs_run.py` (HUGSIM's closed_loop.py from the patched private trees), one scenario per call | `units.csv` (HUGSIM scores, end, cls, spin, max heading error, v_max40, launch_stall, stuck), `results.csv`, run dirs |
| CARLA / B2D | route | `jevdrive.bench.b2d`: any leaderboard agent through `scripts/b2d_run.py` and `jevdrive.cl.pool.b2d_cmd` (Python API, see below) | `units.csv` (DS, RC, status, infractions) |

HUGSIM presets: `exam` (preset exam, tree `fixed`: the wajepa_ref harness and every result before 2026-10-05), `spec` (openpilot's
lateral path, tree opctrl, decision 118), `spec_plan` (spec with the lateral curvature from the model's own plan). Scenario sets
(`$B sets`): `all64`, `spin10`, `spec29`, `small11`, `turn23` (route heading range >= 30 deg), a `.txt` list, or comma-separated
scenario stems. Behaviour classes follow `experiments/op_parity/results/hugsim_spin10.md`: spin = heading error >= 60 deg vs the
recorded route, launch stall = peak speed over the first 40 steps < 1.6 m/s, stuck = `max_steps` end, cls = spin, else the end.

WOD RFS is not wrapped: since decision 107 it is only the op_guard `wod` guard line (`experiments/op_guard/scripts/line_wod.py`).

## Execution (`runner.py`, `hugsim.py`)

- A run dir holds `config.json`, `jobs.json` (stage -> pool job id), `pool/<stage>/` (the pool's log dir: `log.txt`, `STATUS`,
  `DONE` / `ERROR`), a run-level `STATUS` line, and `DONE` (written by the last stage, `collect`, with the summary). `status` /
  `wait` write the run-level `ERROR` when a stage failed. A stage is finished when its output exists (outputs are written
  atomically), so every command is resumable.
- Sizing comes from the live box: navtest scoring shards = quota / 12 cores (12 cores each; small jobs backfill a busy box), HUGSIM worker jobs = one per card (fewer when few
  scenarios are left), W = 6 scenario slots per job, VRAM 8 + 8.5 W GB (measured 48-59 GB at W = 6), CPU 2 W + 3. The pool's
  history (`runs/pool/history.json`) learns the measured peaks under the job names `bn-<bench>-<model>...`.
- HUGSIM: the slots of all worker jobs pull from one claim-file queue, longest expected scenario first (median wall of earlier
  runs). Every scenario is one `zs_run.py run` under a watchdog: no `sim.log` growth for 420 s, or longer than 1500 s (the longest
  scenario so far took 723 s), stops its process tree (`jevdrive.cl.procs`) and retries it (2 retries). Before each scenario the
  job's servers are checked (process alive, socket accepting a connection) and restarted when one is dead or refusing: a refused or
  dead bias-server socket costs one retry instead of the old 5400 s timeout. A scenario that fails all attempts is listed in `summary.json`
  (`failed`, `missing`); a new `run` retries it.
- Jobs go to the pool as `bn-<bench>-<model>[-<preset>]-<stage>`; `--priority`, `--gpus` pass through.

## Reports (`tables.py`)

`report` writes `<out>/<bench>[_<preset>]_{arms,paired,shapley,strata}.{md,csv}` through `jevdrive.stats`:

- arms: `label=spec+spec` averages seeds per unit, then over units; navtest EPDMS = 100 x token mean (as every op_parity table),
  sub-scores; navhard combined / stage1 / stage2; HUGSIM HD, sub-scores, class counts, launch stalls, stuck.
- paired: every arm minus every `--vs` reference (and the references pairwise), percentile bootstrap B 10000 seed 0, clustered by
  log (navtest), by the stage-1 log of the group (navhard), per scenario (HUGSIM).
- shapley (navtest): exact Shapley split of the per-token EPDMS gap ref - arm over the nine sub-scores (columns sum to the gap).
- strata: navtest turn bins (<5 / 5-20 / 20-45 / >45 deg logged 4 s heading change), manoeuvre, speed at t0, city
  (`experiments/op_probe/scripts/opj_build.py` definitions); HUGSIM dataset, difficulty, turning route.
- Sources: the bench run dir if it exists, else the lanes' stored results (op_parity navtest CSVs, navhard harness dirs, the
  op_parity HUGSIM `results.csv`, WA-JEPA's stored runs); the `source` column says which.

## CARLA / Bench2Drive

```python
from jevdrive.bench import b2d, runner
a = b2d.B2DAgent("my-agent", agent="lib/op_arb_agent.py", agent_config="cfg.json", extra=["--rig", "front3"])
d = runner.bench_root("b2d", a.name)
runner.submit(d, "bn-b2d-my-agent", b2d.stages(a, d, routes="bench2drive220", workers=6))
```

Every worker job is one b2d_run on the pool's placeholders (`carla = W` servers; the pool picks card, server indices and cores);
all jobs share one output dir, so b2d_run's route claims spread the routes and a rerun skips finished ones. Read
`b2d/summary.json` (`routes_never_finished`, `restarts`) before quoting a score (runbook trap list).

## Acceptance (2026-10-06, the new path against the lanes' stored results)

| Check | New path (`jevdrive.bench`) | Stored result | Verdict |
|---|---|---|---|
| navtest plans, P2-F-s0 W and P0 navhard G | `plan_mu` / `plan_std` / `plan_pos` | pp_eval.py plans in runs/op_lb | bit-identical |
| navtest EPDMS, P2-F-s0 (3 log shards) | 88.12, all 12 146 tokens | results/full.md s0 88.12 | every token's score and 9 sub-scores identical (max abs diff 0) |
| navtest EPDMS, shipped `cinque` (G) | 81.11 | v2_navtest_opi_lb_navtest_gimm-cinque__base 81.11 | per-token identical |
| navhard two-stage, `P0@gimm` | combined 33.54, stage 1 71.79, stage 2 47.08 | navhard_gimm_arms.md 33.54 / 71.79 / 47.08 | identical |
| HUGSIM 64, P2-F-s0 `spec` | HD 0.3795; ends 18 complete / 30 fg / 12 bg / 4 off_route | pp-spec-P2-F-s0 0.3798 (19 / 30 / 11 / 4) | new - old -0.000 [-0.001, +0.000]; 47 / 64 scenarios identical HD, 63 / 64 same end, none differs by > 0.05: inside the known between-server noise (HD +-0.005, op_control_stack.md) |
| HUGSIM spin10, shipped `cinque` `exam` | HD 0.367, 8 spins | PR #57 run 0.350, its 2026-10-03 rerun 0.315 (8 / 10 spin) | same ends as the rerun in 10 / 10, 6 / 10 identical HD; spinner scenarios are the chaotic ones |
| reports from stored results | navtest P2 - WA-JEPA -3.50 [-4.24, -2.76], P0 - WA-JEPA -11.20 [-12.47, -10.02]; navhard P2 - WA-JEPA -6.10 [-10.04, -2.30]; Shapley WA-JEPA - P0: NC 1.88 [1.44, 2.38], DAC 4.19 [3.18, 5.32] | full_navtest_paired.md, navhard_paired.md, gap_tables.json | identical; HUGSIM exam P2 - WA-JEPA -0.055 [-0.136, +0.023] vs hugsim_full.md -0.055 [-0.135, +0.025] (same mean; the CI differs in the 3rd decimal only through the order the 64 units enter the resampler) |

Time from command to running (2026-10-06, box shared with 9 op_parity HUGSIM jobs): the first stages of all runs were on the cards
20-40 s after `run` (one pool round); the HUGSIM workers then waited ~18 min for VRAM held by the other lane. Once placed, one
worker job (6 slots) had its servers up in ~45 s and ran 58 of the 64 scenarios in 12.5 min; the run finished 15.5 min after its
first worker started, with no retry and no server restart. navtest: plans + export ~1 min, one 4 000-token scoring shard ~3 min
on 24 threads, so ~5 min on an idle 75-core box (here 22 min, waiting for CPU behind the other lane).

## Adding things

- A model: add a `Model` to `STATIC` in `models.py` (or a family branch in `resolve`), its plans function in `navsim.py`, its
  servers in `hugsim.Servers`; a test in `tests/test_bench.py`.
- A benchmark: a module with `stages(model, ..., run_dir) -> [runner.Stage]` and a `collect` that writes `units.csv`,
  `summary.json`, `DONE`; register the stage functions in `stage.py` and the loader in `tables.load`.
- Tests: `python -m unittest tests.test_bench -v` (no GPU; the watchdog test needs Linux). GPU smoke through the pool:
  `$B run --model P0 --bench hugsim --scenarios scene-0013-medium-00 --wait` (one scenario, ~3 min).

Not wrapped yet: WA-JEPA re-runs on NAVSIM (its runner is experiments/top10; the bench reads its stored runs), op_guard
candidates with a command / route adapter, WOD RFS (a guard line only). The lane scripts (pp_eval / pp_hugsim / pp_navhard ...)
are left as they are; new tests use the bench.
