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
| adapt_h | `H-<tag>` (checkpoint in `runs/op_adapt_H/runs/<tag>`) | HUGSIM only | no-adapter ONNX export, training-port stream equivalence gate (plan xy <= 0.5 m), then the shared policy server |
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
lateral path, tree opctrl, decision 118), `spec_plan` (spec with the lateral curvature from the model's own plan). Also supported: `spec_cold`, `spec_hold`, `spec_plan_smooth`, `spec_plan_mpc`; `opctrl_d118` aliases `spec`. Scenario sets
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

## HUGSIM configurations and legacy callers

```bash
# Rule options / controller parameters are recorded and get separate run identities.
$B run --model cinque --bench hugsim --preset spec --opts '{"resume": {}}' --scenarios spin10
$B run --model cinque --bench hugsim --preset exam --controller opctrl_long \
  --opts '{"op_ctrl": true, "op_long": true}' --controller-env '{"OP_CTRL_LONG": {"max_accel": 2}}' --scenarios spin10
$B run --model H-my-checkpoint --bench hugsim --scenarios spin10
$B run --model P2-F-s0 --bench hugsim --preset spec_plan_smooth --repeat r0 r1 --scenarios all64
$B report --bench hugsim --arms run:<exact-run-directory-key> --vs P0 --preset spec
```

Run keys retain `<model>_<preset>` for default arms, add `-c<configuration hash>` for options, controller/env or custom
`--onnx`, and `-r<label>` for independent repeats. The same repeat label resumes; a new label starts an independent run.
Scenario sets are unioned on resume. `config.json` stores both requested options and preset-resolved options/env. Status and
report accept the same identity flags; `run:<key>` selects a precise arm for mixed-configuration reports. Configured arms never
fall back to baseline historical results. Custom ONNX paths must point to immutable exports; replacing a file at the same path
requires a new repeat label.

Legacy HUGSIM orchestrators source `scripts/bench_lane.sh`, which translates their inputs to bench flags and publishes the
finished raw rows to their original `results.csv`/tags. `bench-runs.json` maps each tag to its canonical run directory; trace
paths keep pointing there. Output-directory/tag pairs receive stable repeat labels so old independent lanes remain separate.
`BENCH_REPEAT` overrides that label (used by the guard's force mode). Options/controller changes invalidate old rows of that
legacy tag, and publishing fails for incomplete runs. Reports that intentionally read legacy CSVs, including the spec-plan
publication report, continue to read those exports.

Call from the orchestrator to let bench place its stages. Historical leased workers use `--in-pool` to execute exactly those
stages inside their existing lease, with no nested pool submission. Budget HUGSIM leases as 8 + 8.5 W GB VRAM / 2 W + 3 cores.
`--wait-timeout-s N` is an orchestrator-only deadline: it cancels this run's jobs, waits for cancellation, returns 124 and can
export partial rows for deadline-limited diagnostics. Partial runs carry `WAIT_TIMEOUT`/`ERROR`, never `DONE`. Derotation
selector orchestration preserves its `STOP_AT` cutoff; invoke it outside a GPU lease.

`jevdrive.bench.compat` supplies bench-first navtest CSVs with devkit column names, navhard harness directories, prediction
paths and HUGSIM rows, with historical fallback. Experiment-specific gates, bootstrap methods and training remain in their
original topics.

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

## Migration acceptance (2026-10-07)

Batches 1 and 2 are implemented and verified. Mac / box regression coverage includes run identity, parameter transport,
gate verdicts, bench-first result readers, legacy publishing, trace resolution, independent repeats, leased execution,
completeness failures and deadline cancellation. The Linux watchdog also passes on the box.

- Legacy navtest reader against bench: all 12,146 P2-F-s0 tokens and nine sub-scores have max absolute difference 0.
- Legacy navhard reader against bench: all 225 P0 GIMM groups' combined / stage-1 / stage-2 scores have max difference 0.
- GPU smoke: 18 independent configurations on `scene-0013-medium-00`, all complete without missing scenarios. Covered
  fixed / fixed2 / ideal / official / fixedc / lowspeed / lowsel / opctrl_long trees, launch / derotation / resume rules,
  cold / hold / smooth / MPC presets, H checkpoint export, custom ONNX and WA-JEPA. These are execution checks, not
  replacement full-exam scientific results. Each resumes with every stage already finished.
- `H-it_dw3-s0` no-adapter export: both 134-frame streams have max plan xy difference 0.125 m, below the 0.5 m gate.
- Legacy CSV export preserves all 18 actual trace paths (all have `infos.pkl`; the 17 openpilot runs also have
  `zs_steps.jsonl`; the shipped WA-JEPA client does not emit that file).

Box manifest: `$DATA_DIR/runs/bench/migration-smoke.json`; independent runs use repeat `migration-smoke`. Small compatibility
exports live in `$DATA_DIR/runs/bench/migration-legacy`. Existing scientific result directories were not overwritten.

## Adding things

- A model: add a `Model` to `STATIC` in `models.py` (or a family branch in `resolve`), its plans function in `navsim.py`, its
  servers in `hugsim.Servers`; a test in `tests/test_bench.py`.
- A benchmark: a module with `stages(model, ..., run_dir) -> [runner.Stage]` and a `collect` that writes `units.csv`,
  `summary.json`, `DONE`; register the stage functions in `stage.py` and the loader in `tables.load`.
- Tests: `python -m unittest tests.test_bench tests.test_bench_migration -v` (no GPU; the watchdog test needs Linux). GPU smoke through the pool:
  `$B run --model P0 --bench hugsim --scenarios scene-0013-medium-00 --wait` (one scenario, ~3 min).

Not wrapped yet: WA-JEPA re-runs on NAVSIM (its runner is experiments/top10; the bench reads its stored runs), op_guard
candidates with a command / route adapter, WOD RFS (a guard line only). The first two migration batches below now use bench for their standard evaluation sections; training and topic diagnostics remain local.

## Existing-script migration audit (2026-10-07)

**Assessment:** the standard parity evaluation stages are a small migration; every old experiment chain together is a larger
project. Unify evaluation execution and result loading, while keeping training, experiment gates and diagnostic analysis in
their topics. Batches 1 and 2 have been migrated; the remaining interfaces below describe subsequent work.

Scope: tracked `.py` / `.sh` files in `scripts/`, `experiments/` and `jevdrive/`, excluding `archive/`, `results/` and `figs/`:
49 + 548 + 97 = 694 files. A text search for the existing runners / exporters / scoring helpers / bench found 182 files,
including 37 shell scripts. These are search hits, **not 182 redundant runners**: they include library code, agents, reports,
training and references in comments. The 556 experiment sources under `archive/` (excluding `results/`) were inventoried
separately; historical chains need no bulk rewrite. Some archived files are still execution dependencies, listed below.

### Evaluation stages already covered

Paths in this table are relative to `experiments/`. “Covered” means the model / metric / serving protocol is supported; callers
and readers in batch 1 now use bench. Keep staged training and gate order, and invoke bench from the orchestration process,
outside a job that already reserves GPU resources. Do not replace a leased worker command with a nested `bench run --wait`.

| Existing entry | Bench replacement | What stays / compatibility work |
|---|---|---|
| `op_parity/scripts/pp_full_chain.sh` | `run --model P0 <full tags> --bench navtest hugsim --preset exam spec` | Keep cache sanity, split registration and training; replace the final readout section and its result readers. |
| `op_parity/scripts/pp_hinge_chain.sh` | Per-checkpoint navtest; passing arms on navhard with `@gimm`, HUGSIM exam / spec | Keep lambda selection, train sanity and gate return codes. The hinge-specific navhard report reads its own `hinge/harness/`, so adapt it too. |
| `op_parity/scripts/pp_turn_chain.sh` | Navtest for HP / T*P tags; HUGSIM exam / spec after the gate | Keep turn balancing, overall guard and gap-closure gate. Standard bench strata do not replace the custom closure calculation. |
| `op_parity/scripts/pp_score_chain.sh` | Navtest for its explicit model/frame specs, plus `cinque` | Replace waiting / export / scoring; bench regenerates or reuses plans through its own stages. Confirm checkpoint completion before submission. |
| `op_parity/scripts/pp_navhard_chain.sh` | `run --model P0 <full tags> --bench navhard` | Our arms are covered. WA-JEPA inference / request equivalence is not; preserve that branch until it gets bench stages. Keep the training lane's `GPU_DONE` dependency until callers are updated. |
| `op_parity/scripts/pp_navhard_gimm_chain.sh` | `run --model P0@gimm P2-F-s0@gimm P2-F-s1@gimm --bench navhard` | Replace plans / export / harness; update `pp_navhard.py report-gimm` or use a bench report with matching references. |
| `op_parity/scripts/pp_hugsim.sh arm` | `run --model <tag> --bench hugsim --preset exam/spec/spec_plan --scenarios <list>` | Includes smooth / MPC presets. `equiv` remains local; leased callers execute shared stages in their existing job. |
| `hugsim/scripts/wajepa_run.sh` | `run --model WA-JEPA --bench hugsim --preset exam --scenarios <list>` | Default `fixed` and optional `TREE=fixedc` are covered. Keep the shipped client launcher `wajepa_e2e.sh`. |

The first batch is these **eight entries' standard evaluation sections**, plus the readers they feed. It does not require
another scheduler or changes to model training. `pp_eval.py`'s standard plans / v2 scoring and generic arm / paired tables,
`pp_full_report.py`, and the generic parts of `pp_navhard.py` / `pp_hugsim_report.py` now use the shared execution / readers. Custom extraction,
pixel equivalence, gate calculations, plots and timeline diagnostics stay in their topics and consume bench outputs.

### Shared HUGSIM extensions completed (batch 2)

Options, controller trees/env, interface presets, repeat identities and H checkpoint export/equivalence stages are implemented.
The derotation, launch, low-speed, opctrl/longitudinal, spin comparison, unified interface, H readout and op_resume HUGSIM
wrappers delegate execution to bench. Guard plain-ONNX HUGSIM workers also share these stages and preserve force/completeness
checks. WA-JEPA HUGSIM supports its optional controller tree. Smooth/MPC presets use the existing interface definitions.
CARLA branches, route adapters, custom NAVSIM metrics/predictions and WA-JEPA NAVSIM generation remain outside this batch.

### Remaining interfaces and original size assessment

| Existing family / entries | Missing piece before migration | Size |
|---|---|---|
| `op_guard/scripts/{navlib,line_navtest,line_navhard}.py` | Guard navtest is **v1 PDMS**, bench navtest is **v2 EPDMS**. Add an explicit metric/version identity, v1 collector and score columns. Keep frozen subset membership, force/provenance rules, early-turn readout and the original gate / CI semantics. Navhard full mode also cross-checks the devkit script against the harness. | Medium |
| `op_guard/scripts/{cllib,line_hugsim,guard_hugsim}.py/.sh` | Plain ONNX HUGSIM is migrated, including force / provenance / completeness and the existing spin gate. Remaining: adapter-aware plan and server stages for command / route candidates. | Medium for adapters |
| `op_resume/scripts/{or_hugsim.sh,or_submit.py}` | HUGSIM is migrated, including rule identity, preflight and completeness. Its CARLA branch still needs the resident policy-server lifecycle. | Medium for CARLA |
| `op_parity/scripts/{pp_stageB_submit,pp_unfreeze_chain}.sh` | Standard full-navtest evaluation is covered. Stage B also uses the separate real-history dataset `lb_hq_navtestX`, which `--subset` of `lb_navtest` cannot reproduce. Unfreeze needs its pixel-path equivalence gate and readers of the original plan/pred files; keep P0 / P2 pixel control generation for that check. | Partial migration |
| `op_adapt_h/scripts/{h_onedriver_nav,h_od2_ratio}.sh`; `skill_pack/scripts/{hist_align_chain,hist_align_chain2,hq_chain,edge_chain,edge_vcam_chain,trk_chain,trk_chain2}.sh` | Add scoring of supplied predictions, v1 / navtrain calibration, alternate frame/rig protocols and selector/compensation identities. Bench currently generates `base` exports from registered models; it cannot directly score these transformed pose files. | Medium to large |
| `op_parity/scripts/pp_navhard_wajepa.sh`; WA-JEPA NAVSIM runners in `top10/` | Add request building, shipped inference, shard merge, pose conversion and request-path equivalence as stages; current NAVSIM support only loads stored WA-JEPA references. | Medium |
| `b2d_collect/scripts/b2dc_lane.py` | `B2DAgent` already fits the recording agent / custom env / XML. Reuse CARLA stages, preserve output layout plus labels/check stages and gates; carry its stall/route timeout and per-route seed settings. | Small to medium |
| `b2d_tfv6/lib/{b2d_tfv6_campaign,b2d_tfv6_w2b}.py`; `b2d_controller/lib/b2d_controller_campaign.py`; `b2d_privileged/scripts/{b2d_privileged_chain,b2d_privileged_focus}.py` | Use explicit arm/route/seed identity and pool placement, while preserving attempt archives, paired ordering where required, invariant failures, cancellation and diagnostic recording. Controller campaigns also reuse a world/server across rounds; replacing them with independent route jobs changes that protocol. | Large compared with a wrapper rewrite |
| CARLA consumers of `op_guard/scripts/cllib.py`: guard B2D, `op_resume`, `op_route_ft/scripts/near_lane.py`, `op_wide_ft/scripts/wide_lane.py`, `op_img_cmd/scripts/img_cl_lane.py` | Their agents need a resident policy server / adapter, configuration and lane-specific readouts. `B2DAgent` accepts the agent and env, but does not supply those server stages. Add shared serving hooks before replacing the old `op_arb.sh` launchers. | Medium |

The original audit identified these cross-cutting prerequisites (HUGSIM identity and batch-1 readers are now implemented):

1. **Run identity.** Before batch 2, HUGSIM keyed results by model + preset and unions requested scenarios on rerun. A rule arm,
   controller override or independent repeat must have a distinct identity/config; adding only `--opts` would reuse the
   baseline's finished scenarios. NAVSIM versions and transformed predictions similarly need separate identities.
2. **Result readers.** Before batch 1, compatibility was one-way: `tables.load` can read legacy results; `pp_eval.eval_csv` and
   topic gates cannot read bench's new CSV names. Hinge, turn and unfreeze readers also use old harness/pred/extract paths.
   Prefer a common bench-first loader with legacy fallback; retain diagnostic artifacts not present in `units.csv`.
3. **CARLA completeness.** B2D is a Python stage API, not a `run` / `report` CLI choice, and `tables.load` has no B2D
   branch. `b2d.collect` writes `DONE` even with fewer rows than requested, reporting that in the summary. Data collection
   and invariant-sensitive campaigns must retain their stricter gates; a generic `DONE` is not gate acceptance.

### Code that remains outside generic benchmark execution

Training and feature/bank preparation (`pp_train`, `pp_prep`, `h_train`, `rft`, `img*_chain`, `fw_p2_chain`, etc.), CARLA
pair rendering / label creation (`op_route_cmd`, `near_chain`), offline probes / replay, latency measurements, figures,
downloads, box operations and research serving are not duplicate benchmark runners. A mixed chain can call bench for its
standard evaluation tail without moving its entire experiment into bench. WOD RFS stays the existing guard line (decision 107).

Keep execution dependencies even when their wrappers are retired: `scripts/{op_lb,b2d_run,b2d_route}.py`,
`op_openloop/lib/op_interp.py`, `op_guard/scripts/nav_harness.py`, `op_parity/scripts/{pp_train,pp_prep,pp_unfreeze,pp_hugsim}.py`,
`hugsim/archive/{zs_run,hugsim_zs_server}.py`, `hugsim/scripts/spin_analysis.py`, `hugsim/scripts/wajepa_e2e.sh`, and the shipped model clients.
The existing `op_arb.sh` server setup also remains needed until the CARLA serving hooks above exist. Archive placement alone
does not mean a file is unused.

### Effort and migration order

Static-review estimates, including caller/reader changes and focused regression work, not measured completion times:

| Batch | Scope | Estimate |
|---|---|---|
| 1 | Eight covered entries above; migrate common readers, preserve topic gates, standard parity NAVSIM/HUGSIM smoke | 1-2 engineer-days; hundreds of changed lines across wrappers and readers |
| 2 | HUGSIM options/presets/controller configuration/repeats with distinct identities; migrate related wrappers | Another 1-2 engineer-days, plus box checks for each new controller protocol |
| 3 | v1 / supplied-prediction scoring, route adapters, WA-JEPA NAVSIM stages and CARLA serving / campaign integration | Several more days; invariant-sensitive CARLA campaigns can take longer |

The broad migration is **at least a week-scale effort**, rather than a few command substitutions; adding every historical
model or rewriting archived experiments is outside that estimate. Batches 1 and 2 are complete; batch 3 remains a separate migration.
Compare plans and per-unit scores against the existing path; check gate decisions, frame protocols, resume/skip behavior,
failure propagation and report sources. Use small pool runs on the box for changed runtime paths, rather than rerunning
every stored result. Only retire a wrapper once callers and result readers have moved; keep reproducibility dependencies.
