# Closed-loop runbook (AlpaSim, CARLA / Bench2Drive, every GPU job)

Read this first whenever you run anything on the GPU box: a read, a training, a Bench2Drive exam, a CARLA data generation.
It gives the one way to do each thing. Detail and history live in the docs it links; where they disagree, this page wins.

## Entry points

| Need | Use |
|---|---|
| run anything (GPU or CPU job) | `python -m jevdrive.cl submit --name N --vram GB [--cpu N] [--ram GB] [--after IDS] -- cmd` ([the GPU pool](#the-gpu-pool)); CPU jobs `--vram 0.1` |
| N independent runs (seeds, arms, chunks) | `python -m jevdrive.cl fanout --name N --arms a,b,c ... [--collect "cmd {arms}"] -- cmd {arm}` |
| a lane (rolls, rows, training, reads, summary) | one lane script that submits the whole chain at once ([pipelines](#pipelines-parallel-by-default)) |
| a nuPlan read (arena held / held 460 @18) | `jt.sh aread <tag> off <chunk>` / `rm_chain.sh read <tag> <i> 6`, one job per chunk; each writes its facts at the end |
| a PAI read (281 / LAUNCH-37 / PAI-FIX) | `pai_fix.sh read <tag> <serve>` (chunks + `pai_fix.sh tax` -> `runs/pai_fix/facts/<name>.csv`) |
| a summary of reads | join the facts cache: `python -m jevdrive.facts table ...` first, then lane tables ([facts](#read---facts-cache---summary)) |
| a benchmark (navtest, navhard, HUGSIM, B2D) | `python -m jevdrive.bench` ([bench.md](bench.md)) |
| what runs / waits, and why | `python -m jevdrive.cl queue`, `top`, `usage --hours 24`, `show <id>` |
| stop a job | `python -m jevdrive.cl cancel <id>` (`--drain`: b2d_run finishes its routes first) |
| what the box has right now | `.venv/bin/python -m jevdrive.cl probe` (sizes derived from it: `cl plan`) |
| one card's worth of CARLA routes | `scripts/b2d_run.py`; `jevdrive.cl.pool.b2d_cmd()` builds it on the pool's placeholders |
| one CARLA server by hand (debugging; declare it with `hold`) | `scripts/carla_server.sh start <index>` ([carla.md](carla.md)) |

Library: `jevdrive/cl/` (module map in its `__init__.py`); tests `python -m unittest tests.test_cl tests.test_pool`.

## Pipelines: parallel by default

Everything that can run in parallel runs in parallel; never leave a card idle.

- **Pipeline per chunk.** Each GPU chunk is followed at once by its own CPU stages (extract, rows, tokens, labels, facts),
  chained with `--after` on *that* chunk, so CPU and GPU work overlap. Only true joins (merge, train, summary) wait for
  all chunks.
- **Submit the whole chain in one go**, from one lane script, with `--after` between stages. Stages after a gate may be
  submitted speculatively; the gate is an in-queue CPU job that fails (its dependents fail with it) or cancels them.
  Nobody waits on a gate by hand.
- **N independent runs are N jobs** (`fanout`), never one job that loops: a looping job holds one card while the others idle.
- **Every independent arm's training goes in at the start**, waiting only on what it consumes; backfill with the next
  base training at a lower priority, cancelled if a gate stops the line.
- **Summaries join cached per-read facts and print the scores first** (next section).

Canonical chain (nuplan_dag `nd.sh full` is the model):

```bash
sub() { .venv/bin/python -m jevdrive.cl submit --owner my-lane --priority $PRIO --log-dir $O/pool/$1 --name "$@" | awk '{print $NF}'; }
for c in c000 c001 ...; do                                       # one pipeline per roll chunk
  r=$(sub roll-$c --vram 32 --cpu 8 -- bash $ME roll $c)          # GPU: student rollout of the chunk
  g=$(sub rows-$c --after $r --vram 0.1 --cpu 8 -- bash $ME rows $c)      # CPU: rows of that chunk, while other chunks roll
  k=$(sub tok-$c  --after $g --vram 16 --cpu 8 -- bash $ME tok $c)        # tokens / labels of that chunk
  K+=($k); done
m=$(sub merge --after $(IFS=,; echo "${K[*]}") --vram 0.1 --cpu 4 -- bash $ME merge)   # join: all chunks
gt=$(sub gate --after $m --vram 0.1 --cpu 2 -- bash $ME gate)                          # in-queue gate (exit != 0 stops the chain)
for s in 0 1; do t=$(sub train-s$s --after $gt --train --vram 40 --cpu 6 -- bash $ME train $s)
  for c in $(seq 0 17); do R+=($(sub ar-s$s-c$c --after $t --vram 20 --cpu 6 --env CL_HIST_KEY=alpasim-read4 -- bash $JT aread $TAG-s$s off $c)); done
  for i in 0 1 2 3 4 5; do R+=($(sub held-s$s-$i --after $t --vram 20 --cpu 6 --env CL_HIST_KEY=alpasim-read4 -- bash $RM read $TAG-s$s $i 6)); done
done                                                             # each read chunk writes its facts when it ends
sub summary --after $(IFS=,; echo "${R[*]}") --vram 0.1 --cpu 8 -- bash $ME sum   # joins the facts cache: seconds
```

A lane script records job ids (`$O/jobs.txt`) and stages (`$O/stages.txt`) and offers one `wait` stage (a
`run_in_background` waiter on a done marker or a failed job): agents submit, then stop. Check with `cl usage --hours 24`:
while a lane is live, "no work queued" and "serial" stay near zero.

## Read -> facts cache -> summary

A read's per-window facts are extracted once, per chunk, when the chunk's read job ends; summaries concatenate and join
them ([jevdrive/facts.py](../jevdrive/facts.py); measured and identity-checked in
[read_summaries.md](../experiments/cl_infra/results/read_summaries.md): Part 1 of nuplan_dag in 23 s instead of 3 min).

- **Cache** `$DATA_DIR/runs/facts/<bench>/<read key>/<chunk>/`: `scores.parquet` (every rollout: score, failure class,
  metrics), the native_gap extract (`dec` per decision: fed speed, human speed at the same station, turn, lead gap,
  plan; `win` per window; `steps` per scorer step; `hum.pkl`), `lat.pkl` (arena: the d395 lateral-zero split rows).
  Benches: `arena` (key `<tag>@<serve>`, `jt.sh aread`), `held460` (key `<tag>`, `rm_chain.sh read`). PAI reads keep
  their per-read facts in `runs/pai_fix/facts/<name>.csv` (`pai_fix.sh tax` after the read's chunks).
- **Writing** is the last line of the read job: `$AP -m jevdrive.facts chunk --bench arena --run $D || true` (~9 s on
  the read's cores; never fails the read). A re-read chunk is rebuilt (stamp of its results-summary); writers lock per read.
- **Summaries**, scores first:
  `python -m jevdrive.facts table --bench arena|held460|pai --tags A B --refs REF ... --out X.md` (means, zeros, zero
  classes, paired CIs; seconds), then the mechanism tables from
  `$AP -m jevdrive.facts ext --read LABEL=KEY.. --out DIR` (native_gap-format extract for `free_m.py arena` /
  `native_gap_report`) and `$AP -m jevdrive.facts latsplit --read ... --out CSV`. A missing chunk is extracted and
  cached by the summary itself (`fill`); backfill old reads with `$AP -m jevdrive.facts fill --bench B --keys K..` (one
  CPU job). `AP` = `$DATA_DIR/third_party/alpasim/.venv/bin/python`.

## The box is elastic: never hardcode its size

Cards, cgroup CPU quota, RAM and `pids.max` change between instances (the host always shows 208 CPUs). Every sizing
number comes from `jevdrive.cl.box.probe()` (cgroup `cpu.max`, `pids.max`, `memory.max`, affinity, NUMA, `nvidia-smi`)
or `jevdrive.cl.capacity`; `python -m jevdrive.cl plan` prints them; Python sizes pools with `jevdrive.common.n_cpus()`.
Per card: core slice = quota / cards, CARLA workers = min(GPU knee 6, slice / cores per worker, (VRAM - 8 GB) / 9 GB),
PID plan cap 0.80 x `pids.max`, b2d_run start gate (`B2D_PIDS_WAIT`) 0.85 x `pids.max`, server indices 160-494.

## CARLA worker profile

A worker is one CARLA server plus one route client; thread settings live only in `jevdrive/cl/profiles.py`. Default
`reduced`: CARLA pools 4 each, `carla.Client` 8 threads, OMP/MKL/OPENBLAS/NUMBA 2 (decision 83: same throughput and
route outcomes as stock on 40 PDM-Lite routes, 3.3x fewer threads, ~169 per worker). Camera agents: 6 servers per card
(GPU knee, 5-10 GB VRAM each); camera-less agents fit 10-12 per card; prefer <= 10 until a repeated 12-worker A/B
exists. Torch agents may need their own numeric threads (job `env`). `OPENBLAS_CORETYPE=Haswell` is required for the
NAVSIM devkit on the box (decision 073), in the job's `env`. Measurements and conditions: [carla.md](carla.md),
[bench2drive-cost.md](bench2drive-cost.md); the 2026-10-01 profile experiment: decision 83.

## The GPU pool

One dispatcher (`jevdrive/cl/pool.py`, tmux `jev:pool`) owns every card and core. Jobs are submitted; whichever card
has room takes the next one. Nobody picks a card, core list or CARLA port by hand.

```bash
P=".venv/bin/python -m jevdrive.cl"
$P submit --name dg-train --vram 35 --cpu 12 --ram 45 --train --log-dir $DATA_DIR/runs/x/train -- \
    $DATA_DIR/envs/op-train/bin/python experiments/x/train.py --steps 4000          # prints the job id
$P submit --name dg-eval --vram 12 --cpu 8 --after <id> -- "bash experiments/x/eval.sh {gpu}"
$P fanout --name wax --arms V1,V2,V3 --vram 7 --cpu 4 --collect "python collect.py {arms}" -- python run.py --arm {arm}
$P queue | top | usage --hours 24 | vram | show <id> | cancel <id> [--drain]
$P retarget <id>... --vram 26 [--ram 30] [--gpus 1,2]   # edit a queued job in place; id and --after chains stay
$P trim --margin 120                                     # drop page cache until the working set is 120 GiB under the kill line
```

From Python: `from jevdrive.cl import pool as P; jid = P.submit(cmd, name=..., vram_gb=..., cpu=..., after=[...]);
P.wait([jid])`; `P.fanout(cmd, arms, name, collect=..., ...)`; a CARLA job `P.submit(P.b2d_cmd(out, route_ids,
agent=..., agent_config=...), name=..., carla=6, cpu=12)`.

**What a job declares** (upper bounds; keep names stable, `<what>-<arm>-s<seed>`, so history applies):
`vram_gb` (required; peak VRAM; 9 per CARLA server by default), `cpu` (cores; pins with `taskset`, NUMA-local first),
`ram_gb` (peak private memory of the tree: WOD eval 40, `pp_train.py --host` 45, AlpaSim stack 52), `carla`, `train`
(at most `train_per_card` per card), `priority` (higher first, then submit order), `after` (ids that must end rc 0; a
failed one fails the dependent), `when_exists` (file gate), `gpus`, `exclusive`, `tries`, `timeout_h`, `max_rss_gb`,
`profile`, `env`, `cwd`, `log_dir`, history key env `CL_HIST_KEY`. AlpaSim reads: `--vram 20 --cpu 6 --env
CL_HIST_KEY=alpasim-read4` (4 per card; [research/alpasim-read-profile.md](../research/alpasim-read-profile.md));
rolls with frame dumps `--vram 32`. Fail fast: `submit` refuses missing scripts and `.py` files that do not compile
(`--no-check` skips); `--preflight "<smoke cmd>"` runs a 1-2 item smoke first (shared by a batch).

**What the job gets.** `cmd` (argv or one `bash -c` string) and `env` may use `{gpu} {idx} {span} {carla} {workers}
{cpus} {server_args} {client_threads} {pids_wait} {job_dir} {id}`. Env: `CUDA_VISIBLE_DEVICES`, `CL_GPU`, `CL_IDX`,
`CL_SPAN`, `CL_CPUS`, `CL_POOL_JOB` (a job can submit its own next stage with `--after $CL_POOL_JOB`), `CL_JOB_DIR`,
`B2D_DRAIN_FILE`, `B2D_PIDS_WAIT`, the profile's thread env, and `disk.job_env` (TMPDIR `$DATA_DIR/tmp/cl/<id>`,
removed when the tree exits; shared compile caches under `$DATA_DIR/cache/`). Log dir (default
`$DATA_DIR/runs/pool/jobs/<id>/`): `log.txt`, `STATUS` (also why it waits), `DONE` (JSON) or `ERROR`. Success is rc 0:
put an output check into the command when rc alone is not enough.

**How it places a job** (every 20 s, measured live):

| Resource | Rule |
|---|---|
| VRAM | total - 4 GB headroom - VRAM outside the pool (at least what holds declare) - sum over the card's jobs of max(measured, booked) >= need. With >= 3 finished runs under the history key, need = min(declared, 1.2 x max peak + 1 GB); else the declaration for 10 min, then min(declared, 1.2 x own peak + 1 GB). CARLA / `exclusive` jobs book the declaration. Same-key jobs on one card are booked together from the key's measured concurrent VRAM per job (`cl vram`). |
| VRAM over | a job above 1.1 x declared + 1 GB is booked at 1.2 x its peak + 1 GB from then on (`vram_over` event); its peak goes to history whatever its end state; on a card with < 1 GB free it is stopped and requeued once (`vram_stop`). A CUDA OOM exit is requeued once beyond `--tries`. |
| CARLA, ports | <= 6 servers per card, one CARLA job starts per card per round; a free block of 2 x `carla` indices in 160-494 whose RPC / TM ports miss every job, hold and listening port, freed only when the tree has exited |
| CPU | charged cores <= quota x `cpu_overcommit` (1.4). Charge = declared for 5 min (or from >= 3 runs' history), then max(1, 1.2 x peak measured cores over the last 5 min). Each job is pinned to cores nobody else holds; a GPU job that fits a card while CPU-only jobs hold every core is pinned to the least-pinned cores of the card's NUMA node. |
| memory | see "The kill line" below; PIDs: threads now + young jobs' + the job's estimate <= 0.80 `pids.max` |
| idle card | a card idle >= 120 s takes the highest-priority GPU job blocked only by the CPU budget or the PID cap (never VRAM, RAM, CARLA, ports; never CPU-only jobs) |
| pinning watchdog | a job restricted by `gpus` whose cards are busy gets an idle card added (`--pin-strict` opts out) |
| order | priority, then submit order; jobs that fit start out of order, but a head job blocked > 15 min by VRAM / CARLA / training cap reserves the card closest to fitting it. Among fitting cards the least loaded wins, VRAM best-fit breaks ties. |

**The kill line.** The platform SIGKILLs the largest process while the working set (`memory.current` - `inactive_file`)
is at or above 98 % of `memory.max` ([remote-box.md](remote-box.md), "Host memory"). The pool: holds a start while
working set + what young jobs have not allocated yet + the job's `--ram` + 16 GiB would cross the line (after 10 min
it starts one such job anyway, only if it fits with all page cache dropped); drops page cache (`cl trim`) when the
margin is under 80 GiB + young jobs' needs; stops at once the job furthest above its `--ram` (> 1.1 x + 2 GB) while
the margin is under 48 GiB and requeues it once with `--ram` raised to its peak; retries a SIGKILLed job, or one whose
child died with -9 near the line, 3 times beyond `--tries`. Declare `--ram`; a killed training restarts from step 0.
`top` shows working set, line, margin and the day's kills.

**History.** A job ending `done` (>= 60 s) adds its peak VRAM, cores and private memory to
`runs/pool/history.json` under `CL_HIST_KEY`, else its name with trailing seed / index digits stripped (last 20 kept);
`rm_train.py` trainings get `rm-train`. `submit` without `--vram` / `--cpu` defaults to 1.2 x p95 of that history.

**Reading under-use.** `top` ends with `idle now`, `queued` (ready jobs by cause; jobs waiting on `--after` / gates) and
`UNDER-USED`: no work submitted (submit the next stage with `--after`), a GPU job looping over runs (fan out), or ready
jobs blocked by the named cause. `usage --hours H` splits idle card-hours by cause; `show ID` has each job's `waited`.

**Holds: GPU work outside the pool.** `$P hold --card 1 (--whole | --vram 20 [--carla 2]) [--idx 170-175] [--cpus
52-76] [--cpu 6] [--train] --pid <PID> --note "..."` (ends when the pid exits); `holds`, `unhold <hid>`.

**Restarting the dispatcher** (pool code takes effect only then; running and queued jobs survive, they are adopted
from `state.json`). SIGTERM to its pid only, never Ctrl-C or SIGKILL (can start a job twice):

```bash
kill -TERM $(python3 -c "import json;print(json.load(open('$DATA_DIR/runs/pool/status.json'))['pid'])")
tail -1 $DATA_DIR/runs/pool/events.jsonl      # wait for dispatcher_stop (up to one round)
tmux send-keys -t jev:pool '.venv/bin/python -m jevdrive.cl dispatch' Enter
```

**Config** `$DATA_DIR/runs/pool/config.json` (re-read every round): `carla_per_card` 6, `train_per_card` 2,
`headroom_gb` 4, `cpu_overcommit` 1.4, `max_starts` 4 per round (GPU and CPU-only jobs counted apart), `hold_s` 900,
`poll_s` 20, `idle_s` 120, `ws_reserve_gb` 16, `ws_keep_gb` 80, `ws_wait_s` 600, `ram_default_gb` 8, `kill_retries` 3,
`oom_retries`, `ram_stop_gb` 48, `trim_roots`, `conc_margin` 1.1, `disk`. Processes are tracked by (pid, start time) +
`CL_TOKEN`; cancel, timeout and `max_rss_gb` stop the whole tree. Disk pruning of job temp dirs and caches:
[remote-box.md](remote-box.md) "Disk". Pool history and incidents: decisions 181, 222 and
[pool-fix](../experiments/cl_infra/results/pool-fix/README.md).

## Traps checklist

- [ ] **Never `pkill -f` / `pgrep -f`**: stop recorded identities (`jevdrive.cl.procs`) or the process group; `cl cancel` for pool jobs.
- [ ] **Commit named paths only** (`git add <path>`; never `-a` / `-A`): the tree is shared ([long-runs.md](long-runs.md)).
- [ ] **CARLA ports.** Index i: RPC 2000 + 50i (+1, +2), TM 8000 + 50i .. +49. Keep indices in 160-494 (ports >= 10000,
  TM below the ephemeral range 32768-60999). A startup Signal 11 is a port problem until the log says otherwise.
- [ ] **A TM port outlives its server**: b2d_run scans its own 50-port block each attempt.
- [ ] **Stagger CARLA starts** (`--stagger-s 20`, `B2D_START_SLOTS=2`); many at once wedge the NVIDIA driver.
- [ ] **RenderThread 60 s timeout + Signal 11 at route setup** is the commonest server death; b2d_run retries it; report
  retries with every score.
- [ ] **Threads count against `pids.max`** (20480): `RuntimeError: Resource temporarily unavailable` at `carla.Client()`.
- [ ] **CARLA ignores `CUDA_VISIBLE_DEVICES`**: pass `-graphicsadapter={gpu}`.
- [ ] **`SIGTERM` does not stop a CARLA server promptly**: kill the group, wait for it to empty, then `SIGKILL`.
- [ ] **`-quality-level=Low` with `-RenderOffScreen` segfaults; `SDL_VIDEODRIVER=offscreen` breaks CARLA.**
- [ ] **Closed loop is not bitwise deterministic**: compare arms against the spread of identical baselines.
- [ ] **A score over the routes that finished is a selected subset**: read `summary.json` (`routes_never_finished`,
  `restarts`) before quoting it.
- [ ] **rc 137 / `Killed`** is the kill line: declare `--ram`, read the margin in `cl top`.

## Where the detail lives

- [carla.md](carla.md): Vulkan fix, server start/stop, traps with their incidents, thread census, Town12 / Large Maps.
- [bench2drive-cost.md](bench2drive-cost.md): per-tick cost, servers-per-GPU ladder, reliability and recycling.
- [closed-loop-acceptance.md](closed-loop-acceptance.md): which controllers and harness parts are accepted.
- [long-runs.md](long-runs.md): tmux, run directories, staged launch, reporting.
- [remote-box.md](remote-box.md): box, memory, disk.
- This page before the 2026-10-22 simplification (capacity table with conditions, profile experiment tables, incident
  notes): `git show 354fd7d2:docs/closed-loop-runbook.md`.

Last verified: 2026-10-22 (pool rules, facts cache); CARLA 2026-10-05
