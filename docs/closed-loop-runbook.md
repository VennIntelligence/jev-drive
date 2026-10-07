# Closed-loop runbook (CARLA / Bench2Drive)

Read this first whenever you run anything closed-loop on the GPU box: a Bench2Drive exam, a CARLA data generation, a
controller campaign. It is the authoritative entry point: which tool to launch with, how many workers per card, which
thread settings, and the traps. Detail and history live in the docs it links; where they disagree, this page wins and
says why.

## Entry points

| Need | Use |
|---|---|
| run anything on a GPU (training, eval, render, CARLA) | `python -m jevdrive.cl submit --name N --vram GB [--carla N] [--cpu N] -- cmd` ([the GPU pool](#the-gpu-pool)) |
| what runs / waits, and why | `python -m jevdrive.cl queue`, `top`, `show <id>` |
| stop a job | `python -m jevdrive.cl cancel <id>` (`--drain`: b2d_run finishes its routes first) |
| what the box has right now (cards, quota, pids, NUMA, pool jobs per card) | `.venv/bin/python -m jevdrive.cl probe` |
| one card's worth of routes (the command a CARLA job runs) | `scripts/b2d_run.py`; `jevdrive.cl.pool.b2d_cmd()` builds it on the pool's placeholders |
| one CARLA server, by hand (debugging only; declare it with `hold`) | `scripts/carla_server.sh start <index>` (docs/carla.md) |

The library is `jevdrive/cl/` (module map in its `__init__.py`); tests `python -m unittest tests.test_cl tests.test_pool`
(the dispatcher tests start real processes and run on the box).

## The box is elastic: never hardcode its size

The instance changes between sessions: seven cards / 175 cores / 644 GiB on 2026-09-28, three cards / 75 cores /
276 GiB on 2026-10-01. Every sizing number below is either measured live by `jevdrive.cl.box.probe()` (cgroup
`cpu.max`, `pids.max`, `memory.max`, affinity, NUMA nodes, `nvidia-smi`, CARLA servers per card from `/proc`) or derived
from it in `jevdrive.cl.capacity`, with our best defaults as fallback; the pool's caps live in
`$DATA_DIR/runs/pool/config.json` (see below).

Defaults derived per card: core slice = quota / cards (25 on the 3-card box), workers per card = min(GPU knee 6, slice /
cores per worker, (VRAM - 8 GB) / 9 GB), PID plan cap = 0.80 x `pids.max`, b2d_run's start gate (`B2D_PIDS_WAIT`) =
0.85 x `pids.max`, server indices 160-494 (RPC port >= 10000, TM block below the ephemeral range).
`python -m jevdrive.cl plan` prints them for the live box.

## Worker profiles and the chosen default

A worker is one CARLA server plus one route client. Its thread settings live only in `jevdrive/cl/profiles.py`:

| Profile | CARLA pools (`-RPCThreads/-StreamingThreads/-SecondaryThreads`) | `carla.Client` threads | OMP/MKL/OPENBLAS/NUMBA |
|---|---|---|---|
| `stock` | CARLA default: (host CPUs - 2) / 3 = 68 each on the 208-CPU host | default: one per host CPU (208) | unset |
| **`reduced` (default)** | 4 each | 8 | 2 |

**Decision (2026-10-01, pre-registered rule, [fc65452:todos/2026-10-01-cl-lib.md](https://github.com/VennIntelligence/jev-drive/blob/fc65452/todos/2026-10-01-cl-lib.md)): `reduced`
with numeric threads 2 stays the default.** PDM-Lite on 40 fixed Bench2Drive routes (25 Town12, 5 Town13), one exclusive
run per arm on a 25-core NUMA-local slice of an RTX 6000D, 3-card / 75-core box. Saturated throughput = 8 x mean over
routes of (ticks / route wall); two repeats on two cards (card-to-card spread ~5%).

| Stage A, 8 workers per card | stock T1 | stock T2 | stock T4 | reduced T1 | reduced T2 | reduced T4 |
|---|---:|---:|---:|---:|---:|---:|
| saturated sim ticks/s per card (rep a / b) | 61.9 / 59.0 | 60.2 / 58.1 | 58.5 / 61.1 | 57.8 / 56.7 | 63.6 / 58.3 | 56.8 / 61.3 |
| threads per server / per route client | 346 / 213-219 | | | 154 / 13-19 | | |
| starts that died before ticking | 0 of 48 servers | | | 1 of 49 (one RenderThread timeout) | | |
| routes finished | 40/40 in all 12 runs | | | | | |

- D1 profile: reduced / stock median ratio 0.988 (rule: >= 0.95); start failures 1 vs 0 (rule: <= stock + 2); DS
  identical in 95.1% of stock-reduced route pairs vs 94.5% stock-stock and 94.0% reduced-reduced, every disagreement on
  routes that also flip between identical runs (26408, 2201, 17563, 3564); a worker has ~169 threads instead of ~561.
  So reduced: same throughput and outcomes, 3.3x fewer threads.
- D2 numeric threads: reduced T1 / T2 / T4 = 57.3 / 60.9 / 59.1 ticks/s; T2 is the only one within 0.97 of the best.
  Valid for PDM-Lite; torch agents may need their own T (override per job `env` or `--num-threads`).
- Stage B, workers per card (reduced T2, one run each): 4 -> 31.5, 6 -> 47.4, 8 -> ~60, 10 -> 65.6, 12 -> 71.2
  saturated ticks/s; the 25-core slice was only 8-12 cores busy at every W and GPU util low (no cameras), so for a
  camera-less agent the card takes >= 12 workers (~2 cores per worker budgeted, ~1 used). 16 was not run (16 x 6 GB
  exceeds the 75 GB card budget). Camera agents keep the GPU knee of 6 per card (bench2drive-cost.md).
- **Open caveat:** at 12 workers one stock run gave 87.5 ticks/s against 71.2 for reduced (single runs on different
  cards; reduced also had 2 early start failures there). Stage A at 8 workers showed no difference. Until a repeated
  12-worker A/B says otherwise, prefer <= 10 workers per card with the reduced profile, or rerun the pair before a
  dense CPU-light campaign.
- Stage C, camera-bound (b2d_run stub, front3 1600x900, 6 workers, card 0): reduced 36.4 vs stock 35.9 ticks/s, GPU util
  70 / 68%, 40/40 routes, no start failures: the profile does not cost throughput when the GPU binds either.
- The 2026-10-01 harness reproduces `carla_threads_routes.sh`: 20/20 routes, mean DS 93.3 (old runs 90.6-95.3), DS
  differences only on the known flaky routes 1956 / 3564 / 17563.


`OPENBLAS_CORETYPE` is not a profile knob but a correctness setting of particular envs: `Haswell` for the NAVSIM devkit
on the GPU box (without it NumPy's OpenBLAS returns wrong results there, research/decisions/073.md),
`Barcelona` only to replay controller goldens recorded on Tokyo bit for bit (docs/b2d-controller.md). Put it in a job's
`env`.

## Capacity and thread numbers, with their conditions

Every number here was measured under the conditions in its row; numbers from different rows are not interchangeable.
"Slice" is the CPU affinity the server ran under: UE4 sizes its TaskGraph and PoolThread pools from it, CARLA sizes its
three pools from the host CPU count regardless.

| Quantity | Value | Box / date | Cards, cores | Affinity, flags, profile | Source |
|---|---|---|---|---|---|
| threads, one stock server | ~430 | 5-card RTX PRO 6000, 2026-09-25 | 125-core quota | unpinned (208 CPUs), stock | bench2drive-cost.md |
| threads, one stock server | 374 | same | same | 45-CPU affinity, stock | bench2drive-cost.md |
| threads, one stock server | 301 | 7-card box, 2026-09-27 | 175 | 16-CPU affinity, stock | carla.md "Threads per server" |
| threads, one reduced server | 109 / 69 / 169 | same | same | 16 / 8 / 32-CPU affinity, pools 4 | carla.md |
| threads, one reduced server | 149 / 263 | 7-card box, 2026-09-28 G lane | 175 | 24 / 128-core slice, pools 4 | carla.md "In production" |
| threads, one reduced server | 154 | 3-card box, 2026-10-01 | 75 | 25-core NUMA-local slice, pools 4 | this experiment |
| threads, one stock server | 346 | same | same | 25-core slice, stock | this experiment |
| threads, route client | ~215 / ~16-29 | 2026-09-25 / 27 | - | client threads default (208) / 8 | bench2drive-cost.md, carla.md |
| threads, route client, PDM-Lite | 13-19 (client 8) / 213-219 (default 208) | 2026-10-01 | 75 | profile per row of the experiment table | this experiment |
| threads, route client, SimLingo | 69 | 2026-09-28 | 175 | 24-core slice, client 8 | carla.md |
| threads per worker (admission) | 650 / 450 / 250 / 700 | 2026-09-25 / 25 / 28 / 27 | - | stock / client 8 / reduced SimLingo / stock (G lane) | superseded by `capacity.worker_threads()` |
| cores per worker | 2.5 | 5-card box 2026-09-25 | 45-CPU NUMA slice | Town12, Alpamayo camera rig, 6 per card | bench2drive-cost.md "Recommended layout" |
| cores per worker | ~3 | same | same | heaviest rig (5 cameras) | bench2drive-cost.md |
| cores per worker | 2.0 / 1.2 | 7-card box 2026-09-28 | 24-core slice | SimLingo, TFv6, BLUE / PDM-Lite (G lane `CPU_A`) | experiments/night_queue_4/lib/nq4_g_lane.py |
| cores per worker | ~1.0-1.4 used (11-12 of 25 cores busy at 8-12 workers) | 3-card box 2026-10-01 | 25-core slice | PDM-Lite, reduced, client 8 | this experiment |
| servers per card (GPU knee) | 6 | 5-card RTX PRO 6000 2026-09-25 | - | Town12, Alpamayo camera rig: 34.8 aggregate ticks/s at 6, 38.2 at 12 | bench2drive-cost.md |
| servers per card | 6 (45.8 FPS aggregate, 8.0 GB each) | 7-card RTX 6000D 2026-09-28 | 16 CPUs | six 1600x900 cameras, reduced | remote-box.md |
| VRAM per server | 5.1-6.3 GB Town12, 8-10 GB six cameras | 2026-09-25 / 27 | - | - | bench2drive-cost.md, carla.md |
| numeric threads (OMP...) | 1 / 2 / 4 used by different lanes | - | - | never compared before | this experiment |
| `pids.max` | 20480 | every instance so far | - | counts threads | probe |

## The GPU pool

One dispatcher (`jevdrive/cl/pool.py`, tmux `jev:pool`) owns every card. Agents submit jobs; whichever card has room
takes the next one. Nobody picks a card, a core list or a CARLA port by hand.

```bash
P=".venv/bin/python -m jevdrive.cl"
$P dispatch                                   # once per box: scripts/tmux_run.sh pool $P dispatch (restart-safe)
$P submit --name dg-train --vram 35 --cpu 12 --train --log-dir $DATA_DIR/runs/x/train -- \
    $DATA_DIR/envs/op-train/bin/python experiments/x/train.py --steps 4000          # prints the job id
$P submit --name dg-eval --vram 12 --cpu 8 --after <id> -- "bash experiments/x/eval.sh {gpu}"
$P queue            # running / queued jobs, the card, and why a job waits
$P top              # per card: util, VRAM booked by the pool / used outside it / free, CARLA servers, jobs
$P show <id>        # spec, state, log tail       $P cancel <id> [--drain]
$P retarget <id>... --gpus 0,2   # change the allowed cards of queued jobs; ids and --after chains stay
```

**Fail fast.** `submit` refuses a command whose script paths are missing or whose `.py` files do not compile (`--no-check`
skips). `--preflight "<smoke cmd>"` (the same job on 1-2 items) runs that smoke first at top priority with the job's
resources; the job waits for it and fails with it. Jobs of a batch that pass the same smoke command share one run, so
one broken script costs one 20 s smoke, not N launches. Smoke runs see `CL_PREFLIGHT=1`, log under `<log-dir>/preflight`.

From Python (chains, guards): `from jevdrive.cl import pool as P; jid = P.submit(cmd, name=..., vram_gb=..., carla=...,
cpu=..., after=[...]); P.wait([jid])`. A CARLA job: `P.submit(P.b2d_cmd(out, route_ids, agent=..., agent_config=...),
name=..., carla=6, cpu=12)` (6 servers, 9 GB each unless `vram_gb` says otherwise).

**What a job declares.** `vram_gb` = its peak VRAM on the card (required; default 9 per CARLA server), `carla` = CARLA
servers it starts, `cpu` = cores (> 0 pins it with `taskset` to free physical cores, NUMA-local first), `train` (at
most `train_per_card` per card), `ram_gb`, `priority` (higher first, then submit order), `after` (ids that must end
with rc 0; a failed one fails the dependent), `when_exists` (a file gate), `gpus` (allowed cards), `exclusive` (card
alone), `tries`, `timeout_h`, `max_rss_gb` (stop the tree above it: the openpilot leak cap), `profile` (thread env from
profiles.py; CARLA jobs get the default), `env`, `cwd`, `log_dir`.

**What the job gets.** `cmd` (argv, or one shell string run by `bash -c`) and `env` values may use `{gpu} {idx} {span}
{carla} {workers} {cpus} {server_args} {client_threads} {pids_wait} {job_dir} {id}`. The env carries
`CUDA_VISIBLE_DEVICES`, `CL_GPU`, `CL_IDX`, `CL_SPAN`, `CL_CPUS`, `CL_POOL_JOB` (so a job can submit its own next stage
with `--after $CL_POOL_JOB` or plain `submit`), `CL_JOB_DIR`, `B2D_DRAIN_FILE`, `B2D_PIDS_WAIT` and the profile's
thread env. The log dir (default `$DATA_DIR/runs/pool/jobs/<id>/`) gets `log.txt` (all tries), `STATUS` (one line,
also the reason it waits), `DONE` (JSON) or `ERROR` (reason + log tail). Success is rc 0: put an output check into the
command when rc alone is not enough.

**How it places a job** (every 20 s, measured live; nvidia-smi cached <= 15 s):

| Resource | Rule |
|---|---|
| VRAM | total - 4 GB headroom - (VRAM of processes outside the pool, at least what holds declare) - sum over the card's pool jobs of max(declared, measured) >= `vram_gb`. Declared VRAM stays booked for the job's life, so a job that has not reached its peak is never overbooked. |
| CARLA | servers of pool jobs + servers outside the pool + `carla` <= 6 per card; one CARLA job starts per card per round (staggered starts) |
| ports | a free block of 2 x `carla` server indices in 160-494 whose RPC and TM port blocks (index i: RPC 2000 + 50i, TM = RPC block of i + 120) miss every pool job, every hold and every LISTENING TCP port on the box; freed only when the job's whole process tree has exited |
| CPU | charged cores of pool jobs + holds <= cgroup quota x `cpu_overcommit` (1.0). Charge = declared `cpu` while the job is younger than 5 min (or unmeasured), then max(1, 1.2 x its peak measured cores over the last 5 min) (utime + stime of its process tree, every round); holds keep declared cores. `queue` shows `cpu m/d` = measured peak / declared, `top` the charged total. |
| PIDs / RAM | pids.current + threads of jobs younger than 5 min + the job's estimate (thread model) <= 0.80 pids.max; cgroup memory without page cache (`memory.stat` anon + shmem + kernel) + young `ram_gb` <= 0.85 memory.max |
| idle card (work conservation) | A card with no pool job, no whole hold and <= 1 GB used outside the pool for >= `idle_s` (120 s, config) takes the highest-priority queued job with `vram_gb` > 1 that is blocked **only** by the CPU budget or the PID plan cap (event `admit_idle`). VRAM, RAM, CARLA and ports are never relaxed; CPU-only jobs (`vram_gb` <= 1) never use the rule. |
| pinning watchdog | A queued job restricted by `gpus` whose allowed cards are busy gets an idle card (>= `idle_s`) outside `gpus` added to its `gpus` and starts there (event `auto_retarget`). `submit --pin-strict` opts out. |
| order | priority, then submit order; jobs that fit start out of order (backfill), but a head job blocked > 15 min reserves the card closest to fitting it |

Among the cards that fit, the least loaded (pool jobs + foreign GPU processes) wins, VRAM best-fit breaks ties: spreading
keeps every card computing (one serial chain per card left cards at 0-20% util with one CARLA server where six
fit, 2026-10-05); best-fit plus the head-job reservation keeps room for a large job.

**History defaults.** When a job ends `done` (>= 60 s), the dispatcher appends its peak VRAM and peak cores to
`runs/pool/history.json` under its name prefix (lower case, trailing `-s3`, `-k2`, `_t0`, `-007`, `-pf`, digits stripped
repeatedly: `op-eval-s3-pf` and `op-eval-k2` are `op-eval`; last 20 kept). `submit` without `--vram` / `--cpu` then
defaults to p95 x 1.2 of that history and prints the choice on stderr; explicit values win. Dispatcher config keys:
`idle_s`, `idle_vram_gb` (trivial foreign VRAM), `cpu_overcommit`, `cpu_budget`.

**Process safety.** Each job runs in its own session; its tree is recorded by (pid, start time) plus a `CL_TOKEN` env
entry (`jevdrive.cl.procs`). When the root exits, members still alive (orphan CARLA servers, setsid children) are
stopped through a pidfd; resources are freed only after the tree is empty. Cancel, timeout and the RSS cap stop the
tree the same way. The dispatcher can be restarted at any time (`tmux` window closed, SIGTERM): running jobs keep
running and are adopted from `state.json`.

**Holds: GPU work outside the pool.** Anything not started by the pool (a hand-started server, a process from before
the pool) is declared, or the pool will only see its current VRAM:
`$P hold --card 1 (--whole | --vram 20 [--carla 2]) [--idx 170-175] [--cpus 52-76] [--cpu 6] [--train] --pid <PID> --note "..."`.
With `--pid` the hold ends by itself when that process exits; `holds` lists, `unhold <hid>` drops one.

**Config** (`$DATA_DIR/runs/pool/config.json`, re-read every round): `carla_per_card` (6), `train_per_card` (2),
`headroom_gb` (4), `cpu_overcommit` (1.0) or `cpu_budget`, `max_starts` per round (4), `hold_s` (900), `cards` (all),
`poll_s` (20). Spool layout: `inbox/`, `cancel/`, `holds.json`, `state.json`, `status.json`, `events.jsonl`, `jobs/<id>/`.

**Chains.** A multi-stage chain is either all stages submitted at once with `after=`, or a small driver that submits a
stage, `P.wait`s, checks its outputs and submits the next. Keep the driver itself off the GPU (it can run in tmux or
as a `--vram 0.5` pool job). Resumable stages skip finished units, as before.

## Traps checklist

- [ ] **Ports.** Index i: RPC 2000 + 50i (+1, +2), TM 8000 + 50i .. +49 = the RPC block of i + 120. Keep indices in
  160-494: below 160 the ports are < 10000 (Bench2Drive's README: unsafe), above 494 the TM block reaches the ephemeral
  range 32768-60999, where an outgoing connection can hold a port and CARLA dies at start (`bind: Address already in
  use`, Signal 11). A startup Signal 11 is a port problem until the server log says otherwise. (carla.md, "Traps")
- [ ] **A TM port outlives its server.** b2d_run scans its own 50-port TM block for a free port each attempt.
- [ ] **Stagger server starts.** Launching many at once costs throughput and wedges the NVIDIA driver (2026-09-28: ~50
  starting servers, D-state on the device lock). The pool starts one CARLA job per card per round; b2d_run:
  `--stagger-s 20` and `B2D_START_SLOTS=2` box-wide; cache `nvidia-smi` (the library does).
- [ ] **RenderThread 60 s timeout + Signal 11 during route setup**: the commonest server death (712 of 1558 logs,
  2026-09-24..27), cause open, not flags, not the thread cap. b2d_run retries it; report retries with every score.
- [ ] **Threads, not only cores.** `pids.max` (20480) counts threads. The pool admits jobs from the thread
  model; b2d_run holds a start above `B2D_PIDS_WAIT`. Symptom when exceeded: `RuntimeError: Resource temporarily
  unavailable` at `carla.Client()`.
- [ ] **CARLA ignores `CUDA_VISIBLE_DEVICES`**: the card is `-graphicsadapter` (= CUDA index on this box): pass `{gpu}`.
  The pool still sets `CUDA_VISIBLE_DEVICES` so the agent's model lands on the same card.
- [ ] **NUMA.** Cards 1 and 2 are on NUMA node 1 (CPUs 52-103, 156-207), card 0 on node 0. The pool pins
  NUMA-local physical cores first (`nvidia-smi topo -m`).
- [ ] **Never `pkill -f` / `pgrep -f`**, and never `os.getpgid()` of a recorded pid that may be reused: stop recorded
  identities (`jevdrive.cl.procs`, `experiments/night_queue_4/archive/cx_owned_process.py`) or the process group b2d_run created.
- [ ] **`SIGTERM` does not stop a server promptly**; kill the group, wait for it to empty, then `SIGKILL`.
- [ ] **`-quality-level=Low` with `-RenderOffScreen` segfaults; `SDL_VIDEODRIVER=offscreen` breaks CARLA.**
- [ ] **Closed loop is not bitwise deterministic.** Two identical runs differ; 3 of 20 PDM-Lite routes flip DS between
  identical runs. Compare arms against the spread of identical baselines, not against zero.
- [ ] **A score over the routes that happened to finish is a score on a selected subset**: read `summary.json`
  (`routes_never_finished`, `restarts`) before quoting one.
- [ ] **Memory.** Keep application memory under ~85% of `memory.max`; unexplained SIGKILLs come from the platform
  (closed-loop-acceptance.md, "SIGKILL").

## Where the detail lives

- [carla.md](carla.md): Vulkan fix, server start/stop, traps with their incidents, the thread census and the 2026-09-27
  stock/reduced equivalence test, Town12 / Large Maps.
- [bench2drive-cost.md](bench2drive-cost.md): per-tick cost decomposition, the servers-per-GPU ladder (GPU knee),
  the 220-route round, reliability and recycling.
- [closed-loop-acceptance.md](closed-loop-acceptance.md): which controllers and harness parts are accepted.
- [long-runs.md](long-runs.md): tmux, run directories, sharing the box.
- [fc65452:todos/2026-10-01-cl-lib.md](https://github.com/VennIntelligence/jev-drive/blob/fc65452/todos/2026-10-01-cl-lib.md): the profile experiment's pre-registration, log and raw
  tables.

Last verified: 2026-10-05
