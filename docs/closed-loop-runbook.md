# Closed-loop runbook (CARLA / Bench2Drive)

Read this first whenever you run anything closed-loop on the GPU box: a Bench2Drive exam, a CARLA data generation, a
controller campaign. It is the authoritative entry point: which tool to launch with, how many workers per card, which
thread settings, and the traps. Detail and history live in the docs it links; where they disagree, this page wins and
says why.

## Entry points

| Need | Use |
|---|---|
| what the box has right now (cards, quota, pids, NUMA, who holds what) | `.venv/bin/python -m jevdrive.cl probe` |
| reserve cards, cores and server indices for a lane | `python -m jevdrive.cl lease <lane> --gpus 2` (writes the row in `runs/sched/table.tsv`) |
| run a list of jobs across the leased cards | `python -m jevdrive.cl run <lanefile.py>` inside `scripts/tmux_run.sh` |
| one card's worth of routes, by hand | `scripts/b2d_run.py` (the per-card runner every lane calls) |
| one CARLA server, by hand | `scripts/carla_server.sh start <index>` (docs/carla.md) |
| the schedule table | `python3 scripts/sch_table.py show / check / finish` (now on the same constants as the library) |
| watch / stop a lane | `python -m jevdrive.cl status|drain|stop <root>` |

The library is `jevdrive/cl/` (module map in its `__init__.py`); tests in `tests/test_cl.py` (`python -m unittest
tests.test_cl`; the lane tests need Linux and run on the box).

## The box is elastic: never hardcode its size

The instance changes between sessions: seven cards / 175 cores / 644 GiB on 2026-09-28, three cards / 75 cores /
276 GiB on 2026-10-01. Every sizing number below is either measured live by `jevdrive.cl.box.probe()` (cgroup
`cpu.max`, `pids.max`, `memory.max`, affinity, NUMA nodes, `nvidia-smi`, CARLA servers per card from `/proc`) or derived
from it in `jevdrive.cl.capacity`, with our best defaults as fallback, and every one is overridable on the command line
(`--gpus`, `--cpus`, `--idx0`, `--workers-per-card`, `--profile`, `--num-threads`, `--client-threads`, `--pool-threads`).

Defaults derived per card: core slice = quota / cards (25 on the 3-card box), workers per card = min(GPU knee 6, slice /
cores per worker, (VRAM - 8 GB) / 9 GB), PID plan cap = 0.80 x `pids.max`, b2d_run's start gate (`B2D_PIDS_WAIT`) =
0.85 x `pids.max`, server indices 160-494 (RPC port >= 10000, TM block below the ephemeral range).

## Worker profiles and the chosen default

A worker is one CARLA server plus one route client. Its thread settings live only in `jevdrive/cl/profiles.py`:

| Profile | CARLA pools (`-RPCThreads/-StreamingThreads/-SecondaryThreads`) | `carla.Client` threads | OMP/MKL/OPENBLAS/NUMBA |
|---|---|---|---|
| `stock` | CARLA default: (host CPUs - 2) / 3 = 68 each on the 208-CPU host | default: one per host CPU (208) | unset |
| **`reduced` (default)** | 4 each | 8 | 2 |

**Decision (2026-10-01, pre-registered rule, [todos/2026-10-01-cl-lib.md](../todos/2026-10-01-cl-lib.md)): `reduced`
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
- The library reproduces the old `carla_threads_routes.sh` lane: 20/20 routes, mean DS 93.3 (old runs 90.6-95.3), DS
  differences only on the known flaky routes 1956 / 3564 / 17563.


`OPENBLAS_CORETYPE` is not a profile knob but a correctness setting of particular envs: `Haswell` for the NAVSIM devkit
on the GPU box (without it NumPy's OpenBLAS returns wrong results there, research/navsim-openblas-audit.md),
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
| cores per worker | 2.0 / 1.2 | 7-card box 2026-09-28 | 24-core slice | SimLingo, TFv6, BLUE / PDM-Lite (G lane `CPU_A`) | scripts/nq4_g_lane.py |
| cores per worker | ~1.0-1.4 used (11-12 of 25 cores busy at 8-12 workers) | 3-card box 2026-10-01 | 25-core slice | PDM-Lite, reduced, client 8 | this experiment |
| servers per card (GPU knee) | 6 | 5-card RTX PRO 6000 2026-09-25 | - | Town12, Alpamayo camera rig: 34.8 aggregate ticks/s at 6, 38.2 at 12 | bench2drive-cost.md |
| servers per card | 6 (45.8 FPS aggregate, 8.0 GB each) | 7-card RTX 6000D 2026-09-28 | 16 CPUs | six 1600x900 cameras, reduced | remote-box.md |
| VRAM per server | 5.1-6.3 GB Town12, 8-10 GB six cameras | 2026-09-25 / 27 | - | - | bench2drive-cost.md, carla.md |
| numeric threads (OMP...) | 1 / 2 / 4 used by different lanes | - | - | never compared before | this experiment |
| `pids.max` | 20480 | every instance so far | - | counts threads | probe |

## Launching a lane on the library

1. `python -m jevdrive.cl probe`: check what is free (cards with no row, no compute process, no CARLA).
2. `python -m jevdrive.cl lease my-lane --gpus 2 [--cores-per-card 25] [--workers-per-card 6]`: picks free cards,
   NUMA-local physical cores no other row holds, and server-index blocks clear of every live row (index i's TM ports are
   the RPC block of i + 120), and writes the row. Say what the lane is in `--status`; record it in the lane's todo.
3. Write the lane file (one Python file, ~50 lines; example `scripts/lanes/cl_worker_profile.py`):

   ```python
   from jevdrive.cl import b2d
   NAME, ROOT = "my-lane", "my_lane"                    # ROOT is under $DATA_DIR/runs
   def jobs(args):
       ids = ["1711", "1773", ...]
       return [b2d("seed%d-shard%d" % (s, k), f"/root/autodl-tmp/ujs/runs/my_lane/s{s}", ids[k::4], tm_seed=s,
                   agent="scripts/my_agent.py", agent_config="cfg.json", python=".../envs/x/bin/python",
                   workers=3, vram_gb=14, min_done=0.95)
               for s in range(3) for k in range(4)]
   ```

   A job is any command with placeholders (`{gpu} {idx} {span} {workers} {cpus} {server_args} {client_threads}
   {pids_wait} {out} {job_dir}`); `b2d()` builds a `scripts/b2d_run.py` invocation. Per job: `workers`, `vram_gb` per
   worker, `deps`, `tries`, `profile`, `env`, `exclusive`, `gpus`, `priority`, `ok` (output check).
4. Stage it (CLAUDE.md "Before a long run"): `run ... --only <one job>` with a one-route job, inspect; then ~10 units;
   then the batch. `--dry-run` prints the commands.
5. `scripts/tmux_run.sh my-lane .venv/bin/python -m jevdrive.cl run lanefile.py`. Hand-off files in the root:
   `STATUS` (one sentence), `status.json` (incl. why each card is blocked), `DONE` / `ERROR` / `ERROR.<job>`,
   `util.csv`, `lane/<ts>/{log.txt, events.jsonl, tb/}`; per job `jobs/<name>/{log.<k>.txt, job.<k>.json, rc.<k>,
   owned.json}`.
6. Finish: `python -m jevdrive.cl release my-lane "done: ..."` (archives the row), close your tmux window.

What the lane does for you: dynamic pull of jobs onto cards with room (workers, measured free VRAM minus what young jobs
will still take, PID room from the thread model), one launch per card per round (servers start staggered; b2d_run also
holds one of `B2D_START_SLOTS` box-wide start slots), retries (b2d_run resumes its `--out`, so only unfinished routes
re-run), the lease re-read every round (a card leaving the row takes no new job; a revoked row drains), drain
(`<root>/DRAIN` -> every job's `B2D_DRAIN_FILE`, routes in flight finish), cleanup of each job's process tree by
(pid, start time) through a pidfd, including processes re-parented to init (they carry the job's `CL_TOKEN` env entry),
and adoption of live jobs when the lane process restarts. Nothing is ever found or killed by pattern.

## Traps checklist

- [ ] **Ports.** Index i: RPC 2000 + 50i (+1, +2), TM 8000 + 50i .. +49 = the RPC block of i + 120. Keep indices in
  160-494: below 160 the ports are < 10000 (Bench2Drive's README: unsafe), above 494 the TM block reaches the ephemeral
  range 32768-60999, where an outgoing connection can hold a port and CARLA dies at start (`bind: Address already in
  use`, Signal 11). A startup Signal 11 is a port problem until the server log says otherwise. (carla.md, "Traps")
- [ ] **A TM port outlives its server.** b2d_run scans its own 50-port TM block for a free port each attempt.
- [ ] **Stagger server starts.** Launching many at once costs throughput and wedges the NVIDIA driver (2026-09-28: ~50
  starting servers, D-state on the device lock). b2d_run: `--stagger-s 20` and `B2D_START_SLOTS=2` box-wide; cache
  `nvidia-smi` (the library does, 60 s).
- [ ] **RenderThread 60 s timeout + Signal 11 during route setup**: the commonest server death (712 of 1558 logs,
  2026-09-24..27), cause open, not flags, not the thread cap. b2d_run retries it; report retries with every score.
- [ ] **Threads, not only cores.** `pids.max` (20480) counts threads. The library admits workers from the thread
  model; b2d_run holds a start above `B2D_PIDS_WAIT`. Symptom when exceeded: `RuntimeError: Resource temporarily
  unavailable` at `carla.Client()`.
- [ ] **CARLA ignores `CUDA_VISIBLE_DEVICES`**: the card is `-graphicsadapter` (= CUDA index on this box). The library
  still sets `CUDA_VISIBLE_DEVICES` so the agent's model lands on the same card.
- [ ] **NUMA.** Cards 1 and 2 are on NUMA node 1 (CPUs 52-103, 156-207), card 0 on node 0. Leases take NUMA-local
  physical cores first (`nvidia-smi topo -m`).
- [ ] **Never `pkill -f` / `pgrep -f`**, and never `os.getpgid()` of a recorded pid that may be reused: stop recorded
  identities (`jevdrive.cl.procs`, `scripts/cx_owned_process.py`) or the process group b2d_run created.
- [ ] **`SIGTERM` does not stop a server promptly**; kill the group, wait for it to empty, then `SIGKILL`.
- [ ] **`-quality-level=Low` with `-RenderOffScreen` segfaults; `SDL_VIDEODRIVER=offscreen` breaks CARLA.**
- [ ] **Closed loop is not bitwise deterministic.** Two identical runs differ; 3 of 20 PDM-Lite routes flip DS between
  identical runs. Compare arms against the spread of identical baselines, not against zero.
- [ ] **A score over the routes that happened to finish is a score on a selected subset**: read `summary.json`
  (`routes_never_finished`, `restarts`) before quoting one.
- [ ] **Memory.** Keep application memory under ~85% of `memory.max`; unexplained SIGKILLs come from the platform
  (closed-loop-acceptance.md, "SIGKILL").

## Lane scripts and what the library covers

Audited 2026-10-01. None of the six lane scripts was migrated (they ran or are finished; old lanes are not rewritten).
"Expressible" means a lane file can do it today with `deps`, `ok`, `ready`, `cores`, templated `env` and `more()`.

| Script | Covered / expressible on jevdrive.cl | Still missing in the library | Hazards in the old script |
|---|---|---|---|
| `carla_threads_routes.sh` | **superseded**: `scripts/lanes/cl_worker_profile.py --arg stage=old` reproduces it (verified, 2026-10-01) | - | no lease check, no retries |
| `nq4_g_lane.py` | lease re-read per round, per-card slices / index blocks, staged pilots (`deps` + `ok`), retries, drain, adoption, `min_done=0.9` cell tolerance | demand-file yielding to other lanes; per-agent render-share and cores-per-worker budget (`CAP_A`, `CPU_A`); heavy/light mixing | PID constants 16000 / 700, `B2D_PIDS_WAIT` 16000; fallback layout hardcodes 7 cards |
| `sch_cl_parallel.py` | slot validation (`lease.conflicts`), stage 1 -> 10 via `deps` / `ok`, file gates via `ready` | ESCALATE / auto-SKIP side effects, port bind pre-check, per-arm claim lock | hardcoded GPU 1, index bound 495 |
| `b2d_privileged_chain.py` | phases via `deps`, shard checklists via `ok` (+ `--fail-fast`), retries, util / STATUS / ERROR | provenance lock (sha256 of controls, git commit) with mid-run abort; hard PID-ceiling kill | `B2D_PIDS_WAIT` 17000, hard stop 17500; cards 0-2 / 25 cores hardcoded |
| `op_l_b2d_chain.py` + `op_l_b2d_daemon.py` | priority queue with `deps`, retries, per-slot cores (`cores`), pace-matched follow-ups via `more()`, held-out gate via `ready` | an openpilot server kept alive across units (`KEEP_SRV`): the lane's reaper stops every process of a finished job | daemon uses `pgrep -f` and edits `table.tsv` without the owner lock; chain has no PID admission and `killpg`s a pid file without a start-time check |
| `op_adapt_r2_lane.py` (training, no CARLA) | VRAM packing, per-job core slices, DONE / ERROR / STATUS, retries, checklist `ok` | side jobs that hold no slot; cross-arm batch stop | children not recorded (no cleanup); uncached `nvidia-smi` per decision |
| `b2d_tfv6_campaign.py`, `b2d_tcp_campaign.py`, `b2d_controller_campaign.py` (Tokyo) | jobs + retries, logs, owned cleanup | one server reused across groups; frozen-protocol / provenance checks; windowed Tokyo mode | `DATA=/data` hardcoded; server indices 70-101 (ports < 10000) outside any lease |

Open library work, in order: keep-alive services across jobs (a `keep` flag the reaper honours), demand-file
yielding and per-agent CPU / render budgets, provenance locks, a port bind pre-check.

## Where the detail lives

- [carla.md](carla.md): Vulkan fix, server start/stop, traps with their incidents, the thread census and the 2026-09-27
  stock/reduced equivalence test, Town12 / Large Maps.
- [bench2drive-cost.md](bench2drive-cost.md): per-tick cost decomposition, the servers-per-GPU ladder (GPU knee),
  the 220-route round, reliability and recycling.
- [closed-loop-acceptance.md](closed-loop-acceptance.md): which controllers and harness parts are accepted.
- [long-runs.md](long-runs.md): tmux, run directories, the schedule table rules.
- [todos/2026-10-01-cl-lib.md](../todos/2026-10-01-cl-lib.md): the profile experiment's pre-registration, log and raw
  tables.

Last verified: 2026-10-01
