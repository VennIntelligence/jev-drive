# The SIGKILLs that are not the kernel OOM killer: trigger, reproduction, fix (INFRA2, 2026-10-10)

**Result.** The platform kills the largest process of the container (by RSS), again every 2-5 s, while

    memory.current - inactive_file  >=  0.98 x memory.max        (541.0 of 552 GiB on the 6-card instance)

The sender is outside the container: the kernel counters stay at 0 (`memory.events` oom / oom_kill, host
`/proc/vmstat` oom_kill), the pool logged no stop, and the root processes inside (`autopanel`, supervisord, sshd,
jupyter, tensorboard, proxy) hold no cgroup or kill code (`strings` of the readable `autopanel` binary: no
`memory.stat`, `memory.current`, `inactive_file`, `SIGKILL`). Which host program sends it stays unknown; the rule it
follows is measured.

## Evidence

**1. The night of 2026-10-10** (`kills.csv`, `scripts/sigkill_trigger.py --day 20261010`; box sampler every 5 s).
28 tries of 18 TR1 jobs ended with rc 137 between 00:19 and 01:41 (24 of 32 WOD eval launches, 4 of 6 training
launches; one job needed 14 starts), plus four AlpaSim stacks that lost a worker (rc 1).

| | |
|:--|:--|
| victim | in the sampler's top 3 by RSS in 25 of 28, rank 1 in 19 (the others: the kill fell between two samples); 6.8-32.7 GB; age 10 s - 16 min: WOD evals 10-90 s into loading their model (27-34 GB in 20 s), and old jobs when a newcomer grew |
| `memory.current`, max over the 30 s before | 535.9-550.0 GiB (97.1-99.6 % of `memory.max`); no kill in the 1 480 samples below 530 GiB |
| application memory (anon) at the kill | 102-216 GiB: 18-39 % of the limit |
| `memory.high` events in the 30 s before | 0 in 4 kills (01:38-01:40), 136 - 195 152 in the others |
| not sufficient | `memory.current` >= 536 GiB with the largest process >= 15 GB and no kill: 01:28:10-01:34:13 (6 min, 139 001 high events, anon 95 -> 113), 00:42:17-00:44:58, 01:46:03-01:49:40 |

So neither application memory, nor `memory.current` at `memory.high`, nor reclaim pressure separates kills from
quiet minutes. The sampler did not record the split of the page cache; the next step measured it.

**2. Canaries** (`scripts/canary.py`, `canary_fork.py`; 1 s sampler `scripts/memwatch.py`; box otherwise at 2
inference jobs of 3 GB). `ws` = `memory.current` - `inactive_file`.

| test (02:34-02:48) | state reached | killed |
|:--|:--|:--|
| 1 process, 60 GiB anon at 3 GiB/s, held 90 s | current 535-550, 9 483 high events, ws 494-511 | no |
| 1 process, 110 GiB anon at 3 GiB/s through `memory.high`, held 45 s | current 550 for 70 s, 197 k high events, PSI full 2.6 s, inactive_file 40 -> 13.4, **ws 536.6** | no |
| 30 GiB parent + 12 forked idle children | host Committed_AS 564 GiB (CommitLimit 377), summed RSS 390 GiB | no |
| 2 WOD eval jobs (the killed kind) + 56 GiB filler | current 550, inactive_file 17 -> 9.6, **ws 540.4** at the last sample; host free 95 GiB | **yes**, one job at 17 s (15 GB, rank 1) |
| 4 processes x 12 GiB anon at 2 GiB/s each, next to the surviving WOD job | inactive_file 16.8 -> 8.9, **ws 540.8, 540.9, 541.1** in three consecutive 2 s samples | **yes**: the WOD root (largest, 10 GB), 3 s later the largest canary |

0.98 x 552 = 540.96 GiB. The line does not need a GPU process, fast growth, pressure or many processes: it needs the
working set there. It was missed for two weeks because it is not `memory.current` (page cache counts) and not anon
(most of the page cache counts too).

**3. Why a warm cache is enough.** A file page read twice moves to the active list. The kernel moves pages back to the
inactive list only when inactive is below about active / sqrt(10 x GiB) (active / 64 at 420 GiB: 6.5 GiB), and it
reclaims only at `memory.high` = `memory.max` - 2 GiB = 550 GiB. On a working night active_file was 400-450 GiB and
inactive_file 9-17 GiB, so ws = current - 9..17: at `memory.high` the working set is 533-541 GiB, and once inactive is
under 9 GiB the kill line (541 + inactive) lies below `memory.high`: the platform kills before the kernel reclaims
anything (the four kills without a high event). After one such episode every job that brings `memory.current` back
to ~548 is killed again: the 14 starts. Doubling the quota changed nothing because the cache fills any quota (949
files >= 16 MiB held 276 of 286 GiB of it: AlpaSim scenes 153, op_parity token caches 61, NuRec scenes 34).

## Fix (commit 79419aa2, dispatcher restarted 02:58:28)

No cgroup knob is available (kernel 5.15: no `memory.reclaim`; `/sys/fs/cgroup` read-only; `vm.drop_caches` needs
root). `posix_fadvise(POSIX_FADV_DONTNEED)` on a readable file drops its clean, unmapped pages from either list:
2.39 GiB in 0.147 s, `memory.current` and active_file both down by 2.39 GiB. The pool (`jevdrive/cl/cache.py`,
`pool.py`; docs/closed-loop-runbook.md, "Memory: the kill line"):

- holds a start while ws + unallocated RAM of jobs younger than 5 min + the job's RAM + 16 GiB > the line;
- drops page cache (idle files first) to keep 80 GiB below the line plus what young and held jobs need;
- retries a job whose root died of SIGKILL without counting it (`kill_retries` 3), logs a `kill` event with the memory
  picture, and shows kills per day in `cl top` / `cl usage`.

**Before / after, same load** (56 GiB of anonymous filler, WOD eval jobs `wod_zeroshot_openpilot.py --workers 12`
with a served ONNX, `--ram 40`, cache warmed to active_file 434-441 GiB):

| | jobs | killed | notes |
|:--|--:|--:|:--|
| TR1 night, 00:19-01:41 | 38 launches | 28 | 24 / 32 WOD eval tries, 4 / 6 training tries |
| reproduction, old dispatcher, 02:43 | 2 | 2 | first at 17 s; margin at launch 33 GiB |
| new dispatcher, 03:02 | 4 | 0 | starts spaced 42 / 21 / 41 s by the working-set rule (waited 42-104 s), three trims (62 -> 141, 86 -> 143, 97 -> 162 GiB margin; 48 s the first with the file walk, then 3 s), minimum margin 48.9 GiB, anon peak 187 GiB |

## Open

- The fraction 0.98 is measured on one instance (552 GiB); the 10-08 series on the 276 GiB instance fits it
  (`memory.current` 274.0 at the kills, line 270.5). Re-check after an instance change: `cl top` shows the margin, a
  `kill` event carries the working set at the kill.
- Page cache in files under 16 MiB (camera JPEGs) is not on the trimmer's list; when it cannot make room a held job
  starts after 10 min (`ws_override`).
- A worker killed inside a job whose root survives (AlpaSim) ends as rc 1 and is not counted as a kill.
- Jobs outside the pool are not held by the rule; the trimmer still keeps 80 GiB for them.
- `pp_train.py` / `ap2_train.py` write no checkpoint before the last step, so a killed training restarts from step 0.
  Missing for a resume: a periodic checkpoint with `model.state()`, the AdamW and GradScaler state and the step; the
  row-stream generators (`rng`, `mrng`, `lrng`) as of the consumed step (`Prefetch` draws `--prefetch` batches ahead
  on the main thread, so their state at save time is ahead: store it per drawn batch or replay `draw()` from the
  seed); the geometry tokenizer's parameters when attached; the `Run` directory continued instead of recreated. The
  learning rate is a function of the step alone.
