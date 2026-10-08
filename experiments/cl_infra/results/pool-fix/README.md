# pool-fix: why the box sat idle, and what the pool's accounting change does (2026-10-08)

Source: a copy of the pool spool (`runs/pool/state.json`, `status.json`) and the box sampler
(`runs/boxwatch/<day>.tsv`), read offline by [`../../scripts/pool_usage.py`](../../scripts/pool_usage.py). Full output:
[2026-10-08.md](2026-10-08.md) (00:00-11:07, the day of the complaint), [2026-10-07.md](2026-10-07.md) (a full day).
Box: 3 cards (83.6 GiB each), cgroup quota 75 cores, 276 GiB, `cpu_overcommit` 1.4.

## Diagnosis, 2026-10-08 00:00-11:07

"Card busy" = a pool job with >= 1 GB measured VRAM on it (the pool recorded no utilisation before this change).

| | available | used | idle |
|---|--:|--:|--:|
| card-hours | 33.4 | 18.8 (56 %) | 14.6 |
| core-hours | 834 | <= 357 (43 %; sum of per-job peak cores, an upper bound) | declared: 507 |
| VRAM GB-hours | 2658 | 524 at measured peak | declared: 785 |
| RAM GB-hours | 3071 | 461 measured non-reclaimable | declared: 1493 |

| idle card-hours by cause | h | share |
|:--|--:|--:|
| nothing queued, every card idle (no lane had a ready GPU job in the pool) | 7.88 | 54 % |
| serial in lane (`wax-run`: 5 inference runs in one job, 91 min on one card) | 2.73 | 19 % |
| nothing queued, other card(s) busy | 1.97 | 13 % |
| a ready GPU job was queued (RAM gate 1.04, CPU budget 0.15, dispatch latency 0.14, not reconstructed 0.30) | 1.64 | 11 % |
| CPU-bound stage running (>= 60 % of the quota), no GPU job ready | 0.37 | 3 % |

Over-declaration and placement do not empty a card; they keep jobs in the queue while every card has something on it.
Ready GPU jobs waited 17.4 job-hours (224 jobs, 89 waited > 2 min):

| blocker while a ready GPU job waited | job-hours | share |
|:--|--:|--:|
| RAM gate: the declared RAM of every job younger than 5 min was added on top of the measured cgroup memory | 8.64 | 50 % |
| VRAM over-declared (the job fits by measured peaks x 1.2 + 1 GB) | 5.81 | 33 % |
| VRAM really full | 1.92 | 11 % |
| not reconstructed | 0.47 | 3 % |
| dispatch latency (<= 1 min), CPU budget, training cap | 0.57 | 3 % |

What the leads from the brief came to:
- **Serial in lane**: confirmed for `wax-run` (2.7 idle card-hours). Seed siblings submitted only after the first seed
  ended were found five times (`mxd-e-MX-F`, `wsl-e-SH30-F`, `wsl-e-P2H10-F`, `wods-WLG-full` x2), 1-23 min each.
- **Over-declared VRAM**: `self-consist` declared 24 GB and peaked at 21.6 (not over-declared); the large cases are
  `bn-hugsim` workers (59 GB declared, 37-45 peak, some 2.9), `wax-run` (14 vs 5.7), `mxd-e` / `wsl-e` / `wodl-e`
  eval jobs (8 vs 1.9), `mxd-t` / `sh-t` / `m10-t` training (40 vs 24.3).
- **Uneven placement**: no measurable idle time. One real defect: CPU-only jobs (`vram_gb` 0.5) counted as load on
  the card they were parked on and made it "not idle" for the work-conservation rule.
- **CPU**: `wax-render` declared 48 cores and measured 1.0; CPU-only jobs were queued for 2.07 h of the window. The
  CPU budget blocked GPU jobs for 0.13 job-hours only.
- **Not recorded** (findings in themselves): per-card utilisation over time, why a job waited (only the last reason
  was kept, and cleared at launch), peak RSS per job. All three are recorded from this change on.

2026-10-07 (24 h): 72.0 card-hours, 26.3 used, 45.7 idle; 34.2 of the idle hours had nothing queued on any card and
5.1 were a CPU-bound stage. Ready GPU jobs waited 66.8 job-hours, most of it real contention (three 6-server CARLA
jobs at 46-50 GB per card next to 38 queued 25 GB HUGSIM workers); about 13 of those job-hours are not reproduced by
the placement rules and are not explained (a head-of-queue reservation by priority-3 jobs is the suspect; no record).

## Replay: the same submissions through the old and the new accounting

Recorded submit times, `--after` chains, run times and measured peaks, driven through `jevdrive.cl.pool`'s own
`choose` / `vram_charge` / `cpu_charge` / `ram_reserve`. History starts empty. Per-job RAM use is not on record:
every job gets 0.31 x its declared RAM (the day's measured mean over the declared mean). Submit times are fixed, so a
lane that would have submitted its next stage earlier gains nothing: a lower bound.

| 2026-10-08 replay | GPU-job wait, job-hours | GPU jobs waiting > 2 min | all-job wait, job-hours |
|:--|--:|--:|--:|
| recorded on the box | 18.4 | 89 | - |
| old accounting (declared) | 14.4 | 71 | 40.1 |
| new accounting (measured, history from empty) | 9.7 | 45 | 29.1 |
| new accounting + bench history per stage kind (`CL_HIST_KEY`) | **9.7** | 46 | **27.4** |

The old-accounting replay reaches 78 % of the recorded wait (14.4 of 18.4): the simulator has no head-of-queue
reservation, PID gate or port search. New vs old: GPU-job wait -33 %, all-job wait -32 %; the RAM gate falls from 10.1
to 1.1 job-hours and the remaining wait is the CPU budget (4.2), VRAM (3.3) and the training cap (0.9).
`wax-run` fanned out: 91 min on one card -> 17 min from first start to last end on three.
On 2026-10-07 the same change moves GPU-job wait from 54.1 to 51.7 job-hours: that day was VRAM-full, not mis-booked.

Rejected after the replay: keeping 20 % of the CPU budget for GPU jobs. GPU-job wait 9.7 -> 8.8 job-hours, all-job
wait 27.4 -> 51.9 (the CPU-only scoring stages are the same lanes' next step).

## What changed

`jevdrive/cl/pool.py`, `__main__.py`, `jevdrive/bench/runner.py`; rules in
[docs/closed-loop-runbook.md](../../../../docs/closed-loop-runbook.md#the-gpu-pool) and
[docs/long-runs.md](../../../../docs/long-runs.md) (fan-out rule). Not verified live: the dispatcher was not restarted
on the new code before the box reboot of 2026-10-08; an out-of-memory margin of the tapered RAM reservation under a
real load; per-job RSS (the sum over a process tree counts shared pages once per process).
