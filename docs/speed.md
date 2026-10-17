# Speed: every throughput recipe in one place

Read this when a job is slow, before you write a faster path, or before you change a default that touches speed. One table per
stage; each row says where the recipe lives, what it measured, whether results stay identical, and whether it is on by default.
"Identical" means bit for bit (same scores, plans, weights) unless the row says otherwise. A recipe that changes results
needs a bit-identity check (or an explicit, recorded decision to accept the change) before it becomes a default.

The headroom measured on 2026-10-17 is at the end ([Headroom](#headroom-2026-10-17)).

## Data prep (caches, scoring)

| Recipe | Where | Gain | Identical | Default |
|---|---|---|---|---|
| Slot warps on the GPU in the cache builders | `pp_prep.py --synth gpu` (also `ap2_prep.py`); commit 4a10739a | warp 131 -> 836 pairs/s; ap2 3.5 -> 27 tokens/s (experiments/alpasim/results/lat1_train_audit.md) | yes (`prep_check.py`: 0 of 642 M px) | on |
| Resumable row-wise cache writes | `pp_prep.Part` (open_memmap + done mask) | no rework after a kill | yes | on |
| Process pool + I/O threads | `jevdrive/par.py:70` `pmap` (workers = `n_cpus()`), `shards()` :27; pp_prep uses `n_cpus() - 4` processes + 8 threads | scales with cores | yes | on |
| Idempotent artifact cache | `jevdrive/cache.py:146` `cached` / `key` | skips unchanged work | yes | where used |
| Free graph values after the last reader | `jevdrive/op_torch.py` `OnnxTorch.run`; 7b503a73 | encoder memory at batch 64: 13.8 -> 0.6 GB | yes | on |
| Integer chroma mean, byte gathers | `navsim_zs.OpenpilotMaps`; 7b503a73 | 94 -> 55 ms per token | yes (0 of 7.9 M px) | on |
| NAVSIM scoring: 6 jobs x 12 cores, 1 thread per worker, memoised human IDM | `jevdrive.bench score-poses` (docs/bench.md) | 12 146 tokens x 1 key 11.2 -> 7.2 min; 8 keys 75 -> 43 min | yes under 1 OpenMP thread (inherited `OMP_NUM_THREADS=75` changed 38 rows by <= 3e-11) | on |
| Encoder batch | pp_prep | card-bound from batch 32 (~1 310 pairs/s); larger not worth it | yes | 32+ |

## Training (rm_train, pp_train and the trainers built on them)

All current trainers share `experiments/op_parity/scripts/pp_train.py`'s token store and prefetcher. Precision: the frozen Cinque
weights compute in fp16, the trainable pathway keeps fp32 masters, `torch.amp.GradScaler` scales the loss (`rm_train.py:708`); there is
no autocast. Tokens are memory-mapped on the host (`host=True`) and gathered per batch from the page cache; only small per-row tensors
and the dev rows sit on the card.

| Recipe | Where | Gain | Identical | Default |
|---|---|---|---|---|
| Prefetch N batches on worker threads through page-locked buffers (draws stay on the main thread, in step order) | `pp_train.py:421` `Prefetch`, `Tokens._pinned` :346; `--prefetch 4 --fetch-workers 2` (rm_train :1169) | 4-5 % of a step, 0 on a crowded card (lat1_train_audit.md) | yes (loss curve and ckpt equal) | on |
| Dev rows' tokens held on the card | `--dev-card-gb 4` (`pp_train.py:966`, `dev_store` :413) | dev eval 19.5 -> 4.9 s each (~10 % of a run) | yes | on |
| `torch.compile` of the training step | `pp_train.py --compile` (:857, :967), also ap2 / joint / onp / bd4 / je trainers; **not in rm_train** | 164 -> 76 ms per step on a quiet card (2.15x), 1.9x shared; Tokyo 3090 7.2 vs 4.0 it/s | **no**: gradient differs 1.7 % relative, compiled runs repeat each other bit for bit; navtest / closed loop never checked | off |
| `torch.compile` in rm_train (default since 2026-10-17) | `rm_train.py --compile` (default on; `--no-compile` or `JEV_RM_EAGER=1` = eager) | RSG-L18 recipe 2 500 steps: 1.7x wall, 1.9x steady (197 -> 104 ms per step) | **no**, but within seed noise on dev ADE, held 460 and PAI 281 (experiments/cl_infra/results/compile.md); lanes started before 10-17 pin `JEV_RM_EAGER=1` in their pool `--env` | **on** (new lanes) |
| Hinge rasters stored once per label | `--label-bank` | fits a 24 GB card (memory, not time) | yes | per recipe (on in RB0 / RSG) |
| Two trainings per card | pool `train_per_card=2` (`jevdrive/cl/pool.py:155`) | two on one card: 35-41 min instead of 24-25 each (experiments/alpasim/results/pipeline_profile.md) | yes | on |
| Legacy trainers: DataLoader `num_workers` / `pin_memory` / `prefetch_factor=4` | op_adapt_h, op_route_ft, op_img_cmd, factor_wm | - | - | old code only |

Not used anywhere in the trainers: TF32 / `set_float32_matmul_precision` (the frozen model is fp16 anyway), `cudnn.benchmark`
(several readers set `deterministic=True, benchmark=False` on purpose), `channels_last`, bf16 (bf16 autocast appears only in body1,
corridor, junction_eyes and wajepa probes).

## Closed-loop reads (AlpaSim on nuPlan sh30 and PAI; CARLA)

| Recipe | Where | Gain | Identical | Default |
|---|---|---|---|---|
| `ALPASIM_FAST=1`: every camera kept, unread cameras rendered at 64 px once per rollout, eval video off | `experiments/alpasim/lib/sim_fast.py`, applied by `run_native.py:205`; `pai_box.sh:21` | PAI 2.55x (34.8 vs 88.8 s per scene, shared card), 3x less CPU (docs/alpasim.md, 69638d8d); Tokyo 1.6x. 2/3 of it is the eval video | yes: 36/36 scores and controller traces (box), 6/6 (Tokyo) | on for PAI, off for nuPlan |
| `ALPASIM_FAST=1` on nuPlan | same | 1.00x (MTGS renders all 8 asset cameras anyway; the `full` preset has the video off already) | yes (154/154 held, 150/150 public) | off (no gain) |
| `ALPASIM_FAST=solo` | sim_fast.py | PAI 2.28x | **no**: decision times move 6 us - 30 ms, 26/36 scores identical | off |
| 4 concurrent rollouts, renderer cache 5 | `c0b_chain.OVERRIDES` (`experiments/alpasim/scripts/c0b_chain.py:32`), imported by every lane | VRAM peak p95 26 -> 17.4 GB, RAM -1.7 GB, wall +10 % (in noise): 4 reads per card instead of 3 (research/alpasim-read-profile.md, 5633b19c) | yes (64 scenes; 10/10 chunks to 4 decimals) | on |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, TMPDIR on the data disk | profile `alpasim` (`jevdrive/cl/profiles.py:81`), applied by `run_native.py` to every process | renderer VRAM peak -2.8 GB | yes | on |
| Driver warm-up overlapped with service start | `run_native.py` (5633b19c) | 15-19 s per job (3-4 %) | yes | on |
| Read packing: 4 nuPlan reads per card | reads declare `--vram 20 --cpu 6 --ram 40` under `CL_HIST_KEY=alpasim-read4` (fc86aeb0; e.g. `wait_cap/scripts/nx.sh:201`); PAI `--vram 22 --ram 30` (`pai_box.sh:15`) | see [Headroom](#headroom-2026-10-17): 4 per card costs +12 % wall per read | yes | on |
| Driver: GPU slot warp + fast frame packing | `SH30_SYNTH=gpu` (`experiments/alpasim/lib/sh30_driver.py`) | `drive` 99 -> 35 ms median | yes | on |
| Driver: compiled model passes | `SH30_COMPILE=1` (sh30_driver.py:430) | `drive` 35 -> 18 ms | **no** (plans move ~1 cm; 700 scenes 0.9481 vs 0.9484); `SH30_COMPILE=0` repeats old runs bit for bit; the docker image sets 0 | on (box), off (image) |
| Driver: nvJPEG decode | `SH30_JPEG=nvjpeg` | faster decode | **no** (1 of 700 scenes flips to a collision) | off |
| Persistent inductor / triton / CUDA caches on the data disk | `jevdrive/cl/disk.py` (every pool and tmux_run job) | cold start 60-70 s; warm saving not timed | yes | on |
| On-policy round chunking: 12 rollout jobs, 6 read jobs, labels built beside the tokens | `experiments/onpol_long/scripts/ol_fast.sh` (`OL_NCH`, `OL_RCH`) | critical path 95-110 -> ~45-50 min per round (estimate) | yes off card 0 (48/48 scores, 864/864 plans); card 0 (ECC) is not reproducible, reads avoid it via `OL_SIM_GPUS` | on |
| PAI on Tokyo: one stack per card, `CONC 4` | `pai_eval.sh` | 67-79 scenes/h per card; `CONC 5` no faster; on the GPU box two stacks per card ~1.9x | yes | on |
| CARLA: reduced thread pools (4 / 4 / 4, client 8, numeric 2) | profile `reduced` (`jevdrive/cl/profiles.py`) | threads per server 561 -> 169; tick rate 0.988x stock | DS identical in 95.1 % of pairs vs 94.5 % stock-stock (not bitwise) | on |
| CARLA: 6 servers per card, staggered starts | `capacity.GPU_KNEE = 6` (`jevdrive/cl/capacity.py:26`); `b2d_run.py --stagger-s 20` | knee of the card; staggering 74.2 -> 86.5 FPS (+17 %) | yes | on |
| CARLA agent: `--decimate N`, `--overlap` | `b2d_run.py` | 1.76x / 2.12x / 3.36x at 2 / 4 / 10 (docs/bench2drive-cost.md) | **no** (changes the agent) | off |
| TCP agent CPU overlap | `B2D_TCP_OPTIMIZE=1` (docs/tokyo-python-optimization.md) | +12 % ticks/s | byte-identical inputs | off |

## Downloads

| Recipe | Where | Gain | Identical | Default |
|---|---|---|---|---|
| Ranged parallel HF download from hf-mirror, ModelScope fallback, sha256 check | `jevdrive/hfdl.py` (`download` streams 8, `snapshot` 16) | one mirror stream ~8 MB/s, 8-16 streams ~16 MB/s total; box egress ~16-18 MB/s, more streams are zero-sum | yes (sha256) | on |
| Parallel range requests with resume | `scripts/ranged_get.sh` (a879fcf7); `download_navsim.sh RANGES`, `pai_fetch.sh PAI_RANGES=8` | 16 ranges 1.9 MB/s where one connection is capped ~0.1 MB/s; PAI scenes 4-13 MB/s | yes | per script |
| ModelScope and HF in parallel | docs/network-proxy.md | ~12 MB/s total vs 7 | yes | by hand |
| One camera only | `download_navsim.sh CAMS=CAM_F0` | 1/8 of 1.2 TB | - | per run |
| Mirrors: DaoCloud for nvcr.io; measure Clash nodes over >= 90 s | docs/network-proxy.md | nvcr.io via Clash 0.1 MB/s vs 1.4-3 MB/s mirror | yes | by hand |
| `HF_HUB_DISABLE_XET=1` behind the proxies | `scripts/download_models.sh` | needed to work at all | yes | on |
| Not worth it | `hf_transfer` (3 MB/s, no gain), aria2 (not installed), Tokyo > 8 burst connections (HTTP 429) | - | - | - |

## Headroom (2026-10-17)

Measured on the 5-card box (RTX 6000D 84 GB, 110 cores, 460 GiB) while the pool was busy with other lanes, so every number has
neighbours on the card. Training probes: `experiments/cl_infra/scripts/train_probe.{py,sh}` (pool jobs `speed-probe-*`, out
`$DATA_DIR/runs/speed/train_probe/`); read and pipeline numbers come from the pool's own `state.json` / `usage.jsonl`.

### Where the card-hours went (last 24 h)

108 card-hours: 37 % computing, 11 % a GPU job on the card that was not using it, 49 % no GPU job at all (`cl usage --hours 24`).
Of the 52 idle card-hours, 36 had an empty queue (05:00-08:00 all five cards idle) and 16 had only `--after`-chained work waiting.
Job-hours: PAI read chunks ~52, nuPlan held / public reads ~25, on-policy rollouts ~17, trainings 4.9 (reads pack 3-4 per card). **The largest lever is keeping the
queue fed, not making a step faster.**

### Training (rm_train, RSG-L18 recipe, 300 steps, eager unless stated)

| Probe | ms / step (median) | it/s | vs base | Identical to base (`nx.py bitcmp`) |
|---|--:|--:|--:|---|
| base (two runs) | 195.7 / 196.6 | 5.11 / 5.09 | 1 | base-a = base-b bit for bit |
| time waiting for the prefetched batch | 0.25 | - | 0.13 % of a step | - |
| `--fetch-workers 4 --prefetch 8` | 200.4 | 4.99 | 0.98x | yes |
| loss scalars kept on the card until the log step, row indices uploaded from pinned memory ("nosync") | 199.9 | 5.00 | 0.98x | yes |
| `torch.compile(model, dynamic=False)` on the training forward | 98.6 | 10.14 | **1.98x** | **no**: weights differ by up to 5.3e-3, logged scalars by 6.2e-3 after 300 steps |
| `--batch 256` | 281.3 | 3.56 | 1.39x samples/s | no (recipe change) |
| 2 copies on one card | 345 each | 2.90 each | 1.13x card total, 0.57x per job | - |
| 3 copies on one card | 491 each | 2.03 each | 1.19x card total, 0.40x per job | - |

Profile of 20 steps (torch.profiler): GPU kernels busy 63 % of the wall time; 121 `cudaStreamSynchronize` and 455 `aten::item`
per step (1.15 s of CPU time blocked in syncs over the 20 steps). The step is not data-bound, and it is not the ~15 logged loss
scalars (removing them changed nothing): what is left is dispatch: ~2 970 kernel launches per step (59 373 `cudaLaunchKernel` in 20 steps), with
`aten::copy_` (dtype casts and copies of the ported fp16 graph, 760 per step) the largest GPU op at 24 % of kernel time. Of the
121 syncs per step, 41 come from Python (`train_probe.sh syncs`: 23 logged loss floats at `rm_train.py:1034`, ~16 `if tensor` tests
in `loss43.py` / `pp_train.py` / `drivable_hinge.py`, the finite check); the rest are inside torch ops (boolean-mask indexing,
pageable uploads). The 455 `aten::item` per step are mostly AdamW's CPU step counters (no device sync). This is why compile, which
fuses the small kernels, doubles the step, and why removing Python syncs alone does nothing.

Reading: one training uses the card at ~65-80 %; two or three trainings on one card buy only 13-19 % more total steps while each runs
1.8-2.5x longer, so packing trainings lengthens every chain that waits on them. `torch.compile` is the only large single-job gain,
and it changes results.

### Reads (AlpaSim nuPlan sh30, PAI)

| Measure | nuPlan sh30 held chunk (~77 scenes) | PAI chunk, `ALPASIM_FAST=1` |
|---|---|---|
| wall vs jobs on the card (pool history since 10-13) | 1: 10.2 min (n 8); 2: 8.2; 3: 10.7; 4: 11.4 (n 150); 5: 11.9 (n 18) | s / scene 43 (1 job), 39 (2), 39 (3), 44 (4); n 88 |
| card throughput | 6 -> 21 -> 24 chunks/h at 1 / 4 / 5 per card | 83 -> 304 scenes/h at 1 / 4 per card |
| booked per job | 18.1 GB VRAM (history `alpasim-read`, p95 x 1.1 + 1), RAM 25 GB | 22 GB VRAM (declared), RAM 30 |
| measured per job | VRAM peak p95 15-17 GB, 2-10 GB in the first minutes; anonymous RAM 7-20 GB (driver 3, renderer 2.5, runtime workers 2 x 3-4 GB) | VRAM peak 19-25 GB; RAM ~20 GB |
| card with 4 GPU jobs (72 h, n 1 728 card-minutes) | VRAM used p50 37 / p99 55 / max 59 GB against 72 booked; util p50 52 % | |
| what binds | VRAM booking (4 per card) and box RAM: the pool's 85 % line is 391 GB, ~20 GB per read gives ~19 reads box-wide | VRAM booking (3 per card) |

Reading: reads do not contend on the GPU (wall +12 % from 1 to 4 per card for sh30, flat for PAI), so per-card read throughput
rises almost linearly with packing until VRAM or RAM runs out. Today the two limits meet at ~4 sh30 reads per card x 5 cards. The
`start_limit` (4 starts per 20 s round) cost 0.02 card-hours in 24 h; it is not a limit.

### Headroom table

| # | Item | Expected gain | Identical | Effort | Check before it becomes a default |
|---|---|---|---|---|---|
| 1 | Keep the queue fed: submit every independent arm's training at the start of a lane (arm B trains while arm A reads), queue the next round's base training at lower priority as backfill, keep gates in the queue (`--submit` gate jobs, as `nx.py acgate`) | up to the 52 idle card-hours a day (36 empty-queue + 16 chain gaps): ~1.5-2x box output when lanes have work | yes (scheduling) | low-medium, per lane script | none (no result changes); watch `cl usage` "no work queued" and "serial" |
| 2 | One pool history key per trainer class (`CL_HIST_KEY=rm-train` for every rm_train arm, as reads use `alpasim-read4`) | trainings start under RAM pressure: they declare `--ram 70` and measure 3 GB; per-arm names never reach the 3 runs history needs | yes | trivial (`nx.sh` `FW` / `FK` and the other lane scripts) | none |
| 3 | Book a card by the measured p99 of its concurrent sum instead of the sum of per-job p95 caps (or lower the sh30 read cap toward 15 GB) | PAI: 4 per card instead of 3 (+33 % per card; PAI chunks waited 24.8 job-hours on VRAM and 12.0 on card reservations in 24 h); sh30: 5 per card only while box RAM has room (+25 % per card) | yes (packing; reads are bit-identical across cards except card 0, ECC) | low-medium (pool.py `vram_need` / `vram_charge`; the vram_stop / OOM requeue already guard the tail) | one batch at 5 sh30 / 4 PAI per card: wall per read <= +15 %, no OOM requeues; scores of the batch equal a 4-per-card read |
| 4 | Less RAM per read | would lift the ~19-reads box ceiling, but the cheap knob is spent: `runtime.nr_workers=1` saves only 0.8-0.9 GB per read for +7 % wall ([research/alpasim-read-profile.md](../research/alpasim-read-profile.md) arms B / D); the rest is the third-party stack's own working set | - | high (third-party) | not recommended; RAM per read stays the box-level ceiling for sh30 |
| 5 | `torch.compile` in rm_train (and on_train) | 1.98x per training (9 -> ~5 min for a 2 500-step L18 round, 35 -> ~18 min for 10 000 steps) | **no** | low code (pp_train's `--compile` pattern) | a recorded decision; compiled vs eager seed spread on dev, plus one held 460 and one PAI read of a compiled checkpoint against its eager twin. Compiled runs reproduce each other, so a lane can switch at a boundary |
| 6 | Replay the eager step as a CUDA graph (same kernels, same order: no fusion, so no new numerics); needs the `if tensor` tests in the losses and the boolean-mask indexing made static first | up to ~1.5x (kernels busy 63 % of the step; the rest is launch / dispatch) | expected yes, must be shown | high | `nx.py bitcmp` on 300 steps, as for "nosync" above |
| - | Not worth doing | more fetch workers / prefetch (0.98x); per-term loss syncs (0.98x); 2-3 trainings per card (per job 0.57x / 0.40x); `start_limit`; `--batch 256` (recipe change for 1.39x samples/s) | | | |

### RAM, CPU and kills (2026-10-17 evening)

Symptom at 17:17: 24 running, 36 queued (34 GPU-ready held on RAM), working set 323 of 451 GiB, load 90-100, six platform
kills that day. Audit 17:36-17:55: process census by job tree, cgroup `memory.stat`, pool events, boxwatch.

| Question | Finding |
|---|---|
| Orphans, leaks | None. The 352 processes outside running pool trees held 1.1 GiB and 0.1 cores (the trainval download's 322 curls, autopanel, jupyter); the pool stopped every killed job's tree (`stop` events); /dev/shm 1.7 MB; page cache 61 GiB, 46 of it inactive |
| RAM per class (anon + shmem of the tree; summing RssAnon over-counts forked workers: 303 GiB summed against 255 cgroup anon) | sh30 held read 13-15 GiB (declared 40, booked 25.2 = 1.2 x history max 20.2 + 1); pub read 10-16 GiB, growing to 20; waiting-set read 12-22 GiB (declared 30); static read 7-10 GiB; PAI chunk 11-15 GiB (declared 30, booked 19.5); rm_train 3-6 GiB anon plus 31 GB file-backed RSS (declared 70-90, booked 8) |
| Why 34 jobs were held on RAM | A burst: the NR1F s1 chain's ~20 reads and 8 PAI chunks started into 24 running jobs. For 5 min each start adds the RAM it has not allocated yet under the 85 % anon rule (391 GiB). The hold cleared in ~20 min (17:36: none held on RAM, the rest wait on VRAM). In 12 h no card ever sat idle on RAM ("queued: ram" 0 card-hours): the RAM holds limited packing, they did not idle cards |
| Kills | 15:41:33: CPU-only `df-lat2-build` (declared 160) grew to 344 GiB; cgroup anon went from 304 to 408 GB in 50 s. 16:03:48: `an-x-a_pub` (declared 48, used 240 GiB) and `an-x-pool_rb0` (48, used 67), both started by `ws_override` after 10 min held. Both times the platform killed the three trainings (45-58 GB RSS each, 6 GB anon) and some AlpaSim runtime workers. Five reads failed with rc 1 at the kill times and were not retried. Three logs show "Worker 0 died with exit code -9" (`df-ws-RTB1AR-F-s0`, `df-pub-NR1H0P-L18-s0-6` / `-7`); the logs of the other two (`ws-read-starter`, `ws-read-replay`) were overwritten by later reruns. On the dependency, `ws-gate` and its four reads failed, and so did `df-pilotsum` / `df-pilotsum2`. Earlier, at 14:34, `al-a-pub` (48 declared, 243 used), `al-rsg-pub` (189), `al-v2-pub` (181) and `xg5-geo` (96, 137) had pushed anon to 420 GiB, 19 GiB short of the line |
| CPU | Not binding: 60-80 of 110 cores used, `cpu.pressure` some 4 %, 0.7 % of periods throttled, 77 runnable threads (53 of them renderer threads). Latent: every job without a profile inherits `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = 110 from the platform's profile.d (remote-box.md). A 6-core rm_train runs 126 threads. Not changed mid-lane, because thread counts can change numerics |
| Ceiling | About 25 concurrent AlpaSim jobs, where VRAM booking and host RAM bind together. VRAM: 5 reads per card at 14.3 GB, or 4 reads plus one PAI chunk at 22. RAM: 25 jobs x 13-15 GiB = 300-340 GiB anon at peak, against the 85 % rule (391) and the line (451). Booking reads at per-n concurrent shares (11.0-11.4 GB at n = 3-5 instead of 13.0) would free 7-11 GB per card: not enough for a sixth read |
| Where the card-hours went (12 h to 17:40) | Computing 44 %, no work queued 41 %, a job on the card but the GPU idle 9 %, serial 6 % |

Changes in 086c095f; the dispatcher was restarted at 17:55:51 and adopted 25 running and 31 queued jobs:
- `ram_over` / `ram_stop`: while the margin is under 48 GiB, stop the job furthest above its `--ram` and requeue it once at its measured peak; rounds every 5 s while the margin is under 96 GiB.
- History raises a key's RAM booking when its recorded peak broke the declaration.
- A job whose child was SIGKILLed near the line is retried like a root kill.
- `ws_override` starts a job only when it fits with all page cache dropped.

At deploy time this changed nothing on the live queue: no job was above its declaration.

After: one measurement at 18:08, 12 min after the restart:
- 24 running and 10 queued. Nine GPU-ready jobs were held on RAM by the 85 % rule while a new wave of NR1FB10 reads started; one waited on `--after`.
- Working set 253 GiB (margin 198). The peak since the restart was 352 GiB at 17:59, with 24 jobs and 336 GiB anon.
- Load 89-94; 59 cores measured.
- 22 launches, and no `ram_over`, `ram_stop`, `kill` or `ws_override` event.

Concurrency is unchanged: the ceiling above still holds. What changed: a job that outgrows its declaration is now the one that gets stopped, and a read whose worker the platform kills is retried instead of failing its chain. On 2026-10-17 the kills cost about 90 training-minutes, five reads and two summary jobs.

Open, for the user: a large CPU-only job that is held on RAM now waits for reads to drain (a 344 GiB build needs an almost
empty box) and has no reservation. Per-kind history keys for reads would shorten burst holds: held 15, pub 20, static 10 and
waiting-set 22 GiB peaks all share `alpasim-read4` and are booked at 25.2. The thread env should be pinned per lane at a lane boundary.

Last verified: 2026-10-17
