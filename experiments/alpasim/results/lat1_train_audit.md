# LAT1: where the training side of the openpilot-adapter baseline loses time (2026-10-09)

Measured on small subsets (16-96 tokens, 40-step training benches) and from the logs of finished runs; box loaded by other lanes
(load 44-98 on a 75-core container quota, cards shared). Execution side: [lat1_frame_synthesis.md](lat1_frame_synthesis.md).
Bench scripts: `experiments/alpasim/archive/lat1_train_audit/` (`prep_bench.py`, `train_bench.py`, `dedupe_check.py`,
`sched_audit.py`), raw numbers `results/lat1/train_audit/*.json`.

## Short answers

- Prep is CPU-starved, not serial: the process pool already feeds the encoder, but the slot warp runs on the CPU, so the card encodes
  16 % of the wall time. The GPU lattice gives the same frames and 8.6x the throughput on the same cores.
- The prep encoder is batched well: card-bound from batch 32 (about 1 310 image pairs/s, flat to batch 192).
- Training at batch 128 is card-bound (card 85-99 % per minute in the real runs); the Python dispatch floor of 48 ms per step only
  shows below batch 64. Bigger batches are not free.
- `torch.compile` halves the training step and changes numerics.
- 10 % of every training run is dev eval waiting on host cache reads.
- Pool admission is limited by declared cores, not by busy ones: ready work waited on the CPU budget in 263 of 1 673 minute samples
  (4.5 of 28.7 h) while 42.7 cores were measured busy and 91.8 charged. The container quota is 75 cores, not 208.

## Ranked by wall-clock saved

| # | what | where | before (measured) | after | same bytes | status |
|--:|:--|:--|:--|:--|:--|:--|
| 1 | Slot warps of the cache builders on the card; workers return the 4 keyframes only | `ap2_prep.py`, `pp_prep.py` (`--frames warp`), `ot3_rows.py` -> `sh30_core.lattice_gpu` | `ap2_prep --bw`: 0.83-1.14 s of worker CPU per token (26 warps x 28.6 ms), 3.24 tokens/s on 3 workers; full navtrain as run 49.6 core-h, 2.76 card-h | measured 27.7 tokens/s on the same 3 workers; full navtrain about 62 min on 4 cores and one card | frames yes (0 of 642 M px); tokens equal at the same encoder batch | **done** for `ap2_prep.py` and `pp_prep.py --frames warp` (4a10739a, `--synth gpu` default): ap2 `--bw` 3.48 -> 27.4 tokens/s, pp warp 131 -> 836 image pairs/s on the same 3 workers and 4 cores; cache files identical to the CPU path and to the existing caches (`prep_check.py`, table below). `ot_rows.py` not converted (about 25 lines with `pp_prep.warp_keys`) |
| 2 | Compiled training step (`torch.compile`, inductor) | `ap2_train.py`, `pp_train.py` | 137 ms per step at batch 128 (forward 52.5, backward 75.1) | measured 70.2 ms: 10 000 steps 23 -> 12 min; compile 57 s | no: gradient differs 1.7 % relative (cosine 0.99986) | **opt-in** (`--compile` in both trainers): 164 -> 76 ms per step on a quiet card (13.1 it/s), 1.9-2.15x on a shared one. Validation below: compiled vs eager is 1/3 to 1/5 of the seed spread on dev. Not the default: navtest and closed loop not run |
| 3 | CPU admission by measured cores | pool `--cpu` declarations | trainers declare 6-8 cores and peak at 1.4-5.8; 10 076 queued job-minutes behind the CPU budget | estimated: most of the 4.5 h of blocking goes if trainers declare 2 and closed-loop jobs their measured peak | n/a | open: declarations live in each lane's chain scripts |
| 4 | Free graph values after their last reader | `jevdrive/op_torch.py` `OnnxTorch.run` | encoder 27.6 GB above its inputs at batch 128 (ap2-prep declared 12 GB, peaked at 29.1, two runs died of OOM) | 1.2 GB at batch 128, same time | yes: outputs and all 155 gradients equal | **done** (7b503a73): batch 8 / 32 / 64 1.75 / 6.89 / 13.77 -> 0.08 / 0.30 / 0.61 GB; policy forward + backward at batch 16 2.42 -> 1.23 GB |
| 5 | Dev-eval tokens on the card once | `ap2_train.py:dev_by_m` | 15 s per eval, 150 s of a 1 549 s run | estimated 4 s per eval, 1.9 GB | yes | **done**, default (`--dev-card-gb 4`): 19.5 -> 4.9 s per eval, dev numbers and checkpoint bit-identical |
| 6 | Integer chroma mean and byte gathers in the NAVSIM key renderer | `jevdrive/navsim_zs.py` `OpenpilotMaps` | 94 ms per token for 4 keys (decode 11.5 + packing 12.1 ms each) | 55 ms per token | yes (0 of 7.9 M px old vs new, four map variants; `lat1_check.py remap` 865 M px for the same arithmetic) | **done** (7b503a73) |
| 7 | Encode each real slot once | `ap2_prep.py:make` | 34 image pairs per token | 27 | frames yes; tokens differ 4.2e-4 relative (the encoder's batch-size noise is 7.1e-4) | partly: each lattice is warped once (26 -> 20 warps per token). The encoder still sees 34 pairs, because encoding 27 changes the batch composition and with it the tokens (up to 0.031 at batch 128) |
| 8 | Batch fetch off the step's critical path | `pp_train.py:Tokens.__getitem__` | 25-73 ms per batch of 128 (12 memory maps, three copies, pageable upload); loop 158 ms per step against 140-147 for the step | estimated: prefetch depth 2-3 removes 11-18 ms; required once the step is compiled | yes | **done**, default (`--prefetch 4 --fetch-workers 2`, pinned buffers, non-blocking uploads, the rng stream stays in the main thread): loss curve and checkpoint bit-identical; worth only 4-5 % of a step (0 on a crowded card), the fetch did not become the limit under `--compile` |
| 9 | Fewer fp16 / fp32 casts in the policy | `op_torch.py` LayerNormalization, Softmax | 772 `aten::to` per step, 31 % of GPU kernel time under the profiler | estimated 20-30 % of the card part | no | open (fp16 overflow is why they are fp32) |
| 10 | One packed-key cache per JPEG for all prep families | five builders decode the same CAM_F0 JPEGs (each serves 2.82 navtrain tokens) | 0.094 s CPU per token per family | estimated 2.7 core-h per family | yes | open |
| 11 | Incremental cache writes with a done mask | `ap2_prep.py`, `pp_prep.py` | 6.8 + 2.8 GB per shard held in RAM until the end, no resume | - | yes | **done** with item 1 (`pp_prep.Part`: `open_memmap` + done-rows mask; a build killed after 64 rows resumes to identical files) |

Not worth doing: bigger training batches (332 / 613 / 728 / 872 rows/s at batch 16 / 32 / 64 / 128), bigger encoder batches (flat from
32), constant folding (46 of 388 policy nodes), removing `vmap` (13.4 -> 10.5 ms host per pass, hidden at batch 128), thread-count
tuning, pinned uploads in prep (0.24 ms per token).

## Prep on the card: equality and declarations (`experiments/alpasim/scripts/prep_check.py`)

Old = `--synth cpu`, new = `--synth gpu`, same encoder calls, pair counts and batch size on both sides; frames hashed at the encoder input.

| builder | data | tokens | encoder batch | frames into the encoder | differing | tokens, cpu vs gpu | gpu vs the existing cache rows |
|:--|:--|--:|--:|--:|--:|:--|:--|
| ap2 `--bw` | `navtrain_full.s2of12` | 256 | 128 | 17 920 | 0 | identical | identical |
| ap2 `--bw` | `navtrain_full.s9of12` | 256 | 32 | 17 920 | 0 | identical | identical |
| pp warp | `lb_navtrain` | 256 | 128 | 4 096 | 0 | identical | identical |
| pp full-navtrain job (functions only; the navtrain index is gone) | `s2of12`, 16 camera positions | 224 | - | 2 240 | 0 | frames only | - |

Every ap2 token holds the cold-start lattices for 1, 2 and 3 keyframes. The cache keys are unchanged, so no existing cache is invalidated.
On a card another job holds at 100 % the same build reads 14.7 tokens/s. The build is encoder-bound now (per chunk of 32 tokens: warps
0.14-0.17 s, encoder 0.98 s); 2-3 workers feed it.

Declarations: `--vram 8 --cpu 4 --ram 16 --workers 3` (peak 4.4 GB reserved, 7.6 GB RSS for the whole check chain; the old 29 GB was the
interpreter's retained values). Pass `--workers` under the pool: `jevdrive.common.n_cpus()` reports the container (75), not the pinned
cores, and the log-entry pool then forks 75 workers (94 GB RSS). `ap2_chain.sh` and `ot3_chain.sh` still declare `--cpu 20 --workers 18
--ram 40/60`, VRAM 12 / 44: theirs to change.

## Prep, per token (one process, 4 pinned cores)

| step | ms (median) |
|:--|--:|
| libjpeg decode | 11.5 |
| packing (old `OpenpilotMaps.__call__`) | 12.1 |
| `render_token` (4 keys) | 98.8 |
| `warp_frame`, one slot, both views | 28.6 (of which `warp_map` x 2: 23.9, `cv2.remap` x 6: 2.2) |
| `ap2_prep._job --bw` (4 keys + 26 warps, 20 distinct) | 1 139 |
| `pp_prep._full_job` (4 keys + 6 warps) | 274 |

| variant (96 tokens, 3 workers) | tokens/s | card |
|:--|--:|:--|
| today | 3.24 | busy 16 % of wall: a burst every 8 s |
| keys on CPU, `lattice_gpu` + encoder in the consumer | 27.7 | the limit (consumer thread 89 % busy) |

Encoder (fp16): 759 pairs/s at batch 8 (dispatch-bound, the serving case), 1 310-1 335 from batch 32. Its output depends on the batch
size at the 7e-4 level (up to 0.031 between batch 64 and 128), and the existing AP2 caches already mix `--bs 32` (shards s6-s9) and 128:
"identical tokens" means the same frames at the same encoder batch size.

## Training (AP2-AB recipe, 99 679 rows, fp16)

| batch | forward ms (host) | backward ms (host) | step ms | rows/s | peak GB |
|--:|--:|--:|--:|--:|--:|
| 16 | 17.0 (16.9) | 19.8 (18.9) | 48.3 | 332 | 6.5 |
| 32 | 17.8 (17.5) | 19.3 (19.0) | 52.2 | 613 | 8.9 |
| 64 | 30.0 (17.6) | 38.5 (20.2) | 88.0 | 728 | 13.7 |
| 128 | 51.5 (18.1) | 81.5 (30.0) | 146.9 | 872 | 23.2 |

Real runs: 10 000 steps at batch 128 took 1 549 s alone on a card (6.5 it/s), 2 552-2 742 s with two trainers per card.

## Trainer changes: identity and the compiled step

Bit-identity of the new defaults (prefetch, pinned uploads, dev tokens on the card): a 10 000-step run with them repeats the 10-08
`AP2-AB-s0` in every loss / dev scalar and in `ckpt-final.pt` (which also shows that the `op_torch` change 7b503a73 left training
numerics alone); same for 600 steps of ap2 at batch 128 and of pp_train (P2 hinge) at batch 64. The trainer is run-to-run deterministic.

Compiled step (`--compile`: `torch.compile(model, dynamic=False)` around the training step only; dev eval stays eager, so no
recompiles), AP2-AB, 10 000 steps, dev 1 767 rows, at step 10 000:

| run | ade | ade_m1 | ade_m2 | ade_m3 | drift_on | loss/imit |
|:--|--:|--:|--:|--:|--:|--:|
| eager s0 | 0.5728 | 0.6585 | 0.5801 | 0.5655 | 1.4602 | 0.7019 |
| eager s1 | 0.5726 | 0.6533 | 0.5806 | 0.5665 | 1.4520 | 0.7095 |
| compiled s0 | 0.5727 | 0.6551 | 0.5796 | 0.5647 | 1.4581 | 0.7008 |
| compiled s1 | 0.5720 | 0.6518 | 0.5802 | 0.5662 | 1.4533 | 0.7091 |

| mean abs difference over the 10 evals | ade | ade_m1 | drift_on | weights, relative L2 |
|:--|--:|--:|--:|--:|
| eager s1 - eager s0 (seed spread, two seeds only) | 0.0083 | 0.0124 | 0.0173 | 0.184 |
| compiled s0 - eager s0 | 0.0017 | 0.0036 | 0.0034 | 0.032 |
| compiled s1 - eager s1 | 0.0020 | 0.0034 | 0.0034 | 0.025 |

Compiled runs repeat bit for bit among themselves. 10 000 steps: 1 549 s (eager, 10-08) against an estimated 800-830 s compiled at the
same card load (measured today: 1 433 s each for two compiled runs sharing a card a third job filled). Checkpoints
`LAT1DEV-v-c0`, `-c1` under `$DATA_DIR/runs/alpasim/lat1/train_dev/out/op_parity/runs/`; what would make `--compile` the default is
`ap2_offline.py` (navtest) and a closed-loop run of them against `AP2-AB-s0` / `s1`.

Declarations after 7b503a73 (the chains' 40-48 GB are stale): AP2-AB at batch 128 `--vram 22 --cpu 3` (compiled: 20), pp_train `--host`
at batch 64 `--vram 10 --cpu 3`; every run averaged 1.1-1.4 cores. Not wired: `ap2_ot.py`, `ot_rows.py`, `fw_p2.py` have their own
loops (three lines each: `tok.pin = True`, `T.Prefetch`, `T.card_rows`); `--compile` with DDP, `--mem-e2e`, P3 side, `--route` is
untested; the inductor cache sits in `/tmp/torchinductor_ujs` (4 GB on a 30 GB root overlay: move `TORCHINDUCTOR_CACHE_DIR` to the data
disk in `jevdrive/cl/profiles.py` before compiling routinely).

## Scheduling (pool samples of the last 24 h, 72 card-h)

| | measured |
|:--|:--|
| cards | computing 40 %; no work queued 42 %; a GPU job not using its card 13 % (9.1 card-h); idle beside a long job 4 %; waiting on CPU 0.4 % |
| cores | 391 core-h measured of 1 800 quota (22 %); 767 charged |
| a card with a job on it | mean utilisation 51-64 %; below 10 % in 19-33 % of samples |
| largest over-declarations (declared / peak core-h) | `alpasim-c0b-wajepa` 68.9 / 23.4; `ot3-t` trainers 46.1 / 11.0; `alpasim-c0b-ot0` 35.1 / 13.4; `cf-t` 26.8 / 6.3 |
| VRAM | used / booked 0.59-0.63; `ap2-prep` under-declared (12 against 29.1 GB) |
| summed launch waits by reason | `after` 691 529; `cpu` 567 239; `vram` 150 765; `start_limit` 44 431; `ram` 25 349 |

Restructuring: item 1 turns 12 CPU-starved prep jobs into one card-bound stream per card; trainers declare 2 cores; a chain's
training and closed-loop stages go in as separate pool jobs with their own declarations (three closed-loop jobs booked at 32 GB sat
beside processes using 7-18 % of a card; matched by timing, not by PID).

## Not measured

Everything marked estimated; whether a compiled trainer gives the same closed-loop scores; mean (not peak) cores per job; more than
3 prep workers with the GPU lattice (the consumer thread is then the limit).
