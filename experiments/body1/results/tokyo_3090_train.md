# P2H10S-F training on one RTX 3090 (Tokyo box, 2026-10-11)

Question: does the P2H10S-F recipe (batch 128, 10 000 steps; main box: 28.8 GB peak reserved, 7.2 it/s, 22 to 24 min) train on a 24 GB card,
and how fast. Answer: not as it ships (out of memory), yes with `--label-bank` (same values, bit for bit), and with `--compile` on top one
3090 runs at the main box's eager speed; two seeds at once, one per card, do not slow each other.

Runner: `experiments/body1/scripts/tokyo_bench.sh <name> <card> [flags]` (one timed run; card peak from `nvidia-smi`, process RSS and host page
cache sampled every 2 s). Raw files on the Tokyo box: `/data/runs/tokyo_dual3090/train/<name>/{bench.json, log.txt, usage.tsv}`,
`train_chain_{A,B}.out`. Env `/data/envs/op-train` (torch 2.14.0+cu130).

## Short runs on the one-shard set (measured)

Data: `navtrain_full.s2of12` listed 12 times, i.e. one pulled shard repeated to the row count of the full run (281k Store rows), so the label
and teacher tensors on the card have their full-run size but the token files are 1 / 12 of the full set and fit the page cache. Recipe flags
as the main-box run. "Other card" says what the second 3090 did meanwhile; the data pull (4 MB/s, 32 streams) ran throughout.

| Run | Flags | Steps | it/s | Torch peak GB | Card peak MiB | RSS peak MiB | Page cache MiB | Other card | Full run projected |
|---|---|--:|--:|--:|--:|--:|--:|---|--:|
| `full-dense` | (as shipped) | 100 | out of memory before the first logged step | card full (23.56 GiB) | | | | PAI evaluation | does not run |
| `full-bank` | `--label-bank` | 400 | 3.96 | 16.3 | 17 030 | 16 056 | 47 436 | PAI evaluation | 42 min |
| `full-bank-dev0` | `--label-bank --dev-card-gb 0` | 400 | 3.89 | 15.8 | 16 580 | 15 543 | 41 490 | PAI evaluation | 43 min |
| `full-bank-fw8` | `--label-bank --prefetch 8 --fetch-workers 8` | 400 | 3.92 | 16.4 | 17 162 | 16 254 | 41 932 | PAI evaluation | 43 min |
| `full-bank-b64` | `--label-bank --batch 64` (hinge-only rows 2 / 2 / 2; not the recipe) | 400 | 7.09 | 11.7 | 12 344 | 10 721 | 41 962 | PAI evaluation | (half the rows per step) |
| `full-bank-compile` | `--label-bank --compile` | 800 | 5.64 over the run, 7.1 after step 100 | 13.4 | 14 040 | 23 948 | 46 859 | PAI evaluation | 24 min |
| `pair-e0` / `pair-e1` | `--label-bank`, seeds 0 / 1 at once | 1 200 | 4.04 / 4.00 | 16.3 / 16.3 | 17 030 / 17 017 | 28 338 / 28 371 | 44 321 | the other seed | 42 min each |
| **`pair-c0` / `pair-c1`** | **`--label-bank --compile`, seeds 0 / 1 at once** | 2 400 | **7.14 / 7.04** | 13.4 / 13.4 | 14 058 / 14 027 | 36 436 / 36 433 | 46 428 | the other seed | **24 min each** |
| `full-bank-b64-compile` | `--label-bank --compile --batch 64` (not the recipe) | 1 200 | 9.99 over the run | 10.4 | 10 994 | 20 486 | 46 680 | idle | (half the rows per step) |

Read: the it/s column is the trainer's own running mean, so a compiled run's figure includes its compilation (about 40 s) and rises with the
run length; projections are steps / steady it/s plus compilation, not measurements. Dense batch 128 fails as predicted from the main box (13.8 GB
of its 28.8 GB peak are the two dense raster tensors; the bank holds one raster per distinct label, 103k of 281k rows per file). Moving the dev
tokens to the host or adding fetch threads changes nothing: the step is card-bound (util 87 to 93 %), not fetch-bound. RSS is mostly mapped
token pages (the same pages as the page cache), not private memory: two compiled runs with 36 GB RSS each ran on the 60 GB host with 5 GB "used".

## Identity (measured)

`bd4_train.py ident` on a 60-step pair (one shard, seed 0), dense against `--label-bank`, both on a 3090: 42 logged scalars, adapter and net
weights equal bit for bit (`train/ident_bank.json`; `lib/drivable_hinge.py` RowBank was checked against the dense tensor on the real label files
before, commit 61cb40c2). `--compile` is not bit-identical to eager (as `pp_train.py --compile`); its difference is read on the full runs below.

## Full-length runs on the full set (measured)

`P2H10S-F-T3090-s0` / `-s1`: the recipe on all 12 shards (436 files, 76.5 GiB, each on disk at size with a sha256-verified pull event), 10 000
steps, `--label-bank --compile`, seed 0 on card 0 and seed 1 on card 1 at once, page cache dropped for the token files before the preflight
(`posix_fadvise DONTNEED`; the token set does not fit the 60 GB host). Raw files: `/data/runs/tokyo_dual3090/train_d/{pre-0,pre-1,full-s0,full-s1}/`.

| Run | Steps | it/s over the run | Wall (first to last log line) | Torch peak GB | Card peak MiB | Card util | RSS peak MiB | Page cache peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| preflight, cold cache, both cards | 300 | 6.33 / 6.25 | | 16.7 | 18 094 / 18 117 | 59 % | 13 800 | 51 900 |
| **`full-s0`** (card 0) | 10 000 | **7.29** | **22 min 56 s** | 17.3 | 18 076 | 93 % | 46 469 | 57 911 |
| **`full-s1`** (card 1) | 10 000 | **7.19** | **23 min 15 s** | 17.3 | 18 081 | 94 % | 44 708 | 57 913 |

Measured against projected: the one-shard rows projected 24 min and 13.4 GB per card. Time held (23 min, the main box's 22 to 24 min at 7.2
it/s); memory did not: the full set peaks at 17.3 GB reserved (18.1 GB on the card), 3.9 GB above the one-shard figure, still inside 24 GB.
The token set larger than RAM cost nothing measurable after the first few hundred steps (7.12 / 7.01 it/s at step 1000, flat at 7.29 / 7.19
from step 4000; page cache at 57.9 GB of 60). No card-0 power fault during either run (card 0 at 93 % mean utilisation, clock 1815 to 1860 MHz when sampled early in the run).

### Against the main-box checkpoints of the same seeds

Dev metrics from the training logs (same dev rows, same steps; main box is eager and dense, the 3090 runs are compiled with the bank):

| Step | s0 main ade / drift_on / drift_off | s0 3090 | s1 main ade | s1 3090 ade / drift_on / drift_off |
|--:|---|---|--:|---|
| 1000 | 0.663 / 1.421 / 0.109 | 0.654 / 1.405 / 0.098 | 0.647 | 0.639 / 1.417 / 0.081 |
| 2000 | 0.597 | 0.599 / 1.395 / 0.068 | 0.588 | 0.586 / 1.413 / 0.063 |
| 3000 | 0.577 | 0.577 / 1.421 / 0.064 | 0.580 | 0.582 / 1.473 / 0.063 |
| 4000 | 0.571 | 0.570 / 1.424 / 0.065 | 0.565 | 0.566 / 1.425 / 0.056 |
| 5000 | 0.564 | 0.569 / 1.454 / 0.059 | 0.567 | 0.565 / 1.465 / 0.053 |
| 6000 | 0.558 | 0.557 / 1.448 / 0.052 | 0.551 | 0.557 / 1.447 / 0.051 |
| 7000 | 0.561 | 0.562 / 1.466 / 0.049 | 0.546 | 0.547 / 1.447 / 0.050 |
| 8000 | 0.549 | 0.550 / 1.434 / 0.049 | 0.550 | 0.552 / 1.465 / 0.049 |
| 9000 | 0.546 | 0.546 / 1.454 / 0.045 | 0.545 | 0.546 / 1.459 / 0.045 |
| **10000** | **0.545 / 1.457 / 0.045** | **0.546 / 1.456 / 0.045** | **0.543** (1.459 / 0.044) | **0.545 / 1.460 / 0.044** |

The retrains follow the main-box curves to 0.001 to 0.002 ADE at the end (largest gap anywhere 0.009, at step 1000); the two seeds of one box
differ by the same amount. Open loop, the compiled 3090 run is the same training.

Closed loop, AlpaSim PAI on `s_b1a` (33 scenes, served config, one rollout per scene, each retrain on its own card; per scene:
[tokyo_3090_retrain_b1a_scenes.tsv](tokyo_3090_retrain_b1a_scenes.tsv)):

| Row | Mean | Zeros | at-fault collision | offroad | left corridor |
|---|--:|--:|--:|--:|--:|
| `P2H10S-F-T3090-s0` | 0.4572 | 15 | 4 | 3 | 9 |
| main-box `P2H10S-F-s0` | 0.3905 | 17 | 5 | 3 | 10 |
| `P2H10S-F-T3090-s1` | 0.3767 | 18 | 4 | 4 | 11 |
| main-box `P2H10S-F-s1` | 0.4175 | 16 | 4 | 3 | 11 |

| Pair (33 scenes) | Difference [scene-bootstrap 95% CI] | Better / worse / within 0.01 |
|---|---|---|
| 3090 s0 - main s0 | +0.0667 [+0.0016, +0.1592] | 7 / 1 / 25 |
| 3090 s1 - main s1 | -0.0408 [-0.1151, +0.0037] | 4 / 3 / 26 |
| main s1 - main s0 | +0.0270 [-0.0860, +0.1482] | 4 / 6 / 23 |
| 3090 s1 - 3090 s0 | -0.0805 [-0.2015, +0.0320] | 3 / 8 / 22 |

Read: the two retrain-against-original differences have opposite signs (two-seed means 0.4170 against 0.4040), and each is two scenes
changing between zero and scored (s0: two zero scenes scored, none the other way; s1: two scored scenes went to zero), with +0.013 / +0.005
on the 16 / 15 scenes scored by both. That is the size of a seed-to-seed difference on these 33 scenes (the rows above; decision 235), so the
retrain is not told apart from the original; 33 scenes cannot say more, and it was not pursued. Absolute PAI scores here carry the 10 Hz slot
caveat of [served_plan_length.md](served_plan_length.md); the paired differences do not.
