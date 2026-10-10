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

## Full-length runs on the full set

Pending: `P2H10S-F-T3090-s0` / `-s1`, one seed per card at once, `--label-bank --compile`, from a cold page cache (the full token set is about
74 GB against 60 GB of RAM, which the one-shard rows above do not exercise).
