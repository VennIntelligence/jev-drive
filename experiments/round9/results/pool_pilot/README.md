# POOL-PILOT: can a clean single run carry NTV0's S gain? (round 9, 2026-10-23)

The plan is [../../plans/2026-10-23-pool-design.md](../../plans/2026-10-23-pool-design.md); section 6 is the registration, written before any
training output. Lane: `scripts/pcl.sh`. Gates and verdict: `scripts/pcl.py`. Box dir: `$DATA_DIR/runs/round9/pool_pilot/`; the reads, the
scorer_v2 rescoring and the behavior_stats outputs stay there. Decision 460.

**Question.** NTV0's +0.10 on PAI 281 comes from its S rows (decision 456), and more offline steps do not reproduce it (decision 457). Can one
run from the shipped weights reach NTV0's PAI score, using S rows from its own lineage only? The S pool here is on1_L18RB0: RB0-F-s0's
default-serve roll on 1 020 nuPlan onpol-train scenes, 13 793 live rows.

**Answer.** Nearly, but not cleanly by the registered rule.
- **P1** (19 500 steps, S from step 10 000) reads 0.793 on the new scorer. That is 75 % of the RB0-F -> NTV0 gap, with +0.110 [+0.067, +0.153]
  over RB0L, so the new scorer puts it on "carries the gain". The old scorer puts it at 71 %: "undecided", 0.003 under the line.
- **P1b** (10 000 steps, S from step 5 000) is undecided on both scorers: 50 % / 56 %, but +0.051 [+0.015, +0.087] over RB0-F.
- **P0** (NTV0's recipe with the pool limited to rb0) is at NTV0 level: 88 % / 91 %, -0.012 [-0.044, +0.018] vs NTV0-s0. So the rb0 pool alone
  is enough, and whatever P1 lacks is in the placement, not the pool.
- No arm is slow on highway.

Seed 0 only. No extra seeds were run, as registered.

## Arms

| arm | tag | argv (token diff from `pcl.sh dry`) |
|:--|:--|:--|
| P1 | PCL1-F-s0 | RB0L-F-s0's stored argv + NTV0's S flags (`--on on1_JT --on-k 32 --on-mode mixed --on-wfile JT00S-rb0 --on-hold-pct 15`, lateral terms, `--on-lon-all --on-normal --lat-host --on-arc 0 --on-arc-rows brake --on-split alpasim_nuplan/onpol-train --diag`) + `--on-start 10000`; eager |
| P1b | PCL1B-F-s0 | RB0-F-s0's stored argv + the same S flags + `--on-start 5000`; eager |
| P0 (diagnostic) | PCL0-L18-s0 | NTV0-L18-s0's stored argv with only `--on-wfile JT00S -> JT00S-rb0`; compiled |

JT00S-rb0 is JT00S with every row outside `pool == rb0` set to weight 0. The build asserts 13 793 nonzero rows, all `rb0` and `self`, with
JT00S's weights ([wfile.json](wfile.json)).

## Code change

`rm_train.py --on-start T` (commit da7a47d3), off by default:
- Before draw T, the k_on pool slots are filled with normal rows (115 / 13 / 0), and the pool stream [seed, 0, 53] and the hash are left
  untouched.
- From T on, the batch is 83 / 13 / 32. The lr is not restarted.
- When the flag is set, the trainer logs `loss/on/batch_pool_rows`.

## Gates (all passed; submitted from b3ae2b89, which includes the parallel `--cmd-rule heading-all` commit)

| gate | result |
|:--|:--|
| flag off: RB0-F's argv on HEAD vs the stored RB0-F-s0, steps 0-75 | 22 / 22 scalars bit-identical ([g0_flagoff.json](g0_flagoff.json)) |
| flag on, before T: P1's real argv vs RB0L-F-s0, and P1b's vs RB0-F-s0, steps 0-100 (including a `--diag` step) | 27 / 27 bit-identical each; the only extra keys are diag / S keys |
| T = 30 over 75 steps | step 25 bit-identical to RB0L. Pool rows per logged window 0 / 25.6 / 32 / 32 = planned. S terms (agent, road, lat onc / soft / side) appear after T, finite and nonzero, and carry no mass before T ([g2_T30_batch.json](g2_T30_batch.json)) |
| 60-step smoke, each arm | pass: batch 83 / 13 / 32 mixed, pool rows as planned |
| P0 determinism (NTV0's stored log no longer reproduces bit-for-bit, decision 456) | the same argv twice on HEAD: 254 / 254 scalars and on_hash_60 equal ([det_P0.json](det_P0.json)) |
| S / imit gradient ratio, 500 steps after T | P1 0.584, P1b 0.389, P0 0.440 |
| final navtrain dev ADE <= 0.566 | 0.523 / 0.538 / 0.537 |
| informational: the whole pre-T segment of the full runs | P1 vs RB0L: 2 077 / 2 077 scalars to step 10 000 bit-identical. P1b vs RB0-F: 1 042 / 1 042 to step 5 000 |

On the gradient-ratio gate:
- The registered band is 0.21-1.04.
- The design's "NTV0 0.32-0.46" could not be reproduced from NTV0's stored `--diag` log: on the same quantity NTV0 gives 0.51 overall and
  0.30-0.80 per 500-step window. The band is therefore ±30 % of that measured range.
- All three arms are also inside the design's literal band (0.224-0.598).

## PAI 281, ACC + GAP (g10, FAST, NTV0's chunk lists), Harmonizer OFF

There are 277 scenes that all seven reads scored without a driver RPC error. Both scorers use this same set. Full tables: [verdict.md](verdict.md),
[pai281.md](pai281.md).

| read | old | new | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit / slow p | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL1-F-s0 | 0.717 | **0.793** | 65 / 44 | 16 | 6 + 1 | 14 | 23 | 5 | 27 / 10 | 0.523 |
| PCL1B-F-s0 | 0.703 | **0.768** | 72 / 54 | 21 | 11 + 2 | 13 | 20 | 5 | 23 / 6 | 0.538 |
| PCL0-L18-s0 | 0.737 | **0.806** | 60 / 41 | 14 | 7 + 0 | 16 | 21 | 2 | 27 / 9 | 0.537 |
| NTV0-L18-s0 | 0.745 | 0.817 | 58 / 38 | 12 | 6 + 0 | 15 | 22 | 3 | 25 / 11 | 0.536 |
| RB0-F-s0 | 0.648 | 0.717 | 87 / 68 | 19 | 14 + 4 | 21 | 21 | 8 | 22 / 8 | 0.528 |
| RB0L-F-s0 | 0.600 | 0.683 | 101 / 78 | 37 | 11 + 1 | 18 | 24 | 10 | 22 / 9 | 0.511 |
| NTV0NS-L18-s0 | 0.626 | 0.698 | 92 / 72 | 21 | 13 + 2 | 20 | 22 | 14 | 24 / 6 | 0.529 |

Paired differences (new scorer / old scorer):

| pair | new | old |
|:--|--:|--:|
| P1 - NTV0-s0 | -0.025 [-0.062, +0.010] | -0.029 [-0.067, +0.009] |
| P1 - RB0-F | +0.076 [+0.035, +0.118] | +0.068 [+0.026, +0.112] |
| P1 - RB0L | +0.110 [+0.067, +0.153] | +0.117 [+0.072, +0.163] |
| P1 - NTV0NS | +0.094 [+0.051, +0.138] | +0.091 [+0.045, +0.136] |
| P1b - NTV0-s0 | -0.050 [-0.082, -0.020] | -0.043 [-0.075, -0.012] |
| P1b - RB0-F | +0.051 [+0.015, +0.087] | +0.054 [+0.020, +0.090] |
| P1b - RB0L | +0.085 [+0.037, +0.135] | +0.103 [+0.053, +0.153] |
| P1b - NTV0NS | +0.069 [+0.030, +0.110] | +0.077 [+0.038, +0.117] |
| P0 - NTV0-s0 | -0.012 [-0.044, +0.018] | -0.008 [-0.040, +0.023] |
| P0 - RB0-F | +0.089 [+0.051, +0.128] | +0.089 [+0.051, +0.130] |
| P1 - P1b | +0.025 [-0.013, +0.064] | +0.014 [-0.026, +0.055] |
| P1 - P0 | -0.013 [-0.051, +0.024] | -0.020 [-0.060, +0.019] |

Zero classes against NTV0NS (no S rows) show what the S rows fix. Every arm is back on NTV0's side for AF-front (P1 -9 [-16, -3], P1b -9,
P0 -12). P1 also moves back on offroad -5 [-14, +4], drift -7 [-14, +0] and route-turn -6 [-13, +1], though these CIs reach 0. Against NTV0-s0, P1 leaves no class outside its CI except side +1 /
rear-hit +2, both at the CI edge. P1b keeps +9 [+3, +16] offroad and +5 [+1, +10] drift against NTV0: the shorter run does not fix the lateral
classes.

## Reading lines

g = (arm - RB0-F) / (NTV0-s0 - RB0-F) on the 277 common scenes. Lines: >= 74 % with the CI against the offline counterpart above 0, or <= 21 %.

| arm | new scorer (0.717 -> 0.817; lines 0.791 / 0.738) | old scorer (0.648 -> 0.745; lines 0.720 / 0.669) |
|:--|:--|:--|
| P1 | 0.793, **75 %**, vs RB0L CI > 0 -> **carries the gain** | 0.717, **71 %** -> **undecided** |
| P1b | 0.768, **50 %**, vs RB0-F CI > 0 -> **undecided** | 0.703, **56 %** -> **undecided** |
| P0 | 0.806, **88 %** -> at NTV0 | 0.737, **91 %** -> at NTV0 |

The new scorer is the primary one. P1 clears its line by 0.002, while on the old scorer it falls 0.003 short, so the two scorers land P1 on
different lines. With one seed and seed differences of 0.03-0.04 (decisions 430 / 443), P1 is "at the line": it is not cleanly settled either way.

P0 is at NTV0, so the rb0 pool alone is enough under NTV0's placement. P1's remaining ~0.025 gap to NTV0 (CI spans 0) is the single-run
placement, not the pool.

## Highway read (behavior_stats.py as it ran for NTV0; registered descriptive line)

| read | free-road highway steps | median v_e / v_h | p10 | < 0.85 | > 1.10 | mean dv (m/s) | highway slow + rear-hit pts (new) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| PCL1-F-s0 | 804 | 1.00 | 0.84 | 10 % | 6 % | -0.73 | 1.16 |
| PCL1B-F-s0 | 786 | 1.00 | 0.85 | 10 % | 8 % | -0.57 | 1.29 |
| PCL0-L18-s0 | 824 | 1.00 | 0.86 | 9 % | 6 % | -0.60 | 1.05 |
| NTV0-L18-s0 | 803 | 1.00 | 0.87 | 9 % | 6 % | -0.62 | 1.06 |
| RB0-F-s0 | 827 | 1.00 | 0.87 | 9 % | 8 % | -0.49 | 0.86 |

Every arm reads "this recipe is not slow on highway": median 1.00 >= 0.95, < 0.85 share 10 % < 15 %, and points 1.2-1.3 < 2.0.
- Among the highway-majority scenes with free-road steps (32 of 49), median ratios under 0.90 occur in 1-4 per read. They are mostly the same
  scenes for every read (ffe5b17d 0.81, 1a050ead, 4e454838), including NTV0 and RB0-F.
- The ramp scene has no free-road steps, so ramps cannot be judged.

## Cost

12.0 GPU job-hours in total:
- training: 2.45 job-hours, one card per training;
- 24 read chunks: 9.33 job-hours, at 4 per card about 2.3 card-hours;
- gates: 0.19.

That comes to about 5 card-hours, and 1.9 h wall from the first start to the last end.
