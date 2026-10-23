# P1L: is P1's remaining gap the learning rate at S entry? (round 9, 2026-10-24)

The registration is [../../plans/2026-10-24-p1l.md](../../plans/2026-10-24-p1l.md) (Chinese), written before any training output. The lane is
`scripts/p1l.sh`, with gates and verdict in `scripts/p1l.py` (it reuses `pcl.py`). The box dir is `$DATA_DIR/runs/round9/p1l/`: reads, scorer_v2
rescoring and behavior_stats stay there. Full tables are in [verdict.md](verdict.md) and [pai281.md](pai281.md). Decision 461. Seed 0 only.

**Question.** P1 (PCL1-F-s0, decision 460) reached 75 % / 71 % of the RB0-F -> NTV0 gap. Its S rows enter at step 10 000, where its cosine sits at
0.48 of peak. NTV0's S stage instead ran on a fresh schedule. P1L keeps P1's argv and switches the lr from step 10 000 to NTV0-L18's schedule.

**Answer.** On the registered line, the lr at S entry was the cause. P1L reads **0.816 new / 0.737 old**, which is **g = 98 % / 91 %** of the gap.
- vs P1: +0.023 [-0.010, +0.058] new.
- vs NTV0-s0: -0.002 [-0.039, +0.036] new.

The registered rule is g >= 88 % and P1L - P1 > 0, and both scorers meet it. One seed only: the paired CI against P1 still includes 0.

## Code change

`rm_train.py --lr2-at T [--lr2-scale 0.1 --lr2-warmup 500 --lr2-floor 0.1]`, commit 2dc84752. It is off by default and adds no operation when off.
- From step T, each parameter group's lr is what a fresh run with `--lr-scale S --warmup W --lr-floor F --steps (steps - T)` would give it at
  step - T.
- AdamW moments and the GradScaler state carry on; nothing is reset.
- With the flag, the trainer logs `lr/g<i>`.

P1L's argv is P1's argv plus `--lr2-at 10000 --lr2-scale 0.1 --lr2-warmup 500 --lr2-floor 0.1`. The three values are read from NTV0-L18-s0's stored
argv by `pcl.py argv P1L`. Per group, against NTV0-L18-s0's stored meta (lr 3e-5, lr_new 3e-4, lr_scale 0.1, rm_lr 3e-4, rm_warmup 100):

| group | P1 at step 10 000 | P1L from step 10 000 (= NTV0's L18 stage) |
|:--|--:|:--|
| g0 net | 1.44e-5 | peak 3e-6, warm-up 500, cosine over 9 500 steps to 0.1 x peak |
| g1 old adapter | 1.44e-4 | peak 3e-5, warm-up 500 |
| g2 route branch (`--rm-lr`, not scaled) | 1.44e-4 | peak 3e-4, warm-up 100 |

So the net and the adapter drop about 4.8x at T, while the route branch rises about 2x. This follows NTV0 exactly; the two effects are not split.

## Gates (all passed; submitted from 2dc84752)

| gate | result |
|:--|:--|
| flag off: RB0-F's argv vs the stored RB0-F-s0, steps 0-75 | 22 / 22 scalars bit-identical, no extra or missing keys |
| flag off: P1's argv vs the stored PCL1-F-s0, steps 0-100 (including a `--diag` step) | 119 / 119 bit-identical, no extra or missing keys |
| flag on, before T: P1L's argv vs PCL1-F-s0, steps 0-100 | 119 / 119 bit-identical; the only new keys are `lr/g0-2`; the lr matches P1's formula |
| T = 30, over 75 steps | step 25 bit-identical (75 / 75). Pool rows per window 0 / 25.6 / 32 / 32 = planned. S terms are finite and nonzero after T, none before. The lr after T matches the second schedule |
| 60-step smoke (T = 30) | DONE, finite, batch 83 / 13 / 32 mixed, snapshot written, pool rows as planned, lr formula exact |
| S / imit gradient ratio, 500 steps after T | 0.755. Inside the registered band 0.21-1.04, above the design's literal 0.224-0.598. P1 was 0.584. Per-step values 0.99 / 1.12 / 0.08 / 1.06 / 0.82 |
| post | DONE, final dev ADE 0.525 <= 0.566. The whole pre-T segment vs PCL1-F-s0: 6 471 / 6 471 scalars bit-identical to step 10 000 |
| lr over the whole run | 780 logged points, max relative error vs the formula 0; the remaining horizon (9 500) equals NTV0's |

Logged lr (g0 / g1 / g2):

| step | g0 | g1 | g2 |
|--:|--:|--:|--:|
| 10 025 | 1.50e-7 | 1.50e-6 | 7.50e-5 |
| 10 100 | 6.00e-7 | 6.00e-6 | 3.00e-4 |
| 10 500 | 2.98e-6 | 2.98e-5 | 2.98e-4 |
| 15 000 | 1.54e-6 | 1.54e-5 | 1.54e-4 |
| 19 500 | 3.00e-7 | 3.00e-6 | 3.00e-5 |

## PAI 281, ACC + GAP (g10, FAST, NTV0's chunk lists, P1's serve), Harmonizer OFF

The comparison set is the 278 scenes that all five reads scored on both scorers, with no driver RPC error in any of them.

| read | old | new | zeros old / new | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit / slow p | launched | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **PCL1L-F-s0** | **0.737** | **0.816** | 60 / 38 | 14 | 6 + 3 | 10 | 23 | 4 | 27 / 11 | 32 / 37 | 0.525 |
| PCL1-F-s0 (P1) | 0.718 | 0.793 | 65 / 44 | 16 | 6 + 1 | 14 | 23 | 5 | 28 / 10 | 32 / 37 | 0.523 |
| PCL0-L18-s0 (P0) | 0.738 | 0.806 | 60 / 41 | 14 | 7 + 0 | 16 | 21 | 2 | 27 / 9 | 32 / 37 | 0.537 |
| NTV0-L18-s0 | 0.746 | 0.818 | 58 / 38 | 12 | 6 + 0 | 15 | 22 | 3 | 26 / 11 | 33 / 37 | 0.536 |
| RB0-F-s0 | 0.649 | 0.718 | 87 / 68 | 19 | 14 + 4 | 21 | 21 | 8 | 23 / 8 | 32 / 37 | 0.528 |

Paired differences:

| pair | new | old | zeros (old) |
|:--|--:|--:|--:|
| P1L - P1 | +0.023 [-0.010, +0.058] | +0.020 [-0.016, +0.056] | -5 [-15, +5] |
| P1L - P0 | +0.010 [-0.025, +0.045] | -0.001 [-0.037, +0.035] | +0 [-10, +10] |
| P1L - NTV0-s0 | -0.002 [-0.039, +0.036] | -0.009 [-0.048, +0.029] | +2 [-9, +13] |
| P1L - RB0-F | +0.099 [+0.057, +0.143] | +0.088 [+0.045, +0.133] | -27 [-40, -15] |

Classes against P1, all with CIs that include 0:
- route-turn -4 [-9, +0]
- offroad -2
- AF-front -1
- side +2

Against NTV0-s0, the only class at the CI edge is side +3 [+0, +7]. Route-turn is -5 [-12, +2]. Against RB0-F, P1L has fewer zeros in route-turn
(-11 [-18, -4]) and in drift (-8 [-15, -1]).

## Reading lines

g = (arm - RB0-F) / (NTV0-s0 - RB0-F), computed on the 278 common scenes:

| scorer | RB0-F -> NTV0 | P1L | P1 | P0 | line |
|:--|:--|--:|--:|--:|:--|
| new (registered) | 0.718 -> 0.818 | **98 %** | 75 % | 88 % | g >= 88 % and P1L - P1 +0.023 > 0 -> **the lr at S entry was the cause; use it in NTV1** |
| old (reported) | 0.649 -> 0.746 | **91 %** | 71 % | 92 % | same line |

Caveats:
- One seed. Seed spread is 0.03-0.04 (decisions 430 / 443), larger than P1L - P1.
- The verdict rests on the registered point-estimate rule, not on a CI that excludes 0.
- Because the schedule is copied whole, the change lowers the trunk and adapter lr and also raises the route-branch lr. Which of the two matters is
  not separated.

## Highway read (behavior_stats.py as for NTV0 / P1; registered descriptive line)

| read | free-road highway steps | median v_e / v_h | p10 | < 0.85 | > 1.10 | mean dv (m/s) | highway slow + rear-hit pts (new) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| PCL1L-F-s0 | 820 | 1.00 | 0.85 | 10 % | 6 % | -0.63 | 1.10 |
| PCL1-F-s0 | 804 | 1.00 | 0.84 | 10 % | 6 % | -0.73 | 1.16 |
| NTV0-L18-s0 | 803 | 1.00 | 0.87 | 9 % | 6 % | -0.62 | 1.06 |
| RB0-F-s0 | 827 | 1.00 | 0.87 | 9 % | 8 % | -0.49 | 0.86 |

P1L is not slow on highway. Two highway-majority scenes have a median below 0.90: fd01e9ab 0.74 and ffe5b17d 0.81 (ffe5b17d is also slow in every
earlier read).

## Cost

4.2 GPU job-hours:
- training: 1.01 job-hours, eager, one card;
- 8 read chunks: 3.06 job-hours, at 4 per card about 0.8 card-hours;
- gates: 0.10.

That comes to about 1.9 card-hours, and 1.5 h wall. The snapshot PCL1L-F-s0x10000 is on the box (it is the same state as PCL1-F-s0x10000).
