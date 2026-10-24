# P1L-SPLIT: which half of P1L's lr switch matters? (round 9, 2026-10-24)

The registration is [../../plans/2026-10-24-p1l-split.md](../../plans/2026-10-24-p1l-split.md) (Chinese), written before any training output. The
lane is `scripts/p1ls.sh`, with gates and verdict in `scripts/p1ls.py`. It reuses `pcl.py` and `p1l.py` and edits neither. The box dir is
`$DATA_DIR/runs/round9/p1l_split/`: reads and scorer_v2 rescoring stay there. Full tables are in [verdict.md](verdict.md) and
[pai281.md](pai281.md). Decision 469. Seed 0 only. This is a diagnostic: the lines are descriptive.

**Question.** At T = 10 000, P1L (decision 461) moves every parameter group to NTV0-L18's schedule. That makes two opposite changes at once:
- the net (g0) and the old adapter (g1) drop about 4.8x;
- the route-memory branch (g2) rises about 2x.

P1L read 0.816 on the new scorer, against 0.793 for P1. Which change carries the gain?

**Answer.** Neither change does on its own. Both arms land on P1 on the new scorer:
- **P1La** (only net + adapter switch) reads **0.795**: +0.002 vs P1 and -0.021 vs P1L.
- **P1Lb** (only the route branch switches) reads **0.792**: -0.002 vs P1 and -0.025 vs P1L.

So both arms hit the registered line "that change alone does nothing". On the old scorer, P1La gives the same line. P1Lb is "partial", but on the
wrong side: 0.705, which is -0.012 vs P1.

Two readings fit the data, and one seed cannot tell them apart:
1. P1L's gain needs both changes together.
2. P1L's +0.023 over P1 is mostly seed noise. Its CI is [-0.010, +0.058], and seed spread is 0.03-0.04 (decisions 430 / 443).

The two arms do not differ from each other: P1La - P1Lb is +0.003 [-0.031, +0.038].

## Code change

`rm_train.py --lr2-groups NAMES` (commit 1ca148b6) only works together with `--lr2-at`.
- From T, only the named groups take the second schedule. The others keep their first schedule: base lr x warm-up x a cosine over all 19 500
  steps, with no floor.
- The names are `net`, `adapter`, `route` and `rx`. The default `''` means every group, which is P1L unchanged.
- The change is in separate hunks: one argparse entry, a docstring line, one per-group flag set at optimizer build, one `continue` in the lr
  loop, an info line and a guard.

The arms use P1L's argv plus one option:

| arm | tag | extra tokens | g0 / g1 after T | g2 after T |
|:--|:--|:--|:--|:--|
| P1La | PCL1LA-F-s0 | `--lr2-groups net,adapter` | NTV0-L18 schedule (peak 3e-6 / 3e-5, warm-up 500, cosine to 0.1) | P1's schedule (1.43e-4 at T, cosine to 0) |
| P1Lb | PCL1LB-F-s0 | `--lr2-groups route` | P1's schedule (1.43e-5 / 1.43e-4 at T, cosine to 0) | NTV0-L18 schedule (peak 3e-4, warm-up 100, cosine to 0.1) |

## Gates (all passed; submitted from 1ca148b6)

| gate | result |
|:--|:--|
| flag off: P1's argv vs the stored PCL1-F-s0, steps 0-100 | 119 / 119 scalars bit-identical, no extra or missing keys |
| flag off: P1L's argv (no new option), T = 10 000, vs PCL1L-F-s0, steps 0-100 | 131 / 131 bit-identical, `lr/*` included |
| flag off: P1L's argv, T = 30, vs P1LT30-F-s0, steps 0-75 | 170 / 170 bit-identical, including the lr after T |
| flag off: a whole 60-step run of P1L's argv (T = 30) vs P1L's stored smoke P1LSM-F-s0 | 231 / 231 bit-identical, including every lr before and after T and the final step |
| per arm, before T: the arm's argv vs PCL1L-F-s0, steps 0-100 | 131 / 131 bit-identical for both arms; the lr matches the formula (error 0) |
| per arm, T = 30, over 75 steps | step 25 bit-identical vs PCL1-F-s0 (75 / 75; the only new keys are `lr/*`). Pool rows and S terms (tcheck) pass. The lr matches the per-group formula (error 0) |
| 60-step smoke per arm (T = 30) | DONE, finite, batch 83 / 13 / 32 mixed, snapshot, pool rows, lr formula exact |
| S / imit gradient ratio, 500 steps after T (band 0.21-1.04) | P1La 0.653, P1Lb 0.596. For reference, P1L 0.755 and P1 0.584 |
| post | final dev ADE P1La 0.5229, P1Lb 0.5245 (<= 0.566). Pre-T segment vs P1L: 7 671 / 7 671 scalars bit-identical for both arms |
| lr over the whole run | 780 logged points per arm, max relative error vs the formula 0. The remaining horizon (9 500 steps) equals NTV0's |

The lane's lr check also reproduces P1L's own stored log of 780 points with error 0, so the formula is P1L's.

Logged lr (g0 / g1 / g2):

| step | P1La | P1Lb |
|--:|:--|:--|
| 10 025 | 1.50e-7 / 1.50e-6 / 1.43e-4 | 1.43e-5 / 1.43e-4 / 7.50e-5 |
| 10 500 | 2.98e-6 / 2.98e-5 / 1.32e-4 | 1.32e-5 / 1.32e-4 / 2.98e-4 |
| 15 000 | 1.54e-6 / 1.54e-5 / 3.77e-5 | 3.77e-6 / 3.77e-5 / 1.54e-4 |
| 19 500 | 3.00e-7 / 3.00e-6 / 1.95e-12 | 1.95e-13 / 1.95e-12 / 3.00e-5 |

## PAI 281, ACC + GAP (g10, FAST, P1L's chunk lists and serve), Harmonizer OFF

The comparison set is the 278 scenes that all six reads scored on both scorers, with no driver RPC error in any of them.

| read | new | old | zeros new / old | offroad | drift + side | route-turn | overshoot | AF-front | rear-hit / slow p | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **P1La** PCL1LA-F-s0 | **0.795** | 0.719 | 44 / 65 | 18 | 5 + 1 | 14 | 24 | 3 | 27 / 9 | 0.523 |
| **P1Lb** PCL1LB-F-s0 | **0.792** | 0.705 | 43 / 67 | 17 | 6 + 2 | 11 | 25 | 6 | 32 / 11 | 0.525 |
| P1L PCL1L-F-s0 | 0.816 | 0.737 | 38 / 60 | 14 | 6 + 3 | 10 | 23 | 4 | 27 / 11 | 0.525 |
| P1 PCL1-F-s0 | 0.793 | 0.718 | 44 / 65 | 16 | 6 + 1 | 14 | 23 | 5 | 28 / 10 | 0.523 |
| NTV0-L18-s0 | 0.818 | 0.746 | 38 / 58 | 12 | 6 + 0 | 15 | 22 | 3 | 26 / 11 | 0.536 |
| RB0-F-s0 | 0.718 | 0.649 | 68 / 87 | 19 | 14 + 4 | 21 | 21 | 8 | 23 / 8 | 0.528 |

### Registered lines

The rule: an arm within 0.01 of P1L carries P1L's gain; an arm within 0.01 of P1 alone does nothing; anything else is partial.

| scorer | P1L / P1 | P1La | P1Lb |
|:--|:--|:--|:--|
| new (primary) | 0.816 / 0.793 | 0.795: -0.021 vs P1L, +0.002 vs P1 (8 % of P1L - P1). **Alone does nothing** | 0.792: -0.025 vs P1L, -0.002 vs P1 (-7 %). **Alone does nothing** |
| old (reported) | 0.737 / 0.718 | 0.719: +0.002 vs P1. **Alone does nothing** | 0.705: -0.012 vs P1. Partial (below P1) |

### Paired differences (95 % bootstrap CI)

| pair | new | old | zeros (old) |
|:--|--:|--:|--:|
| P1La - P1L | -0.021 [-0.052, +0.009] | -0.018 [-0.049, +0.014] | +5 [-4, +14] |
| P1La - P1 | +0.002 [-0.028, +0.032] | +0.002 [-0.029, +0.033] | +0 [-9, +9] |
| P1Lb - P1L | -0.025 [-0.053, +0.002] | -0.032 [-0.063, -0.001] | +7 [-2, +16] |
| P1Lb - P1 | -0.002 [-0.028, +0.024] | -0.012 [-0.043, +0.018] | +2 [-7, +11] |
| P1La - P1Lb | +0.003 [-0.031, +0.038] | +0.014 [-0.022, +0.051] | -2 [-12, +8] |
| P1L - P1 (reference) | +0.023 [-0.010, +0.058] | +0.020 [-0.016, +0.056] | -5 [-15, +5] |

### Classes (old-scorer zero / partial classes; count difference with 95 % CI)

| pair | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| P1La - P1L | +1 [-2, +5] | +4 [+0, +9] | +4 [+0, +9] | -1 [-5, +3] | -2 [-5, +0] | -1 [-5, +2] | +0 [-4, +4] | -2 [-5, +0] |
| P1La - P1 | +1 [-2, +4] | +0 [-3, +3] | +2 [-3, +7] | -2 [-6, +2] | +0 [-3, +3] | -1 [-6, +4] | -1 [-6, +4] | -1 [-5, +2] |
| P1Lb - P1L | +2 [-2, +6] | +1 [-2, +5] | +3 [-1, +8] | +2 [-3, +7] | -1 [-4, +2] | +0 [-6, +5] | +5 [+0, +11] | +0 [-4, +4] |
| P1Lb - P1 | +2 [-3, +7] | -3 [-8, +1] | +1 [-4, +6] | +1 [-2, +5] | +1 [+0, +3] | +0 [-3, +3] | +4 [-1, +10] | +1 [+0, +3] |

What the class table shows:
- **P1La** (only the trunk and adapter lr drop) is P1 class for class. Against P1L it loses P1L's route-turn and offroad improvements, +4 each
  with the CI at 0.
- **P1Lb** (only the route-branch lr rises) keeps part of P1L's route-turn gain: 11 against 10 for P1L and 14 for P1. It pays for that in
  rear-hit partials: +5 [+0, +11] vs P1L and +4 vs P1. AF-front also rises, 6 against 4 and 5.
- No class difference excludes 0 beyond the CI edge.

## Reading

Neither half of the switch reproduces P1L. So "the lr at S entry was the cause" (decision 461) does not split into a trunk effect or a route-branch
effect. Either the gain needs both changes at once, or P1L - P1 (+0.023, CI includes 0) was mostly seed spread. One seed per arm cannot separate
these two readings.

Caveat: each arm is about 0.02 below P1L, a distance about as large as the seed spread. The registered lines are point estimates on one seed.

## Cost

The GPU jobs total 10.5 job-hours:
- trainings 3.19 h (2 x ~1.6 h, two to a card);
- reads 7.12 h (16 chunks, four to a card, so about 1.8 card-hours);
- gates and smokes 0.20 h.

That is about 3.6 card-hours, with a wall time of 2.5 h.
