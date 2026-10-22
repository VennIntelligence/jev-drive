# PAI-BASE: a base that sees PAI human-state rows from the start (round 9, 2026-10-22)

Pre-registration (Chinese, amendments 1-2): [../../plans/2026-10-22-pai-base.md](../../plans/2026-10-22-pai-base.md). Lane: `scripts/pb.sh`,
checks and reports: `scripts/pb.py`. Box run dir `$DATA_DIR/runs/round9/pai_base/`.

**Question.** Does a base that imitates PAI human-state rows from the start (RB0-P) beat the navtrain-only base RB0-F (decision 286) on the
PAI board, and does it then beat R6DAG after the unchanged L18 + DAgger stages?

**Answer.** No. All three PAI shares lose to RB0-F on PAI 281 (ACC + GAP, Harmonizer OFF). The registered base gate carries no share, so the
line stopped before the L18 / DAgger stages, as registered.

## Arms

RB0-P{30,50,70}-s0 = RB0-F-s0's meta.json argv unchanged (10 000 steps from the shipped weights, eager, lr, warm-up, 13 yaw-rate rows, route
memory and its 25 % drop). The only change: K = 35 / 58 / 80 of the 115 imitation rows per batch are cleaned PAI human-state rows. These come
from pool `on1_JTPH`, draw file = R6FIX's weights (MV-LC d430 + F1 / F2 d433, input >= 2 m/s) with the S rows at 0. Parts are
ph / ph05 / ph15 = 0.50 / 0.25 / 0.25. Flags: `--on-mode imit --on-normal` (plain imitation rows, the same memory drop as navtrain rows),
`--agent-lam 0`, which is what rm_train makes of RB0-F's `--agent-on-only` without a pool, and `--probe-n 0` (monitoring only, amendment 1).
No trainer code changed.

## Gates (all passed)

| gate | result |
|:--|:--|
| identity: RB0-F-s0's argv rerun on HEAD, eager, 10 000 steps vs the stored RB0-F-s0 (`df.py bitcmp --extra-ok`) | bit-identical: scalars 0.0, weights 0.0, dev ADE 0.5276 both ([identity.json](identity.json)) |
| data check of the draw file | 193 850 drawable rows (ph 81 801, ph05 73 413, ph15 38 636), 709 scenes, all in `physical_ai_av/pai-rows@v1`; 0 in test281 / curated-val / drives_val160 / train_exclude / exam scenes / the N batch's same-drive scenes / trainer-held logs ([wfile.json](wfile.json)) |
| smoke (50 %, 300 steps) | batch 57 / 13 / 58 imit, finite, dev ADE falls ([smoke.json](smoke.json)) |
| draw replay (CPU, the trainer's stream [0, 0, 53]) | on_hash at 60 / 1 000 / 10 000 equal to the trainer's for all three; drawn parts 0.50 / 0.25 / 0.25; no weight-0 or held row drawn; 1.8 / 3.0 / 4.1 passes per drawable row ([draws_*.json](draws_30.json)) |

Navtrain dev ADE at step 10 000: RB0-F 0.528; P30 0.540, P50 0.550, P70 0.583.

## Base reads: PAI 281, ACC + GAP (g10, FAST), Harmonizer OFF, 281 scenes, no RPC error

| base | mean | - RB0-F (paired 95 % CI) | zeros | route-turn | offroad | zero-other (drift / lane change) | AF-front | rear-hit partials | launched |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0-F-s0 | 0.650 | | 88 | 21 | 19 | 14 | 8 | 23 | 32 / 37 |
| RB0-P30-s0 | 0.593 | -0.057 [-0.110, -0.004] | 105 | 9 | 30 | 27 | 10 | 20 | 29 / 37 |
| RB0-P50-s0 | 0.605 | -0.045 [-0.097, +0.008] | 102 | 11 | 28 | 24 | 10 | 18 | 27 / 37 |
| RB0-P70-s0 | 0.564 | -0.085 [-0.140, -0.029] | 114 | 11 | 30 | 28 | 14 | 16 | 29 / 37 |

Full tables with paired class counts: [pai281_base.md](pai281_base.md). Gate table: [gate.md](gate.md).

**Reading.**
- At the base, the PAI rows do what they do in the later stages. Route-turn zeros fall (-10 to -12). Lateral zeros rise: offroad +9 to +11,
  zero-other +10 to +14. This is the MV-family lateral failure (d427 / d430). It outweighs the route-turn gain at every share.
- The PAI rows also cost 3-5 launches (d416: the human launches are short).
- The loss does not shrink with dose. 50 % is the least bad, but no share's CI clears 0.
- **Confound.** PAI rows differ from navtrain rows in two ways at once:
  - frames: real 10 Hz camera tokens against navtrain's warp frames;
  - labels: PAI humans drive in the lane next to the AlpaSim route on 10-14 % of states (d432, [../route_check/README.md](../route_check/README.md)).

  So this tests "PAI human imitation rows in the base", not "real frames vs warp". A frames-only test would need PAI frames with labels that
  follow the served route (e.g. the DAgger teacher's lane labels on human states).

## nuPlan arena held, native (guard, informational)

Amendment 2: the main session cancelled the P30 / P50 arena reads to free cards. Only RB0-F vs P70 was read
([arena_base.md](arena_base.md); 2 228 held windows read by both, log-clustered bootstrap).

| | mean | paired - RB0-F | zeros | corridor | offroad | AF-front |
|:--|--:|--:|--:|--:|--:|--:|
| RB0-F-s0 | 0.779 | | 474 | 392 | 55 | 11 |
| RB0-P70-s0 | 0.767 | -0.012 [-0.034, +0.011] | 498 | 353 | 94 | 37 |

On nuPlan, 70 % PAI rows do not move the mean (CI spans 0), but the zero mix shifts: fewer corridor zeros, more offroad and front collisions.
This is informational only: PAI keeps its own model.

## Cost

About 7 card-h:
- 4 base trainings at about 1 h each (contended box);
- 4 PAI 281 OFF reads;
- RB0-F and P70 arena reads.

The L18 / DAgger stage jobs and their reads failed through their dependency on the gate and never ran.
