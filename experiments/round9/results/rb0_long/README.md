# RB0-LONG: one clean offline training run, 19 500 steps from the shipped weights (round 9, 2026-10-23)

Pre-registration (Chinese, with an amendment for the new scorer): [../../plans/2026-10-23-rb0-long.md](../../plans/2026-10-23-rb0-long.md). Lane:
`scripts/rbl.sh`, gates and report: `scripts/rbl.py`. Box run dir: `$DATA_DIR/runs/round9/rb0_long/`. The read and the scorer_v2 mirror stay on
the box. Checkpoints: `runs/op_parity/runs/RB0L-F-s0` and the snapshot at 10 000 steps, `RB0L-F-s0x10000`, which has not been read.

**Question.** Can a single offline training run, started from the shipped weights, reach NTV0's PAI score if it gets NTV0's total number of
steps (10 000 + 9 500)?

**Answer.** No. On PAI 281 (ACC + GAP, Harmonizer OFF) RB0L-F-s0 scores **0.600**. That is below the registered 0.67 line ("more offline steps
do not give NTV0's gain"). It is also below RB0-F-s0 itself, mostly because offroad zeros double. The new scorer lands on the same line.
Seed 0 only, as approved.

## Arm and gates

The arm is RB0-F-s0's stored meta.json argv run eager (`JEV_RM_EAGER=1`, as RB0-F was), with no `--init`. `rbl.sh dry` gives this token diff:
`--tag RB0-F-s0 -> RB0L-F-s0`, `--steps 10000 -> 19500`, and an appended `--save-at 10000`. The snapshot takes no RNG and does no update. Nothing
else changed: the seed, the warm-up and the lr are the same, and the cosine simply spans 19 500 steps.

| gate | result |
|:--|:--|
| identity: RB0-F's argv at `--steps 10000` on HEAD, stopped after step 75, vs the stored RB0-F-s0 | bit-identical: 22 / 22 scalars at steps 0 / 25 / 50 / 75, max diff 0.0, no missing or extra keys ([identity.json](identity.json)) |
| smoke: the arm's argv with 60 steps and `--save-at 30` | DONE, 32 scalars all finite, last step 60, snapshot written ([smoke.json](smoke.json)) |

## PAI 281, ACC + GAP (g10, FAST, NTV0's chunk lists), Harmonizer OFF

Old scorer (the read's own scores, as registered). The reads share 280 scenes without an RPC error ([pai281.md](pai281.md), [verdict.md](verdict.md)).

| read | mean | zeros | offroad | drift / lane change (zero-other + side) | route-turn | stop-line overshoot | AF-front | rear-hit / slow partials | launched |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RB0L-F-s0 | **0.600** | 102 | 37 | 11 + 1 | 18 | 25 | 10 | 23 / 9 | 33 / 37 |
| RB0-F-s0 | 0.648 | 88 | 19 | 14 + 4 | 21 | 22 | 8 | 23 / 8 | 32 / 37 |
| NTV0-L18-s0 | 0.744 | 59 | 12 | 6 + 0 | 15 | 23 | 3 | 26 / 11 | 33 / 37 |
| NTV0NS-L18-s0 (its own 278-scene set) | 0.624 | 93 | 21 | 13 + 2 | 20 | 23 | 14 | 24 / 6 | 32 / 37 |

Paired differences (95 % bootstrap CI, scenes as units):

| pair | old scorer | new scorer (scorer_v2, d453) |
|:--|--:|--:|
| RB0L - RB0-F-s0 | -0.048 [-0.093, -0.002] | -0.034 [-0.079, +0.012] |
| RB0L - NTV0-L18-s0 | -0.144 [-0.191, -0.099] | -0.133 [-0.179, -0.089] |
| NTV0-L18-s0 - RB0-F-s0 | +0.096 [+0.060, +0.135] | |
| RB0L - NTV0NS-L18-s0 (278 scenes) | -0.026 [-0.070, +0.017] | |

On the new scorer RB0L / RB0-F / NTV0 score 0.682 / 0.716 / 0.816, with 79 / 69 / 39 zeros. That puts RB0L at -34 % of the RB0-F -> NTV0 gap.
The registered lines, converted to shares of that gap, are >= 74 % and <= 21 %, so the new scorer also reads "more offline steps do not give
NTV0's gain". Both scorers land on the same line.

Against RB0-F the extra zeros are offroad: +18 [+8, +28] (23 new, 5 fixed). Route-turn, drift and side zeros fall slightly, and the other
classes stay within noise.

## Navtrain dev ADE (m)

| run | dev ADE |
|:--|--:|
| RB0L-F-s0 @ 10 000 | 0.544 |
| RB0L-F-s0 @ 19 500 | 0.511 |
| RB0-F-s0 @ 10 000 | 0.528 |
| NTV0-L18-s0 (10 000 + 9 500) | 0.536 |

The longer run fits navtrain better than every other run here, yet it drives worse on PAI. Better offline imitation does not turn into
closed-loop score, and the extra steps seem to cost lateral robustness (offroad).

## Cost

| part | wall time | card use |
|:--|--:|:--|
| training | 63 min (eager, 40 GB booked) | 1.05 card-hours |
| gates | under 2 min | negligible |
| 8 read chunks | 20-26 min each, spread over all 6 cards | 3.1 job-hours; at 4 per card about 0.8 card-hours |
| total | about 1.7 h from submit to DONE | about 1.9 card-hours |
