# Harmonizer closed-loop paired read (PARTIAL): PAI 281, ACC + GAP, ON vs stored OFF

**PARTIAL (2026-10-21 20:00).** NTV0-L18-s0 is complete (280 scenes). R6DAG-L18-s1 has only chunks c000-c002 (108 scenes): the main session
cancelled its queued c003-c007 so DAgger iteration 2's 281 reads go first (plan amendment 1). CIs on 108 scenes are wide. These numbers do
not answer the registered submission question. No decisions row yet. The rest of R6DAG-s1 and the full report follow when cards are idle.

Plan and reading rules, registered before any read: [plans/2026-10-21-harmonizer-closedloop.md](../plans/2026-10-21-harmonizer-closedloop.md).
Code: [scripts/harmcl.sh](../scripts/harmcl.sh) (`submit`, `check`, `ptax`, `partial`) and [scripts/harmcl.py](../scripts/harmcl.py).
Raw tables: [pai281_harm_ntv0.md](pai281_harm_ntv0.md) (NTV0, 280 scenes), [pai281_harm_p3.md](pai281_harm_p3.md) (both models, c000-c002), and
`harmonizer_closedloop_{ntv0,p3}.{json,txt}`. Box: `$DATA_DIR/runs/round7/harmcl/` (ON reads; facts `runs/pai_fix/facts/*-281agH*.csv`).

The ON serve is `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 HARMONIZER=1`, g10, ALPASIM_FAST=1, CONC 4, with the chunk table of the stored
NTV0 read. **Harmonizer check:** all 11 renderer logs show `"enable_harmonizer": true` and `DifixModelFactory: checkpoint ready at
.../harmonizer_nontemporal.pt`. OFF reads are the stored ones (NTV0 `runs/defect_fix/pai_accgap/NTV0-L18-s0`, R6DAG-s1 `runs/round6/dag/pai281/R6DAG-L18-s1-ag`).

## Tables

**ON - OFF per model.** Mean difference, paired scene bootstrap 95 % CI. Zeros are counted ON / OFF, with zeros the ON read added and removed.

| Model | Scenes | OFF | ON | ON - OFF | Zeros ON / OFF (new, fixed) | Junction-turn scenes: ON - OFF | LAUNCH-37 launched ON / OFF |
|---|--:|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0, full | 280 | 0.744 | 0.746 | **+0.002 [-0.032, +0.035]** | 58 / 59 (11, 12) | -0.015 [-0.109, +0.078] (74) | 33 / 33 of 37 |
| NTV0-L18-s0, c000-c002 | 108 | 0.754 | 0.734 | -0.020 [-0.084, +0.044] | 24 / 22 (7, 5) | -0.081 [-0.221, +0.057] (36) | 13 / 13 |
| R6DAG-L18-s1, c000-c002 | 108 | 0.815 | 0.800 | -0.014 [-0.063, +0.034] | 18 / 17 (4, 3) | -0.027 [-0.113, +0.057] (36) | 13 / 13 |

**Zero classes, ON - OFF count with 95 % CI and (new, fixed).** Offroad is split by facts `primary`: turn inside = `off_junc_inside` + `off_curve_inside`.

| Class | NTV0 full (280) | NTV0 c000-c002 (108) | R6DAG-s1 c000-c002 (108) |
|---|--:|--:|--:|
| overshoot | 21 / 23: -2 [-6, +2] (1, 3) | 8 / 10: -2 [-5, 0] | 8 / 8: 0 |
| route-turn | 9 / 15: **-6 [-12, -1]** (1, 7) | 3 / 4: -1 [-4, +2] | 1 / 1: 0 |
| offroad, all | 17 / 12: +5 [-1, +11] (7, 2) | 10 / 6: +4 [-1, +10] (6, 2) | 6 / 6: 0 [-4, +4] (2, 2) |
| offroad, turn inside | 5 / 3: +2 [0, +5] (2, 0) | 3 / 1: +2 (2, 0) | 3 / 3: 0 (1, 1) |
| offroad, turn outside | 4 / 3: +1 [0, +3] (1, 0) | 2 / 1: +1 | 1 / 1: 0 |
| offroad, junction straight | 2 / 2: 0 | 1 / 1: 0 | 0 / 1: -1 |
| offroad, straight | 5 / 4: +1 [-3, +5] (3, 2) | 3 / 3: 0 (2, 2) | 1 / 1: 0 (1, 1) |
| AF-front | 5 / 3: +2 [-2, +6] (3, 1) | 0 / 0 | 0 / 0 |
| side | 0 / 0 | 0 / 0 | 2 / 1: +1 |
| zero-other (drift / lane change) | 6 / 6: 0 [-5, +5] (3, 3) | 3 / 2: +1 | 1 / 1: 0 |
| partial rear-hit | 27 / 26: +1 | 10 / 10: 0 | 9 / 8: +1 |
| partial slow | 9 / 11: -2 | 4 / 5: -1 | 5 / 4: +1 |

**Difference in differences, c000-c002 only (108 scenes, 36 junction-turn).**

| Quantity | Value |
|---|--:|
| R6DAG - NTV0, OFF | +0.061 [+0.001, +0.124] |
| R6DAG - NTV0, ON | +0.066 [-0.002, +0.138] |
| DiD (R6DAG - NTV0)_ON - (R6DAG - NTV0)_OFF | **+0.005 [-0.064, +0.077]** |
| DiD on junction-turn scenes | +0.054 [-0.112, +0.223] |

On all 279 scenes, R6DAG - NTV0 under OFF is +0.072 [+0.034, +0.111] (decision 434). The 108-scene subset gives the same OFF margin (+0.061).

**Noise reference (registered).** An OFF re-read is scene-deterministic (decisions 406, 435, 436), so ON - OFF contains no re-read noise. Small
perturbations still churn scenes. The holdlead serve change of decision 370 gave +0.0004 [-0.014, +0.013] with zeros +2. Seed to seed: NTV0
s1 - s0 is -0.006 [-0.026, +0.013], and R6DAG s1 - s0 is +0.040 (new 3, fixed 13 zeros). NTV0's ON churn of 11 new and 12 fixed zeros on 280
scenes is a perturbation of about seed size that nets out to zero.

## Reading (partial; per the registered rules where they apply)

- **NTV0, full read: within the small-perturbation level.** The difference is +0.002 with a CI inside ±0.035, and |point| < 0.02. The
  Harmonizer does not move NTV0's 281 mean. It does move classes in the direction the open loop predicted (tighter turns). Route-turn zeros
  fall by 6, the only class whose CI excludes 0. Offroad zeros rise by 5 (CI includes 0): turn inside +2, turn outside +1, straight +1. The
  registered "consistent with open loop" rule (turn inside up by at least 3 in both models, outside not up) is not met: inside is +2, and
  R6DAG's part is not complete. LAUNCH-37 is unchanged (33 / 33), so no launch flag.
- **R6DAG-s1 (108 scenes):** -0.014 [-0.063, +0.034], with no class change beyond ±1. Its offroad and route-turn counts do not move. That fits
  R6DAG's lateral teacher having already removed most route-turn zeros (1 of 108 under OFF), which leaves fewer turns for tighter plans to rescue.
- **DiD on 108 scenes: +0.005 [-0.064, +0.077].** There is no sign that the Harmonizer shrinks R6DAG's lead over NTV0. Under ON, R6DAG - NTV0
  is +0.066, against +0.061 under OFF on the same scenes. The CI is too wide (±0.07) to call. The registered submission reading needs all 279
  scenes and stays open.
- **Cost so far.** 11 ON chunks (389 scenes) took 15.3 stack-hours, 2.4 min a scene. At 2-3 stacks per card that is about 5-8 card-hours,
  plus two CPU tax jobs. ON output on disk: 5.6 GB (asl of zero-score rollouts only).

## Remaining

R6DAG-s1 c003-c007 (173 scenes, about 7 stack-hours) via `harmcl.sh resume` (moves the cancelled chunks' `SUBMITTED` marks aside, `pai_box.sh submit` with the same env, the full tax, the report job), then `harmcl.sh wait`. After that, the
decisions row.
