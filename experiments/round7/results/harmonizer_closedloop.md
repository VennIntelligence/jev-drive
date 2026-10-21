# Harmonizer closed-loop paired read: PAI 281, ACC + GAP, ON vs stored OFF (NTV0-L18-s0, R6DAG-L18-s1)

Plan and reading rules, registered before any read, plus amendment 1 (the pause): [plans/2026-10-21-harmonizer-closedloop.md](../plans/2026-10-21-harmonizer-closedloop.md).
Code: [scripts/harmcl.sh](../scripts/harmcl.sh) and [scripts/harmcl.py](../scripts/harmcl.py). Raw tables: [pai281_harm.md](pai281_harm.md) (r2.py r281: all
classes, new and fixed zeros for each pair) and [harmonizer_closedloop.txt](harmonizer_closedloop.txt) / [.json](harmonizer_closedloop.json) (offroad split, DiD).
The partial report, written while R6DAG-s1 c003-c007 were paused for DAgger iteration 2's reads, is kept as `pai281_harm_{ntv0,p3}.*` and
`harmonizer_closedloop_{ntv0,p3}.*`. Box: `$DATA_DIR/runs/round7/harmcl/` (the ON reads are new reference reads; facts in `runs/pai_fix/facts/*-281agH.csv`).

The ON serve is `JEV_VCONT=1.0 JEV_LEAD=1 JEV_ACC=1 JEV_GAP=0.7,2.7 HARMONIZER=1`, g10, ALPASIM_FAST=1, CONC 4, with the stored NTV0 read's chunk table
(8 chunks). OFF reads are the stored ones (NTV0 `runs/defect_fix/pai_accgap/NTV0-L18-s0`, R6DAG-s1 `runs/round6/dag/pai281/R6DAG-L18-s1-ag`).
**Harmonizer check:** all 16 renderer logs show `"enable_harmonizer": true` and `DifixModelFactory: checkpoint ready at .../harmonizer_nontemporal.pt`.
279 scenes were scored without an RPC error in all four reads. R6DAG ON had no RPC errors.

## Tables

**ON - OFF per model, 279 scenes.** Paired scene bootstrap 95 % CI (10 000 draws).

| Model | OFF | ON | ON - OFF | Zeros ON / OFF (new, fixed) | Junction-turn scenes (74): ON - OFF | LAUNCH-37 launched ON / OFF |
|---|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0 | 0.743 | 0.745 | +0.002 [-0.031, +0.035] | 58 / 59 (11, 12) | -0.015 [-0.109, +0.078] | 33 / 33 |
| R6DAG-L18-s1 | 0.815 | 0.780 | **-0.035 [-0.066, -0.005]** | 49 / 43 (12, 6) | -0.012 [-0.080, +0.054] | **32 / 35** |

**Classes, ON / OFF count and ON - OFF with 95 % CI (new, fixed).** Offroad is split by facts `primary`: turn inside = `off_junc_inside` + `off_curve_inside`.

| Class | NTV0-L18-s0 | R6DAG-L18-s1 |
|---|--:|--:|
| overshoot | 21 / 23: -2 [-6, +2] (1, 3) | 21 / 19: +2 [0, +5] (2, 0) |
| route-turn | 9 / 15: **-6 [-12, -1]** (1, 7) | 2 / 2: 0 |
| offroad, all | 17 / 12: +5 [-1, +11] (7, 2) | 15 / 12: +3 [-3, +9] (6, 3) |
| offroad, turn inside | 5 / 3: +2 [0, +5] (2, 0) | 8 / 7: +1 [-3, +5] (3, 2) |
| offroad, turn outside | 4 / 3: +1 [0, +3] (1, 0) | 1 / 1: 0 |
| offroad, junction straight | 2 / 2: 0 | 2 / 2: 0 (1, 1) |
| offroad, straight | 5 / 4: +1 [-3, +5] (3, 2) | 3 / 2: +1 [-2, +5] (2, 1) |
| AF-front | 5 / 3: +2 [-2, +6] (3, 1) | 3 / 3: 0 (1, 1) |
| side | 0 / 0 | 3 / 3: 0 (1, 1) |
| zero-other (drift / lane change) | 6 / 6: 0 (3, 3) | 5 / 4: +1 [-2, +5] (2, 1) |
| partial rear-hit | 27 / 26: +1 [-3, +6] (3, 2) | 26 / 21: +5 [-1, +11] (7, 2) |
| partial slow | 9 / 11: -2 [-6, +2] (1, 3) | 8 / 6: +2 [-2, +6] (3, 1) |

**Which model is better, ON vs OFF (279 scenes).**

| Quantity | Value |
|---|--:|
| R6DAG-s1 - NTV0, OFF | +0.072 [+0.035, +0.111] (zeros -16: new 6, fixed 22) |
| R6DAG-s1 - NTV0, ON | +0.035 [-0.006, +0.078] (zeros -9: new 12, fixed 21; route-turn -7 [-13, -2]) |
| **DiD** (R6DAG - NTV0)_ON - (R6DAG - NTV0)_OFF | **-0.037 [-0.078, +0.005]** |
| DiD on junction-turn scenes (74) | +0.003 [-0.106, +0.111] |

R6DAG-s1 lost 3 launches under ON. One scene (268949d2) was already a side-collision zero under OFF; the other two are 5266c7c9 (creep,
1.0 -> 0.12) and a686c7d0 (rear-hit, 1.0 -> 0.16). 15 R6DAG scenes drop by more than 0.5 and 6 gain by more than 0.5. Of the 279 scenes, the 108 in chunks c000-c002
give -0.014 and the other 171 give -0.049. A re-read is scene-deterministic, so this is a spread across scenes, not across runs.

**Noise reference (registered).** An OFF re-read is scene-deterministic (decisions 406, 435, 436), so ON - OFF contains no re-read noise.
For comparison, the holdlead serve change (decision 370) moved the mean by +0.0004 [-0.014, +0.013] with zeros +2. Between seeds, NTV0 s1 - s0 is -0.006 [-0.026, +0.013] and
R6DAG s1 - s0 is +0.040 [+0.014, +0.068].

## Reading (registered rules)

- **NTV0: within the small-perturbation level.** The difference is +0.002, with the CI inside ±0.035. The Harmonizer swaps route-turn zeros
  (-6, the only class whose CI excludes 0) for offroad zeros (+5: turn inside +2, turn outside +1, straight +1), and the two cancel. The
  direction fits the open-loop "turns tighter" finding.
- **R6DAG-s1: material.** The difference is -0.035 [-0.066, -0.005]: the CI excludes 0 and |point| is at least 0.02. The loss is not in turns.
  Route-turn is unchanged (2 / 2), turn-inside offroad is +1, and the junction-turn scenes are unchanged. It is spread over longitudinal
  classes: rear-hit partials +5, overshoot +2, slow +2, and launches -3, which hits the registered launch flag (a drop of 3 or more).
- **Open-loop consistency rule:** not met. Turn-inside offroad is +2 / +1; the rule needs at least +3 in both models.
- **Submission rule:** with the Harmonizer ON, R6DAG - NTV0 is +0.035 [-0.006, +0.078]. The point is above 0 but the CI includes 0, so R6DAG-s1
  is **no longer clearly better** under the finals renderer. The DiD is -0.037 [-0.078, +0.005]. Its CI includes 0 (just), so by the rule
  it is not recorded as a significant loss of the lead. Read plainly, the Harmonizer roughly halves R6DAG's lead and does not reverse it.
- **Speculation, not tested.** In the open loop (decision 435) ON plans of the MV family were shorter: R5LC (R6DAG's student base) gave a
  signed 4 s longitudinal shift of -0.12 m [-0.21, -0.04], -0.16 m at launch and -0.22 m on straights, while NTV0 gave +0.06 m. That matches
  R6DAG's closed-loop losses being longitudinal: shorter plans lead to slow launches, being rear-ended, and partial stops. Check: run
  `harm.py` forward on the stored 300 states with R6DAG-L18-s1 (GPU minutes, no new renders) and compare its signed longitudinal shift
  with NTV0's.

## Cost

16 ON chunks (281 scenes per model) took 18.1 stack-hours. NTV0 took 10.6 h. R6DAG took 7.5 h, because c003-c007 ran on idle
cards in about 34 minutes each against about 90 minutes under contention. At 2-3 stacks per card that is about 6-9 card-hours, plus three
CPU tax jobs. ON output on disk is 7.8 GB (rollout.asl kept for zero-score rollouts only).
