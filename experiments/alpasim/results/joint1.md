# JOINT-1: one joint training for nuPlan submission #2, read with drop-one arms on all 1491 public scenes (2026-10-11, decision 254)

Pre-registration [plans/2026-10-11-joint1-prereg.md](../plans/2026-10-11-joint1-prereg.md) (commit 51608ab0, pushed before any read). Local closed loop only;
nothing was submitted. Generated read-out [joint1/joint1_report.md](joint1/joint1_report.md) (every table below comes from it), numbers
[joint1/joint1_stats.json](joint1/joint1_stats.json), per-scene scores [joint1/joint1_per_scene.csv](joint1/joint1_per_scene.csv). Code: trainer
`scripts/joint_train.py`, chains `scripts/joint_ident.sh`, `scripts/joint1_train.sh`, `scripts/joint1_chain.sh`, reader `scripts/joint1_report.py`.

## Result in short

- **No arm meets the registered line; no candidate for submission #2.** L1 (zeros of the three classes fewer than P2H10-F, scene-bootstrap CI
  excluding 0) fails for every arm: APS +12.5 [-4.0, +29.5], APS - yaw rows -3.0 [-17.0, +11.0], APS - AP2 +2.0 [-13.5, +18.0], APS - S (APY10m10-AB)
  -1.0 [-15.0, +13.5], P2H10S-F -1.5 [-13.5, +11.0]. L2 (mean not below P2H10-F by more than 0.002) holds for all five. L3's navtest clause
  (>= -0.2 against P2H10S-F) fails for every AlpaSim-standard arm (APS -0.44, APS - yaw -0.32, APY10m10-AB -0.59).
- **The candidate APS has more zeros than the base**: 87.5 against 75 (collision 15.5 / offroad 38.5 / corridor 33.5 against 11.5 / 44 / 19.5);
  corridor zeros +14.0 [+3.5, +24.5], collision + offroad -1.5 [-15.0, +12.5], mean 0.9262 against 0.9194 (+0.0068 [-0.0031, +0.0163]).
- **The ingredients interact negatively.** APS is worse than two of its drop-one arms: without the yaw-rate rows (AP2 + S) zeros fall by 15.5
  [2.5, 28.5], collision + offroad zeros by 13.5 [3.0, 24.5] and the mean rises by +0.0098 [+0.0000, +0.0194]; without S (= APY10m10-AB) zeros
  fall by 13.5 [1.5, 26.0] and the mean rises by +0.0072 [+0.0005, +0.0155].
- **Closest arm: APS - yaw rows (`JAPSnoY`, AP2 standard + S)**: mean 0.9360 (+0.0166 [+0.0029, +0.0306] over P2H10-F, the highest of the six),
  fewest at-fault collision + offroad zeros (40.5 against 55.5, -15.0 [-26.0, -4.0]), at-fault events 40.5 against 55.5, slow scenes 125 against
  265.5; it misses L1 because its corridor zeros rise by +12.0 [+2.0, +22.0] (total -3.0 [-17.0, +11.0]) and L3 on navtest (-0.32 against the
  -0.2 limit). P2H10S-F misses L1 only. The user decides what is submitted.

## Arms and the trainer

| Arm | Tags | Recipe | Rows per batch (normal / yaw-rate / hinge-only) | dev ADE (s0 / s1, m) |
|:--|:--|:--|:--|:--|
| APS | `JAPS-s{0,1}` | AP2 standard + S + yaw-rate rows | 102 / 13 / 13 | 0.559 / 0.561 |
| APS - yaw rows | `JAPSnoY-s{0,1}` | AP2 standard + S | 115 / 0 / 13 | 0.558 / 0.561 |
| APS - AP2 | `JAPSnoA-s{0,1}` | S + yaw-rate rows (NAVSIM standard) | 102 / 13 / 13 | 0.548 / 0.545 |
| APS - S | `APY10m10-AB-s{0,1}` (stored) | AP2 standard + yaw-rate rows | 115 / 13 / 0 | 0.552 (s0) |
| reference | `P2H10S-F-s{0,1}` (stored) | S | 115 / 0 / 13 | 0.545 (s0) |
| the line's base | `P2H10-F-s{0,1}` (stored) | lambda-10 hinge | 128 / 0 / 0 | 0.542 (s0) |

S = agent hinge (shape-only) on the own plan of every imitation row of the body1 train logs and of the hinge-only rows, hinge-only off-track rows
(ot1 4 / yr1 4 / bd4 5, weight 3), road-and-lane raster on them (body1 Amendment 6). 10 000 steps x 128, eager, `--label-bank`; one card each at
3.6-5.5 it/s next to closed-loop stacks (31-47 min), 21-24 GB peak.

**Trainer identity** (`results/joint1/ident_*.json`): 60 steps, seed 0, one shard, `joint_train.py` against pp_train.py (plain and with a
memory bank), bd4_train.py, ap2_train.py and ot3_rows.py under both standards: all 109 net and 46 / 53 adapter tensors equal bit for bit in all six
(the merged trainer with `--label-bank` against the originals' dense rasters). Phase 2 extends this to full length: the joint trainer reproduces
`P2H10-F-s0 / s1` and `P2H10S-F-s0 / s1` bit for bit (experiments/eyes_sense/results/identity_*.json). So the stored APY10m10-AB, P2H10S-F and
P2H10-F checkpoints are what this trainer gives for those flags.

**Pilots** (`results/joint1/pilot_*.json`): 300 steps each, then 48 public part001 scenes: composition as registered, every ingredient's loss
term logged, the driver family as registered (`ap2` for JAPS / JAPSnoY, `sh30` for JAPSnoA), 48 / 48 rollouts, 576 drive records, 0 non-finite
plans. All three families passed.

## Closed loop: one driver path for every arm

Five scene lists (c0b chunk0-2 = the 700 of decision 201, cf1fresh 391, ot3new 400), one pool job per checkpoint and list, the box's default driver
(compiled model passes), family by the checkpoint record. Reproducibility checks before reusing stored reads ([joint1/joint1_checks.md](joint1/joint1_checks.md)):

| Rerun | Against | Scenes that differ | Verdict |
|:--|:--|--:|:--|
| P2H10-F-s0, chunk0 | TR1's run (2026-10-10, 894bf6e0) | 0 of 234 | pass |
| P2H10-F-s0, cf1fresh | CF1's run (2026-10-09, c9d2e07b) | 0 of 391 | pass |
| APY10m10-AB-s0, cf1fresh | CF1's run | 22 of 391 (max 0.0051, no zero flips; mean 0.9182 / 0.9183) | fail |

So the stored `sh30` reads were reused (P2H10-F on the 700 from TR1 and on cf1fresh from CF1, P2H10S-F on the 700 from BODY1) and every stored `ap2`
read was rerun (stage `refx`): the `ap2` driver path moved at tolerance level since CF1. OT3's uncompiled runs were rerun on the compiled path
(P2H10-F on ot3new, APY10m10-AB on the 700). P2H10S-F s0-s3 on the 791 are lane nuplan1491's runs (same lists and driver).

## Per arm (two-seed per-scene means; zeros are seed means; 1491 scenes, 44 logs)

| Arm | Mean [CI logs] | Seeds | Zeros: collision / offroad / corridor (sum) | At-fault events | Slow | Progress | > 45 deg (142 scenes): mean, zeros |
|:--|:--|:--|:--|--:|--:|--:|:--|
| P2H10-F | 0.9194 [0.9018, 0.9388] | 0.9183 / 0.9206 | 11.5 / 44 / 19.5 (75) | 55.5 | 265.5 | 0.915 | 0.813, 23 |
| P2H10S-F | 0.9226 [0.9052, 0.9411] | 0.9201 / 0.9251 | 7.5 / 39.5 / 26.5 (73.5) | 47 | 245 | 0.923 | 0.758, 31.5 |
| APY10m10-AB (APS - S) | 0.9334 [0.9172, 0.9502] | 0.9344 / 0.9324 | 12 / 39.5 / 22.5 (74) | 51.5 | 139.5 | 0.963 | 0.821, 24 |
| **APS** | 0.9262 [0.9085, 0.9436] | 0.9266 / 0.9258 | 15.5 / 38.5 / 33.5 (87.5) | 54 | 115 | 0.971 | 0.767, 32 |
| APS - yaw rows | **0.9360** [0.9152, 0.9553] | 0.9338 / 0.9382 | 8.5 / 32 / 31.5 (72) | **40.5** | 125 | 0.969 | 0.771, 31 |
| APS - AP2 | 0.9193 [0.9006, 0.9389] | 0.9195 / 0.9190 | 8.5 / 36 / 32.5 (77) | 44.5 | 256.5 | 0.917 | 0.728, 35 |

Per seed, zeros of the three classes: P2H10-F 78 / 72, P2H10S-F 77 / 70, APY 72 / 76, APS 85 / 90, APS - yaw 74 / 70, APS - AP2 76 / 78; the 12
route-fold scenes (zero, no decision, for every driver) are outside these counts. Descriptive only: P2H10S-F s2 / s3 0.9269 / 0.9250, zeros 68 / 73.
The same reference rows on stored reads of mixed driver paths (decision 253): P2H10-F 0.9196, APY10m10-AB 0.9335, P2H10S-F 0.9243 (four seeds).

## Registered line and guardrails

| Arm | L1 zeros vs P2H10-F [CI scenes] | L2 mean vs P2H10-F [CI logs] | navtest EPDMS, vs P2H10S-F | navhard combined, vs P2H10-F | HUGSIM 64 HD, vs P2H10-F | Verdict |
|:--|:--|:--|:--|:--|:--|:--|
| APS | +12.5 [-4.0, +29.5] no | +0.0068 [-0.0031, +0.0163] yes | 88.61, -0.44 [-0.69, -0.19] no | 32.49, +0.65 [-1.63, +3.08] yes | 0.371, -0.025 [-0.074, +0.023] yes | not a candidate |
| APS - yaw rows | -3.0 [-17.0, +11.0] no | +0.0166 [+0.0029, +0.0306] yes | 88.73, -0.32 [-0.55, -0.08] no | 31.82, -0.02 [-1.86, +1.83] yes | not read | not a candidate |
| APS - AP2 | +2.0 [-13.5, +18.0] no | -0.0002 [-0.0088, +0.0095] yes | 89.11, +0.06 [-0.12, +0.23] yes | 34.95, +3.11 [+0.63, +5.87] yes | not read | not a candidate |
| APY10m10-AB | -1.0 [-15.0, +13.5] no | +0.0140 [+0.0045, +0.0231] yes | 88.46, -0.59 [-0.85, -0.33] no | 30.53, -1.31 [-3.42, +0.88] yes | not read | not a candidate |
| P2H10S-F | -1.5 [-13.5, +11.0] no | +0.0032 [-0.0034, +0.0106] yes | 89.05, 0 (the reference) | 34.06, +2.22 [+0.25, +4.36] yes | 0.403, +0.008 [-0.027, +0.049] yes | not a candidate (L1 only) |

P2H10-F: navtest 88.67, navhard 31.84, HUGSIM 0.395 (two seeds). HUGSIM was registered for APS only; the conditional HUGSIM read of a picked arm
was not needed (no arm passes L1). WOD-E2E was skipped as registered (no read path for the AlpaSim input standard, decision 217).

## Zero composition and fixed vs broken

Fixed / broken scenes (zero in both reference seeds -> in neither arm seed, and the reverse), three classes: APS vs P2H10-F 26 / 32 (corridor
6 / 14, collision + offroad 22 / 21); APS - yaw vs P2H10-F 25 / 18 (collision + offroad 19 / 6, corridor 7 / 14); APS - AP2 vs P2H10-F 26 / 24;
APY vs P2H10-F 23 / 18; P2H10S-F vs P2H10-F 17 / 10 (collision + offroad 14 / 3, corridor 3 / 7). APS against its drop-one arms: vs APS - yaw 9 / 22
(collision + offroad 6 / 18), vs APS - AP2 23 / 25, vs APY 7 / 19. Scenes that are zero in at least one seed: P2H10-F 87, P2H10S-F 88, APY 92, APS 106,
APS - yaw 83, APS - AP2 94; 19 scenes are zero only in APS (in no drop-one arm), 4 are zero in every drop-one arm but not in APS. Every arm with S
has more zeros in the > 45 deg bucket (31-35 against 23-24), mostly corridor (15.5-20 against 8.5-9): decision 233's wide right turns, on 1491
scenes.

## Non-additivity read (decision 244's wording, E = APS - P2H10-F)

| Quantity | E | Yaw-rate rows: X - APS / X - base | AP2 standard: X - APS / X - base | S: X - APS / X - base |
|:--|:--|:--|:--|:--|
| zeros, three classes | +12.5 [-4.0, +29.5] | -15.5 [-28.5, -2.5] / -3.0 [-17.0, +11.0] | -10.5 [-26.5, +5.0] / +2.0 [-13.5, +18.0] | -13.5 [-26.0, -1.5] / -1.0 [-15.0, +13.5] |
| collision + offroad zeros | -1.5 [-15.0, +12.5] | -13.5 [-24.5, -3.0] / -15.0 [-26.0, -4.0] | -9.5 [-21.0, +2.0] / -11.0 [-22.5, +0.0] | -2.5 [-12.5, +7.5] / -4.0 [-14.5, +7.0] |
| corridor zeros | **+14.0 [+3.5, +24.5]** | -2.0 [-10.5, +6.5] / +12.0 [+2.0, +22.0]: **not needed** | -1.0 [-11.5, +10.0] / +13.0 [+2.0, +24.5]: **not needed** | -11.0 [-20.0, -2.5] / +3.0 [-6.5, +12.5]: **carries** |
| mean scene score | +0.0068 [-0.0031, +0.0163] | +0.0098 [+0.0000, +0.0194] / +0.0166 [+0.0029, +0.0306] | -0.0070 [-0.0174, +0.0053] / -0.0002 [-0.0088, +0.0095] | +0.0072 [+0.0005, +0.0155] / +0.0140 [+0.0045, +0.0231] |

E excludes 0 only for corridor zeros; for the other three quantities the registered read is "no effect of APS to attribute". One sentence each:
- **Yaw-rate rows**: not needed for APS's corridor-zero increase, and nothing else of APS over the base is theirs to attribute; inside the joint
  training they cost (dropping them removes 15.5 zeros and 13.5 collision + offroad zeros, CIs excluding 0).
- **AP2 input standard**: not needed for the corridor-zero increase; it is what halves the slow scenes (APS - AP2 256.5 against 115) and lifts
  progress (0.917 to 0.971); its zero and mean effects inside APS are inside the CIs.
- **Body lesson S**: carries APS's corridor-zero increase (dropping it gives back 11.0 of the 14.0); dropping it also removes 13.5 zeros overall
  and adds +0.0072 to the mean, so inside APS it costs more than it brings on this board.

## Limits

- All 1491 public scenes had been read by recipes of this family before (decision 216, amendment 1): a development set, two seeds per arm, five arms
  against one line without multiplicity correction; local render, the compiled driver path (the image serves the interpreted one).
- L1 counts corridor zeros with collision and offroad; the arms with S trade the latter for the former, which the line nets out.
- HUGSIM only for APS; WOD-E2E not read.

## Cost (pool event log, `scripts/joint_usage.py`, both phases together)

495 pool jobs, 41.3 job-hours: trainings 3.7 card-hours (booked VRAM share; 8.5 GPU job-hours), closed loop 7.0 (18.9 stack-hours, including lane
nuplan1491's P2H10S-F s0-s3 on 791 scenes), HUGSIM 0.7, bench navtest / navhard GPU stages under 0.2 (their scoring is CPU); 11.5 card-hours in total,
28.8 GPU job-hours if every GPU job had had a card to itself.
