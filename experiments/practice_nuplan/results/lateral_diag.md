# LATERAL-DIAG: why NTVX is worse laterally than NTV0 on held 460 (2026-10-19)

Diagnosis on stored data, then one confirmatory run that was registered first. Code: [scripts/lat_diag.py](../scripts/lat_diag.py).
- Subcommands: `rows`, `replay`, `trainee`, `wns`, `confirm`, `report`. `report` prints the tables of sections 1-4.
- [scripts/ld.sh](../scripts/ld.sh) is the confirmatory chain.
- Closed-loop paths come from `experiments/route_follow/scripts/rf_follow.py`, run unchanged on the stored held 460 @18 reads.

Box outputs are in `$DATA_DIR/runs/practice_nuplan/lat_diag/`: rows.parquet, follow/, follow1k/, replay.parquet, trainee_<tag>.parquet,
replay_ld.parquet. Pool jobs, all priority 11:
- replay 1019-110431-13b3
- trainee 1019-111330-addf / 1019-111331-02ff / 1019-113422-1b7f
- supplement 2: 1019-121846-e2f5 .. 1019-121849-657d

__ANSWER__

## 0. The new zeros (held 460 @18, NTVX against NTV0)

- **Count.** 17 new zeros and 4 fixed. 14 of the 17 new ones are lateral: 11 offroad, 3 corridor. The other 3 are AF-front.
- **Not seed noise.** The same command with only the seed changed churns far less:

  | pair | new / fixed zeros |
  |---|--:|
  | NTV0 s1 vs s0 | 7 / 6 |
  | JTN s1 vs s0 | 6 / 4 |
  | RSG s1 vs s0 | 3 / 3 |
  | NTVX vs RSG or JTN-s0 | 16-17 new |

- **Direction.** In 12 of the 14 lateral new zeros, the NTVX ego at the zero event is right of where NTV0 was on the same scene and
  decision (one is equal, one is left).
  - 5 are in turns of 73-139 deg. On a left turn, right means wide. rf_follow left-turn completion is 18.8 pp lower than NTV0
    [-40.0, +12.5]; right turns are +12.5 pp.
  - 9 are on non-turn roads.
- **Where.** 10 of the 17 are in two logs, 2021.10.06.08.16.17_veh-52 (5) and 2021.09.16.14.39.34_veh-42 (5). These are the two logs
  with the most decisions and among the largest drifts in section 1. 5 of the 17 are also zeros for RSG or JTN-s0, so these are scenes
  with little margin to begin with.

## 1. Closed loop: NTVX drifts right

The quantity is the signed lateral offset of the 2 Hz anchors to AlpaSim's route (left +). Each pair compares the two reads on the same
(scene, decision), before either read's first zero event. CIs come from a log-clustered bootstrap.

| A - B | non-turn, k 5-10 | non-turn, k 11-17 | turn scenes, k 11-17 |
|---|--:|--:|--:|
| NTVX - NTV0 | -0.041 [-0.050, -0.031] | **-0.108 [-0.131, -0.085]** | -0.363 [-0.629, -0.102] |
| NTV0-s1 - NTV0-s0 (seed) | -0.033 [-0.049, -0.019] | -0.049 [-0.077, -0.018] | +0.137 [+0.001, +0.287] |
| JTN-s1 - JTN-s0 (seed) | -0.007 [-0.015, +0.002] | -0.012 [-0.035, +0.010] | +0.065 [-0.067, +0.187] |
| RSG-s1 - RSG-s0 (seed) | +0.006 [+0.002, +0.010] | +0.011 [-0.010, +0.033] | -0.189 [-0.365, -0.036] |
| NTV0 - RSG | -0.014 [-0.038, +0.006] | -0.069 [-0.121, -0.021] | -0.696 [-0.959, -0.447] |
| NTV0-s1 - RSG | -0.048 [-0.070, -0.025] | -0.121 [-0.177, -0.059] | -0.542 [-0.692, -0.367] |
| JTN - RSG | +0.005 [-0.009, +0.020] | -0.010 [-0.056, +0.034] | -0.460 [-0.685, -0.230] |
| NTVX - RSG | -0.056 [-0.076, -0.036] | -0.176 [-0.227, -0.123] | -1.100 [-1.347, -0.889] |
| JTX1K - JTNC1K (1 000-step pilot) | +0.042 [+0.030, +0.053] | +0.078 [+0.047, +0.108] | -0.061 [-0.198, +0.106] |

- The offset grows with the decision index, so it is a drift. Every one of the 16 held logs has a negative NTVX - NTV0 mean, from -0.04 to
  -0.35 m. The cause is global, not one place.
- Seed noise on this statistic is 0.01-0.05 m. NTVX - NTV0 is twice the largest seed difference.
- **Why the arena did not see it.** Arena held windows last 5.5 s, with 8 executed decisions. On the native arena NTVX does not have more
  offroad zeros than NTV0: non-turn 14 against 15, turn 4 against 12. This count leaves out the 34 windows that every stored policy fails.

## 2. Open loop on the same inputs: a constant right bias of the plan

Each checkpoint's plans were computed on the 5 400 dumped decisions of the arena held windows. The dumps are the native-gap dumps of the
RSG and JTN-L18-s1 closed loops, so every checkpoint sees the same frames, ego row and route memory. Gate: JTN-L18-s1 on its own dumps
reproduces its served plans to 0.024 m (mean largest pose gap).
- Context comes from the dumped decision.
  - standing: v < 0.5 m/s.
  - slow lead: within 15 m, < 1 m/s.
  - turning: |heading change over 20 m| > 20 deg.
  - moving: everything else.
- Decisions with k >= 2.

| A - B | context | decisions | y(2 s) | y(4 s) |
|---|---|--:|--:|--:|
| NTVX - NTV0 | moving | 3567 | -0.015 [-0.019, -0.010] | **-0.032 [-0.047, -0.018]** |
| NTVX - NTV0 | turning | 632 | +0.010 [-0.006, +0.031] | +0.012 [-0.031, +0.067] |
| NTVX - NTV0 | all | 4320 | -0.011 [-0.015, -0.006] | -0.025 [-0.039, -0.011] |
| NTV0 - RSG | moving | 3567 | +0.001 [-0.005, +0.006] | +0.002 [-0.015, +0.021] |
| NTV0 - RSG | turning | 632 | -0.058 [-0.093, -0.027] | -0.155 [-0.274, -0.029] |
| JTN-s1 - RSG | moving | 3567 | -0.001 [-0.008, +0.004] | -0.005 [-0.030, +0.019] |
| JTX1K - JTNC1K | moving | 3567 | +0.016 [+0.013, +0.019] | +0.039 [+0.032, +0.049] |

**Re-centring is unchanged.** We fit y_plan(4 s) = b0 + b1 e + b2 psi_e on moving decisions with |e| < 1.5 m. Here e is the ego's signed
offset to the human's track and psi_e its heading error.

| | b0 | b1 |
|---|--:|--:|
| NTVX | +0.130 | -0.454 |
| NTV0 | +0.162 | -0.458 |
| RSG | +0.161 | -0.399 |
| JTN-s1 | +0.154 | -0.457 |

NTVX re-centres as strongly as NTV0. Its plan is shifted by a constant 3.2 cm to the right. With a gain of -0.45 per 4 s, that shift moves
the closed-loop resting offset by about 3.2 / 0.45 = 7 cm. This has the order of the measured -0.11 m.

## 3. The S rows the two draws see, and the push they give

Pool terms are evaluated as the loss sees them: the trainer's own setup (`lat_diag.py trainee` executes rm_train.py unedited up to its
dispatch, then calls its build()).
- Terms:
  - agent hinge: 0.3 m in the corridor, 0 m beside it.
  - road-and-lane raster: 0.3 m, with the drivable fallback.
  - These two are shape-only.
  - the three nr1_lat terms.
- Rows: the trainable S rows of JTX00S.
- **Push** = the descent direction of each term's gradient on the plan's poses, projected on each pose's left normal and multiplied by the
  term's loss weight (agent / road 3 x 10, lat 3 x 1). Shape-only means the agent and road hinges can only move the plan sideways or turn it.
- "static" = tab speed < 0.5 m/s at the row's t0.

**Trainee = RB0-F-s0** (the init of both arms):

| rows | n | share of NTVX's S draw | plan 4 s arc (med, m) | agent active | push agent | push road | push side | push, all terms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| new moving | 14 955 | 0.149 | 15.66 | 0.079 | +0.012 | +0.016 | +0.062 | +0.090 |
| **new static** | 16 732 | 0.166 | 3.89 | **0.326** | **-0.240** | -0.066 | -0.090 | **-0.397** |
| old moving | 65 751 | 0.653 | 19.01 | 0.073 | +0.006 | +0.029 | +0.034 | +0.045 |
| old static | 3 188 | 0.032 | 5.18 | 0.120 | -0.077 | -0.001 | +0.182 | +0.104 |
| NTV0's draw (old rows) | 68 939 | | | 0.075 | | | | **+0.048** |
| NTVX's draw (all) | 100 626 | | | 0.118 | | | | **-0.020** |

**Trainee = NTV0-L18-s0** (final):
- new static rows: -0.427 per row (agent -0.264). The other classes give -0.011 to +0.061.
- Draw nets: NTV0 draw -0.007, NTVX draw -0.073.
- Of the NTVX draw's -0.073, -0.071 comes from the new static rows.

**Trainee = NTVX-L18-s0** (final):
- new static rows: +0.016 per row. NTVX draw net: +0.041.
- NTVX has moved until the static rows' sideways push balances. The hinge is still active on 33 % of those rows, because the plan still
  creeps 4.0 m.

Reading:
- In the new static rows the human stands (median 4 s arc 0.02 m). The trainee's plan creeps 3.9-4.0 m forward into a stopped vehicle in
  its corridor. On the stored plans, 71 % of the new agent-active rows have an object that moves < 0.5 m in 4 s, and the ego is static in 68 % of them.
- The agent hinge fires on 33-34 % of these rows, against 7 % of the old moving rows. A shape-only hinge cannot shorten the plan. Its only
  gradient moves the plan sideways or turns it, and over these rows that gradient points right on average.
- The new static rows are 16.6 % of NTVX's S draws. They flip the per-row net sideways push of the S draw from left to right: +0.048
  against -0.020 at the init, -0.007 against -0.073 at NTV0's final weights.
- The trained NTVX sits where this push is balanced. The lateral offset bought nothing on those rows: the agent hinge is still active on
  33 %.
- This matches the training logs.

  | log scalar | NTVX | NTV0 |
  |---|--:|--:|
  | loss/act/pool_agent over training | 0.115 | 0.067 |
  | cos(imit, pool) gradient, steps 4-9.5 k | -0.02 | +0.02 / +0.10 |

## 4. The candidates

| candidate | test on stored data | verdict |
|---|---|---|
| (a) 49 % of the new windows are static and carry hinge loss without lateral signal | Static rows are 53 % of the new trainable rows and 16.6 % of NTVX's S draws. With the trainee's plans they carry the largest absolute push per row (1.55-1.66 against 0.93-1.41). Their signed push is right: -0.40 / -0.43 per row, against -0.01 to +0.10 for the other classes. They supply 97 % (RB0 weights) to all (NTV0 weights) of the change in net push between the two draws. | **the cause** ("dilute" undersells it: they push) |
| (b) between-frame starts: interpolated agent boxes and road rasters misaligned, wrong hinge gradients | Label gate on the human's own logged future at k <= 1: agent hinge 0.03 % new against 0.14 % old; road 8.3 % against 14.2 %. On the imitation target `fut`: agent 0.8 % against 2.0 %. The new rows' labels agree with the human at least as well as the old ones. A first pass showed 8-15 % agent activation: that was this script's bug, the log future's unwrapped yaw (2 pi), not a label error. | not supported |
| (c) road raster / drivable labels of the new rows | Same builder (on_rows geo, nuPlan map at the row pose); coverage 100 %; off the raster at t0 1.0 % against 1.5 %. Arena native offroad zeros: NTVX 14 / 4 (non-turn / turn) against NTV0 15 / 12. | not supported |
| (d) per-row passes of the old rows fell 31 % | Taking the old rows alone, their per-row net push is near 0 (old moving -0.011 at NTV0's weights). Fewer passes over them does not produce a right bias. The bias comes from what the new static rows add. | not the mechanism (dilution of corrections is at most second order) |
| (e) which zeros, which mechanism | Global right drift (16 / 16 logs). It is largest in long, narrow-margin logs (veh-52 -0.17 m, veh-42 -0.14 m), where 10 of the 17 new zeros are. 5 are wide on left turns. 5 of the 17 are shared with RSG or JTN. | right offset, not a turn or straight-road skill |

__CONFIRM__

## 6. d372: JTN's extra non-turn offroad on the native arena

This is a different thing from the NTVX regression.
- In this count, with the 34 every-policy windows left out, the native-arena non-turn offroad zeros are: JTN-s1 16, JTN-s0 14, NTV0 15,
  NTVX 14, RSG 8.
- JTN-s1's 14 extra windows over RSG are mostly bends below the turn stratum's threshold: |dyaw| median 18 deg, up to 60 deg; strata
  braking 6, other 5, launch 2.
  - The ego is slightly faster than RSG there (3.5 against 3.0 m/s).
  - It ends right of the human in 9 of 14.
  - 6 of the 14 are in log veh-52 00181_00574 and 4 in veh-42 00990_01609.
  - JTN-s0 and NTV0 fail mostly the same windows.
- JTN never saw the new rows. Open loop on the same inputs, JTN-s1 has no constant offset against RSG (moving -0.005 [-0.030, +0.019];
  turning -0.009). On held 460 it does not drift (-0.010).
- So d372's 21-against-10 is the JT recipe family's line through bends: a little faster and wider, as decision 360 found for turns. It is
  not a lateral bias and not the PN rows.
