# LATERAL-DIAG: why NTVX is worse laterally than NTV0 on held 460 (2026-10-19)

Diagnosis on stored data, no training. Code: [scripts/lat_diag.py](../scripts/lat_diag.py) (`rows`, `replay`, `trainee`, `report`; the
`report` subcommand prints sections 1-3 below). Closed-loop paths: `experiments/route_follow/scripts/rf_follow.py` unchanged on the stored
held 460 @18 reads. Box outputs: `$DATA_DIR/runs/practice_nuplan/lat_diag/` (rows.parquet, follow/, follow1k/, replay.parquet,
trainee_<tag>.parquet). Pool jobs: replay 1019-110431-13b3; trainee 1019-111330-addf (NTV0), 1019-111331-02ff (NTVX), 1019-113422-1b7f (RB0-F).

__ANSWER__

## 0. The zeros themselves (held 460 @18)

- NTVX has 17 new zeros against NTV0 and fixes 4 (net +13). 14 of the 17 are lateral (11 offroad, 3 corridor) and 3 are AF-front.
- This is not seed noise. The same command with another seed changes few zeros: NTV0 s1 against s0 has 7 new / 6 fixed, JTN s1 against
  s0 has 6 / 4, RSG s1 against s0 has 3 / 3. NTVX against RSG or JTN-s0 has 16-17 new.
- Direction: at its zero event, the NTVX ego is right of NTV0's position on the same scene and decision in 12 of the 14 lateral new
  zeros (one equal, one left).
  - 5 of the 14 are in turns of 73-139 deg, where right of NTV0 means wide on a left turn. rf_follow: left-turn completion is 18.8 pp
    lower than NTV0, [-40.0, +12.5]; right turns are +12.5 pp.
  - 9 are on non-turn roads.
- Place: 10 of the 17 are in two logs, 2021.10.06.08.16.17_veh-52 (5) and 2021.09.16.14.39.34_veh-42 (5). 5 of the 17 are also
  zeros for RSG or JTN-s0, so these are scenes where the margin is already small.

## 1. Closed loop: NTVX drifts right, by a few centimetres per second

Signed lateral offset of the 2 Hz anchors to AlpaSim's route (left +). Pairs are taken on the same (scene, decision), before either
read's first zero event. CI: log-clustered bootstrap.

__DRIFT__

- NTVX sits right of NTV0 by 0.04 m over decisions 5-10 and by 0.11 m over decisions 11-17. Every one of the 16 held logs has a negative
  mean (-0.04 to -0.35 m), so this is a global offset, not one place.
- Seed noise on the same statistic is 0.01-0.05 m (NTV0 s1 against s0: -0.049).
- Arena held windows last only 5.5 s (8 executed decisions). The offset is still small there, which is why the arena did not see the
  problem. NTVX has fewer native offroad zeros than NTV0 on the arena (non-turn 14 against 15; turn 4 against 12; 34 windows that every
  stored policy fails are left out).

## 2. Open loop on the same inputs: a constant right bias of the plan, centring unchanged

Plans of each checkpoint on the 5 400 dumped decisions of the arena held windows (native-gap dumps of the RSG and JTN-L18-s1 closed loops:
same frames, ego row and route memory for every checkpoint). Gate: JTN-L18-s1 on its own dumps reproduces its served plans to 0.024 m
(mean largest pose gap, smoke).

__REPLAY__

- On moving decisions, NTVX plans 1.5 cm further right than NTV0 at 2 s and 3.2 cm further right at 4 s. Both CIs exclude 0.
- The re-centring gain is the same for both (b1 -0.454 against -0.458). Only the intercept moves (+0.130 against +0.162 m). A constant
  plan offset of -3.2 cm with a gain of -0.45 per 4 s moves the closed-loop equilibrium by about -0.07 m. That has the order of the
  -0.11 m measured in section 1.
- NTV0 against RSG on moving decisions: +0.002 (no constant offset). JTN-s1 against RSG: -0.005.
- The 1 000-step pilot pair (JTX1K against JTNC1K: RB0 init, with trainval rows) moved the plan the other way, 3.9 cm LEFT at 4 s (closed
  loop +0.078 m). Adding the new rows shifts the plan sideways in both recipes, but the sign is not fixed by the rows alone.

## 3. The S rows the two draws see

Stored own plan of the rolling policy (`plan`); the imitation target `fut` (the human's path blended in from the ego's offset); and, at
k <= 1, the log's own future `log` (label gate). Hinges are set up as the trainer sets them (agent 0.3 m in the corridor, 0 m beside it;
road-and-lane raster 0.3 m, with the drivable fallback for rows off it at t0; shape-only). Push = the descent direction of the term's
gradient on the plan's poses, projected on each pose's left normal, times ho_w x lam (30). Rows: trainable S rows of JTX00S.

__ROWS__

__TRAINEE__

## 4. The candidates

__CANDIDATES__

## 5. d372: JTN's extra non-turn offroad on the native arena

__JTN__
