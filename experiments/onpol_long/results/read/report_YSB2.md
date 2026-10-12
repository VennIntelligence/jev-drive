# ONPOL-LONG Amendment 14: arm B' (separate oncoming hinge channel): stopped at the label gate, no training

Label build `ol_dirraster.py onc` on pool `on1_L18YS` (34 788 rows, 0 errors; `onc_stats_L18YS.json`), job 1012-180644-4535, 1 min on 64 cores:

| registered label gate (before any training) | value | line | met |
|:--|--:|:--|:--|
| oncoming hinge active on the served plan (any footprint corner sample inside the oncoming region) | 0.207 | <= 0.10 | **no** |
| road hinge active on the served plan (pool raster, margin 0.3 m) | 0.122 (0.12185) | = 0.122 | yes |

By log turn: 0.187 on straight rows (< 15 deg), 0.245 at 15-45 deg, 0.335 at > 45 deg. The oncoming region is non-empty on every row (a two-way road
or an opposite roadblock is within the 80 m query almost everywhere). By the registered rule the four trainings (`YSO25` / `YSO50`, seeds 0 / 1) were
not started; their jobs and the 24 read chunks were cancelled / failed on the dependency.

Diagnostic after the gate (read-only, not a re-tuning of the line):

| | share of rows |
|:--|--:|
| minimum corner SDF on the served plan < 0 / < -0.25 / < -0.5 / < -1 / < -2 m | 0.207 / 0.156 / 0.119 / 0.072 / 0.018 |
| ego footprint at t0 already inside the oncoming region | 0.099 (0.48 of the active rows) |

The first entry is at the start of the plan for most active rows (median 0.2 s), and the active rows are only slightly further from the logged path
(|e0| median 0.29 m against 0.13 m). So about half of the activation is the car's own position at t0, close to the log: the region as defined
(baseline heading at the lane point nearest the row, whole roadblocks whose lanes all point away, minus intersections and connectors) marks part of
where the human drove as oncoming. zeros_mech's lane classification (the evaluator's lane polygons, centre-line yaw at the projection; d268 G-dir)
puts the log's own poses in an oncoming lane 0.4 % of the time, so the excess is in this label's geometry (most likely the roadblock-level removal
or lanes of the other direction overlapping the ego's lane polygons near junctions), not in the driving. The same geometry was arm B's removal set,
so arm B's raster (decision 274) carried it too. A rebuild that classifies by the evaluator's own lane polygons and centre-line yaw at the projection
(zeros_mech's rule) and drops the roadblock layer would be the fix; it needs a new amendment.

The arm B collect directory was moved out of `runs/zeros_mech/collect/` (see report_YSB.md); arm B' would have written its collect under
`runs/onpol_long/zm/` through `scripts/ol_zm.py`.


## Amendment 15: the oncoming region by zeros_mech's lane rule, arm B' and arm B redone: stopped again at the label gates, no training

`ol_zmonc.py build` (job 1012-181242-8bc4, 0 errors; `onc_stats_L18YS_zm.json`): every 0.5 m cell classified by `zm_read.Lanes`'s rule (evaluator lane polygons,
centre-line yaw at the projection, same < 60 deg, oncoming > 120 deg, no roadblock layer) at the row's t0 heading.

| gate | value | line | met |
|:--|--:|:--|:--|
| human: the log's own t0 rear axle in an oncoming lane (Lanes.cls itself) | 0.0048 (in a lane: 0.0048; same direction 0.9946) | <= 0.01 | yes |
| arm B': oncoming hinge active on the served plan | 0.227 | <= 0.10 | **no** |
| road hinge active (pool raster) | 0.122 | = 0.122 | yes |
| arm B redone: road hinge active with road minus oncoming cells | 0.374 | <= 0.16 | **no** |

All six trainings (YSO25 / YSO50 / YSBz x 2 seeds), the 36 read chunks and both report jobs were cancelled before starting.

Diagnostic after the gates (read-only): the human's own footprint, not its rear axle, enters the corrected region often. Its box at t0 has a corner in
the oncoming region on 0.128 of rows (0.064 by more than 0.5 m), and its logged 4 s path does somewhere on 0.252 (0.164 by more than 0.5 m). By log turn
the served-plan activity is 0.19 straight, 0.30 at 15-45 deg, 0.47 on turns. The rule is right for a point (the human's rear axle is oncoming 0.5 % of
the time, as in d268), but a car-sized footprint driven normally overlaps oncoming lane polygons at lane edges and in junctions, so a footprint hinge on
this region would penalise the human's own driving on a fifth of the rows. A usable oncoming label would have to be point-based (rear axle or box centre,
as the zeros_mech reading itself is) or tolerate an overlap depth; that is a new design, not a gate tweak.

For decision 274 (the void arm B): 84 % of the cells arm B removed are oncoming by this rule (37 982 212 removed cells), and the corrected directional
raster is *more* active on the served plans than arm B's (0.374 against 0.285). So arm B's high road-hinge activity, which went with its worse zeros
and GT, was not mainly mask error: removing oncoming lanes from a footprint-checked road raster fires on normal driving under either geometry.


## Amendment 16: point-based oncoming hinge variants (label study, no training)

`scripts/ol_onclab.py` (job 1012-181903-051c, CPU, < 2 min; `onclab_L18YS.json`), on the zeros_mech-rule region of Amendment 15, 34 788 rows. Active = the variant enters
the oncoming region at any of the 41 dense steps (0-4 s); the human path = the log's 9 poses at 0-4 s, linear to the same steps; heading of the region = the row's t0 heading.

| variant | served plan | human path | served straight / turn (>= 45 deg) | human straight / turn | usable (served <= 0.10, human <= 0.03) |
|:--|--:|--:|:--|:--|:--|
| (a) rear-axle point | 0.072 | 0.021 | 0.042 / 0.262 | 0.009 / 0.120 | yes |
| (b) box centre | 0.075 | 0.021 | 0.045 / 0.266 | 0.009 / 0.127 | yes |
| (c) footprint, 0.5 m allowed depth | 0.130 | 0.054 | 0.096 / 0.337 | 0.027 / 0.222 | no |
| (d) footprint, 1.0 m allowed depth | 0.083 | 0.023 | 0.055 / 0.253 | 0.011 / 0.120 | yes |

d268's across-traffic early-inside zero events of YS-L18-s1 (11 with a decision two before the event; region computed on the held scene with the same rule):
the served plan two decisions before the event is active in 9 / 11 for (a), 8 / 11 for (b) and (d) (c687799d differs: rear axle only). The two never active:
330166de (corridor) and 8c9e3fd9 (offroad).

Caveat for the usable variants: on turning rows the human's own path is active on 0.12 (straight 0.01), so on turns the hinge would still charge some
human-like plans (the region is classified at the t0 heading for the whole 4 s). No training is queued; the variant is the user's choice.

## Amendment 17: local heading per plan point (label test) -> variant (a) with the t0 heading

`scripts/ol_onclab2.py` (job 1012-182136-9ac0, exact zeros_mech rule per point, no raster; `onclab2_L18YS.json`). Rear-axle point, active rows:

| | all | straight | turn (>= 45 deg) |
|:--|--:|--:|--:|
| served plan, t0 heading | 0.068 | 0.039 | 0.257 |
| served plan, local heading | 0.070 | 0.040 | 0.265 |
| human path, t0 heading | 0.019 | 0.008 | 0.112 |
| human path, local heading | 0.017 | 0.008 | 0.084 |

Across-traffic early-inside events active two decisions ahead: 9 / 11 with either heading. The written rule (human on turns < 0.05 and >= 8 / 11) is
not met by the local heading (0.084), so the trainings use variant (a) with the t0 heading on the Amendment 15 raster (`OL_ONC_MODE=point`). The exact
per-point t0 numbers reproduce the raster version of Amendment 16 (served 0.068 against 0.072, human 0.019 against 0.021).

## Amendment 17 read: arm B' with variant (a) (rear-axle oncoming hinge, t0 heading), held 452 / 18 decisions, cards 1-5

Jobs: trainings 1012-182636-3563 / -182638-745f (YSO25 s0 / s1), 1012-182639-01d8 / -182641-3a69 (YSO50), 24 read chunks, report 1012-182642-d4cd (a first submission failed at setup on a list-for-array bug in `ol_onc.load`, fixed and resubmitted; no training step ran). Timings: training 6.7 min per run, read 8.1 min per 77-scene chunk on average (max 10.0), report 1 min. zm_read collect `runs/onpol_long/zm/zeros_mech/collect/20261012-185754` (through `ol_zm.py`).

| line (seed mean) | YSO25 (lam 2.5) | YSO50 (lam 5) | YS2G | YS-L18 | base |
|:--|--:|--:|--:|--:|--:|
| 1. early-inside zeros, across-traffic turns < 5.5 | 2.5 ok | 2 ok | 5.5 | 10.5 | 6 |
| 2. late-outside zeros, kerb-side turns <= 7.5 | 7.5 ok | 10.5 no | 7.5 | 6 | 17 |
| 3. oncoming 2 s before (corridor + collision zeros) < 0.17 | 0.171 (7 / 41) no | 0.122 (6 / 49) ok | 0.226 | 0.305 | 0.171 |
| 4. zeros <= 41.5 | 37.5 ok | 44 no | 41.5 | 44 | 74 |
| 5. GT distance <= 1.276 m (selection) | 1.306 no | 1.349 no | 1.276 | 1.357 | 1.495 |
| mean score | 0.8584 | 0.8469 | 0.8474 | 0.8376 | 0.7866 |
| at-fault zeros | 22.5 | 24.5 | 22.0 | 23.5 | 41 |
| separate: straight corridor zeros past the log end / all | 2.5 / 3.5 | 2.5 / 4 | 4.5 / 6 | 4.5 / 6 | 2 / 7 |

Paired against YS2G (seed means, log-clustered 95 % CI): YSO25 +0.011 [-0.007, +0.026], zeros -4.0 [-10.9, +4.4], at-fault +0.5, GT +0.030 [-0.022, +0.079] m; YSO50 -0.001 [-0.021, +0.017], zeros +2.5, at-fault +2.5, GT +0.074 [-0.014, +0.160] m. Against YS-L18: YSO25 +0.021 [+0.000, +0.039], GT -0.051 [-0.137, +0.033].

**Verdict: neither weight adopted** (line 5, the selection quantity, fails for both). YSO25 meets lines 1, 2 and 4 and misses line 3 by one event (7 / 41 = 0.1707 against < 0.17, the base level) and line 5 by +0.030 m (CI includes 0); it has the highest mean score of this topic (0.858) and the fewest zeros (37.5). YSO50 meets 1 and 3, fails 2, 4 and 5: the stronger weight pushes plans to the kerb side again (offroad zeros 19.5, kerb-side late-outside 10.5).

---

# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSO25-L18 (SO25) | 0.8584 | 37.5 | 5.5 / 17.0 / 16.0 / 0.0 | 22.5 | 1.306 | 0.0921 | YSO25-L18-s0 0.8584 / 37 / 22 / 1.318 ; YSO25-L18-s1 0.8585 / 38 / 23 / 1.293 |
| control YS2G-L18 | 0.8474 | 41.5 | 7.0 / 15.0 / 19.5 / 0.0 | 22.0 | 1.276 | 0.0903 | YS2G-L18-s0 0.8434 / 43 / 24 / 1.272 ; YS2G-L18-s1 0.8514 / 40 / 20 / 1.279 |
| reference YS-L18-s0 | 0.8345 | 45.0 | 9.0 / 14.0 / 23.0 / 0.0 | 23.0 | 1.367 | 0.0962 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 |
| reference YS-L18-s0, YS-L18-s1 | 0.8376 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0960 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0110 [-0.0069, +0.0255] | -4.0 [-10.9, +4.4] | +0.5 [-6.0, +7.4] | -1.5 / +2.0 / -3.5 / +0.0 | +0.030 [-0.022, +0.079] (452 scenes) | 6 / 3 |
| arm - reference YS-L18-s0+YS-L18-s1 | +0.0208 [+0.0003, +0.0386] | -6.5 [-14.5, +2.6] | -1.0 [-8.7, +7.1] | -3.5 / +2.5 / -6.0 / +0.0 | -0.051 [-0.137, +0.033] (452 scenes) | 9 / 7 |
| control - reference YS-L18-s0+YS-L18-s1 | +0.0098 [-0.0003, +0.0205] | -2.5 [-7.3, +1.8] | -1.5 [-6.1, +3.3] | -2.0 / +0.5 / -2.5 / +0.0 | -0.081 [-0.141, -0.024] (452 scenes) | 5 / 2 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 3.

**Adoption line (registered): NOT adopted** - fewer zeros than the control True (37.5 vs 41.5); at-fault zeros not above the control's (hard gate) False (22.5 vs 22.0); GT distance not larger False (1.306 vs 1.276 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s0+YS-L18-s1: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 37.5 vs 44.0, at-fault 22.5 vs 23.5, GT 1.306 vs 1.357).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
# ONPOL-LONG round 1, family Y (base YS-L18-s0): 452 scenes, length `18`, 39 logs

Dropped for every label alike: 8 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YSO50-L18 (SO50) | 0.8469 | 44.0 | 5.0 / 19.5 / 21.0 / 0.0 | 24.5 | 1.349 | 0.1058 | YSO50-L18-s0 0.8385 / 48 / 27 / 1.338 ; YSO50-L18-s1 0.8553 / 40 / 22 / 1.361 |
| control YS2G-L18 | 0.8474 | 41.5 | 7.0 / 15.0 / 19.5 / 0.0 | 22.0 | 1.276 | 0.0903 | YS2G-L18-s0 0.8434 / 43 / 24 / 1.272 ; YS2G-L18-s1 0.8514 / 40 / 20 / 1.279 |
| reference YS-L18-s0 | 0.8345 | 45.0 | 9.0 / 14.0 / 23.0 / 0.0 | 23.0 | 1.367 | 0.0962 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 |
| reference YS-L18-s0, YS-L18-s1 | 0.8376 | 44.0 | 9.0 / 14.5 / 22.0 / 0.0 | 23.5 | 1.357 | 0.0960 | YS-L18-s0 0.8345 / 45 / 23 / 1.367 ; YS-L18-s1 0.8407 / 43 / 24 / 1.346 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | -0.0005 [-0.0214, +0.0167] | +2.5 [-5.8, +12.5] | +2.5 [-5.0, +10.4] | -2.0 / +4.5 / +1.5 / +0.0 | +0.074 [-0.014, +0.160] (452 scenes) | 5 / 9 |
| arm - reference YS-L18-s0+YS-L18-s1 | +0.0093 [-0.0148, +0.0308] | +0.0 [-9.7, +10.9] | +1.0 [-6.2, +8.6] | -4.0 / +5.0 / -1.0 / +0.0 | -0.007 [-0.127, +0.111] (452 scenes) | 11 / 11 |
| control - reference YS-L18-s0+YS-L18-s1 | +0.0098 [-0.0003, +0.0205] | -2.5 [-7.3, +1.8] | -1.5 [-6.1, +3.3] | -2.0 / +0.5 / -2.5 / +0.0 | -0.081 [-0.141, -0.024] (452 scenes) | 5 / 2 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 and YS-L18 s0 / s1 = 3.

**Adoption line (registered): NOT adopted** - fewer zeros than the control False (44.0 vs 41.5); at-fault zeros not above the control's (hard gate) False (24.5 vs 22.0); GT distance not larger False (1.349 vs 1.276 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YS-L18-s0+YS-L18-s1: not passed - {"fewer_zeros": false, "atfault_not_above": false, "gt_not_larger": true, "passed": false, "score_not_lower": true} (zeros 44.0 vs 44.0, at-fault 24.5 vs 23.5, GT 1.349 vs 1.357).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": false, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
