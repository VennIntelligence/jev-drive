> PAI transfer read (Amendment 1), Y family, 281 scenes x 5 drivers (8 chunks each, all complete, 281 rollouts each, 0 dropped). Produced by
> `ol_report.py --family Y --arm S --base-tags YR10m10-F-s0 --length pai`; PAI labels are read on the run's own clock, every simulator step
> (`long_read.py`, `@pai`). Notes: the "drift slope" column is over simulator-step index, not decisions, and is not comparable to the held-460 slopes;
> "adopted" / "round gate" / "CL-1" lines below are the script's mechanical output with the held-460 wording, not a verdict;
> "arm" = YS-L18, "control" = YNONE-L18. Raw data: `$DATA_DIR/runs/onpol_long/pai281/{ybase,ynone0,ynone1,ys0,ys1}/` on the main box.

# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 281 scenes, length `pai`, 281 logs

Dropped for every label alike: 0 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS-L18 (S hinges only on the same rows (Amendment 3)) | 0.5264 | 92.5 | 12.5 / 24.0 / 59.0 / 0.0 | 36.5 | 2.319 | 0.1158 | YS-L18-s0 0.5165 / 96 / 39 / 2.366 ; YS-L18-s1 0.5363 / 89 / 34 / 2.271 |
| control YNONE-L18 (same continuation, no rows) | 0.4292 | 131.0 | 20.0 / 49.0 / 66.5 / 0.0 | 68.5 | 2.564 | 0.1372 | YNONE-L18-s0 0.4230 / 133 / 67 / 2.569 ; YNONE-L18-s1 0.4355 / 129 / 70 / 2.559 |
| reference YR10m10-F-s0 | 0.4857 | 110.0 | 14.0 / 37.0 / 65.0 / 0.0 | 51.0 | 2.468 | 0.1316 | YR10m10-F-s0 0.4857 / 110 / 51 / 2.468 |
| reference YR10m10-F-s0 | 0.4857 | 110.0 | 14.0 / 37.0 / 65.0 / 0.0 | 51.0 | 2.468 | 0.1316 | YR10m10-F-s0 0.4857 / 110 / 51 / 2.468 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0971 [+0.0656, +0.1298] | -38.5 [-51.0, -26.5] | -32.0 [-43.5, -20.0] | -7.5 / -25.0 / -7.5 / +0.0 | -0.245 [-0.375, -0.118] (281 scenes) | 31 / 4 |
| arm - reference YR10m10-F-s0 | +0.0407 [+0.0050, +0.0770] | -17.5 [-30.0, -5.0] | -14.5 [-26.0, -3.5] | -1.5 / -13.0 / -6.0 / +0.0 | -0.149 [-0.278, -0.018] (281 scenes) | 28 / 11 |
| control - reference YR10m10-F-s0 | -0.0564 [-0.0926, -0.0199] | +21.0 [+8.5, +34.0] | +17.5 [+5.0, +30.0] | +6.0 / +12.0 / +1.5 / +0.0 | +0.096 [-0.039, +0.232] (281 scenes) | 11 / 25 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 = 4.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (92.5 vs 131.0); at-fault zeros not above the control's (hard gate) True (36.5 vs 68.5); GT distance not larger True (2.319 vs 2.564 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 92.5 vs 110.0, at-fault 36.5 vs 51.0, GT 2.319 vs 2.468).

CL-1 as written in closed-loop.md (reported beside it): met - {"score_ci_low_gt0": true, "atfault_not_above_base_mean": true, "gt_lower_by_010": true, "fixed_minus_broken_over_null": true, "met": true}.
