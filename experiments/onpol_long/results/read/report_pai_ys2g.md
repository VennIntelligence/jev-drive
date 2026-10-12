> PAI transfer read of YS2G-L18 (Amendment 8's round-2b rows, gentle continuation), Y family, 281 scenes x 5 drivers (8 chunks each, all complete). Produced by
> `ol_report.py --family Y --arm S2G --ctl S --base-tags YR10m10-F-s0 --length pai` (same labels and clock as report_pai.md; `@pai`). Notes: "arm" = YS2G-L18,
> **"control" = YS-L18** (not YNONE; `--ctl S`), reference = YR10m10-F-s0; the "adopted" / "round gate" / "CL-1" lines are the script's mechanical output, not a verdict.
> The drift slope column is over simulator-step index. **4 scenes are dropped for every label alike (277 scenes, not 281)**: in `YS2G-L18-s1` chunk c003 (clipgt-33c8976b, -444fca69,
> -51aee21d, -52003cee) the driver stack failed to connect (gRPC UNAVAILABLE on 127.0.0.1:6400 at start, score 0, no metrics), an infrastructure failure, not driving; the
> script drops them from all labels as route-fold scenes. Raw data: `$DATA_DIR/runs/onpol_long/pai281/{ybase,ys0,ys1,ys2g0,ys2g1}/` on the main box; outputs in `$DATA_DIR/runs/onpol_long/read_pai_ys2g/`.
> Run on Tokyo 2026-10-12 20:05 - 2026-10-13 00:24 JST (562 scenes in 15522 s).

# ONPOL-LONG round 1, family Y (base YR10m10-F-s0): 277 scenes, length `pai`, 277 logs

Dropped for every label alike: 4 scenes without a rollout in at least one label (route generation failed: route folds back).

| | mean score | zeros | collision at fault / offroad / corridor / other zeros | at-fault zeros | GT distance (m) | drift slope k >= 4 (m / decision) | per seed: score / zeros / at-fault zeros / GT |
|:--|--:|--:|:--|--:|--:|--:|:--|
| arm YS2G-L18 (round 2b: round-2 rows, gentle, from YS-L18-s1 (Amendment 8)) | 0.5356 | 87.5 | 11.5 / 21.5 / 56.0 / 0.0 | 33.0 | 2.219 | 0.1163 | YS2G-L18-s0 0.5279 / 90 / 34 / 2.250 ; YS2G-L18-s1 0.5433 / 85 / 32 / 2.187 |
| control YS-L18 | 0.5268 | 90.5 | 12.5 / 23.0 / 58.0 / 0.0 | 35.5 | 2.305 | 0.1146 | YS-L18-s0 0.5167 / 94 / 38 / 2.352 ; YS-L18-s1 0.5368 / 87 / 33 / 2.258 |
| reference YR10m10-F-s0 | 0.4855 | 108.0 | 14.0 / 36.0 / 64.0 / 0.0 | 50.0 | 2.447 | 0.1303 | YR10m10-F-s0 0.4855 / 108 / 50 / 2.447 |
| reference YR10m10-F-s0 | 0.4855 | 108.0 | 14.0 / 36.0 / 64.0 / 0.0 | 50.0 | 2.447 | 0.1303 | YR10m10-F-s0 0.4855 / 108 / 50 / 2.447 |

Zeros = scenes scored 0; a zero can carry more than one kind (offroad and corridor together), `other` = none of the three. Paired per-scene seed-mean differences, log-clustered 95 % CI (10 000 draws); zero columns are totals over the scenes:

| a - b | mean score | zeros | at-fault zeros | collision / offroad / corridor / other zeros | GT distance (m) | fixed / broken |
|:--|:--|:--|:--|:--|:--|:--|
| arm - control | +0.0088 [-0.0069, +0.0248] | -3.0 [-8.5, +2.5] | -2.5 [-7.0, +1.5] | -1.0 / -1.5 / -2.0 / +0.0 | -0.086 [-0.152, -0.026] (277 scenes) | 1 / 1 |
| arm - reference YR10m10-F-s0 | +0.0501 [+0.0108, +0.0887] | -20.5 [-34.0, -7.0] | -17.0 [-28.5, -6.0] | -2.5 / -14.5 / -8.0 / +0.0 | -0.229 [-0.365, -0.095] (277 scenes) | 32 / 13 |
| control - reference YR10m10-F-s0 | +0.0413 [+0.0042, +0.0787] | -17.5 [-30.5, -5.0] | -14.5 [-26.0, -3.5] | -1.5 / -13.0 / -6.0 / +0.0 | -0.142 [-0.273, -0.012] (277 scenes) | 28 / 11 |

Null of fixed - broken: max |fixed - broken| of the single-seed comparisons YNONE-L18 s0 / s1 = 7.

**Adoption line (registered): ADOPTED** - fewer zeros than the control True (87.5 vs 90.5); at-fault zeros not above the control's (hard gate) True (33.0 vs 35.5); GT distance not larger True (2.219 vs 2.305 m).

Round gate (closed-loop.md; registered for round 2 in Amendment 5; with score for Amendment 6) against YR10m10-F-s0: PASSED - {"fewer_zeros": true, "atfault_not_above": true, "gt_not_larger": true, "passed": true, "score_not_lower": true} (zeros 87.5 vs 108.0, at-fault 33.0 vs 50.0, GT 2.219 vs 2.447).

CL-1 as written in closed-loop.md (reported beside it): not met - {"score_ci_low_gt0": false, "atfault_not_above_base_mean": true, "gt_lower_by_010": false, "fixed_minus_broken_over_null": false, "met": false}.
