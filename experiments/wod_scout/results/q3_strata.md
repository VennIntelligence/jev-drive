# Q3. Night, pedestrians and the clusters that lose on test: for WLG on val the loss there is speed, not path, and night is no longer a geometry gap

Written 2026-10-10, lane WODSCOUT. Same run, conventions and limits as [q2_path_budget.md](q2_path_budget.md) (`scripts/ws_val.py`; 479 val rater
frames, cluster-mean RFS, points add to the 1.400 gap, paired bootstrap over sequences, B 4 000). Night = sequence front-camera luma < 50, day >= 120
(the labels of decision 131, `experiments/leaderboard_audit/results/night_gap/seq_lum.csv`). Tables: [q3_strata/](q3_strata/) and
[q2_path_budget/budget.md](q2_path_budget/budget.md). Pre-registration: [plans/2026-10-10-wodscout-prereg.md](../plans/2026-10-10-wodscout-prereg.md).

## Answer

| stratum | n | RFS top / log / WLG / shipped | points WLG loses [95 % CI] (share of 1.400) | speed only (O1) | path only (O2) | label | WLG - shipped |
|:--|--:|:--|:--|:--|:--|:--|:--|
| night (luma < 50) | 133 | 9.665 / 8.328 / 7.976 / 7.610 | 0.386 [0.272, 0.515] (28 %) | 0.167 [0.071, 0.274] | 0.065 [0.019, 0.122] | speed | +0.366 [+0.087, +0.766] |
| day (luma >= 120) | 325 | 9.564 / 8.012 / 8.201 / 8.100 | 0.913 [0.733, 1.101] (65 %) | 0.432 [0.289, 0.588] | 0.215 [0.099, 0.349] | speed | +0.101 [-0.060, +0.275] |
| Pedestrian | 52 | 9.442 / 8.306 / 8.334 / 8.234 | 0.111 [0.063, 0.165] (8 %) | 0.055 [0.008, 0.106] | 0.018 [-0.014, 0.053] | speed | +0.100 [-0.194, +0.448] |
| Cyclist | 71 | 9.831 / 7.559 / 8.439 / 8.228 | 0.139 [0.103, 0.177] (10 %) | 0.063 [0.022, 0.106] | 0.002 [-0.018, 0.018] | speed | +0.212 [-0.157, +0.617] |
| Others | 22 | 9.500 / 8.722 / 8.127 / 7.855 | 0.137 [0.069, 0.208] (10 %) | 0.089 [0.033, 0.155] | 0.042 [0.004, 0.085] | speed | +0.272 [-0.434, +1.063] |
| Single-Lane Maneuvers | 38 | 9.447 / 8.825 / 8.480 / 8.110 | 0.097 [0.047, 0.156] (7 %) | 0.042 [-0.002, 0.089] | 0.042 [0.005, 0.092] | undetermined | +0.370 [+0.053, +0.812] |
| Interections | 116 | 9.595 / 8.367 / 8.085 / 7.743 | 0.151 [0.115, 0.187] (11 %) | 0.065 [0.033, 0.097] | 0.014 [-0.006, 0.034] | speed | +0.342 [+0.129, +0.574] |

Every cluster is one tenth of the score, so a cluster's points are (its gap) / 10: the clusters differ little (0.097 to 0.159 each; all ten in
`q2_path_budget/budget.md`). Stratum RFS columns are cluster means within the stratum and do not add; the points columns do.

1. **Night.** 133 frames (28 %) hold 28 % of the gap: night is not over-represented in WLG's loss. Night minus day of the per-frame gap is +0.29
   [-0.10, +0.69] unmatched and +0.23 [-0.15, +0.62] matched on the speed bin; neither excludes 0. The label is speed (0.167 against 0.065 path only).
   On geometry the night gap of decision 131 is gone for WLG: ADE@3s against the log (1 374 night / day frames of rater + extra) is 0.50 m at night and
   0.60 m by day, night - day -0.10 [-0.20, +0.01], while shipped on the same frames is 1.31 against 0.92, +0.39 [+0.19, +0.60] (decision 131's
   +0.43 matched). WLG - shipped is larger at night (+0.366) than by day (+0.101); the difference of the two is +0.32 [+0.01, +0.63], a weak read.
   The log itself scores higher at night than by day (+0.32 [-0.11, +0.71]): WLG is 0.35 below the log at night and 0.19 above it by day.
2. **Pedestrian.** 0.111 points, label speed; path only +0.018 with a CI containing 0. WLG equals the log there (8.334 against 8.306).
3. **Others, Single-Lane, Intersection** (the three other clusters where WLG is furthest behind the leader on test, decision 180): Others and
   Intersection are speed; Single-Lane is the one cluster where speed only and path only are equal (0.042 each, both CIs touching 0, undetermined).
   Within Intersection the 11 turn-intent frames are path (Q2); the 105 straight ones are speed with a path-only gain of -0.002.
4. **Perception-side reading.** Nothing here points at a perception gap that a path or detection fix would close: where WLG loses in these strata,
   re-timing its own path to the top-rated speed profile recovers two to thirty times what the top-rated path at its own speed does (except
   Single-Lane). What is being missed is how far to go given the scene, which is the known bottleneck (decisions 164, 168, 218 amendment).

About 20 strata and three contrasts each were read; CIs with a bound near 0 are weak.

## Val against test, per cluster (descriptive; test has no CI)

Val score of the submitted trajectory (mean of the two seeds) with its by-sequence bootstrap CI, against the official test cluster scores of
decision 180. Table: [q3_strata/val_vs_test.md](q3_strata/val_vs_test.md).

| cluster | n val | val RFS [95 % CI] | test RFS | test - val | test inside the val CI | test: WLG - leader |
|:--|--:|:--|--:|--:|:--|--:|
| Cyclist | 71 | 8.455 [8.067, 8.834] | 8.007 | -0.448 | **no** | +0.040 |
| Others | 22 | 8.110 [7.249, 8.932] | 7.616 | -0.494 | yes | -0.369 |
| Pedestrian | 52 | 8.322 [7.779, 8.827] | 7.957 | -0.365 | yes | -0.270 |
| Single-Lane Maneuvers | 38 | 8.500 [7.871, 9.106] | 8.377 | -0.123 | yes | -0.241 |
| Interections | 116 | 8.086 [7.708, 8.455] | 8.073 | -0.013 | yes | -0.222 |
| Multi-Lane Maneuvers | 42 | 8.007 [7.396, 8.583] | 8.079 | +0.072 | yes | +0.244 |
| Cut_ins | 20 | 8.268 [7.458, 9.018] | 8.353 | +0.085 | yes | +0.116 |
| Foreign Object Debris | 78 | 8.082 [7.603, 8.519] | 8.265 | +0.183 | yes | |
| Construction | 15 | 8.329 [7.171, 9.335] | 8.725 | +0.396 | yes | |
| Special Vehicles | 25 | 7.618 [6.751, 8.439] | 8.289 | +0.671 | yes | -0.155 |
| mean of the 10 | 479 | 8.178 [7.960, 8.397] | 8.174 | -0.004 | yes | |

The val CIs are 0.7 to 2.2 wide, so "inside" says little: only Cyclist falls outside (test 0.45 lower). Others and Pedestrian drop by 0.4-0.5 from
val to test and are also where WLG trails the leader most, but with 22 and 52 val frames neither drop is distinguishable from sampling. The equality
of the two means (8.178 against 8.174) is the sum of per-cluster moves of up to +/- 0.67 that cancel.

## Limits

- Night labels are luma thresholds on sequence means; dusk (luma 50-120, 21 frames) is in neither group.
- Cluster tags are Waymo's and each frame has one; "pedestrians" means the Pedestrian cluster, not "a pedestrian is visible".
- Swaps are oracle reads in sample (see Q2); ADE is against the log, which the model was trained to imitate.
