# LON-DIAG reading 2: the slow scenes of RSG-L18-s0 on PAI 281

Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md` (reading 2). Script: `rf_londiag.py table` (per-decision table, Tokyo `/data/runs/route_follow/lon_diag/decisions.parquet`, not in git) and `rf_londiag.py slow`. Definitions in the script header. 55693 decisions (10 Hz) of 281 rollouts; 42129 before the first zero-score event / contact (`valid`); 12455 valid free decisions at v_f >= 1 m/s.

## R2a handoff (k = 0-2, the fed history is still the log's)

| v_log bin (m/s) | decisions | scenes | median v_p / v_f (k 0-2) | median at k = 0 |
|:--|--:|--:|--:|--:|
| <8 | 233 | 81 | 1.010 | 1.016 |
| 8-12 | 168 | 58 | 0.972 | 0.947 |
| 12-16 | 110 | 37 | 0.945 | 0.934 |
| >=16 | 197 | 66 | 0.867 | 0.853 |

Line (>= 16 bin < 0.85 and < 8 bin >= 0.95 -> distribution ceiling; all bins >= 0.95 -> not short at the handoff): **neither line**.

## R2b onset of the slow 73 (`low_slow_free`)

Onset = first valid decision (v_f >= 1 m/s) with v_p < 0.9 v_f whose next 3 decisions (0.3 s) stay below 0.95 v_f. Found in 67 / 73 scenes; onset time median 0.0 s (IQR 0.0-2.2), 45 at k <= 2 (the handoff), 59 before 3 s. At the onset v_f median 16.3 m/s, v_log 16.2 m/s, v_p / v_f 0.86. RR uses the 6 scenes with >= 1 s of control decisions in [onset - 6, onset - 3] s: per scene the fraction of decisions with the feature in [onset - 1 s, onset] against the fraction in the control window; RR = ratio of the scene means; scene bootstrap B 10 000, seed 0.

| feature | onset window | control window | RR [95 % CI] |
|:--|--:|--:|:--|
| vehicle in corridor <= 40 m | 0.000 | 0.091 | 0.00 [0.00, 0.00] |
| stop line <= 60 m | 0.667 | 0.667 | 1.00 [1.00, 1.00] |
| route turn >= 30 deg within 60 m | 0.136 | 0.000 | undefined (control fraction 0) |
| in junction | 0.030 | 0.167 | 0.18 [0.00, 0.36] |
| (extra) serve lead filter sees a lead | 0.106 | 0.167 | 0.64 [0.64, 0.64] |

Handoff onsets (k <= 2, 45 scenes) against later onsets (22), medians: v_log at the onset 21.1 vs 10.4 m/s, v_f 20.4 vs 9.7 m/s, v_p / v_f 0.85 vs 0.89; later onsets at t = 2.5 s (median). Per-scene rows: `onsets.csv`.

Line (any registered feature RR >= 2 and CI low > 1 -> trigger; all CIs contain 1 -> no trigger): **neither line (some CI excludes 1 without RR >= 2)**.

## R2c fixed point (valid free decisions, v_f >= 1 m/s)

| set | decisions | scenes | slope g [95 % CI] | intercept | v* = a / (1 - g) | v* from 1 m/s bins (mean v_p crosses v_f downward) |
|:--|--:|--:|:--|--:|--:|:--|
| pooled | 12455 | 157 | 0.936 [0.912, 0.956] | +0.35 | 5.5 | 3.1 |
| exam | 2772 | 39 | 0.886 [0.829, 0.927] | +0.87 | 7.6 | 3.2 |
| non-exam | 9683 | 118 | 0.956 [0.933, 0.973] | +0.13 | 3.0 | 3.0 |

Line (g < 1 and v* <= 10 m/s -> a comfort speed the ratchet decays to; g >= 0.95 -> constant offset): pooled **comfort speed: g < 1 and v* <= 10 m/s**. Bin table (pooled):

| v_f bin | n | mean v_f | mean v_p | v_p - v_f |
|--:|--:|--:|--:|--:|
| 1 | 181 | 1.55 | 1.97 | +0.42 |
| 2 | 420 | 2.58 | 2.68 | +0.11 |
| 3 | 649 | 3.51 | 3.41 | -0.10 |
| 4 | 560 | 4.43 | 4.23 | -0.20 |
| 5 | 324 | 5.52 | 5.02 | -0.49 |
| 6 | 299 | 6.53 | 5.93 | -0.60 |
| 7 | 451 | 7.51 | 7.13 | -0.37 |
| 8 | 386 | 8.46 | 8.02 | -0.44 |
| 9 | 484 | 9.43 | 9.04 | -0.39 |
| 10 | 499 | 10.58 | 10.26 | -0.32 |
| 11 | 503 | 11.51 | 11.04 | -0.46 |
| 12 | 657 | 12.50 | 12.08 | -0.42 |
| 13 | 587 | 13.52 | 13.03 | -0.49 |
| 14 | 736 | 14.54 | 14.15 | -0.39 |
| 15 | 670 | 15.53 | 15.18 | -0.35 |
| 16 | 685 | 16.50 | 16.18 | -0.33 |
| 17 | 783 | 17.48 | 17.10 | -0.39 |
| 18 | 790 | 18.54 | 18.25 | -0.29 |
| 19 | 590 | 19.49 | 19.04 | -0.45 |
| 20 | 349 | 20.55 | 19.89 | -0.66 |
| 21 | 331 | 21.48 | 20.41 | -1.07 |
| 22 | 241 | 22.48 | 21.16 | -1.31 |
| 23 | 152 | 23.45 | 22.28 | -1.17 |
| 24 | 169 | 24.51 | 22.76 | -1.75 |
| 25 | 156 | 25.51 | 23.74 | -1.77 |
| 26 | 132 | 26.53 | 24.63 | -1.90 |
| 27 | 104 | 27.47 | 26.00 | -1.47 |
| 28 | 82 | 28.41 | 26.54 | -1.87 |
| 29 | 78 | 29.53 | 27.69 | -1.84 |
| 30 | 64 | 30.46 | 28.48 | -1.98 |
| 31 | 58 | 31.44 | 29.94 | -1.50 |
| 32 | 62 | 32.52 | 31.72 | -0.81 |
| 33 | 44 | 33.32 | 31.68 | -1.65 |
| 34 | 49 | 34.70 | 30.87 | -3.83 |
| 35 | 66 | 35.56 | 32.56 | -2.99 |
| 38 | 34 | 38.40 | 31.57 | -6.84 |

## R2d following (the 12 `low_slow_behind_lead`)

Decisions with a moving (>= 0.5 m/s) same-direction vehicle in the corridor <= 40 m; the human's gap to the same actor id when the log was at the ego's station (asl actors replay the log); time gap = gap / own speed (both >= 1 m/s). Per-scene medians:

| scene | decisions | ego gap m | human gap m | ego time gap s | human time gap s | ratio | ego v | human v | lead v |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 098e2482 | 139 | 13.8 | 10.3 | 9.40 | 4.37 | 2.15 | 1.7 | 2.4 | 2.4 |
| 132ae6cb | 26 | 13.8 | 6.7 | 5.47 | 2.39 | 2.28 | 2.5 | 2.8 | 5.5 |
| 21829be5 | 86 | 13.0 | 7.7 | 3.39 | 1.84 | 1.84 | 3.6 | 4.5 | 4.1 |
| 404541da | 91 | 16.3 | 14.1 | 3.41 | 2.25 | 1.52 | 4.8 | 6.2 | 7.1 |
| 593ebf97 | 109 | 14.9 | 10.3 | 4.31 | 2.23 | 1.93 | 3.2 | 4.6 | 5.2 |
| 6fd8cf30 | 64 | 16.0 | 12.4 | 7.26 | 2.79 | 2.60 | 2.4 | 4.3 | 3.1 |
| a686c7d0 | 20 | 14.7 | 8.6 | 6.96 | 2.67 | 2.61 | 2.1 | 3.2 | 6.3 |
| bf56fd00 | 86 | 17.5 | 16.5 | 2.00 | 1.81 | 1.11 | 8.7 | 9.1 | 10.0 |
| cbb22323 | 32 | 8.5 | 6.2 | 5.26 | 2.88 | 1.83 | 1.6 | 2.3 | 1.3 |
| db8693db | 106 | 12.9 | 10.0 | 2.63 | 1.59 | 1.66 | 4.9 | 6.3 | 6.7 |
| df99cdc4 | 84 | 18.5 | 14.9 | 2.29 | 1.49 | 1.53 | 8.1 | 10.3 | 10.5 |

11 of 12 scenes with paired decisions; median over scenes of the time-gap ratio = 1.84 (ego 4.31 s, human 2.25 s). Line (median time gap >= 1.5 x the human's -> follows far, else follows slowly): **follows far (gap)**.

## Figures

![fixed point](../../figs/lon_diag_fixed_point.png)

R2c: grey = decision density, orange = bin means; the bin mean sits a near-constant 0.3-0.5 m/s below the diagonal from 3 to 19 m/s and 1-2 m/s below above 20 m/s, so g < 1 comes mostly from the fast end and the bin crossing (v* ~ 3 m/s) is where the slow scenes settle (onset figure: the cars decay to 3-4 m/s and hold).

![onsets](../../figs/lon_diag_onsets.png)

R2b figure: the 8 lowest-score `low_slow_free` scenes with an onset at >= 1 s (most onsets are at the handoff and have no frame worth showing); red line = onset; the human (black) often speeds up while the plan decays.
