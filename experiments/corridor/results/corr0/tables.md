## 0. Lane-sequence matching (privileged: map + logged future)

| tokens | n | matched | failed % | no candidate t0 | < 5 of 9 poses | no connected sequence | lane change % | gap frames >= 1 % | log leaves interior % | median max |d_log| m | mean |d0| m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all navtest | 12146 | 12023 | 1.01 | 64 | 19 | 40 | 5.15 | 0.90 | 70.85 | 0.46 | 0.49 |
| >20 | 3154 | 3083 | 2.25 | 36 | 12 | 23 | 8.40 | 2.89 | 91.05 | 0.82 | 0.64 |
| >45 | 1517 | 1476 | 2.70 | 18 | 6 | 17 | 8.33 | 4.61 | 94.04 | 0.85 | 0.58 |
| 20-45 | 1637 | 1607 | 1.83 | 18 | 6 | 6 | 8.46 | 1.31 | 88.30 | 0.76 | 0.70 |

`gap frames`: token has a 0-4 s log pose with no lane candidate (heading > 60 deg off every containing lane, or > 3 m from any). `log leaves interior`: the logged path needs a shift > 5 cm to keep the ego body 0.2 m inside its own lane (the KP clamp). `max |d_log|`: largest lateral distance of the logged path from the corridor centreline; d0 = the ego's offset at t0.

## 1. Error decomposition on the lane graph (SH30, token-seeds; shares in % with log-cluster 95% CIs)

**>20 deg** (3083 matched tokens x 2 seeds; per-seed means: DAC failures 225.5, inside-cut 123.0, cannot-make-turn 47.0, lost EPDMS sum 30970)

| class | tokens | DAC failures | inside-cut | cannot-make-turn | lost EPDMS (LL - PP) |
|:--|--:|--:|--:|--:|--:|
| a1 other exit | 1.1 [0.6, 1.6] | 0.7 [0.0, 1.4] | 0.0 [0.0, 0.0] | 3.2 [0.0, 7.0] | 0.9 [0.2, 1.8] |
| a2 other lane | 3.1 [2.1, 4.4] | 0.9 [0.0, 2.4] | 0.0 [0.0, 0.0] | 3.2 [0.0, 8.7] | 1.7 [0.4, 3.3] |
| b along-track | 69.1 [66.0, 72.2] | 61.9 [54.0, 70.4] | 39.8 [30.6, 51.1] | 83.0 [67.9, 94.8] | 57.4 [50.8, 64.5] |
| c cross-track | 23.6 [21.4, 26.0] | 30.8 [22.3, 38.8] | 52.8 [40.5, 63.0] | 4.3 [0.0, 10.5] | 29.3 [23.4, 35.9] |
| unmatched end | 3.1 [1.8, 4.6] | 5.8 [1.7, 10.3] | 7.3 [1.3, 15.2] | 6.4 [0.0, 19.4] | 10.6 [5.0, 16.2] |
| short (< 2 m) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| same exit, small error (r0 < 0.3 m) | 54.8 [51.4, 57.9] | 46.6 [36.8, 55.5] | 52.4 [36.7, 64.7] | 24.5 [11.8, 40.0] | 39.8 [32.6, 46.9] |
| same exit, neither model explains half | 37.8 [35.4, 40.5] | 45.7 [36.9, 54.1] | 66.3 [54.0, 76.6] | 14.9 [5.0, 28.0] | 41.0 [34.7, 47.5] |
| same_amb (overlap rule) | 3.0 [2.3, 3.8] | 2.4 [0.5, 5.0] | 1.6 [0.0, 4.3] | 5.3 [0.0, 14.3] | 2.1 [0.8, 3.9] |
| lane-change tokens | 8.4 [5.9, 11.6] | 2.2 [0.4, 5.1] | 2.4 [0.0, 7.1] | 0.0 [0.0, 0.0] | 3.3 [1.2, 6.2] |

**>45 deg** (1476 matched tokens x 2 seeds; per-seed means: DAC failures 134.0, inside-cut 64.0, cannot-make-turn 36.0, lost EPDMS sum 17615)

| class | tokens | DAC failures | inside-cut | cannot-make-turn | lost EPDMS (LL - PP) |
|:--|--:|--:|--:|--:|--:|
| a1 other exit | 1.8 [1.0, 2.8] | 1.1 [0.0, 2.3] | 0.0 [0.0, 0.0] | 4.2 [0.0, 8.9] | 1.6 [0.5, 3.2] |
| a2 other lane | 3.8 [2.6, 5.4] | 1.5 [0.0, 4.1] | 0.0 [0.0, 0.0] | 4.2 [0.0, 11.5] | 1.9 [0.1, 4.4] |
| b along-track | 70.4 [66.3, 74.4] | 66.4 [57.6, 75.8] | 46.9 [33.6, 60.7] | 79.2 [61.3, 94.4] | 59.0 [51.1, 68.0] |
| c cross-track | 19.4 [16.8, 22.3] | 23.5 [15.2, 33.0] | 43.8 [29.5, 57.9] | 4.2 [0.0, 11.8] | 24.0 [17.4, 31.5] |
| unmatched end | 4.6 [2.2, 7.0] | 7.5 [1.3, 13.8] | 9.4 [0.0, 21.7] | 8.3 [0.0, 24.0] | 13.5 [5.8, 20.1] |
| short (< 2 m) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| same exit, small error (r0 < 0.3 m) | 42.6 [38.5, 46.2] | 31.7 [20.3, 42.4] | 36.7 [19.6, 53.2] | 16.7 [5.3, 31.9] | 29.2 [20.1, 37.5] |
| same exit, neither model explains half | 34.0 [31.1, 37.0] | 38.1 [28.2, 48.3] | 62.5 [47.7, 76.3] | 9.7 [1.4, 21.4] | 36.0 [28.8, 43.6] |
| same_amb (overlap rule) | 2.0 [1.2, 2.8] | 3.4 [0.4, 7.4] | 1.6 [0.0, 5.0] | 6.9 [0.0, 18.8] | 2.8 [0.7, 5.7] |
| lane-change tokens | 8.3 [5.6, 11.9] | 1.5 [0.0, 4.1] | 1.6 [0.0, 5.5] | 0.0 [0.0, 0.0] | 2.1 [0.3, 4.7] |

Fit sizes on same-exit token-seeds (r0 = RMS distance plan curve to log curve at equal arc length; `explained` = 1 - min(rA, rC) / r0):

| bucket | PP kind | token-seeds | r0 RMS m | r after shift | r after offset | median delta m (late +) | median c m (inside +) | mean explained share | along-track wins % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| >20 | inside | 228 | 0.38 | 0.25 | 0.28 | 0.00 | 0.15 | 0.37 | 42.98 |
| >20 | cannot | 82 | 0.56 | 0.15 | 0.42 | 0.60 | -0.30 | 0.67 | 95.12 |
| >20 | pass | 5300 | 0.31 | 0.13 | 0.22 | 0.00 | -0.05 | 0.55 | 75.17 |
| >45 | inside | 116 | 0.48 | 0.28 | 0.36 | -0.20 | 0.25 | 0.40 | 51.72 |
| >45 | cannot | 60 | 0.63 | 0.16 | 0.47 | 0.55 | -0.35 | 0.70 | 95.00 |
| >45 | pass | 2410 | 0.38 | 0.16 | 0.28 | 0.10 | -0.10 | 0.55 | 78.84 |

Fallback for the unmatched plan ends (> 20 deg, class of the last plan pose that has a lane candidate): {'same': 188, 'same_amb': 2, 'a1': 1}

## 2. Re-target swap (no-EC EPDMS x 100, non-reactive; seed means; privileged arms)

| bucket | arm | n | EPDMS | vs PP | DAC | NC | TTC | EP | LK | DDC |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| >20 | PP stored plan | 3083 | 83.89 |  | 92.69 | 98.73 | 97.88 | 85.13 | 94.94 | 98.61 |
| >20 | CE corridor path + plan error | 3083 | 74.01 | -9.89 [-11.95, -7.87] | 83.54 | 97.44 | 92.83 | 84.25 | 93.04 | 98.99 |
| >20 | CP corridor path (no error) | 3083 | 80.44 | -3.46 [-6.47, -0.41] | 88.68 | 98.70 | 94.21 | 84.67 | 98.90 | 100.00 |
| >20 | KE clamped log + plan error | 3083 | 76.23 | -7.66 [-9.08, -6.37] | 86.30 | 97.69 | 95.60 | 83.93 | 94.31 | 98.51 |
| >20 | KP clamped log (no error) | 3083 | 79.19 | -4.70 [-7.03, -2.53] | 87.79 | 98.69 | 96.76 | 84.32 | 97.32 | 99.25 |
| >20 | LP log path, plan timing (d207) | 3083 | 91.17 | +7.28 [+5.70, +8.78] | 98.07 | 99.71 | 99.12 | 85.52 | 99.16 | 99.81 |
| >20 | LL log (d207) | 3083 | 93.94 | +10.05 [+8.62, +11.48] | 100.00 | 100.00 | 100.00 | 85.89 | 100.00 | 99.77 |
| 20-45 | PP stored plan | 1607 | 85.69 |  | 94.31 | 99.13 | 97.60 | 86.29 | 94.90 | 98.99 |
| 20-45 | CE corridor path + plan error | 1607 | 75.01 | -10.68 [-13.36, -8.07] | 84.47 | 97.39 | 91.94 | 85.75 | 94.43 | 99.39 |
| 20-45 | CP corridor path (no error) | 1607 | 80.37 | -5.32 [-9.50, -1.25] | 88.86 | 98.60 | 93.22 | 85.95 | 99.44 | 100.00 |
| 20-45 | KE clamped log + plan error | 1607 | 77.18 | -8.51 [-10.47, -6.53] | 86.93 | 97.79 | 95.08 | 85.41 | 94.12 | 98.90 |
| 20-45 | KP clamped log (no error) | 1607 | 79.42 | -6.27 [-9.15, -3.45] | 88.24 | 98.44 | 96.24 | 85.54 | 96.42 | 99.32 |
| 20-45 | LP log path, plan timing (d207) | 1607 | 91.91 | +6.22 [+4.79, +7.74] | 98.97 | 99.69 | 99.00 | 86.43 | 98.76 | 99.77 |
| 20-45 | LL log (d207) | 1607 | 94.00 | +8.31 [+6.88, +9.85] | 100.00 | 100.00 | 100.00 | 86.31 | 100.00 | 99.81 |
| >45 | PP stored plan | 1476 | 81.94 |  | 90.92 | 98.29 | 98.17 | 83.87 | 94.99 | 98.20 |
| >45 | CE corridor path + plan error | 1476 | 72.92 | -9.02 [-11.53, -6.50] | 82.52 | 97.49 | 93.80 | 82.61 | 91.53 | 98.56 |
| >45 | CP corridor path (no error) | 1476 | 80.51 | -1.42 [-4.41, +1.73] | 88.48 | 98.81 | 95.29 | 83.27 | 98.31 | 100.00 |
| >45 | KE clamped log + plan error | 1476 | 75.19 | -6.74 [-8.66, -5.00] | 85.60 | 97.58 | 96.17 | 82.33 | 94.51 | 98.09 |
| >45 | KP clamped log (no error) | 1476 | 78.94 | -2.99 [-5.79, -0.38] | 87.30 | 98.95 | 97.32 | 82.99 | 98.31 | 99.19 |
| >45 | LP log path, plan timing (d207) | 1476 | 90.36 | +8.43 [+5.82, +10.91] | 97.09 | 99.73 | 99.25 | 84.53 | 99.59 | 99.85 |
| >45 | LL log (d207) | 1476 | 93.87 | +11.93 [+9.84, +13.99] | 100.00 | 100.00 | 100.00 | 85.42 | 100.00 | 99.73 |

DAC failures of the stored plan removed by each arm (gross = share of PP's failures that pass; net counts the arm's new failures):

| bucket | PP failures | arm | PP failures (seed mean) | gross removed % | net removed % | arm DAC failures (seed mean) |
|:--|--:|--:|--:|--:|--:|--:|
| >20 | DAC | CE corridor path + plan error | 225.50 | 25.3 [17.5, 32.8] | -125.1 [-172.2, -88.2] | 507.50 |
| >20 | DAC | CP corridor path (no error) | 225.50 | 70.7 [59.1, 81.0] | -54.8 [-112.4, -8.4] | 349.00 |
| >20 | DAC | KE clamped log + plan error | 225.50 | 22.2 [15.9, 28.6] | -87.4 [-119.2, -63.8] | 422.50 |
| >20 | DAC | KP clamped log (no error) | 225.50 | 73.4 [65.6, 80.6] | -67.0 [-113.3, -33.8] | 376.50 |
| >20 | DAC | LP log path, plan timing (d207) | 225.50 | 94.5 [91.7, 97.1] | 73.6 [61.1, 83.7] | 59.50 |
| >20 | inside-cut | CE corridor path + plan error | 123.00 | 35.4 [25.8, 44.9] |  |  |
| >20 | inside-cut | CP corridor path (no error) | 123.00 | 85.0 [76.6, 91.7] |  |  |
| >20 | inside-cut | KE clamped log + plan error | 123.00 | 29.3 [22.4, 36.8] |  |  |
| >20 | inside-cut | KP clamped log (no error) | 123.00 | 82.5 [75.6, 88.8] |  |  |
| >20 | inside-cut | LP log path, plan timing (d207) | 123.00 | 95.1 [91.0, 98.6] |  |  |
| >20 | cannot-make-turn | CE corridor path + plan error | 47.00 | 8.5 [1.1, 18.6] |  |  |
| >20 | cannot-make-turn | CP corridor path (no error) | 47.00 | 72.3 [55.9, 86.1] |  |  |
| >20 | cannot-make-turn | KE clamped log + plan error | 47.00 | 11.7 [3.6, 20.0] |  |  |
| >20 | cannot-make-turn | KP clamped log (no error) | 47.00 | 72.3 [58.2, 85.1] |  |  |
| >20 | cannot-make-turn | LP log path, plan timing (d207) | 47.00 | 95.7 [89.9, 100.0] |  |  |
| 20-45 | DAC | CE corridor path + plan error | 91.50 | 23.0 [13.7, 31.9] | -172.7 [-272.0, -112.7] | 249.50 |
| 20-45 | DAC | CP corridor path (no error) | 91.50 | 73.2 [59.2, 84.2] | -95.6 [-219.8, -20.2] | 179.00 |
| 20-45 | DAC | KE clamped log + plan error | 91.50 | 21.9 [13.4, 30.7] | -129.5 [-199.3, -84.3] | 210.00 |
| 20-45 | DAC | KP clamped log (no error) | 91.50 | 77.6 [68.9, 84.6] | -106.6 [-200.0, -50.0] | 189.00 |
| 20-45 | DAC | LP log path, plan timing (d207) | 91.50 | 95.1 [88.8, 99.4] | 82.0 [68.1, 92.0] | 16.50 |
| 20-45 | inside-cut | CE corridor path + plan error | 59.00 | 25.4 [13.9, 36.6] |  |  |
| 20-45 | inside-cut | CP corridor path (no error) | 59.00 | 84.7 [72.4, 93.5] |  |  |
| 20-45 | inside-cut | KE clamped log + plan error | 59.00 | 28.0 [17.0, 39.2] |  |  |
| 20-45 | inside-cut | KP clamped log (no error) | 59.00 | 84.7 [76.1, 91.4] |  |  |
| 20-45 | inside-cut | LP log path, plan timing (d207) | 59.00 | 95.8 [88.6, 100.0] |  |  |
| 20-45 | cannot-make-turn | CE corridor path + plan error | 11.00 | 13.6 [0.0, 34.6] |  |  |
| 20-45 | cannot-make-turn | CP corridor path (no error) | 11.00 | 63.6 [33.3, 89.5] |  |  |
| 20-45 | cannot-make-turn | KE clamped log + plan error | 11.00 | 9.1 [0.0, 25.9] |  |  |
| 20-45 | cannot-make-turn | KP clamped log (no error) | 11.00 | 59.1 [33.3, 85.7] |  |  |
| 20-45 | cannot-make-turn | LP log path, plan timing (d207) | 11.00 | 95.5 [82.6, 100.0] |  |  |
| >45 | DAC | CE corridor path + plan error | 134.00 | 26.9 [17.1, 37.3] | -92.5 [-135.5, -56.9] | 258.00 |
| >45 | DAC | CP corridor path (no error) | 134.00 | 69.0 [56.7, 81.5] | -26.9 [-72.0, 12.8] | 170.00 |
| >45 | DAC | KE clamped log + plan error | 134.00 | 22.4 [13.5, 32.6] | -58.6 [-87.5, -37.8] | 212.50 |
| >45 | DAC | KP clamped log (no error) | 134.00 | 70.5 [61.4, 79.7] | -39.9 [-82.3, -10.0] | 187.50 |
| >45 | DAC | LP log path, plan timing (d207) | 134.00 | 94.0 [90.0, 97.7] | 67.9 [47.7, 82.4] | 43.00 |
| >45 | inside-cut | CE corridor path + plan error | 64.00 | 44.5 [30.6, 56.8] |  |  |
| >45 | inside-cut | CP corridor path (no error) | 64.00 | 85.2 [74.3, 94.1] |  |  |
| >45 | inside-cut | KE clamped log + plan error | 64.00 | 30.5 [20.2, 41.7] |  |  |
| >45 | inside-cut | KP clamped log (no error) | 64.00 | 80.5 [71.9, 88.6] |  |  |
| >45 | inside-cut | LP log path, plan timing (d207) | 64.00 | 94.5 [88.0, 99.3] |  |  |
| >45 | cannot-make-turn | CE corridor path + plan error | 36.00 | 6.9 [0.0, 15.9] |  |  |
| >45 | cannot-make-turn | CP corridor path (no error) | 36.00 | 75.0 [59.6, 88.0] |  |  |
| >45 | cannot-make-turn | KE clamped log + plan error | 36.00 | 12.5 [1.6, 25.4] |  |  |
| >45 | cannot-make-turn | KP clamped log (no error) | 36.00 | 76.4 [61.4, 89.1] |  |  |
| >45 | cannot-make-turn | LP log path, plan timing (d207) | 36.00 | 95.8 [88.6, 100.0] |  |  |

Without lane-change tokens (> 20 deg, 2824 tokens): CE vs PP -10.24 [-12.50, -8.10], CP vs PP -2.91 [-6.28, +0.46].

Post hoc (not registered): side of the first footprint corner outside the drivable area, devkit LQR replay (decision 153's hook):

| bucket | arm | DAC failures (seed mean) | first corner out on the inside of the turn % | on the outside % | raw poses already outside % | mean depth m |
|:--|--:|--:|--:|--:|--:|--:|
| >20 | PP stored plan | 225.50 | 54.55 | 45.45 | 53.88 | 0.49 |
| >20 | CP corridor path (no error) | 349.00 | 30.37 | 69.63 | 14.18 | 0.28 |
| >20 | CE corridor path + plan error | 507.50 | 30.94 | 69.06 | 47.29 | 0.54 |
| >20 | KP clamped log (no error) | 376.50 | 51.66 | 48.34 | 58.83 | 0.53 |
| >20 | KE clamped log + plan error | 422.50 | 45.56 | 54.44 | 67.46 | 0.63 |
| >45 | PP stored plan | 134.00 | 47.76 | 52.24 | 61.19 | 0.55 |
| >45 | CP corridor path (no error) | 170.00 | 33.82 | 66.18 | 12.94 | 0.36 |
| >45 | CE corridor path + plan error | 258.00 | 25.00 | 75.00 | 55.23 | 0.62 |
| >45 | KP clamped log (no error) | 187.50 | 61.07 | 38.93 | 57.07 | 0.58 |
| >45 | KE clamped log + plan error | 212.50 | 44.71 | 55.29 | 73.18 | 0.71 |

Post hoc: where the logged path sits relative to the corridor centreline (rear axle, tokens without a lane change):

| bucket | n (no lane change) | log minus centreline at 4 s, inside + (m) | > 0.3 m inside % | > 0.3 m outside % | mean |offset| m | mean |d0| m |
|:--|--:|--:|--:|--:|--:|--:|
| >20 | 2824 | +0.24 [+0.19, +0.28] | 44.97 | 15.40 | 0.52 | 0.51 |
| 20-45 | 1471 | +0.20 [+0.15, +0.26] | 42.56 | 15.30 | 0.48 | 0.52 |
| >45 | 1353 | +0.27 [+0.21, +0.34] | 47.60 | 15.52 | 0.57 | 0.50 |

## 3. Is map + exit choice enough for the 4 s heading (error against the logged 4 s heading, deg)

| tokens | source | n | RMS deg | robust sigma | mean abs | > 10 deg % |
|:--|--:|--:|--:|--:|--:|--:|
| all navtest | SH30 plan (own 4 s heading) | 12023 | 5.62 [4.94, 6.28] | 1.17 | 2.63 | 6.39 |
| all navtest | map centreline at the plan's 4 s arc length | 12023 | 4.51 [4.08, 4.92] | 1.17 | 2.27 | 4.97 |
| all navtest | map centreline at the log's 4 s arc length | 12023 | 3.20 [2.93, 3.45] | 1.11 | 1.76 | 2.40 |
| all navtest | exit pose (end of the first connector ahead) | 8674 | 26.63 [24.83, 28.33] | 3.21 | 13.07 | 27.30 |
| < 20 | SH30 plan (own 4 s heading) | 8940 | 3.40 [2.82, 3.95] | 0.77 | 1.40 | 1.98 |
| < 20 | map centreline at the plan's 4 s arc length | 8940 | 3.17 [2.81, 3.53] | 0.85 | 1.44 | 2.40 |
| < 20 | map centreline at the log's 4 s arc length | 8940 | 2.44 [2.19, 2.67] | 0.83 | 1.25 | 1.51 |
| < 20 | exit pose (end of the first connector ahead) | 5846 | 25.42 [23.03, 27.70] | 1.81 | 9.98 | 15.67 |
| >20 | SH30 plan (own 4 s heading) | 3083 | 9.47 [8.49, 10.41] | 5.81 | 6.22 | 19.19 |
| >20 | map centreline at the plan's 4 s arc length | 3083 | 7.07 [6.38, 7.74] | 4.27 | 4.69 | 12.44 |
| >20 | map centreline at the log's 4 s arc length | 3083 | 4.77 [4.29, 5.25] | 3.15 | 3.21 | 4.96 |
| >20 | exit pose (end of the first connector ahead) | 2828 | 28.95 [26.86, 31.00] | 15.95 | 19.47 | 51.34 |
| >45 | SH30 plan (own 4 s heading) | 1476 | 11.04 [9.86, 12.10] | 7.59 | 7.53 | 25.20 |
| >45 | map centreline at the plan's 4 s arc length | 1476 | 7.25 [6.54, 7.93] | 5.35 | 5.18 | 13.69 |
| >45 | map centreline at the log's 4 s arc length | 1476 | 4.87 [4.34, 5.38] | 3.96 | 3.55 | 5.01 |
| >45 | exit pose (end of the first connector ahead) | 1406 | 24.43 [22.47, 26.21] | 15.76 | 16.99 | 51.28 |

## 4. Counterfactual supply on navtrain (lane-graph paths within 40 m of arc ahead of the ego's own lane)

| tokens | n | matched | driven path known | >= 2 paths | 1 alt | 2 alts | 3+ alts | alt >= 20 deg | rows >= 20 | alt >= 45 deg | rows >= 45 | roadblock-level >= 20 | roadblock-level >= 45 | distinct branch nodes (>= 20) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| navtrain all | 103288 | 102019 | 93922 | 41395 / 1029 logs | 20841 | 7040 | 3029 | 18989 / 821 logs | 26351 | 16197 / 749 logs | 22165 | 35261 / 1103 logs | 28763 / 1060 logs | 491 |
| turn > 20 deg | 28322 | 27403 | 22604 | 15963 / 792 logs | 7846 | 2839 | 1421 | 7140 / 601 logs | 10436 | 5978 / 556 logs | 8698 | 8890 / 664 logs | 6918 / 633 logs | 285 |

Match status: {'ok': 102019, 'no_connected_sequence': 570, 'no_candidate_t0': 545, 'no_candidate_4s': 154}; lane-change tokens (driven path not one of the own-lane paths): 8097. Tokens per log with a >= 20 deg alternative: 10 / 50 / 90th percentile [2.0, 16.0, 52.0], top log 210; start on a lane 14212, inside a connector 4777.

