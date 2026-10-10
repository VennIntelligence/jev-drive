**EXPLORATORY (HEAD1b: second attempt after the missed registered gate G3, user-authorised; not a registered read).**

`b`: navtest 12146 tokens, per-token seed means x 100; differences to H0 = GH0-F-s0, GH0-F-s1 (88.48; per seed 88.46 / 88.51) with 95% CI (log-cluster paired bootstrap, B 4000). Lines: memory.

| arm | tags | EPDMS - base | per seed | - shuffled | memory masked - on (seeds) | dev ADE on / masked / mismatched, per seed (m) | channel read | verdict |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| HP | H1P-F-s0, H1P-F-s1 | -0.03 [-0.21, +0.16] | -0.04 / -0.01 | +0.14 [-0.03, +0.31] | -0.11 [-0.31, +0.10] (2) | 0.628 / 0.637 / 0.682 ; 0.625 / 0.638 / 0.651 | NO | ceiling not measured (the channel was not read) |
| HPX | H1PX-F-s0, H1PX-F-s1 | -0.16 [-0.30, -0.02] | -0.12 / -0.21 | - | - | 0.626 / 0.625 / 0.626 ; 0.631 / 0.631 / 0.632 | NO | ceiling not measured (the channel was not read) |

Baseline dev ADE (m): 0.632 ; 0.629

**EPDMS by logged heading change over 4 s (difference to the baseline).**

| arm / contrast | < 5 deg (straight) | 5-20 deg | 20-45 deg | > 45 deg | > 20 deg |
|:--|:--|:--|:--|:--|:--|
| HP - H0 | -0.07 [-0.27, +0.10] | +0.12 [-0.23, +0.51] | +0.03 [-0.55, +0.54] | -0.14 [-0.92, +0.73] | -0.05 [-0.57, +0.45] |
| HP - HPX | +0.00 [-0.16, +0.18] | +0.13 [-0.32, +0.59] | +0.39 [-0.23, +0.97] | +0.44 [-0.32, +1.24] | - |
| HP masked - HP | -0.04 [-0.15, +0.07] | -0.47 [-0.86, -0.11] | -0.26 [-1.05, +0.65] | +0.34 [-0.54, +1.26] | - |
| HP masked - H0 | -0.11 [-0.27, +0.03] | -0.35 [-0.69, +0.00] | -0.23 [-0.75, +0.33] | +0.21 [-0.42, +0.80] | - |
| HPX - H0 | -0.08 [-0.22, +0.06] | -0.01 [-0.36, +0.36] | -0.36 [-0.88, +0.08] | -0.57 [-1.06, -0.10] | -0.47 [-0.83, -0.14] |

**Turn failures at > 45 deg (1 517 tokens; %, difference in pp), DAC failures, sub-scores (difference to the baseline).**

| arm | cut inside: arm / base, diff | cannot make the turn: arm / base, diff | DAC fail %, > 45 | DAC fail %, > 20 | NC | DAC | EP | TTC | NC + TTC fail % |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| HP | 4.78 / 4.12, +0.66 [+0.24, +1.06] | 3.10 / 4.19, -1.09 [-1.82, -0.36] | -0.03 [-0.72, +0.62] | -0.02 [-0.48, +0.45] | -0.07 [-0.17, +0.02] | +0.03 [-0.11, +0.16] | +0.14 [+0.09, +0.20] | -0.06 [-0.19, +0.08] | +0.07 [-0.07, +0.20] |
| HPX | 4.25 / 4.12, +0.13 [-0.25, +0.55] | 4.19 / 4.19, +0.00 [-0.35, +0.40] | +0.26 [-0.26, +0.84] | +0.19 [-0.19, +0.62] | -0.06 [-0.13, +0.01] | -0.02 [-0.15, +0.11] | -0.00 [-0.05, +0.04] | -0.09 [-0.17, -0.01] | +0.09 [-0.00, +0.19] |

**4 s arc length of the plan and the plan's own 4 s heading error (RMS, deg; token x seed pooled; before = baseline, after = arm).**

| arm | arc ratio to base | arm / log, base / log | straight / > 45 deg ratio | heading error all: before, after, diff | > 20 deg | > 45 deg |
|:--|:--|:--|:--|:--|:--|:--|
| HP | 0.9991 [0.9982, 1.0000] | 0.9976, 0.9984 | 0.9991 / 0.9970 | 7.38 [6.24, 8.50], 6.23 [5.32, 7.10], -1.15 [-1.52, -0.79] | 12.90 [11.03, 14.74], 10.57 [9.16, 11.93], -2.33 [-3.06, -1.65] | 15.60 [13.31, 17.57], 12.40 [10.59, 13.98], -3.19 [-4.06, -2.35] |
| HPX | 0.9990 [0.9984, 0.9996] | 0.9974, 0.9984 | 0.9990 / 0.9982 | 7.38 [6.24, 8.50], 7.72 [6.50, 8.96], +0.35 [+0.21, +0.50] | 12.90 [11.03, 14.74], 13.61 [11.56, 15.64], +0.71 [+0.44, +0.99] | 15.60 [13.31, 17.57], 16.55 [14.05, 18.76], +0.95 [+0.55, +1.36] |

**The fed heading profile's realised quality (error to the logged path; pilot train rows = navsim/op-parity-s234-train@v1:08fc1d5edd65 against navtest).** Profiles from /root/autodl-tmp/ujs/runs/corridor/head1/final/prof.

| rows | bucket | n | heading error at the logged 4 s arc, RMS deg | polyline position error at the logged 4 s arc, RMS m | heading error over the grid 2.5-40 m, RMS deg |
|:--|:--|--:|:--|:--|--:|
| pilot_train_rows (25415 rows, 1141 logs) | all | 25415 | 4.18 [3.69, 4.72] | 0.58 [0.53, 0.63] | 6.93 |
| pilot_train_rows (25415 rows, 1141 logs) | > 20 deg | 7040 | 6.49 [5.57, 7.59] | 0.91 [0.82, 1.01] | 8.24 |
| pilot_train_rows (25415 rows, 1141 logs) | > 45 deg | 2740 | 8.10 [6.37, 10.21] | 1.14 [0.99, 1.32] | 9.56 |
| navtest (12146 rows, 136 logs) | all | 12146 | 3.99 [3.29, 4.62] | 0.63 [0.54, 0.72] | 8.91 |
| navtest (12146 rows, 136 logs) | > 20 deg | 3154 | 6.65 [5.35, 7.84] | 1.03 [0.87, 1.23] | 9.40 |
| navtest (12146 rows, 136 logs) | > 45 deg | 1517 | 7.71 [5.98, 9.24] | 1.24 [0.99, 1.52] | 10.01 |

Train rows / navtest, heading RMS: all 1.05, > 20 deg 0.98, > 45 deg 1.05 (decision 204's QH head: ADE 0.55 m on its pilot train rows against 0.76 m on navtest, ratio 0.72: an in-sample head).

**R2 conversion (decision 243) for the fed navtest profile against GH0-F-s0, GH0-F-s1, read at the plan's own 4 s arc length; measured gain of HP: -0.03 [-0.21, +0.16].**

| stratum | head RMS deg | plan RMS deg | diff | corr | R2 |
|:--|:--|:--|:--|:--|:--|
| all | 5.19 [4.54, 5.85] | 7.38 [6.24, 8.50] | -2.18 [-2.93, -1.46] | 0.54 [0.46, 0.60] | 0.524 [0.439, 0.600] |
| > 20 deg | 8.42 [7.35, 9.53] | 12.90 [11.03, 14.74] | -4.48 [-5.80, -3.25] | 0.54 [0.45, 0.61] | 0.584 [0.500, 0.658] |
| > 45 deg | 9.29 [7.89, 10.74] | 15.60 [13.31, 17.57] | -6.30 [-8.12, -4.64] | 0.53 [0.43, 0.61] | 0.648 [0.552, 0.736] |

| prediction | predicted gain | inside the measured 95% CI | measured / predicted |
|:--|--:|:--|--:|
| stage1-head-decision-243 | +0.48 | does not hold | -0.05 |
| 0.93 x R2 of the fed profile | +0.49 | does not hold | -0.05 |
