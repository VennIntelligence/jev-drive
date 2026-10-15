# PLAN SWITCHING (rf_switching.py)

Prereg `plans/2026-10-15-plan-switching-prereg.md`. Window: decisions with the ego in entry - 20 .. entry + 20 m, before the first zero. r = plan 4 s heading change / route heading change over the plan's own arc (route >= 15 deg); switching rate = share of consecutive decision pairs with one r >= 0.6 and the other <= 0.3; j = perpendicular offset of the plan's 2 s point from the previous plan carried forward by the ego motion (+ = inside); e = offset of the car from the plan served 1 s earlier (+ = inside). Ratios failed / completed (S1: means of the per-turn rate; S2: medians of the per-turn median |j|) with a scene-clustered (held: log) bootstrap 95 % CI, B 10 000. PAI decides every 0.1 s, held every 0.5 s: pair statistics compare within a population only.

Gates: G-frame pass (median |anchor - controller| per population held_RSG-L18-s0 0.000 m / 0.00 deg, held_RSG-L18-s1 0.000 m / 0.00 deg, held_RV2a-L18-s0 0.000 m / 0.00 deg, held_RV2a-L18-s1 0.000 m / 0.00 deg, held_RV2b-L18-s0 0.000 m / 0.00 deg, held_RV2b-L18-s1 0.000 m / 0.00 deg, pai_APP1X-L18-s0 0.000 m / 0.00 deg, pai_RSG-L18-s0 0.000 m / 0.00 deg, pai_RV2a-L18-s0 0.000 m / 0.00 deg, pai_RV2b-L18-s1 0.000 m / 0.00 deg, held_RSG 0.000 m / 0.00 deg); G-human (pai_RSG, human log as a plan) mean switching rate 0.0000 (n 35), median |j_h| 0.015 m (n 35): pass. Classes agree with RECEDING turns.csv on 1.000 of the turns. Decisions 11487, consecutive pairs 11194, r defined 7946, e defined 10537.

## S1 / S2

| population | turns f / c | S1 mean switching rate f / c | ratio [CI] | S1 | S2 median abs j f / c (m) | ratio [CI] | S2 | flip /s f / c | ratio [CI] |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| held_RSG-L18-s0 | 8 / 16 (rate 8 / 16) | 0.000 / 0.000 | - [-, -] | descriptive | 0.114 / 0.175 | 0.65 [0.43, 1.20] | descriptive | 0.00 / 0.00 | - [-, -] |
| held_RSG-L18-s1 | 10 / 14 (rate 10 / 14) | 0.000 / 0.000 | - [-, -] | descriptive | 0.145 / 0.152 | 0.96 [0.48, 1.59] | descriptive | 0.01 / 0.00 | inf [inf, inf] |
| held_RV2a-L18-s0 | 6 / 18 (rate 6 / 18) | 0.000 / 0.000 | - [-, -] | descriptive | 0.109 / 0.162 | 0.67 [0.29, 1.67] | descriptive | 0.00 / 0.01 | 0.00 [0.00, 0.00] |
| held_RV2a-L18-s1 | 9 / 15 (rate 9 / 15) | 0.000 / 0.000 | - [-, -] | descriptive | 0.118 / 0.155 | 0.76 [0.58, 1.33] | descriptive | 0.02 / 0.00 | inf [inf, inf] |
| held_RV2b-L18-s0 | 8 / 16 (rate 8 / 16) | 0.000 / 0.000 | - [-, -] | descriptive | 0.149 / 0.167 | 0.89 [0.38, 1.43] | descriptive | 0.00 / 0.01 | 0.00 [0.00, 0.00] |
| held_RV2b-L18-s1 | 7 / 17 (rate 7 / 17) | 0.000 / 0.000 | - [-, -] | descriptive | 0.124 / 0.163 | 0.76 [0.20, 1.09] | descriptive | 0.02 / 0.00 | inf [inf, inf] |
| pai_APP1X-L18-s0 | 25 / 11 (rate 21 / 11) | 0.001 / 0.000 | inf [inf, inf] | supported (degenerate: completed 0, failed < 0.01; no switching) | 0.041 / 0.057 | 0.71 [0.39, 1.50] | refuted | 0.10 / 0.00 | inf [inf, inf] |
| pai_RSG-L18-s0 | 31 / 5 (rate 30 / 5) | 0.000 / 0.000 | - [-, -] | inconclusive (0 / 0: no switching in either group) | 0.030 / 0.058 | 0.53 [0.21, 1.02] | refuted | 0.01 / 0.00 | inf [inf, inf] |
| pai_RV2a-L18-s0 | 31 / 5 (rate 30 / 5) | 0.000 / 0.000 | - [-, -] | inconclusive (0 / 0: no switching in either group) | 0.032 / 0.041 | 0.79 [0.39, 2.42] | refuted | 0.02 / 0.00 | inf [inf, inf] |
| pai_RV2b-L18-s1 | 29 / 7 (rate 28 / 7) | 0.000 / 0.000 | - [-, -] | inconclusive (0 / 0: no switching in either group) | 0.032 / 0.067 | 0.48 [0.30, 1.33] | refuted | 0.03 / 0.00 | inf [inf, inf] |
| held_RSG | 18 / 30 (rate 18 / 30) | 0.000 / 0.000 | - [-, -] | inconclusive (0 / 0: no switching in either group) | 0.133 / 0.169 | 0.79 [0.49, 1.21] | refuted | 0.01 / 0.00 | inf [inf, inf] |
| held_all6 | 48 / 96 (rate 48 / 96) | 0.000 / 0.000 | - [-, -] | descriptive | 0.133 / 0.162 | 0.82 [0.47, 1.14] | descriptive | 0.01 / 0.00 | 3.00 [1.28, 4.66] |
| pai_all4 | 116 / 28 (rate 109 / 28) | 0.000 / 0.000 | inf [inf, inf] | descriptive | 0.032 / 0.057 | 0.57 [0.38, 1.14] | descriptive | 0.04 / 0.00 | inf [inf, inf] |

Bootstrap bookkeeping (dropped replicates / replicates with ratio +inf) and the human log treated as a plan (mean switching rate, median abs j_h; f / c):

- held_RSG-L18-s0: S1 dropped 10000 inf 0; S2 dropped 5 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.006 m
- held_RSG-L18-s1: S1 dropped 10000 inf 0; S2 dropped 1 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.004 m
- held_RV2a-L18-s0: S1 dropped 10000 inf 0; S2 dropped 16 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.006 / 0.005 m
- held_RV2a-L18-s1: S1 dropped 10000 inf 0; S2 dropped 1 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.005 m
- held_RV2b-L18-s0: S1 dropped 10000 inf 0; S2 dropped 0 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.006 / 0.005 m
- held_RV2b-L18-s1: S1 dropped 10000 inf 0; S2 dropped 19 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.005 m
- pai_APP1X-L18-s0: S1 dropped 3573 inf 6427; S2 dropped 0 inf 0; turns without a rate 4, without j 3; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.015 / 0.008 m
- pai_RSG-L18-s0: S1 dropped 10000 inf 0; S2 dropped 53 inf 0; turns without a rate 1, without j 1; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.015 / 0.041 m
- pai_RV2a-L18-s0: S1 dropped 10000 inf 0; S2 dropped 47 inf 0; turns without a rate 1, without j 1; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.016 / 0.007 m
- pai_RV2b-L18-s1: S1 dropped 10000 inf 0; S2 dropped 2 inf 0; turns without a rate 1, without j 1; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.015 / 0.007 m
- held_RSG: S1 dropped 10000 inf 0; S2 dropped 1 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.005 m
- held_all6: S1 dropped 10000 inf 0; S2 dropped 0 inf 0; turns without a rate 0, without j 0; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.005 / 0.005 m
- pai_all4: S1 dropped 3602 inf 6398; S2 dropped 0 inf 0; turns without a rate 7, without j 6; pairs skipped (gap) 0; human switching 0.0000 / 0.0000, human abs j 0.015 / 0.010 m

## S3: tracking error and what the car executes

| population | median abs e f / c (m) | diff [CI] | signed e f / c (m) | diff [CI] | signed j f / c (m) | p90 abs j f / c (m) |
|:--|:--|:--|:--|:--|:--|:--|
| held_RSG-L18-s0 | 0.02 / 0.03 | -0.01 [-0.02, +0.01] | +0.02 / +0.03 | -0.01 [-0.02, +0.00] | -0.002 / +0.017 | 0.28 / 0.35 |
| held_RSG-L18-s1 | 0.02 / 0.02 | -0.00 [-0.01, +0.01] | +0.02 / +0.02 | -0.00 [-0.02, +0.00] | -0.019 / +0.019 | 0.32 / 0.34 |
| held_RV2a-L18-s0 | 0.02 / 0.03 | -0.01 [-0.02, +0.01] | +0.02 / +0.02 | -0.00 [-0.02, +0.01] | +0.011 / -0.005 | 0.28 / 0.37 |
| held_RV2a-L18-s1 | 0.02 / 0.03 | -0.01 [-0.02, +0.00] | +0.02 / +0.03 | -0.01 [-0.02, +0.00] | -0.071 / -0.021 | 0.29 / 0.34 |
| held_RV2b-L18-s0 | 0.02 / 0.03 | -0.01 [-0.02, +0.00] | +0.01 / +0.03 | -0.01 [-0.02, -0.00] | +0.035 / -0.018 | 0.39 / 0.37 |
| held_RV2b-L18-s1 | 0.02 / 0.02 | -0.00 [-0.02, +0.03] | +0.02 / +0.02 | -0.01 [-0.02, -0.00] | +0.006 / -0.005 | 0.27 / 0.36 |
| pai_APP1X-L18-s0 | 0.02 / 0.03 | -0.02 [-0.03, -0.00] | +0.01 / +0.01 | -0.01 [-0.02, +0.00] | -0.012 / -0.003 | 0.16 / 0.24 |
| pai_RSG-L18-s0 | 0.02 / 0.03 | -0.01 [-0.04, +0.00] | +0.01 / +0.02 | -0.01 [-0.03, +0.06] | -0.026 / -0.011 | 0.11 / 0.25 |
| pai_RV2a-L18-s0 | 0.01 / 0.02 | -0.01 [-0.04, +0.00] | +0.01 / +0.02 | -0.01 [-0.03, +0.06] | -0.020 / -0.016 | 0.13 / 0.13 |
| pai_RV2b-L18-s1 | 0.01 / 0.02 | -0.01 [-0.03, -0.00] | +0.01 / +0.02 | -0.01 [-0.03, +0.02] | -0.020 / -0.013 | 0.12 / 0.20 |
| held_RSG | 0.02 / 0.03 | -0.01 [-0.02, +0.01] | +0.02 / +0.02 | -0.01 [-0.02, -0.00] | -0.010 / +0.019 | 0.30 / 0.34 |
| held_all6 | 0.02 / 0.03 | -0.01 [-0.02, +0.00] | +0.02 / +0.02 | -0.01 [-0.02, -0.00] | +0.005 / -0.003 | 0.31 / 0.35 |
| pai_all4 | 0.01 / 0.03 | -0.01 [-0.03, -0.00] | +0.01 / +0.02 | -0.01 [-0.02, +0.00] | -0.020 / -0.010 | 0.13 / 0.21 |

Executed ratio R_exec (ego yaw change / route heading change over the window) against the time-average of r and against its p90 (robust max), across turns: OLS slope, intercept, R^2, median abs (R_exec - x); reading per the prereg.

| population | turns | median R_exec / mean r / p90 r | vs mean r: slope, icpt, R^2, mad | vs p90 r: slope, icpt, R^2, mad | reading |
|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 all | 31 | 0.83 / 1.02 / 1.42 | +0.83, -0.09, 0.73, 0.26 | +0.49, +0.01, 0.58, 0.68 | averages the sequence |
| pai_RSG-L18-s0 failed | 26 | 0.76 / 1.03 / 1.43 | +0.82, -0.11, 0.76, 0.30 | +0.49, -0.02, 0.61, 0.72 | averages the sequence |
| pai_RSG-L18-s0 completed | 5 | 0.91 / 1.01 / 1.42 | +0.46, +0.46, 0.04, 0.11 | -0.71, +1.95, 0.91, 0.51 | - |
| pai_all4 all | 119 | 0.86 / 1.01 / 1.41 | +0.87, -0.08, 0.69, 0.22 | +0.53, +0.02, 0.53, 0.59 | averages the sequence |
| pai_all4 failed | 91 | 0.77 / 1.02 / 1.42 | +0.86, -0.11, 0.71, 0.27 | +0.54, -0.04, 0.58, 0.67 | averages the sequence |
| pai_all4 completed | 28 | 0.93 / 0.99 / 1.40 | +0.36, +0.57, 0.21, 0.10 | -0.17, +1.16, 0.11, 0.42 | - |
| held_RSG all | 48 | 1.01 / 1.02 / 1.25 | +0.13, +0.87, 0.02, 0.12 | +0.04, +0.95, 0.01, 0.20 | neither |
| held_RSG failed | 18 | 0.98 / 1.12 / 1.29 | -0.15, +1.19, 0.03, 0.14 | -0.03, +1.07, 0.01, 0.34 | neither |
| held_RSG completed | 30 | 1.02 / 0.99 / 1.14 | +0.64, +0.36, 0.20, 0.08 | +0.40, +0.53, 0.22, 0.15 | - |
| held_all6 all | 144 | 1.00 / 1.01 / 1.20 | +0.13, +0.87, 0.02, 0.09 | +0.04, +0.95, 0.01, 0.19 | neither |
| held_all6 failed | 48 | 0.99 / 1.12 / 1.31 | -0.11, +1.14, 0.02, 0.13 | -0.02, +1.05, 0.01, 0.34 | neither |
| held_all6 completed | 96 | 1.02 / 0.98 / 1.16 | +0.67, +0.34, 0.23, 0.09 | +0.41, +0.54, 0.27, 0.15 | - |

## By failure class (vs the completed turns of the same population; ratio [CI], descriptive)

| population | class | n | switching ratio [CI] | abs j ratio [CI] | flip ratio [CI] | abs e f / c (m) | signed e | signed j | median mean r | median R_exec |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | late_wide | 12 | - [-, -] (0.000 / 0.000) | 0.77 [0.26, 1.46] (0.044 / 0.058) | - [-, -] (0.000 / 0.000) | 0.02 / 0.03 | +0.02 | -0.033 | 1.14 | 0.76 |
| pai_RSG-L18-s0 | straight | 8 | - [-, -] (0.000 / 0.000) | 0.42 [0.13, 0.86] (0.024 / 0.058) | inf [inf, inf] (0.028 / 0.000) | 0.01 / 0.03 | -0.01 | -0.022 | 0.09 | -0.02 |
| pai_RSG-L18-s0 | over_turn | 2 | - [-, -] (0.000 / 0.000) | 3.90 [1.01, 7.66] (0.224 / 0.058) | - [-, -] (0.000 / 0.000) | 0.05 / 0.03 | +0.03 | +0.197 | 0.31 | - |
| pai_RSG-L18-s0 | no_zero | 10 | - [-, -] (0.000 / 0.000) | 0.50 [0.14, 1.26] (0.029 / 0.058) | inf [inf, inf] (0.023 / 0.000) | 0.02 / 0.03 | +0.01 | -0.015 | 1.14 | 0.92 |
| pai_RSG-L18-s0 | other | 1 | - [-, -] (0.000 / 0.000) | 4.08 [1.60, 6.72] (0.235 / 0.058) | - [-, -] (0.000 / 0.000) | 0.05 / 0.03 | -0.05 | -0.235 | 1.00 | - |
| pai_all4 | late_wide | 42 | - [-, -] (0.000 / 0.000) | 0.70 [0.44, 1.44] (0.040 / 0.057) | inf [inf, inf] (0.014 / 0.000) | 0.02 / 0.03 | +0.02 | -0.025 | 1.10 | 0.77 |
| pai_all4 | straight | 26 | - [-, -] (0.000 / 0.000) | 0.36 [0.23, 0.97] (0.021 / 0.057) | inf [inf, inf] (0.077 / 0.000) | 0.01 / 0.03 | -0.01 | -0.018 | 0.11 | -0.02 |
| pai_all4 | over_turn | 2 | - [-, -] (0.000 / 0.000) | 3.93 [1.53, 8.98] (0.224 / 0.057) | - [-, -] (0.000 / 0.000) | 0.05 / 0.03 | +0.03 | +0.197 | 0.31 | - |
| pai_all4 | no_zero | 43 | inf [inf, inf] (0.000 / 0.000) | 0.45 [0.26, 1.11] (0.026 / 0.057) | inf [inf, inf] (0.039 / 0.000) | 0.01 / 0.03 | +0.01 | -0.010 | 1.15 | 0.92 |
| pai_all4 | other | 5 | - [-, -] (0.000 / 0.000) | 3.16 [0.95, 5.49] (0.181 / 0.057) | - [-, -] (0.000 / 0.000) | 0.04 / 0.03 | -0.04 | -0.181 | 1.01 | 1.09 |
| held_RSG | late_wide | 2 | - [-, -] (0.000 / 0.000) | 0.86 [0.73, 1.08] (0.145 / 0.169) | - [-, -] (0.000 / 0.000) | 0.02 / 0.03 | +0.02 | -0.123 | 1.41 | 0.80 |
| held_RSG | over_turn | 9 | - [-, -] (0.000 / 0.000) | 0.71 [0.27, 1.84] (0.120 / 0.169) | inf [inf, inf] (0.065 / 0.000) | 0.03 / 0.03 | +0.03 | +0.014 | 0.95 | 1.30 |
| held_RSG | no_zero | 16 | - [-, -] (0.000 / 0.000) | 0.72 [0.40, 1.27] (0.122 / 0.169) | inf [inf, inf] (0.009 / 0.000) | 0.02 / 0.03 | +0.02 | -0.002 | 1.11 | 0.99 |
| held_all6 | late_wide | 6 | - [-, -] (0.000 / 0.000) | 0.95 [0.77, 1.12] (0.154 / 0.162) | 0.00 [0.00, 0.00] (0.000 / 0.003) | 0.02 / 0.03 | +0.02 | -0.104 | 1.41 | 0.81 |
| held_all6 | over_turn | 9 | - [-, -] (0.000 / 0.000) | 0.74 [0.27, 1.83] (0.120 / 0.162) | 21.78 [0.00, inf] (0.065 / 0.003) | 0.03 / 0.03 | +0.03 | +0.014 | 0.95 | 1.30 |
| held_all6 | no_zero | 42 | - [-, -] (0.000 / 0.000) | 0.73 [0.41, 1.16] (0.118 / 0.162) | 3.43 [1.44, 5.02] (0.010 / 0.003) | 0.02 / 0.03 | +0.02 | +0.010 | 1.11 | 0.99 |

## Amendment 1 (post hoc, descriptive)

(a) Plan creep: signed j per metre of ego travel (per-turn median; - = each new plan's 2 s point lies outside the previous plan's path) and the summed signed j over the window (m); median failed / completed, difference [CI]; control = the same per-metre j 40-80 m before the entry (turns that reach back that far). (c) Executed - planned heading toward the turn over the next 1 s / 2 s (deg, per-turn median; planned = the plan's own heading at 1 / 2 s, executed = the car's yaw change), with the planned / executed medians.

| population | approach v f / c | j per m f / c | diff [CI] | summed j f / c (m) | diff [CI] | control j per m f / c (n) | exec - plan 1 s f / c | exec - plan 2 s f / c | diff 2 s [CI] | plan / exec 2 s f | plan / exec 2 s c |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| held_RSG-L18-s0 | 2.3 / 4.8 | -0.002 / +0.007 | -0.009 [-0.070, +0.053] | -0.35 / -0.10 | -0.26 [-1.04, +0.95] | - / - (0 / 0) | -0.2 / +0.3 | -1.3 / +0.4 | -1.6 [-4.2, +1.5] | +27.5 / +25.6 | +29.1 / +30.5 |
| held_RSG-L18-s1 | 3.4 / 4.0 | -0.011 / +0.009 | -0.020 [-0.055, +0.023] | -0.73 / -0.04 | -0.68 [-1.28, +0.30] | - / - (0 / 0) | -0.5 / +0.3 | -1.8 / +0.3 | -2.1 [-3.8, -0.1] | +26.8 / +25.8 | +31.1 / +31.6 |
| held_RV2a-L18-s0 | 3.3 / 4.0 | +0.004 / -0.007 | +0.011 [-0.122, +0.042] | -0.46 / -0.15 | -0.31 [-1.06, +0.89] | - / - (0 / 0) | -0.6 / +0.3 | -1.3 / +0.4 | -1.6 [-4.7, +0.6] | +27.7 / +24.5 | +29.0 / +29.2 |
| held_RV2a-L18-s1 | 4.4 / 3.6 | -0.040 / -0.009 | -0.031 [-0.064, +0.022] | -0.72 / -0.18 | -0.54 [-1.65, +0.20] | - / - (0 / 0) | -0.4 / +0.2 | -1.4 / +0.6 | -2.1 [-3.6, -0.0] | +27.3 / +24.7 | +30.1 / +31.2 |
| held_RV2b-L18-s0 | 2.3 / 4.8 | +0.019 / -0.015 | +0.034 [-0.067, +0.065] | +0.13 / -0.07 | +0.20 [-1.11, +1.23] | - / - (0 / 0) | -0.1 / +0.3 | -0.0 / +0.3 | -0.3 [-4.5, +1.9] | +24.9 / +25.2 | +30.3 / +30.9 |
| held_RV2b-L18-s1 | 2.2 / 4.4 | +0.005 / -0.009 | +0.013 [-0.084, +0.057] | -0.02 / +0.03 | -0.06 [-0.98, +0.96] | - / - (0 / 0) | -0.8 / +0.4 | -2.4 / +0.5 | -2.9 [-4.5, +2.0] | +26.8 / +23.1 | +30.0 / +30.3 |
| pai_APP1X-L18-s0 | 2.5 / 2.9 | -0.041 / -0.007 | -0.034 [-0.106, +0.017] | -1.47 / +0.01 | -1.48 [-3.01, +0.47] | +0.014 / +0.014 (11 / 3) | -0.4 / +0.1 | -1.0 / +0.7 | -1.7 [-4.0, -0.1] | +10.1 / +4.6 | +13.9 / +17.0 |
| pai_RSG-L18-s0 | 3.3 / 5.3 | -0.063 / -0.015 | -0.048 [-0.104, -0.016] | -2.10 / -0.40 | -1.70 [-4.86, -0.81] | +0.003 / -0.002 (14 / 1) | -0.7 / +0.2 | -2.5 / +1.6 | -4.1 [-6.6, -2.3] | +6.7 / +5.1 | +29.7 / +26.7 |
| pai_RV2a-L18-s0 | 3.2 / 3.4 | -0.056 / -0.034 | -0.022 [-0.046, +0.014] | -1.74 / -1.65 | -0.09 [-1.79, +2.50] | +0.004 / +0.006 (13 / 2) | -0.6 / -0.4 | -2.1 / +0.1 | -2.2 [-4.8, +0.8] | +11.3 / +9.4 | +21.6 / +21.8 |
| pai_RV2b-L18-s1 | 3.3 / 3.2 | -0.054 / -0.033 | -0.021 [-0.076, +0.035] | -1.80 / -0.74 | -1.06 [-3.51, +2.41] | +0.003 / +0.007 (13 / 2) | -0.7 / +0.2 | -2.7 / +1.8 | -4.5 [-6.1, +0.0] | +6.7 / +4.9 | +21.6 / +20.8 |
| held_RSG | 2.3 / 4.4 | -0.007 / +0.008 | -0.016 [-0.055, +0.038] | -0.49 / -0.04 | -0.45 [-1.12, +0.69] | - / - (0 / 0) | -0.4 / +0.3 | -1.8 / +0.4 | -2.1 [-3.9, +0.5] | +26.8 / +25.7 | +29.6 / +30.9 |
| held_all6 | 2.3 / 4.0 | +0.003 / -0.006 | +0.009 [-0.067, +0.040] | -0.37 / -0.06 | -0.31 [-0.99, +0.61] | - / - (0 / 0) | -0.4 / +0.3 | -1.6 / +0.4 | -2.0 [-4.0, +0.5] | +26.4 / +25.3 | +29.8 / +30.8 |
| pai_all4 | 3.2 / 3.3 | -0.055 / -0.021 | -0.034 [-0.069, -0.008] | -1.86 / -0.54 | -1.32 [-2.59, -0.19] | +0.003 / +0.008 (51 / 8) | -0.6 / +0.1 | -2.1 / +1.0 | -3.2 [-4.7, -1.4] | +8.5 / +4.9 | +20.9 / +20.9 |

(a) by class (vs completed of the same population): j per m failed / completed, diff [CI]; summed j; exec - plan 2 s (failed median):

| population | class | n | j per m | diff [CI] | summed j (m) | diff [CI] | exec - plan 2 s |
|:--|:--|:--|:--|:--|:--|:--|:--|
| pai_RSG-L18-s0 | late_wide | 12 | -0.133 / -0.015 | -0.117 [-0.156, -0.066] | -3.67 / -0.40 | -3.26 [-6.35, -1.46] | -2.7 |
| pai_RSG-L18-s0 | straight | 8 | -0.023 / -0.015 | -0.007 [-0.042, +0.025] | -0.50 / -0.40 | -0.10 [-3.26, +0.74] | -2.6 |
| pai_RSG-L18-s0 | over_turn | 2 | +0.140 / -0.015 | +0.156 [+0.046, +0.279] | +1.39 / -0.40 | +1.80 [-1.37, +3.55] | - |
| pai_RSG-L18-s0 | no_zero | 10 | -0.055 / -0.015 | -0.040 [-0.064, -0.007] | -2.18 / -0.40 | -1.78 [-4.94, -0.33] | -1.3 |
| pai_RSG-L18-s0 | other | 1 | -0.198 / -0.015 | -0.183 [-0.189, -0.150] | -1.78 / -0.40 | -1.37 [-4.54, -0.54] | - |
| pai_all4 | late_wide | 42 | -0.121 / -0.021 | -0.100 [-0.138, -0.034] | -3.28 / -0.54 | -2.73 [-3.92, -1.05] | -2.4 |
| pai_all4 | straight | 26 | -0.025 / -0.021 | -0.004 [-0.035, +0.025] | -0.59 / -0.54 | -0.05 [-1.64, +1.04] | -2.7 |
| pai_all4 | over_turn | 2 | +0.140 / -0.021 | +0.161 [+0.051, +0.279] | +1.39 / -0.54 | +1.93 [-0.03, +3.75] | - |
| pai_all4 | no_zero | 43 | -0.040 / -0.021 | -0.019 [-0.038, +0.009] | -1.56 / -0.54 | -1.02 [-2.29, +0.23] | -0.7 |
| pai_all4 | other | 5 | -0.163 / -0.021 | -0.142 [-0.191, -0.086] | -1.78 / -0.54 | -1.24 [-4.32, -0.13] | -4.9 |
| held_RSG | late_wide | 2 | -0.114 / +0.008 | -0.122 [-0.134, -0.104] | -1.08 / -0.04 | -1.04 [-1.43, -0.71] | -4.3 |
| held_RSG | over_turn | 9 | +0.024 / +0.008 | +0.016 [-0.045, +0.157] | +0.34 / -0.04 | +0.38 [-0.98, +0.81] | +1.1 |
| held_RSG | no_zero | 16 | -0.002 / +0.008 | -0.010 [-0.031, +0.055] | -0.35 / -0.04 | -0.31 [-0.84, +0.87] | -1.4 |
| held_all6 | late_wide | 6 | -0.110 / -0.006 | -0.104 [-0.127, -0.097] | -1.04 / -0.06 | -0.99 [-1.39, -0.69] | -4.0 |
| held_all6 | over_turn | 9 | +0.024 / -0.006 | +0.030 [-0.040, +0.163] | +0.34 / -0.06 | +0.40 [-0.90, +0.78] | +1.1 |
| held_all6 | no_zero | 42 | +0.005 / -0.006 | +0.010 [-0.021, +0.048] | -0.23 / -0.06 | -0.18 [-0.69, +0.79] | -1.1 |

(b) M4 composition, failed turns, 0-10 m before the entry (S r < 0.25, P, F r >= 0.6 as d307 M4): decision shares, the set of states each turn shows, and the share of the variance of r that lies between turns:

- pai_RSG-L18-s0: 647 decisions, 30 turns; shares S 0.09 / P 0.04 / F 0.86; state sets F 20, S 5, FP 3, FPS 1, PS 1; between-turn variance share 0.76
- pai_RV2a-L18-s0: 599 decisions, 30 turns; shares S 0.10 / P 0.07 / F 0.84; state sets F 18, FP 6, S 5, PS 1; between-turn variance share 0.76
- pai_RV2b-L18-s1: 541 decisions, 28 turns; shares S 0.12 / P 0.05 / F 0.84; state sets F 18, S 5, FP 3, FPS 1, PS 1; between-turn variance share 0.79
- pai_APP1X-L18-s0: 497 decisions, 20 turns; shares S 0.04 / P 0.10 / F 0.85; state sets F 13, FP 3, FPS 2, P 1, FS 1; between-turn variance share 0.50
- pai_all4: 2284 decisions, 108 turns; shares S 0.09 / P 0.06 / F 0.85; state sets F 69, FP 15, S 15, FPS 4, PS 3, P 1, FS 1; between-turn variance share 0.73
- held_RSG: 78 decisions, 18 turns; shares S 0.00 / P 0.00 / F 1.00; state sets F 18; between-turn variance share 0.53

## Per turn (pai_RSG-L18-s0)

| scene | side | class | win dec | sw pairs | switching | flip /s | abs j m | signed j | j per m | summed j | abs e m | signed e | mean r | p90 r | R_exec | exec - plan 2 s | human sw |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 22fa5c92 | L | late_wide | 82 | 54 | 0.000 | 0.00 | 0.015 | -0.012 | -0.155 | -2.06 | 0.03 | +0.03 | 1.25 | 1.88 | 1.17 | -3.8 | 0.000 |
| 33c8976b | L | late_wide | 51 | 50 | 0.000 | 0.00 | 0.060 | -0.037 | -0.189 | -1.64 | 0.05 | +0.05 | 1.16 | 1.68 | 1.39 | -9.5 | 0.000 |
| 37d660b1 | R | late_wide | 66 | 52 | 0.000 | 0.00 | 0.040 | -0.025 | -0.058 | -4.01 | 0.02 | +0.01 | 1.02 | 1.46 | 0.58 | -1.9 | 0.000 |
| 3ce13eed | R | late_wide | 85 | 67 | 0.000 | 0.00 | 0.049 | -0.037 | -0.129 | -3.99 | 0.04 | +0.03 | 0.90 | 1.15 | 0.60 | -4.5 | 0.000 |
| 68e53160 | L | late_wide | 109 | 68 | 0.000 | 0.00 | 0.027 | -0.021 | -0.114 | -3.74 | 0.02 | +0.02 | 1.12 | 1.44 | 0.77 | -1.3 | 0.000 |
| 94c1d722 | R | late_wide | 137 | 75 | 0.000 | 0.00 | 0.011 | -0.005 | -0.025 | -4.06 | 0.01 | +0.00 | 1.46 | 2.77 | 0.75 | -0.5 | 0.000 |
| 9f393c8a | R | late_wide | 49 | 48 | 0.000 | 0.00 | 0.101 | -0.083 | -0.164 | -2.07 | 0.03 | +0.01 | 1.04 | 1.51 | 0.72 | -11.8 | 0.000 |
| a1b166cc | L | late_wide | 116 | 67 | 0.000 | 0.00 | 0.016 | -0.012 | -0.121 | -3.59 | 0.02 | +0.02 | 1.22 | 1.41 | 0.88 | -2.5 | 0.000 |
| ce161092 | L | late_wide | 51 | 50 | 0.000 | 0.00 | 0.064 | -0.040 | -0.049 | -2.13 | 0.02 | +0.01 | 1.09 | 1.59 | 0.83 | -1.7 | 0.000 |
| ddc3e8df | R | late_wide | 61 | 47 | 0.000 | 0.00 | 0.093 | -0.093 | -0.248 | -5.06 | 0.06 | +0.06 | 1.49 | 1.78 | 0.98 | -11.2 | 0.000 |
| e37aa159 | L | late_wide | 54 | 53 | 0.000 | 0.00 | 0.084 | -0.084 | -0.136 | -4.84 | 0.01 | +0.01 | 1.03 | 1.48 | 0.70 | -2.7 | 0.000 |
| fd01e9ab | L | late_wide | 78 | 56 | 0.000 | 0.00 | 0.031 | -0.030 | -0.179 | -3.34 | 0.02 | +0.02 | 1.16 | 1.94 | 0.58 | -2.8 | 0.000 |
| 09528162 | R | no_zero | 136 | 93 | 0.000 | 0.21 | 0.057 | -0.044 | -0.163 | -7.06 | 0.04 | +0.02 | 1.20 | 1.82 | 0.93 | -8.6 | 0.000 |
| 268949d2 | R | no_zero | 100 | 55 | 0.000 | 0.00 | 0.014 | -0.005 | -0.040 | -2.21 | 0.01 | -0.01 | 1.14 | 1.66 | 0.92 | +0.1 | 0.000 |
| 6fd8cf30 | R | no_zero | 97 | 42 | 0.000 | 0.00 | 0.010 | -0.008 | -0.036 | -1.64 | 0.01 | +0.01 | 1.32 | 2.01 | 0.91 | -0.4 | 0.000 |
| a4d2f7cf | L | no_zero | 91 | 61 | 0.000 | 0.00 | 0.029 | -0.015 | -0.041 | -1.56 | 0.01 | +0.01 | 0.98 | 1.31 | 0.69 | -3.1 | 0.000 |
| abd45a30 | L | no_zero | 39 | 11 | 0.000 | 0.00 | 0.013 | -0.006 | -0.019 | -0.20 | 0.00 | +0.00 | 0.89 | 1.15 | - | -1.2 | 0.000 |
| b40f4729 | R | no_zero | 67 | 49 | 0.000 | 0.00 | 0.075 | -0.027 | -0.083 | -0.40 | 0.03 | +0.03 | 0.85 | 1.36 | 0.86 | -0.1 | 0.000 |
| c4a58f83 | R | no_zero | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| e0f6ebe1 | R | no_zero | 52 | 39 | 0.000 | 0.00 | 0.108 | -0.044 | -0.055 | -2.18 | 0.02 | +0.01 | 0.96 | 1.34 | 0.92 | -1.7 | 0.000 |
| e848c843 | R | no_zero | 117 | 55 | 0.000 | 0.00 | 0.031 | -0.022 | -0.073 | -4.12 | 0.02 | +0.02 | 1.22 | 1.77 | 0.95 | -2.8 | 0.000 |
| ea468790 | R | no_zero | 153 | 72 | 0.000 | 0.00 | 0.014 | -0.007 | -0.068 | -4.40 | 0.02 | +0.02 | 1.26 | 1.49 | 0.94 | -1.3 | 0.000 |
| 1e8e939e | R | other | 10 | 9 | 0.000 | 0.00 | 0.235 | -0.235 | -0.198 | -1.78 | 0.05 | -0.05 | 1.00 | 1.06 | - | - | 0.000 |
| 2f227f2c | R | straight | 24 | 23 | 0.000 | 0.00 | 0.030 | -0.030 | -0.031 | -0.66 | 0.01 | -0.01 | 0.06 | 0.09 | - | -0.8 | 0.000 |
| 302c5c99 | L | straight | 42 | 41 | 0.000 | 0.00 | 0.030 | -0.028 | -0.055 | -1.75 | 0.01 | -0.01 | 0.29 | 0.50 | -0.22 | -7.4 | 0.000 |
| 3232fdb7 | R | straight | 66 | 18 | 0.000 | 0.00 | 0.009 | +0.000 | +0.001 | +0.07 | 0.00 | -0.00 | 0.65 | 0.80 | - | -0.9 | 0.000 |
| 403d558a | R | straight | 25 | 24 | 0.000 | 0.00 | 0.049 | -0.049 | -0.048 | -1.09 | 0.01 | -0.01 | 0.12 | 0.21 | 0.02 | -2.6 | 0.000 |
| a174faed | R | straight | 44 | 43 | 0.000 | 0.23 | 0.068 | -0.040 | -0.107 | -3.52 | 0.01 | -0.01 | 0.75 | 1.35 | 0.06 | -4.5 | 0.000 |
| e5e47fd2 | L | straight | 20 | 19 | 0.000 | 0.00 | 0.017 | -0.007 | -0.006 | -0.09 | 0.01 | -0.01 | -0.00 | -0.00 | -0.02 | - | 0.000 |
| eaba3f6a | R | straight | 20 | 19 | 0.000 | 0.00 | 0.018 | -0.017 | -0.014 | -0.30 | 0.01 | -0.01 | 0.02 | 0.04 | -0.08 | - | 0.000 |
| f9e361aa | R | straight | 22 | 21 | 0.000 | 0.00 | 0.009 | -0.009 | -0.009 | -0.34 | 0.00 | -0.00 | -0.00 | 0.00 | -0.03 | - | 0.000 |
| 132ae6cb | L | completed | 69 | 59 | 0.000 | 0.00 | 0.058 | -0.005 | -0.011 | -0.04 | 0.02 | +0.01 | 1.01 | 1.53 | 0.88 | +1.0 | 0.000 |
| 382a7247 | L | completed | 52 | 49 | 0.000 | 0.00 | 0.147 | -0.012 | -0.019 | +2.76 | 0.03 | +0.03 | 0.93 | 1.41 | 1.00 | +4.2 | 0.000 |
| 3c2822d9 | L | completed | 41 | 23 | 0.000 | 0.00 | 0.072 | -0.031 | -0.048 | -0.83 | 0.06 | -0.05 | 1.04 | 1.33 | 1.00 | +0.1 | 0.000 |
| 8a143cb2 | R | completed | 55 | 35 | 0.000 | 0.00 | 0.035 | -0.005 | -0.009 | -0.40 | 0.04 | +0.04 | 0.96 | 1.68 | 0.76 | +2.6 | 0.000 |
| d90fe123 | L | completed | 50 | 49 | 0.000 | 0.00 | 0.053 | -0.011 | -0.015 | -1.23 | 0.03 | +0.02 | 1.02 | 1.42 | 0.91 | +1.6 | 0.000 |
