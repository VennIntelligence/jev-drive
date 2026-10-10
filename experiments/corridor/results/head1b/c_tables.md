**EXPLORATORY (HEAD1b: second attempt after the missed registered gate G3, user-authorised; not a registered read).**

`c`: navtest 12146 tokens, per-token seed means x 100; differences to S = P2H10S-P-s0, P2H10S-P-s1 (88.16; per seed 88.08 / 88.24) with 95% CI (log-cluster paired bootstrap, B 4000). Lines: memory.

| arm | tags | EPDMS - base | per seed | - shuffled | memory masked - on (seeds) | dev ADE on / masked / mismatched, per seed (m) | channel read | verdict |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| SHP | P2H10S-HP-P-s0, P2H10S-HP-P-s1 | +0.18 [-0.00, +0.38] | +0.20 / +0.15 | - | -0.17 [-0.41, +0.09] (2) | 0.579 / 0.598 / 0.615 ; 0.572 / 0.595 / 0.624 | yes | negative |

Baseline dev ADE (m): 0.588 ; 0.601

**EPDMS by logged heading change over 4 s (difference to the baseline).**

| arm / contrast | < 5 deg (straight) | 5-20 deg | 20-45 deg | > 45 deg | > 20 deg |
|:--|:--|:--|:--|:--|:--|
| SHP - S | -0.05 [-0.18, +0.07] | +0.25 [-0.08, +0.60] | +0.90 [+0.06, +1.74] | +0.23 [-0.31, +0.79] | +0.58 [+0.05, +1.11] |
| SHP masked - SHP | -0.00 [-0.15, +0.15] | -0.27 [-0.78, +0.24] | -0.85 [-1.79, +0.26] | +0.04 [-0.77, +0.82] | - |
| SHP masked - S | -0.05 [-0.19, +0.08] | -0.02 [-0.43, +0.47] | +0.06 [-0.73, +0.84] | +0.27 [-0.51, +1.02] | - |

**Turn failures at > 45 deg (1 517 tokens; %, difference in pp), DAC failures, sub-scores (difference to the baseline).**

| arm | cut inside: arm / base, diff | cannot make the turn: arm / base, diff | DAC fail %, > 45 | DAC fail %, > 20 | NC | DAC | EP | TTC | NC + TTC fail % |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| SHP | 3.89 / 3.99, -0.10 [-0.61, +0.36] | 3.46 / 3.96, -0.49 [-1.24, +0.19] | -0.10 [-0.74, +0.53] | -0.54 [-1.04, +0.00] | -0.05 [-0.12, +0.01] | +0.15 [-0.02, +0.33] | +0.14 [+0.10, +0.18] | +0.01 [-0.07, +0.10] | +0.01 [-0.08, +0.10] |

**4 s arc length of the plan and the plan's own 4 s heading error (RMS, deg; token x seed pooled; before = baseline, after = arm).**

| arm | arc ratio to base | arm / log, base / log | straight / > 45 deg ratio | heading error all: before, after, diff | > 20 deg | > 45 deg |
|:--|:--|:--|:--|:--|:--|:--|
| SHP | 1.0011 [1.0005, 1.0016] | 0.9983, 0.9972 | 1.0005 / 1.0038 | 6.59 [5.70, 7.46], 6.05 [5.19, 6.88], -0.53 [-0.70, -0.37] | 11.20 [9.87, 12.49], 10.10 [8.81, 11.31], -1.10 [-1.47, -0.76] | 13.47 [11.82, 14.91], 11.80 [10.13, 13.22], -1.67 [-2.25, -1.18] |

**Decision 244's widening measures, navtest tokens over 45 deg, own plan.**

| arm | W2 per seed (plan > 2 m outside the logged path) | baseline W2 | signed lateral at 4 s, arm - base (m, + = outside) | arm / base mean (m) |
|:--|:--|:--|:--|:--|
| SHP | 99 / 120 | 141 / 150 | -0.117 [-0.158, -0.079] | +0.342 / +0.458 |

**The fed heading profile's realised quality (error to the logged path; pilot train rows = navsim/op-parity-full-train@v1:fc6eff272683 against navtest).** Profiles from /root/autodl-tmp/ujs/runs/corridor/head1/final/prof.

| rows | bucket | n | heading error at the logged 4 s arc, RMS deg | polyline position error at the logged 4 s arc, RMS m | heading error over the grid 2.5-40 m, RMS deg |
|:--|:--|--:|:--|:--|--:|
| pilot_train_rows (16937 rows, 1120 logs) | all | 16937 | 4.31 [3.69, 5.01] | 0.59 [0.53, 0.66] | 6.98 |
| pilot_train_rows (16937 rows, 1120 logs) | > 20 deg | 4720 | 6.82 [5.67, 8.25] | 0.94 [0.82, 1.07] | 8.48 |
| pilot_train_rows (16937 rows, 1120 logs) | > 45 deg | 1845 | 8.67 [6.64, 11.30] | 1.18 [0.99, 1.42] | 10.30 |
| navtest (12146 rows, 136 logs) | all | 12146 | 3.99 [3.29, 4.62] | 0.63 [0.54, 0.72] | 8.91 |
| navtest (12146 rows, 136 logs) | > 20 deg | 3154 | 6.65 [5.35, 7.84] | 1.03 [0.87, 1.23] | 9.40 |
| navtest (12146 rows, 136 logs) | > 45 deg | 1517 | 7.71 [5.98, 9.24] | 1.24 [0.99, 1.52] | 10.01 |

Train rows / navtest, heading RMS: all 1.08, > 20 deg 1.03, > 45 deg 1.12 (decision 204's QH head: ADE 0.55 m on its pilot train rows against 0.76 m on navtest, ratio 0.72: an in-sample head).
