# LON-DIAG reading 3: open-loop counterfactuals (RSG-L18-s0)

Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md (读数 3)`. Script: `experiments/route_follow/scripts/rf_lonol.py` (docstring = definitions). Store: navtrain_full shard(s) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (26421 offline rows, rm_train.build as rf_lonread.py). CIs: log-clustered percentile bootstrap, B 10 000, seed 0.

## Gate

Edits off vs rm_train's dev evaluation (rows_ade, memory on): 1789 dev rows, per-row ADE bit-for-bit equal: **True** (max |diff| 0, dev ADE 0.528447); a repeated pass equal bit for bit: True; x 1.0 scale path = measured row bit for bit: True (max 0 m); F touches only the speed columns and the history yaws: True.

## Cells

| cell | rows | logs |
|:--|--:|--:|
| free | 20889 | 1010 |
| red | 960 | 224 |
| green | 177 | 70 |
| lead | 2388 | 415 |
| red_all | 1286 | 253 |
| green_all | 553 | 136 |
| lead_all | 3838 | 497 |
| red_nolead | 262 | 84 |

## (i) History gain (free straight rows)

gamma = fixed-effects OLS slope of v_p on the scaled v_f over x 0.6 .. 1.4 (pooled within-row sums).

| human speed | rows | logs | gamma [95 % CI] | median per-row slope | median v_p / v (measured speed) at x 0.6 / 0.8 / 1.0 / 1.2 / 1.4 |
|:--|--:|--:|:--|--:|:--|
| all | 20889 | 1010 | 0.172 [0.170, 0.175] | 0.176 |  |
| < 8 | 9949 | 889 | 0.222 [0.218, 0.226] | 0.219 | 0.906 / 0.957 / 1.005 / 1.050 / 1.090 |
| 8-12 | 9197 | 719 | 0.164 [0.162, 0.165] | 0.167 | 0.889 / 0.930 / 0.967 / 0.998 / 1.021 |
| 12-16 | 1735 | 240 | 0.137 [0.135, 0.139] | 0.137 | 0.891 / 0.931 / 0.964 / 0.987 / 1.001 |
| >= 16 | 8 | 1 | 0.081 [0.081, 0.081] | 0.080 | 0.897 / 0.931 / 0.952 / 0.961 / 0.963 |

Line (i) (gamma >= 0.6 history-led; <= 0.3 frame-led; between: mixed): gamma 0.172 -> **frame-led**.

## (ii) Brake source: b = (v_f - v_p) / v_f under H (own history) vs F (flat history)

| cell | rows | logs | median b_H | median b_F | b_F / b_H (ratio of medians) | row-wise median ratio (b_H > 0.05, rows) | v_f | ax | line / lead m |
|:--|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| red | 960 | 224 | 0.192 [0.188, 0.197] | 0.114 [0.112, 0.118] | 0.597 [0.587, 0.615] | 0.604 (960) | 6.9 | -1.04 | 32.2 |
| green | 177 | 70 | 0.181 [0.170, 0.191] | 0.089 [0.078, 0.100] | 0.489 [0.442, 0.537] | 0.508 (177) | 7.3 | -1.10 | 28.7 |
| lead | 2388 | 415 | 0.177 [0.173, 0.180] | 0.100 [0.097, 0.102] | 0.564 [0.553, 0.572] | 0.556 (2384) | 7.4 | -1.04 | 18.2 |
| red_all | 1286 | 253 | 0.174 [0.168, 0.180] | 0.108 [0.105, 0.111] | 0.623 [0.607, 0.638] | 0.629 (1234) | 6.8 | -0.91 | 32.7 |
| green_all | 553 | 136 | 0.064 [0.044, 0.083] | 0.054 [0.040, 0.061] | 0.831 [0.685, 1.090] | 0.570 (304) | 7.9 | -0.18 | 28.3 |
| lead_all | 3838 | 497 | 0.145 [0.140, 0.149] | 0.086 [0.083, 0.089] | 0.594 [0.581, 0.608] | 0.596 (3444) | 7.5 | -0.75 | 18.1 |
| red_nolead | 262 | 84 | 0.185 [0.177, 0.196] | 0.112 [0.107, 0.119] | 0.606 [0.584, 0.628] | 0.605 (262) | 7.2 | -1.04 | 33.0 |

Line (ii) red (median b_F / b_H < 0.3 reads stop from history; 0.3-0.7 partial; >= 0.7 reads the light): ratio 0.597 -> **partial**.
Positive control (lead ratio >= 0.7 and red < 0.3 -> the light / line channel is what is missing): lead 0.564, red 0.597 -> **not met**.
Negative control (negative control: b under H near 0 on green lines): b_H green 0.181 vs red 0.192.

## (iii) Cap (free rows, H)

| human speed | rows | logs | median v_p / v_f [95 % CI] |
|:--|--:|--:|:--|
| < 8 | 9949 | 889 | 1.005 [1.000, 1.011] |
| 8-12 | 9197 | 719 | 0.967 [0.965, 0.969] |
| 12-16 | 1735 | 240 | 0.964 [0.961, 0.966] |
| >= 16 | 8 | 1 | 0.952 [0.952, 0.952] |

Line (iii) (>= 16 m/s bin median v_p / v_f < 0.85 -> distribution cap): 0.952 -> **no cap**.

Training store row speed (m/s, the cache's t0 speed):

| rows | n | p50 | p90 | p99 | max | share >= 16 |
|:--|--:|--:|--:|--:|--:|--:|
| train rows (12 shards) | 101499 | 4.44 | 9.28 | 12.79 | 17.11 | 0.0001 |
| all rows (12 shards) | 103288 | 4.43 | 9.27 | 12.78 | 17.11 | 0.0001 |
| shard 0 (this read) | 8608 | 4.39 | 9.24 | 12.73 | 16.80 | 0.0001 |

Figures: `figs/lon_diag_history_gain.png` (median v_p against the scaled v_f per speed bin, the diagonal = plan follows the history one for one; look at the slope against the diagonal), `figs/lon_diag_red_brake.png` (b under H and F per cell; look at whether the red and lead bars keep their H height under F).

## Notes (hand-written after the read)

- The edit changes the numeric speed history only; the 8-slot frame stack still shows the logged motion (a decelerating human is visible
  as deceleration in the video). gamma and b_F therefore measure the numeric history against everything in the frames, own-motion cues
  included, not against the light or the lead alone. The closed-loop trend coefficient of d294 (0.71-0.86) is not this quantity: there
  the frames and the history move together.
- Negative control fails as a control: with the same deceleration condition, green rows brake as much as red rows under H (0.181 vs
  0.192) and nearly as much under F (0.089 vs 0.114). Without the ax condition green brakes little (0.064 under H). The brake follows the
  human's deceleration (history and frames), not the light state; the light adds at most the small red - green gap under F.
- (iii) cannot be judged at >= 16 m/s: the training store has 8 rows at >= 16 m/s out of 101 499 train rows (p99 12.8 m/s, max 17.1);
  the free >= 16 bin holds those 8 rows from 1 log (shard 0: 1 row). Up to 16 m/s there is no cap (12-16 m/s: 0.964).
- Sensitivity: an input change of one float32 ulp (x * 10 / 10 in the first launch) moved plans by up to 4.8 cm; every number here comes
  from deterministic kernels with the x 1.0 path bit-equal to the measured row.
