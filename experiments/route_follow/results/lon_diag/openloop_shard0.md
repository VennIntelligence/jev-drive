# LON-DIAG reading 3: open-loop counterfactuals (RSG-L18-s0)

Prereg: `experiments/route_follow/plans/2026-10-15-longitudinal-diag-prereg.md (读数 3)`. Script: `experiments/route_follow/scripts/rf_lonol.py` (docstring = definitions). Store: navtrain_full shard(s) [0] (2134 offline rows, rm_train.build as rf_lonread.py). CIs: log-clustered percentile bootstrap, B 10 000, seed 0.

## Gate

Edits off vs rm_train's dev evaluation (rows_ade, memory on): 1789 dev rows, per-row ADE bit-for-bit equal: **True** (max |diff| 0, dev ADE 0.528447); a repeated pass equal bit for bit: True; x 1.0 scale path = measured row bit for bit: True (max 0 m); F touches only the speed columns and the history yaws: True.

## Cells

| cell | rows | logs |
|:--|--:|--:|
| free | 1682 | 582 |
| red | 79 | 61 |
| green | 15 | 15 |
| lead | 179 | 132 |
| red_all | 112 | 87 |
| green_all | 32 | 29 |
| lead_all | 317 | 200 |
| red_nolead | 24 | 20 |

## (i) History gain (free straight rows)

gamma = fixed-effects OLS slope of v_p on the scaled v_f over x 0.6 .. 1.4 (pooled within-row sums).

| human speed | rows | logs | gamma [95 % CI] | median per-row slope | median v_p / v (measured speed) at x 0.6 / 0.8 / 1.0 / 1.2 / 1.4 |
|:--|--:|--:|:--|--:|:--|
| all | 1682 | 582 | 0.171 [0.167, 0.174] | 0.176 |  |
| < 8 | 803 | 404 | 0.216 [0.209, 0.224] | 0.219 | 0.904 / 0.958 / 1.005 / 1.048 / 1.086 |
| 8-12 | 736 | 346 | 0.163 [0.160, 0.166] | 0.167 | 0.890 / 0.932 / 0.969 / 1.000 / 1.023 |
| 12-16 | 142 | 83 | 0.137 [0.134, 0.140] | 0.137 | 0.887 / 0.927 / 0.958 / 0.982 / 0.997 |
| >= 16 | 0 | | | | |

Line (i) (gamma >= 0.6 history-led; <= 0.3 frame-led; between: mixed): gamma 0.171 -> **frame-led**.

## (ii) Brake source: b = (v_f - v_p) / v_f under H (own history) vs F (flat history)

| cell | rows | logs | median b_H | median b_F | b_F / b_H (ratio of medians) | row-wise median ratio (b_H > 0.05, rows) | v_f | ax | line / lead m |
|:--|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| red | 79 | 61 | 0.193 [0.183, 0.203] | 0.112 [0.107, 0.118] | 0.580 [0.552, 0.612] | 0.595 (79) | 6.7 | -1.05 | 32.4 |
| green | 15 | 15 | 0.194 [0.151, 0.225] | 0.087 [0.064, 0.126] | 0.447 [0.379, 0.637] | 0.511 (15) | 6.7 | -1.00 | 25.2 |
| lead | 179 | 132 | 0.177 [0.168, 0.184] | 0.098 [0.095, 0.103] | 0.555 [0.535, 0.586] | 0.555 (179) | 7.3 | -1.05 | 18.1 |
| red_all | 112 | 87 | 0.169 [0.156, 0.183] | 0.107 [0.101, 0.112] | 0.634 [0.587, 0.682] | 0.616 (105) | 6.7 | -0.92 | 32.5 |
| green_all | 32 | 29 | 0.112 [0.078, 0.149] | 0.072 [0.055, 0.097] | 0.643 [0.463, 0.969] | 0.577 (22) | 7.3 | -0.35 | 25.5 |
| lead_all | 317 | 200 | 0.141 [0.124, 0.150] | 0.086 [0.079, 0.090] | 0.606 [0.572, 0.671] | 0.609 (279) | 7.5 | -0.67 | 18.1 |
| red_nolead | 24 | 20 | 0.175 [0.158, 0.195] | 0.106 [0.103, 0.112] | 0.609 [0.558, 0.677] | 0.609 (24) | 7.3 | -0.97 | 32.9 |

Line (ii) red (median b_F / b_H < 0.3 reads stop from history; 0.3-0.7 partial; >= 0.7 reads the light): ratio 0.580 -> **partial**.
Positive control (lead ratio >= 0.7 and red < 0.3 -> the light / line channel is what is missing): lead 0.555, red 0.580 -> **not met**.
Negative control (negative control: b under H near 0 on green lines): b_H green 0.194 vs red 0.193.

## (iii) Cap (free rows, H)

| human speed | rows | logs | median v_p / v_f [95 % CI] |
|:--|--:|--:|:--|
| < 8 | 803 | 404 | 1.005 [0.996, 1.016] |
| 8-12 | 736 | 346 | 0.969 [0.966, 0.972] |
| 12-16 | 142 | 83 | 0.958 [0.955, 0.964] |
| >= 16 | 1 |  | n/a |

Line (iii) (>= 16 m/s bin median v_p / v_f < 0.85 -> distribution cap): nan -> **no cap**.

Training store row speed (m/s, the cache's t0 speed):

| rows | n | p50 | p90 | p99 | max | share >= 16 |
|:--|--:|--:|--:|--:|--:|--:|
| train rows (12 shards) | 101499 | 4.44 | 9.28 | 12.79 | 17.11 | 0.0001 |
| all rows (12 shards) | 103288 | 4.43 | 9.27 | 12.78 | 17.11 | 0.0001 |
| shard 0 (this read) | 8608 | 4.39 | 9.24 | 12.73 | 16.80 | 0.0001 |

Figures: `figs/lon_diag_history_gain.png` (median v_p against the scaled v_f per speed bin, the diagonal = plan follows the history one for one; look at the slope against the diagonal), `figs/lon_diag_red_brake.png` (b under H and F per cell; look at whether the red and lead bars keep their H height under F).
