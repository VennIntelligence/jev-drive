# image-command fine-tune Q3: q3SB-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.51 [0.49, 0.54] (b 0.51, n 237.0) | +0.00 [-0.01, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 384 | +0.78 | +0.79 [+0.52, +1.10] | 0.13 [0.09, 0.17] (b -0.00) | +0.14 [+0.10, +0.18] | 0.59 [0.56, 0.63] (b 0.50, n 237.0) | +0.09 [+0.06, +0.13] | +0.09 / +0.08 / +0.97 | -0.30 | +0.89 [+0.72, +1.08] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 293 | -0.00 [-0.00, +0.00] | +0.00 | -0.00 [-0.01, -0.00] |
| straight:sky_disc | 293 | -0.00 [-0.00, +0.00] | +0.00 | -0.00 [-0.01, -0.00] |

Guards: {"none_drift_median_junction": 0.08820599317550659, "none_drift_median_straight": 0.05401163548231125, "sky_disc_move_median": 0.08968617320058901, "sky_disc_abs_dy4_mean": 0.28340488766347605, "sky_disc_move_median_ref": 0.17344454004797757, "sky_disc_abs_dy4_mean_ref": 0.2044576100881637, "sky_disc_n": 385, "sky_wrong_move_median": 0.09902847337170595, "sky_wrong_abs_dy4_mean": 0.5774431687004773, "sky_wrong_move_median_ref": 0.139451117697597, "sky_wrong_abs_dy4_mean_ref": 0.21610442541656485, "sky_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": 0.012297425959305972, "lo": -0.01723831678783523, "hi": 0.041690509535932965, "mean_a": 1.3412120663400295, "mean_b": 1.3289146403807235}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.0005624018465317852, "lo": -0.0034522758465081295, "hi": 0.004613212323238613, "mean_a": 0.1613246711432054, "mean_b": 0.1607622692966736}}
