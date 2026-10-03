# image-command fine-tune Q3: q3SA-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.47, 0.53] (b 0.51, n 246.0) | -0.01 [-0.03, -0.00] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 384 | +1.72 | +1.73 [+1.29, +2.22] | 0.29 [0.23, 0.34] (b -0.00) | +0.29 [+0.24, +0.34] | 0.71 [0.67, 0.76] (b 0.50, n 246.0) | +0.21 [+0.16, +0.27] | +0.67 / +0.20 / +1.68 | -0.45 | +0.74 [+0.56, +0.93] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 293 | +0.01 [+0.00, +0.02] | +0.00 | +0.00 [-0.00, +0.01] |
| straight:sky_disc | 293 | +0.02 [+0.01, +0.04] | +0.00 | +0.02 [+0.01, +0.03] |

Guards: {"none_drift_median_junction": 0.13210073113441467, "none_drift_median_straight": 0.07184960693120956, "sky_disc_move_median": 0.13834147895328736, "sky_disc_abs_dy4_mean": 0.43648765698458264, "sky_disc_move_median_ref": 0.17344454004797757, "sky_disc_abs_dy4_mean_ref": 0.2044576100881637, "sky_disc_n": 385, "sky_wrong_move_median": 0.17810327812072604, "sky_wrong_abs_dy4_mean": 0.9995365933737597, "sky_wrong_move_median_ref": 0.139451117697597, "sky_wrong_abs_dy4_mean_ref": 0.21610442541656485, "sky_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": -0.03335192827444608, "lo": -0.07977881744864994, "hi": 0.013512891922290552, "mean_a": 1.2955627121062776, "mean_b": 1.3289146403807235}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.005609484506235536, "lo": -0.002429440566581686, "hi": 0.01386075437823854, "mean_a": 0.1663717538029091, "mean_b": 0.1607622692966736}}
