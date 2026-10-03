# image-command fine-tune Q3: q3NB-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.51 [0.48, 0.53] (b 0.51, n 229.0) | -0.01 [-0.03, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 384 | +0.44 | +0.45 [+0.31, +0.62] | 0.07 [0.05, 0.11] (b -0.00) | +0.08 [+0.05, +0.11] | 0.56 [0.53, 0.60] (b 0.50, n 229.0) | +0.07 [+0.03, +0.10] | +0.10 / +0.09 / +0.49 | -0.24 | +0.95 [+0.78, +1.13] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 293 | -0.00 [-0.00, +0.00] | +0.00 | -0.01 [-0.01, -0.00] |
| straight:sky_disc | 293 | -0.00 [-0.00, +0.00] | +0.00 | -0.00 [-0.01, -0.00] |

Guards: {"none_drift_median_junction": 0.09646286815404892, "none_drift_median_straight": 0.058873679488897324, "sky_disc_move_median": 0.07782765730646725, "sky_disc_abs_dy4_mean": 0.10573647665763816, "sky_disc_move_median_ref": 0.17344454004797757, "sky_disc_abs_dy4_mean_ref": 0.2044576100881637, "sky_disc_n": 385, "sky_wrong_move_median": 0.1000074311755061, "sky_wrong_abs_dy4_mean": 0.37831159697489264, "sky_wrong_move_median_ref": 0.139451117697597, "sky_wrong_abs_dy4_mean_ref": 0.21610442541656485, "sky_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": 0.015997615243307322, "lo": -0.016946438716359657, "hi": 0.05170560686087215, "mean_a": 1.344912255624031, "mean_b": 1.3289146403807235}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.00036623438128181667, "lo": -0.004421646291018477, "hi": 0.0050748578842527954, "mean_a": 0.1611285036779554, "mean_b": 0.1607622692966736}}
