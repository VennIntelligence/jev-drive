# image-command fine-tune Q3: O vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.51 [0.49, 0.54] (b 0.51, n 237.0) | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| sg | 384 | +0.00 | +0.00 [+0.00, +0.00] | -0.01 [-0.02, 0.01] (b -0.01) | +0.00 [+0.00, +0.00] | 0.51 [0.48, 0.55] (b 0.51, n 237.0) | +0.00 [+0.00, +0.00] | +0.00 / -0.12 / +0.12 | -1.30 | +0.00 [+0.00, +0.00] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 293 | +0.01 [-0.00, +0.02] | +0.01 | +0.00 [+0.00, +0.00] |

Guards: {"none_drift_median_junction": 0.0, "none_drift_median_straight": 0.0, "sg_wrong_move_median": 0.2612884469846402, "sg_wrong_abs_dy4_mean": 0.3043523218799173, "sg_wrong_move_median_ref": 0.2612884469846402, "sg_wrong_abs_dy4_mean_ref": 0.3043523218799173, "sg_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": 0.0, "lo": 0.0, "hi": 0.0, "mean_a": 1.3288051303246329, "mean_b": 1.3288051303246329}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.0, "lo": 0.0, "hi": 0.0, "mean_a": 0.1607757679680714, "mean_b": 0.1607757679680714}}
