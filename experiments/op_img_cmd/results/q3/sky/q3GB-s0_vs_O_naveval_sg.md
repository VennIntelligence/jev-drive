# image-command fine-tune Q3: q3GB-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.52 [0.49, 0.55] (b 0.51, n 234.0) | +0.00 [-0.01, +0.02] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sg | 384 | +1.35 | +1.35 [+0.94, +1.83] | 0.22 [0.17, 0.27] (b -0.01) | +0.23 [+0.18, +0.28] | 0.66 [0.62, 0.71] (b 0.51, n 234.0) | +0.15 [+0.10, +0.20] | +0.15 / +0.34 / +1.47 | -0.30 | +1.00 [+0.80, +1.20] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 293 | +0.00 [-0.00, +0.01] | +0.01 | -0.00 [-0.01, +0.00] |

Guards: {"none_drift_median_junction": 0.10782124847173691, "none_drift_median_straight": 0.07767631113529205, "sg_wrong_move_median": 0.24218265401552114, "sg_wrong_abs_dy4_mean": 0.9514453061273054, "sg_wrong_move_median_ref": 0.2612884469846402, "sg_wrong_abs_dy4_mean_ref": 0.3043523218799173, "sg_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": 0.00384070115484278, "lo": -0.0309580305075234, "hi": 0.03905603896829327, "mean_a": 1.3326458314794758, "mean_b": 1.3288051303246329}, "straight_none_lat_err_3s": {"n": 293, "mean": -9.651255481757325e-05, "lo": -0.004899437796751387, "hi": 0.004831506523087369, "mean_a": 0.1606792554132538, "mean_b": 0.1607757679680714}}
