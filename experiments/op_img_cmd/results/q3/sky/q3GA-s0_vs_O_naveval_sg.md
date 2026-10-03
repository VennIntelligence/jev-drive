# image-command fine-tune Q3: q3GA-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.51 [0.48, 0.54] (b 0.51, n 240.0) | -0.01 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sg | 384 | +1.97 | +1.97 [+1.49, +2.50] | 0.33 [0.27, 0.38] (b -0.01) | +0.33 [+0.28, +0.39] | 0.82 [0.77, 0.86] (b 0.51, n 240.0) | +0.31 [+0.25, +0.36] | +0.74 / +0.57 / +1.62 | -0.31 | +0.99 [+0.79, +1.19] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 293 | +0.01 [-0.00, +0.01] | +0.01 | -0.00 [-0.01, +0.00] |

Guards: {"none_drift_median_junction": 0.14943866431713104, "none_drift_median_straight": 0.09109116345643997, "sg_wrong_move_median": 0.2968929495563509, "sg_wrong_abs_dy4_mean": 1.3295980572644563, "sg_wrong_move_median_ref": 0.2612884469846402, "sg_wrong_abs_dy4_mean_ref": 0.3043523218799173, "sg_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": -0.008387409038112845, "lo": -0.07468366529072566, "hi": 0.051453450535926856, "mean_a": 1.3204177212865198, "mean_b": 1.3288051303246329}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.0017487660893414452, "lo": -0.005422646600755226, "hi": 0.009250020312824823, "mean_a": 0.16252453405741285, "mean_b": 0.1607757679680714}}
