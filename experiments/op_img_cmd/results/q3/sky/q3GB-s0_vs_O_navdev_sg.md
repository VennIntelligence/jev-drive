# image-command fine-tune Q3: q3GB-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 109.0) | +0.00 [-0.03, +0.03] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sg | 136 | +2.13 | +2.25 [+0.86, +3.86] | 0.27 [0.13, 0.36] (b -0.01) | +0.28 [+0.15, +0.37] | 0.69 [0.58, 0.77] (b 0.50, n 109.0) | +0.19 [+0.09, +0.27] | -0.02 / +0.12 / +2.92 | -0.60 | +0.24 [-0.25, +0.73] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 30 | -0.02 [-0.04, +0.00] | -0.01 | -0.01 [-0.03, +0.01] |

Guards: {"none_drift_median_junction": 0.10614694654941559, "none_drift_median_straight": 0.08405830711126328, "sg_wrong_move_median": 0.24270738339485043, "sg_wrong_abs_dy4_mean": 0.8406466963447558, "sg_wrong_move_median_ref": 0.2186037774391925, "sg_wrong_abs_dy4_mean_ref": 0.23093070915422625, "sg_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": 0.01465560964411915, "lo": -0.05831789069206427, "hi": 0.10415459536499043, "mean_a": 2.3766200694230197, "mean_b": 2.3619644597789002}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.03760915770689572, "lo": 0.009015010138380151, "hi": 0.08180768687637098, "mean_a": 0.29614210933368906, "mean_b": 0.2585329516267933}}
