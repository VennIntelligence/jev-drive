# image-command fine-tune Q3: q3GA-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.49 [0.45, 0.52] (b 0.50, n 110.0) | -0.00 [-0.03, +0.02] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sg | 136 | +2.78 | +2.91 [+1.32, +4.75] | 0.35 [0.21, 0.45] (b -0.01) | +0.37 [+0.23, +0.46] | 0.75 [0.63, 0.84] (b 0.50, n 110.0) | +0.25 [+0.14, +0.34] | +0.37 / +0.31 / +3.36 | -0.64 | +0.20 [-0.30, +0.71] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 30 | -0.01 [-0.03, +0.02] | -0.01 | -0.00 [-0.02, +0.02] |

Guards: {"none_drift_median_junction": 0.13564065098762512, "none_drift_median_straight": 0.10972334444522858, "sg_wrong_move_median": 0.26182537090512364, "sg_wrong_abs_dy4_mean": 1.1390965486039843, "sg_wrong_move_median_ref": 0.2186037774391925, "sg_wrong_abs_dy4_mean_ref": 0.23093070915422625, "sg_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": 0.020315717001028025, "lo": -0.05783492891490086, "hi": 0.1046232092488419, "mean_a": 2.3822801767799286, "mean_b": 2.3619644597789002}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.039590095799960506, "lo": 0.009348758608937618, "hi": 0.08803385925122338, "mean_a": 0.2981230474267538, "mean_b": 0.2585329516267933}}
