# image-command fine-tune Q3: q3GC-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.54] (b 0.50, n 108.0) | +0.00 [-0.02, +0.04] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sg | 136 | +0.16 | +0.28 [+0.18, +0.40] | 0.02 [-0.00, 0.03] (b -0.01) | +0.03 [+0.02, +0.04] | 0.50 [0.46, 0.54] (b 0.50, n 108.0) | +0.00 [-0.02, +0.04] | -0.08 / +0.07 / +0.19 | -0.24 | +0.59 [+0.34, +0.88] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sg | 30 | -0.02 [-0.05, -0.00] | -0.01 | -0.02 [-0.03, -0.00] |

Guards: {"none_drift_median_junction": 0.1300801783800125, "none_drift_median_straight": 0.11055377125740051, "sg_wrong_move_median": 0.20186949393075632, "sg_wrong_abs_dy4_mean": 0.21853327368707356, "sg_wrong_move_median_ref": 0.2186037774391925, "sg_wrong_abs_dy4_mean_ref": 0.23093070915422625, "sg_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": -0.06041046410982581, "lo": -0.13498495867865204, "hi": 0.015977263674225934, "mean_a": 2.301553995669075, "mean_b": 2.3619644597789002}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.04347067913534778, "lo": 0.010544262333330487, "hi": 0.09434160272695133, "mean_a": 0.3020036307621411, "mean_b": 0.2585329516267933}}
