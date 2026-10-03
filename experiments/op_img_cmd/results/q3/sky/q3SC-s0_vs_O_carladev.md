# image-command fine-tune Q3: q3SC-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.43 [0.40, 0.47] (b 0.43, n 106) | +0.00 [-0.01, +0.02] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 75 | +0.22 | +0.23 [+0.17, +0.30] | 0.01 [-0.00, 0.01] (b 0.00) | +0.00 [-0.00, +0.01] | 0.43 [0.40, 0.47] (b 0.43, n 106) | +0.00 [-0.01, +0.02] | -0.52 / +0.25 / +0.64 | -0.37 | +0.27 [+0.04, +0.52] |

Guards: {"none_drift_median_junction": 0.22221708297729492, "sky_disc_move_median": 0.24090974349091268, "sky_disc_abs_dy4_mean": 0.8250081860172955, "sky_disc_move_median_ref": 0.2535011741188035, "sky_disc_abs_dy4_mean_ref": 0.37946966347835787, "sky_disc_n": 75, "sky_wrong_move_median": 0.21269587647713015, "sky_wrong_abs_dy4_mean": 0.867196734884362, "sky_wrong_move_median_ref": 0.22848393117832577, "sky_wrong_abs_dy4_mean_ref": 0.5065774470283951, "sky_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": -0.1462856632537419, "lo": -0.3500413452360428, "hi": 0.056567580359298014, "mean_a": 3.0159780514827266, "mean_b": 3.1622637147364685}}
