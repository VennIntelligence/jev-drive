# image-command fine-tune Q3: q3SA-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.43 [0.40, 0.46] (b 0.43, n 107) | -0.00 [-0.01, +0.00] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 75 | +9.59 | +9.60 [+7.73, +11.92] | 0.43 [0.37, 0.49] (b 0.00) | +0.42 [+0.36, +0.49] | 0.81 [0.75, 0.89] (b 0.43, n 107) | +0.38 [+0.31, +0.46] | +1.33 / +3.38 / +10.40 | -1.03 | -0.39 [-1.09, +0.21] |

Guards: {"none_drift_median_junction": 0.24265924096107483, "sky_disc_move_median": 0.31705203641990976, "sky_disc_abs_dy4_mean": 2.701055038500104, "sky_disc_move_median_ref": 0.2535011741188035, "sky_disc_abs_dy4_mean_ref": 0.37946966347835787, "sky_disc_n": 75, "sky_wrong_move_median": 0.7908267107362024, "sky_wrong_abs_dy4_mean": 7.628197986718858, "sky_wrong_move_median_ref": 0.22848393117832577, "sky_wrong_abs_dy4_mean_ref": 0.5065774470283951, "sky_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.08201447521044741, "lo": -0.5861033052237885, "hi": 0.6111442110661301, "mean_a": 3.244278189946916, "mean_b": 3.1622637147364685}}
