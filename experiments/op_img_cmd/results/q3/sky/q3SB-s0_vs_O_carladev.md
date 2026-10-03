# image-command fine-tune Q3: q3SB-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.41, 0.47] (b 0.43, n 105) | +0.01 [+0.00, +0.02] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 75 | +6.70 | +6.71 [+4.94, +8.86] | 0.27 [0.20, 0.35] (b 0.00) | +0.27 [+0.20, +0.35] | 0.69 [0.62, 0.76] (b 0.43, n 105) | +0.25 [+0.18, +0.33] | +0.14 / +2.43 / +8.26 | -0.79 | -0.15 [-0.73, +0.33] |

Guards: {"none_drift_median_junction": 0.20061704516410828, "sky_disc_move_median": 0.2399033543091369, "sky_disc_abs_dy4_mean": 2.396787040398878, "sky_disc_move_median_ref": 0.2535011741188035, "sky_disc_abs_dy4_mean_ref": 0.37946966347835787, "sky_disc_n": 75, "sky_wrong_move_median": 0.286726977257837, "sky_wrong_abs_dy4_mean": 5.618390476223332, "sky_wrong_move_median_ref": 0.22848393117832577, "sky_wrong_abs_dy4_mean_ref": 0.5065774470283951, "sky_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.2706677722405425, "lo": 0.01977195814815983, "hi": 0.5608852148167701, "mean_a": 3.4329314869770102, "mean_b": 3.1622637147364685}}
