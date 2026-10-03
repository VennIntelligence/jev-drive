# image-command fine-tune Q3: q3NB-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.43 [0.40, 0.46] (b 0.43, n 108) | -0.01 [-0.02, +0.00] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 75 | +4.09 | +4.10 [+2.96, +5.37] | 0.18 [0.13, 0.23] (b 0.00) | +0.17 [+0.12, +0.23] | 0.58 [0.51, 0.66] (b 0.43, n 108) | +0.15 [+0.08, +0.23] | +0.20 / +1.64 / +4.89 | -0.61 | +0.03 [-0.33, +0.39] |

Guards: {"none_drift_median_junction": 0.2017180621623993, "sky_disc_move_median": 0.17089448235994392, "sky_disc_abs_dy4_mean": 0.29170801190018203, "sky_disc_move_median_ref": 0.2535011741188035, "sky_disc_abs_dy4_mean_ref": 0.37946966347835787, "sky_disc_n": 75, "sky_wrong_move_median": 0.1968007283590115, "sky_wrong_abs_dy4_mean": 3.464057612982803, "sky_wrong_move_median_ref": 0.22848393117832577, "sky_wrong_abs_dy4_mean_ref": 0.5065774470283951, "sky_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.23541546126157772, "lo": -0.05845903493560615, "hi": 0.5809467549275442, "mean_a": 3.397679175998046, "mean_b": 3.1622637147364685}}
