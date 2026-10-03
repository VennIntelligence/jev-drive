# image-command fine-tune Q3: q3NA-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.41, 0.47] (b 0.43, n 105) | +0.01 [+0.00, +0.02] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sky | 75 | +7.45 | +7.46 [+5.59, +9.87] | 0.34 [0.26, 0.43] (b 0.00) | +0.34 [+0.26, +0.42] | 0.76 [0.69, 0.84] (b 0.43, n 105) | +0.33 [+0.26, +0.40] | +0.83 / +2.83 / +8.06 | -0.67 | -0.04 [-0.71, +0.52] |

Guards: {"none_drift_median_junction": 0.22102133929729462, "sky_disc_move_median": 0.15573567744448139, "sky_disc_abs_dy4_mean": 0.30595514980653915, "sky_disc_move_median_ref": 0.2535011741188035, "sky_disc_abs_dy4_mean_ref": 0.37946966347835787, "sky_disc_n": 75, "sky_wrong_move_median": 0.3758531577112285, "sky_wrong_abs_dy4_mean": 3.78414355318292, "sky_wrong_move_median_ref": 0.22848393117832577, "sky_wrong_abs_dy4_mean_ref": 0.5065774470283951, "sky_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.37877018442962596, "lo": 0.01665198046553876, "hi": 0.7418829151088459, "mean_a": 3.541033899166095, "mean_b": 3.1622637147364685}}
