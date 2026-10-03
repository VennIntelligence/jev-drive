# image-command fine-tune Q3: O vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.43, 0.48] (b 0.45, n 345) | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| sky | 225 | -0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.01] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.42, 0.47] (b 0.45, n 345) | +0.00 [+0.00, +0.00] | +0.18 / -0.49 / +0.35 | -0.61 | +0.00 [+0.00, +0.00] |

Guards: {"none_drift_median_junction": 0.0, "sky_disc_move_median": 0.24228987798510915, "sky_disc_abs_dy4_mean": 0.2796735839161208, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.2688277182613713, "sky_wrong_abs_dy4_mean": 0.29156829488924196, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": 0.0, "lo": 0.0, "hi": 0.0, "mean_a": 3.6276072001980135, "mean_b": 3.6276072001980135}}
