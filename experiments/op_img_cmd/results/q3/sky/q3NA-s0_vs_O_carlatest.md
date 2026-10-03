# image-command fine-tune Q3: q3NA-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.43, 0.47] (b 0.45, n 354) | -0.01 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| sky | 225 | +7.38 | +7.38 [+6.13, +8.69] | 0.30 [0.26, 0.33] (b 0.00) | +0.29 [+0.26, +0.33] | 0.73 [0.70, 0.77] (b 0.45, n 354) | +0.29 [+0.24, +0.32] | +0.95 / +2.82 / +7.69 | -0.77 | -0.16 [-0.38, +0.06] |

Guards: {"none_drift_median_junction": 0.25240886211395264, "sky_disc_move_median": 0.17765220677130045, "sky_disc_abs_dy4_mean": 0.35599359750590137, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.4169670922987875, "sky_wrong_abs_dy4_mean": 4.910183010359459, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": 0.06236414927929221, "lo": -0.12934391707153958, "hi": 0.25634565311343543, "mean_a": 3.6899713494773065, "mean_b": 3.6276072001980135}}
