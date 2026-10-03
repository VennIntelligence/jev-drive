# image-command fine-tune Q3: q3SC-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.42, 0.46] (b 0.45, n 355) | -0.01 [-0.02, +0.00] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 225 | +0.23 | +0.23 [+0.18, +0.29] | 0.00 [-0.00, 0.01] (b 0.00) | -0.00 [-0.01, +0.00] | 0.46 [0.44, 0.48] (b 0.45, n 355) | +0.01 [-0.01, +0.03] | -0.49 / +0.21 / +0.63 | -0.41 | +0.20 [+0.11, +0.30] |

Guards: {"none_drift_median_junction": 0.23383621871471405, "sky_disc_move_median": 0.3039694958321957, "sky_disc_abs_dy4_mean": 0.8997284840951617, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.2836320472108598, "sky_wrong_abs_dy4_mean": 0.8058245716206904, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": -0.24617667670051332, "lo": -0.3734227969546363, "hi": -0.11663860780255617, "mean_a": 3.381430523497501, "mean_b": 3.6276072001980135}}
