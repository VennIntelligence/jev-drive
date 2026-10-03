# image-command fine-tune Q3: q3NB-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.43, 0.47] (b 0.45, n 352) | -0.00 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 225 | +4.03 | +4.03 [+3.20, +4.92] | 0.15 [0.12, 0.17] (b 0.00) | +0.14 [+0.11, +0.17] | 0.61 [0.58, 0.65] (b 0.45, n 352) | +0.17 [+0.13, +0.20] | +0.24 / +1.52 / +4.58 | -0.70 | -0.09 [-0.28, +0.09] |

Guards: {"none_drift_median_junction": 0.21406342089176178, "sky_disc_move_median": 0.15828371653609813, "sky_disc_abs_dy4_mean": 0.27442566118674466, "sky_disc_move_median_ref": 0.24228987798510915, "sky_disc_abs_dy4_mean_ref": 0.2796735839161208, "sky_disc_n": 225, "sky_wrong_move_median": 0.2837044715984137, "sky_wrong_abs_dy4_mean": 3.286541651674812, "sky_wrong_move_median_ref": 0.2688277182613713, "sky_wrong_abs_dy4_mean_ref": 0.29156829488924196, "sky_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": 0.17541568857477485, "lo": 0.0031068515699432054, "hi": 0.36008442461255896, "mean_a": 3.8030228887727886, "mean_b": 3.6276072001980135}}
