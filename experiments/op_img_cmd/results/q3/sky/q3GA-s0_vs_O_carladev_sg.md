# image-command fine-tune Q3: q3GA-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.43 [0.40, 0.46] (b 0.43, n 107) | -0.00 [-0.01, +0.00] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sg | 75 | +10.07 | +8.89 [+6.96, +11.30] | 0.44 [0.39, 0.51] (b 0.07) | +0.38 [+0.33, +0.45] | 0.85 [0.79, 0.91] (b 0.51, n 107) | +0.34 [+0.27, +0.41] | +1.22 / +4.51 / +9.88 | -0.60 | -0.26 [-0.99, +0.38] |

Guards: {"none_drift_median_junction": 0.33655890822410583, "sg_wrong_move_median": 1.0154059695355369, "sg_wrong_abs_dy4_mean": 7.905267043469758, "sg_wrong_move_median_ref": 0.4942496132499641, "sg_wrong_abs_dy4_mean_ref": 0.5756932344192192, "sg_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.4772322819672025, "lo": 0.09706540664692237, "hi": 0.881932210479977, "mean_a": 3.640568898417222, "mean_b": 3.163336616450019}}
