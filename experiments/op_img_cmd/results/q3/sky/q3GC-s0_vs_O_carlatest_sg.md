# image-command fine-tune Q3: q3GC-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.43, 0.47] (b 0.45, n 346) | -0.01 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | -0.00 | +0.00 [-0.00, +0.00] |
| sg | 225 | +3.60 | +2.64 [+2.03, +3.30] | 0.15 [0.13, 0.18] (b 0.05) | +0.11 [+0.09, +0.13] | 0.60 [0.55, 0.64] (b 0.48, n 346) | +0.12 [+0.08, +0.15] | +0.30 / +2.63 / +2.34 | -0.43 | +0.06 [-0.11, +0.23] |

Guards: {"none_drift_median_junction": 0.331499308347702, "sg_wrong_move_median": 0.6000795262009235, "sg_wrong_abs_dy4_mean": 2.359889525286186, "sg_wrong_move_median_ref": 0.35234799441978865, "sg_wrong_abs_dy4_mean_ref": 0.4282392917071503, "sg_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": -0.19166395823713592, "lo": -0.5227907499514015, "hi": 0.1496357243190509, "mean_a": 3.435869443840613, "mean_b": 3.627533402077749}}
