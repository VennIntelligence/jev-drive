# image-command fine-tune Q3: q3GA-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.42, 0.46] (b 0.45, n 358) | -0.01 [-0.03, +0.00] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| sg | 225 | +10.30 | +9.35 [+8.20, +10.56] | 0.39 [0.35, 0.43] (b 0.05) | +0.34 [+0.30, +0.38] | 0.84 [0.80, 0.87] (b 0.48, n 358) | +0.36 [+0.31, +0.40] | +2.56 / +4.45 / +8.90 | -0.73 | -0.24 [-0.57, +0.09] |

Guards: {"none_drift_median_junction": 0.3653716742992401, "sg_wrong_move_median": 1.1602431876342347, "sg_wrong_abs_dy4_mean": 8.081701518736356, "sg_wrong_move_median_ref": 0.35234799441978865, "sg_wrong_abs_dy4_mean_ref": 0.4282392917071503, "sg_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": -0.1096896366552355, "lo": -0.6519173724146636, "hi": 0.35411646975435096, "mean_a": 3.5178437654225148, "mean_b": 3.627533402077749}}
