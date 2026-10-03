# image-command fine-tune Q3: O vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.43, 0.48] (b 0.45, n 345) | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 | -0.00 | +0.00 [+0.00, +0.00] |
| sg | 225 | +0.95 | +0.00 [+0.00, +0.00] | 0.05 [0.03, 0.06] (b 0.05) | +0.00 [+0.00, +0.00] | 0.48 [0.46, 0.51] (b 0.48, n 345) | +0.00 [+0.00, +0.00] | +0.40 / +0.29 / +0.72 | -0.49 | +0.00 [+0.00, +0.00] |

Guards: {"none_drift_median_junction": 0.0, "sg_wrong_move_median": 0.35234799441978865, "sg_wrong_abs_dy4_mean": 0.4282392917071503, "sg_wrong_move_median_ref": 0.35234799441978865, "sg_wrong_abs_dy4_mean_ref": 0.4282392917071503, "sg_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": 0.0, "lo": 0.0, "hi": 0.0, "mean_a": 3.627533402077749, "mean_b": 3.627533402077749}}
