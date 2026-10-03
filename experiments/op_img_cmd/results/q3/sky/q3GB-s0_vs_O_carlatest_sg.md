# image-command fine-tune Q3: q3GB-s0 vs O (carlatest, tau = 4 s)

Junction samples 225 (60 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 225 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.45 [0.42, 0.47] (b 0.45, n 355) | -0.01 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sg | 225 | +8.18 | +7.23 [+6.11, +8.43] | 0.30 [0.27, 0.34] (b 0.05) | +0.26 [+0.22, +0.29] | 0.73 [0.70, 0.77] (b 0.48, n 355) | +0.25 [+0.21, +0.29] | +0.69 / +3.62 / +8.31 | -0.76 | -0.28 [-0.58, +0.03] |

Guards: {"none_drift_median_junction": 0.27406594157218933, "sg_wrong_move_median": 0.9510566561178121, "sg_wrong_abs_dy4_mean": 6.703269889597996, "sg_wrong_move_median_ref": 0.35234799441978865, "sg_wrong_abs_dy4_mean_ref": 0.4282392917071503, "sg_wrong_n": 64, "none_toward_taken": {"n": 225, "mean": -0.03303310971629192, "lo": -0.32091778596062304, "hi": 0.22632379898060556, "mean_a": 3.5945002923614573, "mean_b": 3.627533402077749}}
