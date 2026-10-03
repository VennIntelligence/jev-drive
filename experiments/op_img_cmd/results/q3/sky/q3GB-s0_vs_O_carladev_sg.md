# image-command fine-tune Q3: q3GB-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.44 [0.41, 0.47] (b 0.43, n 105) | +0.01 [+0.00, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sg | 75 | +8.80 | +7.62 [+5.68, +10.17] | 0.35 [0.30, 0.43] (b 0.07) | +0.29 [+0.24, +0.36] | 0.80 [0.75, 0.85] (b 0.51, n 105) | +0.29 [+0.24, +0.34] | +0.28 / +3.90 / +9.43 | -0.66 | -0.32 [-1.02, +0.27] |

Guards: {"none_drift_median_junction": 0.2554677128791809, "sg_wrong_move_median": 0.8745617961868135, "sg_wrong_abs_dy4_mean": 6.4293565723422805, "sg_wrong_move_median_ref": 0.4942496132499641, "sg_wrong_abs_dy4_mean_ref": 0.5756932344192192, "sg_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.33152811267096177, "lo": 0.07566813784844566, "hi": 0.6640269726429077, "mean_a": 3.494864729120981, "mean_b": 3.163336616450019}}
