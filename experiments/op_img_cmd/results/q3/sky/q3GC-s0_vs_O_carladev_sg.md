# image-command fine-tune Q3: q3GC-s0 vs O (carladev, tau = 4 s)

Junction samples 75 (18 clusters), straight 0. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 75 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.42 [0.39, 0.46] (b 0.43, n 106) | -0.01 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sg | 75 | +3.61 | +2.43 [+1.79, +3.10] | 0.17 [0.13, 0.21] (b 0.07) | +0.10 [+0.08, +0.13] | 0.60 [0.54, 0.66] (b 0.51, n 106) | +0.09 [+0.04, +0.14] | -0.08 / +2.84 / +2.39 | -0.20 | +0.14 [-0.10, +0.42] |

Guards: {"none_drift_median_junction": 0.2899336516857147, "sg_wrong_move_median": 0.63865420667652, "sg_wrong_abs_dy4_mean": 1.971654610119102, "sg_wrong_move_median_ref": 0.4942496132499641, "sg_wrong_abs_dy4_mean_ref": 0.5756932344192192, "sg_wrong_n": 30, "none_toward_taken": {"n": 75, "mean": 0.09151474382436343, "lo": -0.25875220248067704, "hi": 0.44241291207765, "mean_a": 3.254851360274383, "mean_b": 3.163336616450019}}
