# image-command fine-tune Q3: q3NB-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 108.0) | +0.00 [-0.01, +0.02] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 136 | +0.39 | +0.38 [+0.16, +0.70] | 0.04 [0.02, 0.06] (b 0.00) | +0.04 [+0.01, +0.06] | 0.51 [0.47, 0.54] (b 0.48, n 108.0) | +0.03 [+0.00, +0.08] | +0.07 / +0.04 / +0.47 | -0.25 | +0.69 [+0.41, +0.99] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 30 | -0.01 [-0.02, +0.00] | +0.00 | -0.01 [-0.03, +0.01] |
| straight:sky_disc | 30 | -0.01 [-0.01, -0.00] | +0.00 | -0.01 [-0.03, +0.00] |

Guards: {"none_drift_median_junction": 0.08607068657875061, "none_drift_median_straight": 0.0675668865442276, "sky_disc_move_median": 0.06730537146873411, "sky_disc_abs_dy4_mean": 0.07883734005700493, "sky_disc_move_median_ref": 0.19100464033475484, "sky_disc_abs_dy4_mean_ref": 0.16747363358218734, "sky_disc_n": 136, "sky_wrong_move_median": 0.10486668306667897, "sky_wrong_abs_dy4_mean": 0.27930755930870144, "sky_wrong_move_median_ref": 0.16426193588976942, "sky_wrong_abs_dy4_mean_ref": 0.17866676077937702, "sky_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": 0.003616558085937246, "lo": -0.057348529927950614, "hi": 0.06889612729663246, "mean_a": 2.365509136352421, "mean_b": 2.361892578266484}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.0010516665015738536, "lo": -0.013935185784763736, "hi": 0.018402846629246353, "mean_a": 0.2595224646686295, "mean_b": 0.25847079816705554}}
