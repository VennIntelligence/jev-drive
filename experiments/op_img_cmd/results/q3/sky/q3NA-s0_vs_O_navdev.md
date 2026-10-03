# image-command fine-tune Q3: q3NA-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.54] (b 0.50, n 108.0) | +0.00 [-0.02, +0.03] | +0.00 / +0.00 / +0.00 | +0.00 | -0.00 [-0.00, +0.00] |
| sky | 136 | +1.31 | +1.30 [+0.59, +2.25] | 0.16 [0.08, 0.23] (b 0.00) | +0.15 [+0.08, +0.23] | 0.59 [0.50, 0.67] (b 0.48, n 108.0) | +0.12 [+0.03, +0.20] | +0.42 / +0.07 / +1.52 | -0.35 | +0.59 [+0.24, +0.93] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 30 | +0.00 [-0.01, +0.01] | +0.00 | +0.00 [-0.01, +0.02] |
| straight:sky_disc | 30 | +0.00 [-0.01, +0.01] | +0.00 | -0.00 [-0.02, +0.01] |

Guards: {"none_drift_median_junction": 0.1000487357378006, "none_drift_median_straight": 0.09583401679992676, "sky_disc_move_median": 0.08171408634449344, "sky_disc_abs_dy4_mean": 0.14482791175501156, "sky_disc_move_median_ref": 0.19100464033475484, "sky_disc_abs_dy4_mean_ref": 0.16747363358218734, "sky_disc_n": 136, "sky_wrong_move_median": 0.14066670803789558, "sky_wrong_abs_dy4_mean": 0.48959980263951824, "sky_wrong_move_median_ref": 0.16426193588976942, "sky_wrong_abs_dy4_mean_ref": 0.17866676077937702, "sky_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": -0.014090782716379003, "lo": -0.07397803637123229, "hi": 0.048181478738225286, "mean_a": 2.347801795550105, "mean_b": 2.361892578266484}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.00525569279801773, "lo": -0.007572067914782776, "hi": 0.021381142332182773, "mean_a": 0.26372649096507333, "mean_b": 0.25847079816705554}}
