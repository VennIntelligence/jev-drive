# image-command fine-tune Q3: q3SB-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.54] (b 0.50, n 107.0) | +0.01 [-0.01, +0.04] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sky | 136 | +1.16 | +1.15 [+0.41, +2.08] | 0.14 [0.06, 0.21] (b 0.00) | +0.14 [+0.05, +0.21] | 0.58 [0.49, 0.65] (b 0.48, n 107.0) | +0.10 [+0.03, +0.18] | +0.07 / +0.04 / +1.57 | -0.36 | +0.58 [+0.20, +0.94] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 30 | -0.01 [-0.02, -0.00] | +0.00 | -0.01 [-0.03, +0.00] |
| straight:sky_disc | 30 | -0.01 [-0.01, -0.00] | +0.00 | -0.01 [-0.03, -0.00] |

Guards: {"none_drift_median_junction": 0.07521364837884903, "none_drift_median_straight": 0.0656673014163971, "sky_disc_move_median": 0.08229268529326693, "sky_disc_abs_dy4_mean": 0.17576045664540926, "sky_disc_move_median_ref": 0.19100464033475484, "sky_disc_abs_dy4_mean_ref": 0.16747363358218734, "sky_disc_n": 136, "sky_wrong_move_median": 0.11238491567938144, "sky_wrong_abs_dy4_mean": 0.5539962949957074, "sky_wrong_move_median_ref": 0.16426193588976942, "sky_wrong_abs_dy4_mean_ref": 0.17866676077937702, "sky_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": 0.016279339162555057, "lo": -0.030186135982882036, "hi": 0.0695257400407545, "mean_a": 2.3781719174290394, "mean_b": 2.361892578266484}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.005113849406993358, "lo": -0.005128186488681466, "hi": 0.016682336512273757, "mean_a": 0.2635846475740489, "mean_b": 0.25847079816705554}}
