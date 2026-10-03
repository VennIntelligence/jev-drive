# image-command fine-tune Q3: q3SA-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 108.0) | +0.00 [-0.01, +0.03] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| sky | 136 | +2.53 | +2.53 [+1.16, +4.11] | 0.32 [0.20, 0.40] (b 0.00) | +0.31 [+0.19, +0.40] | 0.75 [0.64, 0.84] (b 0.48, n 108.0) | +0.27 [+0.17, +0.36] | +0.61 / +0.06 / +3.15 | -0.65 | +0.28 [-0.23, +0.73] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 30 | -0.01 [-0.05, +0.01] | +0.00 | -0.01 [-0.04, +0.01] |
| straight:sky_disc | 30 | -0.01 [-0.06, +0.02] | +0.00 | -0.01 [-0.06, +0.02] |

Guards: {"none_drift_median_junction": 0.10687577724456787, "none_drift_median_straight": 0.09777147322893143, "sky_disc_move_median": 0.1117556745524826, "sky_disc_abs_dy4_mean": 0.27478074000058095, "sky_disc_move_median_ref": 0.19100464033475484, "sky_disc_abs_dy4_mean_ref": 0.16747363358218734, "sky_disc_n": 136, "sky_wrong_move_median": 0.23099288115871025, "sky_wrong_abs_dy4_mean": 1.2399368756458282, "sky_wrong_move_median_ref": 0.16426193588976942, "sky_wrong_abs_dy4_mean_ref": 0.17866676077937702, "sky_wrong_n": 123, "none_toward_taken": {"n": 136, "mean": -0.05963802678957609, "lo": -0.21376929874953463, "hi": 0.06481065885839195, "mean_a": 2.302254551476908, "mean_b": 2.361892578266484}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.017211228178000415, "lo": -0.00500360525189834, "hi": 0.04910832722421548, "mean_a": 0.27568202634505595, "mean_b": 0.25847079816705554}}
