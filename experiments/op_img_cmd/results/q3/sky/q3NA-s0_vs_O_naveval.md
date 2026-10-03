# image-command fine-tune Q3: q3NA-s0 vs O (naveval, tau = 4 s)

Junction samples 385 (247 clusters), straight 293. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 384 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.47, 0.53] (b 0.51, n 241.0) | -0.01 [-0.03, -0.00] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, -0.00] |
| sky | 384 | +1.03 | +1.03 [+0.74, +1.37] | 0.18 [0.13, 0.24] (b -0.00) | +0.18 [+0.14, +0.24] | 0.62 [0.58, 0.67] (b 0.50, n 241.0) | +0.12 [+0.08, +0.17] | +0.40 / +0.14 / +1.00 | -0.37 | +0.82 [+0.65, +1.00] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:sky | 293 | +0.00 [-0.00, +0.01] | +0.00 | -0.00 [-0.01, +0.00] |
| straight:sky_disc | 293 | +0.01 [+0.00, +0.02] | +0.00 | +0.01 [+0.00, +0.01] |

Guards: {"none_drift_median_junction": 0.11922046542167664, "none_drift_median_straight": 0.06709346920251846, "sky_disc_move_median": 0.09602374692742559, "sky_disc_abs_dy4_mean": 0.1578287669443428, "sky_disc_move_median_ref": 0.17344454004797757, "sky_disc_abs_dy4_mean_ref": 0.2044576100881637, "sky_disc_n": 385, "sky_wrong_move_median": 0.11832562139357412, "sky_wrong_abs_dy4_mean": 0.42946693233985866, "sky_wrong_move_median_ref": 0.139451117697597, "sky_wrong_abs_dy4_mean_ref": 0.21610442541656485, "sky_wrong_n": 313, "none_toward_taken": {"n": 385, "mean": 0.026421499503057815, "lo": -0.01748478350318598, "hi": 0.07388107895285236, "mean_a": 1.3553361398837815, "mean_b": 1.3289146403807235}, "straight_none_lat_err_3s": {"n": 293, "mean": 0.0005381294824735656, "lo": -0.005084227704279283, "hi": 0.006096462186802406, "mean_a": 0.16130039877914715, "mean_b": 0.1607622692966736}}
