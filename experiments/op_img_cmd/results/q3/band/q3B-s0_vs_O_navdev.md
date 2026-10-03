# image-command fine-tune Q3: q3B-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 122.0) | -0.00 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| band | 136 | +0.54 | +0.52 [+0.21, +0.90] | 0.05 [0.02, 0.09] (b -0.00) | +0.06 [+0.02, +0.09] | 0.53 [0.49, 0.58] (b 0.50, n 122.0) | +0.04 [+0.00, +0.08] | +0.12 / +0.38 / +0.21 | -0.08 | -0.14 [-0.45, +0.11] |
| barrier | 136 | +2.61 | +0.78 [+0.09, +1.61] | 0.29 [0.20, 0.37] (b 0.23) | +0.07 [-0.04, +0.14] | 0.66 [0.58, 0.72] (b 0.58, n 122.0) | +0.07 [-0.03, +0.17] | +1.51 / +0.50 / +2.11 | -0.61 | +2.06 [+1.71, +2.41] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | +0.02 [-0.00, +0.04] | +0.02 | +0.00 [-0.02, +0.02] |

Guards: {"none_drift_median_junction": 0.16507193446159363, "none_drift_median_straight": 0.1029169112443924, "none_toward_taken": {"n": 136, "mean": 0.04210424253999415, "lo": -0.07165485219048907, "hi": 0.20756161875453047, "mean_a": 2.586986305430285, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.003166096386694712, "lo": -0.012472487246802158, "hi": 0.019266732564215256, "mean_a": 0.25155188738656314, "mean_b": 0.24838579099986843}}
