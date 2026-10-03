# image-command fine-tune Q3: q3A-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.47, 0.54] (b 0.50, n 121.0) | -0.00 [-0.02, +0.02] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| band | 136 | +2.75 | +2.72 [+1.44, +4.44] | 0.30 [0.20, 0.41] (b -0.00) | +0.30 [+0.19, +0.42] | 0.76 [0.68, 0.84] (b 0.50, n 121.0) | +0.26 [+0.20, +0.34] | +1.67 / +0.56 / +2.09 | -0.39 | -0.46 [-0.98, -0.04] |
| barrier | 136 | +3.50 | +1.66 [+0.70, +2.81] | 0.38 [0.29, 0.47] (b 0.23) | +0.16 [+0.05, +0.24] | 0.76 [0.67, 0.85] (b 0.58, n 121.0) | +0.18 [+0.05, +0.29] | +2.56 / +0.56 / +2.59 | -0.71 | +1.95 [+1.62, +2.30] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | +0.03 [+0.00, +0.07] | +0.02 | +0.01 [-0.01, +0.04] |

Guards: {"none_drift_median_junction": 0.2028556764125824, "none_drift_median_straight": 0.125189870595932, "none_toward_taken": {"n": 136, "mean": 0.04000283661649425, "lo": -0.07779283293729283, "hi": 0.2069138053817134, "mean_a": 2.5848848995067857, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": -0.0016455513986023819, "lo": -0.018323414986390653, "hi": 0.017186227872308255, "mean_a": 0.24674023960126598, "mean_b": 0.24838579099986843}}
