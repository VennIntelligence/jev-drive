# image-command fine-tune Q3: q3C-s0 vs O (navdev, tau = 4 s)

Junction samples 136 (35 clusters), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap (log / route). Metrics as img_report.py.

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 122.0) | -0.00 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| band | 136 | +0.16 | +0.13 [+0.06, +0.22] | 0.02 [0.00, 0.03] (b -0.00) | +0.02 [+0.00, +0.03] | 0.51 [0.49, 0.53] (b 0.50, n 122.0) | +0.01 [-0.01, +0.04] | -0.05 / +0.19 / +0.04 | +0.04 | -0.02 [-0.27, +0.19] |
| barrier | 136 | +1.57 | -0.27 [-0.75, +0.28] | 0.18 [0.09, 0.26] (b 0.23) | -0.05 [-0.13, +0.02] | 0.56 [0.48, 0.62] (b 0.58, n 122.0) | -0.03 [-0.13, +0.06] | +0.65 / +0.34 / +1.41 | -0.49 | +2.18 [+1.81, +2.56] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | +0.00 [-0.01, +0.02] | +0.02 | -0.01 [-0.04, +0.01] |

Guards: {"none_drift_median_junction": 0.13618203997612, "none_drift_median_straight": 0.09643872082233429, "none_toward_taken": {"n": 136, "mean": 0.027603241894597313, "lo": -0.08215542321452997, "hi": 0.18706454301976821, "mean_a": 2.572485304784888, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.006207319366318852, "lo": -0.007015039909748255, "hi": 0.023832939596545272, "mean_a": 0.25459311036618726, "mean_b": 0.24838579099986843}}
