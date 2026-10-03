# image-command fine-tune: ft3-s0 vs O (dev, tau = 4 s)

Junction samples 136 (35 logs), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap by log (2000 resamples for ratios, jevdrive.stats otherwise). Metrics as img_report.py (Delta m, uptake = fraction of a full switch, fcorrect = fixed-set correct, tw_c = move toward class c vs the model's own `none`, dx = plan x change vs `none`).

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.47, 0.53] (b 0.50, n 122.0) | -0.00 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | -0.00 | -0.00 [-0.00, +0.00] |
| band | 136 | +2.49 | +2.46 [+1.45, +3.70] | 0.25 [0.16, 0.35] (b -0.00) | +0.25 [+0.16, +0.35] | 0.61 [0.55, 0.68] (b 0.50, n 122.0) | +0.11 [+0.07, +0.18] | +0.28 / +1.88 / +1.07 | +2.80 | +2.74 [+2.06, +3.33] |
| barrier | 136 | +6.01 | +4.18 [+2.63, +5.92] | 0.62 [0.51, 0.74] (b 0.23) | +0.40 [+0.27, +0.53] | 0.80 [0.71, 0.89] (b 0.58, n 122.0) | +0.22 [+0.09, +0.34] | +3.97 / +1.81 / +3.64 | +1.80 | +4.47 [+3.88, +5.03] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | +0.00 [-0.06, +0.08] | +0.02 | -0.01 [-0.10, +0.08] |

Guards: {"none_drift_median_junction": 0.2053549885749817, "none_drift_median_straight": 0.11756065487861633, "none_drift_median_all": 0.18492728471755981, "none_toward_taken": {"n": 136, "mean": 0.11358686977271606, "lo": -0.08726093585393678, "hi": 0.38630631169737983, "mean_a": 2.658468932663007, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.004956063540593774, "lo": -0.011768210618616975, "hi": 0.02257988411742214, "mean_a": 0.2533418545404622, "mean_b": 0.24838579099986843}}
