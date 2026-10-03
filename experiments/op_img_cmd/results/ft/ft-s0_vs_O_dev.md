# image-command fine-tune: ft-s0 vs O (dev, tau = 4 s)

Junction samples 136 (35 logs), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap by log (2000 resamples for ratios, jevdrive.stats otherwise). Metrics as img_report.py (Delta m, uptake = fraction of a full switch, fcorrect = fixed-set correct, tw_c = move toward class c vs the model's own `none`, dx = plan x change vs `none`).

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.49 [0.45, 0.51] (b 0.50, n 123.0) | -0.02 [-0.03, -0.00] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [-0.00, +0.00] |
| band | 136 | +0.72 | +0.69 [+0.38, +1.06] | 0.06 [0.03, 0.10] (b -0.00) | +0.06 [+0.04, +0.10] | 0.50 [0.46, 0.55] (b 0.50, n 123.0) | +0.01 [-0.02, +0.05] | -0.27 / +1.08 / -0.13 | +2.01 | +1.94 [+1.47, +2.37] |
| barrier | 136 | +1.90 | +0.07 [-0.43, +0.67] | 0.21 [0.15, 0.28] (b 0.23) | -0.02 [-0.10, +0.07] | 0.54 [0.49, 0.62] (b 0.58, n 123.0) | -0.04 [-0.14, +0.07] | +0.52 / +0.79 / +1.45 | +0.98 | +3.64 [+3.13, +4.14] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | -0.04 [-0.09, -0.00] | +0.02 | -0.06 [-0.13, -0.00] |

Guards: {"none_drift_median_junction": 0.4510963261127472, "none_drift_median_straight": 0.24410763382911682, "none_drift_median_all": 0.37130773067474365, "none_toward_taken": {"n": 136, "mean": 0.2649347244845076, "lo": -0.040218341703498015, "hi": 0.6083036289605683, "mean_a": 2.809816787374799, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.004026978462574186, "lo": -0.018734689346666063, "hi": 0.03971118913536925, "mean_a": 0.2524127694624426, "mean_b": 0.24838579099986843}}
