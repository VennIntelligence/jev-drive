# image-command fine-tune: ft2-s0 vs O (dev, tau = 4 s)

Junction samples 136 (35 logs), straight 30. a = model, b = ref; d = a - b, 95% paired cluster bootstrap by log (2000 resamples for ratios, jevdrive.stats otherwise). Metrics as img_report.py (Delta m, uptake = fraction of a full switch, fcorrect = fixed-set correct, tw_c = move toward class c vs the model's own `none`, dx = plan x change vs `none`).

| family | n | Delta a | Delta d | uptake a | uptake d | fcorrect a | fcorrect d | toward L / S / R a | dx a | dx d |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| none | 136 | +0.00 | +0.00 [+0.00, +0.00] | 0.00 [0.00, 0.00] (b 0.00) | +0.00 [+0.00, +0.00] | 0.50 [0.46, 0.53] (b 0.50, n 120.0) | -0.00 [-0.02, +0.01] | +0.00 / +0.00 / +0.00 | +0.00 | +0.00 [+0.00, +0.00] |
| band | 136 | +5.70 | +5.67 [+3.31, +8.48] | 0.57 [0.38, 0.74] (b -0.00) | +0.57 [+0.39, +0.74] | 0.77 [0.67, 0.86] (b 0.50, n 120.0) | +0.27 [+0.18, +0.36] | +2.89 / +1.99 / +3.75 | +2.24 | +2.18 [+1.38, +2.88] |
| barrier | 136 | +6.44 | +4.61 [+2.93, +6.56] | 0.64 [0.51, 0.78] (b 0.23) | +0.41 [+0.26, +0.57] | 0.82 [0.73, 0.90] (b 0.58, n 120.0) | +0.23 [+0.10, +0.35] | +4.44 / +1.87 / +3.91 | +1.93 | +4.60 [+3.95, +5.23] |

Straight frames (lane keeping, 3 s): lateral error change vs the model's own `none`.

| family | n | d lat err a | d lat err b | a - b |
|:--|--:|:--|:--|:--|
| straight:band | 30 | -0.02 [-0.07, +0.03] | +0.02 | -0.04 [-0.11, +0.04] |

Guards: {"none_drift_median_junction": 0.28140008449554443, "none_drift_median_straight": 0.19973310828208923, "none_drift_median_all": 0.2538168132305145, "none_toward_taken": {"n": 136, "mean": 0.11727671205109956, "lo": -0.1066081091135242, "hi": 0.388545962204281, "mean_a": 2.6621587749413904, "mean_b": 2.544882062890291}, "straight_none_lat_err_3s": {"n": 30, "mean": 0.008962846262572417, "lo": -0.010024436649140635, "hi": 0.028712177661749452, "mean_a": 0.2573486372624408, "mean_b": 0.24838579099986843}}
