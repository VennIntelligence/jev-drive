# Approach gate: APP0C-L18-s1 vs RSG-L18-s1

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.255 | 0.249 | +0.025 [-0.008, +0.066] | 306 / 13 |
| speed | 0.007 | 0.009 | -0.356 [-9.706, -0.268] | 116 / 7 |
| head_deg | 2.890 | 2.860 | +0.010 [-0.033, +0.067] | 306 / 13 |
| lat_nup | 0.268 | 0.263 | +0.018 [-0.011, +0.053] | 249 / 11 |
| lat_pai | 0.200 | 0.186 | +0.070 [-0.039, +0.134] | 57 / 2 |
| speed_nup | 0.009 | 0.012 | -0.356 [-9.706, -0.268] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.076 | +0.050 [+0.005, +0.162] | 306 / 13 |
| loss2 | 0.114 | 0.117 | -0.024 [-0.174, +0.069] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.432 | 0.429 | +0.008 [-0.002, +0.018] | 313 / 25 |
| neg_dv | -1.942 | -1.921 | +0.021 [+0.016, +0.029] (difference) | 193 / 25 |
| over10_share | 0.026 | 0.026 | +0.000 [+0.000, +0.000] (difference) | 306 / 13 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | 0.0253 | >= 0.3 (reported) | FAIL (not gating) |
| A-speed | -0.3557 | >= 0.3 (reported) | FAIL (not gating) |
| A-neg | -0.0034 | <= +0.1 m (reported) | PASS (not gating) |
| G-real | 0.5272 | <= 0.5420 (base + 0.015) | PASS |
| G-ST | 0.005, 0.029 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0067 | <= 0.0067 | PASS |
| G-rely | 0.0243 | >= 0.01 | PASS |

all pass: True
