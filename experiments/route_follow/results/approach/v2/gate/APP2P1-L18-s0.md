# Approach gate: APP2P1-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.268 | -0.046 [-0.329, +0.188] | 306 / 13 |
| speed | 0.006 | 0.009 | -0.390 [-0.498, +1.000] | 116 / 7 |
| head_deg | 2.937 | 2.562 | +0.128 [+0.019, +0.229] | 306 / 13 |
| lat_nup | 0.267 | 0.269 | -0.007 [-0.250, +0.233] | 249 / 11 |
| lat_pai | 0.206 | 0.261 | -0.269 [-1.165, +0.217] | 57 / 2 |
| speed_nup | 0.008 | 0.012 | -0.390 [-0.498, +1.000] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.098 | -0.223 [-1.300, +0.359] | 306 / 13 |
| loss2 | 0.115 | 0.119 | -0.036 [-0.484, +0.391] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.430 | +0.007 [-0.018, +0.034] | 313 / 25 |
| neg_dv | -1.948 | -1.803 | +0.145 [+0.038, +0.227] (difference) | 192 / 25 |
| over10_share | 0.029 | 0.023 | -0.007 [-0.023, +0.000] (difference) | 306 / 13 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | -0.0459 | >= 0.3 (reported) | FAIL (not gating) |
| A-speed | -0.3897 | >= 0.3 (reported) | FAIL (not gating) |
| A-neg | -0.0029 | <= +0.1 m (reported) | PASS (not gating) |
| G-real | 0.5508 | <= 0.5435 (base + 0.015) | FAIL |
| G-ST | 0.005, 0.027 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0056 | <= 0.0067 | PASS |
| G-rely | 0.0176 | >= 0.01 | PASS |

all pass: False
