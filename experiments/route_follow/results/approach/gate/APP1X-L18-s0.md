# Approach gate: APP1X-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.295 | -0.152 [-0.534, +0.225] | 306 / 13 |
| speed | 0.006 | 0.006 | +0.090 [-0.199, +0.929] | 116 / 7 |
| head_deg | 2.937 | 2.757 | +0.061 [-0.036, +0.168] | 306 / 13 |
| lat_nup | 0.267 | 0.298 | -0.114 [-0.544, +0.340] | 249 / 11 |
| lat_pai | 0.206 | 0.282 | -0.368 [-1.048, +0.000] | 57 / 2 |
| speed_nup | 0.008 | 0.008 | +0.090 [-0.199, +0.929] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.120 | -0.495 [-1.626, +0.367] | 306 / 13 |
| loss2 | 0.115 | 0.116 | -0.013 [-0.419, +0.562] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.394 | +0.092 [+0.019, +0.152] | 313 / 25 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | -0.1522 | >= 0.3 | FAIL |
| A-speed | 0.0899 | >= 0.3 | FAIL |
| A-neg | -0.0398 | <= +0.1 m | PASS |
| G-real | 0.5600 | <= 0.5385 | FAIL |
| G-ST | 0.013, 0.041 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0056 | <= 0.0067 | PASS |
| G-rely | 0.0150 | >= 0.01 | PASS |

all pass: False
