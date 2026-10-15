# Approach gate: APP1P0-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.256 | -0.001 [-0.030, +0.027] | 306 / 13 |
| speed | 0.006 | 0.007 | -0.084 [-0.336, +0.122] | 116 / 7 |
| head_deg | 2.937 | 2.977 | -0.014 [-0.072, +0.037] | 306 / 13 |
| lat_nup | 0.267 | 0.265 | +0.007 [-0.023, +0.043] | 249 / 11 |
| lat_pai | 0.206 | 0.215 | -0.047 [-0.097, -0.019] | 57 / 2 |
| speed_nup | 0.008 | 0.009 | -0.084 [-0.336, +0.122] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.080 | +0.000 [-0.107, +0.086] | 306 / 13 |
| loss2 | 0.115 | 0.116 | -0.016 [-0.143, +0.041] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.434 | -0.002 [-0.009, +0.005] | 313 / 25 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | -0.0009 | >= 0.3 | FAIL |
| A-speed | -0.0837 | >= 0.3 | FAIL |
| A-neg | 0.0008 | <= +0.1 m | PASS |
| G-real | 0.5286 | <= 0.5385 | PASS |
| G-ST | 0.003, 0.020 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0067 | <= 0.0067 | PASS |
| G-rely | 0.0216 | >= 0.01 | PASS |

all pass: False
