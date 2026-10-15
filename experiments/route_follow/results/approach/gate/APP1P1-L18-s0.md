# Approach gate: APP1P1-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.262 | -0.023 [-0.333, +0.230] | 306 / 13 |
| speed | 0.006 | 0.007 | -0.024 [-0.044, +1.000] | 116 / 7 |
| head_deg | 2.937 | 2.701 | +0.080 [-0.000, +0.194] | 306 / 13 |
| lat_nup | 0.267 | 0.260 | +0.028 [-0.190, +0.279] | 249 / 11 |
| lat_pai | 0.206 | 0.270 | -0.314 [-1.427, +0.290] | 57 / 2 |
| speed_nup | 0.008 | 0.008 | -0.024 [-0.044, +1.000] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.098 | -0.227 [-1.414, +0.447] | 306 / 13 |
| loss2 | 0.115 | 0.120 | -0.048 [-0.590, +0.420] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.425 | +0.020 [-0.019, +0.056] | 313 / 25 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | -0.0234 | >= 0.3 | FAIL |
| A-speed | -0.0238 | >= 0.3 | FAIL |
| A-neg | -0.0085 | <= +0.1 m | PASS |
| G-real | 0.5490 | <= 0.5385 | FAIL |
| G-ST | 0.004, 0.026 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0056 | <= 0.0067 | PASS |
| G-rely | 0.0196 | >= 0.01 | PASS |

all pass: False
