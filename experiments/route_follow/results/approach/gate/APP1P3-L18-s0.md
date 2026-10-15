# Approach gate: APP1P3-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.277 | -0.083 [-0.465, +0.205] | 306 / 13 |
| speed | 0.006 | 0.007 | -0.039 [-0.082, +1.000] | 116 / 7 |
| head_deg | 2.937 | 2.693 | +0.083 [-0.002, +0.173] | 306 / 13 |
| lat_nup | 0.267 | 0.264 | +0.011 [-0.244, +0.291] | 249 / 11 |
| lat_pai | 0.206 | 0.333 | -0.619 [-1.901, +0.076] | 57 / 2 |
| speed_nup | 0.008 | 0.009 | -0.039 [-0.082, +1.000] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.112 | -0.399 [-2.036, +0.298] | 306 / 13 |
| loss2 | 0.115 | 0.128 | -0.118 [-0.808, +0.442] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.416 | +0.041 [-0.004, +0.089] | 313 / 25 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | -0.0833 | >= 0.3 | FAIL |
| A-speed | -0.0387 | >= 0.3 | FAIL |
| A-neg | -0.0177 | <= +0.1 m | PASS |
| G-real | 0.5655 | <= 0.5385 | FAIL |
| G-ST | 0.007, 0.034 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0045 | <= 0.0067 | PASS |
| G-rely | 0.0128 | >= 0.01 | PASS |

all pass: False
