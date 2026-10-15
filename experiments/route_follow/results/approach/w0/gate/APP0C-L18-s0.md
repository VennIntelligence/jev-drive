# Approach gate: APP0C-L18-s0 vs RSG-L18-s0

dev rows: 306 turn ({"nup": {"turn": 249, "neg": 249}, "pai": {"turn": 57, "neg": 64}}), 116 with the speed read, 313 straight negatives; held-out logs, never in the pool.

| read | base | pilot | drop [95 % CI] | rows / logs |
|:--|--:|--:|:--|:--|
| lat | 0.256 | 0.246 | +0.038 [-0.015, +0.081] | 306 / 13 |
| speed | 0.006 | 0.009 | -0.337 [-5.928, +0.037] | 116 / 7 |
| head_deg | 2.937 | 2.834 | +0.035 [+0.005, +0.075] | 306 / 13 |
| lat_nup | 0.267 | 0.262 | +0.019 [-0.053, +0.058] | 249 / 11 |
| lat_pai | 0.206 | 0.176 | +0.146 [+0.119, +0.194] | 57 / 2 |
| speed_nup | 0.008 | 0.011 | -0.337 [-5.928, +0.037] | 89 / 6 |
| speed_pai | 0.000 | 0.000 | +nan [+nan, +nan] | 27 / 1 |
| loss1 | 0.080 | 0.075 | +0.064 [-0.029, +0.196] | 306 / 13 |
| loss2 | 0.115 | 0.113 | +0.013 [-0.043, +0.089] | 306 / 13 |
| loss3 | 0.000 | 0.000 | +nan [+nan, +nan] | 116 / 7 |
| neg_lat | 0.433 | 0.432 | +0.003 [-0.009, +0.013] | 313 / 25 |
| neg_dv | -1.948 | -1.914 | +0.034 [+0.022, +0.053] (difference) | 192 / 25 |
| over10_share | 0.029 | 0.026 | -0.003 [-0.011, +0.000] (difference) | 306 / 13 |

| gate | value | line | pass |
|:--|--:|:--|:--|
| A-lat | 0.0382 | >= 0.3 (reported) | FAIL (not gating) |
| A-speed | -0.3369 | >= 0.3 (reported) | FAIL (not gating) |
| A-neg | -0.0014 | <= +0.1 m (reported) | PASS (not gating) |
| G-real | 0.5288 | <= 0.5435 (base + 0.015) | PASS |
| G-ST | 0.005, 0.028 | <= 0.10 / <= 0.10 | PASS |
| G-over | 0.0067 | <= 0.0067 | PASS |
| G-rely | 0.0234 | >= 0.01 | PASS |

all pass: True
