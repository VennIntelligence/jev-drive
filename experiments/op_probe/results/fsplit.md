| set                             |   n |   raw_plan_out |   lqr_only |   depth_lt_0_3 |   depth_median_m |   depth_p90_m |
|:--------------------------------|----:|---------------:|-----------:|---------------:|-----------------:|--------------:|
| F (P2-s0 DAC fail, WA pass)     | 433 |          0.716 |      0.284 |          0.503 |            0.297 |         1.207 |
| F-plan (raw plan footprint out) | 310 |          1.000 |      0.000 |          0.371 |            0.422 |         1.384 |
| F-core (F-plan, depth >= 0.3 m) | 195 |        nan     |    nan     |        nan     |          nan     |       nan     |

opb_score.py on P2-F-s0's navtest plans (scorer reproduces the devkit DAC on all checked tokens)
