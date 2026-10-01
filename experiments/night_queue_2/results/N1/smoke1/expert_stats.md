## x10: expert mode per scenario

| scenario              |   n |   bypass_L |   wait_then_bypass_L |   bypass_share | usable (>= 0.70)   |
|:----------------------|----:|-----------:|---------------------:|---------------:|:-------------------|
| Accident              |   1 |          1 |                    0 |              1 | yes                |
| HazardAtSideLane      |   1 |          1 |                    0 |              1 | yes                |
| ParkedObstacleTwoWays |   1 |          0 |                    1 |              1 | yes                |

## mode per world and class

|                    |   n |   keep |   bypass_L |   wait_then_bypass_L |   bypass_share |
|:-------------------|----:|-------:|-----------:|---------------------:|---------------:|
| ('1W', 'shoulder') |   1 |      1 |          0 |                    0 |              0 |
| ('1W', 'x00')      |   2 |      2 |          0 |                    0 |              0 |
| ('1W', 'x10')      |   2 |      0 |          2 |                    0 |              1 |
| ('2W', 'mirror')   |   1 |      0 |          0 |                    1 |              1 |
| ('2W', 'x00')      |   1 |      1 |          0 |                    0 |              0 |
| ('2W', 'x01')      |   1 |      1 |          0 |                    0 |              0 |
| ('2W', 'x10')      |   1 |      0 |          0 |                    1 |              1 |
| ('2W', 'x11')      |   1 |      0 |          0 |                    1 |              1 |

## t_div vs t_vis (x10 vs x00)

| scenario              |   pairs |   ok |   early |   never_visible |   med_div_minus_vis_s |   med_divlat_minus_vis_s |
|:----------------------|--------:|-----:|--------:|----------------:|----------------------:|-------------------------:|
| Accident              |       1 |    1 |       0 |               0 |                  3.8  |                     6.7  |
| HazardAtSideLane      |       1 |    1 |       0 |               0 |                  4    |                     6.2  |
| ParkedObstacleTwoWays |       1 |    1 |       0 |               0 |                  0.45 |                     7.35 |

smoke 1: x00 |d(k+3 s)| < 0.3 m on 40 / 40 x10-bypass frames (1.000; gate >= 0.95), 3 pairs with such frames of 3

## negotiation: x11 - x10 (2W)

| scenario              |   cases |   wait_share_x11 |   wait_share_x10 |   med_lat_delay_s |   med_dv_x11_minus_x10 |
|:----------------------|--------:|-----------------:|-----------------:|------------------:|-----------------------:|
| ParkedObstacleTwoWays |       1 |                1 |                1 |                 0 |                      0 |

pooled wait share x11 1.0 (gate >= 0.50), x10 1.0

placement null keep: 1 / 1 = 1.000 (gate >= 0.90)

| scenario   |   n |   keep |   bypass_share |
|:-----------|----:|-------:|---------------:|
| Accident   |   1 |      1 |              0 |

mirror stop: 0 / 1 = 0.000 (gate >= 0.80)

| scenario              |   n |   wait_then_bypass_L |   bypass_share |
|:----------------------|----:|---------------------:|---------------:|
| ParkedObstacleTwoWays |   1 |                    1 |              1 |

## smoke 2: oncoming vehicles within 50 m in the lane-change window (2W)

| world   |   worlds |   with >= 1 oncoming |   mean oncoming |
|:--------|---------:|---------------------:|----------------:|
| mirror  |        1 |                    1 |               5 |
| x00     |        1 |                    1 |               4 |
| x01     |        1 |                    1 |               5 |
| x10     |        1 |                    1 |               3 |
| x11     |        1 |                    1 |               4 |

## frame-level section 2.1 modes (mode_x10, frames from t_vis on)

| cls   |   curve_or_other |   keep |   lane_change |   nudge_return |   stop |   turn |
|:------|-----------------:|-------:|--------------:|---------------:|-------:|-------:|
| 1W    |               27 |     18 |            23 |             10 |      1 |     11 |
| 2W    |               10 |     10 |             0 |             19 |     16 |      0 |

## frame-level section 2.1 modes (mode_x00, frames from t_vis on)

| cls   |   keep |   stop |
|:------|-------:|-------:|
| 1W    |     61 |      0 |
| 2W    |     23 |      3 |
