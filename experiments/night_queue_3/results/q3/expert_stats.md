## x10: expert mode per scenario

| scenario                    |   n |   keep |   stop |   bypass_L |   bypass_R |   wait_then_bypass_L |   wait_then_bypass_R |   bypass_share | usable (>= 0.70)   |
|:----------------------------|----:|-------:|-------:|-----------:|-----------:|---------------------:|---------------------:|---------------:|:-------------------|
| Accident                    |  15 |      0 |      2 |          7 |          0 |                    6 |                    0 |          0.867 | yes                |
| AccidentTwoWays             |  14 |      2 |      1 |         11 |          0 |                    0 |                    0 |          0.786 | yes                |
| ConstructionObstacle        |  16 |      1 |      1 |          8 |          1 |                    5 |                    0 |          0.875 | yes                |
| ConstructionObstacleTwoWays |  15 |      0 |      0 |         15 |          0 |                    0 |                    0 |          1     | yes                |
| HazardAtSideLane            |  15 |      0 |      0 |          9 |          0 |                    6 |                    0 |          1     | yes                |
| HazardAtSideLaneTwoWays     |  16 |      2 |      0 |         14 |          0 |                    0 |                    0 |          0.875 | yes                |
| ParkedObstacle              |  16 |      0 |      3 |          6 |          3 |                    3 |                    1 |          0.812 | yes                |
| ParkedObstacleTwoWays       |  16 |      1 |      2 |         13 |          0 |                    0 |                    0 |          0.812 | yes                |
| VehicleOpensDoorTwoWays     |  16 |      2 |      2 |         12 |          0 |                    0 |                    0 |          0.75  | yes                |

## mode per world and class

|                    |   n |   keep |   stop |   bypass_L |   bypass_R |   wait_then_bypass_L |   wait_then_bypass_R |   bypass_share |
|:-------------------|----:|-------:|-------:|-----------:|-----------:|---------------------:|---------------------:|---------------:|
| ('1W', 'shoulder') |  64 |     58 |      3 |          2 |          1 |                    0 |                    0 |          0.047 |
| ('1W', 'wnull')    |  63 |      1 |      6 |         31 |          4 |                   20 |                    1 |          0.889 |
| ('1W', 'x00')      |  62 |     56 |      3 |          2 |          1 |                    0 |                    0 |          0.048 |
| ('1W', 'x10')      |  62 |      1 |      6 |         30 |          4 |                   20 |                    1 |          0.887 |
| ('2W', 'mirror')   |  80 |     17 |     61 |          1 |          0 |                    1 |                    0 |          0.025 |
| ('2W', 'shoulder') |  76 |     60 |     16 |          0 |          0 |                    0 |                    0 |          0     |
| ('2W', 'wnull')    |  77 |      6 |      5 |         66 |          0 |                    0 |                    0 |          0.857 |
| ('2W', 'x00')      |  79 |     66 |     13 |          0 |          0 |                    0 |                    0 |          0     |
| ('2W', 'x01')      |  79 |     66 |     13 |          0 |          0 |                    0 |                    0 |          0     |
| ('2W', 'x10')      |  77 |      7 |      5 |         65 |          0 |                    0 |                    0 |          0.844 |
| ('2W', 'x11')      |  73 |     10 |     26 |         25 |          0 |                   12 |                    0 |          0.507 |

## t_div vs t_vis (x10 vs x00)

| scenario                    |   pairs |   ok |   early |   never_visible |   med_div_minus_vis_s |   med_divlat_minus_vis_s |
|:----------------------------|--------:|-----:|--------:|----------------:|----------------------:|-------------------------:|
| Accident                    |      16 |   15 |       0 |               0 |                  4.7  |                     6.8  |
| AccidentTwoWays             |      16 |   14 |       0 |               0 |                  4.75 |                     7.65 |
| ConstructionObstacle        |      16 |   15 |       0 |               0 |                  4.2  |                     6.95 |
| ConstructionObstacleTwoWays |      16 |   14 |       0 |               0 |                  4.85 |                     8.1  |
| HazardAtSideLane            |      16 |   14 |       0 |               0 |                  1.78 |                     7.65 |
| HazardAtSideLaneTwoWays     |      16 |   16 |       0 |               0 |                  3.03 |                     6.68 |
| ParkedObstacle              |      16 |   15 |       0 |               1 |                  6.3  |                     8.7  |
| ParkedObstacleTwoWays       |      16 |   16 |       0 |               0 |                  4    |                     7    |
| VehicleOpensDoorTwoWays     |      16 |   16 |       0 |               0 |                  2.45 |                     6.33 |

smoke 1: x00 |d(k+3 s)| < 0.3 m on 844 / 1186 x10-bypass frames (0.712; gate >= 0.95), 99 pairs with such frames of 144

## negotiation: x11 - x10 (2W)

| scenario                    |   cases |   wait_share_x11 |   wait_share_x10 |   med_lat_delay_s |   med_dv_x11_minus_x10 |
|:----------------------------|--------:|-----------------:|-----------------:|------------------:|-----------------------:|
| AccidentTwoWays             |      16 |            0.625 |            0.062 |           nan     |                 -7.809 |
| ConstructionObstacleTwoWays |      16 |            0.938 |            0     |             5.8   |                 -9.219 |
| HazardAtSideLaneTwoWays     |      16 |            0     |            0     |            18.95  |                 -1.172 |
| ParkedObstacleTwoWays       |      16 |            0.625 |            0.125 |             3.925 |                 -6.841 |
| VehicleOpensDoorTwoWays     |      16 |            0.188 |            0.125 |             0.3   |                 -2.47  |

pooled wait share x11 0.475 (gate >= 0.50), x10 0.062

placement null keep: 118 / 140 = 0.843 (gate >= 0.90)

| scenario                    |   n |   keep |   stop |   bypass_L |   bypass_R |   bypass_share |
|:----------------------------|----:|-------:|-------:|-----------:|-----------:|---------------:|
| Accident                    |  16 |     14 |      1 |          1 |          0 |          0.062 |
| AccidentTwoWays             |  15 |     15 |      0 |          0 |          0 |          0     |
| ConstructionObstacle        |  16 |     15 |      0 |          1 |          0 |          0.062 |
| ConstructionObstacleTwoWays |  15 |     15 |      0 |          0 |          0 |          0     |
| HazardAtSideLane            |  16 |     14 |      2 |          0 |          0 |          0     |
| HazardAtSideLaneTwoWays     |  16 |     12 |      4 |          0 |          0 |          0     |
| ParkedObstacle              |  16 |     15 |      0 |          0 |          1 |          0.062 |
| ParkedObstacleTwoWays       |  15 |     10 |      5 |          0 |          0 |          0     |
| VehicleOpensDoorTwoWays     |  15 |      8 |      7 |          0 |          0 |          0     |

mirror stop: 61 / 80 = 0.762 (gate >= 0.80)

| scenario                    |   n |   keep |   stop |   bypass_L |   wait_then_bypass_L |   bypass_share |
|:----------------------------|----:|-------:|-------:|-----------:|---------------------:|---------------:|
| AccidentTwoWays             |  16 |      2 |     14 |          0 |                    0 |          0     |
| ConstructionObstacleTwoWays |  16 |      0 |     16 |          0 |                    0 |          0     |
| HazardAtSideLaneTwoWays     |  16 |     14 |      2 |          0 |                    0 |          0     |
| ParkedObstacleTwoWays       |  16 |      1 |     14 |          1 |                    0 |          0.062 |
| VehicleOpensDoorTwoWays     |  16 |      0 |     15 |          0 |                    1 |          0.062 |

weather null: same world mode as x10 in 135 / 135

## smoke 2: oncoming vehicles within 50 m in the lane-change window (2W)

| world    |   worlds |   with >= 1 oncoming |   mean oncoming |
|:---------|---------:|---------------------:|----------------:|
| mirror   |       80 |                   71 |            8.41 |
| shoulder |       76 |                    0 |            0    |
| wnull    |       77 |                    1 |            0.01 |
| x00      |       79 |                    0 |            0    |
| x01      |       79 |                   64 |            3.67 |
| x10      |       77 |                    1 |            0.01 |
| x11      |       73 |                   71 |            5.04 |

## frame-level section 2.1 modes (mode_x10, frames from t_vis on)

| cls   |   curve_or_other |   keep |   lane_change |   nudge_hold |   nudge_return |   stop |   turn |
|:------|-----------------:|-------:|--------------:|-------------:|---------------:|-------:|-------:|
| 1W    |              516 |   1007 |           450 |            4 |            284 |   1583 |     81 |
| 2W    |              312 |   1420 |           211 |           17 |            762 |   1071 |     37 |

## frame-level section 2.1 modes (mode_x00, frames from t_vis on)

| cls   |   curve_or_other |   keep |   nudge_return |   stop |   turn |
|:------|-----------------:|-------:|---------------:|-------:|-------:|
| 1W    |                2 |   1961 |              2 |    525 |      8 |
| 2W    |                0 |   2671 |              0 |    559 |      0 |
