## x10: expert mode per scenario

| scenario                    |   n |   keep |   stop |   bypass_L |   bypass_R |   wait_then_bypass_L |   bypass_share | usable (>= 0.70)   |
|:----------------------------|----:|-------:|-------:|-----------:|-----------:|---------------------:|---------------:|:-------------------|
| Accident                    |  15 |      0 |      0 |         11 |          0 |                    4 |          1     | yes                |
| AccidentTwoWays             |  15 |      0 |      0 |         15 |          0 |                    0 |          1     | yes                |
| ConstructionObstacle        |  15 |      0 |      0 |         11 |          0 |                    4 |          1     | yes                |
| ConstructionObstacleTwoWays |  15 |      0 |      0 |         15 |          0 |                    0 |          1     | yes                |
| HazardAtSideLane            |  15 |      0 |      0 |         13 |          0 |                    2 |          1     | yes                |
| HazardAtSideLaneTwoWays     |  15 |      0 |      0 |         15 |          0 |                    0 |          1     | yes                |
| InvadingTurn                |  15 |      6 |      3 |          0 |          6 |                    0 |          0.4   | no                 |
| ParkedObstacle              |  15 |      0 |      0 |         10 |          0 |                    5 |          1     | yes                |
| ParkedObstacleTwoWays       |  15 |      0 |      0 |         15 |          0 |                    0 |          1     | yes                |
| VehicleOpensDoorTwoWays     |  15 |      0 |      5 |         10 |          0 |                    0 |          0.667 | no                 |
| YieldToEmergencyVehicle     |  15 |      0 |      0 |          9 |          6 |                    0 |          1     | yes                |

## mode per world and class

|                    |   n |   keep |   stop |   bypass_L |   bypass_R |   wait_then_bypass_L |   no_window |   bypass_share |
|:-------------------|----:|-------:|-------:|-----------:|-----------:|---------------------:|------------:|---------------:|
| ('1W', 'shoulder') |  20 |     20 |      0 |          0 |          0 |                    0 |           0 |          0     |
| ('1W', 'wnull')    |  20 |      0 |      0 |         17 |          0 |                    3 |           0 |          1     |
| ('1W', 'x00')      |  60 |     60 |      0 |          0 |          0 |                    0 |           0 |          0     |
| ('1W', 'x10')      |  60 |      0 |      0 |         45 |          0 |                   15 |           0 |          1     |
| ('2W', 'mirror')   |  25 |      5 |     20 |          0 |          0 |                    0 |           0 |          0     |
| ('2W', 'shoulder') |  25 |     20 |      5 |          0 |          0 |                    0 |           0 |          0     |
| ('2W', 'wnull')    |  25 |      0 |      1 |         24 |          0 |                    0 |           0 |          0.96  |
| ('2W', 'x00')      |  75 |     63 |     12 |          0 |          0 |                    0 |           0 |          0     |
| ('2W', 'x01')      |  75 |     63 |     12 |          0 |          0 |                    0 |           0 |          0     |
| ('2W', 'x10')      |  75 |      0 |      5 |         70 |          0 |                    0 |           0 |          0.933 |
| ('2W', 'x11')      |  75 |      0 |     29 |         26 |          0 |                   20 |           0 |          0.613 |
| ('EV', 'wnull')    |   5 |      0 |      0 |          3 |          2 |                    0 |           0 |          1     |
| ('EV', 'x00')      |  15 |     13 |      0 |          0 |          0 |                    0 |           2 |          0     |
| ('EV', 'x10')      |  15 |      0 |      0 |          9 |          6 |                    0 |           0 |          1     |
| ('IT', 'wnull')    |   5 |      2 |      1 |          0 |          2 |                    0 |           0 |          0.4   |
| ('IT', 'x00')      |  15 |     15 |      0 |          0 |          0 |                    0 |           0 |          0     |
| ('IT', 'x10')      |  15 |      6 |      3 |          0 |          6 |                    0 |           0 |          0.4   |

## t_div vs t_vis (x10 vs x00)

| scenario                    |   pairs |   ok |   early |   never_visible |   med_div_minus_vis_s |   med_divlat_minus_vis_s |
|:----------------------------|--------:|-----:|--------:|----------------:|----------------------:|-------------------------:|
| Accident                    |      15 |   15 |       0 |               0 |                  3.7  |                     6.78 |
| AccidentTwoWays             |      15 |   15 |       0 |               0 |                  3.4  |                     6.4  |
| ConstructionObstacle        |      15 |   15 |       0 |               0 |                  3.3  |                     6.1  |
| ConstructionObstacleTwoWays |      15 |   15 |       0 |               0 |                  3.2  |                     6.2  |
| HazardAtSideLane            |      15 |   15 |       0 |               0 |                  3.65 |                     7    |
| HazardAtSideLaneTwoWays     |      15 |   15 |       0 |               0 |                  3.1  |                     5.8  |
| InvadingTurn                |      15 |   12 |       3 |               0 |                  0.6  |                     1.2  |
| ParkedObstacle              |      15 |   15 |       0 |               0 |                  3.3  |                     6.3  |
| ParkedObstacleTwoWays       |      15 |   15 |       0 |               0 |                  3.25 |                     6.4  |
| VehicleOpensDoorTwoWays     |      15 |   15 |       0 |               0 |                  2.45 |                     6.05 |
| YieldToEmergencyVehicle     |      15 |    0 |      12 |               3 |                 -4.5  |                    -1.65 |

smoke 1: x00 |d(k+3 s)| < 0.3 m on 1670 / 1670 x10-bypass frames (1.000; gate >= 0.95), 143 pairs with such frames of 165

## negotiation: x11 - x10 (2W)

| scenario                    |   cases |   wait_share_x11 |   wait_share_x10 |   med_lat_delay_s |   med_dv_x11_minus_x10 |
|:----------------------------|--------:|-----------------:|-----------------:|------------------:|-----------------------:|
| AccidentTwoWays             |      15 |            1     |            0     |            29.4   |                 -6.72  |
| ConstructionObstacleTwoWays |      15 |            1     |            0     |            12.425 |                 -9.485 |
| HazardAtSideLaneTwoWays     |      15 |            0.267 |            0     |            20.2   |                 -0.656 |
| ParkedObstacleTwoWays       |      15 |            0.4   |            0     |             3     |                 -7.186 |
| VehicleOpensDoorTwoWays     |      15 |            0.6   |            0.333 |             2.15  |                 -8.357 |

pooled wait share x11 0.653 (gate >= 0.50), x10 0.067

placement null keep: 40 / 45 = 0.889 (gate >= 0.90)

| scenario                    |   n |   keep |   stop |   bypass_share |
|:----------------------------|----:|-------:|-------:|---------------:|
| Accident                    |   5 |      5 |      0 |              0 |
| AccidentTwoWays             |   5 |      3 |      2 |              0 |
| ConstructionObstacle        |   5 |      5 |      0 |              0 |
| ConstructionObstacleTwoWays |   5 |      5 |      0 |              0 |
| HazardAtSideLane            |   5 |      5 |      0 |              0 |
| HazardAtSideLaneTwoWays     |   5 |      5 |      0 |              0 |
| ParkedObstacle              |   5 |      5 |      0 |              0 |
| ParkedObstacleTwoWays       |   5 |      3 |      2 |              0 |
| VehicleOpensDoorTwoWays     |   5 |      4 |      1 |              0 |

mirror stop: 20 / 25 = 0.800 (gate >= 0.80)

| scenario                    |   n |   keep |   stop |   bypass_share |
|:----------------------------|----:|-------:|-------:|---------------:|
| AccidentTwoWays             |   5 |      0 |      5 |              0 |
| ConstructionObstacleTwoWays |   5 |      0 |      5 |              0 |
| HazardAtSideLaneTwoWays     |   5 |      5 |      0 |              0 |
| ParkedObstacleTwoWays       |   5 |      0 |      5 |              0 |
| VehicleOpensDoorTwoWays     |   5 |      0 |      5 |              0 |

weather null: same world mode as x10 in 55 / 55

## smoke 2: oncoming vehicles within 50 m in the lane-change window (2W)

| world    |   worlds |   with >= 1 oncoming |   mean oncoming |
|:---------|---------:|---------------------:|----------------:|
| mirror   |       25 |                   23 |            7.52 |
| shoulder |       25 |                    1 |            0.04 |
| wnull    |       25 |                    1 |            0.04 |
| x00      |       75 |                    1 |            0.01 |
| x01      |       75 |                   53 |            3.01 |
| x10      |       75 |                    2 |            0.03 |
| x11      |       75 |                   69 |            4.72 |

## frame-level section 2.1 modes (mode_x10, frames from t_vis on)

| cls   |   curve_or_other |   keep |   lane_change |   nudge_hold |   nudge_return |   stop |   turn |
|:------|-----------------:|-------:|--------------:|-------------:|---------------:|-------:|-------:|
| 1W    |              760 |    713 |           718 |            4 |            438 |    775 |    219 |
| 2W    |              345 |    686 |           335 |           19 |           1179 |   1151 |    119 |
| IT    |               10 |     86 |             0 |           67 |              0 |    488 |      0 |

## frame-level section 2.1 modes (mode_x00, frames from t_vis on)

| cls   |   keep |   stop |
|:------|-------:|-------:|
| 1W    |   2256 |      0 |
| 2W    |   2648 |    422 |
| IT    |     97 |      0 |
