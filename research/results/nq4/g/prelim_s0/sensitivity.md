# Seed-0 sensitivity: following windows kept (descriptive, not the registered readout)

## Reaction rate with following windows kept

| cand        | variant   |   trigger_rate_all |    lo |    hi |   routes |   control_rate_all |   c_lo |   c_hi |   c_routes |   trig_minus_ctrl |   p_lo |   p_hi |   p_routes |
|:------------|:----------|-------------------:|------:|------:|---------:|-------------------:|-------:|-------:|-----------:|------------------:|-------:|-------:|-----------:|
| blue        | ghost     |              0.287 | 0.188 | 0.388 |       80 |              0.185 |  0.037 |  0.333 |         27 |             0.000 | -0.148 |  0.148 |         27 |
| bridgedrive | ghost     |              0.188 | 0.113 | 0.275 |       80 |              0.111 |  0.000 |  0.222 |         27 |             0.000 | -0.185 |  0.185 |         27 |
| pdm         | ghost     |              0.163 | 0.087 | 0.250 |       80 |              0.037 |  0.000 |  0.111 |         27 |             0.000 | -0.111 |  0.111 |         27 |
| simlingo    | ghost     |              0.300 | 0.200 | 0.400 |       80 |              0.185 |  0.037 |  0.333 |         27 |            -0.037 | -0.222 |  0.148 |         27 |
| tfv6        | ghost     |              0.225 | 0.138 | 0.325 |       80 |              0.074 |  0.000 |  0.185 |         27 |             0.000 | -0.148 |  0.148 |         27 |
| tfv6        | orig      |              0.875 | 0.800 | 0.938 |       80 |              0.407 |  0.222 |  0.593 |         27 |             0.556 |  0.370 |  0.741 |         27 |

PDM-Lite ghost trigger rate with following kept: 0.163

## Ghost world: mean speed in the control windows (all reached windows, m/s)

| cand        |   ghost_control_mean_mps |   routes |
|:------------|-------------------------:|---------:|
| blue        |                    10.52 |    27.00 |
| bridgedrive |                     7.65 |    27.00 |
| pdm         |                    10.84 |    27.00 |
| simlingo    |                    11.02 |    27.00 |
| tfv6        |                     7.42 |    27.00 |

## TFv6 trigger window, orig vs ghost, following kept (paired by route)

|   routes |   orig_rate |   ghost_rate |   orig_minus_ghost |    lo |    hi |   orig_decel_med |   ghost_decel_med |   orig_lat_med |   ghost_lat_med |   orig_ventry_med |   ghost_ventry_med |
|---------:|------------:|-------------:|-------------------:|------:|------:|-----------------:|------------------:|---------------:|----------------:|------------------:|-------------------:|
|   80.000 |       0.875 |        0.225 |              0.650 | 0.537 | 0.750 |            6.448 |             1.082 |          0.946 |           0.024 |             6.847 |              7.523 |

## Ghost trigger reactions, following kept

| cand        | routes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| blue        | 24816(obstacle_bypass) 25854(obstacle_bypass) 25896(obstacle_bypass) 26950(junction_violator) 27515(vru_crossing) 26944(junction_violator) 24759(cut_in) 25318(obstacle_bypass) 25955(obstacle_bypass) 24367(obstacle_bypass) 24757(obstacle_bypass) 1852(obstacle_bypass) 2082(junction_violator) 2164(vru_crossing) 2539(obstacle_bypass) 2643(obstacle_bypass) 2664(obstacle_bypass) 2844(junction_violator) 2847(junction_violator) 3410(obstacle_bypass) 3737(vru_crossing) 24333(vru_emerging) 25845(obstacle_bypass)     |
| bridgedrive | 26950(junction_violator) 26944(junction_violator) 25955(obstacle_bypass) 24240(obstacle_bypass) 25865(obstacle_bypass) 11381(vru_crossing) 1773(obstacle_bypass) 2082(junction_violator) 2164(vru_crossing) 2509(obstacle_bypass) 2513(obstacle_bypass) 2844(junction_violator) 2847(junction_violator) 3307(obstacle_bypass) 3731(vru_crossing)                                                                                                                                                                                |
| pdm         | 24206(vru_emerging) 24816(obstacle_bypass) 26950(junction_violator) 26944(junction_violator) 25857(obstacle_bypass) 10857(vru_crossing) 2082(junction_violator) 2534(obstacle_bypass) 2554(obstacle_bypass) 2844(junction_violator) 2847(junction_violator) 3307(obstacle_bypass) 3737(vru_crossing)                                                                                                                                                                                                                            |
| simlingo    | 24816(obstacle_bypass) 25854(obstacle_bypass) 25896(obstacle_bypass) 26950(junction_violator) 27515(vru_crossing) 26944(junction_violator) 24759(cut_in) 25955(obstacle_bypass) 24757(obstacle_bypass) 25358(cut_in) 24252(vru_emerging) 25865(obstacle_bypass) 11381(vru_crossing) 1852(obstacle_bypass) 2082(junction_violator) 2539(obstacle_bypass) 2664(obstacle_bypass) 2844(junction_violator) 2847(junction_violator) 3436(obstacle_bypass) 3737(vru_crossing) 24333(vru_emerging) 25845(obstacle_bypass) 26405(cut_in) |
| tfv6        | 24211(vru_emerging) 26950(junction_violator) 27515(vru_crossing) 26944(junction_violator) 27529(vru_crossing) 24240(obstacle_bypass) 25424(obstacle_bypass) 10857(vru_crossing) 11381(vru_crossing) 18252(vru_emerging) 2082(junction_violator) 2164(vru_crossing) 2643(obstacle_bypass) 2844(junction_violator) 2847(junction_violator) 3436(obstacle_bypass) 3731(vru_crossing) 3737(vru_crossing)                                                                                                                            |
