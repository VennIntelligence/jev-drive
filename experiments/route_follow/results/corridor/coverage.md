# Route-lane corridor labels: coverage (rf_corridor.py build)

Found = the route's lane chain covers >= 90 % of the 1 m route samples in [s_e, s_e + 40] m (s_e = the ego's route station at the decision); ambiguous = a route sample in the corridor window had two distinct candidate lanes (resolved by the human's path); found_h = the same at the human's own station (used by the label sanity check).

| population | source | decisions | rollouts | scenes | found | ambiguous | found (human station) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| all decisions | held | 48774 | 2712 | 452 | 0.980 | 0.019 | 0.980 |
| all decisions | pai | 55693 | 281 | 281 | 1.000 | 0.357 | 0.999 |
| all decisions | set held_APP1X_s0 | 8129 | 452 | 452 | 0.980 | 0.019 | 0.980 |
| all decisions | set held_RBSG2_s0 | 8129 | 452 | 452 | 0.979 | 0.018 | 0.980 |
| all decisions | set held_RSG_s0 | 8129 | 452 | 452 | 0.981 | 0.019 | 0.980 |
| all decisions | set held_RSG_s1 | 8129 | 452 | 452 | 0.980 | 0.019 | 0.980 |
| all decisions | set held_RV2c_s0 | 8129 | 452 | 452 | 0.980 | 0.019 | 0.980 |
| all decisions | set held_RV2c_s1 | 8129 | 452 | 452 | 0.979 | 0.019 | 0.980 |
| all decisions | set pai | 55693 | 281 | 281 | 1.000 | 0.357 | 0.999 |
| strict-turn window (ego) | held | 2108 | 144 | 24 | 0.961 | 0.000 | 0.952 |
| strict-turn window (ego) | pai | 3755 | 35 | 35 | 1.000 | 0.699 | 1.000 |
| strict-turn window (ego) | set held_APP1X_s0 | 361 | 24 | 24 | 0.956 | 0.000 | 0.947 |
| strict-turn window (ego) | set held_RBSG2_s0 | 347 | 24 | 24 | 0.960 | 0.000 | 0.951 |
| strict-turn window (ego) | set held_RSG_s0 | 360 | 24 | 24 | 0.961 | 0.000 | 0.953 |
| strict-turn window (ego) | set held_RSG_s1 | 367 | 24 | 24 | 0.967 | 0.000 | 0.954 |
| strict-turn window (ego) | set held_RV2c_s0 | 337 | 24 | 24 | 0.961 | 0.000 | 0.953 |
| strict-turn window (ego) | set held_RV2c_s1 | 336 | 24 | 24 | 0.961 | 0.000 | 0.952 |
| strict-turn window (ego) | set pai | 3755 | 35 | 35 | 1.000 | 0.699 | 1.000 |

| set | rollouts | with a lane change in the chain | with a break | with ambiguous samples | first decision - first controller sample (s, median) |
|:--|--:|--:|--:|--:|--:|
| held_APP1X_s0 | 452 | 159 | 1 | 15 | -0.10 |
| held_RBSG2_s0 | 452 | 161 | 1 | 15 | -0.10 |
| held_RSG_s0 | 452 | 158 | 1 | 15 | -0.10 |
| held_RSG_s1 | 452 | 159 | 1 | 15 | -0.10 |
| held_RV2c_s0 | 452 | 162 | 1 | 15 | -0.10 |
| held_RV2c_s1 | 452 | 161 | 1 | 15 | -0.10 |
| pai | 281 | 99 | 0 | 150 | -0.10 |

Rollouts without plans or lanes: {}; job errors: 0.
