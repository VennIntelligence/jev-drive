# STEER-LONG read (pre-registration: research/next-round/steer-long.md section 3, commit 8f48d0c0)

## Gates

| track / label | summary rollouts | with controller + driver | G-frame | G-rec |
|:--|--:|--:|--:|--:|
| nuplan/YR10m10-F-s0 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| nuplan/YR10m10-F-s1 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| nuplan/YS-L18-s0 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| nuplan/YS-L18-s1 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| nuplan/YS2G-L18-s0 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| nuplan/YS2G-L18-s1 | 460 | 452 | 7653.0 / 7677 (99.7 %) | 40169.0 / 40193 (99.9 %) |
| pai/ybase | 281 | 281 | 55412.0 / 55412 (100.0 %) | 55412.0 / 55412 (100.0 %) |
| pai/ys0 | 281 | 281 | 55412.0 / 55412 (100.0 %) | 55412.0 / 55412 (100.0 %) |
| pai/ys1 | 281 | 281 | 55412.0 / 55412 (100.0 %) | 55412.0 / 55412 (100.0 %) |

G-ev: zeros_mech zero rollouts 316, this read 316, mismatched rows 0

## Thresholds (p99 of clean-rollout rows, v >= 1 m/s)

| track / bucket | n rows | A_path | A_cmd | A_exec | A_req | ddelta (rad/s) |
|:--|--:|--:|--:|--:|--:|--:|
| nuplan/high | 427 | 0.47 | 0.67 | 0.50 | 0.43 | 0.011 |
| nuplan/low | 89516 | 2.19 | 2.04 | 1.96 | 1.77 | 0.810 |
| nuplan/mid | 108640 | 2.22 | 2.59 | 2.43 | 2.11 | 0.217 |
| pai/high | 19778 | 1.58 | 9.34 | 5.03 | 3.63 | 0.153 |
| pai/low | 28458 | 0.88 | 2.08 | 1.89 | 1.99 | 1.016 |
| pai/mid | 46762 | 1.38 | 3.85 | 3.45 | 3.17 | 0.382 |

## Zero events: who moved first (section 3.5)

| track | bucket | n | plan_first | steer_first_amp | steer_first_ref | quiet | spin_like | loop_O | loop_R | loop_R2deg | amp_median |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nuplan | low | 235 | 55 | 1 | 7 | 172 | 23 | 8 | 98 | 97 | 1.23 |
| nuplan | mid | 57 | 12 | 0 | 7 | 38 | 2 | 0 | 7 | 7 | 1.31 |
| pai | high | 83 | 7 | 32 | 0 | 44 | 35 | 3 | 9 | 8 | 5.72 |
| pai | low | 106 | 30 | 0 | 7 | 69 | 10 | 1 | 40 | 38 | 1.38 |
| pai | mid | 106 | 23 | 1 | 0 | 82 | 2 | 7 | 18 | 15 | 1.84 |

- nuplan / low: plan first 23.4 %; steer first amplified 0.4 %
  low-speed spin-like events 23: O or R 56.5 % -> HL3 visible
- nuplan / mid: plan first 21.1 %; steer first amplified 0.0 %
- pai / high: plan first 8.4 %; steer first amplified 38.6 %
- pai / low: plan first 28.3 %; steer first amplified 0.0 %
  low-speed spin-like events 10: O or R 70.0 % -> HL3 visible
- pai / mid: plan first 21.7 %; steer first amplified 0.9 %

By zero type:

| track | kind | plan_first | quiet | steer_first_amp | steer_first_ref |
|:--|--:|--:|--:|--:|--:|
| nuplan | collision | 6 | 38 | 0 | 2 |
| nuplan | corridor | 38 | 96 | 1 | 6 |
| nuplan | offroad | 23 | 76 | 0 | 6 |
| pai | collision | 9 | 30 | 0 | 0 |
| pai | corridor | 31 | 127 | 10 | 3 |
| pai | offroad | 20 | 38 | 23 | 4 |

Event window maxima (median [p90]) by track x bucket:

| track / bucket | A_path | A_cmd | A_exec | ddelta | v at event |
|:--|--:|--:|--:|--:|--:|
| nuplan / low | 1.18 [2.42] | 1.11 [2.45] | 0.99 [2.32] | 0.24 [0.90] | 3.77 [5.48] |
| nuplan / mid | 1.76 [2.48] | 1.75 [2.96] | 1.61 [2.75] | 0.13 [0.25] | 5.99 [7.19] |
| pai / high | 0.72 [1.48] | 5.10 [486.50] | 2.98 [98.03] | 0.11 [4.83] | 20.30 [32.59] |
| pai / low | 0.57 [1.33] | 0.72 [2.45] | 0.54 [1.65] | 0.55 [1.61] | 3.35 [5.32] |
| pai / mid | 0.33 [1.94] | 0.88 [3.64] | 0.59 [2.89] | 0.06 [0.25] | 12.21 [14.64] |

Exploratory (amendment 1): the same rule with A_delta (steering state) in place of A_cmd:

| track | bucket | plan_first | quiet | steer_first_amp | steer_first_ref |
|:--|--:|--:|--:|--:|--:|
| nuplan | low | 55 | 172 | 1 | 7 |
| nuplan | mid | 12 | 38 | 0 | 7 |
| pai | high | 7 | 44 | 32 | 0 |
| pai | low | 30 | 69 | 1 | 6 |
| pai | mid | 24 | 82 | 0 | 0 |

## High-speed large-steer episodes (v >= 15 m/s, |vx wz| >= 3 m/s^2 for >= 0.3 s, before any event)

| track | km_high | episodes | ep_to_zero | per_100km |
|:--|--:|--:|--:|--:|
| nuplan | 0.69 | 0 | 0 | 0.00 |
| pai | 49.97 | 73 | 38 | 146.08 |

nuPlan 2 Hz, human log vs ego (YR10m10-F-s0 visits; ego of each label): share of 0.5 s steps with a_lat >= 3 m/s^2

| v bucket | human n | human share | ego (all six labels) n | ego share |
|:--|--:|--:|--:|--:|
| low | 2387 | 0.0 % | 26693 | 0.0 % |
| mid | 4717 | 0.2 % | 29979 | 0.1 % |
| high | 0 | nan % | 92 | 0.0 % |

## Cap grid (section 3.7)

navhard tokens with a PDM reference: 5912; human 2 Hz steps (v >= 1): 7104

| track | a_max | j_max | coll_clean | coll_clean_low | coll_clean_mid | coll_clean_high | coll_human | coll_navhard | eligible | events | cover | high_plan_first | cover_high_plan_first |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nuplan | 2.000 | 2.500 | 0.096 | 0.157 | 0.038 | 0.000 | 0.023 | 0.026 | False | 292 | 17 | 0 | 0 |
| pai | 2.000 | 2.500 | 0.225 | 0.356 | 0.176 | 0.064 | 0.023 | 0.026 | False | 295 | 43 | 7 | 3 |
| nuplan | 2.000 | 5.000 | 0.072 | 0.119 | 0.027 | 0.000 | 0.023 | 0.026 | False | 292 | 12 | 0 | 0 |
| pai | 2.000 | 5.000 | 0.158 | 0.267 | 0.107 | 0.046 | 0.023 | 0.026 | False | 295 | 20 | 7 | 0 |
| nuplan | 2.000 | nan | 0.061 | 0.096 | 0.027 | 0.000 | 0.023 | 0.026 | False | 292 | 11 | 0 | 0 |
| pai | 2.000 | nan | 0.047 | 0.072 | 0.031 | 0.034 | 0.023 | 0.026 | False | 295 | 19 | 7 | 0 |
| nuplan | 3.000 | 2.500 | 0.078 | 0.147 | 0.013 | 0.000 | 0.001 | 0.008 | False | 292 | 17 | 0 | 0 |
| pai | 3.000 | 2.500 | 0.220 | 0.355 | 0.170 | 0.051 | 0.001 | 0.008 | False | 295 | 42 | 7 | 3 |
| nuplan | 3.000 | 5.000 | 0.052 | 0.105 | 0.001 | 0.000 | 0.001 | 0.008 | False | 292 | 8 | 0 | 0 |
| pai | 3.000 | 5.000 | 0.151 | 0.267 | 0.099 | 0.032 | 0.001 | 0.008 | False | 295 | 21 | 7 | 0 |
| nuplan | 3.000 | nan | 0.034 | 0.071 | 0.000 | 0.000 | 0.001 | 0.008 | False | 292 | 3 | 0 | 0 |
| pai | 3.000 | nan | 0.031 | 0.055 | 0.016 | 0.016 | 0.001 | 0.008 | False | 295 | 12 | 7 | 0 |
| nuplan | 4.000 | 2.500 | 0.079 | 0.149 | 0.013 | 0.000 | 0.000 | 0.003 | False | 292 | 19 | 0 | 0 |
| pai | 4.000 | 2.500 | 0.217 | 0.355 | 0.165 | 0.047 | 0.000 | 0.003 | False | 295 | 42 | 7 | 3 |
| nuplan | 4.000 | 5.000 | 0.052 | 0.106 | 0.001 | 0.000 | 0.000 | 0.003 | False | 292 | 7 | 0 | 0 |
| pai | 4.000 | 5.000 | 0.148 | 0.266 | 0.093 | 0.027 | 0.000 | 0.003 | False | 295 | 21 | 7 | 0 |
| nuplan | 4.000 | nan | 0.031 | 0.064 | 0.000 | 0.000 | 0.000 | 0.003 | False | 292 | 1 | 0 | 0 |
| pai | 4.000 | nan | 0.023 | 0.049 | 0.007 | 0.010 | 0.000 | 0.003 | False | 295 | 9 | 7 | 0 |
| nuplan | 5.000 | 2.500 | 0.080 | 0.150 | 0.013 | 0.000 | 0.000 | 0.002 | False | 292 | 19 | 0 | 0 |
| pai | 5.000 | 2.500 | 0.217 | 0.355 | 0.165 | 0.046 | 0.000 | 0.002 | False | 295 | 42 | 7 | 3 |
| nuplan | 5.000 | 5.000 | 0.053 | 0.109 | 0.001 | 0.000 | 0.000 | 0.002 | False | 292 | 6 | 0 | 0 |
| pai | 5.000 | 5.000 | 0.147 | 0.266 | 0.091 | 0.024 | 0.000 | 0.002 | False | 295 | 21 | 7 | 0 |
| nuplan | 5.000 | nan | 0.029 | 0.061 | 0.000 | 0.000 | 0.000 | 0.002 | False | 292 | 0 | 0 | 0 |
| pai | 5.000 | nan | 0.020 | 0.045 | 0.003 | 0.005 | 0.000 | 0.002 | True | 295 | 6 | 7 | 0 |
- nuplan: no eligible cap
- pai: chosen a_max 5.0, j_max nan: covers 0 / 7 high-speed plan-first events (0.0 %) -> below the 25 % line: exploratory arm only; all events 6 / 295

Slowness, stops, onsets, rear impacts, corridor positions: sl_long.md (amendment 1 replaced the first-pass log alignment).
