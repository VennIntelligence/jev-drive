## A. The target events (R6DAG-L18 s0 / s1)

| read | scene | fclass_primary | t_leave | v_leave | flips_lu | cmdLR_lu | cmd_to_route_lu | cmdH_LR_lu | dth40_lu | arc4_lu | beyond_lu | rh_start | rh_end | y_rt0_end | d_log_end | latcov_lu | mech |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 0e05cb0c | cor_drift | -8.90 | 31.80 | 0 | 1.00 | 1.00 | 0.00 | 0.09 | 127.67 | 1.00 | -3.84 | 0.39 | -3.63 | 4.02 | 3.37 | route in another lane; human merges into it, ego stays |
| R6DAG-s0 | 26004ca8 | cor_lane_change | -2.60 | 22.70 | 1 | 0.95 | 0.95 | 0.00 | 9.40 | 93.41 | 1.00 | -3.12 | -3.11 | 0.86 | -3.96 | 1.00 | route in another lane than the human; ego follows the route |
| R6DAG-s0 | 302c5c99 | cor_straight_at_turn | -1.20 | 4.81 | 0 | 1.00 | 1.00 | 0.00 | 0.56 | 17.24 | 0.20 | 10.22 | 10.44 | nan | -4.84 | 10.65 | route turn not taken |
| R6DAG-s0 | 404cc92f | cor_drift | -6.10 | 16.56 | 0 | 0.00 | nan | 0.00 | 0.24 | 78.47 | 0.71 | -0.65 | -3.45 | 0.53 | -3.98 | 0.60 | human leaves the route's lane; ego stays on the route |
| R6DAG-s0 | 4cb05d65 | cor_drift | -3.40 | 38.53 | 1 | 0.90 | 0.90 | 0.00 | 0.37 | 154.11 | 1.00 | 3.58 | 3.33 | -0.79 | 4.12 | -2.73 | route in another lane than the human; ego follows the route |
| R6DAG-s0 | 4f7cb072 | off_straight | 0.00 | 20.54 | 0 | 1.00 | 1.00 | 0.00 | 5.47 | 84.66 | 0.86 | 3.32 | 3.39 | 4.60 | -1.21 | -2.61 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s0 | 596af0c7 | off_straight | -5.00 | 19.18 | 2 | 0.12 | nan | 0.00 | 9.10 | 80.47 | 0.92 | -0.09 | 0.42 | 3.68 | -3.26 | -0.51 | policy drift, route fine |
| R6DAG-s0 | 7750265d | off_straight | 0.00 | 10.90 | 2 | 0.31 | nan | 0.00 | 11.68 | 44.19 | 0.00 | -0.08 | -0.58 | -0.90 | 0.32 | -0.10 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s0 | 7dc59d0d | cor_drift | -9.40 | 28.28 | 1 | 0.80 | 0.80 | 0.00 | 0.27 | 112.05 | 1.00 | 3.23 | 3.56 | -0.45 | 4.01 | -2.74 | route in another lane than the human; ego follows the route |
| R6DAG-s0 | a5599793 | off_junc_straight | 0.00 | 2.41 | 1 | 0.35 | nan | 0.00 | nan | 9.29 | 0.00 | nan | nan | nan | 0.95 | nan | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s0 | a68ad579 | off_junc_straight | -1.40 | 10.63 | 2 | 0.69 | nan | 0.27 | 16.16 | 44.35 | 0.00 | 0.49 | -0.06 | 0.90 | -0.96 | 0.08 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s0 | baabc57d | off_straight | -0.60 | 9.04 | 1 | 0.86 | nan | 0.00 | 15.35 | 24.15 | 0.00 | -1.24 | -1.30 | 0.22 | -1.52 | 0.65 | command flip (old rule L/R on a non-turn), route fine |
| R6DAG-s0 | e1c6ae9f | off_junc_straight | 0.00 | 13.71 | 2 | 0.53 | nan | 0.00 | 5.61 | 60.05 | 0.00 | -0.92 | -0.22 | 0.55 | -0.77 | 0.81 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s0 | e5e47fd2 | cor_straight_at_turn | -1.40 | 8.49 | 1 | 0.62 | nan | 0.60 | 112.80 | 34.20 | 0.00 | 0.27 | -0.48 | 3.69 | -4.17 | -1.28 | route turn not taken |
| R6DAG-s1 | 0e05cb0c | cor_drift | -9.30 | 31.78 | 0 | 1.00 | 1.00 | 0.00 | 0.09 | 127.29 | 1.00 | -3.83 | 0.40 | -3.61 | 4.01 | 3.29 | route in another lane; human merges into it, ego stays |
| R6DAG-s1 | 26004ca8 | cor_lane_change | -2.90 | 22.54 | 1 | 0.95 | 0.95 | 0.00 | 9.25 | 93.36 | 1.00 | -3.12 | -3.19 | 0.82 | -4.00 | 1.11 | route in another lane than the human; ego follows the route |
| R6DAG-s1 | 302c5c99 | cor_straight_at_turn | -1.30 | 4.50 | 0 | 1.00 | 1.00 | 0.00 | 0.57 | 17.24 | 0.24 | 10.22 | 10.41 | nan | -4.52 | 10.82 | route turn not taken |
| R6DAG-s1 | 4cb05d65 | cor_drift | -3.40 | 38.53 | 1 | 0.90 | 0.90 | 0.00 | 0.37 | 154.11 | 1.00 | 3.58 | 3.33 | -0.81 | 4.14 | -2.69 | route in another lane than the human; ego follows the route |
| R6DAG-s1 | 4f7cb072 | off_straight | -0.10 | 20.79 | 0 | 1.00 | 1.00 | 0.00 | 5.47 | 84.62 | 0.89 | 3.32 | 3.38 | 4.71 | -1.34 | -2.67 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s1 | 7dc59d0d | cor_drift | -8.70 | 29.30 | 1 | 0.80 | 0.80 | 0.00 | 0.24 | 114.68 | 1.00 | 3.23 | 3.50 | -0.51 | 4.01 | -2.85 | route in another lane than the human; ego follows the route |
| R6DAG-s1 | a5599793 | off_junc_straight | 0.00 | 2.87 | 1 | 0.25 | nan | 0.00 | nan | 10.53 | 0.00 | nan | nan | nan | 0.97 | nan | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s1 | a68ad579 | off_junc_straight | -1.50 | 10.75 | 2 | 0.69 | nan | 0.29 | 17.03 | 44.19 | 0.00 | 0.49 | 0.04 | 0.89 | -0.85 | 0.09 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s1 | b661da67 | off_straight | 0.00 | 3.83 | 0 | 0.00 | nan | 0.00 | 5.02 | 12.23 | 0.00 | 0.13 | -0.44 | -0.43 | -0.01 | 0.39 | on the route and the human's path (<= 1.5 m); offroad at a junction / lane edge |
| R6DAG-s1 | e5e47fd2 | cor_straight_at_turn | -1.30 | 8.19 | 1 | 0.65 | nan | 0.63 | 112.83 | 33.89 | 0.00 | 0.27 | -0.54 | 3.45 | -3.99 | -1.18 | route turn not taken |

Scores of every read on the same scenes:

| event_read | scene | NTV0 | R6DAG-s0 | R6DAG-s1 | R7J-s0 | R7J-s1 |
|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 0e05cb0c | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | 26004ca8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | 302c5c99 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | 404cc92f | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| R6DAG-s0 | 4cb05d65 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | 4f7cb072 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | 596af0c7 | 0.61 | 0.00 | 0.62 | 0.00 | 0.00 |
| R6DAG-s0 | 7750265d | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| R6DAG-s0 | 7dc59d0d | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| R6DAG-s0 | a5599793 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | a68ad579 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s0 | baabc57d | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| R6DAG-s0 | e1c6ae9f | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| R6DAG-s0 | e5e47fd2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 0e05cb0c | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 26004ca8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 302c5c99 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 4cb05d65 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 4f7cb072 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | 7dc59d0d | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| R6DAG-s1 | a5599793 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | a68ad579 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| R6DAG-s1 | b661da67 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| R6DAG-s1 | e5e47fd2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Ego ends on the route and > 2.5 m from the human (followed the route into the other lane):

| event_read | scene | NTV0 | R6DAG-s0 | R6DAG-s1 | R7J-s0 | R7J-s1 |
|--:|--:|:--|:--|:--|:--|:--|
| R6DAG-s0 | 0e05cb0c | False | nan | False | False | False |
| R6DAG-s0 | 26004ca8 | True | nan | True | True | True |
| R6DAG-s0 | 302c5c99 | False | nan | False | False | False |
| R6DAG-s0 | 404cc92f | True | nan | True | True | True |
| R6DAG-s0 | 4cb05d65 | False | nan | True | True | True |
| R6DAG-s0 | 4f7cb072 | False | nan | False | False | False |
| R6DAG-s0 | 596af0c7 | False | nan | False | False | False |
| R6DAG-s0 | 7750265d | False | nan | False | False | False |
| R6DAG-s0 | 7dc59d0d | True | nan | True | False | False |
| R6DAG-s0 | a5599793 | False | nan | False | False | False |
| R6DAG-s0 | a68ad579 | False | nan | False | False | False |
| R6DAG-s0 | baabc57d | False | nan | False | False | False |
| R6DAG-s0 | e1c6ae9f | False | nan | False | False | False |
| R6DAG-s0 | e5e47fd2 | False | nan | False | False | False |
| R6DAG-s1 | 0e05cb0c | False | False | nan | False | False |
| R6DAG-s1 | 26004ca8 | True | True | nan | True | True |
| R6DAG-s1 | 302c5c99 | False | False | nan | False | False |
| R6DAG-s1 | 4cb05d65 | True | True | nan | True | True |
| R6DAG-s1 | 4f7cb072 | False | False | nan | False | False |
| R6DAG-s1 | 7dc59d0d | True | True | nan | False | False |
| R6DAG-s1 | a5599793 | False | False | nan | False | False |
| R6DAG-s1 | a68ad579 | False | False | nan | False | False |
| R6DAG-s1 | b661da67 | False | False | nan | False | False |
| R6DAG-s1 | e5e47fd2 | False | False | nan | False | False |

Share of decisions whose stored command points to the route's side (route in another lane only), same human-log stretch:

| event_read | scene | NTV0 | R6DAG-s0 | R6DAG-s1 | R7J-s0 | R7J-s1 |
|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 0e05cb0c | 1.00 | nan | 1.00 | 0.00 | 0.00 |
| R6DAG-s0 | 26004ca8 | 0.68 | nan | 0.65 | 0.00 | 0.00 |
| R6DAG-s0 | 302c5c99 | 1.00 | nan | 1.00 | 0.12 | 0.12 |
| R6DAG-s0 | 4cb05d65 | 0.65 | nan | 0.56 | 0.00 | 0.00 |
| R6DAG-s0 | 4f7cb072 | 1.00 | nan | 1.00 | 0.00 | 0.00 |
| R6DAG-s0 | 7dc59d0d | 0.32 | nan | 0.29 | 0.00 | 0.00 |
| R6DAG-s1 | 0e05cb0c | 1.00 | 1.00 | nan | 0.00 | 0.00 |
| R6DAG-s1 | 26004ca8 | 0.65 | 0.65 | nan | 0.00 | 0.00 |
| R6DAG-s1 | 302c5c99 | 1.00 | 1.00 | nan | 0.10 | 0.10 |
| R6DAG-s1 | 4cb05d65 | 0.64 | 0.55 | nan | 0.00 | 0.00 |
| R6DAG-s1 | 4f7cb072 | 1.00 | 1.00 | nan | 0.00 | 0.00 |
| R6DAG-s1 | 7dc59d0d | 0.33 | 0.32 | nan | 0.00 | 0.00 |

## B. Plan 4 s arc vs route coverage, every decision (v >= 0.5 m/s)

| read | speed | n | arc4_p50 | arc4_p90 | cov_p50 | cov_lt72 | no_slot_at_ego | beyond | beyond10 | beyond_m_p50 | t_exit_p50 | arc4_gt72 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 0.5-5 | 9964 | 11.58 | 24.95 | 72.00 | 9.81 | 55.80 | 0.49 | 0.27 | 0.00 | 1.86 | 0.00 |
| R6DAG-s0 | 5-10 | 16034 | 29.73 | 40.75 | 72.00 | 2.75 | 31.31 | 0.00 | 0.00 | 0.00 | nan | 0.00 |
| R6DAG-s0 | 10-15 | 12056 | 46.36 | 59.25 | 72.00 | 1.20 | 24.49 | 0.20 | 0.09 | 0.00 | 3.35 | 0.00 |
| R6DAG-s0 | 15-20 | 4923 | 68.65 | 78.82 | 72.00 | 0.00 | 12.41 | 32.32 | 3.11 | 0.00 | 3.74 | 32.32 |
| R6DAG-s0 | 20-25 | 4019 | 88.06 | 97.09 | 72.00 | 0.00 | 9.98 | 94.08 | 76.54 | 16.06 | 3.25 | 94.08 |
| R6DAG-s0 | 25-45 | 4573 | 118.66 | 147.23 | 72.00 | 0.00 | 7.76 | 99.83 | 99.48 | 46.66 | 2.42 | 99.83 |
| R6DAG-s1 | 0.5-5 | 9741 | 11.76 | 24.97 | 72.00 | 10.72 | 58.31 | 0.52 | 0.31 | 0.00 | 1.88 | 0.00 |
| R6DAG-s1 | 5-10 | 15971 | 29.55 | 40.85 | 72.00 | 2.49 | 31.21 | 0.03 | 0.01 | 0.00 | 2.80 | 0.00 |
| R6DAG-s1 | 10-15 | 12234 | 46.44 | 59.09 | 72.00 | 1.19 | 24.02 | 0.27 | 0.16 | 0.00 | 3.10 | 0.00 |
| R6DAG-s1 | 15-20 | 5060 | 68.62 | 78.88 | 72.00 | 0.00 | 12.35 | 33.40 | 3.28 | 0.00 | 3.74 | 33.40 |
| R6DAG-s1 | 20-25 | 3998 | 88.14 | 97.13 | 72.00 | 0.00 | 10.03 | 94.80 | 78.86 | 16.14 | 3.25 | 94.80 |
| R6DAG-s1 | 25-45 | 4590 | 121.80 | 150.97 | 72.00 | 0.00 | 7.73 | 99.93 | 99.41 | 49.80 | 2.36 | 99.93 |

Command one-hot by segment (stored = what the model was fed; cmdH = the heading rule on the same memory; R7J was fed cmdH):

| read | segment | n | cmd_L | cmd_R | cmdH_L | cmdH_R | flips_per_min |
|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | straight (|dth40| < 3), v >= 15 | 10110 | 6.56 | 5.24 | 0.00 | 0.00 | 1.66 |
| R6DAG-s0 | straight, 10 <= v < 15 | 6168 | 1.86 | 8.24 | 0.00 | 0.00 | nan |
| R6DAG-s0 | straight, v < 10 | 7178 | 5.35 | 9.21 | 0.00 | 0.00 | nan |
| R6DAG-s0 | gentle (3-15), v >= 15 | 1950 | 15.18 | 10.26 | 0.00 | 0.00 | nan |
| R6DAG-s0 | no near slot (route gap at the ego) | 14900 | 15.42 | 19.80 | 8.66 | 10.91 | nan |
| R6DAG-s1 | straight (|dth40| < 3), v >= 15 | 10274 | 5.49 | 5.35 | 0.00 | 0.00 | 1.69 |
| R6DAG-s1 | straight, 10 <= v < 15 | 6164 | 1.77 | 8.61 | 0.00 | 0.00 | nan |
| R6DAG-s1 | straight, v < 10 | 6931 | 6.36 | 10.06 | 0.00 | 0.00 | nan |
| R6DAG-s1 | gentle (3-15), v >= 15 | 1900 | 12.74 | 13.37 | 0.00 | 0.00 | nan |
| R6DAG-s1 | no near slot (route gap at the ego) | 14984 | 15.81 | 20.52 | 8.77 | 10.99 | nan |
| NTV0 | straight (|dth40| < 3), v >= 15 | 9023 | 8.13 | 6.08 | 0.00 | 0.00 | 1.40 |
| NTV0 | straight, 10 <= v < 15 | 7170 | 1.97 | 19.22 | 0.00 | 0.00 | nan |
| NTV0 | straight, v < 10 | 7137 | 4.20 | 12.78 | 0.00 | 0.00 | nan |
| NTV0 | gentle (3-15), v >= 15 | 1607 | 12.01 | 10.70 | 0.00 | 0.00 | nan |
| NTV0 | no near slot (route gap at the ego) | 15338 | 15.64 | 22.47 | 8.48 | 11.56 | nan |
| R7J-s0 | straight (|dth40| < 3), v >= 15 | 10396 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 |
| R7J-s0 | straight, 10 <= v < 15 | 6482 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s0 | straight, v < 10 | 6324 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s0 | gentle (3-15), v >= 15 | 2000 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s0 | no near slot (route gap at the ego) | 14742 | 8.73 | 10.77 | 8.73 | 10.77 | nan |
| R7J-s1 | straight (|dth40| < 3), v >= 15 | 10638 | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 |
| R7J-s1 | straight, 10 <= v < 15 | 6272 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s1 | straight, v < 10 | 6376 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s1 | gentle (3-15), v >= 15 | 2057 | 0.00 | 0.00 | 0.00 | 0.00 | nan |
| R7J-s1 | no near slot (route gap at the ego) | 14664 | 8.57 | 10.85 | 8.57 | 10.85 | nan |

Plan lateral vs the route at the last covered arc (straight route, + = plan left of route):

| read | speed | n | latcov_p50 | abs_latcov_p90 | abs_lat4_p90_within | on_route_n | on_route_abs_latcov_p90 | on_route_latcov_gt1 | route_side_ge2p5 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R6DAG-s0 | 0.5-5 | 2175 | 0.11 | 1.25 | 1.14 | 1947 | 0.72 | 5.60 | 4.87 |
| R6DAG-s0 | 5-10 | 5003 | 0.19 | 1.72 | 1.72 | 3807 | 0.98 | 9.69 | 2.06 |
| R6DAG-s0 | 10-15 | 6168 | 0.23 | 1.77 | 1.77 | 4757 | 1.10 | 10.93 | 0.68 |
| R6DAG-s0 | 15-20 | 3109 | 0.16 | 1.85 | 1.95 | 2584 | 0.58 | 3.17 | 4.66 |
| R6DAG-s0 | 20-25 | 3034 | 0.22 | 1.88 | 3.15 | 2593 | 0.75 | 4.97 | 9.53 |
| R6DAG-s0 | 25-45 | 3967 | 0.15 | 3.30 | 2.91 | 3136 | 0.70 | 2.23 | 14.67 |
| R6DAG-s1 | 0.5-5 | 1901 | 0.11 | 1.51 | 1.30 | 1709 | 0.76 | 4.92 | 5.73 |
| R6DAG-s1 | 5-10 | 5030 | 0.28 | 1.83 | 1.83 | 3702 | 1.09 | 11.32 | 3.18 |
| R6DAG-s1 | 10-15 | 6164 | 0.29 | 1.83 | 1.83 | 4551 | 1.02 | 10.22 | 2.00 |
| R6DAG-s1 | 15-20 | 3259 | 0.17 | 1.62 | 1.77 | 2771 | 0.60 | 4.40 | 2.09 |
| R6DAG-s1 | 20-25 | 3040 | 0.19 | 1.92 | 3.65 | 2591 | 0.74 | 5.13 | 9.61 |
| R6DAG-s1 | 25-45 | 3975 | 0.16 | 3.20 | 2.81 | 3266 | 0.76 | 2.79 | 14.31 |

