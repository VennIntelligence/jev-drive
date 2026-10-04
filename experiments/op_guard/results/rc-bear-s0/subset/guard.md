# Guard set: rc-bear-s0 (subset mode)

Verdict: **FAIL** (8 pass, 2 fail, 33 not evaluable). 2026-10-05 03:57:43

| line | readout | candidate | shipped | delta | rule | result | note |
|---|---|---:|---:|---:|---|:--|---|
| navtest (subset) | PDMS v1 (subset, 4026 tokens) | 84.258 | 84.03 | 0.228 [-0.13, 0.614] | not below shipped - 0.3 | pass | 4026 / 4026 tokens valid in both; paired mean delta, 95% CI log-cluster bootstrap over 136 logs; sub-score deltas (pp) NC +0.04 DAC +0.40 EP +0.24 TTC -0.12 C +0.00 |
| navhard (subset) | EPDMS two-stage (5912 tokens) | 32.897 | 33.331 | -0.434 [-1.673, 0.837] | no drop | **FAIL** | devkit harness (official aggregation; no official CSV in subset mode); harness - official combined: shipped -0.0000; CI paired over 225 mapping groups |
| navhard (subset) | EPDMS stage 1 | 71.67 | 71.704 | -0.034 [-1.682, 1.731] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS stage 2 | 46.166 | 46.897 | -0.731 [-1.818, 0.327] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS early-turn set (stages pooled) | 43.586 | 43.888 | -0.303 [-0.655, 0.044] | no drop | **FAIL** | 3852 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 220 mapping groups |
| navhard (subset) | EPDMS early-turn stage 1 | 66.974 | 66.724 | 0.25 [-2.581, 3.095] |  | n/a | 231 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 128 mapping groups; diagnostic (no rule) |
| navhard (subset) | EPDMS early-turn stage 2 | 42.094 | 42.432 | -0.338 [-0.672, -0.006] |  | n/a | 3621 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 219 mapping groups; diagnostic (no rule) |
| wod (subset) | RFS, 479 rater frames (serving ONNX) | 8.07 | 8.005 | 0.065 [-0.004, 0.134] | RFS no drop (paired 95% upper bound >= 0) | pass | segment bootstrap B=2000; shipped vs itself = 0 |
| wod (subset) | false-start rate, WOD val stay rows (port, plan >= 3 m at 4 s) | 0.043 | 0.041 | 0.002 [-0.002, 0.005] | false start <= shipped + 2 pp | pass | n 4116 rows / 144 segments; op_adapt_L/H metric and rows |
| wod (subset) | false-start rate, the stay frames among the rater frames (serving) | 0.091 | 0.045 | 0.045 | info (n too small for a 2 pp line) | n/a | n 22 frames (1 frame = 4.5 pp) |
| hugsim (subset) | spins (heading err >= 60 deg) | 0 | 0 | 0 | not up vs shipped | pass | 11 scenes |
| hugsim (subset) | stuck (end max_steps) | 4 | 6 | -2 | reported (stuck: reduction is the goal) | n/a |  |
| hugsim (subset) | completes | 4 | 3 | 1 | reported | n/a |  |
| hugsim (subset) | HD-Score mean (paired over scenes) | 0.479 | 0.544 | -0.065 [-0.234, 0.096] | reported | n/a | paired mean diff -0.065 |
| b2d_turns (subset) | took exit, choice turns | 0.077 | 0 | 0.077 [0, 0.231] | primary readout (up is the goal) | n/a | 1 / 13 vs 0 / 13 |
| b2d_turns (subset) | took exit, forced turns | 0.25 | 0.083 | 0.167 [0, 0.455] | primary readout (up is the goal) | n/a | 3 / 12 vs 1 / 12 |
| b2d_turns (subset) | took exit, all turns | 0.16 | 0.04 | 0.12 [0, 0.304] | primary readout (up is the goal) | n/a | 4 / 25 vs 1 / 25 |
| b2d_turns (subset) | leaves lane, share of entered turns | 0.857 | 0.9 | -0.043 | reported | n/a | 21 entered |
| b2d_turns (subset) | collisions in turn windows | 5 | 5 | 0 | not up vs shipped | pass | 25 turns |
| b2d_turns (subset) | collisions, whole runs (20 routes) | 17 | 21 | -4 | reported | n/a |  |
| b2d_turns (subset) | route DS mean, zones off | 36.215 | 26.287 | 9.928 | reported | n/a |  |
| b2d_ds (subset) | DS mean (paired over seed x route) | 66.822 | 69.265 | -2.443 [-10.779, 6.642] | delta >= -7.6 (no drop beyond decision-38 noise) | pass | n 19, paired diff -2.44 |
| b2d_ds (subset) | RC mean | 86.108 | 78.571 | 7.537 | reported | n/a |  |
| b2d_ds (subset) | collisions (sum) | 4 | 5 | -1 | reported | n/a |  |
| b2d_ds (subset) | red lights (sum) | 8 | 3 | 5 | reported | n/a |  |
| b2d_ds (subset) | routes completed | 15 | 13 | 2 | reported | n/a |  |
| drift (subset) | 4 s lateral drift vs shipped, max over sets / kinds (median per set) | 0.038 | 0 | 0.038 | <= 0.10 m | pass | max at carladev/junction |
| drift (subset) | 4 s lateral drift median, navdev junction (nav) | 0.025 | 0 | 0.025 | info | n/a | n 136; p90 0.108; plan_drift (0-5 s L2) median 0.115 |
| drift (subset) | 4 s lateral drift median, navdev straight (nav) | 0.017 | 0 | 0.017 | info | n/a | n 30; p90 0.072; plan_drift (0-5 s L2) median 0.097 |
| drift (subset) | 4 s lateral drift median, carladev junction (carla) | 0.038 | 0 | 0.038 | info | n/a | n 75; p90 0.386; plan_drift (0-5 s L2) median 0.178 |
| drift (subset) | 4 s lateral drift median, dist:nav all (nav) | 0.011 | 0 | 0.011 | info | n/a | n 153; p90 0.093; plan_drift (0-5 s L2) median 0.092 |
| drift (subset) | 4 s lateral drift median, dist:wod all (wod) | 0.011 | 0 | 0.011 | info | n/a | n 380; p90 0.093; plan_drift (0-5 s L2) median 0.087 |
| drift (subset) | 4 s lateral drift median, dist:carla all (carla) | 0.012 | 0 | 0.012 | info | n/a | n 333; p90 0.049; plan_drift (0-5 s L2) median 0.085 |
| drift (subset) | plan_drift (0-5 s mean L2, x and y) median, max over sets / kinds | 0.178 | 0 | 0.178 | info: the metric op_img_cmd's 0.10 m line was set on | n/a | max at carladev.junction |
| negatives (subset) | |y(4 s) - logged| on the negative frames, mean (m) | 0.743 | 0.779 | -0.036 [-0.07, -0.003] | <= original + 0.3 m | pass | offset under the negative command (adapter) vs shipped without a command; n 158, 135 of 158 await the visual check (all used) |
| negatives (subset) | same, the map-certain N1 tier-A rows (no visual check needed) | 0.486 | 0.551 | -0.065 | info | n/a | n 23 |
| negatives (subset) | same, N1_exit | 0.721 | 0.771 | -0.05 | info | n/a | n 58 |
| negatives (subset) | same, N2_side | 0.455 | 0.457 | -0.002 | info | n/a | n 40 |
| negatives (subset) | same, N3_wrong | 1.262 | 1.332 | -0.069 | info | n/a | n 30 |
| negatives (subset) | same, N4_uturn | 0.651 | 0.673 | -0.023 | info | n/a | n 30 |
| negatives (subset) | offset with command 'none' | 0.773 | 0.779 | -0.007 | info | n/a |  |
| negatives (subset) | offset with command 'correct' | 0.753 | 0.779 | -0.027 | info | n/a |  |
| negatives (subset) | |y_negative - y_none| at 4 s, mean | 0.141 | 0 | 0.141 | info | n/a |  |

Runtime per line (min): navtest 8.6, navhard 12.4, wod 6.4, hugsim 40.0, b2d_turns 39.5, b2d_ds 0.0, drift 0.3, negatives 0.3; sum 107.7
