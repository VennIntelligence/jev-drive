# Guard set: rc-all-s0 (subset mode)

Verdict: **FAIL** (5 pass, 3 fail, 33 not evaluable). 2026-10-05 04:35:26

| line | readout | candidate | shipped | delta | rule | result | note |
|---|---|---:|---:|---:|---|:--|---|
| navtest (subset) | PDMS v1 (subset, 4026 tokens) | 84.808 | 84.03 | 0.778 [0.334, 1.267] | not below shipped - 0.3 | pass | 4026 / 4026 tokens valid in both; paired mean delta, 95% CI log-cluster bootstrap over 136 logs; sub-score deltas (pp) NC +0.14 DAC +0.82 EP +0.75 TTC +0.25 C +0.00 |
| navhard (subset) | EPDMS two-stage (5912 tokens) | 35.233 | 33.331 | 1.902 [-0.496, 4.291] | no drop | pass | devkit harness (official aggregation; no official CSV in subset mode); harness - official combined: shipped -0.0000; CI paired over 225 mapping groups |
| navhard (subset) | EPDMS stage 1 | 71.748 | 71.704 | 0.043 [-1.925, 2.02] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS stage 2 | 48.748 | 46.897 | 1.851 [-0.878, 4.566] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS early-turn set (stages pooled) | 43.241 | 43.888 | -0.647 [-1.575, 0.28] | no drop | **FAIL** | 3852 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 220 mapping groups |
| navhard (subset) | EPDMS early-turn stage 1 | 66.819 | 66.724 | 0.095 [-3.231, 3.372] |  | n/a | 231 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 128 mapping groups; diagnostic (no rule) |
| navhard (subset) | EPDMS early-turn stage 2 | 41.737 | 42.432 | -0.694 [-1.664, 0.252] |  | n/a | 3621 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 219 mapping groups; diagnostic (no rule) |
| wod (subset) | error | - | - | - | RFS no drop; false start <= shipped + 2 pp | n/a | ModuleNotFoundError: No module named 'onnx' |
| hugsim (subset) | spins (heading err >= 60 deg) | 1 | 0 | 1 | not up vs shipped | **FAIL** | 11 scenes |
| hugsim (subset) | stuck (end max_steps) | 1 | 6 | -5 | reported (stuck: reduction is the goal) | n/a |  |
| hugsim (subset) | completes | 5 | 3 | 2 | reported | n/a |  |
| hugsim (subset) | HD-Score mean (paired over scenes) | 0.537 | 0.544 | -0.006 [-0.193, 0.185] | reported | n/a | paired mean diff -0.006 |
| b2d_turns (subset) | took exit, choice turns | 0.231 | 0 | 0.231 [0, 0.462] | primary readout (up is the goal) | n/a | 3 / 13 vs 0 / 13 |
| b2d_turns (subset) | took exit, forced turns | 0.25 | 0.083 | 0.167 [0, 0.455] | primary readout (up is the goal) | n/a | 3 / 12 vs 1 / 12 |
| b2d_turns (subset) | took exit, all turns | 0.24 | 0.04 | 0.2 [0.038, 0.4] | primary readout (up is the goal) | n/a | 6 / 25 vs 1 / 25 |
| b2d_turns (subset) | leaves lane, share of entered turns | 0.905 | 0.9 | 0.005 | reported | n/a | 21 entered |
| b2d_turns (subset) | collisions in turn windows | 8 | 5 | 3 | not up vs shipped | **FAIL** | 25 turns |
| b2d_turns (subset) | collisions, whole runs (20 routes) | 24 | 21 | 3 | reported | n/a |  |
| b2d_turns (subset) | route DS mean, zones off | 31.036 | 26.287 | 4.749 | reported | n/a |  |
| b2d_ds (subset) | DS mean (paired over seed x route) | 62.528 | 69.265 | -6.737 [-19.721, 4.949] | delta >= -7.6 (no drop beyond decision-38 noise) | pass | n 19, paired diff -6.74 |
| b2d_ds (subset) | RC mean | 85.949 | 78.571 | 7.378 | reported | n/a |  |
| b2d_ds (subset) | collisions (sum) | 8 | 5 | 3 | reported | n/a |  |
| b2d_ds (subset) | red lights (sum) | 7 | 3 | 4 | reported | n/a |  |
| b2d_ds (subset) | routes completed | 15 | 13 | 2 | reported | n/a |  |
| drift (subset) | 4 s lateral drift vs shipped, max over sets / kinds (median per set) | 0.041 | 0 | 0.041 | <= 0.10 m | pass | max at navdev/junction |
| drift (subset) | 4 s lateral drift median, navdev junction (nav) | 0.041 | 0 | 0.041 | info | n/a | n 136; p90 0.336; plan_drift (0-5 s L2) median 0.195 |
| drift (subset) | 4 s lateral drift median, navdev straight (nav) | 0.013 | 0 | 0.013 | info | n/a | n 30; p90 0.197; plan_drift (0-5 s L2) median 0.133 |
| drift (subset) | 4 s lateral drift median, carladev junction (carla) | 0.036 | 0 | 0.036 | info | n/a | n 75; p90 0.840; plan_drift (0-5 s L2) median 0.389 |
| drift (subset) | 4 s lateral drift median, dist:nav all (nav) | 0.017 | 0 | 0.017 | info | n/a | n 153; p90 0.196; plan_drift (0-5 s L2) median 0.144 |
| drift (subset) | 4 s lateral drift median, dist:wod all (wod) | 0.021 | 0 | 0.021 | info | n/a | n 380; p90 0.208; plan_drift (0-5 s L2) median 0.126 |
| drift (subset) | 4 s lateral drift median, dist:carla all (carla) | 0.019 | 0 | 0.019 | info | n/a | n 333; p90 0.201; plan_drift (0-5 s L2) median 0.158 |
| drift (subset) | plan_drift (0-5 s mean L2, x and y) median, max over sets / kinds | 0.389 | 0 | 0.389 | info: the metric op_img_cmd's 0.10 m line was set on | n/a | max at carladev.junction |
| negatives (subset) | |y(4 s) - logged| on the negative frames, mean (m) | 0.728 | 0.779 | -0.052 [-0.101, -0.008] | <= original + 0.3 m | pass | offset under the negative command (adapter) vs shipped without a command; n 158, 135 of 158 await the visual check (all used) |
| negatives (subset) | same, the map-certain N1 tier-A rows (no visual check needed) | 0.507 | 0.551 | -0.044 | info | n/a | n 23 |
| negatives (subset) | same, N1_exit | 0.731 | 0.771 | -0.041 | info | n/a | n 58 |
| negatives (subset) | same, N2_side | 0.461 | 0.457 | 0.005 | info | n/a | n 40 |
| negatives (subset) | same, N3_wrong | 1.148 | 1.332 | -0.183 | info | n/a | n 30 |
| negatives (subset) | same, N4_uturn | 0.656 | 0.673 | -0.017 | info | n/a | n 30 |
| negatives (subset) | offset with command 'none' | 0.756 | 0.779 | -0.024 | info | n/a |  |
| negatives (subset) | offset with command 'correct' | 0.738 | 0.779 | -0.042 | info | n/a |  |
| negatives (subset) | |y_negative - y_none| at 4 s, mean | 0.143 | 0 | 0.143 | info | n/a |  |

Runtime per line (min): navtest 10.5, navhard 11.5, wod 2.1, hugsim 37.7, b2d_turns 33.1, b2d_ds 0.0, drift 0.3, negatives 0.3; sum 95.5
