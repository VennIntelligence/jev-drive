# Guard set: rc-bear-pre-s0 (subset mode)

Verdict: **FAIL** (7 pass, 3 fail, 33 not evaluable). 2026-10-05 10:28:04

| line | readout | candidate | shipped | delta | rule | result | note |
|---|---|---:|---:|---:|---|:--|---|
| navtest (subset) | PDMS v1 (subset, 4026 tokens) | 84.338 | 84.03 | 0.308 [-0.064, 0.726] | not below shipped - 0.3 | pass | 4026 / 4026 tokens valid in both; paired mean delta, 95% CI log-cluster bootstrap over 136 logs; sub-score deltas (pp) NC -0.01 DAC +0.52 EP +0.28 TTC -0.10 C +0.00 |
| navhard (subset) | EPDMS two-stage (5912 tokens) | 32.865 | 33.331 | -0.465 [-1.673, 0.741] | no drop | **FAIL** | devkit harness (official aggregation; no official CSV in subset mode); harness - official combined: shipped -0.0000; CI paired over 225 mapping groups |
| navhard (subset) | EPDMS stage 1 | 71.701 | 71.704 | -0.004 [-1.657, 1.742] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS stage 2 | 46.381 | 46.897 | -0.517 [-1.609, 0.502] |  | n/a | diagnostic (no rule); CI over mapping groups |
| navhard (subset) | EPDMS early-turn set (stages pooled) | 43.662 | 43.888 | -0.226 [-0.579, 0.127] | no drop | **FAIL** | 3852 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 220 mapping groups |
| navhard (subset) | EPDMS early-turn stage 1 | 67.374 | 66.724 | 0.651 [-2.237, 3.572] |  | n/a | 231 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 128 mapping groups; diagnostic (no rule) |
| navhard (subset) | EPDMS early-turn stage 2 | 42.149 | 42.432 | -0.282 [-0.604, 0.037] |  | n/a | 3621 tokens, PDM ref heading change >= 5 deg in 2 s (decision 110); per-token score, CI clustered by 219 mapping groups; diagnostic (no rule) |
| wod (subset) | RFS, 479 rater frames (serving ONNX) | 8.064 | 8.005 | 0.06 [-0.008, 0.128] | RFS no drop (paired 95% upper bound >= 0) | pass | segment bootstrap B=2000; shipped vs itself = 0 |
| wod (subset) | false-start rate, WOD val stay rows (port, plan >= 3 m at 4 s) | 0.043 | 0.041 | 0.001 [-0.002, 0.004] | false start <= shipped + 2 pp | pass | n 4116 rows / 144 segments; op_adapt_L/H metric and rows |
| wod (subset) | false-start rate, the stay frames among the rater frames (serving) | 0.091 | 0.045 | 0.045 | info (n too small for a 2 pp line) | n/a | n 22 frames (1 frame = 4.5 pp) |
| hugsim (subset) | spins (heading err >= 60 deg) | 0 | 0 | 0 | not up vs shipped | pass | 11 scenes |
| hugsim (subset) | stuck (end max_steps) | 4 | 6 | -2 | reported (stuck: reduction is the goal) | n/a |  |
| hugsim (subset) | completes | 3 | 3 | 0 | reported | n/a |  |
| hugsim (subset) | HD-Score mean (paired over scenes) | 0.43 | 0.544 | -0.113 [-0.26, 0.003] | reported | n/a | paired mean diff -0.113 |
| b2d_turns (subset) | took exit, choice turns | 0 | 0 | 0 [0, 0] | primary readout (up is the goal) | n/a | 0 / 13 vs 0 / 13 |
| b2d_turns (subset) | took exit, forced turns | 0 | 0.083 | -0.083 [-0.273, 0] | primary readout (up is the goal) | n/a | 0 / 12 vs 1 / 12 |
| b2d_turns (subset) | took exit, all turns | 0 | 0.04 | -0.04 [-0.12, 0] | primary readout (up is the goal) | n/a | 0 / 25 vs 1 / 25 |
| b2d_turns (subset) | leaves lane, share of entered turns | 0.947 | 0.9 | 0.047 | reported | n/a | 19 entered |
| b2d_turns (subset) | collisions in turn windows | 6 | 5 | 1 | not up vs shipped | **FAIL** | 25 turns |
| b2d_turns (subset) | collisions, whole runs (20 routes) | 26 | 21 | 5 | reported | n/a |  |
| b2d_turns (subset) | route DS mean, zones off | 23.091 | 26.287 | -3.196 | reported | n/a |  |
| b2d_ds (subset) | DS mean (paired over seed x route) | 67.981 | 69.265 | -1.284 [-13.512, 10.509] | delta >= -7.6 (no drop beyond decision-38 noise) | pass | n 19, paired diff -1.28 |
| b2d_ds (subset) | RC mean | 83.586 | 78.571 | 5.015 | reported | n/a |  |
| b2d_ds (subset) | collisions (sum) | 6 | 5 | 1 | reported | n/a |  |
| b2d_ds (subset) | red lights (sum) | 5 | 3 | 2 | reported | n/a |  |
| b2d_ds (subset) | routes completed | 14 | 13 | 1 | reported | n/a |  |
| drift (subset) | 4 s lateral drift vs shipped, max over sets / kinds (median per set) | 0.038 | 0 | 0.038 | <= 0.10 m | pass | max at carladev/junction |
| drift (subset) | 4 s lateral drift median, navdev junction (nav) | 0.026 | 0 | 0.026 | info | n/a | n 136; p90 0.102; plan_drift (0-5 s L2) median 0.115 |
| drift (subset) | 4 s lateral drift median, navdev straight (nav) | 0.012 | 0 | 0.012 | info | n/a | n 30; p90 0.086; plan_drift (0-5 s L2) median 0.103 |
| drift (subset) | 4 s lateral drift median, carladev junction (carla) | 0.038 | 0 | 0.038 | info | n/a | n 75; p90 0.381; plan_drift (0-5 s L2) median 0.173 |
| drift (subset) | 4 s lateral drift median, dist:nav all (nav) | 0.01 | 0 | 0.01 | info | n/a | n 153; p90 0.085; plan_drift (0-5 s L2) median 0.094 |
| drift (subset) | 4 s lateral drift median, dist:wod all (wod) | 0.012 | 0 | 0.012 | info | n/a | n 380; p90 0.090; plan_drift (0-5 s L2) median 0.086 |
| drift (subset) | 4 s lateral drift median, dist:carla all (carla) | 0.01 | 0 | 0.01 | info | n/a | n 333; p90 0.050; plan_drift (0-5 s L2) median 0.086 |
| drift (subset) | plan_drift (0-5 s mean L2, x and y) median, max over sets / kinds | 0.173 | 0 | 0.173 | info: the metric op_img_cmd's 0.10 m line was set on | n/a | max at carladev.junction |
| negatives (subset) | |y(4 s) - logged| on the negative frames, mean (m) | 0.745 | 0.779 | -0.035 [-0.069, -0.001] | <= original + 0.3 m | pass | offset under the negative command (adapter) vs shipped without a command; n 158, 135 of 158 await the visual check (all used) |
| negatives (subset) | same, the map-certain N1 tier-A rows (no visual check needed) | 0.506 | 0.551 | -0.045 | info | n/a | n 23 |
| negatives (subset) | same, N1_exit | 0.725 | 0.771 | -0.047 | info | n/a | n 58 |
| negatives (subset) | same, N2_side | 0.449 | 0.457 | -0.008 | info | n/a | n 40 |
| negatives (subset) | same, N3_wrong | 1.276 | 1.332 | -0.056 | info | n/a | n 30 |
| negatives (subset) | same, N4_uturn | 0.646 | 0.673 | -0.028 | info | n/a | n 30 |
| negatives (subset) | offset with command 'none' | 0.773 | 0.779 | -0.007 | info | n/a |  |
| negatives (subset) | offset with command 'correct' | 0.754 | 0.779 | -0.025 | info | n/a |  |
| negatives (subset) | |y_negative - y_none| at 4 s, mean | 0.138 | 0 | 0.138 | info | n/a |  |

Runtime per line (min): navtest 8.7, navhard 11.5, wod 6.4, hugsim 41.4, b2d_turns 2.9, b2d_ds 0.0, drift 0.2, negatives 0.3; sum 71.4
