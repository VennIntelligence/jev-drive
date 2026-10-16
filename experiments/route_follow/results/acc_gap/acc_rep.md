# ACC-GAP diagnosis: acc

table `$DATA_DIR/runs/route_follow/acc_gap/acc_tab.parquet`, 280 scenes, 55495 decisions, valid 45949

## Winning term at 1.5 s (valid decisions)

| term | share |
|:--|--:|
| set | 0.487 |
| lead | 0.323 |
| model | 0.069 |
| ramp | 0.066 |
| curve | 0.055 |

## Following decisions (true moving lead <= 40 m, ego >= 2 m/s): 9706 in 124 scenes

| winning term | decisions | scenes | ego time gap s (median) | human same place s | ratio median | ratio >= 1.5 | ego gap m | human gap m (same t) | ego lags human m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 9706 | 124 | 3.02 | 2.28 | 1.10 | 0.24 | 22.7 | 19.9 | 0.8 |
| lead | 6216 | 109 | 3.10 | 2.25 | 1.16 | 0.26 | 21.8 | 18.5 | 1.2 |
| set | 2045 | 78 | 2.94 | 2.29 | 1.05 | 0.20 | 24.5 | 23.8 | 0.3 |
| ramp | 466 | 22 | 2.90 | 1.96 | 1.01 | 0.24 | 23.3 | 18.6 | 1.0 |
| model | 624 | 47 | 3.01 | 2.88 | 1.02 | 0.17 | 25.0 | 26.3 | 0.0 |
| curve | 355 | 21 | 2.56 | 2.40 | 1.04 | 0.21 | 23.5 | 21.2 | 0.8 |

Per scene (>= 5 paired following decisions): 115 scenes; median ego 3.17 s, human 2.45 s, ratio median 1.11

## The model's lead against the true lead (following decisions)

- serve lead present (filtered p > 0.5): 8827 / 9706 (0.91)
- distance error d - true gap: median -1.06 m, p10 -4.29, p90 +1.95
- relative speed error (vl - v_lead_true): median -0.83 m/s, p10 -2.09, p90 +0.09
  - true gap 0-15 m: n 1753, d - gap median -0.31 m, vl - v median -0.66
  - true gap 15-25 m: n 3315, d - gap median -1.38 m, vl - v median -0.79
  - true gap 25-40 m: n 3750, d - gap median -1.22 m, vl - v median -0.93

## Phantom and missed leads (valid decisions)

- serve lead present: 22378 decisions; no true corridor vehicle within d + 15 m: 4060 (0.18), lead term wins on 1249 of them, in 65 scenes
- true moving lead <= 30 m and serve p <= 0.5: 662

## Decisions the lead term wins

- 14864 decisions in 214 scenes; true lead moving <= 40 m 7553, stationary vehicle <= 40 m 4500, none within 40 m 2811
- with no true vehicle within 40 m: serve p > 0.5 2542, p <= 0.5 (the MPC's no-lead acceleration limit) 269
- served speed at 1.5 s below the set speed by: median 1.59 m/s

## Time gap by speed (following decisions where lead wins)

| ego speed m/s | n | ego s | human same place s | openpilot steady state (1.45 + 6 / v) s |
|:--|--:|--:|--:|--:|
| 2-5 | 1937 | 4.92 | 3.52 | 3.29 |
| 5-8 | 1339 | 3.22 | 2.25 | 2.36 |
| 8-12 | 1640 | 3.00 | 2.26 | 2.09 |
| 12-18 | 723 | 2.29 | 2.04 | 1.88 |
| 18-40 | 577 | 1.08 | 1.09 | 1.69 |
