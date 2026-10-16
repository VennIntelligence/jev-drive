# ACC-GAP diagnosis: s1

table `$DATA_DIR/runs/route_follow/acc_gap/s1_tab.parquet`, 37 scenes, 7341 decisions, valid 6870

## Winning term at 1.5 s (valid decisions)

| term | share |
|:--|--:|
| lead | 0.628 |
| set | 0.292 |
| model | 0.047 |
| curve | 0.017 |
| ramp | 0.016 |

## Following decisions (true moving lead <= 40 m, ego >= 2 m/s): 4980 in 34 scenes

| winning term | decisions | scenes | ego time gap s (median) | human same place s | ratio median | ratio >= 1.5 | ego gap m | human gap m (same t) | ego lags human m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 4980 | 34 | 2.46 | 2.23 | 1.03 | 0.13 | 20.0 | 18.0 | 0.1 |
| lead | 3243 | 34 | 2.47 | 2.24 | 1.04 | 0.15 | 18.8 | 17.3 | 0.3 |
| set | 1333 | 22 | 2.40 | 2.26 | 1.00 | 0.07 | 25.5 | 24.3 | 0.0 |
| ramp | 96 | 6 | 3.07 | 1.74 | 1.74 | 0.66 | 20.2 | 15.3 | 5.5 |
| model | 246 | 14 | 2.41 | 2.23 | 1.06 | 0.10 | 18.0 | 15.7 | 0.0 |
| curve | 62 | 3 | 2.83 | 2.01 | 1.33 | 0.17 | 19.0 | 17.6 | 2.5 |

Per scene (>= 5 paired following decisions): 34 scenes; median ego 2.53 s, human 2.24 s, ratio median 1.01

## The model's lead against the true lead (following decisions)

- serve lead present (filtered p > 0.5): 4819 / 4980 (0.97)
- distance error d - true gap: median -0.96 m, p10 -3.29, p90 +1.06
- relative speed error (vl - v_lead_true): median -0.82 m/s, p10 -1.77, p90 -0.20
  - true gap 0-15 m: n 1527, d - gap median -0.68 m, vl - v median -0.73
  - true gap 15-25 m: n 1798, d - gap median -1.25 m, vl - v median -0.78
  - true gap 25-40 m: n 1494, d - gap median -1.03 m, vl - v median -1.00

## Phantom and missed leads (valid decisions)

- serve lead present: 6337 decisions; no true corridor vehicle within d + 15 m: 168 (0.03), lead term wins on 66 of them, in 7 scenes
- true moving lead <= 30 m and serve p <= 0.5: 130

## Decisions the lead term wins

- 4316 decisions in 35 scenes; true lead moving <= 40 m 3583, stationary vehicle <= 40 m 588, none within 40 m 145
- with no true vehicle within 40 m: serve p > 0.5 140, p <= 0.5 (the MPC's no-lead acceleration limit) 5
- served speed at 1.5 s below the set speed by: median 1.10 m/s

## Time gap by speed (following decisions where lead wins)

| ego speed m/s | n | ego s | human same place s | openpilot steady state (1.45 + 6 / v) s |
|:--|--:|--:|--:|--:|
| 2-5 | 887 | 4.12 | 3.67 | 3.40 |
| 5-8 | 631 | 2.31 | 2.10 | 2.32 |
| 8-12 | 1126 | 2.38 | 2.10 | 2.08 |
| 12-18 | 422 | 2.15 | 2.07 | 1.88 |
| 18-40 | 177 | 0.72 | 0.74 | 1.64 |
