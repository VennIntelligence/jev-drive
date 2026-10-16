# ACC-GAP diagnosis: s1acc

table `$DATA_DIR/runs/route_follow/acc_gap/s1acc_tab.parquet`, 37 scenes, 7341 decisions, valid 6504

## Winning term at 1.5 s (valid decisions)

| term | share |
|:--|--:|
| lead | 0.773 |
| set | 0.175 |
| model | 0.026 |
| curve | 0.017 |
| ramp | 0.009 |

## Following decisions (true moving lead <= 40 m, ego >= 2 m/s): 4570 in 34 scenes

| winning term | decisions | scenes | ego time gap s (median) | human same place s | ratio median | ratio >= 1.5 | ego gap m | human gap m (same t) | ego lags human m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 4570 | 34 | 3.01 | 2.23 | 1.17 | 0.28 | 22.6 | 18.1 | 1.6 |
| lead | 3834 | 34 | 3.02 | 2.22 | 1.19 | 0.29 | 22.3 | 18.1 | 1.9 |
| set | 488 | 20 | 2.99 | 2.57 | 1.06 | 0.16 | 24.8 | 19.7 | 0.4 |
| ramp | 43 | 3 | 3.46 | 1.81 | 1.90 | 0.65 | 21.5 | 15.3 | 6.1 |
| model | 145 | 13 | 2.35 | 2.23 | 1.06 | 0.17 | 26.2 | 24.7 | 0.0 |
| curve | 60 | 3 | 3.26 | 1.88 | 1.23 | 0.46 | 21.4 | 15.9 | 14.6 |

Per scene (>= 5 paired following decisions): 34 scenes; median ego 3.11 s, human 2.22 s, ratio median 1.23

## The model's lead against the true lead (following decisions)

- serve lead present (filtered p > 0.5): 4443 / 4570 (0.97)
- distance error d - true gap: median -1.08 m, p10 -4.09, p90 +1.30
- relative speed error (vl - v_lead_true): median -0.83 m/s, p10 -1.87, p90 -0.08
  - true gap 0-15 m: n 831, d - gap median -0.72 m, vl - v median -0.69
  - true gap 15-25 m: n 1715, d - gap median -1.38 m, vl - v median -0.80
  - true gap 25-40 m: n 1897, d - gap median -1.08 m, vl - v median -0.92

## Phantom and missed leads (valid decisions)

- serve lead present: 5993 decisions; no true corridor vehicle within d + 15 m: 158 (0.03), lead term wins on 85 of them, in 7 scenes
- true moving lead <= 30 m and serve p <= 0.5: 94

## Decisions the lead term wins

- 5027 decisions in 36 scenes; true lead moving <= 40 m 4176, stationary vehicle <= 40 m 528, none within 40 m 323
- with no true vehicle within 40 m: serve p > 0.5 314, p <= 0.5 (the MPC's no-lead acceleration limit) 9
- served speed at 1.5 s below the set speed by: median 1.18 m/s

## Time gap by speed (following decisions where lead wins)

| ego speed m/s | n | ego s | human same place s | openpilot steady state (1.45 + 6 / v) s |
|:--|--:|--:|--:|--:|
| 2-5 | 941 | 5.27 | 3.71 | 3.42 |
| 5-8 | 918 | 3.08 | 2.16 | 2.34 |
| 8-12 | 1130 | 3.01 | 2.25 | 2.09 |
| 12-18 | 455 | 2.23 | 2.09 | 1.88 |
| 18-40 | 390 | 1.01 | 1.02 | 1.69 |
