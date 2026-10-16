# ACC-GAP diagnosis: g10

table `$DATA_DIR/runs/route_follow/acc_gap/g10_tab.parquet`, 280 scenes, 55495 decisions, valid 42042

## Winning term at 1.5 s (valid decisions)

| term | share |
|:--|--:|
|  | 1.000 |

## Following decisions (true moving lead <= 40 m, ego >= 2 m/s): 8406 in 119 scenes

| winning term | decisions | scenes | ego time gap s (median) | human same place s | ratio median | ratio >= 1.5 | ego gap m | human gap m (same t) | ego lags human m |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all | 8406 | 119 | 3.22 | 2.30 | 1.16 | 0.32 | 22.4 | 18.5 | 1.0 |

Per scene (>= 5 paired following decisions): 109 scenes; median ego 3.34 s, human 2.47 s, ratio median 1.17

## The model's lead against the true lead (following decisions)

- serve lead present (filtered p > 0.5): 7563 / 8406 (0.90)
- distance error d - true gap: median -0.93 m, p10 -4.33, p90 +2.34
- relative speed error (vl - v_lead_true): median -0.90 m/s, p10 -2.25, p90 +0.05
  - true gap 0-15 m: n 1542, d - gap median -0.33 m, vl - v median -0.65
  - true gap 15-25 m: n 2897, d - gap median -1.31 m, vl - v median -0.80
  - true gap 25-40 m: n 3116, d - gap median -0.90 m, vl - v median -1.14

## Phantom and missed leads (valid decisions)

- serve lead present: 19733 decisions; no true corridor vehicle within d + 15 m: 3646 (0.18), lead term wins on 0 of them, in 0 scenes
- true moving lead <= 30 m and serve p <= 0.5: 585

## Decisions the lead term wins

- 0 decisions in 0 scenes; true lead moving <= 40 m 0, stationary vehicle <= 40 m 0, none within 40 m 0
- with no true vehicle within 40 m: serve p > 0.5 0, p <= 0.5 (the MPC's no-lead acceleration limit) 0
- served speed at 1.5 s below the set speed by: median nan m/s

## Time gap by speed (following decisions where lead wins)

| ego speed m/s | n | ego s | human same place s | openpilot steady state (1.45 + 6 / v) s |
|:--|--:|--:|--:|--:|
