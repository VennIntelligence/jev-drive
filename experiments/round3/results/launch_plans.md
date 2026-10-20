# ROUND3: model plans at the human's launch (stored reads, launch scenes)

t_L = the log is 0.5 m past its handoff pose; `arc4@tL` = path length of the model plan (poses_model) at the first decision at or after t_L; `go` = time from t_L to the first decision whose plan covers >= 2 m (negative: before the human); `short3` = share of decisions in [t_L, t_L + 3 s] with the ego still (< 0.5 m/s) and a plan under 1 m; `ego` = time from t_L until the ego is 0.5 m past its handoff pose. Medians; n = launch scenes of the read.

## PAI 281, all launch scenes (ACC + GAP)

| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-281ag | 37 | 33 | 11.19 | 1 | -3.90 | 0 | 0.00 | -1.16 | 24 |
| R2J-281ag | 37 | 21 | 8.07 | 1 | -3.09 | 0 | 0.01 | -0.10 | 24 |

## PAI 281, the 14 launches NTV0 makes and R2J loses

| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-281ag | 14 | 14 | 11.32 | 1 | -4.30 | 0 | 0.00 | -1.39 | 8 |
| R2J-281ag | 14 | 0 | 4.29 | 1 | -3.28 | 0 | 0.00 | +0.68 | 8 |

## PAI-FIX G-LAUNCH (VC + LEAD)

| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-vl | 6 | 5 | 7.53 | 1 | -4.71 | 0 | 0.02 | +0.07 | 6 |
| R2C-vl | 6 | 5 | 7.40 | 0 | -4.71 | 0 | 0.00 | +0.07 | 6 |
| R2J-vl | 6 | 3 | 5.99 | 1 | -1.87 | 0 | 0.17 | +0.38 | 6 |
| R2JH-vl | 6 | 4 | 5.95 | 1 | -1.82 | 0 | 0.17 | +0.38 | 6 |

## PAI-FIX all launch scenes (G-LAUNCH + TARGET, VC + LEAD)

| read | n | launched | arc4@tL med | arc4@tL < 1 m | go med (s) | no go | short3 mean | ego move med (s) | lead at t_L |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-vl | 18 | 13 | 14.63 | 1 | -4.97 | 0 | 0.01 | -1.18 | 7 |
| R2C-vl | 18 | 13 | 14.19 | 0 | -4.97 | 0 | 0.00 | -1.18 | 7 |
| R2J-vl | 18 | 8 | 6.55 | 1 | -3.39 | 0 | 0.06 | +0.01 | 7 |
| R2JH-vl | 18 | 8 | 4.94 | 1 | -3.39 | 0 | 0.06 | +0.08 | 7 |

## The lost launches, per scene (R2J vs NTV0, ACC + GAP)

| scene | lead at t_L (v) | t_L (s) | human 4 s (m) | NTV0 arc4@tL / go / ego | R2J arc4@tL / go / ego | R2J score | R2J class |
|:--|:--|--:|--:|:--|:--|--:|:--|
| 22fa5c92 | no (nan) | 6.9 | 10.9 | 11.9 / -6.9 / -5.1 | 5.3 / -6.9 / -4.2 | 0.05 | slow |
| 52bd89ce | yes (0.1) | 0.7 | 2.9 | 1.0 / -0.7 / 0.0 | 0.8 / 0.6 / 0.0 | 0.08 | rear-hit |
| 68e53160 | no (nan) | 9.0 | 15.6 | 37.9 / -9.0 / -7.0 | 4.3 / -4.1 / 4.7 | 0.01 | rear-hit |
| 76f7e025 | no (nan) | 7.5 | 27.1 | 49.5 / -7.5 / -5.8 | 3.5 / -2.3 / 6.5 | 0.00 | rear-hit |
| 7c6529f6 | yes (0.2) | 11.1 | 6.6 | 4.6 / -11.1 / -9.5 | 2.0 / -11.0 / -9.5 | 0.24 | rear-hit |
| 9fa1d6eb | yes (3.2) | 3.2 | 8.2 | 10.7 / -3.2 / -0.8 | 10.2 / -3.1 / -0.7 | 0.17 | rear-hit |
| a1b166cc | no (nan) | 7.6 | 14.7 | 24.0 / -7.6 / -5.6 | 2.5 / -3.2 / 5.3 | 0.01 | rear-hit |
| c8cab401 | yes (1.3) | 2.5 | 4.7 | 6.8 / -2.5 / -0.2 | 5.9 / -1.6 / 0.0 | 0.16 | rear-hit |
| ca173de1 | no (nan) | 3.2 | 19.0 | 28.9 / -3.2 / -1.6 | 4.2 / -2.3 / 14.2 | 0.00 | rear-hit |
| de9db5dc | yes (2.9) | 4.4 | 14.0 | 10.2 / -4.4 / -3.1 | 10.0 / -4.3 / -2.8 | 0.37 | rear-hit |
| e37aa159 | yes (7.1) | 5.2 | 13.8 | 20.8 / -5.1 / -1.0 | 3.7 / -4.9 / 5.6 | 0.06 | rear-hit |
| ea468790 | no (nan) | 4.3 | 6.0 | 9.7 / -4.2 / -1.2 | 9.8 / -4.1 / -1.3 | 0.00 | offroad |
| f1069db5 | yes (3.6) | 3.6 | 21.2 | 15.2 / -3.6 / -1.2 | 7.5 / -2.9 / 1.3 | 0.02 | slow |
| fda39805 | yes (0.0) | 3.4 | 9.0 | 5.4 / -3.3 / 0.7 | 4.1 / -3.3 / 2.9 | 0.07 | slow |
