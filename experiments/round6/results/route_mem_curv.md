# ROUTE-CURV: is the route input a lateral intent in disguise? (ROUND6, 2026-10-21)

Diagnosis, stored reads only, no GPU (pool CPU jobs). Code `experiments/round6/scripts/rmc.py` (`extract`, `scenes`, `rows`, `report`), lane `rmc.sh`. Generated tables; the reading follows at the end.

## 0. Sanity

Decisions read: 277274 (NTV0 55494, R3MV-s0 55494, R3MV-s1 55099, MVLC-s0 55494, MVLC-s1 55693); human states: test 7729, train 18745. Fed +40 m route point vs AlpaSim's generator polyline rebuilt on the log: median 0.002 m, p99 31.066 m, > 0.5 m on 2.70 % (n 263756). Current lane found on 98.9 % of decisions, 100.0 / 99.8 % of human states (test / train).

## 1. The route against the lane on the human log

Every 0.5 s of the human log (v >= 2 m/s); route = AlpaSim's map route (what the messages are cut from) windowed as the fed memory. `y_rt40` = the route point 40 m of route arc ahead, lateral in the ego heading frame (+ left); parts: `pose` = the ego's lane offset and heading error carried 40 m straight, `curv` = the lane centre's own bend over those 40 m, `lane off` = the route point's signed Frenet offset from the ego's current lane centre (the human's lane here). `off > 2.5 m` = the route sits in another lane. cmd = the command rule on the route's ~42 m point.

**test 281**

| cls | n | |y_rt40| p50 | p90 | |pose| p50 | |curv| p50 | |curv| p90 | |lane off| p50 | |lane off| p90 | off>1 m % | off>2.5 m % | off<-1 (right) % | off>1 (left) % | cmd L % | cmd R % |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| straight (<3 deg / 40 m) | 5156 | 0.51 | 2.98 | 0.40 | 0.13 | 0.84 | 0.00 | 0.62 | 9.97 | 9.50 | 3.39 | 6.57 | 8.01 | 9.06 |
| gentle curve (3-15) | 1264 | 1.80 | 6.02 | 0.74 | 2.06 | 5.00 | 0.00 | 0.06 | 9.34 | 7.12 | 6.80 | 2.53 | 19.30 | 29.91 |
| bend (15-45) | 517 | 8.00 | 18.29 | 1.22 | 8.66 | 17.94 | 0.00 | 0.05 | 5.61 | 2.51 | 5.22 | 0.39 | 35.01 | 56.09 |
| turn (>=45) | 781 | 17.91 | 29.79 | 1.35 | 21.15 | 32.70 | 0.00 | 19.29 | 15.49 | 15.24 | 11.40 | 4.10 | 41.23 | 52.24 |

**train 719 (the PAI human rows' scenes)**

| cls | n | |y_rt40| p50 | p90 | |pose| p50 | |curv| p50 | |curv| p90 | |lane off| p50 | |lane off| p90 | off>1 m % | off>2.5 m % | off<-1 (right) % | off>1 (left) % | cmd L % | cmd R % |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| straight (<3 deg / 40 m) | 11005 | 0.54 | 3.32 | 0.38 | 0.14 | 0.83 | 0.00 | 3.28 | 15.30 | 14.41 | 7.76 | 7.54 | 8.83 | 11.59 |
| gentle curve (3-15) | 3204 | 1.98 | 6.02 | 0.66 | 2.19 | 5.48 | 0.00 | 2.80 | 12.17 | 10.67 | 8.77 | 3.40 | 21.47 | 31.77 |
| bend (15-45) | 2048 | 8.27 | 17.72 | 0.96 | 8.86 | 17.84 | 0.00 | 0.06 | 7.18 | 6.45 | 5.27 | 1.90 | 39.60 | 54.30 |
| turn (>=45) | 2387 | 19.40 | 29.58 | 1.37 | 20.96 | 32.47 | 0.01 | 6.01 | 15.63 | 15.21 | 10.10 | 5.53 | 45.20 | 50.52 |

- test: route >= 1.75 m off the human's lane centre at 40 m on 10.0 % of states (771; right 4.8 %, left 5.2 %); of these the human's 4 s point stays within 1 m of its lane centre on 71.7 % and ends within 1 m of the route on 16.3 %; scenes with any such state 92 / 278.
- train: route >= 1.75 m off the human's lane centre at 40 m on 13.7 % of states (2545; right 7.8 %, left 5.8 %); of these the human's 4 s point stays within 1 m of its lane centre on 65.1 % and ends within 1 m of the route on 19.0 %; scenes with any such state 287 / 707.

Turn commands (L / R) on non-turn states (|lane bend over 40 m| < 15 deg): the part that puts the ~42 m point beyond 2 m (largest signed contribution):

| split | side | cause | gentle curve (3-15) | straight (<3 deg / 40 m) |
|--:|--:|--:|--:|--:|
| test | L | curvature (lane bends) | 211 | 23 |
| test | L | lane offset (route not on the ego's lane centre) | 10 | 317 |
| test | L | pose (ego offset / heading) | 23 | 73 |
| test | R | curvature (lane bends) | 254 | 30 |
| test | R | lane offset (route not on the ego's lane centre) | 34 | 148 |
| test | R | pose (ego offset / heading) | 90 | 289 |
| train | L | curvature (lane bends) | 610 | 52 |
| train | L | lane offset (route not on the ego's lane centre) | 29 | 686 |
| train | L | pose (ego offset / heading) | 49 | 234 |
| train | R | curvature (lane bends) | 746 | 61 |
| train | R | lane offset (route not on the ego's lane centre) | 126 | 671 |
| train | R | pose (ego offset / heading) | 146 | 544 |

## 2. The fed route at the decisions of the five reads

Every inference decision (10 Hz) of the stored PAI 281 ACC + GAP reads; route = the fed memory itself; lane = the ego's own current lane.

| g | n | |y_rt40| p50 | p90 | |pose| p50 | |curv| p50 | |curv| p90 | |lane off| p50 | |lane off| p90 | off>1 m % | off>2.5 m % | off<-1 (right) % | off>1 (left) % | cmd L % | cmd R % |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 / straight (<3 deg / 40 m) | 34585 | 0.58 | 2.92 | 0.52 | 0.12 | 0.81 | 0.00 | 2.95 | 10.85 | 10.40 | 5.94 | 4.91 | 6.41 | 11.17 |
| NTV0 / gentle curve (3-15) | 8087 | 1.52 | 6.19 | 0.95 | 1.93 | 4.89 | 0.01 | 3.05 | 12.81 | 10.71 | 8.57 | 4.24 | 15.94 | 24.74 |
| R3MV-s0 / straight (<3 deg / 40 m) | 34258 | 0.72 | 3.45 | 0.60 | 0.12 | 0.82 | 0.00 | 3.31 | 13.79 | 13.23 | 6.34 | 7.46 | 10.08 | 13.05 |
| R3MV-s0 / gentle curve (3-15) | 8445 | 1.69 | 6.42 | 1.07 | 1.97 | 5.01 | 0.01 | 3.38 | 16.02 | 14.59 | 9.67 | 6.35 | 20.70 | 25.99 |
| R3MV-s1 / straight (<3 deg / 40 m) | 34075 | 0.72 | 3.45 | 0.59 | 0.12 | 0.83 | 0.00 | 3.30 | 13.52 | 12.92 | 7.21 | 6.31 | 9.44 | 13.96 |
| R3MV-s1 / gentle curve (3-15) | 8275 | 1.70 | 6.23 | 1.03 | 1.93 | 5.05 | 0.01 | 3.41 | 16.12 | 14.14 | 9.72 | 6.40 | 19.87 | 26.83 |
| MVLC-s0 / straight (<3 deg / 40 m) | 34306 | 0.69 | 3.50 | 0.57 | 0.12 | 0.84 | 0.00 | 3.29 | 13.58 | 12.97 | 6.58 | 7.00 | 9.31 | 13.16 |
| MVLC-s0 / gentle curve (3-15) | 8289 | 1.73 | 6.50 | 1.00 | 1.98 | 5.04 | 0.01 | 3.33 | 14.88 | 13.20 | 8.52 | 6.36 | 20.92 | 26.52 |
| MVLC-s1 / straight (<3 deg / 40 m) | 34325 | 0.73 | 3.52 | 0.59 | 0.12 | 0.83 | 0.00 | 3.29 | 13.63 | 12.96 | 6.48 | 7.15 | 10.04 | 13.35 |
| MVLC-s1 / gentle curve (3-15) | 8533 | 1.74 | 6.65 | 1.04 | 2.03 | 5.02 | 0.01 | 3.30 | 14.53 | 13.00 | 9.32 | 5.22 | 19.31 | 28.38 |

## 3. The route before the MV family's offroad / zero-other events

Events: every offroad or zero-other zero of R3MV s0 / s1 and MV-LC s0 / s1 on a scene where NTV0 does not have that class (92 events: offroad 51, zero-other 41); decision = the first one at or after t_ev - 3 / 2 / 1 s (offroad_mv's rule); NTV0 = its decision nearest the same human-log station (< 3 m). `right route` = y_rt40 < -1.5 m, `right lane off` = d_rt40 < -1 m, `plan right of lane` = the plan's 4 s point > 1 m right of the ego lane's centre.

| set | n | y_rt40_med | d_rt40_med | y_cv40_med | right_route_pct | right_laneoff_pct | cmd_R_pct | cmd_L_pct | d_f4_med | plan_right_of_lane_pct |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| offroad events, -3 s, MV | 51 | 0.04 | 0.00 | -0.03 | 21.57 | 1.96 | 21.57 | 27.45 | -0.02 | 19.61 |
| zero-other events, -3 s, MV | 41 | -1.27 | 0.00 | 0.02 | 46.34 | 29.27 | 43.90 | 4.88 | 0.31 | 7.32 |
| offroad events, -3 s, NTV0 same station | 51 | -0.37 | 0.00 | 0.01 | 25.49 | 3.92 | 25.49 | 19.61 | -0.05 | 35.29 |
| zero-other events, -3 s, NTV0 same station | 41 | -0.41 | 0.00 | 0.02 | 19.51 | 12.20 | 14.63 | 12.20 | 0.15 | 0.00 |
| offroad events, -2 s, MV | 51 | 0.81 | 0.00 | -0.04 | 23.53 | 0.00 | 21.57 | 31.37 | -0.38 | 31.37 |
| zero-other events, -2 s, MV | 41 | -3.28 | -0.00 | 0.08 | 63.41 | 43.90 | 56.10 | 21.95 | 0.26 | 7.32 |
| offroad events, -2 s, NTV0 same station | 51 | -0.47 | 0.00 | -0.03 | 27.45 | 7.84 | 27.45 | 21.57 | -0.11 | 39.22 |
| zero-other events, -2 s, NTV0 same station | 41 | -0.28 | 0.00 | 0.03 | 14.63 | 7.32 | 14.63 | 17.07 | 0.24 | 0.00 |
| offroad events, -1 s, MV | 50 | 1.53 | 0.00 | 0.05 | 32.00 | 0.00 | 30.00 | 44.00 | -0.54 | 38.00 |
| zero-other events, -1 s, MV | 41 | -3.96 | -3.47 | -0.04 | 63.41 | 58.54 | 58.54 | 21.95 | 0.33 | 14.63 |
| offroad events, -1 s, NTV0 same station | 49 | -0.55 | 0.00 | 0.05 | 38.78 | 8.16 | 34.69 | 24.49 | -0.00 | 34.69 |
| zero-other events, -1 s, NTV0 same station | 41 | -0.34 | 0.00 | -0.01 | 14.63 | 7.32 | 14.63 | 17.07 | 0.28 | 2.44 |
| NTV0 clean scenes (score 1), all moving decisions | 28571 | -0.36 | 0.00 | -0.00 | 21.49 | 5.45 | 17.54 | 10.55 | 0.33 | 6.65 |
| NTV0 clean scenes, non-turn moving decisions | 25357 | -0.34 | 0.00 | -0.00 | 17.84 | 4.39 | 13.68 | 7.02 | 0.33 | 4.60 |
| MV clean scenes (score 1), all moving decisions | 116163 | -0.31 | 0.00 | -0.00 | 22.14 | 4.49 | 18.67 | 14.20 | 0.28 | 7.38 |
| MV clean scenes, non-turn moving decisions | 100973 | -0.28 | 0.00 | -0.00 | 17.37 | 3.20 | 13.70 | 10.91 | 0.28 | 5.49 |

Per event scene (median over its reads and the three leads; `d_hl4` = the human log's offset from the ego lane's centre at the plan's 4 s station):

| scene | fclass | mech | reads | y_rt40 | d_rt40 | y_cv40 | cmd_R | d_f4 | d_hl4 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 596af0c7 | offroad | turn / curve: cut inside | 4 | 1.03 | 3.50 | -3.43 | 0.00 | 0.98 | 3.50 |
| 6e7b42a3 | offroad | over-speed drift | 4 | 0.46 | -0.00 | 0.04 | 0.00 | -1.10 | -0.57 |
| 94c1d722 | offroad | turn / curve: cut inside | 4 | -24.80 | 0.00 | -23.73 | 1.00 | 0.75 | 0.39 |
| a2b415d5 | offroad | straight: lateral offset | 4 | 0.81 | 0.00 | -0.01 | 0.00 | -0.53 | 0.20 |
| dc401449 | offroad | slow-down drift | 4 | 12.62 | 0.01 | 9.43 | 0.00 | 0.02 | 0.71 |
| e1c6ae9f | offroad | turn / curve: cut inside | 4 | -1.38 | 6.29 | -7.60 | 0.33 | 23.69 | 25.43 |
| ea468790 | offroad | turn / curve: cut inside | 4 | -28.26 | -0.00 | -30.72 | 1.00 | -2.44 | -0.50 |
| 0f4b8753 | offroad | slow-down drift | 3 | 4.99 | -0.00 | 0.04 | 0.00 | -0.67 | 0.92 |
| 7750265d | offroad | straight: lateral offset | 3 | 1.12 | 0.00 | 1.89 | 0.00 | -1.86 | -0.49 |
| 13d646af | offroad | slow-down drift | 2 | 15.01 | 11.80 | -10.69 | 0.00 | -1.25 | 1.71 |
| 4e4a2a51 | offroad | turn / curve: cut inside | 2 | -8.32 | 0.00 | -16.45 | 0.67 | 0.06 | -0.08 |
| abd45a30 | offroad | turn / curve: wide outside | 2 | 19.37 | -0.01 | 19.86 | 0.00 | -0.82 | -1.64 |
| b680a689 | offroad | turn / curve: cut inside | 2 | 26.22 | 0.00 | 31.10 | 0.00 | 0.78 | 1.03 |
| bbc53aa8 | offroad | slow-down drift | 2 | 1.96 | 0.00 | 0.43 | 0.00 | -0.52 | 3.29 |
| 1e8e939e | offroad | turn / curve: cut inside | 1 | 0.88 | 1.76 | -2.33 | 0.33 | 2.84 | 5.62 |
| 3c2822d9 | offroad | slow-down drift | 1 | 0.78 | -0.01 | -0.54 | 0.00 | -0.26 | 0.84 |
| 68e53160 | offroad | turn / curve: cut inside | 1 | 30.72 | -0.00 | 34.77 | 0.00 | -2.37 | -0.24 |
| c92137b4 | offroad | slow-down drift | 1 | 7.04 | 0.00 | 1.62 | 0.00 | -0.31 | 2.17 |
| db8693db | offroad | straight: lateral offset | 1 | -0.97 | 0.00 | -0.75 | 0.33 | 0.34 | 0.40 |
| de9db5dc | offroad | turn / curve: cut inside | 1 | -36.25 | 0.00 | -31.20 | 1.00 | 1.28 | nan |
| f7ded74f | offroad | straight: lateral offset | 1 | -1.52 | -0.00 | -0.22 | 0.00 | 1.02 | nan |
| 4a863a41 | zero-other |  | 4 | -4.49 | -3.74 | -0.36 | 1.00 | -0.20 | -3.66 |
| 8a143cb2 | zero-other |  | 4 | 8.09 | 3.26 | 1.91 | 0.00 | -0.54 | 3.21 |
| 98f88a4e | zero-other |  | 4 | -10.28 | 0.18 | -7.82 | 1.00 | 3.51 | -0.02 |
| e904e9c0 | zero-other |  | 4 | -1.50 | -3.38 | 2.00 | 0.08 | -0.02 | -5.51 |
| 0feb3afa | zero-other |  | 3 | -3.57 | -0.00 | 0.26 | 0.67 | 2.14 | 0.29 |
| 36a2ab7f | zero-other |  | 3 | -4.63 | -3.62 | -0.85 | 1.00 | 3.66 | -3.43 |
| 05855d57 | zero-other |  | 2 | -3.33 | -3.67 | 0.89 | 1.00 | 0.74 | -3.34 |
| 2fe500f7 | zero-other |  | 2 | -6.79 | -6.75 | -0.10 | 1.00 | -0.15 | -3.97 |
| 4cb05d65 | zero-other |  | 2 | 0.00 | -0.00 | -0.08 | 0.00 | 1.24 | -3.30 |
| 8120803d | zero-other |  | 2 | 2.18 | 0.00 | 0.01 | 0.00 | -3.11 | 0.63 |
| 87a3b6d4 | zero-other |  | 2 | -0.80 | -3.59 | 4.80 | 0.00 | 0.26 | -3.53 |
| b27b3375 | zero-other |  | 2 | -3.36 | -3.89 | 1.08 | 1.00 | 0.42 | -4.05 |
| 13d646af | zero-other |  | 1 | -2.39 | 0.00 | 3.58 | 0.67 | 4.40 | 1.53 |
| 16300901 | zero-other |  | 1 | -3.54 | 0.03 | 0.72 | 0.67 | 5.10 | 0.38 |
| 1e31b2ad | zero-other |  | 1 | 5.11 | -0.00 | -0.89 | 0.00 | -4.72 | 0.08 |
| 422c7297 | zero-other |  | 1 | -1.39 | 0.01 | -0.04 | 0.00 | 1.10 | -2.83 |
| 478d6ec4 | zero-other |  | 1 | 4.84 | 4.68 | -1.18 | 0.00 | -0.14 | 3.80 |
| c0c88642 | zero-other |  | 1 | -3.28 | -3.58 | 0.07 | 1.00 | 3.15 | -3.62 |
| f1575120 | zero-other |  | 1 | -0.04 | 0.00 | -0.20 | 0.00 | 0.21 | -3.64 |

### 3b. Before the ego leaves the human's lane

Leave time = the first decision whose ego is more than 1.75 m (zero-other) / 1.0 m (offroad) from the human log (d_log), else the event; the route at 2 / 1 / 0.5 s before it. Classes of the fed route against the ego's lane at 40 m: lane-centred (|off| < 1 m); off lane with the human (log, at the plan's 4 s station) still in the ego's lane = a route-lane artefact; off lane with the human in the route's lane = the human's own lane change ahead.

| fclass | n | left_lane | lead_med | to_right | to_left |
|--:|--:|--:|--:|--:|--:|
| offroad | 51 | 0.51 | 0.00 | 17.00 | 9.00 |
| zero-other | 41 | 1.00 | 3.40 | 4.00 | 37.00 |

| fclass | lead | who | n | lane-centred % | off lane, human in ego lane % | off lane, human in route lane % | off lane, other % | route off on the side the ego leaves to % | cmd R % | cmd L % |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| offroad | 0.50 | MV | 51.00 | 82.35 | 3.92 | 1.96 | 11.76 | 3.92 | 33.33 | 35.29 |
| offroad | 0.50 | NTV0 same station | 51.00 | 84.31 | 3.92 | 0.00 | 11.76 | 3.92 | 37.25 | 25.49 |
| offroad | 1.00 | MV | 51.00 | 86.27 | 3.92 | 1.96 | 7.84 | 3.92 | 33.33 | 33.33 |
| offroad | 1.00 | NTV0 same station | 51.00 | 86.27 | 3.92 | 0.00 | 9.80 | 3.92 | 37.25 | 25.49 |
| offroad | 2.00 | MV | 51.00 | 88.24 | 3.92 | 0.00 | 7.84 | 3.92 | 35.29 | 21.57 |
| offroad | 2.00 | NTV0 same station | 51.00 | 88.24 | 3.92 | 0.00 | 7.84 | 3.92 | 29.41 | 25.49 |
| zero-other | 0.50 | MV | 41.00 | 87.80 | 12.20 | 0.00 | 0.00 | 7.32 | 65.85 | 14.63 |
| zero-other | 0.50 | NTV0 same station | 41.00 | 87.80 | 12.20 | 0.00 | 0.00 | 7.32 | 21.95 | 14.63 |
| zero-other | 1.00 | MV | 41.00 | 87.80 | 12.20 | 0.00 | 0.00 | 7.32 | 46.34 | 9.76 |
| zero-other | 1.00 | NTV0 same station | 41.00 | 85.37 | 14.63 | 0.00 | 0.00 | 9.76 | 21.95 | 9.76 |
| zero-other | 2.00 | MV | 41.00 | 87.80 | 12.20 | 0.00 | 0.00 | 7.32 | 31.71 | 17.07 |
| zero-other | 2.00 | NTV0 same station | 41.00 | 87.80 | 12.20 | 0.00 | 0.00 | 7.32 | 26.83 | 12.20 |
| NTV0 clean scenes | nan | all moving | 29022.00 | 88.17 | 3.87 | 1.70 | 1.40 | nan | 17.59 | 10.55 |
| MV clean scenes | nan | all moving | 118036.00 | 87.65 | 3.49 | 1.88 | 1.62 | nan | 18.66 | 14.34 |

## 4. Does the plan / the human label follow the route's lateral offset or the lane?

Moving (v0 >= 4), non-turn states; bins by the route's offset from the ego's lane centre at the plan's (label's) own 4 s arc; `d_f4` = the plan / label 4 s point's offset from the same lane centre; follow ratio = median d_f4 / median route offset; `stays` = |d_f4| < 1 m, `goes to route` = within 1 m of the route point.

| who | route_vs_lane_at_4s | n | d_route_med | d_f4_med | follow_ratio | stays_in_lane_pct | goes_to_route_pct |
|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 plan | route >= 2.5 m right of lane | 1641 | -3.60 | -0.87 | 0.24 | 47.41 | 12.37 |
| NTV0 plan | 1-2.5 m right | 264 | -2.17 | 0.16 | -0.07 | 89.02 | 4.55 |
| NTV0 plan | within 1 m (lane-centred) | 24251 | 0.00 | 0.34 | nan | 76.20 | 76.15 |
| NTV0 plan | 1-2.5 m left | 50 | 2.17 | 0.96 | 0.44 | 56.00 | 24.00 |
| NTV0 plan | >= 2.5 m left | 1018 | 3.51 | 1.50 | 0.43 | 30.65 | 23.58 |
| R3MV-s0 plan | route >= 2.5 m right of lane | 1782 | -3.51 | -0.96 | 0.27 | 48.93 | 15.32 |
| R3MV-s0 plan | 1-2.5 m right | 244 | -2.16 | 0.14 | -0.06 | 95.90 | 2.46 |
| R3MV-s0 plan | within 1 m (lane-centred) | 21871 | 0.00 | 0.32 | nan | 72.62 | 72.59 |
| R3MV-s0 plan | 1-2.5 m left | 42 | 2.26 | 1.22 | 0.54 | 19.05 | 35.71 |
| R3MV-s0 plan | >= 2.5 m left | 1408 | 3.46 | 0.70 | 0.20 | 57.03 | 17.12 |
| R3MV-s1 plan | route >= 2.5 m right of lane | 1454 | -3.57 | -0.93 | 0.26 | 48.28 | 13.48 |
| R3MV-s1 plan | 1-2.5 m right | 241 | -2.16 | 0.18 | -0.08 | 88.80 | 3.73 |
| R3MV-s1 plan | within 1 m (lane-centred) | 21600 | 0.00 | 0.30 | nan | 72.68 | 72.68 |
| R3MV-s1 plan | 1-2.5 m left | 101 | 2.26 | 1.19 | 0.53 | 28.71 | 31.68 |
| R3MV-s1 plan | >= 2.5 m left | 1200 | 3.45 | 0.93 | 0.27 | 48.25 | 17.17 |
| MVLC-s0 plan | route >= 2.5 m right of lane | 1751 | -3.50 | -0.75 | 0.21 | 55.17 | 9.48 |
| MVLC-s0 plan | 1-2.5 m right | 246 | -2.16 | 0.13 | -0.06 | 86.99 | 2.85 |
| MVLC-s0 plan | within 1 m (lane-centred) | 22043 | 0.00 | 0.32 | nan | 74.12 | 74.19 |
| MVLC-s0 plan | 1-2.5 m left | 94 | 2.26 | 1.16 | 0.52 | 32.98 | 24.47 |
| MVLC-s0 plan | >= 2.5 m left | 1310 | 3.49 | 0.84 | 0.24 | 51.45 | 20.31 |
| MVLC-s1 plan | route >= 2.5 m right of lane | 1733 | -3.52 | -0.76 | 0.22 | 51.53 | 11.37 |
| MVLC-s1 plan | 1-2.5 m right | 249 | -2.16 | 0.20 | -0.09 | 81.53 | 4.82 |
| MVLC-s1 plan | within 1 m (lane-centred) | 21881 | 0.00 | 0.31 | nan | 73.48 | 73.51 |
| MVLC-s1 plan | 1-2.5 m left | 81 | 2.30 | 1.25 | 0.54 | 9.88 | 38.27 |
| MVLC-s1 plan | >= 2.5 m left | 1197 | 3.44 | 0.86 | 0.25 | 52.05 | 18.05 |
| human test | route >= 2.5 m right of lane | 94 | -3.34 | -0.10 | 0.03 | 68.09 | 21.28 |
| human test | 1-2.5 m right | 50 | -2.16 | 0.04 | -0.02 | 88.00 | 2.00 |
| human test | within 1 m (lane-centred) | 3815 | -0.00 | 0.23 | nan | 80.50 | 80.45 |
| human test | 1-2.5 m left | 2 | 1.36 | -0.68 | -0.50 | 50.00 | 0.00 |
| human test | >= 2.5 m left | 123 | 3.29 | 0.45 | 0.14 | 66.67 | 21.95 |
| human train | route >= 2.5 m right of lane | 550 | -3.25 | -0.20 | 0.06 | 64.36 | 18.00 |
| human train | 1-2.5 m right | 77 | -2.06 | -0.79 | 0.38 | 74.03 | 16.88 |
| human train | within 1 m (lane-centred) | 8631 | 0.00 | 0.17 | nan | 81.65 | 81.68 |
| human train | 1-2.5 m left | 26 | 1.77 | 0.67 | 0.38 | 30.77 | 30.77 |
| human train | >= 2.5 m left | 471 | 3.29 | 0.58 | 0.18 | 62.21 | 17.41 |

Lane-centred route (|lane off| < 1 m at 40 m), by the lane's own bend (the curvature part of y_rt40): the plan / label 4 s point against the lane centre (negative = right).

| who | lane_bend | n | d_f4_med | d_f4_p10 | d_f4_p90 | right_of_lane_gt1_pct | left_of_lane_gt1_pct |
|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 plan | right bend (curv part < -3 m at 40 m) | 3366 | 0.57 | -0.76 | 3.71 | 7.72 | 37.37 |
| NTV0 plan | gentle right (-3..-1) | 2552 | 0.15 | -0.87 | 1.71 | 8.97 | 19.47 |
| NTV0 plan | straight (|curv| < 1) | 24810 | 0.34 | -0.25 | 1.64 | 2.56 | 17.96 |
| NTV0 plan | gentle left (1..3) | 2765 | 0.32 | -0.67 | 1.45 | 6.62 | 15.33 |
| NTV0 plan | left bend (> 3) | 3177 | 0.38 | -3.40 | 2.35 | 23.36 | 32.67 |
| R3MV-s0 plan | right bend (curv part < -3 m at 40 m) | 3466 | 0.36 | -1.36 | 2.21 | 13.85 | 28.74 |
| R3MV-s0 plan | gentle right (-3..-1) | 2394 | 0.23 | -0.56 | 1.66 | 6.73 | 21.51 |
| R3MV-s0 plan | straight (|curv| < 1) | 24103 | 0.31 | -0.34 | 1.72 | 4.01 | 19.21 |
| R3MV-s0 plan | gentle left (1..3) | 2771 | 0.16 | -1.05 | 1.16 | 10.50 | 12.09 |
| R3MV-s0 plan | left bend (> 3) | 3226 | 0.48 | -1.75 | 2.64 | 14.48 | 35.55 |
| R3MV-s1 plan | right bend (curv part < -3 m at 40 m) | 3378 | 0.27 | -1.48 | 2.14 | 15.19 | 27.92 |
| R3MV-s1 plan | gentle right (-3..-1) | 2360 | 0.26 | -0.66 | 1.71 | 6.99 | 22.71 |
| R3MV-s1 plan | straight (|curv| < 1) | 23981 | 0.29 | -0.38 | 1.75 | 4.25 | 19.22 |
| R3MV-s1 plan | gentle left (1..3) | 2780 | 0.17 | -1.14 | 1.20 | 11.33 | 12.48 |
| R3MV-s1 plan | left bend (> 3) | 3242 | 0.34 | -1.97 | 2.35 | 17.61 | 30.97 |
| MVLC-s0 plan | right bend (curv part < -3 m at 40 m) | 3545 | 0.37 | -1.16 | 2.15 | 11.48 | 28.72 |
| MVLC-s0 plan | gentle right (-3..-1) | 2302 | 0.28 | -0.43 | 1.73 | 5.69 | 23.50 |
| MVLC-s0 plan | straight (|curv| < 1) | 24013 | 0.29 | -0.34 | 1.70 | 4.06 | 18.35 |
| MVLC-s0 plan | gentle left (1..3) | 2919 | 0.16 | -1.05 | 1.01 | 10.79 | 10.21 |
| MVLC-s0 plan | left bend (> 3) | 3302 | 0.40 | -1.58 | 2.42 | 13.66 | 34.04 |
| MVLC-s1 plan | right bend (curv part < -3 m at 40 m) | 3576 | 0.30 | -1.37 | 1.93 | 13.90 | 26.85 |
| MVLC-s1 plan | gentle right (-3..-1) | 2332 | 0.27 | -0.41 | 1.68 | 4.25 | 21.96 |
| MVLC-s1 plan | straight (|curv| < 1) | 24059 | 0.30 | -0.29 | 1.73 | 3.54 | 18.94 |
| MVLC-s1 plan | gentle left (1..3) | 2894 | 0.13 | -1.12 | 1.05 | 11.13 | 10.61 |
| MVLC-s1 plan | left bend (> 3) | 3084 | 0.40 | -1.39 | 2.41 | 12.61 | 32.62 |
| human test | right bend (curv part < -3 m at 40 m) | 587 | 0.16 | -0.60 | 1.27 | 4.77 | 14.65 |
| human test | gentle right (-3..-1) | 423 | 0.18 | -0.45 | 1.38 | 1.65 | 17.26 |
| human test | straight (|curv| < 1) | 4099 | 0.18 | -0.26 | 1.28 | 2.76 | 13.44 |
| human test | gentle left (1..3) | 470 | 0.19 | -0.21 | 1.10 | 1.91 | 12.34 |
| human test | left bend (> 3) | 569 | 0.24 | -0.44 | 1.38 | 4.04 | 18.63 |
| human train | right bend (curv part < -3 m at 40 m) | 2173 | 0.14 | -0.73 | 1.05 | 6.26 | 11.09 |
| human train | gentle right (-3..-1) | 1018 | 0.14 | -0.54 | 1.16 | 3.63 | 12.18 |
| human train | straight (|curv| < 1) | 8290 | 0.15 | -0.36 | 1.22 | 4.16 | 12.01 |
| human train | gentle left (1..3) | 938 | 0.20 | -0.32 | 1.35 | 2.45 | 15.57 |
| human train | left bend (> 3) | 1853 | 0.34 | -0.47 | 1.54 | 5.50 | 20.67 |

## 5. Training rows (map-free): does the label go where the fed route points?

Rows fed v0 >= 4 m/s with the route covering the label's 4 s arc; `route lat4` = the fed route's lateral (ego frame) at the label's 4 s arc; `label - route` = the label's 4 s lateral minus it. PAI parts unweighted (not the draw weights).

| rows | set | n | cmd_L_pct | cmd_R_pct | cmd_unk_pct | route_lat4_gt175_pct | label_minus_route_abs_med | label_off_route_gt175_pct | when_route_gt175_label_follows_pct | when_route_gt175_label_straight_pct |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | moving | 69952 | 20.02 | 26.56 | 0.00 | 46.27 | 0.43 | 18.08 | 55.55 | 18.28 |
| ph | moving, label and route heading < 10 deg | 46077 | 9.57 | 13.08 | 0.00 | 26.44 | 0.35 | 16.60 | 41.11 | 43.64 |
| ph05 | moving | 56526 | 15.63 | 22.88 | 0.00 | 31.20 | 0.36 | 18.62 | 39.96 | 45.56 |
| ph05 | moving, label and route heading < 10 deg | 45631 | 11.99 | 13.23 | 0.00 | 22.78 | 0.33 | 19.48 | 18.07 | 71.80 |
| ph15 | moving | 45886 | 22.17 | 28.37 | 0.00 | 50.75 | 0.47 | 19.55 | 53.30 | 11.97 |
| ph15 | moving, label and route heading < 10 deg | 26742 | 8.19 | 13.06 | 0.00 | 26.39 | 0.31 | 15.11 | 44.09 | 35.24 |
| navtrain | moving | 34990 | 19.00 | 8.52 | 0.00 | 49.07 | 0.28 | 16.54 | 61.25 | 8.95 |
| navtrain | moving, label and route heading < 10 deg | 20560 | 1.82 | 1.18 | 0.00 | 24.33 | 0.20 | 10.55 | 59.16 | 23.71 |

<!-- generated end -->

## Route input: what the model receives

NTV0, R3MV and MV-LC all carry the ROUTE-MEM branch in its **projected layout**: the sidecar `route_mem.json` has no `order` / `anchor` head, n_pt 40, nf 5. There are two route inputs.

1. **Route memory, 200 numbers** (`experiments/route_mem/lib/rmem.py`: `Tracker.features` -> `memory` -> `merge` + `window`; served in
   `experiments/alpasim/lib/pai_driver.py` `submit_route` / `rt.features`, logged as `rmem`).
   - Source: every AlpaSim route message of the episode. A message holds up to 20 waypoints every 80 / 19 m, resampled from the ego's projection on AlpaSim's map route, with points closer than 40 m chord dropped, so the first point is about 42 m ahead. Waypoints are in the rig frame at the message time.
   - Each message is carried into the current rear-axle frame with the logged ego poses, then merged oldest first (a newer batch only appends points beyond the merged end).
   - The ego is projected onto the merged polyline. The memory is sampled at **40 route-arc offsets, -6 .. +72 m every 2 m**. Each point is [valid, x / 20, y / 20 (clipped to +-4), cos h, sin h] in the **current rear-axle frame** (x forward, y left); invalid points are zeros.
   - It is a geometric polyline in the ego frame. It is neither Frenet nor lane-relative, and it is not normalised for curvature.
   - At the first decisions only the far points (from about 42 m) are valid.
   - `rm_drop` 0.25 in training.
2. **Command**, 3 numbers in the ego row (`ego[1:4]`, `lib/parity_adapter.py` `ego_features`). PAI serve and PAI rows use the challenge sample's `command_from_route`: the first waypoint at least 5 m away, which is about 42 m ahead; y > 2 m gives LEFT, y < -2 m gives RIGHT, otherwise STRAIGHT. navtrain rows carry NAVSIM's own driving command, which is junction intent.

**Where the route comes from.** AlpaSim's `RouteGeneratorMap` (`third_party/alpasim/src/runtime/alpasim_runtime/route_generator.py`):
- every log pose is projected onto the centre line of one NuRec lane, chosen among the lanes within 4 m and pi / 8 of the heading;
- the candidate that persists over the most upcoming poses wins, and it is kept while it stays a candidate;
- past the log, the route follows that lane's successors.

So the route is lane-centred by construction, but the lane need not be the human's. The fed +40 m point lies on this polyline (median 0.002 m, section 0; the 2.7 % above 0.5 m are the documented merge fold-backs at loops).

## Reading

**Is the route lane-centred? Yes, on the NuRec lane centre, and usually on the ego's own lane** (section 1: |lane off| p50 0.00 m in every road class). What makes it look like lateral intent in the ego frame:
- **Straight roads:** the ego's own pose. The human drives about 0.2-0.4 m left of the NuRec centre, and a small heading error carried 40 m adds to it (|pose| p50 0.4 m, |curv| p50 0.13 m). The fed route therefore sits about 0.3-0.5 m right of the heading line at 40 m on straight roads (y_rt40 median -0.28 to -0.36 m on clean scenes). This is a steady small right bias.
- **Gentle curves (3-15 deg over 40 m):** the curvature (|curv| p50 2.1 m at 40 m).
- **The lane-choice artefact:** on 10 % (test 281) / 14 % (train 719) of straight human states the route is in another lane (>= 2.5 m off at 40 m). There the human's 4 s point stays in its own lane 65-72 % of the time and reaches the route only 16-19 % of the time. 92 of 278 test scenes have at least one such state.

**The command is where the ego-frame geometry becomes an explicit intent.**
- On human states the rule fires RIGHT on 9-12 % of straight states and on 30-32 % of gentle-curve states (LEFT on 8-9 % and 19-21 %).
- Cause of the non-turn turn commands:
  - straight roads: pose and lane offset;
  - gentle curves: curvature.
- Training rows (section 5): non-turn rows carry a turn command 23 % of the time in the PAI human pools (L 9.6 / R 13.1 %) against 3 % in navtrain (1.8 / 1.2 %). The same one-hot means "junction" in navtrain and "geometry at 42 m" in PAI.

**Failure link: weak. The hypothesis is not the main cause of the MV family's extra zeros.**
- **Offroad (51 events):** the route is lane-centred on the ego's lane before 82-88 % of them (section 3b), the same as clean scenes (88 %) and NTV0 at the same station (84-88 %). A right lane offset before the event is rarer for MV (0-2 %) than for NTV0 (4-8 %) or clean scenes (3-4 %). cmd R is 21-35 % for MV and 25-37 % for NTV0: the events sit on bends, so it is not MV-specific.
- **Zero-other (41 events):** the ego leaves the human's lane a median 3.4 s before the event, **to the left in 37 / 41**. The right-biased route seen at the event leads is the route pointing back, not a precursor. Before leaving, the route is lane-centred 88 % (same as clean and NTV0). cmd R climbs from 32 % to 66 % over the last 2 s (NTV0 22-27 %) because the MV's own heading has already turned left. The MV ignores a RIGHT command that points back to the lane.
- **Where the artefact is causal:** a route placed in the adjacent lane, while the human stays in the ego's lane, precedes exactly **5 / 92 events in 3 scenes**, with the MV going to the route's side:
  - bbc53aa8: offroad to the right, 2 reads;
  - 4cb05d65: lane change to the left, 2 reads;
  - 422c7297: lane change to the left, 1 read.

  NTV0 saw the same input at the same station without failing. That is at most about 1.2 zeros per read.
- **Following:** where the route sits 3.5 m to the right, plans move about 25 % of the way (NTV0 0.24, MV 0.21-0.27; humans 0.03-0.06). This is a family trait, not an MV one. For routes on the left, MV follows less than NTV0 (0.20-0.27 vs 0.43).
- **Bends with a lane-centred route** (section 4, second table):
  - MV medians are closer to the lane centre than NTV0 (right bends +0.27..+0.37 vs +0.57 m), but MV has a heavier inside tail (> 1 m right on right bends 11-15 % vs 7.7 %; humans 5-6 %). That is corner cutting, as in d431.
  - On gentle right curves MV is less often right of the lane than NTV0 (4-7 % vs 9 %). So "gentle curve read as a right lane change" does not show in aggregate.

**The two scenes the user flagged.**
- **596af0c7:** a right bend (curvature -3.3 m at 40 m) plus the MV's own pose turn the command to RIGHT for 0.2-0.4 s at t = 3.0-3.4 s. NTV0 never gets RIGHT there; its 40 m point was -1.3 m against -1.7 m for MV. MV leaves the lane to the right at 5.0-5.2 s, stays a lane right while the route points one lane back left (d_rt40 +3.5 m, its plan moves only about +1 m), and goes offroad at 6.4-6.9 s. NTV0 also drifts 1.2-1.9 m right and recovers. This is a real but single instance of curvature -> command -> lateral shift.
- **6e7b42a3:** over the 3 s before the event (12.7-13.3 s, 35-37 m/s) the route is lane-centred and points **left** in the ego frame (+0.5..+1.3 m; the ego is 0.5-0.9 m right of the lane centre), and the command is STRAIGHT. The MV plan goes 1.1-1.5 m right of the lane anyway. RIGHT appears only at 18-19 s, after the ego is 1.8-3.9 m left of the human: there the route is pulling it back.

**Human rows.** The PAI human label follows the lane, not a misplaced route: it stays within 1 m of its lane centre in 65-72 % of off-lane-route states and reaches the route in 16-19 % (section 4: follow ratio 0.03-0.06 for routes on the right). Map-free (section 5): when the fed route is more than 1.75 m lateral at the label's 4 s arc on non-turn rows, the label goes there 41 % (ph) / 18 % (ph05) / 44 % (ph15) of the time against 59 % for navtrain. The PAI rows teach a weaker route-to-label coupling than navtrain, not a stronger one.

**Verdict.** The route memory is lane-centred. The ego-frame "right arc" is curvature on bends and a small steady pose offset on straight roads, and it does not precede the MV family's offroad / zero-other zeros more often than on clean scenes or for NTV0. Two real representation defects remain:
1. the generator sometimes places the route in the adjacent lane (behind 5 / 92 events);
2. the PAI command rule turns gentle curvature, pose and that artefact into L / R intents, ten times as often as navtrain on non-turn rows. It shows in 596af0c7.

Neither explains the bulk of the zeros: corner cutting, slow-down drift, over-speed and the leftward lane departures of the zero-others. Line C (F1 / F2), DAgger and REC stay first. The representation sketch, with a cheap closed-loop test (`route_generator_type: RECORDED` on the 3 artefact scenes + the 9 stable ones), is in [plans/2026-10-21-route-frenet-sketch.md](../plans/2026-10-21-route-frenet-sketch.md) (Chinese).

Caveats: the current lane is the nearest NuRec lane centre within 4 m and pi / 8 (98.9 % of decisions have one). Its forward chain picks the successor nearest the route at forks, so at junctions the lane offset is unreliable (turn rows are excluded from the follow tables). The thresholds (1 / 1.75 / 2.5 m, 3 / 15 / 45 deg) were set here, not registered. There is one read per checkpoint.
