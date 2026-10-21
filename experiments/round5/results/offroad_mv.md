# OFFROAD-MV: what the PAI-human-row recipe's extra offroad zeros are, and which rows could cause them (ROUND5, 2026-10-21)

Diagnosis, no training, no GPU (pool CPU jobs only). The 9-12 extra offroad zeros that R3MV and MV-LC add on PAI 281 under ACC + GAP are
**not lane changes**: MV-LC keeps all 9 stable scenes. They split into four mechanisms:
- corner cutting on curves and turns;
- an unforced slow-down followed by a drift to the right edge;
- over-speed on a highway;
- a steady lateral offset on straight roads.

Tables first; the reading and the two recommended fixes follow the generated part.

<!-- generated: offroad_mv.py report -->


Stored reads only, no GPU. Reads: R3MV-L18 s0 / s1, MV-LC (R5LC-L18) s0 / s1, PAI 281 under ACC + GAP; NTV0-L18-s0 the same serve as reference. Event = an offroad zero (pai_fix fclass) on a scene where NTV0 is not offroad. Code: `scripts/offroad_mv.py` (`events`, `data`, `report`), lane `scripts/offroad_mv.sh` (pool CPU jobs, PRIO 15). Files: per event [offroad_mv_events.csv](offroad_mv_events.csv), per scene [offroad_mv.csv](offroad_mv.csv), plans at 3 / 2 / 1 s [offroad_mv_plan.csv](offroad_mv_plan.csv), read-wide plan scan [offroad_mv_pullscan.csv](offroad_mv_pullscan.csv).

## 1. Events

51 offroad zeros on 21 scenes (R3MV s0 / s1 / MV-LC s0 / s1: 12 / 12 / 12 / 15). Recurrence across the four reads (scenes): 4 of 4: 7, 3 of 4: 2, 2 of 4: 5, 1 of 4: 7.

Mechanism per event (rule, in this order): **slow-down drift** = ego speed < 0.6 x the human's at the event and the model plan 1 s before < 0.6 x the human's speed there; **over-speed drift** = straight (log turns < 10 deg over the 60 m around the event) and ego > 1.2 x the human; **turn / curve** = the log turns >= 10 deg over those 60 m, inside / outside by the sign of the ego's lateral offset; else **straight: lateral offset**. `d1_class` = D1's geometry class (d1_mac.lateral_one; its 10-15 deg bends are always called `cut`). Lateral: + left of the human log.

| scene | n_of_4 | mech | d1_class | v_ego | v_human | lat_ev | turn60 | ctx | NTV0 | R2J | reads |
|:--|--:|:--|:--|--:|--:|--:|--:|:--|:--|:--|:--|
| 6e7b42a3 | 4 | over-speed drift | straight: edge (\|lat\| < 3 m) | 36.58 | 26.45 | -0.42 | 0.13 | - | ok | offroad | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| dc401449 | 4 | slow-down drift | bend cut | 1.45 | 9.29 | -0.88 | 20.09 | - | ok | ok | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| a2b415d5 | 4 | straight: lateral offset | straight: edge (\|lat\| < 3 m) | 16.63 | 18.64 | -0.89 | 0.09 | - | ok | offroad | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| 596af0c7 | 4 | turn / curve: cut inside | straight: both crossed | 18.27 | 26.28 | -3.49 | -14.83 | - | rear-hit | offroad | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| 94c1d722 | 4 | turn / curve: cut inside | curve turn cut | 6.07 | 3.48 | -0.88 | -81.35 | - | slow | slow | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| e1c6ae9f | 4 | turn / curve: cut inside | straight: edge (\|lat\| < 3 m) | 13.74 | 11.43 | -0.77 | -11.33 | - | ok | offroad | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| ea468790 | 4 | turn / curve: cut inside | curve turn cut | 6.16 | 7.42 | -1.58 | -83.28 | stood before | ok | offroad | MVLC-s0 MVLC-s1 R3MV-s0 R3MV-s1 |
| 0f4b8753 | 3 | slow-down drift | straight: edge (\|lat\| < 3 m) | 4.32 | 10.37 | -1.62 | 1.38 | - | ok | offroad | MVLC-s0 MVLC-s1 R3MV-s1 |
| 7750265d | 3 | straight: lateral offset | straight: edge (\|lat\| < 3 m) | 11.31 | 14.40 | 0.33 | 8.55 | - | overshoot | offroad | MVLC-s0 R3MV-s0 R3MV-s1 |
| 13d646af | 2 | slow-down drift | straight: both crossed | 1.60 | 15.86 | -3.12 | 11.74 | - | ok | zero-other | MVLC-s0 R3MV-s0 |
| bbc53aa8 | 2 | slow-down drift | straight: ego left its lane | 5.83 | 34.71 | -3.64 | 2.17 | - | ok | ok | MVLC-s1 R3MV-s0 |
| 4e4a2a51 | 2 | turn / curve: cut inside | bend cut | 3.81 | 2.32 | -0.84 | -15.33 | - | overshoot | offroad | MVLC-s1 R3MV-s1 |
| b680a689 | 2 | turn / curve: cut inside | junction turn cut | 4.67 | 2.69 | 2.36 | 87.46 | - | ok | offroad | MVLC-s0 MVLC-s1 |
| abd45a30 | 2 | turn / curve: wide outside | junction turn wide | 5.22 | 6.48 | -0.66 | 89.25 | - | route-turn | offroad | MVLC-s0 R3MV-s0 |
| 3c2822d9 | 1 | slow-down drift | straight: edge (\|lat\| < 3 m) | 3.75 | 10.31 | -1.20 | 1.55 | - | ok | slow | MVLC-s1 |
| c92137b4 | 1 | slow-down drift | straight: edge (\|lat\| < 3 m) | 3.02 | 12.31 | -2.34 | 6.22 | - | ok | ok | R3MV-s1 |
| db8693db | 1 | straight: lateral offset | straight: edge (\|lat\| < 3 m) | 5.97 | 8.63 | 0.06 | -5.24 | - | ok | offroad | MVLC-s1 |
| f7ded74f | 1 | straight: lateral offset | straight: edge (\|lat\| < 3 m) | 26.60 | 26.91 | 1.18 | -0.84 | - | ok | ok | MVLC-s1 |
| 1e8e939e | 1 | turn / curve: cut inside | straight: edge (\|lat\| < 3 m) | 6.15 | 3.52 | -1.38 | -30.93 | - | route-turn | ok | MVLC-s1 |
| 68e53160 | 1 | turn / curve: cut inside | junction turn cut | 9.61 | 6.68 | 2.16 | 113.70 | stood before | route-turn | rear-hit | R3MV-s0 |
| de9db5dc | 1 | turn / curve: cut inside | curve turn cut | 5.18 | 2.25 | -1.72 | -80.57 | stood before | ok | rear-hit | R3MV-s1 |

Events by mechanism and recurrence (count of events; `scenes` = distinct scenes):

| mechanism | 1 of 4 | 2 of 4 | stable (3-4 of 4) | events | scenes |
|:--|--:|--:|--:|--:|--:|
| over-speed drift | 0 | 0 | 4 | 4 | 1 |
| slow-down drift | 2 | 4 | 7 | 13 | 6 |
| straight: lateral offset | 2 | 0 | 7 | 9 | 4 |
| turn / curve: cut inside | 3 | 4 | 16 | 23 | 9 |
| turn / curve: wide outside | 0 | 2 | 0 | 2 | 1 |

## 2. Stable events (offroad in >= 3 of the 4 reads): plan vs the human log and vs NTV0 at the same log station

Plan = the model plan (`poses_model`) at the decision 3 / 2 / 1 s before the event (MV: median over the reads that have the event); NTV0 = its decision whose ego is closest to the same log station (within 3 m). `lat_ego` / `lat_plan4` = lateral offset of the ego / the plan's 4 s point from the human log (m, + left); `hdg_ego` = ego heading minus the log's (deg); `turn_plan` / `turn_log` = the plan's 4 s heading change and the log's over the same arc from the ego's station (deg); `curv_ratio` = their ratio when the log turns >= 5 deg; `won` = the ACC term that set the speed (model = the model plan's own speed); `lead_d` = ACC lead gap (m, blank = no lead).

| scene | mech | t | who | v0 | v_h | v_plan | lat_ego | lat_plan4 | hdg_ego | turn_plan | turn_log | curv_ratio | won | lead_d |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|
| 0f4b8753 | slow-down drift | -3 s | MV (4 reads) | 12.87 | 10.77 | 7.71 | 0.47 | -0.62 | -0.41 | 5.12 | 4.00 | nan | model | nan |
| 0f4b8753 | slow-down drift | -3 s | NTV0 | 10.32 | 10.77 | 9.47 | 0.34 | -0.22 | -1.36 | 6.29 | 4.23 | nan | set | nan |
| 0f4b8753 | slow-down drift | -2 s | MV (4 reads) | 8.96 | 10.65 | 3.27 | -0.13 | -1.07 | -7.59 | 9.08 | 2.10 | nan | model | nan |
| 0f4b8753 | slow-down drift | -2 s | NTV0 | 10.32 | 10.67 | 9.16 | 0.13 | -0.53 | -1.49 | 6.51 | 3.92 | nan | set | nan |
| 0f4b8753 | slow-down drift | -1 s | MV (4 reads) | 6.19 | 10.51 | 2.81 | -1.10 | -2.02 | -6.81 | 6.60 | 2.23 | nan | model | nan |
| 0f4b8753 | slow-down drift | -1 s | NTV0 | 10.32 | 10.52 | 9.36 | -0.08 | -0.05 | -1.94 | 6.50 | 2.72 | nan | set | nan |
| 596af0c7 | turn / curve: cut inside | -3 s | MV (4 reads) | 21.63 | 26.45 | 20.90 | -0.00 | 0.32 | -1.01 | -16.30 | -19.91 | 0.82 | curve | 39.26 |
| 596af0c7 | turn / curve: cut inside | -3 s | NTV0 | 21.63 | 26.45 | 18.55 | -0.24 | -0.22 | -0.98 | -16.34 | -17.44 | 0.94 | curve | 39.19 |
| 596af0c7 | turn / curve: cut inside | -2 s | MV (4 reads) | 20.43 | 26.43 | 18.82 | -0.76 | -2.18 | -3.46 | -13.85 | -19.16 | 0.72 | curve | nan |
| 596af0c7 | turn / curve: cut inside | -2 s | NTV0 | 20.41 | 26.43 | 17.53 | -0.61 | -0.90 | -1.07 | -16.29 | -17.86 | 0.91 | curve | 49.40 |
| 596af0c7 | turn / curve: cut inside | -1 s | MV (4 reads) | 18.92 | 26.41 | 16.40 | -2.30 | -2.32 | -4.54 | -10.54 | -17.65 | 0.60 | model | nan |
| 596af0c7 | turn / curve: cut inside | -1 s | NTV0 | 19.01 | 26.41 | 16.78 | -1.13 | -1.10 | -1.32 | -15.59 | -18.12 | 0.86 | ramp | 51.74 |
| 6e7b42a3 | over-speed drift | -3 s | MV (4 reads) | 34.20 | 27.06 | 35.73 | 0.53 | -0.15 | -0.45 | 0.19 | -0.06 | nan | ramp | nan |
| 6e7b42a3 | over-speed drift | -3 s | NTV0 | 27.42 | 27.06 | 26.16 | 0.62 | 0.59 | -0.09 | -0.20 | -0.21 | nan | set | nan |
| 6e7b42a3 | over-speed drift | -2 s | MV (4 reads) | 35.00 | 26.90 | 37.07 | 0.22 | -0.55 | -0.61 | 0.60 | 0.01 | nan | ramp | nan |
| 6e7b42a3 | over-speed drift | -2 s | NTV0 | 27.42 | 26.89 | 26.34 | 0.59 | 0.46 | -0.04 | -0.29 | 0.08 | nan | set | nan |
| 6e7b42a3 | over-speed drift | -1 s | MV (4 reads) | 35.80 | 26.64 | 37.62 | -0.14 | -0.73 | -0.61 | 0.65 | 0.09 | nan | ramp | nan |
| 6e7b42a3 | over-speed drift | -1 s | NTV0 | 27.42 | 26.65 | 26.17 | 0.61 | 0.24 | 0.04 | -0.23 | 0.17 | nan | set | nan |
| 7750265d | straight: lateral offset | -3 s | MV (4 reads) | 15.61 | 17.96 | 13.05 | 0.05 | -0.14 | 0.48 | 0.80 | 7.13 | 0.11 | model | 40.99 |
| 7750265d | straight: lateral offset | -3 s | NTV0 | 15.77 | 17.96 | 14.00 | 0.20 | -0.76 | 0.11 | 0.89 | 8.26 | 0.11 | curve | 44.24 |
| 7750265d | straight: lateral offset | -2 s | MV (4 reads) | 14.21 | 17.03 | 11.59 | 0.21 | -1.29 | 0.62 | 0.51 | 8.62 | 0.06 | model | 49.33 |
| 7750265d | straight: lateral offset | -2 s | NTV0 | 14.53 | 17.07 | 12.63 | 0.24 | -1.89 | 0.00 | 1.22 | 8.61 | 0.14 | curve | 58.14 |
| 7750265d | straight: lateral offset | -1 s | MV (4 reads) | 12.80 | 15.75 | 9.75 | 0.25 | -1.87 | -0.54 | 2.70 | 7.62 | 0.35 | model | nan |
| 7750265d | straight: lateral offset | -1 s | NTV0 | 12.87 | 15.75 | 10.39 | 0.24 | -1.55 | 0.35 | 1.65 | 5.09 | 0.32 | model | 42.83 |
| 94c1d722 | turn / curve: cut inside | -3 s | MV (4 reads) | 2.84 | 8.31 | 3.84 | -0.51 | 0.49 | 3.63 | -11.55 | -10.86 | 1.21 | ramp | nan |
| 94c1d722 | turn / curve: cut inside | -3 s | NTV0 | 6.03 | 8.31 | 2.95 | -0.37 | -0.11 | -0.67 | -5.51 | -4.96 | nan | model | nan |
| 94c1d722 | turn / curve: cut inside | -2 s | MV (4 reads) | 4.53 | 6.97 | 4.88 | -0.31 | -0.16 | 3.47 | -48.13 | -44.10 | 1.09 | curve | nan |
| 94c1d722 | turn / curve: cut inside | -2 s | NTV0 | 4.89 | 6.99 | 2.72 | -0.39 | -0.04 | -0.11 | -8.81 | -7.64 | 1.15 | model | nan |
| 94c1d722 | turn / curve: cut inside | -1 s | MV (4 reads) | 6.09 | 4.59 | 7.12 | -0.08 | -0.35 | -1.26 | -72.27 | -79.06 | 0.93 | curve | nan |
| 94c1d722 | turn / curve: cut inside | -1 s | NTV0 | 4.98 | 4.56 | 5.33 | -0.34 | -0.66 | 0.62 | -73.66 | -66.97 | 1.08 | curve | nan |
| a2b415d5 | straight: lateral offset | -3 s | MV (4 reads) | 16.63 | 18.02 | 15.62 | -0.63 | -0.67 | -0.42 | 0.44 | -0.06 | nan | set | nan |
| a2b415d5 | straight: lateral offset | -3 s | NTV0 | 16.63 | 18.02 | 15.01 | -0.43 | -0.32 | -0.20 | 0.23 | 0.01 | nan | set | nan |
| a2b415d5 | straight: lateral offset | -2 s | MV (4 reads) | 16.63 | 18.24 | 16.10 | -0.73 | -0.76 | -0.35 | 0.74 | 0.15 | nan | set | nan |
| a2b415d5 | straight: lateral offset | -2 s | NTV0 | 16.63 | 18.24 | 15.08 | -0.40 | -0.29 | 0.24 | 0.17 | 0.14 | nan | set | nan |
| a2b415d5 | straight: lateral offset | -1 s | MV (4 reads) | 16.63 | 18.45 | 15.64 | -0.79 | -0.76 | -0.26 | 0.51 | 0.11 | nan | set | nan |
| a2b415d5 | straight: lateral offset | -1 s | NTV0 | 16.63 | 18.45 | 14.84 | -0.29 | -0.19 | 0.31 | -0.11 | 0.09 | nan | set | nan |
| dc401449 | slow-down drift | -3 s | MV (4 reads) | 8.41 | 9.47 | 4.04 | -0.14 | -0.39 | 1.16 | 7.97 | 7.68 | 1.00 | ramp | nan |
| dc401449 | slow-down drift | -3 s | NTV0 | 10.81 | 9.47 | 13.02 | -0.26 | -2.90 | 1.42 | 23.37 | 32.75 | 0.71 | curve | nan |
| dc401449 | slow-down drift | -2 s | MV (4 reads) | 5.32 | 9.40 | 1.80 | -0.26 | -0.79 | -5.16 | 10.58 | 4.13 | nan | model | nan |
| dc401449 | slow-down drift | -2 s | NTV0 | 11.51 | 9.40 | 13.81 | -0.06 | -3.90 | 1.72 | 21.89 | 37.92 | 0.58 | curve | nan |
| dc401449 | slow-down drift | -1 s | MV (4 reads) | 2.85 | 9.34 | 0.76 | -0.68 | -0.56 | -6.70 | 4.77 | 2.04 | nan | model | nan |
| dc401449 | slow-down drift | -1 s | NTV0 | 11.71 | 9.34 | 13.24 | 0.10 | -2.78 | 1.77 | 22.77 | 37.08 | 0.61 | curve | nan |
| e1c6ae9f | turn / curve: cut inside | -3 s | MV (4 reads) | 17.23 | 13.25 | 19.10 | -0.41 | 4.85 | 0.23 | -0.23 | -11.34 | 0.02 | curve | nan |
| e1c6ae9f | turn / curve: cut inside | -3 s | NTV0 | 17.16 | 13.24 | 17.67 | -0.21 | 3.35 | -0.34 | 0.96 | -11.47 | -0.08 | curve | nan |
| e1c6ae9f | turn / curve: cut inside | -2 s | MV (4 reads) | 15.89 | 13.38 | 13.73 | -0.30 | 3.05 | 0.11 | -0.48 | -11.89 | 0.04 | model | nan |
| e1c6ae9f | turn / curve: cut inside | -2 s | NTV0 | 16.08 | 13.38 | 16.18 | -0.27 | 5.30 | -0.25 | 0.51 | -11.67 | -0.04 | curve | nan |
| e1c6ae9f | turn / curve: cut inside | -1 s | MV (4 reads) | 13.66 | 13.03 | 12.64 | -0.38 | -1.62 | -0.85 | -17.74 | -11.86 | 1.49 | model | nan |
| e1c6ae9f | turn / curve: cut inside | -1 s | NTV0 | 14.63 | 13.02 | 14.87 | -0.37 | 2.03 | -0.37 | -8.33 | -11.65 | 0.72 | ramp | nan |
| ea468790 | turn / curve: cut inside | -3 s | MV (4 reads) | 3.79 | 1.54 | 6.19 | 0.21 | -0.93 | 4.75 | -72.80 | -66.05 | 1.10 | ramp | nan |
| ea468790 | turn / curve: cut inside | -3 s | NTV0 | 3.21 | 1.55 | 5.17 | 0.09 | -0.96 | 5.26 | -72.73 | -59.37 | 1.23 | ramp | nan |
| ea468790 | turn / curve: cut inside | -2 s | MV (4 reads) | 5.03 | 3.31 | 6.66 | 0.40 | -1.99 | -0.70 | -57.35 | -58.64 | 0.98 | ramp | nan |
| ea468790 | turn / curve: cut inside | -2 s | NTV0 | 5.07 | 3.28 | 7.07 | 0.53 | -0.18 | 4.67 | -65.39 | -59.45 | 1.10 | ramp | nan |
| ea468790 | turn / curve: cut inside | -1 s | MV (4 reads) | 5.19 | 5.98 | 6.94 | -0.06 | -2.62 | -10.14 | -30.60 | -43.52 | 0.70 | ramp | nan |
| ea468790 | turn / curve: cut inside | -1 s | NTV0 | 5.23 | 6.00 | 5.91 | 0.78 | -1.68 | -1.35 | -42.70 | -42.41 | 1.01 | set | nan |

## 3. Read-wide: unforced slow-down plans

Every 0.5 s decision with ego v0 >= 4 m/s where the human, at the ego's log station, keeps >= 0.8 of its speed (>= 3 m/s) over the next 4 s: `slow` = the model plan's end speed < 0.6 v0; `right` = the plan's 4 s point > 0.8 m right with |yaw| < 20 deg; `both`. `scenes_slow5` = scenes with >= 5 such slow plans (>= 2.5 s of them).

| read | scenes | decisions | slow | right | both | scenes_slow5 | scenes_both3 | slow % | right % | both % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0 | 280 | 6689 | 284 | 1201 | 17 | 12 | 1 | 4.25 | 17.95 | 0.25 |
| R2C | 281 | 6706 | 290 | 1220 | 18 | 18 | 1 | 4.32 | 18.19 | 0.27 |
| R2J | 281 | 6551 | 618 | 1384 | 54 | 61 | 6 | 9.43 | 21.13 | 0.82 |
| R3MV-s0 | 280 | 6676 | 672 | 1379 | 64 | 66 | 8 | 10.07 | 20.66 | 0.96 |
| R3MV-s1 | 278 | 6658 | 654 | 1323 | 50 | 67 | 8 | 9.82 | 19.87 | 0.75 |
| MVLC-s0 | 280 | 6736 | 610 | 1350 | 55 | 60 | 7 | 9.06 | 20.04 | 0.82 |
| MVLC-s1 | 281 | 6701 | 672 | 1384 | 53 | 70 | 8 | 10.03 | 20.65 | 0.79 |

## 4. Training rows by state (moving rows, fed v0 >= 2 m/s; PAI parts under R3MV's draw weights, navtrain uniform)

`alat` = peak lateral acceleration of the label path (v^2 x curvature of consecutive label points, m/s^2); `y4 R / L` = share of rows whose 4 s point is > 0.8 m right / left; `lead` = r5 LEAD flag (PAI only). Road read-outs only on labels inside the raster (x < 52 m, |y| < 20 m; `in raster` share): `ml<0` = the label footprint leaves the road surface (PAI: round 5 road_pai raster = the evaluator's NuRec surface; navtrain: the nuPlan road raster of the S rows' hinge, a different surface: compare PAI parts with each other first); `ins tight` = the car centre comes within 1.6 m of the inside edge on the label (bends / turns).


**all moving**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 90631 | 9.6 | 0.28 | 2.11 | 3.8% | 26.0% | 22.2% | 1.00 | 40% | 68% | 0.6% | 2.9% | 6.50 |
| ph05 | 74250 | 5.2 | 0.08 | 0.53 | 0.0% | 15.4% | 13.3% | 1.00 | 39% | 90% | 0.4% | 2.3% | 6.75 |
| ph15 | 69095 | 13.5 | 0.68 | 5.06 | 21.9% | 32.2% | 26.5% | 1.00 | 41% | 44% | 1.1% | 3.3% | 7.00 |
| navtrain | 78144 | 5.5 | 0.35 | 1.58 | 0.3% | 20.0% | 39.0% | 0.87 |  | 99% | 2.2% | 3.5% | 5.75 |

**cruise straight (h < 5 deg, v_end >= 0.8 v0)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 43132 | 11.7 | 0.17 | 0.56 | 0.0% | 14.4% | 15.3% | 1.02 | 45% | 53% | 0.4% | 2.6% | 7.25 |
| ph05 | 47338 | 5.9 | 0.06 | 0.19 | 0.0% | 3.9% | 5.1% | 1.01 | 41% | 85% | 0.3% | 2.0% | 7.00 |
| ph15 | 25840 | 16.4 | 0.33 | 1.03 | 0.0% | 21.2% | 20.1% | 1.04 | 50% | 31% | 1.2% | 3.1% | 7.25 |
| navtrain | 10584 | 8.8 | 0.21 | 0.41 | 0.0% | 18.2% | 33.0% | 0.99 |  | 96% | 3.2% | 3.1% | 5.75 |

**curve / bend (15-45 deg, v0 >= 4)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 10612 | 8.8 | 1.81 | 3.06 | 11.5% | 60.3% | 38.7% | 1.05 | 27% | 80% | 1.1% | 4.1% | 4.75 |
| ph05 | 4061 | 5.7 | 0.75 | 1.17 | 0.0% | 65.4% | 34.5% | 1.01 | 31% | 100% | 0.0% | 2.2% | 5.00 |
| ph15 | 9663 | 13.7 | 3.36 | 6.27 | 60.2% | 62.0% | 36.2% | 1.07 | 29% | 40% | 2.1% | 5.3% | 5.00 |
| navtrain | 12307 | 5.4 | 1.07 | 1.95 | 0.3% | 30.6% | 68.8% | 0.98 |  | 100% | 0.8% | 4.3% | 5.00 |

**junction turn (>= 45 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 6684 | 6.1 | 2.51 | 4.07 | 32.5% | 54.9% | 45.1% | 1.12 | 30% | 78% | 0.5% | 2.8% | 3.25 |
| ph05 | 723 | 3.0 | 0.75 | 1.33 | 2.5% | 49.8% | 42.0% | 1.07 | 23% | 100% | 5.3% | 0.6% | 3.25 |
| ph15 | 8645 | 9.6 | 5.19 | 8.63 | 96.0% | 54.7% | 45.2% | 1.19 | 31% | 35% | 1.2% | 5.6% | 2.75 |
| navtrain | 8991 | 4.4 | 1.69 | 2.51 | 2.0% | 32.4% | 67.6% | 1.17 |  | 99% | 2.4% | 7.1% | 4.00 |

**decelerating (v_end < 0.6 v0, h < 20 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 12923 | 6.3 | 0.06 | 0.44 | 0.0% | 7.9% | 4.8% | 0.26 | 43% | 99% | 0.7% | 1.3% | 8.00 |
| ph05 | 3672 | 2.9 | 0.03 | 0.12 | 0.0% | 2.4% | 0.4% | 0.47 | 37% | 100% | 0.7% | 0.8% | 8.00 |
| ph15 | 13442 | 10.4 | 0.12 | 1.05 | 0.3% | 11.4% | 9.5% | 0.13 | 43% | 88% | 1.0% | 1.7% | 8.00 |
| navtrain | 25987 | 5.0 | 0.10 | 0.27 | 0.0% | 2.8% | 5.6% | 0.32 |  | 100% | 2.3% | 2.7% | 8.00 |

**pull right while slowing (v_end < 0.6 v0, y4 < -0.8 m, h < 20 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 1021 | 8.9 | 0.54 | 1.01 | 0.0% | 100.0% | 0.0% | 0.43 | 39% | 95% | 0.0% | 2.5% | 5.50 |
| ph05 | 87 | 3.4 | 0.21 | 0.38 | 0.0% | 100.0% | 0.0% | 0.54 | 48% | 100% | 0.0% | 0.0% | 8.00 |
| ph15 | 1532 | 15.5 | 0.89 | 1.97 | 1.4% | 100.0% | 0.0% | 0.33 | 37% | 68% | 0.0% | 3.1% | 7.00 |
| navtrain | 736 | 6.4 | 0.37 | 0.65 | 0.0% | 100.0% | 0.0% | 0.47 |  | 100% | 6.9% | 1.0% | 5.00 |

**pull left while slowing (v_end < 0.6 v0, y4 > 0.8 m, h < 20 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 619 | 8.8 | 0.50 | 1.03 | 0.5% | 0.0% | 100.0% | 0.44 | 37% | 96% | 3.0% | 3.7% | 8.00 |
| ph05 | 14 | 4.5 | 0.29 | 0.44 | 0.0% | 0.0% | 100.0% | 0.57 | 7% | 100% | 0.0% | 0.0% | 4.50 |
| ph15 | 1273 | 16.0 | 0.97 | 1.91 | 0.9% | 0.0% | 100.0% | 0.34 | 37% | 67% | 4.5% | 6.5% | 8.00 |
| navtrain | 1466 | 4.8 | 0.39 | 0.64 | 0.0% | 0.0% | 100.0% | 0.44 |  | 100% | 1.0% | 1.2% | 8.00 |

**fast (v0 >= 25)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 8748 | 29.6 | 0.40 | 1.15 | 1.0% | 28.6% | 31.2% | 1.00 | 50% | 0% | 0.0% | 0.0% |  |
| ph05 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| ph15 | 13291 | 36.6 | 0.94 | 4.02 | 13.9% | 40.7% | 33.2% | 1.00 | 44% | 0% | 0.0% | 0.0% | 8.00 |
| navtrain | 0 |  |  |  |  |  |  |  |  |  |  |  |  |

**fast (v0 >= 30)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 3919 | 33.5 | 0.52 | 1.06 | 0.0% | 23.1% | 40.7% | 1.00 | 48% | 0% | 0.0% | 0.0% |  |
| ph05 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| ph15 | 9878 | 40.7 | 0.91 | 3.70 | 13.0% | 37.9% | 34.9% | 1.00 | 51% | 0% | 0.0% | 0.0% |  |
| navtrain | 0 |  |  |  |  |  |  |  |  |  |  |  |  |

**slow-down to a stop (v0 >= 4, v_end < 0.5, h < 20 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 1978 | 5.4 | 0.05 | 0.26 | 0.0% | 2.8% | 1.4% | 0.00 | 35% | 100% | 0.8% | 0.0% | 8.00 |
| ph05 | 7 |  |  |  |  |  |  |  |  |  |  |  |  |
| ph15 | 4387 | 8.3 | 0.07 | 0.63 | 0.1% | 6.0% | 3.4% | 0.00 | 45% | 100% | 0.9% | 0.5% | 8.00 |
| navtrain | 1509 | 4.6 | 0.10 | 0.33 | 0.0% | 0.9% | 6.3% | 0.03 |  | 100% | 0.8% | 2.8% | 6.50 |

**slow-down, no stop (v0 >= 4, 0.5 <= v_end < 0.6 v0, h < 20 deg)**

| part | n | v50 | alat50 | alat90 | alat_gt3 | y4_right08 | y4_left08 | vend_v0 | lead | in_raster | ml_lt0 | ins_tight | ins_min50 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 7913 | 8.0 | 0.09 | 0.57 | 0.0% | 12.1% | 7.3% | 0.38 | 41% | 98% | 0.8% | 1.8% | 8.00 |
| ph05 | 676 | 4.7 | 0.05 | 0.19 | 0.0% | 3.8% | 1.3% | 0.53 | 30% | 100% | 0.0% | 0.4% | 8.00 |
| ph15 | 7538 | 13.5 | 0.21 | 1.23 | 0.4% | 16.8% | 14.8% | 0.33 | 39% | 78% | 1.3% | 2.6% | 8.00 |
| navtrain | 15895 | 6.3 | 0.11 | 0.30 | 0.0% | 4.4% | 5.5% | 0.41 |  | 100% | 3.1% | 2.6% | 8.00 |

Stretch f = 1.5 rows vs their plain base row (69095 drawn pairs): fed speed x 1.50, label peak lateral acceleration x 2.19 (median); > 3 m/s^2 21.9% vs 4.0%; fed v0 > 30 m/s 14.3% vs 4.1%; label footprint within 0.3 m of the edge (in-raster ph15 rows) 8.2% vs the same base rows 6.6%.

- ph: pull-right-while-slowing rows sit at clip position k / k_max median 0.39; 9% in the last 20 % of the clip (all drawn moving rows 20%): not an end-of-clip pull-over; 0.18 rows a batch.
- ph15: pull-right-while-slowing rows sit at clip position k / k_max median 0.54; 14% in the last 20 % of the clip (all drawn moving rows 24%): not an end-of-clip pull-over; 0.18 rows a batch.

Candidate fixes (rows at weight 0 under R3MV's draw; drawn moving rows, rows a batch of 64, share of the part's drawn moving mass):

| rule | drawn rows | rows a batch | share of part | scenes |
|:--|--:|--:|--:|--:|
| F1 stretch-1.5 rows with label lat. accel > 3 m/s^2 or fed v0 > 30 m/s | 23693 | 2.74 | ph15 34.3% | 313 |
| F1a stretch-1.5 rows with label lat. accel > 3 m/s^2 | 15098 | 1.75 | ph15 21.9% | 240 |
| F1b stretch-1.5 rows fed v0 > 30 m/s | 9878 | 1.14 | ph15 14.3% | 99 |
| F2 PAI rows slowing without a lead and without stopping (v0 >= 4, 0.5 <= v_end < 0.6 v0, h < 20 deg, no LEAD) | 9735 | 1.41 | ph 5.1% / ph05 0.6% / ph15 6.7% | 205 |
| F2a PAI rows slowing and pulling right (v_end < 0.6 v0, y4 < -0.8 m, h < 20 deg) | 2640 | 0.37 | ph 1.1% / ph05 0.1% / ph15 2.2% | 100 |

## 5. R5-R (R3MV + PAI road hinge, PAI-FIX native VC + LEAD, 118 scenes)

- offroad zeros: R5-R 15, R3MV 11, NTV0 14 (same 118-scene read, native serve); R5-R and R3MV share 10; new in R5-R 5 (15fd88d6 (TURN, off_curve_inside), 1d6e30bc (TURN, off_straight), 213dfdac (TURN, off_curve_outside), 59e085d7 (STOP, off_junc_inside), a68ad579 (T281-TURN, off_junc_outside)); fixed 1 (9e3fd12d).
- of the 21 OFFROAD-MV event scenes, 5 are in the PAI-FIX exam; R5-R offroad on 0 of them, R3MV-vl on 0: 1e8e939e, 4e4a2a51, 68e53160, 7750265d, abd45a30.
- R5-R's offroads are 13 / 15 on TURN-subset scenes: the PAI-FIX exam samples turns and stops, not the straight-road / slow-down scenes where the 281 events sit, so R5-R cannot say whether the hinge would touch them.

<!-- end generated -->


## Reading

**1. What the events are.** There are 51 offroad zeros on 21 scenes:

- 9 scenes recur in at least 3 of the 4 reads (36 events). The other 12 scenes are seed noise (1-2 of 4).
- The whole PAI-human-row family shares them. R2J (round 2, D1) is offroad on 11 of the 21 scenes and on 7 of the 9 stable ones.
- MV-LC removed the off-route lane-change rows. It keeps every stable scene, so these are not D1's lane-change hypothesis.
- 43 of 51 exits are on the right (the curb side).
- 6 events on 3 scenes follow a standing start (ea468790, de9db5dc, 68e53160: the turn right after a launch). No event follows a rear hit.

The mechanisms (rule in §1):

- **Corner cutting on curves and turns.** 23 events on 9 scenes, 16 of them stable (596af0c7, 94c1d722, e1c6ae9f, ea468790).
- **Slow-down drift.** 13 events on 6 scenes, stable on 0f4b8753 and dc401449. The model plan brakes on its own: ACC `won = model`, and
  ACC has no lead. The human keeps 9-35 m/s.
  - In 5 of the 6 scenes the plan 1 s before the event is below 3 m/s (0.8-2.8 m/s).
  - Once crawling, the ego heading turns 2.5-20 deg right (typically 7-8 deg), and the ego rolls off the right edge at 1-6 m/s.
  - At the same station NTV0 plans the human's speed.
  - In bbc53aa8 (2 of 4) the model slows from 35 m/s to 3 m/s on a highway and moves 3.6 m right: a pull-over with no cue.
- **Over-speed drift.** 6e7b42a3, 4 of 4. The ACC ramp follows the model's own speed, and the MV plan rises to 36-38 m/s. The human and NTV0
  drive 27 m/s. A 0.6 deg heading bias then reaches the right edge after about 8 s.
- **Straight-road lateral offset.** 9 events on 4 scenes.
  - In a2b415d5 MV rides 0.4-0.5 m right of NTV0 at the same speed, with no heading error. The road edge is close there.
  - 7750265d is also an NTV0 zero (overshoot), so it costs no points.

**2. The stable events, plan against the human and NTV0 (§2).**

- **Corner cutting.** In two scenes the cut is a chord; in none is the plan's curvature too high early on.
  - 596af0c7: the MV ego is 0.8 m inside the human path at -2 s and 2.3 m inside at -1 s, heading 3.5-4.5 deg further turned than the log.
    The plan then turns less than the log (curv_ratio 0.72 at -2 s, 0.60 at -1 s), so it straightens out of the cut.
  - ea468790: at -1 s the ego heading is 10 deg further right than the log, and the plan ends 2.6 m inside while turning less
    (curv_ratio 0.70).
  - 94c1d722: the plan follows the log's curvature (1.09 / 0.93 at -2 / -1 s) at a higher speed than NTV0, and it slides 0.9 m inside
    late in the curve.
  - NTV0 at the same station is 0.3-1.1 m off the log at -1 s, with curv_ratio 0.72-1.08.
  - In 3 of the 4 scenes the MV plan enters the curve faster than NTV0:
    - 94c1d722: 4.9 vs 2.7 m/s at -2 s, 7.1 vs 5.3 at -1 s, while the human slows to 4.6.
    - e1c6ae9f: 19.1 vs 17.7 m/s at -3 s (human 13.3).
    - 596af0c7: 20.9 vs 18.6 m/s at -3 s.
    - ACC's curve term binds in 596af0c7, 94c1d722 and e1c6ae9f (`won = curve`).
  - e1c6ae9f is a right fork. Both models under-turn it at -3 / -2 s: plan about 0 deg against the log's -11 deg, 3-5 m left of the log. At
    -1 s MV over-corrects (curv_ratio 1.49, plan end 1.6 m right) while NTV0 corrects gently (0.72).
- **Slow-down.** In 0f4b8753 the MV plan speed falls 7.7 -> 3.3 -> 2.8 m/s and the ego heading turns -7 deg; in dc401449 the plan falls
  4.0 -> 1.8 -> 0.8 m/s. The human holds 10.5 / 9.4 m/s and NTV0 plans 9.4 / 13 m/s.
- **Over-speed.** The plan is at 35.7-37.6 m/s against 26.2 m/s for NTV0. The ego goes from +0.5 m at -3 s to -0.4 m at the event,
  and the plan's 4 s point is 0.7 m right.
- **What none of the stable events show.** No clear early over-curvature: curv_ratio is 0.8-1.2 at -3 s, and NTV0 is the same. No plan
  overshoots a turn by under-curving it, except the late fork in e1c6ae9f.

**3. Read-wide (§3).** The slow-down is a property of the recipe, not of a few scenes:

- Count plans that slow while the human at that station keeps its speed (end speed below 0.6 x v0).
  - These are 9.1-10.1 % of eligible decisions in R2J and in both seeds of both R3MV and MV-LC.
  - They are 4.3 % in NTV0 and in R2C (the same continuation without PAI human rows).
  - Scenes with at least 2.5 s of them: 60-70 against 12-18.
- Rightward plans alone move only 18 -> 20 %. Slow plus right triples (0.25 -> 0.8 %).

**4. Training rows (§4).**

- **The stretch f = 1.5 rows are outside the physical envelope on turns.**
  - They keep the human path and run it at 1.5 x the speed, so the label's peak lateral acceleration is x 2.19 (median).
  - On curve / bend rows the median is 3.36 m/s^2 and 60 % of rows exceed 3 m/s^2. On junction turns the median is 5.19 m/s^2 and 96 %
    exceed 3 m/s^2.
  - Plain PAI is at 1.81 / 2.51 m/s^2 (11.5 % / 32.5 % above 3). navtrain is at 1.07 / 1.69 m/s^2 (0.3 % / 2.0 %).
  - The stretch also makes the only rows above 30 m/s: 14.3 % of drawn ph15 rows, median 40.7 m/s. navtrain has no row at or above 25 m/s.
    This is the regime of the over-speed scene. On straight cruise rows ph15 labels also accelerate more (v_end / v0 1.04; 37 % of rows
    accelerate by more than 15 %, navtrain 26 %).
- **The human paths themselves are not tight.**
  - On in-raster curve rows the label footprint leaves the NuRec surface on 1.1 % (ph) and 2.1 % (ph15). The car centre comes within
    1.6 m of the inside edge on 4.1 % / 5.3 %. navtrain on its own raster: 0.8 % / 4.3 %.
  - The rows do not teach a cut line. What PAI adds is speed on the same line. Plain PAI curve rows are already 1.7 x navtrain's lateral
    acceleration, and the stretch doubles that again.
- **Slow-downs.** The rows give no clean handle.
  - PAI moving rows that slow without stopping are 8.7 % of ph and 11 % of ph15, against 20 % of navtrain. The doubling of unforced
    slow plans is therefore not a share effect.
  - 59-61 % of the PAI slow-downs have no LEAD (r5 rule). Lights, stop lines and pedestrians are not labelled, so "no LEAD" is not "no cue".
  - Rows that slow and pull right are rare (0.37 rows a batch, ph 1.1 % / ph15 2.2 %). They are not clustered at clip ends, so they are
    not end-of-recording pull-overs.
- **The road hinge (R5-R) cannot reach these mechanisms.** The human labels are on the road (footprint off on about 1 %), and the hinge only
  fires on plans. On its own exam R5-R has 15 offroad zeros against R3MV's 11, and 13 of the 15 are on TURN scenes. None of the 5 event
  scenes that are in the PAI-FIX exam goes offroad in either native read. The exam has none of the straight-road slow-down / over-speed
  scenes, so R5-R gives no read on them.

## Recommended data-only fixes (ranked by expected recovered offroad zeros)

Both are draw-file changes on R3MV's argv (or MV-LC's), built like R5LC: rows at weight 0, each PAI part renormalised to its R3MV share. Row
counts are under R3MV's draw (§4 table).

1. **F1: drop the out-of-envelope stretch rows.**
   - Rows: ph15 rows whose label peak lateral acceleration is above 3 m/s^2 or whose fed v0 is above 30 m/s. That is 23 693 drawn rows,
     2.74 rows a batch, 34 % of ph15's drawn mass, on 313 scenes.
   - Split: F1a (lateral acceleration) 15 098 rows / 1.75 rows a batch; F1b (speed) 9 878 / 1.14.
   - It targets the curve-entry speed behind the corner cuts (16 stable events on 4 scenes, 23 events in all) and the over-speed scene
     (4 of 4).
   - Expected: 2-4 of the 9-12 extra zeros a read. This is not verified. The corner cut is a chord, and the link to entry speed rests on 3 of 4
     scenes plus the row envelope.
   - Risk: ph15 turn rows may carry part of the route-turn gain (13 zeros fixed). The 96 % of ph15 junction-turn rows above 3 m/s^2 would go.
     Keep the T281 route-turn count and LAUNCH-37 as guards.
2. **F2: drop PAI slow-downs that have no LEAD and do not stop.**
   - Rows: PAI human rows with v0 >= 4, 0.5 <= v_end < 0.6 v0, heading below 20 deg, and no r5 LEAD flag. That is 9 735 drawn rows,
     1.41 rows a batch (ph 5.1 %, ph05 0.6 %, ph15 6.7 %), on 205 scenes.
   - Slow-downs that end at a standstill are kept, so the stop rows behind the overshoot gains stay.
   - It targets the slow-down drift (13 events on 6 scenes, 2 stable) and the read-wide 2.3 x excess of unforced slow plans.
   - Expected: 1-2 zeros a read, possibly fewer rear-hit partials.
   - The evidence is weaker than for F1, because the row shares do not single these rows out. The run is a test of the cue hypothesis.

The straight-road lateral offset (a2b415d5) has no row-level handle here: PAI straight cruise labels are symmetric (14 % vs 15 % beyond
0.8 m right / left). Neither fix addresses it.

Limits:
- Mechanism and class thresholds are mine and were not registered.
- The model plan and NTV0's plan are compared at the closest station within 3 m, which is not always the same time.
- The road read-outs use the extract's road polygons (`fig`) for events and the round 5 rasters for rows. Labels that run past the raster
  (x > 52 m: most rows above about 13 m/s) carry no margin read-out.
- The LEAD flag covers vehicles in path only.
- Each event is one read per seed, 2 seeds per arm.
