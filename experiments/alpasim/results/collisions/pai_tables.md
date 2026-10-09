# COL1 PAI tables (40 rollouts of P2H10-F-s0, one plan per call, Harmonizer off)

Mean scene score 0.1694; zeros 30: collision_at_fault 12, offroad 9, left_corridor_laterally 9; replay reproduces the run's plan (4 s end point within 0.05 m) in 100.0% of decisions (max 0.000 m).

## At-fault collisions, one row per rollout

| scene | class | contact | t s | driven m | ego m/s | object m/s | ego off the logged path m | ego ahead of the log m | first visible: s before / gap m / TTC s | plan 4 s arc / (4 v): min, last 1 s | plan / log 4 s | last 4 s: served plan reaches the object / shipped plan reaches it / shipped arc over served | in the ego's own corridor ahead, s of the last 8 / shipped lead head on it then | log's own min distance m | clear on the logged path | clear at the log's position | alpamayo1 | start m/s |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|
| 071d15c4 | slow lead | front | 7.7 | 16 | 8.0 | 6.0 | -0.48 | +13.3 | 7.7 / 4.4 / inf | 0.56, 1.39 | 2.10 | 95% / 80% / 0.68 | 7.6 / 79% | 4.36 | no | no | 1.00 | 0.0 |
| 13fb89b9 | adjacent-lane vehicle | front | 6.0 | 68 | 13.9 | 10.6 | +2.24 | +13.4 | 4.5 / 0.7 / inf | 0.87, 0.90 | 1.31 | 68% / 48% / 0.84 | 0.0 / 0% | 2.43 | yes | yes | 1.00 | 8.6 |
| 9ea70552 | lead stopped | front | 11.8 | 155 | 16.0 | 0.0 | +0.46 | +36.2 | 11.6 / 59.2 / inf | 0.77, 0.90 | 1.42 | 72% / 50% / 0.74 | 6.6 / 74% | 3.13 | yes | yes | 0.00 | 10.7 |
| a28b6685 | parked / static | front | 3.3 | 3 | 3.1 | 0.1 | -0.22 | +1.3 | 3.3 / 2.0 / 9.7 | 1.57, 2.66 | 1.39 | 94% / 76% / 0.53 | 2.8 / 7% | 0.36 | yes | yes | 1.00 | 0.3 |
| 24a50fcc | adjacent-lane vehicle | front | 9.2 | 120 | 12.5 | 9.1 | +2.21 | +38.7 | 9.1 / 24.0 / inf | 0.85, 0.89 | 1.79 | 72% / 8% / 0.78 | 8.0 / 88% | 23.98 | yes | yes | 0.00 | 9.4 |
| 605bf77a | parked / static | front | 12.0 | 141 | 13.6 | 0.2 | -3.59 | +33.2 | 4.5 / 59.1 / 4.3 | 0.64, 0.86 | 1.29 | 80% / 50% / 0.74 | 7.4 / 41% | 1.28 | yes | yes | 0.00 | 5.1 |
| 0ee89cab | adjacent-lane vehicle | front | 8.7 | 40 | 13.4 | 0.1 | +2.99 | +30.3 | 7.4 / 59.9 / 8.7 | 0.95, 1.04 | 2.40 | 80% / 65% / 0.73 | 2.5 / 44% | 0.92 | no | no | 1.00 | 0.0 |
| 8a365ac2 | crossing | lateral | 5.2 | 9 | 5.4 | 10.1 | -0.68 | +8.6 | 0.0 / -4.2 / inf | 0.86, 1.10 | 6.78 | 100% / 65% / 0.57 | 0.0 / 0% | 5.85 | no | yes | 1.00 | 0.0 |
| 8ecf02e3 | adjacent-lane vehicle | front | 8.6 | 112 | 5.0 | 1.6 | -2.42 | +8.0 | 8.4 / 44.6 / 68.4 | 0.28, 0.30 | 1.15 | 45% / 18% / 0.82 | 5.9 / 92% | 3.83 | yes | yes | 1.00 | 14.6 |
| 0e3771c0 | lead stopped | front | 6.5 | 6 | 2.5 | 0.0 | +0.01 | +3.1 | 6.5 / 5.6 / 6.1 | 0.53, 1.49 | 1.72 | 48% / 20% / 0.50 | 6.5 / 89% | 2.88 | no | no | 1.00 | 1.0 |
| 8f3902f9 | other (person) | front | 14.0 | 50 | 1.3 | 1.6 | -1.65 | -0.2 | 11.6 / 59.4 / 10.6 | 0.46, 0.72 | 0.64 | 55% / 38% / 0.87 | 5.0 / 0% | 2.26 | yes | yes | 1.00 | 3.9 |
| c6c01d25 | parked / static | front | 5.0 | 22 | 3.9 | 0.2 | -2.16 | -1.2 | 4.9 / 14.5 / 6.5 | 0.20, 0.44 | 0.96 | 90% / 78% / 0.85 | 1.8 / 0% | 2.57 | yes | yes | 1.00 | 2.2 |

## Class counts

| class | n | share | ego off the logged path > 1 m | ego ahead of the log > 2 m | clear at the log's position | alpamayo1 mean |
|---|--:|--:|--:|--:|--:|--:|
| adjacent-lane vehicle | 4 | 33% | 4 | 4 | 3 | 0.75 |
| crossing | 1 | 8% | 0 | 1 | 1 | 1.00 |
| lead stopped | 2 | 17% | 0 | 2 | 1 | 0.50 |
| other (person) | 1 | 8% | 1 | 0 | 1 | 1.00 |
| parked / static | 3 | 25% | 2 | 1 | 3 | 0.67 |
| slow lead | 1 | 8% | 0 | 1 | 0 | 1.00 |

## Lead outputs (pre-registered)

C = 12 at-fault-collision rollouts, C_A = 3 (class A), N = 21 control rollouts (2192 decisions).

### Quantity 1: seen in time, per class-A rollout

| scene | class | P0 seen in time | P0 lead time s | P0 hit decisions in the last 8 s | FT seen in time | FT lead time s | FT hits |
|---|---|---|--:|--:|---|--:|--:|
| 071d15c4 | slow lead | yes | 7.7 | 60 | yes | 7.7 | 66 |
| 9ea70552 | lead stopped | yes | 6.2 | 55 | yes | 6.2 | 54 |
| 0e3771c0 | lead stopped | yes | 6.5 | 58 | yes | 6.5 | 47 |

S_time (P0) = 1.00 [1.00, 1.00] (n = 3)

S_time (FT) = 1.00 [1.00, 1.00] (n = 3)

### Lead head against the label by gap (all decisions with a labelled object ahead: collision rollouts up to impact, controls)

| gap m | decisions | P0: p >= 0.5 | P0: median d - g m | P0: |d - g| <= max(2, 0.3 g) | FT: p >= 0.5 | FT: median d - g m | FT: within |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0-5 | 403 | 71% | +3.4 | 36% | 68% | +3.8 | 31% |
| 5-10 | 83 | 63% | +0.5 | 61% | 60% | +1.4 | 48% |
| 10-20 | 216 | 60% | -0.4 | 64% | 60% | +0.2 | 64% |
| 20-40 | 295 | 62% | -1.0 | 60% | 62% | -0.3 | 69% |
| 40-80 | 468 | 38% | -15.1 | 49% | 36% | -14.7 | 48% |
| no object in the corridor | 1626 | 17% | | | 16% | | |

### Quantity 2: guard ceiling G(p*, a*)

| source | p* | a* | turned / C | turned among class A | post hoc: turned with the lead head on the struck object | too late (trigger, no stop possible) | no trigger | false-alarm decisions | false-alarm scenes (>= 1.0 s) / N | supported triggers on controls (scenes) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P0 | 0.3 | 1.0 | 8 / 12 | 2 / 3 | 6 / 12 | 1 | 3 | 51 / 2192 (2.3%) | 0 / 21 | 6 |
| P0 | 0.3 | 1.5 | 7 / 12 | 2 / 3 | 6 / 12 | 1 | 4 | 24 / 2192 (1.1%) | 0 / 21 | 6 |
| P0 | 0.3 | 2.5 | 7 / 12 | 2 / 3 | 6 / 12 | 1 | 4 | 13 / 2192 (0.6%) | 0 / 21 | 2 |
| P0 | 0.5 | 1.0 | 8 / 12 | 2 / 3 | 6 / 12 | 1 | 3 | 36 / 2192 (1.6%) | 0 / 21 | 3 |
| P0 **(primary)** | 0.5 | 1.5 | 7 / 12 [0.33, 0.83] | 2 / 3 | 6 / 12 | 1 | 4 | 15 / 2192 (0.7%) | 0 / 21 [0.00, 0.00] | 3 |
| P0 | 0.5 | 2.5 | 7 / 12 | 2 / 3 | 6 / 12 | 1 | 4 | 6 / 2192 (0.3%) | 0 / 21 | 1 |
| P0 | 0.7 | 1.0 | 8 / 12 | 2 / 3 | 6 / 12 | 1 | 3 | 29 / 2192 (1.3%) | 0 / 21 | 3 |
| P0 | 0.7 | 1.5 | 7 / 12 | 2 / 3 | 6 / 12 | 1 | 4 | 13 / 2192 (0.6%) | 0 / 21 | 3 |
| P0 | 0.7 | 2.5 | 7 / 12 | 2 / 3 | 6 / 12 | 1 | 4 | 4 / 2192 (0.2%) | 0 / 21 | 1 |
| P0 | 0.9 | 1.0 | 6 / 12 | 2 / 3 | 4 / 12 | 2 | 4 | 10 / 2192 (0.5%) | 0 / 21 | 3 |
| P0 | 0.9 | 1.5 | 5 / 12 | 2 / 3 | 4 / 12 | 2 | 5 | 7 / 2192 (0.3%) | 0 / 21 | 3 |
| P0 | 0.9 | 2.5 | 5 / 12 | 2 / 3 | 4 / 12 | 2 | 5 | 1 / 2192 (0.0%) | 0 / 21 | 1 |
| FT | 0.3 | 1.0 | 8 / 12 | 2 / 3 | 5 / 12 | 1 | 3 | 65 / 2192 (3.0%) | 1 / 21 | 4 |
| FT | 0.3 | 1.5 | 7 / 12 | 2 / 3 | 5 / 12 | 2 | 3 | 30 / 2192 (1.4%) | 0 / 21 | 4 |
| FT | 0.3 | 2.5 | 7 / 12 | 2 / 3 | 5 / 12 | 0 | 5 | 19 / 2192 (0.9%) | 0 / 21 | 2 |
| FT | 0.5 | 1.0 | 8 / 12 | 2 / 3 | 5 / 12 | 1 | 3 | 41 / 2192 (1.9%) | 1 / 21 | 4 |
| FT **(primary)** | 0.5 | 1.5 | 7 / 12 [0.33, 0.83] | 2 / 3 | 5 / 12 | 2 | 3 | 14 / 2192 (0.6%) | 0 / 21 [0.00, 0.00] | 4 |
| FT | 0.5 | 2.5 | 7 / 12 | 2 / 3 | 5 / 12 | 0 | 5 | 6 / 2192 (0.3%) | 0 / 21 | 2 |
| FT | 0.7 | 1.0 | 8 / 12 | 2 / 3 | 5 / 12 | 1 | 3 | 29 / 2192 (1.3%) | 0 / 21 | 3 |
| FT | 0.7 | 1.5 | 7 / 12 | 2 / 3 | 5 / 12 | 2 | 3 | 10 / 2192 (0.5%) | 0 / 21 | 3 |
| FT | 0.7 | 2.5 | 7 / 12 | 2 / 3 | 5 / 12 | 0 | 5 | 3 / 2192 (0.1%) | 0 / 21 | 1 |
| FT | 0.9 | 1.0 | 7 / 12 | 2 / 3 | 5 / 12 | 1 | 4 | 10 / 2192 (0.5%) | 0 / 21 | 3 |
| FT | 0.9 | 1.5 | 6 / 12 | 2 / 3 | 5 / 12 | 2 | 4 | 5 / 2192 (0.2%) | 0 / 21 | 3 |
| FT | 0.9 | 2.5 | 5 / 12 | 2 / 3 | 4 / 12 | 1 | 6 | 2 / 2192 (0.1%) | 0 / 21 | 1 |

P0 primary point, turned: 071d15c4 (slow lead), 9ea70552 (lead stopped), a28b6685 (parked / static), 24a50fcc (adjacent-lane vehicle), 605bf77a (parked / static), 0ee89cab (adjacent-lane vehicle), 8ecf02e3 (adjacent-lane vehicle); not turned: 13fb89b9 (adjacent-lane vehicle), 8a365ac2 (crossing), 0e3771c0 (lead stopped), 8f3902f9 (other (person)), c6c01d25 (parked / static)

FT primary point, turned: 071d15c4 (slow lead), 9ea70552 (lead stopped), a28b6685 (parked / static), 24a50fcc (adjacent-lane vehicle), 605bf77a (parked / static), 0ee89cab (adjacent-lane vehicle), 8ecf02e3 (adjacent-lane vehicle); not turned: 13fb89b9 (adjacent-lane vehicle), 8a365ac2 (crossing), 0e3771c0 (lead stopped), 8f3902f9 (other (person)), c6c01d25 (parked / static)

## Command and shipped-plan sensitivity (offline replay, same tokens)

| scene | flag | plan 4 s end, fed command vs straight: median / p95 abs lateral m | 4 s arc, fed / straight (median ratio) | 4 s arc, shipped P0 / served (median ratio) | served arc / (4 v) median | P0 arc / (4 v) median |
|---|---|--:|--:|--:|--:|--:|
| 071d15c4 | collision_at_fault | 0.08 / 0.28 | 0.97 | 0.64 | 1.43 | 0.57 |
| 13fb89b9 | collision_at_fault | 0.00 / 0.06 | 1.00 | 0.83 | 1.03 | 0.88 |
| 213dfdac | left_corridor_laterally | 0.21 / 2.73 | 1.03 | 0.78 | 1.04 | 0.75 |
| 49597f01 |  | 0.02 / 0.03 | 1.01 | 1.01 | 0.98 | 1.00 |
| 4f779a92 | offroad | 0.04 / 0.19 | 1.00 | 1.00 | 0.82 | 0.86 |
| 5d794411 |  | 0.01 / 0.08 | 1.00 | 0.93 | 0.97 | 0.91 |
| 7a824ffa | left_corridor_laterally | 0.11 / 0.45 | 1.00 | 0.85 | 1.12 | 0.94 |
| 94877a4a | offroad | 0.15 / 1.74 | 1.00 | 0.93 | 0.90 | 0.80 |
| 9ea70552 | collision_at_fault | 0.04 / 0.47 | 1.00 | 0.95 | 0.94 | 0.90 |
| a28b6685 | collision_at_fault | 0.11 / 0.51 | 0.98 | 0.57 | 2.55 | 1.41 |
| 1d6e30bc | left_corridor_laterally | 0.39 / 0.79 | 1.00 | 0.84 | 1.52 | 1.15 |
| 24a50fcc | collision_at_fault | 0.00 / 0.37 | 1.00 | 0.90 | 0.93 | 0.85 |
| 2e76251a |  | 0.00 / 0.04 | 1.00 | 0.87 | 1.18 | 0.97 |
| 3a48e906 | left_corridor_laterally | 0.00 / 0.08 | 1.00 | 0.93 | 1.01 | 0.94 |
| 4726de45 |  | 0.00 / 1.44 | 1.00 | 0.94 | 1.37 | 1.12 |
| 59e085d7 | offroad | 0.43 / 3.76 | 1.01 | 0.86 | 0.82 | 0.69 |
| 605bf77a | collision_at_fault | 0.00 / 0.95 | 1.00 | 0.88 | 0.90 | 0.82 |
| a45776a3 | offroad | 0.00 / 0.80 | 1.00 | 1.11 | 0.71 | 0.74 |
| b45734f4 | offroad | 0.02 / 0.23 | 1.00 | 0.99 | 0.86 | 0.86 |
| b988494a |  | 0.00 / 0.07 | 1.00 | 0.99 | 1.00 | 0.98 |
| 05f35348 | left_corridor_laterally | 0.21 / 0.82 | 1.00 | 1.03 | 0.63 | 0.68 |
| 0ee89cab | collision_at_fault | 0.00 / 0.73 | 1.00 | 0.69 | 1.64 | 0.91 |
| 1fcacc17 | offroad | 0.23 / 1.26 | 1.00 | 1.02 | 0.89 | 0.97 |
| 21626256 | left_corridor_laterally | 0.00 / 1.45 | 1.00 | 0.93 | 1.00 | 0.92 |
| 75c23f07 | left_corridor_laterally | 0.01 / 0.12 | 1.00 | 0.99 | 1.01 | 1.01 |
| 8a365ac2 | collision_at_fault | 1.48 / 2.97 | 1.04 | 0.59 | 2.35 | 1.06 |
| 8ecf02e3 | collision_at_fault | 0.04 / 0.22 | 1.01 | 0.86 | 0.68 | 0.60 |
| 92163671 |  | 0.00 / 0.46 | 1.00 | 1.41 | 0.59 | 0.80 |
| 96da4f1a | left_corridor_laterally | 0.03 / 1.44 | 1.00 | 0.90 | 1.01 | 0.91 |
| d7328fed | offroad | 0.21 / 0.87 | 1.00 | 0.99 | 1.01 | 0.99 |
| 0e3771c0 | collision_at_fault | 0.00 / 0.00 | 1.00 | 0.52 | 0.64 | 0.34 |
| 145bf9cc |  | 0.00 / 0.15 | 1.00 | 0.92 | 0.99 | 0.85 |
| 44fdaa0f | offroad | 0.08 / 0.78 | 1.00 | 0.93 | 0.97 | 0.89 |
| 5a38c811 | offroad | 0.00 / 0.21 | 1.00 | 0.79 | 1.10 | 0.85 |
| 8f3902f9 | collision_at_fault | 0.08 / 0.86 | 0.97 | 0.95 | 0.90 | 0.85 |
| c6c01d25 | collision_at_fault | 0.39 / 0.64 | 0.99 | 0.84 | 1.15 | 0.99 |
| dfb0277e | left_corridor_laterally | 0.03 / 0.44 | 1.00 | 1.01 | 0.64 | 0.66 |
| e9fe4c3b |  | 0.00 / 0.00 | 1.00 | 0.97 | 1.01 | 0.96 |
| eea2fc6d |  | 0.00 / 0.00 | 1.00 | 1.49 | 0.59 | 0.88 |
| fdfb5073 |  | 0.00 / 0.08 | 1.00 | 0.96 | 0.98 | 0.95 |

## Plan speed against the ego's speed just before the hand-over (decisions at 1.45-1.75 s, the state is the log's)

| scene | ego m/s | served: first 0.5 s, 4 s mean (m/s, % of ego) | shipped P0: first 0.5 s, 4 s mean | flag |
|---|--:|--:|--:|---|
| 071d15c4 | 0.0 | 0.0 (-91%), 1.1 (+120%) | 0.0 (-99%), 0.0 (-96%) | collision_at_fault |
| 1d6e30bc | 0.0 | 0.0 (-98%), 1.7 (+242%) | -0.0 (-100%), 0.1 (-87%) | left_corridor_laterally |
| 0ee89cab | 0.0 | 0.0 (-96%), 0.6 (+16%) | 0.0 (-100%), 0.0 (-97%) | collision_at_fault |
| 8a365ac2 | 0.0 | -0.0 (-108%), 1.6 (+216%) | 0.0 (-99%), 0.1 (-86%) | collision_at_fault |
| a28b6685 | 0.5 | 0.8 (+53%), 0.9 (+65%) | 0.5 (-11%), 0.8 (+38%) | collision_at_fault |
| 0e3771c0 | 0.7 | 0.8 (+23%), 0.4 (-45%) | 0.7 (-2%), 0.2 (-65%) | collision_at_fault |
| 4726de45 | 1.1 | 1.0 (-9%), 0.8 (-22%) | 0.9 (-19%), 0.6 (-42%) |  |
| c6c01d25 | 3.8 | 3.9 (+3%), 5.3 (+42%) | 3.1 (-17%), 4.5 (+19%) | collision_at_fault |
| 96da4f1a | 3.8 | 4.1 (+9%), 2.5 (-33%) | 3.7 (-2%), 3.4 (-9%) | left_corridor_laterally |
| 8f3902f9 | 4.0 | 3.9 (-2%), 4.1 (+3%) | 3.4 (-15%), 3.8 (-4%) | collision_at_fault |
| 5a38c811 | 5.0 | 6.0 (+20%), 6.1 (+20%) | 4.8 (-5%), 5.3 (+4%) | offroad |
| 7a824ffa | 5.7 | 7.3 (+28%), 7.3 (+27%) | 6.7 (+18%), 6.8 (+19%) | left_corridor_laterally |
| 2e76251a | 5.8 | 6.6 (+14%), 4.5 (-23%) | 6.0 (+4%), 5.5 (-4%) |  |
| 145bf9cc | 6.0 | 6.0 (+1%), 5.1 (-15%) | 5.1 (-15%), 4.1 (-31%) |  |
| 605bf77a | 7.1 | 7.8 (+9%), 8.9 (+25%) | 6.6 (-7%), 8.1 (+13%) | collision_at_fault |
| e9fe4c3b | 7.9 | 7.8 (-2%), 7.9 (-0%) | 7.3 (-7%), 7.9 (+0%) |  |
| d7328fed | 8.2 | 9.9 (+21%), 6.7 (-18%) | 9.0 (+10%), 6.1 (-26%) | offroad |
| 13fb89b9 | 8.8 | 9.9 (+13%), 9.8 (+12%) | 8.4 (-5%), 8.2 (-6%) | collision_at_fault |
| 24a50fcc | 9.4 | 10.5 (+11%), 10.1 (+7%) | 9.9 (+5%), 9.7 (+2%) | collision_at_fault |
| 3a48e906 | 10.0 | 10.9 (+9%), 10.3 (+3%) | 10.0 (+0%), 9.8 (-2%) | left_corridor_laterally |
| 9ea70552 | 10.3 | 10.9 (+6%), 10.5 (+2%) | 10.4 (+1%), 10.6 (+3%) | collision_at_fault |
| 213dfdac | 10.5 | 10.1 (-3%), 8.1 (-23%) | 9.9 (-6%), 8.2 (-22%) | left_corridor_laterally |
| 5d794411 | 10.7 | 11.3 (+5%), 11.3 (+5%) | 10.4 (-3%), 10.8 (+1%) |  |
| 75c23f07 | 10.8 | 11.4 (+6%), 11.4 (+6%) | 10.9 (+1%), 11.1 (+3%) | left_corridor_laterally |
| fdfb5073 | 10.8 | 11.5 (+6%), 11.1 (+3%) | 10.8 (-1%), 10.2 (-6%) |  |
| 44fdaa0f | 11.9 | 13.6 (+14%), 13.5 (+13%) | 12.9 (+8%), 12.7 (+7%) | offroad |
| 59e085d7 | 12.5 | 13.5 (+8%), 12.2 (-2%) | 12.4 (-1%), 11.4 (-8%) | offroad |
| 21626256 | 12.7 | 13.5 (+6%), 11.6 (-9%) | 12.2 (-4%), 9.3 (-27%) | left_corridor_laterally |
| 94877a4a | 12.8 | 11.9 (-8%), 11.7 (-9%) | 11.0 (-14%), 10.8 (-16%) | offroad |
| 8ecf02e3 | 14.2 | 14.1 (-1%), 12.9 (-9%) | 13.7 (-3%), 12.8 (-10%) | collision_at_fault |
| eea2fc6d | 14.6 | 13.5 (-8%), 12.9 (-12%) | 13.9 (-5%), 13.2 (-10%) |  |
| 92163671 | 15.2 | 13.4 (-12%), 13.6 (-10%) | 12.6 (-17%), 12.5 (-18%) |  |
| a45776a3 | 16.6 | 14.7 (-11%), 13.1 (-21%) | 14.5 (-12%), 12.9 (-22%) | offroad |
| b988494a | 19.5 | 19.0 (-3%), 19.5 (+0%) | 19.2 (-1%), 19.8 (+2%) |  |
| 49597f01 | 22.3 | 21.5 (-4%), 21.9 (-2%) | 21.9 (-2%), 22.4 (+0%) |  |
| 4f779a92 | 23.9 | 22.2 (-7%), 22.0 (-8%) | 22.3 (-7%), 22.0 (-8%) | offroad |
| 05f35348 | 29.7 | 26.5 (-11%), 26.9 (-10%) | 27.9 (-6%), 28.4 (-4%) | left_corridor_laterally |
| 1fcacc17 | 29.8 | 29.4 (-1%), 29.5 (-1%) | 29.5 (-1%), 29.6 (-1%) | offroad |
| dfb0277e | 33.0 | 31.5 (-5%), 31.5 (-5%) | 31.6 (-4%), 31.8 (-4%) | left_corridor_laterally |
| b45734f4 | 34.8 | 32.5 (-6%), 32.8 (-6%) | 32.3 (-7%), 32.5 (-7%) | offroad |

ego 0-2 m/s (n = 7): median first-0.5 s speed against the ego's, served -91%, shipped -99%; 4 s mean, served +65%, shipped -86%.

ego 2-5 m/s (n = 3): median first-0.5 s speed against the ego's, served +3%, shipped -15%; 4 s mean, served +3%, shipped -4%.

ego 5-13 m/s (n = 19): median first-0.5 s speed against the ego's, served +8%, shipped -1%; 4 s mean, served +3%, shipped -2%.

ego 13-23 m/s (n = 6): median first-0.5 s speed against the ego's, served -6%, shipped -4%; 4 s mean, served -10%, shipped -10%.

ego 23-99 m/s (n = 5): median first-0.5 s speed against the ego's, served -6%, shipped -6%; 4 s mean, served -6%, shipped -4%.
