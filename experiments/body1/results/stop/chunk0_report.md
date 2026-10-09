Scenes common to all runs: 234 from 27 logs; logged 4 s turn > 45 deg: 24 (token not in lb_navtest: 0).

## Drivers

| driver | n | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0 | 234 | 0.9518 [0.9165, 0.9773] | 184 | 7 | 4 | 2 | 1 | 43 | 0.937 |
| stop2-s0 | 234 | 0.9451 [0.9135, 0.9691] | 179 | 5 | 2 | 2 | 1 | 50 | 0.920 |

## Recipes (per-scene mean of the two seeds; counts are seed means)

| recipe | seeds | mean [95 % CI, logs] | zeros | at-fault collision | offroad | left corridor | slow |
|:--|:--|:--|--:|--:|--:|--:|--:|
| base | 0.9518 | 0.9518 [0.9165, 0.9773] | 7 | 4 | 2 | 1 | 43 |
| stop2 | 0.9451 | 0.9451 [0.9135, 0.9691] | 5 | 2 | 2 | 1 | 50 |

## Lines (Amendment 2 item 8; the registered configuration is the first arm, any other arm is a labelled secondary)

| arm | L1 at-fault collision zeros, seed mean (per seed) arm vs base | L1 | L2 difference [95 % CI by log] | by scene | L2 | L3 slow arm vs 1.1 x base | L3 | all three |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| stop2 | 2 (2) vs 4 (4) | met | -0.0067 [-0.0159, +0.0011] | [-0.0191, +0.0059] | not met | 50 vs 47.3 (base 43) | not met | **no** |

Per seed (arm seed i against baseline seed i, paired by scene, CI by log): stop2 s0 -0.0067 [-0.0159, +0.0011].

## Zero / non-zero changes per seed

| arm | seed | base zeros removed (collision / offroad / corridor) | new zeros (collision / offroad / corridor) | new zeros in scenes with a flag | base at-fault collisions: passed / still collision / other zero | of them with >= 1 flag |
|:--|:--|:--|:--|--:|:--|:--|
| stop2 | s0 | 2 (2 / 0 / 0) | 0 (0 / 0 / 0) | 0 | 2 / 2 / 0 | 2 / 2 / 0 |

## Flags and what a flag costs

| driver | decisions | flagged (%) | scenes with a flag (%) | flags that remove speed | at the 6 m/s^2 cap | ego under 0.5 m/s | median stop point (m) / deceleration / m removed in 2 s | flags by decision index 0..9 |
|:--|--:|:--|:--|--:|--:|--:|:--|:--|
| stop2-s0 | 2340 | 86 (3.68) | 36 (15.4) | 85 | 11 | 8 | 5.6 / 1.00 / 1.08 | [8, 9, 6, 4, 8, 9, 12, 14, 9, 7] |

| arm, seed | scenes with a flag: n, mean score arm / base, difference | scenes without a flag: n, mean arm / base, difference | flagged scenes 1.0 -> slow | unflagged 1.0 -> slow | slow -> 1.0 (all) | mean progress change in flagged scenes |
|:--|:--|:--|--:|--:|--:|--:|
| stop2 s0 | 36, 0.7911 / 0.8349, -0.0439 | 198, 0.9731 / 0.9731, +0.0000 | 5 | 0 | 0 | -0.1094 |

## Driver latency per `drive` call (ms, whole call inside the driver; target 100)

| driver | p50 | p90 | p99 | max | hook alone p50 / p90 |
|:--|--:|--:|--:|--:|:--|
| P2H10-F-s0 | 18.8 | 32.1 | 46.8 | 81 | - |
| stop2-s0 | 29.4 | 47.8 | 69.9 | 126 | 9.4 / 20.3 |

## Scenes whose logged 4 s future turns > 45 deg (24 scenes, 13 logs)

| recipe | mean | zeros (seed mean) | collision / offroad / corridor | flagged scenes (seed mean) | difference to base [95 % CI, logs] |
|:--|--:|--:|:--|--:|:--|
| base | 0.7898 | 4 | 2 / 1 / 1 | 0 | - |
| stop2 | 0.8186 | 3 | 1 / 1 / 1 | 4 | +0.0288 [-0.0000, +0.0901] |
