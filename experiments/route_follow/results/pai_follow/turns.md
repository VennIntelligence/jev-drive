# PAI turn completion with the real route (rf_pai_follow.py)

Turn = AlpaSim route (MAP, the log on its lanes + extensions) net heading change >= 60 deg within 30 m, inside the log's travel, exit + 10 m inside it; completed = the ego crosses exit + 10 m before any zero-score event, <= 2.5 m from the route. event = zero event first; off = crossed > 2.5 m off / last seen > 2.5 m off; short = still on the route at the end (rear = truncated by a rear contact). Rates with scene-bootstrap 95 % CI; cover = logged memory +20 m valid at the decision 20 m before the entry.

| label | rollouts / turn scenes (L / R) | completion all | left | right | covered | uncovered | completed / event / off / short slow / short rear | event kinds | ego / log speed at entry (median ratio) | in the 48 list |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| rsg@bend | 281 / 46 (18 / 28) | 19.6 % (46) [9, 30] | 33.3 % (18) [11, 56] | 10.7 % (28) [0, 25] | 3.8 % (26) [0, 12] | 40.0 % (20) [20, 60] | 9 / 20 / 10 / 4 / 3 | corridor 13, offroad 4, collision 3 | 1.05 | 27 of 48 |
| rsg@open | 281 / 51 (21 / 30) | 13.7 % (51) [6, 24] | 23.8 % (21) [5, 43] | 6.7 % (30) [0, 17] | 8.3 % (36) [0, 19] | 26.7 % (15) [7, 47] | 7 / 32 / 6 / 4 / 2 | corridor 24, collision 4, offroad 4 | 1.14 | 39 of 48 |
| rsg | 281 / 36 (15 / 21) | 13.9 % (36) [3, 25] | 26.7 % (15) [7, 53] | 4.8 % (21) [0, 14] | 4.5 % (22) [0, 14] | 28.6 % (14) [7, 50] | 5 / 21 / 6 / 2 / 2 | corridor 15, collision 3, offroad 3 | 1.10 | 28 of 48 |
| rbsg2@bend | 281 / 46 (18 / 28) | 23.9 % (46) [13, 37] | 22.2 % (18) [6, 44] | 25.0 % (28) [11, 43] | 0.0 % (26) [0, 0] | 55.0 % (20) [35, 75] | 11 / 22 / 4 / 6 / 3 | corridor 13, offroad 6, collision 3 | 1.01 | 27 of 48 |
| rbsg2@open | 281 / 51 (21 / 30) | 21.6 % (51) [12, 33] | 19.0 % (21) [5, 38] | 23.3 % (30) [10, 40] | 8.3 % (36) [0, 19] | 53.3 % (15) [27, 80] | 11 / 29 / 3 / 7 / 1 | corridor 17, collision 6, offroad 6 | 1.20 | 39 of 48 |
| rbsg2 | 281 / 36 (15 / 21) | 25.0 % (36) [11, 39] | 20.0 % (15) [0, 40] | 28.6 % (21) [10, 48] | 4.5 % (22) [0, 14] | 57.1 % (14) [29, 86] | 9 / 21 / 1 / 4 / 1 | corridor 12, offroad 5, collision 4 | 1.05 | 28 of 48 |

Per-decision junction lateral, pair rbsg2:rsg (|lat| median A / B, A - B [scene-clustered 95 % CI]); discordant 31 of 2197 left out:

| half | decisions (scenes) | median A / B (m) | A - B |
|:--|--:|:--|:--|
| covered | 1308 (31) | 0.53 / 0.63 | -0.102 [-0.263, +0.032] |
| uncovered | 858 (15) | 0.43 / 0.44 | -0.007 [-0.185, +0.054] |

Per scene (turn scenes, both labels):

| scene | side / deg | in 48 | rsg outcome (score) | rbsg2 outcome (score) |
|:--|:--|:--|--|--|
| 09528162 | R -100 | y | off (0.28) | event (0.00) |
| 132ae6cb | L +86 |  | completed (0.04) | completed (0.05) |
| 1e8e939e | R -49 |  | event (0.00) | event (0.00) |
| 22fa5c92 | L +91 | y | event (0.00) | event (0.00) |
| 268949d2 | R -88 | y | off (1.00) | completed (0.91) |
| 2f227f2c | R -66 | y | event (0.00) | event (0.00) |
| 302c5c99 | L +163 |  | event (0.00) | event (0.00) |
| 3232fdb7 | R -87 | y | event (0.00) | short_slow (0.94) |
| 33c8976b | L +107 | y | event (0.00) | event (0.00) |
| 37d660b1 | R -85 | y | event (0.00) | event (0.00) |
| 382a7247 | L +83 | y | completed (0.00) | event (0.00) |
| 3c2822d9 | L +81 |  | completed (0.88) | completed (0.91) |
| 3ce13eed | R -100 | y | event (0.00) | event (0.00) |
| 403d558a | R -93 | y | event (0.00) | event (0.00) |
| 68e53160 | L +108 | y | event (0.00) | event (0.00) |
| 6fd8cf30 | R -88 |  | short_rear (0.30) | short_rear (0.32) |
| 8a143cb2 | R -76 |  | completed (1.00) | completed (1.00) |
| 94c1d722 | R -86 | y | event (0.00) | completed (0.54) |
| 9f393c8a | R -84 | y | event (0.00) | event (0.00) |
| a174faed | R -90 | y | event (0.00) | event (0.00) |
| a1b166cc | L +91 | y | event (0.00) | completed (1.00) |
| a4d2f7cf | L +110 | y | short_slow (0.98) | short_slow (1.00) |
| abd45a30 | L +90 | y | short_rear (0.81) | short_slow (0.90) |
| b40f4729 | R -89 | y | off (1.00) | completed (1.00) |
| c4a58f83 | R -85 |  | short_slow (0.68) | short_slow (0.82) |
| ce161092 | L +85 | y | event (0.00) | event (0.00) |
| d90fe123 | L +85 | y | completed (1.00) | event (0.00) |
| ddc3e8df | R -66 | y | event (0.00) | completed (0.68) |
| e0f6ebe1 | R -90 | y | off (1.00) | event (0.00) |
| e37aa159 | L +82 | y | event (0.00) | off (1.00) |
| e5e47fd2 | L +95 |  | event (0.00) | event (0.00) |
| e848c843 | R -90 | y | off (0.79) | event (0.00) |
| ea468790 | R -83 | y | off (0.55) | completed (0.68) |
| eaba3f6a | R -90 | y | event (0.00) | event (0.00) |
| f9e361aa | R -81 | y | event (0.00) | event (0.00) |
| fd01e9ab | L +98 | y | event (0.00) | event (0.00) |
