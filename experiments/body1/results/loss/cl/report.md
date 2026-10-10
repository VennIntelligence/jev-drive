Scenes common to all runs: 700 from 27 logs; logged 4 s turn > 45 deg: 61 (token not in lb_navtest: 0). (seed, scene) pairs not in a development list: 933 of 1400.

## Drivers

| driver | n | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0 | 700 | 0.9468 [0.9252, 0.9652] | 562 | 26 | 8 | 10 | 8 | 112 | 0.942 |
| P2H10-F-s1 | 700 | 0.9495 [0.9300, 0.9669] | 572 | 23 | 6 | 10 | 7 | 105 | 0.936 |
| P2H10B-F-s0 | 700 | 0.9441 [0.9238, 0.9617] | 541 | 22 | 5 | 10 | 7 | 137 | 0.928 |
| P2H10B-F-s1 | 700 | 0.9430 [0.9234, 0.9599] | 538 | 22 | 4 | 11 | 7 | 140 | 0.921 |

## Lines (the stricter of the two readings decides)

| reading | pairs | L1a collision + offroad + corridor zeros, arm vs base (collision / offroad / corridor); per seed | L1a | L1b collision + offroad, per seed | L1b | L2 mean difference [95 % CI by log] | L2 | L3 slow arm vs 1.1 x base | L3 | all |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A: all | 1400 | 44 (9 / 21 / 14) vs 49 (14 / 20 / 15); s0 22 vs 26, s1 22 vs 23 | met | 30 vs 34; s0 15 vs 18, s1 15 vs 16 | met | -0.0046 [-0.0131, +0.0052] (by scene [-0.0153, +0.0064]) | not met | 277 vs 238.7 (base 217) | not met | **no** |
| B: never-switched pairs | 933 | 29 (6 / 13 / 10) vs 32 (8 / 13 / 11); s0 7 vs 9, s1 22 vs 23 | met | 19 vs 21; s0 4 vs 5, s1 15 vs 16 | met | -0.0058 [-0.0160, +0.0050] (by scene [-0.0169, +0.0054]) | not met | 182 vs 150.7 (base 137) | not met | **no** |

Counts are sums over the (seed, scene) pairs of the reading. Verdict: **not met**. Per seed (paired by scene, CI by log): s0 -0.0026 [-0.0117, +0.0071]; s1 -0.0065 [-0.0164, +0.0044].

## Zero changes per seed (all scenes)

| seed | removed (collision / offroad / corridor) | new (collision / offroad / corridor) | base zero, still zero | new, mean score in scenes both ran | slow base -> arm |
|:--|:--|:--|--:|--:|:--|
| s0 | 11 (5 / 2 / 4) | 7 (2 / 2 / 3) | 15 | 0.9441 vs 0.9468 | 112 -> 137 |
| s1 | 10 (3 / 3 / 4) | 9 (1 / 4 / 4) | 13 | 0.9430 vs 0.9495 | 105 -> 140 |

Every flip (seed, scene tail, base -> arm, zero class, turn deg):

- s0 `78b4153a6d3e5b33` new: 1.0 (-) -> 0.0 (corridor); turn -48.1; dev
- s0 `00ff8eeb53cc598b` removed: 0.0 (offroad) -> 0.6738 (-); turn -44.9; fresh
- s0 `6e3c7a34388e5ae3` new: 1.0 (-) -> 0.0 (collision); turn -28.1; dev
- s0 `77155a60b2ae5e75` new: 0.8921 (-) -> 0.0 (corridor); turn -67.0; dev
- s0 `abe4fa26de85552d` new: 1.0 (-) -> 0.0 (collision); turn 3.7; fresh
- s0 `6fc4fc2702305dfa` removed: 0.0 (collision) -> 0.9789 (-); turn 28.3; fresh
- s0 `fcb45b2aa29356d9` removed: 0.0 (offroad) -> 1.0 (-); turn 40.9; dev
- s0 `1347c91c511a5918` removed: 0.0 (corridor) -> 1.0 (-); turn 3.7; fresh
- s0 `927b73fea33f5218` removed: 0.0 (corridor) -> 1.0 (-); turn 46.0; fresh
- s0 `a04628cdd3f25947` removed: 0.0 (corridor) -> 1.0 (-); turn 61.6; dev
- s0 `52d3f15d6ca75701` new: 1.0 (-) -> 0.0 (corridor); turn -55.0; fresh
- s0 `6d08dce7cfaa5035` new: 1.0 (-) -> 0.0 (offroad); turn 8.9; dev
- s0 `994ab680895a55d2` removed: 0.0 (collision) -> 0.7789 (-); turn -8.2; dev
- s0 `b53b172e95895a12` removed: 0.0 (corridor) -> 1.0 (-); turn 66.0; dev
- s0 `a7fa8bccce9253cb` removed: 0.0 (collision) -> 1.0 (-); turn -61.4; dev
- s0 `80f6c94fed0c5519` removed: 0.0 (collision) -> 1.0 (-); turn -1.5; dev
- s0 `52d9d533ca4e5980` new: 1.0 (-) -> 0.0 (offroad); turn 40.3; dev
- s0 `39cbed62434a528b` removed: 0.0 (collision) -> 1.0 (-); turn -4.2; dev
- s1 `78b4153a6d3e5b33` new: 1.0 (-) -> 0.0 (corridor); turn -48.1; fresh
- s1 `748779bcdb5b5a49` new: 1.0 (-) -> 0.0 (collision); turn 1.3; fresh
- s1 `5e9e8c31277d5edc` removed: 0.0 (offroad) -> 0.9569 (-); turn -49.7; fresh
- s1 `00ff8eeb53cc598b` removed: 0.0 (offroad) -> 0.8419 (-); turn -44.9; fresh
- s1 `f0a6222ab3e55174` new: 1.0 (-) -> 0.0 (corridor); turn -56.6; fresh
- s1 `77155a60b2ae5e75` new: 0.8342 (-) -> 0.0 (corridor); turn -67.0; fresh
- s1 `fcb45b2aa29356d9` removed: 0.0 (offroad) -> 1.0 (-); turn 40.9; fresh
- s1 `1347c91c511a5918` removed: 0.0 (corridor) -> 1.0 (-); turn 3.7; fresh
- s1 `927b73fea33f5218` removed: 0.0 (corridor) -> 1.0 (-); turn 46.0; fresh
- s1 `e933d70dd378598f` new: 0.9519 (-) -> 0.0 (corridor); turn -70.5; fresh
- s1 `a04628cdd3f25947` removed: 0.0 (corridor) -> 1.0 (-); turn 61.6; fresh
- s1 `6d08dce7cfaa5035` new: 1.0 (-) -> 0.0 (offroad); turn 8.9; fresh
- s1 `994ab680895a55d2` removed: 0.0 (collision) -> 0.7428 (-); turn -8.2; fresh
- s1 `87339a4d32305504` new: 1.0 (-) -> 0.0 (offroad); turn 55.9; fresh
- s1 `80f6c94fed0c5519` removed: 0.0 (collision) -> 0.8469 (-); turn -1.5; fresh
- s1 `52d9d533ca4e5980` new: 1.0 (-) -> 0.0 (offroad); turn 40.3; fresh
- s1 `39cbed62434a528b` removed: 0.0 (collision) -> 1.0 (-); turn -4.2; fresh
- s1 `fe6105aa925d5621` removed: 0.0 (corridor) -> 0.8885 (-); turn -3.7; fresh
- s1 `365c0937de9d5885` new: 1.0 (-) -> 0.0 (offroad); turn 68.1; fresh

## Scenes whose logged 4 s future turns > 45 deg (61 scenes, 20 logs)

| recipe | mean | zeros (sum of the two seeds) | collision / offroad / corridor | difference to base [95 % CI by log] |
|:--|--:|--:|:--|:--|
| base | 0.8343 | 18 | 3 / 10 / 5 | - |
| P2H10B-F | 0.8136 | 20 | 2 / 11 / 7 | -0.0207 [-0.0825, +0.0453] |
