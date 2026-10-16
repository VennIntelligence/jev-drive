# ACC-GAP stage 1: $DATA_DIR/runs/route_follow/acc_gap/read/s1 vs $DATA_DIR/runs/route_follow/pai_g10/acc

37 scenes scored by both (exam 9)

| label | mean | zeros | at-fault collision | offroad | corridor | rear-ended | spins |
|:--|--:|--:|--:|--:|--:|--:|--:|
| gap | 0.8858 | 4 | 1 | 2 | 1 | 2 | 0 |
| acc | 0.8717 | 3 | 1 | 2 | 0 | 9 | 0 |
| accre | 0.8717 | 3 | 1 | 2 | 0 | 9 | 0 |

gap - acc: +0.0142 [-0.0741, +0.0997] (scene bootstrap 10 000, seed 0); exam +0.0255

| line | holds |
|:--|:--|
| rear-ended down | yes |
| at-fault collisions <= ACC + 1 | yes |
| zeros <= ACC + 1 | yes |

new zeros (2): 0feb3afa corridor, fce18b62 collision
fixed zeros (1): 382a7247 collision
rear-ended only in gap: ; only in acc: 82f505c9, a309e228, cbb22323, df99cdc4, e682c5d7, fb138448, fce18b62

verdict: PASS
