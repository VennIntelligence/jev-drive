# ACC-GAP stage 2: $DATA_DIR/runs/route_follow/acc_gap/read/s2 vs $DATA_DIR/runs/route_follow/pai_g10/acc

280 scenes scored by both (exam 71)

| label | mean | zeros | at-fault collision | offroad | corridor | rear-ended | spins |
|:--|--:|--:|--:|--:|--:|--:|--:|
| gap | 0.6934 | 75 | 9 | 14 | 52 | 28 | 0 |
| acc | 0.6760 | 75 | 9 | 16 | 50 | 40 | 0 |

gap - acc: +0.0173 [-0.0082, +0.0436] (scene bootstrap 10 000, seed 0); exam +0.0047

| line | holds |
|:--|:--|
| mean CI lo > -0.01 and point > 0 | yes |
| exam point >= -0.01 | yes |
| zeros <= ACC + 2 | yes |
| spins <= ACC + 1 | yes |

new zeros (7): 098e2482 corridor, 0feb3afa corridor, 2bbf1efc corridor, a4d2f7cf corridor, dbfd6c72 collision, f8ee31cb collision, fce18b62 collision
fixed zeros (7): 268949d2 corridor, 37d660b1 corridor, 382a7247 collision, 4e4a2a51 offroad, 5ddb96bd collision, b44a4a44 collision, ea468790 offroad
rear-ended only in gap: 268949d2, 5ddb96bd; only in acc: 098e2482, 5c4a277a, 7b5c9fd6, 7c6529f6, 82f505c9, a309e228, ad69dd31, ca7accbc, cbb22323, d5f732f6, df99cdc4, e682c5d7, fb138448, fce18b62

verdict: PASS
