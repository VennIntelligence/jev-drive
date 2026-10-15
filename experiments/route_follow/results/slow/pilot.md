# SLOW pilot gate (1000 steps; prereg 闸 + Amendment 1: judged on seed 0, seed 1 reported)

| arm | seed | dev ADE @1000 (<= 0.538) | G-rely read | dropped share / expected (0.8 .. 1.2) | open share (off / yr / pool) | PAITV imit first 100 -> last 100 steps (<= 0.8) | pass |
|:--|--:|--:|:--|:--|:--|:--|:--|
| S1 | 0 | 0.5304 | True (d_junction +0.0268, diff lo +0.0103) | 0.1164 / 0.1166 = 0.999 | 0.202 / 0.279 / 0.370 | - | **pass** |
| S1 | 1 | 0.5308 | True (d_junction +0.0272, diff lo +0.0115) | 0.1181 / 0.1181 = 1.000 | 0.205 / 0.289 / 0.367 | - | **pass** |
| Sall | 0 | 0.5460 | True (d_junction +0.0260, diff lo +0.0106) | - | - | - | **FAIL ade** |
| Sall | 1 | 0.5502 | True (d_junction +0.0237, diff lo +0.0083) | - | - | - | **FAIL ade** |
| S3 | 0 | 0.5293 | True (d_junction +0.0242, diff lo +0.0105) | - | - | 3.3739 -> 2.5592 (x 0.759) | **pass** |
| S3 | 1 | 0.5282 | True (d_junction +0.0272, diff lo +0.0143) | - | - | 3.6667 -> 2.3052 (x 0.629) | **pass** |
| S2 | 0 | 0.5326 | True (d_junction +0.0248, diff lo +0.0106) | 0.1146 / 0.1141 = 1.004 | 0.202 / 0.279 / 0.332 | 3.3319 -> 2.5771 (x 0.773) | **pass** |
| S2 | 1 | 0.5325 | True (d_junction +0.0262, diff lo +0.0123) | 0.1162 / 0.1160 = 1.001 | 0.205 / 0.289 / 0.335 | 3.7556 -> 2.3194 (x 0.618) | **pass** |

Gate (prereg): dev ADE (memory on) <= RSG-L18 0.528 + 0.01; G-rely read true; S1 / S2 dropped share within +-20 % of 0.5 x open share;
S2 (and S3) PAITV imitation loss (`loss/on/imit`, the 25-step means of the mixed `both` rows) at the end <= 0.8 x its start. Arm verdict = seed 0.
