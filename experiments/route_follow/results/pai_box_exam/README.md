# PAI reads on the GPU box (pai_box.sh): turn26 and exam147

Gains 20, JEV_VCONT=1.0 JEV_LEAD=1, pool jobs pinned to cards 1-5 (`GPUS=1,2,3,4,5`). Rows: tok* = Tokyo stored reads (r281/rsg; exam/rsgboth-l18-s0_turn26), box* = this box.

## turn26 (15 of the 26 scenes; the other 11 were not on the box yet)
Full table: [turn26_read.md](turn26_read.md), per-scene scores: [turn26.json](turn26.json).

| Row (15 scenes) | Mean | Zeros |
|---|--:|--:|
| Tokyo RSG | 0.2065 | 11 |
| box RSG | 0.1881 | 12 |
| Tokyo RSGboth | 0.2000 | 12 |
| box RSGboth | 0.2000 | 12 |
| box RSGPAIT | 0.2734 | 8 |

Box vs Tokyo: RSGboth identical on all 15 scenes (diff 0.0000); RSG -0.0185 [-0.057, +0.002], one scene differs (0.28), the cm-level numerics flip of decision 215.
box RSGPAIT - box RSG: +0.085 [-0.12, +0.31], 4 better / 2 worse. Turn completion (rf_pai_follow.py) needs its Tokyo-container extract step and was not run.
