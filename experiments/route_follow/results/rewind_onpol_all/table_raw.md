# Recovery-expert rewind on every onpol-train turn: RSG-L18-s0 (box, held-read serving, 30 decisions), expert XP-L12-E46

Scenes with a strict turn in the identity's 15 s window: 166 (60 logs); identity outcome completed 71, event 65, off 17, short 13. Failed (A) 95: off 16, off_cross 1, short_end 13, zero 65; handed over (event / off with h >= 1 for some Delta) 78; short listed, not rewound.

Gate thresholds: PAI (d293 all-turn human p95) kmax 0.184 1/m, dk 0.0425 1/m^2, alat 4.42 m/s^2, entry 5.6 m/s; nuPlan (human p95 of these 166 turns) kmax 0.151, dk 0.0406, alat 2.83, entry 6.7. Human logs passing: PAI 122 / 166, nuPlan 136 / 166.

## Expert from h = k_ref - 2 Delta (failed units)

| Delta | units (not rewindable) | valid handovers | within tolerance | completed (log CI) | completed / event / off / short | any zero after h | at-fault after h | rear after h | creep v(h+3 s) >= 2 | completed + gate PAI | completed + gate nuPlan |
|--:|--:|--:|--:|:--|:--|--:|--:|--:|:--|--:|--:|
| 2 s | 78 (4) | 78 | 69 | 0.65 [0.55, 0.76] | 51 / 20 / 0 / 7 | 25 | 20 | 21 | 68 / 77 | 24 | 16 |
| 3 s | 77 (5) | 77 | 73 | 0.68 [0.57, 0.79] | 52 / 18 / 0 / 7 | 23 | 18 | 19 | 69 / 77 | 28 | 25 |

## Label rows (2 Hz, k >= h, rf_follow junction stretch, zero-free completed continuations)

| Delta | category | units | rows | unique turns | logs | top-3 turns' share |
|--:|:--|--:|--:|--:|--:|--:|
| 2 s | completed + gate (PAI) | 22 | 234 | 22 | 21 | 0.24 |
| 2 s | completed + gate (nuPlan) | 16 | 192 | 16 | 15 | 0.29 |
| 2 s | completed, zero-free, no gate | 46 | 483 | 46 | 34 | 0.11 |
| 3 s | completed + gate (PAI) | 24 | 308 | 24 | 22 | 0.20 |
| 3 s | completed + gate (nuPlan) | 23 | 301 | 23 | 21 | 0.20 |
| 3 s | completed, zero-free, no gate | 47 | 581 | 47 | 35 | 0.11 |
| both | completed + gate (PAI) | - | 542 | 29 | 25 | 0.21 |
| both | completed + gate (nuPlan) | - | 493 | 26 | 23 | 0.23 |
| both | completed, zero-free, no gate | - | 1064 | 49 | 36 | 0.11 |

d316's 64 train turns (stored 18-decision units), those in this read: event / off units 32; also event / off here 30; turn here but not failed 1; no strict turn here 1.

Completed but excluded by a zero after h (zero kind, after the last junction row or inside it): 2 s: collision after exit 1, corridor after exit 4; 3 s: collision after exit 1, corridor after exit 4
