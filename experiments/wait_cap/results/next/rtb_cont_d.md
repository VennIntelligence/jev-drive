# 补记 1 (exploratory): RTB1OB1-L18-s1 closed loop, one seed

| vs | zeros (<= C + 3) | mean | paired [session CI] | progress_rel | offroad | at-fault | turns /24 (>= C - 2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | 39 (C 34) | 0.864 (C 0.866) | -0.001 [-0.021, 0.016] | 0.802 (C 0.788) | 14 (C 17) | 11 (C 4) | 17 (C 16) |
| WTC-L18-s0 | 39 (C 34) | 0.864 (C 0.866) | -0.001 [-0.021, 0.016] | 0.802 (C 0.788) | 14 (C 17) | 11 (C 4) | 17 (C 16) |
| RSG-L18-s1 | 39 (C 34) | 0.864 (C 0.868) | -0.003 [-0.029, 0.015] | 0.802 (C 0.788) | 14 (C 13) | 11 (C 6) | 17 (C 14) |
| WTC-L18-s1 | 39 (C 34) | 0.864 (C 0.868) | -0.003 [-0.029, 0.015] | 0.802 (C 0.788) | 14 (C 13) | 11 (C 6) | 17 (C 14) |

| tag | static: distance (m) | static: contact | static: min gap (m) | stop17 zeros | arrival 4 s arc @k=1 (m) | dev ADE |
|---|--:|:-:|--:|--:|--:|--:|
| RTB1OB1-L18-s1 | 0.63 | no | 1.40 | 15 / 17 | 1.51 | 0.5447 |
| P0 | 0.48 | no | 1.55 | 14 / 17 | 0.43 | - |
| RSG-L18-s0 | 2.61 | yes | 0.00 | 16 / 17 | 2.90 | 0.5284 |
| WTCP-L18-s0 | 2.40 | yes | 0.00 | 16 / 17 | 3.06 | 0.5290 |

arrival-stop seen in closed loop (static no contact and <= 1.0 m, or stop17 zeros <= 15): True
dev ADE: RTB1OB1-L18-s1 0.5447, RTB1-F-s0 0.5556, RB0-F-s0 0.5276, RSG-L18-s0 0.5284, WTCP-L18-s0 0.5290
