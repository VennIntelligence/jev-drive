# 补记 1 (exploratory): RTB05O-L18-s0 closed loop, one seed

| vs | zeros (<= C + 3) | mean | paired [session CI] | progress_rel | offroad | at-fault | turns /24 (>= C - 2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0 | 41 (C 34) | 0.858 (C 0.866) | -0.007 [-0.029, 0.015] | 0.796 (C 0.788) | 18 (C 17) | 8 (C 4) | 18 (C 16) |
| WTC-L18-s0 | 41 (C 34) | 0.858 (C 0.866) | -0.007 [-0.029, 0.015] | 0.796 (C 0.788) | 18 (C 17) | 8 (C 4) | 18 (C 16) |
| RSG-L18-s1 | 41 (C 34) | 0.858 (C 0.868) | -0.009 [-0.036, 0.008] | 0.796 (C 0.788) | 18 (C 13) | 8 (C 6) | 18 (C 14) |
| WTC-L18-s1 | 41 (C 34) | 0.858 (C 0.868) | -0.009 [-0.036, 0.008] | 0.796 (C 0.788) | 18 (C 13) | 8 (C 6) | 18 (C 14) |

| tag | static: distance (m) | static: contact | static: min gap (m) | stop17 zeros | arrival 4 s arc @k=1 (m) | dev ADE |
|---|--:|:-:|--:|--:|--:|--:|
| RTB05O-L18-s0 | 0.54 | no | 1.49 | 16 / 17 | 1.20 | 0.5434 |
| P0 | 0.48 | no | 1.55 | 14 / 17 | 0.43 | - |
| RSG-L18-s0 | 2.61 | yes | 0.00 | 16 / 17 | 2.90 | 0.5284 |
| WTCP-L18-s0 | 2.40 | yes | 0.00 | 16 / 17 | 3.06 | 0.5290 |

arrival-stop seen in closed loop (static no contact and <= 1.0 m, or stop17 zeros <= 15): True
dev ADE: RTB05O-L18-s0 0.5434, RTB1-F-s0 0.5556, RB0-F-s0 0.5276, RSG-L18-s0 0.5284, WTCP-L18-s0 0.5290
