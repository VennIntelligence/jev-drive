Scenes scored by every row: 139 (base_s0 139, base_s1 139, s0 281, s1 281, base 139, both 281).

| Row | Mean | Zeros | at-fault collision | offroad | left corridor | other zero |
|---|--:|--:|--:|--:|--:|--:|
| base_s0 | 0.3245 | 79 | 12 | 21 | 47 | 1 |
| base_s1 | 0.3121 | 82 | 13 | 25 | 46 | 0 |
| s0 | 0.4086 | 69 | 12 | 19 | 40 | 0 |
| s1 | 0.4440 | 60 | 9 | 15 | 40 | 0 |
| base | 0.3183 | 72 |  | | |  |
| both | 0.4263 | 55 |  | | |  |

Paired difference base_s1 - base_s0 on 139 scenes: -0.0124, bootstrap 95% CI [-0.0596, +0.0353]; means 0.3121 / 0.3245; base_s1 better on 13, worse on 28, within 0.01 on 98 (max |difference| 1.0000).

Paired difference s0 - base_s0 on 139 scenes: +0.0841, bootstrap 95% CI [+0.0193, +0.1503]; means 0.4086 / 0.3245; s0 better on 27, worse on 24, within 0.01 on 88 (max |difference| 1.0000).

Paired difference s1 - base_s1 on 139 scenes: +0.1319, bootstrap 95% CI [+0.0683, +0.1970]; means 0.4440 / 0.3121; s1 better on 38, worse on 16, within 0.01 on 85 (max |difference| 1.0000).

Paired difference both - base on 139 scenes: +0.1080, bootstrap 95% CI [+0.0528, +0.1646]; means 0.4263 / 0.3183; both better on 41, worse on 20, within 0.01 on 78 (max |difference| 1.0000).

Paired difference s0 - base_s1 on 139 scenes: +0.0965, bootstrap 95% CI [+0.0283, +0.1632]; means 0.4086 / 0.3121; s0 better on 40, worse on 13, within 0.01 on 86 (max |difference| 1.0000).

Paired difference s1 - base_s0 on 139 scenes: +0.1195, bootstrap 95% CI [+0.0544, +0.1873]; means 0.4440 / 0.3245; s1 better on 32, worse on 26, within 0.01 on 81 (max |difference| 1.0000).

| Run | Scenes | CONC | Card | Wall s | Scenes / h | Render share | Drive share | Driver ms / call | VRAM peaks MiB | Card peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| base_s0: full1_ab | 99 | 4 | 1 | 4646 | 77 | 0.82 | 0.08 | 50 | driver 3108, physics 2498, renderer 18150 | 23790 |
| base_s0: base_q40_c1 | 40 | 4 | 1 | 1821 | 79 | 0.80 | 0.08 | 50 | driver 3108, physics 2146, renderer 17854 | 23142 |
| base_s1: p2h10-f-s1_c000 | 35 | 4 | 0 | 1844 | 68 | 0.82 | 0.08 | 51 | driver 3108, physics 2498, renderer 18466 | 24119 |
| base_s1: p2h10-f-s1_c001 | 35 | 4 | 1 | 1814 | 69 | 0.80 | 0.08 | 51 | driver 3108, physics 2050, renderer 18528 | 24119 |
| base_s1: p2h10-f-s1_c002 | 35 | 4 | 1 | 1850 | 68 | 0.80 | 0.08 | 51 | driver 3108, physics 2242, renderer 18540 | 23901 |
| base_s1: p2h10-f-s1_c003 | 34 | 4 | 0 | 1770 | 69 | 0.79 | 0.08 | 51 | driver 3108, physics 2146, renderer 18952 | 23901 |
| s0: p2h10s-f-s0_b1a | 33 | 4 | 1 | 1647 | 72 | 0.82 | 0.07 | 49 | driver 3108, physics 2498, renderer 17932 | 23572 |
| s0: p2h10s-f-s0_b1b | 33 | 4 | 1 | 1605 | 74 | 0.82 | 0.08 | 50 | driver 3108, physics 1922, renderer 16778 | 21842 |
| s0: p2h10s-f-s0_b1c | 33 | 4 | 1 | 1607 | 74 | 0.81 | 0.08 | 50 | driver 3108, physics 2242, renderer 18640 | 23896 |
| s0: p2h10s-f-s0_b2 | 56 | 4 | 1 | 2672 | 75 | 0.83 | 0.08 | 50 | driver 3108, physics 2210, renderer 18798 | 23924 |
| s0: p2h10s-f-s0_n1 | 58 | 4 | 1 | 2773 | 75 | 0.83 | 0.08 | 49 | driver 3108, physics 2114, renderer 18238 | 23494 |
| s0: p2h10s-f-s0_n2 | 22 | 4 | 1 | 1141 | 69 | 0.81 | 0.08 | 50 | driver 3108, physics 1986, renderer 18998 | 23582 |
| s0: p2h10s-f-s0_n3 | 6 | 4 | 1 | 451 | 48 | 0.80 | 0.06 | 49 | driver 3108, physics 834, renderer 14198 | 17854 |
| s0: p2h10s-f-s0_q40 | 40 | 4 | 1 | 1892 | 76 | 0.81 | 0.08 | 51 | driver 3108, physics 2146, renderer 17720 | 22848 |
| s1: p2h10s-f-s1_c000 | 42 | 4 | 0 | 2090 | 72 | 0.79 | 0.08 | 52 | driver 3108, physics 2082, renderer 18992 | 24069 |
| s1: p2h10s-f-s1_c001 | 42 | 4 | 1 | 2066 | 73 | 0.79 | 0.09 | 51 | driver 3108, physics 2242, renderer 18472 | 24069 |
| s1: p2h10s-f-s1_c002 | 42 | 4 | 1 | 2142 | 71 | 0.83 | 0.08 | 51 | driver 3108, physics 2338, renderer 18592 | 23967 |
| s1: p2h10s-f-s1_c003 | 42 | 4 | 0 | 2251 | 67 | 0.82 | 0.08 | 51 | driver 3108, physics 2242, renderer 18986 | 24085 |
| s1: p2h10s-f-s1_c004 | 42 | 4 | 1 | 2182 | 69 | 0.80 | 0.08 | 51 | driver 3108, physics 2114, renderer 19684 | 24085 |
| s1: p2h10s-f-s1_c005 | 38 | 4 | 0 | 1938 | 71 | 0.82 | 0.08 | 51 | driver 3108, physics 2082, renderer 19298 | 23937 |
| s1: s1_b1a_c4 | 33 | 4 | 1 | 1737 | 68 | 0.83 | 0.08 | 51 | driver 3108, physics 2498, renderer 17868 | 24007 |
