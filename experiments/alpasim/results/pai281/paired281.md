Scenes scored by every row: 281 (base 281, s0 281, s1 281, both 281).

| Row | Mean | Zeros | at-fault collision | offroad | left corridor | other zero |
|---|--:|--:|--:|--:|--:|--:|
| base | 0.3746 | 144 | 23 | 42 | 80 | 1 |
| s0 | 0.4629 | 119 | 18 | 33 | 70 | 0 |
| s1 | 0.4837 | 110 | 14 | 29 | 72 | 0 |
| both | 0.4733 | 99 |  | | |  |

Paired difference s0 - base on 281 scenes: +0.0883, bootstrap 95% CI [+0.0472, +0.1304]; means 0.4629 / 0.3746; s0 better on 60, worse on 38, within 0.01 on 183 (max |difference| 1.0000).

Paired difference s1 - base on 281 scenes: +0.1091, bootstrap 95% CI [+0.0655, +0.1536]; means 0.4837 / 0.3746; s1 better on 62, worse on 49, within 0.01 on 170 (max |difference| 1.0000).

Paired difference both - base on 281 scenes: +0.0987, bootstrap 95% CI [+0.0594, +0.1394]; means 0.4733 / 0.3746; both better on 69, worse on 47, within 0.01 on 165 (max |difference| 1.0000).

Paired difference s1 - s0 on 281 scenes: +0.0208, bootstrap 95% CI [-0.0098, +0.0516]; means 0.4837 / 0.4629; s1 better on 38, worse on 50, within 0.01 on 193 (max |difference| 1.0000).

| Run | Scenes | CONC | Card | Wall s | Scenes / h | Render share | Drive share | Driver ms / call | VRAM peaks MiB | Card peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| base: full1_ab | 99 | 4 | 1 | 4646 | 77 | 0.82 | 0.08 | 50 | driver 3108, physics 2498, renderer 18150 | 23790 |
| base: base_q40_c1 | 40 | 4 | 1 | 1821 | 79 | 0.80 | 0.08 | 50 | driver 3108, physics 2146, renderer 17854 | 23142 |
| base: p2h10-f-s0_c000 | 36 | 4 | 0 | 1866 | 69 | 0.81 | 0.08 | 51 | driver 3108, physics 2210, renderer 18626 | 24042 |
| base: p2h10-f-s0_c001 | 36 | 4 | 1 | 1940 | 67 | 0.83 | 0.08 | 51 | driver 3108, physics 1986, renderer 18914 | 24042 |
| base: p2h10-f-s0_c002 | 36 | 4 | 0 | 1753 | 74 | 0.81 | 0.08 | 51 | driver 3108, physics 2050, renderer 17234 | 23518 |
| base: p2h10-f-s0_c003 | 34 | 4 | 1 | 1786 | 69 | 0.82 | 0.08 | 51 | driver 3108, physics 1986, renderer 18774 | 23518 |
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
