Scenes scored by every row: 15 (tokrsg 281, tokboth 26, boxrsg 15, boxboth 15, boxpait 15).

| Row | Mean | Zeros | at-fault collision | offroad | left corridor | other zero |
|---|--:|--:|--:|--:|--:|--:|
| tokrsg | 0.2065 | 11 | 4 | 3 | 4 | 0 |
| tokboth | 0.2000 | 12 | 6 | 2 | 4 | 0 |
| boxrsg | 0.1881 | 12 | 4 | 3 | 5 | 0 |
| boxboth | 0.2000 | 12 | 6 | 2 | 4 | 0 |
| boxpait | 0.2734 | 8 | 3 | 1 | 4 | 0 |

Paired difference boxrsg - tokrsg on 15 scenes: -0.0185, bootstrap 95% CI [-0.0569, +0.0015]; means 0.1881 / 0.2065; boxrsg better on 0, worse on 1, within 0.01 on 14 (max |difference| 0.2847).

Paired difference boxboth - tokboth on 15 scenes: +0.0000, bootstrap 95% CI [+0.0000, +0.0000]; means 0.2000 / 0.2000; boxboth better on 0, worse on 0, within 0.01 on 15 (max |difference| 0.0000).

Paired difference boxboth - boxrsg on 15 scenes: +0.0119, bootstrap 95% CI [-0.1642, +0.2000]; means 0.2000 / 0.1881; boxboth better on 1, worse on 1, within 0.01 on 13 (max |difference| 1.0000).

Paired difference boxpait - boxrsg on 15 scenes: +0.0853, bootstrap 95% CI [-0.1214, +0.3083]; means 0.2734 / 0.1881; boxpait better on 4, worse on 2, within 0.01 on 9 (max |difference| 1.0000).

| Run | Scenes | CONC | Card | Wall s | Scenes / h | Render share | Drive share | Driver ms / call | VRAM peaks MiB | Card peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| tokrsg: rsg-l18-s0_c000 | 36 | 4 | 0 | 1859 | 70 | 0.80 | 0.08 | 54 | driver 3112, physics 2498, renderer 18478 | 23823 |
| tokrsg: rsg-l18-s0_c001 | 36 | 4 | 1 | 1928 | 67 | 0.81 | 0.09 | 57 | driver 3112, physics 2050, renderer 17424 | 23789 |
| tokrsg: rsg-l18-s0_c002 | 36 | 4 | 0 | 1998 | 65 | 0.82 | 0.08 | 57 | driver 3112, physics 2498, renderer 19188 | 23789 |
| tokrsg: rsg-l18-s0_c003 | 36 | 4 | 1 | 1965 | 66 | 0.78 | 0.08 | 55 | driver 3112, physics 2146, renderer 18922 | 23894 |
| tokrsg: rsg-l18-s0_c004 | 36 | 4 | 0 | 2014 | 64 | 0.81 | 0.08 | 55 | driver 3112, physics 2242, renderer 18578 | 23979 |
| tokrsg: rsg-l18-s0_c005 | 36 | 4 | 1 | 1959 | 66 | 0.81 | 0.08 | 55 | driver 3112, physics 1922, renderer 17766 | 23979 |
| tokrsg: rsg-l18-s0_c006 | 36 | 4 | 0 | 1898 | 68 | 0.77 | 0.09 | 57 | driver 3112, physics 2050, renderer 18040 | 23153 |
| tokrsg: rsg-l18-s0_c007 | 29 | 4 | 1 | 1563 | 67 | 0.82 | 0.07 | 54 | driver 3112, physics 2146, renderer 18734 | 23630 |
| tokboth: rsgboth-l18-s0-tk_c000 | 26 | 4 | 0 | 1522 | 61 | 0.80 | 0.08 | 55 | driver 3112, physics 1826, renderer 19062 | 23905 |

Detail on the 15 scenes scored by every row. Score 1 = perfect scene, slow = 0 < score < 1; zero kinds are the scorer's flags on zero-score scenes (two flags can sit on one scene); driver errors = inference_error + input_error counters over all scenes of the row's run dirs; a mean row has no flags and is zero on a scene only if every seed is zero.

| Row | Scenes | Mean | Score 1 | Slow | Zeros | at-fault collision | offroad | left corridor | other zero | Driver errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tokrsg | 15 | 0.2065 | 2 | 2 | 11 | 4 | 3 | 4 | 0 | 0 |
| tokboth | 15 | 0.2000 | 3 | 0 | 12 | 6 | 2 | 4 | 0 | 0 |
| boxrsg | 15 | 0.1881 | 2 | 1 | 12 | 4 | 3 | 5 | 0 | 0 |
| boxboth | 15 | 0.2000 | 3 | 0 | 12 | 6 | 2 | 4 | 0 | 0 |
| boxpait | 15 | 0.2734 | 3 | 4 | 8 | 3 | 1 | 4 | 0 | 0 |

Every pair a - b with its decomposition by zero / scored status. Fixed = b zero and a scored, broken = b scored and a zero; a part is the summed difference of those scenes divided by the scenes of the pair, so the three parts add up to the mean difference (check column). Better / worse / within = scenes where a is above b by more than 0.01 / below by more than 0.01 / within 0.01.

| Pair (a - b) | Scenes | Mean difference | Bootstrap 95 % CI | Better / worse / within | Fixed: scenes | part | Broken: scenes | part | Both scored: scenes | part | Both zero: scenes | Check |
|---|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| boxrsg - tokrsg | 15 | -0.0185 | [-0.0569, +0.0015] | 0 / 1 / 14 | 0 | +0.0000 | 1 | -0.0190 | 3 | +0.0005 | 11 | +0e+00 |
| boxboth - tokboth | 15 | +0.0000 | [+0.0000, +0.0000] | 0 / 0 / 15 | 0 | +0.0000 | 0 | +0.0000 | 3 | +0.0000 | 12 | +0e+00 |
| boxboth - boxrsg | 15 | +0.0119 | [-0.1642, +0.2000] | 1 / 1 / 13 | 1 | +0.0667 | 1 | -0.0547 | 2 | +0.0000 | 11 | -3e-18 |
| boxpait - boxrsg | 15 | +0.0853 | [-0.1214, +0.3083] | 4 / 2 / 9 | 5 | +0.1546 | 1 | -0.0547 | 2 | -0.0146 | 7 | +0e+00 |
