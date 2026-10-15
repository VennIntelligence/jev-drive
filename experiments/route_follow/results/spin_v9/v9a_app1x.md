# SPIN v9 read: v9a APP1X-L18-s0, arm `/data/runs/route_night/spin_v9/v9a_APP1X-L18-s0` vs identity `/data/runs/route_night/v2pai/APP1X-L18-s0`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 29 | 1 (29) | 1 (29) | 3 (29) | 0 / 26 | -26 [-29, -22] | 1 / 0 / 16 (4 / 2 / 0) | 0.3993 (0.0000) | +0.3993 [+0.3342, +0.4606] |
| controls | 33 | 0 (0) | 0 (0) | 7 (6) | 2 / 1 | +1 [-2, +4] | 3 / 1 / 14 (1 / 0 / 15) | 0.4416 (0.4529) | -0.0113 [-0.1099, +0.0726] |
| controls, all 46 | 46 | 0 (13) | 0 (10) | 7 (16) | 2 / 11 | -9 [-16, -3] | 3 / 1 / 22 (3 / 1 / 17) | 0.4550 (0.3709) | +0.0841 [-0.0031, +0.1637] |
| union | 90 | 1 (38) | 1 (29) | 21 (45) | 3 / 27 | -24 [-34, -15] | 6 / 3 / 33 (5 / 2 / 18) | 0.4577 (0.3425) | +0.1152 [+0.0526, +0.1747] |

- L1 spin set spins 1 <= 2: pass
- L2 controls zeros CI upper +4 <= +2 and mean diff -0.0113 >= -0.01: FAIL
- L3 new offroad zeros with the event <= 4.7 s over the union: 0 : pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, controls, all 46 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 05855d57 | spin set + controls, all 46 | 0.000 | 0.457 | collision / fast+handoff (1.8) | - / - | 2.3 / - |
| 0e05cb0c | spin set + controls, all 46 | 0.000 | 0.379 | offroad / fast+handoff+slow (1.8) | - / - | 1.9 / 9.2 |
| 16300901 | spin set + controls, all 46 | 0.000 | 0.333 | collision / fast+handoff (1.8) | - / - | 2.1 / 7.3 |
| 1a050ead | spin set | 0.000 | 0.665 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 26004ca8 | spin set | 0.000 | 0.000 | offroad / fast+handoff+slow (1.9) | corridor / - | 2.2 / 11.3 |
| 2eaf1872 | spin set | 0.000 | 0.334 | corridor / fast+handoff (1.8) | - / - | 2.3 / 9.0 |
| 2fe500f7 | spin set | 0.000 | 0.000 | corridor / fast+handoff+slow (1.8) | corridor / fast+handoff (2.0) | 2.2 / 2.5 |
| 32be03e4 | spin set | 0.000 | 0.432 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 3fa5e06b | spin set | 0.000 | 0.462 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 478d6ec4 | spin set + controls, all 46 | 0.000 | 0.626 | corridor / fast+handoff (1.8) | - / - | 2.2 / - |
| 4a863a41 | spin set | 0.000 | 0.391 | offroad / fast+handoff+slow (1.8) | - / - | 2.2 / 10.9 |
| 4cb05d65 | spin set | 0.000 | 0.605 | offroad / fast+handoff+slow (1.8) | - / - | 2.1 / - |
| 596af0c7 | spin set | 0.000 | 0.329 | offroad / fast+handoff (1.8) | - / - | 2.3 / 7.3 |
| 61e73e86 | spin set | 0.000 | 0.412 | collision / fast+handoff (1.8) | - / - | 2.2 / 9.0 |
| 7dc59d0d | spin set | 0.000 | 0.406 | corridor / handoff+fast (1.9) | - / - | 2.5 / 8.7 |
| 8120803d | spin set | 0.000 | 0.373 | offroad / fast+handoff (1.8) | - / - | 2.2 / 9.5 |
| 87a3b6d4 | spin set + controls, all 46 | 0.000 | 0.334 | offroad / fast+handoff (1.8) | - / - | 2.4 / 7.1 |
| 90ab8c24 | spin set + controls, all 46 | 0.000 | 0.371 | offroad / fast+handoff (1.8) | - / - | 2.2 / 8.2 |
| a9c7f2c7 | spin set | 0.000 | 0.663 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| afcb2149 | spin set + controls, all 46 | 0.000 | 0.348 | corridor / fast+handoff+slow (1.8) | - / - | 2.3 / 7.3 |
| bbc53aa8 | spin set | 0.000 | 0.000 | offroad / fast+handoff+slow (1.8) | collision / - | 2.2 / 6.1 |
| ca481583 | spin set | 0.000 | 0.529 | corridor / fast+handoff (1.8) | - / - | 2.6 / 12.9 |
| d2a7ae02 | spin set | 0.000 | 0.622 | offroad / fast+handoff (1.8) | - / - | 2.1 / - |
| dc222bc3 | spin set + controls, all 46 | 0.000 | 0.467 | offroad / fast+handoff (1.8) | - / - | 2.0 / 6.1 |
| e7db8971 | spin set | 0.000 | 0.643 | offroad / fast+handoff (1.8) | - / - | 2.1 / - |
| f1575120 | spin set + controls, all 46 | 0.000 | 0.366 | offroad / fast+handoff+slow (1.8) | - / - | 2.0 / 7.1 |
| f7ded74f | spin set | 0.000 | 0.322 | collision / fast+handoff+slow (1.8) | - / - | 2.2 / 6.5 |
| fcd8cc77 | spin set | 0.000 | 0.350 | offroad / fast+handoff (1.8) | - / - | 2.1 / 7.9 |
| ffe5b17d | spin set + controls, all 46 | 0.000 | 0.362 | offroad / fast+handoff (1.8) | - / - | 2.3 / - |
| 00fabf60 |  | 0.000 | 0.000 | corridor / fast (19.8) | corridor / - | 16.4 / 16.4 |
| 06b3e399 |  | 0.370 | 0.420 | - / - | - / - | 8.4 / 9.3 |
| 1e8e939e |  | 1.000 | 0.000 | - / - | collision / - | - / 7.8 |
| 302c5c99 |  | 0.000 | 0.000 | corridor / fast (12.0) | corridor / - | 9.4 / 9.2 |
| 403d558a |  | 0.000 | 0.000 | offroad / - | collision / - | 6.0 / 9.9 |
| 4f7cb072 |  | 0.395 | 0.309 | - / - | - / - | - / 7.8 |
| 7381c9fa |  | 0.000 | 0.000 | offroad / fast (19.2) | offroad / - | 3.5 / 3.6 |
| 9d9052bd |  | 0.998 | 1.000 | - / - | - / - | - / - |
| b40f4729 |  | 1.000 | 1.000 | - / fast (4.8) | - / - | - / - |
| eaba3f6a |  | 1.000 | 1.000 | - / fast (9.2) | - / - | - / - |
| f9e361aa |  | 1.000 | 1.000 | - / fast (8.3) | - / - | - / - |
| 1a5da90a | controls + controls, all 46 | 0.266 | 0.289 | - / - | - / - | 5.6 / 6.0 |
| 524089d2 | controls + controls, all 46 | 0.328 | 0.382 | - / - | - / - | 13.0 / 13.8 |
| a6941bb6 | controls, all 46 | 0.275 | 0.314 | - / fast (6.1) | - / - | 5.6 / 6.2 |
| 91d84204 | controls + controls, all 46 | 0.536 | 0.594 | - / - | - / - | 9.4 / 10.2 |
| b88aa3d5 | controls + controls, all 46 | 0.346 | 0.385 | - / - | - / - | 6.3 / 6.9 |
| 901c2cff | controls + controls, all 46 | 0.432 | 0.490 | - / - | - / - | - / - |
| 6e7b42a3 | controls + controls, all 46 | 0.370 | 0.409 | - / - | - / - | 8.0 / 8.6 |
| 450d5c8d | controls + controls, all 46 | 0.455 | 0.502 | - / - | - / - | - / - |
| b27b3375 | controls + controls, all 46 | 0.358 | 0.540 | - / - | - / - | 6.3 / 10.6 |
| 7c346fd5 | controls + controls, all 46 | 0.267 | 0.290 | - / - | - / - | 6.3 / 6.7 |
| c0c88642 | controls + controls, all 46 | 0.286 | 0.324 | - / - | - / - | 5.6 / 6.2 |
| b727530b | controls + controls, all 46 | 0.334 | 0.461 | - / - | - / - | 11.1 / 13.2 |
| ea468790 | controls + controls, all 46 | 0.553 | 0.521 | - / - | - / - | 19.6 / 19.4 |
| ec1d0b2b | controls + controls, all 46 | 0.168 | 0.105 | - / - | - / - | 10.2 / 8.9 |
| f9121c36 | controls + controls, all 46 | 0.475 | 0.508 | - / - | - / - | 6.5 / 6.8 |
| 66b15fbe | controls + controls, all 46 | 0.549 | 0.605 | - / - | - / - | 11.8 / 12.4 |
| e9990bdd | controls, all 46 | 1.000 | 1.000 | - / fast (18.6) | - / - | - / 18.8 |
| 268949d2 | controls + controls, all 46 | 0.896 | 0.000 | - / - | collision / - | 13.9 / 14.5 |
| a68ad579 | controls + controls, all 46 | 0.384 | 0.435 | - / - | - / - | - / - |
| c4a58f83 | controls + controls, all 46 | 0.543 | 0.596 | - / - | - / - | - / - |
| e848c843 | controls + controls, all 46 | 0.669 | 0.657 | - / - | - / - | - / - |
| 09528162 | controls + controls, all 46 | 0.000 | 0.738 | collision / - | - / - | 18.0 / - |
| a4d2f7cf | controls + controls, all 46 | 0.694 | 0.716 | - / - | - / - | - / - |
| e0f6ebe1 | controls, all 46 | 0.843 | 1.000 | - / fast (15.9) | - / - | 13.5 / - |
| 132ae6cb | controls + controls, all 46 | 0.035 | 0.025 | - / - | - / - | 6.3 / 6.0 |
| 68e53160 | controls + controls, all 46 | 0.000 | 0.000 | corridor / - | collision / - | 11.4 / 12.4 |
| f8ee31cb | controls + controls, all 46 | 1.000 | 0.000 | - / - | collision / - | 18.7 / 5.2 |

VERDICT v9a APP1X-L18-s0: L1 pass, L2 FAIL, L3 pass
