# SPIN v9 read: v9a RSG-L18-s0, arm `/data/runs/route_night/spin_v9/v9a_RSG-L18-s0` vs identity `/data/runs/route_night/r281/rsg`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 38 | 1 (26) | 0 (2) | 17 (16) | 3 / 2 | +1 [-3, +5] | 2 / 1 / 5 (1 / 1 / 6) | 0.4329 (0.4437) | -0.0109 [-0.0903, +0.0580] |
| controls | 46 | 0 (0) | 0 (0) | 7 (6) | 2 / 1 | +1 [-2, +5] | 1 / 1 / 19 (0 / 0 / 20) | 0.5885 (0.5668) | +0.0217 [-0.0428, +0.0855] |
| union | 90 | 1 (26) | 0 (2) | 24 (22) | 5 / 3 | +2 [-3, +8] | 3 / 2 / 27 (1 / 1 / 29) | 0.5372 (0.5294) | +0.0078 [-0.0388, +0.0520] |

- L1 spin set spins 1 <= 2: pass
- L2 controls zeros CI upper +5 <= +2 and mean diff +0.0217 >= -0.01: FAIL
- L3 new offroad zeros with the event <= 4.7 s over the union: 0 : pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 05855d57 | controls | 0.532 | 0.552 | - / - | - / - | - / - |
| 0e05cb0c | controls | 0.467 | 0.467 | - / - | - / - | 9.4 / 9.4 |
| 16300901 | controls | 0.394 | 0.412 | - / - | - / - | 8.2 / 8.5 |
| 1a050ead |  | 0.897 | 0.939 | - / - | - / - | - / - |
| 26004ca8 | spin set | 0.316 | 0.000 | - / - | corridor / - | 5.7 / 6.9 |
| 2fe500f7 |  | 0.485 | 0.509 | - / - | - / - | 8.6 / 9.0 |
| 32be03e4 | spin set | 0.000 | 0.456 | corridor / fast+handoff (1.8) | - / - | 2.3 / - |
| 3fa5e06b | spin set | 0.587 | 0.644 | - / - | - / - | - / - |
| 478d6ec4 | controls | 0.868 | 0.874 | - / - | - / - | - / - |
| 4a863a41 | spin set | 0.450 | 0.499 | - / - | - / - | 10.7 / 11.5 |
| 4cb05d65 | spin set | 0.000 | 0.714 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 596af0c7 | spin set | 0.408 | 0.428 | - / - | - / - | 8.6 / 8.9 |
| 61e73e86 | spin set | 0.695 | 0.710 | - / fast (13.3) | - / - | 12.7 / 12.9 |
| 7dc59d0d |  | 0.709 | 0.730 | - / - | - / - | 13.1 / 13.4 |
| 8120803d | spin set | 0.480 | 0.497 | - / - | - / - | 11.1 / 11.4 |
| 90ab8c24 | controls | 0.519 | 0.549 | - / - | - / - | 10.6 / 11.1 |
| a9c7f2c7 | spin set | 0.885 | 0.918 | - / - | - / - | - / - |
| afcb2149 | controls | 0.383 | 0.398 | - / - | - / - | 7.8 / 8.0 |
| bbc53aa8 | spin set | 0.531 | 0.000 | - / - | collision / - | - / 12.2 |
| ca481583 |  | 0.998 | 1.000 | - / - | - / - | - / - |
| d2a7ae02 |  | 0.900 | 0.912 | - / - | - / - | - / - |
| dc222bc3 | controls | 0.571 | 0.659 | - / - | - / - | 7.6 / 8.9 |
| e7db8971 | spin set | 0.844 | 0.852 | - / - | - / - | - / - |
| f1575120 | controls | 0.495 | 0.509 | - / - | - / - | 9.1 / 9.3 |
| f7ded74f |  | 0.719 | 0.737 | - / - | - / - | 12.8 / 13.1 |
| fcd8cc77 | spin set | 0.380 | 0.405 | - / - | - / - | 8.4 / 8.8 |
| ffe5b17d | controls | 0.410 | 0.450 | - / - | - / - | - / - |
| 00fabf60 | spin set | 0.000 | 0.000 | corridor / fast (17.5) | corridor / - | 14.7 / 14.9 |
| 06b3e399 | spin set | 0.858 | 0.860 | - / - | - / - | - / - |
| 0a228e44 | spin set | 0.000 | 0.000 | corridor / fast (10.3) | corridor / - | 7.8 / 7.8 |
| 1e8e939e | spin set | 0.000 | 0.000 | collision / fast (18.9) | collision / - | 6.5 / 6.4 |
| 23666aa2 | spin set | 1.000 | 1.000 | - / fast (7.1) | - / - | 6.7 / 6.8 |
| 2466266e | spin set | 0.000 | 0.000 | offroad / fast (10.3) | offroad / - | 9.1 / 9.0 |
| 300465f7 | spin set | 0.000 | 0.000 | offroad / fast (16.3) | offroad / - | 8.7 / 9.0 |
| 302c5c99 | spin set | 0.000 | 0.000 | corridor / fast (19.1) | corridor / - | 7.7 / 7.7 |
| 3ffad395 | spin set | 1.000 | 1.000 | - / fast (18.2) | - / - | 14.1 / 13.7 |
| 403d558a | spin set | 0.000 | 0.000 | corridor / fast (10.1) | corridor / - | 4.1 / 4.1 |
| 4e8c8f9e | spin set | 1.000 | 1.000 | - / fast (17.5) | - / - | 18.2 / 18.4 |
| 4f7cb072 | spin set | 0.427 | 0.464 | - / - | - / - | - / - |
| 51400458 | spin set | 1.000 | 1.000 | - / fast (14.5) | - / - | 12.6 / 12.9 |
| 60cc35db | spin set | 1.000 | 1.000 | - / fast (15.5) | - / - | 15.6 / 15.9 |
| 7381c9fa | spin set | 0.000 | 0.000 | offroad / fast (12.4) | offroad / - | 3.5 / 3.6 |
| 9d9052bd | spin set | 1.000 | 1.000 | - / fast (17.6) | - / - | 16.8 / 16.7 |
| b40f4729 | spin set | 1.000 | 0.000 | - / fast+slow (16.0) | corridor / - | 19.2 / 7.8 |
| ba40be62 | spin set | 0.000 | 0.000 | corridor / fast (7.3) | corridor / - | 6.3 / 6.3 |
| ce63741d | spin set | 1.000 | 1.000 | - / fast (9.2) | - / - | 8.9 / 9.2 |
| dc401449 | spin set | 1.000 | 1.000 | - / fast (18.9) | - / - | 17.6 / 17.3 |
| e1c6ae9f | spin set | 1.000 | 1.000 | - / fast (14.4) | - / - | 15.8 / 14.9 |
| eaba3f6a | spin set | 0.000 | 0.000 | corridor / fast (10.6) | corridor / - | 5.6 / 5.6 |
| f9e361aa | spin set | 0.000 | 0.000 | corridor / fast (17.9) | corridor / - | 6.6 / 6.7 |
| fc9fde87 | spin set | 0.000 | 0.000 | corridor / fast (14.9) | corridor / - | 10.6 / 10.8 |
| 1a5da90a | controls | 0.396 | 0.415 | - / - | - / - | 7.7 / 8.0 |
| 524089d2 | controls | 0.466 | 0.532 | - / - | - / - | 15.1 / 16.1 |
| a6941bb6 | controls | 0.459 | 0.477 | - / - | - / - | 8.5 / 8.8 |
| 91d84204 | controls | 0.825 | 0.857 | - / - | - / - | 13.6 / 14.1 |
| b88aa3d5 | controls | 0.567 | 0.609 | - / - | - / - | 9.5 / 10.1 |
| 901c2cff | controls | 0.532 | 0.647 | - / - | - / - | - / - |
| 6e7b42a3 | controls | 0.895 | 0.909 | - / - | - / - | 16.0 / 16.2 |
| 450d5c8d | controls | 0.680 | 0.738 | - / - | - / - | - / - |
| 7c346fd5 | controls | 0.343 | 0.359 | - / - | - / - | 7.6 / 7.9 |
| c0c88642 | controls | 0.509 | 0.517 | - / - | - / - | 8.8 / 8.9 |
| b727530b | controls | 0.633 | 0.823 | - / - | - / - | 15.8 / 19.6 |
| ea468790 | controls | 0.549 | 0.522 | - / - | - / - | - / - |
| ec1d0b2b | controls | 0.230 | 0.137 | - / - | - / - | 11.1 / 9.6 |
| f9121c36 | controls | 0.596 | 0.769 | - / - | - / - | 7.8 / 9.7 |
| 66b15fbe | controls | 0.658 | 0.859 | - / - | - / - | 13.0 / 15.4 |
| 268949d2 | controls | 1.000 | 0.000 | - / - | corridor / - | 15.7 / 10.6 |
| a68ad579 | controls | 0.443 | 0.443 | - / - | - / - | - / - |
| 62cfe9d3 | controls | 0.883 | 0.911 | - / - | - / - | - / - |
| c4a58f83 | controls | 0.684 | 0.862 | - / - | - / - | - / - |
| e848c843 | controls | 0.787 | 0.789 | - / - | - / - | - / - |
| 09528162 | controls | 0.285 | 0.000 | - / - | corridor / - | 5.8 / 5.7 |
| a4d2f7cf | controls | 0.976 | 1.000 | - / - | - / - | - / - |
| 132ae6cb | controls | 0.041 | 0.029 | - / - | - / - | 6.5 / 6.1 |
| 1bfad87c | controls | 0.000 | 1.000 | offroad / - | - / - | 18.1 / 18.1 |
| 68e53160 | controls | 0.000 | 0.000 | corridor / - | collision / - | 11.1 / 11.8 |

VERDICT v9a RSG-L18-s0: L1 pass, L2 FAIL, L3 pass
