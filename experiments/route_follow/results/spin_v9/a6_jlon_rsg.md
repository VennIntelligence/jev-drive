# SPIN v9 read: A6 jlon_v=8 RSG-L18-s0 vs v8 (RSG-L18-s0-gate reread), arm `/data/runs/route_night/spin_v9/jlon_RSG-L18-s0` vs identity `/data/runs/route_night/v2pai/RSG-L18-s0-gate`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | at-fault diff 95 % CI | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 38 | 1 (1) | 0 (0) | 15 (15) | 0 / 0 | +0 [+0, +0] | 1 / 0 / 6 (1 / 0 / 6) | +0 [+0, +0] | 0.4514 (0.4516) | -0.0002 [-0.0007, +0.0002] |
| controls | 46 | 0 (0) | 0 (0) | 8 (7) | 1 / 0 | +1 [+0, +3] | 1 / 1 / 19 (1 / 1 / 19) | +0 [+0, +0] | 0.5612 (0.5823) | -0.0210 [-0.0617, -0.0001] |
| union | 84 | 1 (1) | 0 (0) | 23 (22) | 1 / 0 | +1 [+0, +3] | 2 / 1 / 25 (2 / 1 / 25) | +0 [+0, +0] | 0.5115 (0.5231) | -0.0116 [-0.0339, -0.0001] |

- A6 L1 spin set spins 1 <= reference 1: pass
- A6 L2 control zeros 8 <= reference 7, scored: ['268949d2', '09528162'] (zero: ['268949d2', '09528162']): FAIL
- A6 L3 new zeros (any kind) with the first event <= 4.7 s: 0 []: pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 32be03e4 | spin set | 0.439 | 0.441 | - / - | - / - | - / - |
| 4f7cb072 | spin set | 0.451 | 0.442 | - / - | - / - | - / - |
| bbc53aa8 | spin set | 0.576 | 0.576 | - / - | - / - | - / - |
| 524089d2 | controls | 0.511 | 0.499 | - / - | - / - | 15.8 / 15.6 |
| 901c2cff | controls | 0.612 | 0.605 | - / - | - / - | - / - |
| 450d5c8d | controls | 0.724 | 0.722 | - / - | - / - | - / - |
| ffe5b17d | controls | 0.452 | 0.442 | - / - | - / - | - / - |
| 05855d57 | controls | 0.545 | 0.541 | - / - | - / - | - / - |
| b727530b | controls | 0.823 | 0.807 | - / - | - / - | 19.6 / 19.3 |
| ea468790 | controls | 0.522 | 0.527 | - / - | - / - | - / - |
| ec1d0b2b | controls | 0.137 | 0.142 | - / - | - / - | 9.6 / 9.7 |
| f9121c36 | controls | 0.728 | 0.727 | - / - | - / - | 9.2 / 9.2 |
| 66b15fbe | controls | 0.859 | 0.842 | - / - | - / - | 15.4 / 15.2 |
| a68ad579 | controls | 0.425 | 0.441 | - / - | - / - | - / - |
| 62cfe9d3 | controls | 0.915 | 0.000 | - / - | corridor / - | - / 19.1 |
| c4a58f83 | controls | 0.839 | 0.832 | - / - | - / - | - / - |
| e848c843 | controls | 0.789 | 0.781 | - / - | - / - | - / - |
| 132ae6cb | controls | 0.029 | 0.031 | - / - | - / - | 6.1 / 6.2 |

VERDICT A6 jlon_v=8 RSG-L18-s0 vs v8 (RSG-L18-s0-gate reread): L1 pass, L2 FAIL, L3 pass
