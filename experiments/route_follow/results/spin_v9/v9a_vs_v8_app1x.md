# SPIN v9 read: v9a vs v8 (APP1X, 29), arm `/data/runs/route_night/spin_v9/v9a_APP1X-L18-s0` vs identity `/data/runs/route_night/spin_v9/v8_APP1X-L18-s0`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 29 | 1 (1) | 1 (1) | 3 (2) | 1 / 0 | +1 [+0, +3] | 1 / 0 / 16 (0 / 0 / 17) | 0.3993 (0.3867) | +0.0126 [-0.0157, +0.0298] |
| controls | 29 | 1 (1) | 1 (1) | 3 (2) | 1 / 0 | +1 [+0, +3] | 1 / 0 / 16 (0 / 0 / 17) | 0.3993 (0.3867) | +0.0126 [-0.0166, +0.0296] |
| union | 29 | 1 (1) | 1 (1) | 3 (2) | 1 / 0 | +1 [+0, +3] | 1 / 0 / 16 (0 / 0 / 17) | 0.3993 (0.3867) | +0.0126 [-0.0159, +0.0296] |

- L1 spin set spins 1 <= 2: pass
- L2 controls zeros CI upper +3 <= +2 and mean diff +0.0126 >= -0.01: FAIL
- L3 new offroad zeros with the event <= 4.7 s over the union: 0 : pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 05855d57 | spin set + controls | 0.424 | 0.457 | - / - | - / - | - / - |
| 0e05cb0c | spin set + controls | 0.307 | 0.379 | - / - | - / - | 6.8 / 9.2 |
| 16300901 | spin set + controls | 0.311 | 0.333 | - / - | - / - | 7.0 / 7.3 |
| 1a050ead | spin set + controls | 0.629 | 0.665 | - / - | - / - | - / - |
| 2eaf1872 | spin set + controls | 0.305 | 0.334 | - / - | - / - | 8.6 / 9.0 |
| 32be03e4 | spin set + controls | 0.392 | 0.432 | - / - | - / - | - / - |
| 3fa5e06b | spin set + controls | 0.437 | 0.462 | - / - | - / - | - / - |
| 478d6ec4 | spin set + controls | 0.616 | 0.626 | - / - | - / - | - / - |
| 4a863a41 | spin set + controls | 0.353 | 0.391 | - / - | - / - | 10.2 / 10.9 |
| 4cb05d65 | spin set + controls | 0.585 | 0.605 | - / - | - / - | - / - |
| 596af0c7 | spin set + controls | 0.310 | 0.329 | - / - | - / - | 7.0 / 7.3 |
| 61e73e86 | spin set + controls | 0.388 | 0.412 | - / - | - / - | 8.7 / 9.0 |
| 7dc59d0d | spin set + controls | 0.376 | 0.406 | - / - | - / - | 8.2 / 8.7 |
| 8120803d | spin set + controls | 0.345 | 0.373 | - / - | - / - | 9.1 / 9.5 |
| 87a3b6d4 | spin set + controls | 0.334 | 0.334 | - / - | - / - | 7.1 / 7.1 |
| 90ab8c24 | spin set + controls | 0.342 | 0.371 | - / - | - / - | 7.7 / 8.2 |
| a9c7f2c7 | spin set + controls | 0.640 | 0.663 | - / - | - / - | - / - |
| afcb2149 | spin set + controls | 0.320 | 0.348 | - / - | - / - | 6.9 / 7.3 |
| bbc53aa8 | spin set + controls | 0.350 | 0.000 | - / - | collision / - | 9.0 / 6.1 |
| ca481583 | spin set + controls | 0.496 | 0.529 | - / - | - / - | 12.5 / 12.9 |
| d2a7ae02 | spin set + controls | 0.602 | 0.622 | - / - | - / - | - / - |
| dc222bc3 | spin set + controls | 0.416 | 0.467 | - / - | - / - | 5.4 / 6.1 |
| e7db8971 | spin set + controls | 0.622 | 0.643 | - / - | - / - | - / - |
| f1575120 | spin set + controls | 0.337 | 0.366 | - / - | - / - | 6.7 / 7.1 |
| f7ded74f | spin set + controls | 0.301 | 0.322 | - / - | - / - | 6.2 / 6.5 |
| fcd8cc77 | spin set + controls | 0.322 | 0.350 | - / - | - / - | 7.4 / 7.9 |
| ffe5b17d | spin set + controls | 0.352 | 0.362 | - / - | - / - | - / - |

VERDICT v9a vs v8 (APP1X, 29): L1 pass, L2 FAIL, L3 pass
