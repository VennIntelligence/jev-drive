# SPIN v9 read: v8 diagnostic APP1X-L18-s0, arm `/data/runs/route_night/spin_v9/v8_APP1X-L18-s0` vs identity `/data/runs/route_night/v2pai/APP1X-L18-s0`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 29 | 1 (29) | 1 (29) | 2 (29) | 0 / 27 | -27 [-29, -24] | 0 / 0 / 17 (4 / 2 / 0) | 0.3867 (0.0000) | +0.3867 [+0.3298, +0.4422] |
| controls | 29 | 1 (29) | 1 (29) | 2 (29) | 0 / 27 | -27 [-29, -24] | 0 / 0 / 17 (4 / 2 / 0) | 0.3867 (0.0000) | +0.3867 [+0.3302, +0.4416] |
| union | 29 | 1 (29) | 1 (29) | 2 (29) | 0 / 27 | -27 [-29, -24] | 0 / 0 / 17 (4 / 2 / 0) | 0.3867 (0.0000) | +0.3867 [+0.3295, +0.4418] |

- L1 spin set spins 1 <= 2: pass
- L2 controls zeros CI upper -24 <= +2 and mean diff +0.3867 >= -0.01: pass
- L3 new offroad zeros with the event <= 4.7 s over the union: 0 : pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 05855d57 | spin set + controls | 0.000 | 0.424 | collision / fast+handoff (1.8) | - / - | 2.3 / - |
| 0e05cb0c | spin set + controls | 0.000 | 0.307 | offroad / fast+handoff+slow (1.8) | - / - | 1.9 / 6.8 |
| 16300901 | spin set + controls | 0.000 | 0.311 | collision / fast+handoff (1.8) | - / - | 2.1 / 7.0 |
| 1a050ead | spin set + controls | 0.000 | 0.629 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 26004ca8 | spin set + controls | 0.000 | 0.000 | offroad / fast+handoff+slow (1.9) | corridor / - | 2.2 / 13.0 |
| 2eaf1872 | spin set + controls | 0.000 | 0.305 | corridor / fast+handoff (1.8) | - / - | 2.3 / 8.6 |
| 2fe500f7 | spin set + controls | 0.000 | 0.000 | corridor / fast+handoff+slow (1.8) | corridor / fast+handoff (2.0) | 2.2 / 2.5 |
| 32be03e4 | spin set + controls | 0.000 | 0.392 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 3fa5e06b | spin set + controls | 0.000 | 0.437 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| 478d6ec4 | spin set + controls | 0.000 | 0.616 | corridor / fast+handoff (1.8) | - / - | 2.2 / - |
| 4a863a41 | spin set + controls | 0.000 | 0.353 | offroad / fast+handoff+slow (1.8) | - / - | 2.2 / 10.2 |
| 4cb05d65 | spin set + controls | 0.000 | 0.585 | offroad / fast+handoff+slow (1.8) | - / - | 2.1 / - |
| 596af0c7 | spin set + controls | 0.000 | 0.310 | offroad / fast+handoff (1.8) | - / - | 2.3 / 7.0 |
| 61e73e86 | spin set + controls | 0.000 | 0.388 | collision / fast+handoff (1.8) | - / - | 2.2 / 8.7 |
| 7dc59d0d | spin set + controls | 0.000 | 0.376 | corridor / handoff+fast (1.9) | - / - | 2.5 / 8.2 |
| 8120803d | spin set + controls | 0.000 | 0.345 | offroad / fast+handoff (1.8) | - / - | 2.2 / 9.1 |
| 87a3b6d4 | spin set + controls | 0.000 | 0.334 | offroad / fast+handoff (1.8) | - / - | 2.4 / 7.1 |
| 90ab8c24 | spin set + controls | 0.000 | 0.342 | offroad / fast+handoff (1.8) | - / - | 2.2 / 7.7 |
| a9c7f2c7 | spin set + controls | 0.000 | 0.640 | offroad / fast+handoff (1.8) | - / - | 2.2 / - |
| afcb2149 | spin set + controls | 0.000 | 0.320 | corridor / fast+handoff+slow (1.8) | - / - | 2.3 / 6.9 |
| bbc53aa8 | spin set + controls | 0.000 | 0.350 | offroad / fast+handoff+slow (1.8) | - / - | 2.2 / 9.0 |
| ca481583 | spin set + controls | 0.000 | 0.496 | corridor / fast+handoff (1.8) | - / - | 2.6 / 12.5 |
| d2a7ae02 | spin set + controls | 0.000 | 0.602 | offroad / fast+handoff (1.8) | - / - | 2.1 / - |
| dc222bc3 | spin set + controls | 0.000 | 0.416 | offroad / fast+handoff (1.8) | - / - | 2.0 / 5.4 |
| e7db8971 | spin set + controls | 0.000 | 0.622 | offroad / fast+handoff (1.8) | - / - | 2.1 / - |
| f1575120 | spin set + controls | 0.000 | 0.337 | offroad / fast+handoff+slow (1.8) | - / - | 2.0 / 6.7 |
| f7ded74f | spin set + controls | 0.000 | 0.301 | collision / fast+handoff+slow (1.8) | - / - | 2.2 / 6.2 |
| fcd8cc77 | spin set + controls | 0.000 | 0.322 | offroad / fast+handoff (1.8) | - / - | 2.1 / 7.4 |
| ffe5b17d | spin set + controls | 0.000 | 0.352 | offroad / fast+handoff (1.8) | - / - | 2.3 / - |

VERDICT v8 diagnostic APP1X-L18-s0: L1 pass, L2 pass, L3 pass
