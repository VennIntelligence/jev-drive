# SPIN v9 read: A6 jlon_v=8 APP1X-L18-s0 vs v8 (v8 diagnostic), arm `/data/runs/route_night/spin_v9/jlon_APP1X-L18-s0` vs identity `/data/runs/route_night/spin_v9/v8_APP1X-L18-s0`

| set | scenes (both) | spins sf any (id) | handoff spins (id) | zeros (id) | new / recovered zeros | zeros diff 95 % CI | at-fault / front / rear (id) | at-fault diff 95 % CI | mean (id) | diff 95 % CI |
|:--|--:|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| spin set | 29 | 1 (1) | 1 (1) | 2 (2) | 0 / 0 | +0 [+0, +0] | 0 / 0 / 17 (0 / 0 / 17) | +0 [+0, +0] | 0.3869 (0.3867) | +0.0002 [-0.0012, +0.0019] |
| controls | 29 | 1 (1) | 1 (1) | 2 (2) | 0 / 0 | +0 [+0, +0] | 0 / 0 / 17 (0 / 0 / 17) | +0 [+0, +0] | 0.3869 (0.3867) | +0.0002 [-0.0012, +0.0020] |
| union | 29 | 1 (1) | 1 (1) | 2 (2) | 0 / 0 | +0 [+0, +0] | 0 / 0 / 17 (0 / 0 / 17) | +0 [+0, +0] | 0.3869 (0.3867) | +0.0002 [-0.0012, +0.0019] |

- A6 L1 spin set spins 1 <= reference 1: pass
- A6 L2 control zeros 2 <= reference 2, scored: [] (zero: []): pass
- A6 L3 new zeros (any kind) with the first event <= 4.7 s: 0 []: pass
- missing rollouts (no metrics in either root): spin set 0, controls 0, union 0

## Per scene (union; changed score, zero or spin only)

| scene | sets | id score | arm score | id zero / spin (onset s) | arm zero / spin (onset s) | id / arm first event s |
|:--|:--|--:|--:|:--|:--|--:|
| 05855d57 | spin set + controls | 0.424 | 0.442 | - / - | - / - | - / - |
| 16300901 | spin set + controls | 0.311 | 0.311 | - / - | - / - | 7.0 / 7.0 |
| 1a050ead | spin set + controls | 0.629 | 0.623 | - / - | - / - | - / - |
| 2eaf1872 | spin set + controls | 0.305 | 0.308 | - / - | - / - | 8.6 / 8.7 |
| 32be03e4 | spin set + controls | 0.392 | 0.399 | - / - | - / - | - / - |
| 3fa5e06b | spin set + controls | 0.437 | 0.433 | - / - | - / - | - / - |
| 4a863a41 | spin set + controls | 0.353 | 0.347 | - / - | - / - | 10.2 / 10.1 |
| 596af0c7 | spin set + controls | 0.310 | 0.310 | - / - | - / - | 7.0 / 7.0 |
| 8120803d | spin set + controls | 0.345 | 0.346 | - / - | - / - | 9.1 / 9.1 |
| 90ab8c24 | spin set + controls | 0.342 | 0.342 | - / - | - / - | 7.7 / 7.7 |
| afcb2149 | spin set + controls | 0.320 | 0.319 | - / - | - / - | 6.9 / 6.9 |
| bbc53aa8 | spin set + controls | 0.350 | 0.350 | - / - | - / - | 9.0 / 9.0 |
| dc222bc3 | spin set + controls | 0.416 | 0.416 | - / - | - / - | 5.4 / 5.4 |
| fcd8cc77 | spin set + controls | 0.322 | 0.322 | - / - | - / - | 7.4 / 7.4 |
| ffe5b17d | spin set + controls | 0.352 | 0.346 | - / - | - / - | - / - |

VERDICT A6 jlon_v=8 APP1X-L18-s0 vs v8 (v8 diagnostic): L1 pass, L2 pass, L3 pass
