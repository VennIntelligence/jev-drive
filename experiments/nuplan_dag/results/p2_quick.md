# Part 2: NDAG vs B2: score-only headline (scorer outputs, no extract)

Arena: held windows read by both tags, route-fail windows dropped (free_m arena's rule); held 460 @18: scenes read by both. Paired = arm - ref, log-clustered bootstrap 95 % CI (jevdrive.stats.paired).

| arm | ref | read | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|:--|--:|--:|--:|--:|--:|
| NDAG-L18-s0 | STANDFMXH-L18-s0 | arena | 2228 | 0.9354 / 0.9373 | -0.0018 [-0.0071, +0.0035] | 99 / 92 | 95 / 89 |
| NDAG-L18-s0 | STANDFMXH-L18-s0 | held 460 | 460 | 0.8634 / 0.8439 | +0.0195 [+0.0009, +0.0365] | 35 / 41 | 22 / 27 |
| NDAG-L18-s0 | STANDFMXH-L18-s1 | arena | 2228 | 0.9354 / 0.9358 | -0.0003 [-0.0055, +0.0051] | 99 / 101 | 95 / 99 |
| NDAG-L18-s0 | STANDFMXH-L18-s1 | held 460 | 460 | 0.8634 / 0.8527 | +0.0107 [-0.0105, +0.0291] | 35 / 41 | 22 / 30 |
| NDAG-L18-s0 | NTV0-L18-s0 | arena | 2228 | 0.9354 / 0.7699 | +0.1655 [+0.1135, +0.2206] | 99 / 494 | 95 / 475 |
| NDAG-L18-s0 | NTV0-L18-s0 | held 460 | 460 | 0.8634 / 0.8876 | -0.0241 [-0.0587, +0.0088] | 35 / 32 | 22 / 20 |
| NDAG-L18-s1 | STANDFMXH-L18-s0 | arena | 2228 | 0.9304 / 0.9373 | -0.0068 [-0.0166, +0.0015] | 111 / 92 | 107 / 89 |
| NDAG-L18-s1 | STANDFMXH-L18-s0 | held 460 | 460 | 0.8538 / 0.8439 | +0.0099 [-0.0118, +0.0291] | 38 / 41 | 24 / 27 |
| NDAG-L18-s1 | STANDFMXH-L18-s1 | arena | 2228 | 0.9304 / 0.9358 | -0.0053 [-0.0126, +0.0001] | 111 / 101 | 107 / 99 |
| NDAG-L18-s1 | STANDFMXH-L18-s1 | held 460 | 460 | 0.8538 / 0.8527 | +0.0011 [-0.0166, +0.0185] | 38 / 41 | 24 / 30 |
| NDAG-L18-s1 | NTV0-L18-s0 | arena | 2228 | 0.9304 / 0.7699 | +0.1605 [+0.1056, +0.2182] | 111 / 494 | 107 / 475 |
| NDAG-L18-s1 | NTV0-L18-s0 | held 460 | 460 | 0.8538 / 0.8876 | -0.0337 [-0.0639, -0.0008] | 38 / 32 | 24 / 20 |

Zero classes, arena held (nx_regress.failure_type):

| class | NDAG-L18-s0 | NDAG-L18-s1 | STANDFMXH-L18-s0 | STANDFMXH-L18-s1 | NTV0-L18-s0 |
|:--|--:|--:|--:|--:|--:|
| corridor | 37 | 43 | 35 | 43 | 396 |
| offroad | 57 | 61 | 51 | 53 | 61 |
| route-fail | 18 | 18 | 18 | 18 | 18 |
| AF-front | 4 | 4 | 3 | 2 | 19 |
| AF-front+corridor | 0 | 0 | 0 | 0 | 18 |
| offroad+corridor | 1 | 1 | 2 | 2 | 0 |
| AF-side | 0 | 1 | 1 | 1 | 0 |
| AF-front+offroad | 0 | 1 | 0 | 0 | 0 |

Zero classes, held 460 (nx_regress.failure_type):

| class | NDAG-L18-s0 | NDAG-L18-s1 | STANDFMXH-L18-s0 | STANDFMXH-L18-s1 | NTV0-L18-s0 |
|:--|--:|--:|--:|--:|--:|
| offroad | 16 | 15 | 17 | 19 | 15 |
| route-fail | 8 | 8 | 8 | 8 | 8 |
| corridor | 4 | 8 | 8 | 9 | 4 |
| AF-front | 5 | 6 | 6 | 3 | 4 |
| AF-front+corridor | 1 | 1 | 1 | 1 | 0 |
| offroad+corridor | 1 | 0 | 1 | 1 | 0 |
| AF-side | 0 | 0 | 0 | 0 | 1 |
