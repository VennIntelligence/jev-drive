# nuPlan arena held, native: D3 (NT4D3-F-s0) vs B2 / NTV0: score-only (arena, facts cache)

Arena held windows read by both, route-fail windows dropped (free_m arena's rule); log-clustered bootstrap. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| NT4D3-F-s0 | STANDFMXH-L18-s1 | 2228 | 0.7879 / 0.9358 | -0.1479 [-0.2056, -0.0942] | 456 / 101 | 440 / 99 |
| NT4D3-F-s0 | NTV0-L18-s0 | 2228 | 0.7879 / 0.7699 | +0.0180 [+0.0028, +0.0361] | 456 / 494 | 440 / 475 |
| R8K-L18-s2 | STANDFMXH-L18-s1 | 2228 | 0.7626 / 0.9358 | -0.1732 [-0.2311, -0.1180] | 512 / 101 | 495 / 99 |
| R8K-L18-s2 | NTV0-L18-s0 | 2228 | 0.7626 / 0.7699 | -0.0074 [-0.0200, +0.0042] | 512 / 494 | 495 / 475 |
| R6DAG-L18-s1 | STANDFMXH-L18-s1 | 2228 | 0.7671 / 0.9358 | -0.1686 [-0.2265, -0.1132] | 501 / 101 | 487 / 99 |
| R6DAG-L18-s1 | NTV0-L18-s0 | 2228 | 0.7671 / 0.7699 | -0.0028 [-0.0166, +0.0104] | 501 / 494 | 487 / 475 |

Zero classes (held windows):

| class | NT4D3-F-s0 | R8K-L18-s2 | R6DAG-L18-s1 | STANDFMXH-L18-s1 | NTV0-L18-s0 |
|:--|--:|--:|--:|--:|--:|
| corridor | 359 | 409 | 399 | 43 | 396 |
| offroad | 59 | 65 | 68 | 53 | 61 |
| route-fail | 18 | 18 | 18 | 18 | 18 |
| AF-front+corridor | 20 | 19 | 20 | 0 | 18 |
| AF-front | 16 | 17 | 14 | 2 | 19 |
| offroad+corridor | 2 | 2 | 0 | 2 | 0 |
| AF-side | 0 | 0 | 0 | 1 | 0 |
