# B2 native re-reads vs the stored reads, arena, old scorer: score-only (arena, facts cache)

Arena held windows read by both, route-fail windows dropped (free_m arena's rule); log-clustered bootstrap. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| STANDFMXH-L18-s1@cb2 | STANDFMXH-L18-s1 | 2228 | 0.9358 / 0.9358 | -0.0000 [-0.0000, +0.0000] | 101 / 101 | 99 / 99 |
| STANDFMXH-L18-s1@cb2 | STANDFMXH-L18-s0 | 2228 | 0.9358 / 0.9373 | -0.0015 [-0.0093, +0.0054] | 101 / 92 | 99 / 89 |
| STANDFMXH-L18-s0@cb2 | STANDFMXH-L18-s1 | 2228 | 0.9373 / 0.9358 | +0.0015 [-0.0054, +0.0093] | 92 / 101 | 89 / 99 |
| STANDFMXH-L18-s0@cb2 | STANDFMXH-L18-s0 | 2228 | 0.9373 / 0.9373 | -0.0000 [-0.0000, +0.0000] | 92 / 92 | 89 / 89 |

Zero classes (held windows):

| class | STANDFMXH-L18-s1@cb2 | STANDFMXH-L18-s0@cb2 | STANDFMXH-L18-s1 | STANDFMXH-L18-s0 |
|:--|--:|--:|--:|--:|
| offroad | 53 | 51 | 53 | 51 |
| corridor | 43 | 35 | 43 | 35 |
| route-fail | 18 | 18 | 18 | 18 |
| AF-front | 2 | 3 | 2 | 3 |
| offroad+corridor | 2 | 2 | 2 | 2 |
| AF-side | 1 | 1 | 1 | 1 |
