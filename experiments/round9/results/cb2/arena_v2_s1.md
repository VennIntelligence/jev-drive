# CB2 stage 1, arena, fixed scorer (1b024f0): score-only (arena, facts cache)

Arena held windows read by both, route-fail windows dropped (free_m arena's rule); log-clustered bootstrap. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| CB2P1N-F-s0@cb2 | STANDFMXH-L18-s1@cb2 | 2228 | 0.7844 / 0.9412 | -0.1567 [-0.2168, -0.0993] | 474 / 89 | 459 / 87 |
| CB2P1N-F-s0@cb2 | STANDFMXH-L18-s0@cb2 | 2228 | 0.7844 / 0.9431 | -0.1587 [-0.2180, -0.1029] | 474 / 79 | 459 / 76 |

Zero classes (held windows):

| class | CB2P1N-F-s0@cb2 | STANDFMXH-L18-s1@cb2 | STANDFMXH-L18-s0@cb2 |
|:--|--:|--:|--:|
| corridor | 363 | 25 | 19 |
| offroad | 54 | 53 | 51 |
| route-fail | 18 | 18 | 18 |
| corridor-v2 | 25 | 6 | 3 |
| AF-front | 15 | 2 | 3 |
| AF-front+corridor | 17 | 0 | 0 |
| offroad+corridor | 0 | 2 | 2 |
| AF-side | 0 | 1 | 1 |
