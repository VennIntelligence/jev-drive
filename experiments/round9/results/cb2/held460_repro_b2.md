# B2 native re-reads vs the stored reads, held460, old scorer: score-only (held460, facts cache)

Held 460 @18: scenes read by both; bootstrap clustered by log. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| STANDFMXH-L18-s1-cb2 | STANDFMXH-L18-s1 | 460 | 0.8527 / 0.8527 | +0.0000 [+0.0000, +0.0000] | 41 / 41 | 30 / 30 |
| STANDFMXH-L18-s1-cb2 | STANDFMXH-L18-s0 | 460 | 0.8527 / 0.8439 | +0.0088 [-0.0061, +0.0233] | 41 / 41 | 30 / 27 |
| STANDFMXH-L18-s0-cb2 | STANDFMXH-L18-s1 | 460 | 0.8439 / 0.8527 | -0.0088 [-0.0232, +0.0061] | 41 / 41 | 27 / 30 |
| STANDFMXH-L18-s0-cb2 | STANDFMXH-L18-s0 | 460 | 0.8439 / 0.8439 | -0.0000 [-0.0004, +0.0004] | 41 / 41 | 27 / 27 |

Zero classes:

| class | STANDFMXH-L18-s1-cb2 | STANDFMXH-L18-s0-cb2 | STANDFMXH-L18-s1 | STANDFMXH-L18-s0 |
|:--|--:|--:|--:|--:|
| offroad | 19 | 17 | 19 | 17 |
| corridor | 9 | 8 | 9 | 8 |
| route-fail | 8 | 8 | 8 | 8 |
| AF-front | 3 | 6 | 3 | 6 |
| AF-front+corridor | 1 | 1 | 1 | 1 |
| offroad+corridor | 1 | 1 | 1 | 1 |
