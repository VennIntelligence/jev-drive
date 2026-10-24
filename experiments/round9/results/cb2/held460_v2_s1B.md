# CB2 stage 1, held460, fixed scorer (1b024f0): score-only (held460, facts cache)

Held 460 @18: scenes read by both; bootstrap clustered by log. Paired = arm - ref, 95 % CI (jevdrive.stats.paired).

| arm | ref | n | mean arm / ref | paired [CI] | zeros arm / ref | lateral zeros arm / ref |
|:--|:--|--:|--:|--:|--:|--:|
| CB2P1NB-F-s0-cb2 | STANDFMXH-L18-s1-cb2 | 460 | 0.8282 / 0.8549 | -0.0267 [-0.0489, -0.0060] | 42 / 40 | 29 / 29 |
| CB2P1NB-F-s0-cb2 | STANDFMXH-L18-s0-cb2 | 460 | 0.8282 / 0.8461 | -0.0179 [-0.0410, +0.0039] | 42 / 40 | 29 / 26 |
| CB2P1N-F-s0-cb2 | STANDFMXH-L18-s1-cb2 | 460 | 0.8976 / 0.8549 | +0.0427 [+0.0179, +0.0730] | 44 / 40 | 31 / 29 |
| CB2P1N-F-s0-cb2 | STANDFMXH-L18-s0-cb2 | 460 | 0.8976 / 0.8461 | +0.0515 [+0.0286, +0.0791] | 44 / 40 | 31 / 26 |

Zero classes:

| class | CB2P1NB-F-s0-cb2 | CB2P1N-F-s0-cb2 | STANDFMXH-L18-s1-cb2 | STANDFMXH-L18-s0-cb2 |
|:--|--:|--:|--:|--:|
| offroad | 22 | 20 | 19 | 17 |
| route-fail | 8 | 8 | 8 | 8 |
| corridor | 5 | 10 | 8 | 7 |
| AF-front | 5 | 5 | 3 | 6 |
| AF-side | 2 | 1 | 0 | 0 |
| offroad+corridor | 0 | 0 | 1 | 1 |
| AF-front+corridor | 0 | 0 | 1 | 1 |
