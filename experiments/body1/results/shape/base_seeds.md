Scenes common to the 4 runs: 700 from 27 logs; > 45 deg: 61.

| driver | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | collision + offroad | slow | mean progress | > 45 deg mean | > 45 deg zeros |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0 | 0.9468 [0.9252, 0.9652] | 562 | 26 | 8 | 10 | 8 | 18 | 112 | 0.942 | 0.8143 | 10 |
| P2H10-F-s1 | 0.9495 [0.9300, 0.9669] | 572 | 23 | 6 | 10 | 7 | 16 | 105 | 0.936 | 0.8544 | 8 |
| P2H10-F-s2 | 0.9498 [0.9287, 0.9685] | 576 | 24 | 7 | 11 | 6 | 18 | 100 | 0.939 | 0.8682 | 7 |
| P2H10-F-s3 | 0.9446 [0.9222, 0.9639] | 558 | 25 | 7 | 11 | 7 | 18 | 117 | 0.932 | 0.8443 | 8 |

Range over the seeds: mean 0.9446 to 0.9498; collision + offroad + corridor zeros 23 to 26; at-fault collision 6 to 8; offroad 10 to 11; corridor 6 to 8; slow 100 to 117; progress 0.932 to 0.942.

## One seed read as an arm against another as the base (same recipe; the lines' quantities)

| "arm" vs "base" | L1a zeros (collision / offroad / corridor) | L1b collision + offroad | L2 mean difference [95 % CI by log] | L3 slow vs 1.1 x base | removed / new zeros | score 1 -> slow / slow -> 1 | lines that would read as met |
|:--|:--|:--|:--|:--|:--|:--|:--|
| s0 vs s1 | 26 (8 / 10 / 8) vs 23 (6 / 10 / 7) | 18 vs 16 | -0.0028 [-0.0098, +0.0037] | 112 vs 115.5 | 1 / 4 | 17 / 10 | L3 |
| s0 vs s2 | 26 (8 / 10 / 8) vs 24 (7 / 11 / 6) | 18 vs 18 | -0.0031 [-0.0114, +0.0049] | 112 vs 110.0 | 2 / 4 | 17 / 5 | - |
| s0 vs s3 | 26 (8 / 10 / 8) vs 25 (7 / 11 / 7) | 18 vs 18 | +0.0022 [-0.0089, +0.0130] | 112 vs 128.7 | 4 / 5 | 15 / 20 | L3 |
| s1 vs s0 | 23 (6 / 10 / 7) vs 26 (8 / 10 / 8) | 16 vs 18 | +0.0028 [-0.0037, +0.0098] | 105 vs 123.2 | 4 / 1 | 10 / 17 | L1a, L1b, L2, L3 |
| s1 vs s2 | 23 (6 / 10 / 7) vs 24 (7 / 11 / 6) | 16 vs 18 | -0.0003 [-0.0067, +0.0051] | 105 vs 110.0 | 3 / 2 | 17 / 12 | L1a, L1b, L3 |
| s1 vs s3 | 23 (6 / 10 / 7) vs 25 (7 / 11 / 7) | 16 vs 18 | +0.0049 [-0.0034, +0.0138] | 105 vs 128.7 | 5 / 3 | 10 / 22 | L1a, L1b, L2, L3 |
| s2 vs s0 | 24 (7 / 11 / 6) vs 26 (8 / 10 / 8) | 18 vs 18 | +0.0031 [-0.0049, +0.0114] | 100 vs 123.2 | 4 / 2 | 5 / 17 | L1a, L2, L3 |
| s2 vs s1 | 24 (7 / 11 / 6) vs 23 (6 / 10 / 7) | 18 vs 16 | +0.0003 [-0.0051, +0.0067] | 100 vs 115.5 | 2 / 3 | 12 / 17 | L3 |
| s2 vs s3 | 24 (7 / 11 / 6) vs 25 (7 / 11 / 7) | 18 vs 18 | +0.0052 [-0.0004, +0.0121] | 100 vs 128.7 | 3 / 2 | 4 / 21 | L1a, L2, L3 |
| s3 vs s0 | 25 (7 / 11 / 7) vs 26 (8 / 10 / 8) | 18 vs 18 | -0.0022 [-0.0130, +0.0089] | 117 vs 123.2 | 5 / 4 | 20 / 15 | L1a, L3 |
| s3 vs s1 | 25 (7 / 11 / 7) vs 23 (6 / 10 / 7) | 18 vs 16 | -0.0049 [-0.0138, +0.0034] | 117 vs 115.5 | 3 / 5 | 22 / 10 | - |
| s3 vs s2 | 25 (7 / 11 / 7) vs 24 (7 / 11 / 6) | 18 vs 18 | -0.0052 [-0.0121, +0.0004] | 117 vs 110.0 | 2 / 3 | 21 / 4 | - |

Over the 12 ordered pairs: largest |mean difference| 0.0052, mean CI half-width 0.0078; zero flips per pair 6.3 on average; scenes with a zero in all seeds 20, in any seed 31; score 1 -> slow per pair 14.2; L2 would read as met in 4, L3 in 9, all four lines in 2.
