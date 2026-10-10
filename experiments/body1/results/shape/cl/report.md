Scenes common to all 10 runs: 700 from 27 logs; logged 4 s turn > 45 deg: 61 (token missing: 0).

## 1. Per seed

| driver | mean scene score [95 % CI, logs] | score 1 | zeros | at-fault collision | offroad | left corridor | slow (0 < score < 1) | mean progress | > 45 deg mean | > 45 deg zeros |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| P2H10-F-s0 | 0.9468 [0.9252, 0.9652] | 562 | 26 | 8 | 10 | 8 | 112 | 0.942 | 0.8143 | 10 |
| P2H10-F-s1 | 0.9495 [0.9300, 0.9669] | 572 | 23 | 6 | 10 | 7 | 105 | 0.936 | 0.8544 | 8 |
| P2H10-F-s2 | 0.9498 [0.9287, 0.9685] | 576 | 24 | 7 | 11 | 6 | 100 | 0.939 | 0.8682 | 7 |
| P2H10-F-s3 | 0.9446 [0.9222, 0.9639] | 558 | 25 | 7 | 11 | 7 | 117 | 0.932 | 0.8443 | 8 |
| P2H10S-F-s0 | 0.9486 [0.9305, 0.9647] | 568 | 24 | 4 | 10 | 10 | 108 | 0.946 | 0.8233 | 10 |
| P2H10S-F-s1 | 0.9482 [0.9278, 0.9667] | 576 | 25 | 5 | 11 | 9 | 99 | 0.944 | 0.7578 | 14 |
| P2H10S-F-s2 | 0.9538 [0.9403, 0.9675] | 576 | 21 | 5 | 7 | 9 | 103 | 0.947 | 0.8244 | 10 |
| P2H10S-F-s3 | 0.9562 [0.9413, 0.9694] | 589 | 20 | 4 | 7 | 9 | 91 | 0.949 | 0.8453 | 9 |
| P2H10B-F-s0 | 0.9441 [0.9238, 0.9617] | 541 | 22 | 5 | 10 | 7 | 137 | 0.928 | 0.8298 | 9 |
| P2H10B-F-s1 | 0.9430 [0.9234, 0.9599] | 538 | 22 | 4 | 11 | 7 | 140 | 0.921 | 0.7973 | 11 |

Ranges over the four seeds, base against arm: mean 0.9446 to 0.9498 / 0.9482 to 0.9562; zeros 23 to 26 / 20 to 25; at-fault collision 6 to 8 / 4 to 5; offroad 10 to 11 / 7 to 11; corridor 6 to 8 / 9 to 10; slow 100 to 117 / 91 to 108; progress 0.932 to 0.942 / 0.944 to 0.949.

Mean over the four seeds, base / arm: mean 0.9477 / 0.9517; zeros 24.5 / 22.5; collision 7 / 4.5; offroad 10.5 / 8.75; corridor 7 / 9.25; slow 108.5 / 100.2; ones 567 / 577.2; progress 0.9375 / 0.9467.

## 2. The four lines (descriptive; no promotion)

| read | pairs | L1a zeros arm vs base (collision / offroad / corridor) | L1b | L2 mean difference [95 % CI by log] | L3 slow arm vs 1.1 x base | lines that would read as met |
|:--|--:|:--|:--|:--|:--|:--|
| S 2 seeds (s0-1) vs TR1 base s0-1 | 1400 | 49 (9 / 21 / 19) vs 49 (14 / 20 / 15); per pair [24, 25] vs [26, 23] (not met) | 30 vs 34; per pair [14, 16] vs [18, 16] (met) | +0.0003 [-0.0060, +0.0078] (not met) | 207 vs 238.7 (base 217) (met) | L1b, L3 |
| S 4 seeds paired by seed index | 2800 | 90 (18 / 35 / 37) vs 98 (28 / 42 / 28); per pair [24, 25, 21, 20] vs [26, 23, 24, 25] (not met) | 53 vs 70; per pair [14, 16, 12, 11] vs [18, 16, 18, 18] (met) | +0.0040 [-0.0038, +0.0134] (met) | 401 vs 477.4 (base 434) (met) | L1b, L2, L3 |
| B (loss arm, Amendment 5) 2 seeds, for comparison | 1400 | 44 (9 / 21 / 14) vs 49 (14 / 20 / 15); per pair [22, 22] vs [26, 23] (met) | 30 vs 34; per pair [15, 15] vs [18, 16] (met) | -0.0046 [-0.0131, +0.0052] (not met) | 277 vs 238.7 (base 217) (not met) | L1a, L1b |

Per pair, paired by scene (CI by log): s0 vs base s0 +0.0018 [-0.0054, +0.0102]; s1 vs base s1 -0.0013 [-0.0101, +0.0091]; s2 vs base s2 +0.0040 [-0.0079, +0.0173]; s3 vs base s3 +0.0116 [-0.0001, +0.0272].

## 3. Each statistic against the base's seed spread

N1 = the 12 ordered pairs of base seeds (one pair each), N2 = the 12 disjoint 2-vs-2 splits (two pairs each); per-pair normalised values. Cells: arm value; N1 range / N2 range; number of null members below the value (of 12 / of 12); **outside** only if beyond every member of both.

| statistic | S 2 seeds (s0-1) vs TR1 base s0-1 | S 4 seeds paired by seed index | B (loss arm, Amendment 5) 2 seeds, for comparison |
|:--|:--|:--|:--|
| L2 mean difference | +0.0003; -0.0052 to +0.0052 / -0.0040 to +0.0040; 6 of 12 / 6 of 12 below; **no** | +0.0040; -0.0052 to +0.0052 / -0.0040 to +0.0040; 10 of 12 / 12 of 12 below; **no** | -0.0046; -0.0052 to +0.0052 / -0.0040 to +0.0040; 2 of 12 / 0 of 12 below; **no** |
| L2 CI lower bound | -0.0060; -0.0138 to -0.0004 / -0.0085 to +0.0004; 7 of 12 / 8 of 12 below; **no** | -0.0038; -0.0138 to -0.0004 / -0.0085 to +0.0004; 9 of 12 / 10 of 12 below; **no** | -0.0131; -0.0138 to -0.0004 / -0.0085 to +0.0004; 1 of 12 / 0 of 12 below; **no** |
| L1a zeros per pair (arm - base) | +0.00; -3.00 to +3.00 / -2.00 to +2.00; 6 of 12 / 4 of 12 below; **no** | -2.00; -3.00 to +3.00 / -2.00 to +2.00; 1 of 12 / 0 of 12 below; **no** | -2.50; -3.00 to +3.00 / -2.00 to +2.00; 1 of 12 / 0 of 12 below; **no** |
| L1b zeros per pair (arm - base) | -2.00; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **no** | -4.25; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **below** | -2.00; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **no** |
| at-fault collision zeros per pair | -2.50; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **below** | -2.50; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **below** | -2.50; -2.00 to +2.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **below** |
| offroad zeros per pair | +0.50; -1.00 to +1.00 / -1.00 to +1.00; 8 of 12 / 10 of 12 below; **no** | -1.75; -1.00 to +1.00 / -1.00 to +1.00; 0 of 12 / 0 of 12 below; **below** | +0.50; -1.00 to +1.00 / -1.00 to +1.00; 8 of 12 / 10 of 12 below; **no** |
| corridor zeros per pair | +2.00; -2.00 to +2.00 / -1.00 to +1.00; 11 of 12 / 12 of 12 below; **no** | +2.25; -2.00 to +2.00 / -1.00 to +1.00; 12 of 12 / 12 of 12 below; **above** | -0.50; -2.00 to +2.00 / -1.00 to +1.00; 5 of 12 / 4 of 12 below; **no** |
| slow scenes per pair (arm - base) | -5.0; -17.0 to +17.0 / -12.0 to +12.0; 4 of 12 / 2 of 12 below; **no** | -8.2; -17.0 to +17.0 / -12.0 to +12.0; 3 of 12 / 2 of 12 below; **no** | +30.0; -17.0 to +17.0 / -12.0 to +12.0; 12 of 12 / 12 of 12 below; **above** |
| mean progress per pair (arm - base) | +0.0060; -0.0097 to +0.0097 / -0.0066 to +0.0066; 9 of 12 / 10 of 12 below; **no** | +0.0091; -0.0097 to +0.0097 / -0.0066 to +0.0066; 11 of 12 / 12 of 12 below; **no** | -0.0145; -0.0097 to +0.0097 / -0.0066 to +0.0066; 0 of 12 / 0 of 12 below; **below** |

L2 / L3 / L1 verdict of the null members (how often a same-recipe pair would read as met): N1 L1a 6, L1b 3, L2 4, L3 9, all four 2 of 12; N2 L1a 3, L1b 6, L2 4, L3 10, all four 2 of 12.

## 4. Zero flips per scene against all four base seeds

Rule: removed = zero in at least 3 of 4 base seeds and in at most 1 of 4 arm seeds; new = the mirror image.

| | scenes | by class (collision / offroad / corridor; class by majority over the seeds that are zeros) |
|:--|--:|:--|
| removed | 5 | 2 / 2 / 1 |
| new | 4 | 0 / 0 / 4 |
| zero in at least 3 of 4 seeds of both | 15 | - |

Scenes with a zero in all four base seeds 20, in any 31; in all four arm seeds 17, in any 30. Flips (scene tail, base zeros of 4 -> arm zeros of 4, class, turn):

- `78b4153a6d3e5b33` new: 0/4 (-) -> 4/4 (corridor); turn -48.1; scores base [1.0, 1.0, 1.0, 1.0] arm [0.0, 0.0, 0.0, 0.0]
- `00ff8eeb53cc598b` removed: 4/4 (offroad) -> 1/4 (offroad); turn -44.9; scores base [0.0, 0.0, 0.0, 0.0] arm [0.0, 1.0, 1.0, 1.0]
- `77155a60b2ae5e75` new: 0/4 (-) -> 4/4 (corridor); turn -67.0; scores base [0.892, 0.834, 0.912, 0.778] arm [0.0, 0.0, 0.0, 0.0]
- `fcb45b2aa29356d9` removed: 4/4 (offroad) -> 0/4 (-); turn 40.9; scores base [0.0, 0.0, 0.0, 0.0] arm [1.0, 1.0, 1.0, 1.0]
- `927b73fea33f5218` removed: 4/4 (corridor) -> 0/4 (-); turn 46.0; scores base [0.0, 0.0, 0.0, 0.0] arm [1.0, 1.0, 1.0, 1.0]
- `e933d70dd378598f` new: 0/4 (-) -> 4/4 (corridor); turn -70.5; scores base [1.0, 0.952, 1.0, 0.963] arm [0.0, 0.0, 0.0, 0.0]
- `52d3f15d6ca75701` new: 0/4 (-) -> 4/4 (corridor); turn -55.0; scores base [1.0, 1.0, 1.0, 1.0] arm [0.0, 0.0, 0.0, 0.0]
- `994ab680895a55d2` removed: 3/4 (collision) -> 0/4 (-); turn -8.2; scores base [0.0, 0.0, 0.0, 0.808] arm [0.727, 0.729, 0.86, 0.775]
- `80f6c94fed0c5519` removed: 4/4 (collision) -> 0/4 (-); turn -1.5; scores base [0.0, 0.0, 0.0, 0.0] arm [1.0, 0.854, 1.0, 1.0]

## 5. Lead / open split (proximity group of the base plan at the scene's navtest token, decision 230)

| group | scenes | base mean progress | arm mean progress | difference per pair [95 % CI by log] | slow base -> arm (sum of 4 pairs) | N1 progress range | N2 progress range | slow per pair N1 range | outside |
|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|
| lead | 159 | 0.919 | 0.930 | +0.0115 [+0.0038, +0.0203] | 137 -> 137 | -0.0203 to +0.0203 | -0.0128 to +0.0128 | -10.0 to +10.0 | progress no; slow no |
| open | 329 | 0.945 | 0.952 | +0.0078 [+0.0032, +0.0126] | 190 -> 164 | -0.0058 to +0.0058 | -0.0038 to +0.0038 | -8.0 to +8.0 | progress above; slow no |
| contact | 26 | 0.927 | 0.953 | +0.0264 [+0.0129, +0.0399] | 8 -> 1 | -0.0138 to +0.0138 | -0.0086 to +0.0086 | +0.0 to +0.0 | progress above; slow below |
| near obj | 99 | 0.928 | 0.932 | +0.0040 [-0.0070, +0.0135] | 71 -> 73 | -0.0113 to +0.0113 | -0.0057 to +0.0057 | -5.0 to +5.0 | progress no; slow no |
| near edge | 87 | 0.960 | 0.970 | +0.0104 [+0.0033, +0.0214] | 28 -> 26 | -0.0167 to +0.0167 | -0.0102 to +0.0102 | -2.0 to +2.0 | progress no; slow no |

## 6. Scenes turning more than 45 deg (61 scenes, 20 logs; logged 4 s future of the scene's navtest token)

| driver | mean | zeros | collision / offroad / corridor |
|:--|--:|--:|:--|
| P2H10-F-s0 | 0.8143 | 10 | 2 / 5 / 3 |
| P2H10-F-s1 | 0.8544 | 8 | 1 / 5 / 2 |
| P2H10-F-s2 | 0.8682 | 7 | 1 / 5 / 1 |
| P2H10-F-s3 | 0.8443 | 8 | 2 / 4 / 2 |
| P2H10S-F-s0 | 0.8233 | 10 | 1 / 5 / 4 |
| P2H10S-F-s1 | 0.7578 | 14 | 2 / 7 / 5 |
| P2H10S-F-s2 | 0.8244 | 10 | 1 / 5 / 4 |
| P2H10S-F-s3 | 0.8453 | 9 | 1 / 4 / 4 |
| P2H10B-F-s0 | 0.8298 | 9 | 1 / 5 / 3 |
| P2H10B-F-s1 | 0.7973 | 11 | 1 / 6 / 4 |

Arm minus base, four seeds paired by seed index: mean difference -0.0326 [-0.0858, +0.0200] (CI by log); zeros per pair +2.50 (arm 43 vs base 33, classes 5 / 21 / 17 vs 6 / 19 / 8). Null, same bucket: mean difference N1 -0.0539 to +0.0539, N2 -0.0320 to +0.0320; zeros per pair N1 -3.00 to +3.00, N2 -1.50 to +1.50. Outside the base's spread: mean no, zeros no.
