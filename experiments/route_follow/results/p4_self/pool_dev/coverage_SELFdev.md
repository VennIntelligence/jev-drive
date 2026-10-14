# Pool on1_SELFdev: gates and coverage (rf_selfpool.py pack)

| gate | value | pass |
|:--|:--|:--|
| G-in (replayed inputs = served: PAI_DUMP files / drive.jsonl records) | dumps 0 / 0, records 6357 / 6357 | True |
| G-route (regenerated first route point vs the logged route0 <= 0.05 m) | 0.978978978978979 of 666 rows, max 0.102 m | False |
| G-rmem (route memory input from the logged messages vs the served rmem) | max abs diff 0.00011688843369483948 over 666 rows | True |
| G-road (demo crosses a road-edge segment; <= 1 %) | 0.0000 (served plan 0.0000) | True |
| G-speed (demo 4 s chord vs the controller trace within 10 %; >= 99 %) | 1.0000, median ratio 1.0000 | True |
| G-r1 (pool tokens / ego bit-identical to the cache) | front True (256 rows), ego True, feat f16 max diff 9.77e-04 | True |
| G-shape (decision 293 on the branch's realized track) | 17 / 32 branches pass; fails: kmax 6, dk 8, alat 7, v_entry 7; rows dropped 607 | - |
| G-crawl (approach rows: min(v0, future segment speeds before the entry) >= 0.5 v_e) | rows dropped 370 | - |

Rows 666 from 13 branches over 6 scenes (turns). Route memory input: logged; regenerated memory vs served: 0.338 of rows differ by > 0.02 (p99 0.095).

| coverage | rows |
|:--|:--|
| phase | inside 316, exit 180, approach 170 |
| D | D10 334, D0 332 |
| cell | g1-s1 208, g1.25-s1 136, g1-s0.6 108, g1-s0.8 103, g1.5-s0.8 58, g1.5-s0.6 53 |
| src | log 666 |
| push | off 439, on 227 |
| topo | straight_exit 468, no_straight_exit 117, straight_road 81 |
| part | hard_train 468, hard_exam 198 |
| scene | clipgt-eaba3f6a-3629-42a6-aa86-4bf8dd1f569b 387, clipgt-2f227f2c-95e9-4098-a9b1-01d64c697226 81, clipgt-94c1d722-1fce-46b4-9aa9-c78389fae1ac 59, clipgt-fd01e9ab-05d6-445a-b974-02a33793ab49 58, clipgt-f9e361aa-5e93-41bd-aa5c-406999a5b9db 47, clipgt-a174faed-c8e1-4b48-b85e-fecdff7ccca9 34 |
| d_entry | > 40 m 0, 40 .. 20 0, 20 .. 0 170, inside / exit (<= 0) 496 |

| branch | cell | D | src | h | k0 .. k1 | shape kmax / alat / dk / v_entry | G-shape | window | usable | crawl | rows |
|:--|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|--:|
| 2f227f2c~019fed5c | g1.5-s0.8 | D10 | log | 53 | 53 .. 141 | 0.0646 / 2.6479 / 0.0134 / 4.5974 | pass | 89 | 37 | 37 | 0 |
| 2f227f2c~0e2f2486 | g1.25-s1 | D0 | log | 74 | 74 .. 116 | 0.1083 / 3.691 / 0.0241 / 5.5001 | pass | 43 | 43 | 0 | 43 |
| 2f227f2c~22f14334 | g1.5-s0.6 | D0 | log | 74 | 74 .. 192 | 0.2018 / 1.3758 / 0.0797 / 0.6756 | fail: kmax,dk | 119 | 85 | 62 | 0 |
| 2f227f2c~2fd5e8c2 | g1-s0.8 | D10 | log | 53 | 53 .. 142 | 0.0639 / 2.6771 / 0.0117 / 4.5031 | pass | 90 | 37 | 37 | 0 |
| 2f227f2c~90c6b02c | g1-s1 | D10 | log | 53 | 53 .. 103 | 0.0572 / 3.4091 / 0.0064 / 7.1556 | fail: v_entry | 51 | 51 | 0 | 0 |
| 2f227f2c~9a3f4926 | g1-s0.6 | D10 | log | 53 | 53 .. 148 | 0.0637 / 2.6098 / 0.0132 / 5.7615 | pass | 96 | 31 | 31 | 0 |
| 2f227f2c~aa61e126 | g1.5-s1 | D0 | log | 74 | 74 .. 112 | 0.167 / 4.9482 / 0.0475 / 5.5555 | fail: dk,alat | 39 | 39 | 0 | 0 |
| 2f227f2c~c62a302a | g1.25-s1 | D10 | log | 53 | 53 .. 103 | 0.1169 / 5.6779 / 0.041 / 6.9716 | fail: dk,alat,v_entry | 51 | 51 | 0 | 0 |
| 2f227f2c~da2efc16 | g1.5-s0.6 | D10 | log | 53 | 53 .. 148 | 0.0637 / 2.6224 / 0.0122 / 5.7925 | pass | 96 | 31 | 31 | 0 |
| 2f227f2c~dcb51d54 | g1-s1 | D0 | log | 74 | 74 .. 111 | 0.0567 / 2.4449 / 0.0098 / 5.5011 | pass | 38 | 38 | 0 | 38 |
| 302c5c99~4cca4946 | g1.5-s0.8 | D0 | log | 65 | 65 .. 116 | 0.2911 / 2.4987 / 0.0719 / 2.0857 | fail: kmax,dk | 52 | 52 | 24 | 0 |
| 302c5c99~a8d7f55c | g1.5-s1 | D0 | log | 65 | 65 .. 108 | 0.2868 / 3.9531 / 0.0537 / 3.6942 | fail: kmax,dk | 44 | 44 | 0 | 0 |
| 382a7247~7773f93c | g1-s0.8 | D0 | log | 42 | 42 .. 123 | 0.0859 / 2.1393 / 0.0248 / 6.2071 | fail: v_entry | 82 | 82 | 0 | 0 |
| 382a7247~8f2d3e98 | g1-s1 | D10 | log | 28 | 28 .. 79 | 0.0483 / 4.659 / 0.0042 / 8.607 | fail: alat,v_entry | 52 | 52 | 0 | 0 |
| 382a7247~db292a02 | g1-s1 | D0 | log | 42 | 42 .. 79 | 0.0487 / 4.5609 / 0.0044 / 8.5874 | fail: alat,v_entry | 38 | 38 | 0 | 0 |
| 3ce13eed~0e31164c | g1.25-s1 | D0 | log | 53 | 53 .. 96 | 0.1385 / 4.686 / 0.0178 / 3.3464 | fail: alat | 44 | 44 | 0 | 0 |
| 94c1d722~9704040c | g1.25-s1 | D0 | log | 66 | 66 .. 124 | 0.0842 / 2.9087 / 0.0118 / 3.5621 | pass | 59 | 59 | 0 | 59 |
| a174faed~99afad82 | g1.25-s1 | D0 | log | 96 | 96 .. 129 | 0.1534 / 4.2578 / 0.0241 / 4.5336 | pass | 34 | 34 | 0 | 34 |
| a174faed~e93c1b6c | g1.5-s1 | D10 | log | 76 | 76 .. 128 | 0.242 / 7.5109 / 0.0658 / 5.4396 | fail: kmax,dk,alat | 53 | 53 | 0 | 0 |
| e37aa159~587431b0 | g1.25-s1 | D10 | log | 117 | 117 .. 170 | 0.0522 / 3.8936 / 0.0049 / 8.5236 | fail: v_entry | 54 | 41 | 0 | 0 |
| e37aa159~9f8ccb90 | g1.25-s1 | D0 | log | 130 | 130 .. 172 | 0.0755 / 5.7413 / 0.0105 / 8.6809 | fail: alat,v_entry | 43 | 28 | 0 | 0 |
| eaba3f6a~17adbeb6 | g1.5-s0.8 | D10 | log | 56 | 56 .. 155 | 0.0944 / 3.7154 / 0.0149 / 4.4306 | pass | 100 | 100 | 42 | 58 |
| eaba3f6a~46d70b40 | g1-s0.8 | D10 | log | 56 | 56 .. 153 | 0.0914 / 3.7505 / 0.0155 / 4.478 | pass | 98 | 98 | 42 | 56 |
| eaba3f6a~ab202be6 | g1-s1 | D10 | log | 56 | 56 .. 122 | 0.0698 / 3.59 / 0.0071 / 5.5839 | pass | 67 | 67 | 0 | 67 |
| eaba3f6a~b03036fe | g1-s0.6 | D10 | log | 56 | 56 .. 161 | 0.0961 / 3.6629 / 0.0136 / 5.0611 | pass | 106 | 102 | 49 | 53 |
| eaba3f6a~f2b2ac74 | g1-s1 | D0 | log | 90 | 90 .. 134 | 0.0811 / 3.6295 / 0.0137 / 4.5224 | pass | 45 | 45 | 0 | 45 |
| eaba3f6a~f8c1de04 | g1.5-s0.6 | D10 | log | 56 | 56 .. 161 | 0.0974 / 3.7012 / 0.0139 / 5.0945 | pass | 106 | 102 | 49 | 53 |
| eaba3f6a~fb4dd8a0 | g1-s0.6 | D0 | log | 90 | 90 .. 172 | 0.096 / 2.9802 / 0.0181 / 1.1622 | pass | 83 | 68 | 13 | 55 |
| f9e361aa~59457b5e | g1-s0.8 | D10 | log | 53 | 53 .. 138 | 0.0944 / 3.051 / 0.012 / 3.0291 | pass | 86 | 86 | 39 | 47 |
| fd01e9ab~0add5286 | g1-s1 | D0 | log | 61 | 61 .. 118 | 0.1253 / 1.6871 / 0.0249 / 3.9153 | pass | 58 | 58 | 0 | 58 |
| fd01e9ab~53a5c026 | g1.5-s0.6 | D0 | log | 61 | 61 .. 166 | 0.2811 / 1.6007 / 0.0885 / 0.9737 | fail: kmax,dk | 106 | 42 | 42 | 0 |
| fd01e9ab~a94d0754 | g1-s0.8 | D0 | log | 61 | 61 .. 153 | 0.2322 / 1.3297 / 0.071 / 1.6789 | fail: kmax,dk | 93 | 49 | 16 | 0 |
