# Pool on1_SELF4R: gates and coverage (rf_selfpool.py pack)

| gate | value | pass |
|:--|:--|:--|
| G-in (replayed inputs = served: PAI_DUMP files / drive.jsonl records) | dumps 400 / 400, records 3979 / 3979 | True |
| G-route (regenerated first route point vs the logged route0 <= 0.05 m) | 1.0 of 753 rows, max 0.049 m (row filter: True, 10 rows dropped) | True |
| G-rmem (route memory input from the logged messages vs the served rmem) | max abs diff 0.00014129281044006348 over 753 rows | True |
| G-road (demo crosses a road-edge segment; <= 1 %) | 0.0000 (served plan 0.0066) | True |
| G-speed (demo 4 s chord vs the controller trace within 10 %; >= 99 %) | 1.0000, median ratio 1.0000 | True |
| G-r1 (pool tokens / ego bit-identical to the cache) | front True (256 rows), ego True, feat f16 max diff 9.77e-04 | True |
| G-shape (decision 293 on the branch's realized track) | 12 / 20 branches pass; fails: kmax 7, dk 2, alat 3, v_entry 1; rows dropped 487 | - |
| G-crawl (approach rows: min(v0, future segment speeds before the entry) >= 0.5 v_e) | rows dropped 391 | - |

Rows 753 from 12 branches over 8 scenes (turns). Route memory input: logged; regenerated memory vs served: 0.278 of rows differ by > 0.02 (p99 0.031).

| coverage | rows |
|:--|:--|
| phase | inside 370, exit 226, approach 157 |
| D | D40 476, D0 277 |
| cell | g1.25-raw 753 |
| src | own 731, log 22 |
| push | off 506, on 247 |
| topo | curve_single 486, no_straight_exit 190, straight_exit 77 |
| part | train 402, hard_train 351 |
| scene | clipgt-ea468790-9c38-406d-a4cf-4c43fbc091d8 168, clipgt-232d240e-5ccb-41ce-9954-97c336fa45af 157, clipgt-8a365ac2-e5d5-4fd6-903d-497b2225151c 108, clipgt-268949d2-3e20-4972-8785-c15e4fcb5fde 82, clipgt-ddc3e8df-b368-40f0-bb6a-9d9f4f568cd4 79, clipgt-a4be61d6-099b-4380-8c32-ad176781b288 77, clipgt-c6c01d25-ff8a-4cc3-9d33-27598d48dd90 60, clipgt-1e8e939e-932b-4511-a231-0282942cb4ca 22 |
| d_entry | > 40 m 0, 40 .. 20 0, 20 .. 0 157, inside / exit (<= 0) 596 |

| branch | cell | D | src | h | k0 .. k1 | shape kmax / alat / dk / v_entry | G-shape | window | usable | crawl | rows |
|:--|:--|:--|:--|--:|:--|:--|:--|--:|--:|--:|--:|
| 1e8e939e~3b28f3ea | g1.25-raw | D0 | log | 121 | 121 .. 142 | 0.0848 / 2.9706 / 0.0174 / 4.6119 | pass | 22 | 22 | 0 | 22 |
| 213dfdac~095fc108 | g1.25-raw | D0 | own | 111 | 19 .. 166 | 0.227 / 4.8245 / 0.0395 / 3.59 | fail: kmax,alat | 148 | 140 | 56 | 0 |
| 232d240e~86f3ad6e | g1.25-raw | D0 | own | 26 | 0 .. 78 | 0.1453 / 3.4328 / 0.0261 / 4.2337 | pass | 79 | 79 | 0 | 79 |
| 232d240e~ff18a35a | g1.25-raw | D40 | own | 17 | 0 .. 77 | 0.1475 / 3.5987 / 0.0332 / 4.2515 | pass | 78 | 78 | 0 | 78 |
| 268949d2~0c47ede2 | g1.25-raw | D40 | own | 17 | 0 .. 102 | 0.0868 / 2.9685 / 0.0171 / 2.685 | pass | 103 | 103 | 60 | 43 |
| 268949d2~99b0051a | g1.25-raw | D0 | own | 62 | 0 .. 114 | 0.173 / 3.9586 / 0.0294 / 2.6326 | pass | 115 | 99 | 60 | 39 |
| 37d660b1~67347f1a | g1.25-raw | D0 | own | 54 | 0 .. 96 | 0.175 / 6.0851 / 0.0303 / 4.5529 | fail: kmax,alat | 97 | 97 | 0 | 0 |
| 8a365ac2~156c2e6a | g1.25-raw | D40 | own | 17 | 0 .. 96 | 0.1097 / 1.5048 / 0.0324 / 3.4337 | pass | 97 | 97 | 42 | 55 |
| 8a365ac2~5a9efef8 | g1.25-raw | D0 | own | 52 | 0 .. 104 | 0.0922 / 2.3247 / 0.0221 / 3.1662 | pass | 105 | 105 | 42 | 53 |
| 96eb4700~33f928d8 | g1.25-raw | D40 | own | 17 | 0 .. 42 | 0.0654 / 3.4275 / 0.0107 / 6.3871 | fail: v_entry | 43 | 43 | 0 | 0 |
| 98694f91~9b0a7cb0 | g1.25-raw | D0 | own | 128 | 15 .. 186 | 0.1606 / 2.5538 / 0.0289 / 3.3786 | fail: kmax | 172 | 144 | 81 | 0 |
| a4be61d6~faae00b0 | g1.25-raw | D40 | own | 17 | 0 .. 113 | 0.1094 / 1.9376 / 0.0274 / 3.9293 | pass | 114 | 114 | 37 | 77 |
| b40f4729~07d6d9dc | g1.25-raw | D40 | own | 17 | 0 .. 65 | 0.1642 / 4.3981 / 0.032 / 2.9833 | fail: kmax | 66 | 66 | 30 | 0 |
| b40f4729~caeadd6e | g1.25-raw | D0 | own | 34 | 0 .. 66 | 0.1758 / 5.0544 / 0.0331 / 2.9698 | fail: kmax,alat | 67 | 67 | 30 | 0 |
| c6c01d25~92db6214 | g1.25-raw | D40 | own | 17 | 0 .. 59 | 0.0987 / 2.8889 / 0.0149 / 3.7645 | pass | 60 | 60 | 0 | 60 |
| dc1966f4~1238a382 | g1.25-raw | D40 | own | 19 | 19 .. 174 | 0.2152 / 1.4163 / 0.0737 / 3.8554 | fail: kmax,dk | 156 | 140 | 81 | 0 |
| dc1966f4~f8940970 | g1.25-raw | D0 | own | 131 | 19 .. 182 | 0.2107 / 2.022 / 0.077 / 3.5818 | fail: kmax,dk | 164 | 140 | 72 | 0 |
| ddc3e8df~cf8bc19e | g1.25-raw | D40 | own | 17 | 0 .. 78 | 0.1122 / 2.4276 / 0.0177 / 2.5777 | pass | 79 | 79 | 0 | 79 |
| ea468790~69ea542e | g1.25-raw | D0 | own | 96 | 0 .. 165 | 0.1341 / 2.9946 / 0.0334 / 3.2163 | pass | 166 | 159 | 75 | 84 |
| ea468790~f700c4e4 | g1.25-raw | D40 | own | 17 | 0 .. 181 | 0.0633 / 0.9927 / 0.0096 / 3.2281 | pass | 182 | 159 | 75 | 84 |
