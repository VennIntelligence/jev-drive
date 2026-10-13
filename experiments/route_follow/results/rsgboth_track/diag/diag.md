# RSGboth extra zeros: diagnosis (held 460 @18, rf_rsgbv.py diag)

## RSGboth-L18-s0-rtgate vs RSG-L18-s0

zeros 41 vs 26; extra in A 22, extra in B 7, shared 19

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 22 | 8 / 7 / 7 | 6 / 12 / 4 | 1 | 8 / 9 | 3.3 (2.0) |
| extra B | 7 | 3 / 3 / 1 | 4 / 2 / 1 | 0 | 1 / 1 | 3.0 (4.4) |
| shared (A) | 19 | 13 / 2 / 4 | 5 / 9 / 5 | 1 | 0 / 2 | 4.5 (4.3) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2121 | 4.99 / 4.11 / 4.83 | +0.87 [+0.72, +1.03] | +0.16 / -0.71 | 0.17 / 0.07 |
| junction | 728 | 5.52 / 5.15 / 5.66 | +0.37 [+0.18, +0.56] | -0.14 / -0.51 | 0.14 / 0.10 |
| straight | 3512 | 7.20 / 6.24 / 7.25 | +0.97 [+0.83, +1.11] | -0.05 / -1.02 | 0.12 / 0.04 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.06.28.16.29.11_veh-38_03263_03766-b87c15c9f6cf54eb collision curve k 12 v 1.3 / 0.8 lead 1 box 0 turn 0 hit static False dpsi -1 x 6.8 y -0.1
- 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e collision curve k 15 v 2.5 / 0.9 lead 1 box 1 turn 0 hit static False dpsi 12 x 8.9 y -0.7
- 2021.05.25.17.54.41_veh-35_04111_04288-8051c60f20705909 collision straight k 14 v 5.7 / 3.1 lead 1 box 12 turn 0 hit static False dpsi -6 x 5.1 y 2.5
- 2021.06.03.12.02.06_veh-35_00233_00609-abe4fa26de85552d collision straight k 17 v 10.1 / 7.5 lead 1 box 2 turn 0
- 2021.06.03.17.06.58_veh-35_02571_02742-9a89dccc70835d69 collision straight k 17 v 1.3 / 0.1 lead 0 box 1 turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-67077701cdd85c9d collision straight k 15 v 6.1 / 3.8 lead 0 box nan turn 0 hit static False dpsi -175 x 19.0 y 0.9
- 2021.10.06.08.16.17_veh-52_00612_00782-b5668089b793502b collision straight k 16 v 0.8 / 0.1 lead 1 box 0 turn 0 hit static False dpsi -0 x 6.6 y -0.0
- 2021.06.28.16.29.11_veh-38_03263_03766-330166def5a35f4d corridor curve k 17 v 1.7 / 0.2 lead 1 box 1 turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-b7c35b12755c5d44 corridor curve k 12 v 3.1 / 0.1 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-ded4ce0e91b4596a corridor curve k 17 v 1.7 / 0.3 lead 1 box 1 turn 0
- 2021.09.29.15.23.04_veh-28_00601_00802-fc0a60a9d8245aea corridor curve k 17 v 1.9 / 0.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00922_01296-d81c4398a40b52b9 corridor curve k 16 v 2.9 / 0.3 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-9b1bea0cc0d75583 corridor junction k 13 v 7.4 / 4.0 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-0fa603b5789956f1 corridor junction k 17 v 4.7 / 6.0 lead 0 box nan turn 0
- 2021.05.25.17.54.41_veh-35_02723_02902-f42258b638fc5b64 offroad curve k 14 v 1.7 / 0.0 lead 1 box 4 turn 0
- 2021.09.16.17.40.09_veh-45_02539_02745-661ce644db6d5546 offroad curve k 12 v 5.6 / 4.7 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-caa3170f8c3b5cfe offroad curve k 8 v 3.4 / 2.2 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc offroad curve k 5 v 1.6 / 1.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-a67f68a6c9e85668 offroad curve k 15 v 7.6 / 4.9 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-f2592c08589e5398 offroad junction k 11 v 5.3 / 4.7 lead 0 box nan turn 1
- 2021.10.06.08.16.17_veh-52_01590_01725-320446eca62c59fe offroad junction k 16 v 8.0 / 7.0 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00032_00186-83e895f722575452 offroad straight k 17 v 4.8 / 6.0 lead 0 box nan turn 0

## RSGboth-L18-s0-rtgatep vs RSG-L18-s0

zeros 39 vs 26; extra in A 21, extra in B 8, shared 18

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 21 | 7 / 7 / 7 | 6 / 11 / 4 | 0 | 8 / 9 | 3.1 (1.7) |
| extra B | 8 | 4 / 3 / 1 | 4 / 2 / 2 | 0 | 1 / 1 | 3.5 (4.4) |
| shared (A) | 18 | 13 / 1 / 4 | 4 / 9 / 5 | 1 | 0 / 1 | 4.6 (4.2) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2129 | 4.98 / 4.11 / 4.83 | +0.87 [+0.72, +1.03] | +0.16 / -0.72 | 0.17 / 0.07 |
| junction | 728 | 5.53 / 5.15 / 5.65 | +0.38 [+0.18, +0.56] | -0.13 / -0.50 | 0.14 / 0.10 |
| straight | 3514 | 7.20 / 6.24 / 7.25 | +0.97 [+0.83, +1.11] | -0.05 / -1.02 | 0.12 / 0.04 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.06.28.16.29.11_veh-38_03263_03766-b87c15c9f6cf54eb collision curve k 12 v 1.3 / 0.8 lead 1 box 0 turn 0 hit static False dpsi -1 x 6.8 y -0.1
- 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e collision curve k 15 v 2.5 / 0.9 lead 1 box 1 turn 0 hit static False dpsi 12 x 8.9 y -0.7
- 2021.05.25.17.54.41_veh-35_04111_04288-8051c60f20705909 collision straight k 14 v 5.7 / 3.1 lead 1 box 12 turn 0 hit static False dpsi -6 x 5.1 y 2.5
- 2021.06.03.12.02.06_veh-35_00233_00609-abe4fa26de85552d collision straight k 17 v 10.1 / 7.5 lead 1 box 2 turn 0
- 2021.06.03.17.06.58_veh-35_02571_02742-9a89dccc70835d69 collision straight k 17 v 1.3 / 0.1 lead 0 box 1 turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-67077701cdd85c9d collision straight k 15 v 6.1 / 3.8 lead 0 box nan turn 0 hit static False dpsi -175 x 19.0 y 0.9
- 2021.10.06.08.16.17_veh-52_00612_00782-b5668089b793502b collision straight k 16 v 0.8 / 0.1 lead 1 box 0 turn 0 hit static False dpsi -0 x 6.6 y -0.0
- 2021.06.28.16.29.11_veh-38_03263_03766-330166def5a35f4d corridor curve k 17 v 1.7 / 0.2 lead 1 box 1 turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-b7c35b12755c5d44 corridor curve k 12 v 3.1 / 0.1 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-ded4ce0e91b4596a corridor curve k 17 v 1.7 / 0.3 lead 1 box 1 turn 0
- 2021.09.29.15.23.04_veh-28_00601_00802-fc0a60a9d8245aea corridor curve k 17 v 1.9 / 0.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00922_01296-d81c4398a40b52b9 corridor curve k 16 v 2.9 / 0.3 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-9b1bea0cc0d75583 corridor junction k 13 v 7.4 / 4.0 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-0fa603b5789956f1 corridor junction k 17 v 4.7 / 6.0 lead 0 box nan turn 0
- 2021.05.25.17.54.41_veh-35_02723_02902-f42258b638fc5b64 offroad curve k 14 v 1.7 / 0.0 lead 1 box 4 turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-caa3170f8c3b5cfe offroad curve k 8 v 3.4 / 2.2 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc offroad curve k 5 v 1.6 / 1.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-a67f68a6c9e85668 offroad curve k 15 v 7.6 / 4.9 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-727ed49214315cd6 offroad junction k 14 v 5.2 / 5.4 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-320446eca62c59fe offroad junction k 16 v 8.0 / 7.0 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00032_00186-83e895f722575452 offroad straight k 17 v 4.8 / 6.0 lead 0 box nan turn 0

## RSGboth-L18-s0-rttimep vs RSG-L18-s0

zeros 41 vs 26; extra in A 21, extra in B 6, shared 20

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 21 | 7 / 7 / 7 | 6 / 12 / 3 | 0 | 8 / 9 | 3.1 (1.7) |
| extra B | 6 | 3 / 2 / 1 | 3 / 2 / 1 | 0 | 0 / 0 | 2.8 (4.1) |
| shared (A) | 20 | 13 / 3 / 4 | 6 / 9 / 5 | 1 | 2 / 3 | 4.6 (4.3) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2123 | 4.98 / 4.11 / 4.83 | +0.87 [+0.72, +1.02] | +0.16 / -0.71 | 0.17 / 0.07 |
| junction | 726 | 5.54 / 5.13 / 5.64 | +0.41 [+0.22, +0.60] | -0.10 / -0.50 | 0.14 / 0.10 |
| straight | 3518 | 7.21 / 6.24 / 7.25 | +0.97 [+0.83, +1.11] | -0.05 / -1.02 | 0.12 / 0.04 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.06.28.16.29.11_veh-38_03263_03766-b87c15c9f6cf54eb collision curve k 12 v 1.3 / 0.8 lead 1 box 0 turn 0 hit static False dpsi -1 x 6.8 y -0.1
- 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e collision curve k 15 v 2.5 / 0.9 lead 1 box 1 turn 0 hit static False dpsi 12 x 8.9 y -0.7
- 2021.05.25.17.54.41_veh-35_04111_04288-8051c60f20705909 collision straight k 14 v 5.7 / 3.1 lead 1 box 12 turn 0 hit static False dpsi -6 x 5.1 y 2.5
- 2021.06.03.12.02.06_veh-35_00233_00609-abe4fa26de85552d collision straight k 17 v 10.1 / 7.5 lead 1 box 2 turn 0
- 2021.06.03.17.06.58_veh-35_02571_02742-9a89dccc70835d69 collision straight k 17 v 1.3 / 0.1 lead 0 box 1 turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-67077701cdd85c9d collision straight k 15 v 6.1 / 3.8 lead 0 box nan turn 0 hit static False dpsi -175 x 19.0 y 0.9
- 2021.10.06.08.16.17_veh-52_00612_00782-b5668089b793502b collision straight k 16 v 0.8 / 0.1 lead 1 box 0 turn 0 hit static False dpsi -0 x 6.6 y -0.0
- 2021.06.28.16.29.11_veh-38_03263_03766-330166def5a35f4d corridor curve k 17 v 1.7 / 0.2 lead 1 box 1 turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-b7c35b12755c5d44 corridor curve k 12 v 3.1 / 0.1 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-ded4ce0e91b4596a corridor curve k 17 v 1.7 / 0.3 lead 1 box 1 turn 0
- 2021.09.29.15.23.04_veh-28_00601_00802-fc0a60a9d8245aea corridor curve k 17 v 1.9 / 0.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00922_01296-d81c4398a40b52b9 corridor curve k 16 v 2.9 / 0.3 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-9b1bea0cc0d75583 corridor junction k 13 v 7.4 / 4.0 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-0fa603b5789956f1 corridor junction k 17 v 4.7 / 6.0 lead 0 box nan turn 0
- 2021.05.25.17.54.41_veh-35_02723_02902-f42258b638fc5b64 offroad curve k 14 v 1.6 / 0.0 lead 1 box 4 turn 0
- 2021.09.16.17.40.09_veh-45_02539_02745-661ce644db6d5546 offroad curve k 12 v 5.8 / 4.7 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-caa3170f8c3b5cfe offroad curve k 8 v 3.4 / 2.2 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc offroad curve k 5 v 1.6 / 1.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-a67f68a6c9e85668 offroad curve k 15 v 7.6 / 4.9 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-320446eca62c59fe offroad junction k 16 v 8.0 / 7.0 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00032_00186-83e895f722575452 offroad straight k 17 v 5.0 / 6.0 lead 0 box nan turn 0

## RSG-L18-s0-rtgatep vs RSG-L18-s0

zeros 28 vs 26; extra in A 6, extra in B 4, shared 22

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 6 | 4 / 0 / 2 | 3 / 3 / 0 | 0 | 0 / 0 | 3.7 (2.8) |
| extra B | 4 | 2 / 1 / 1 | 1 / 2 / 1 | 0 | 1 / 1 | 2.8 (3.9) |
| shared (A) | 22 | 14 / 2 / 6 | 5 / 10 / 7 | 1 | 0 / 1 | 4.5 (4.2) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2120 | 4.16 / 4.13 / 4.80 | +0.03 [-0.01, +0.08] | -0.64 / -0.67 | 0.06 / 0.07 |
| junction | 765 | 4.97 / 5.21 / 5.69 | -0.24 [-0.31, -0.16] | -0.72 / -0.48 | 0.06 / 0.11 |
| straight | 3526 | 6.18 / 6.18 / 7.22 | +0.00 [-0.04, +0.04] | -1.04 / -1.05 | 0.04 / 0.04 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.09.09.18.04.06_veh-40_00743_01071-1b18327179f15a8f corridor straight k 17 v 4.3 / 7.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-0fa603b5789956f1 corridor straight k 17 v 4.6 / 6.0 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-caa3170f8c3b5cfe offroad curve k 7 v 3.0 / 2.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00032_00170-6b68c09f87b85dc8 offroad curve k 8 v 1.1 / 2.0 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-77424ed925b45c77 offroad curve k 12 v 1.2 / 2.5 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-078bc1027dde5d1a offroad straight k 10 v 5.1 / 3.0 lead 0 box nan turn 0

## RSGboth-L18-s0 vs RSG-L18-s0

zeros 33 vs 26; extra in A 14, extra in B 7, shared 19

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 14 | 5 / 4 / 5 | 4 / 8 / 2 | 0 | 5 / 5 | 4.0 (4.2) |
| extra B | 7 | 4 / 2 / 1 | 3 / 1 / 3 | 0 | 0 / 0 | 4.1 (4.4) |
| shared (A) | 19 | 12 / 3 / 4 | 3 / 10 / 6 | 1 | 1 / 2 | 4.9 (3.9) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2133 | 5.00 / 4.08 / 4.80 | +0.92 [+0.78, +1.06] | +0.20 / -0.72 | 0.17 / 0.07 |
| junction | 735 | 5.69 / 5.15 / 5.63 | +0.54 [+0.39, +0.70] | +0.06 / -0.48 | 0.16 / 0.10 |
| straight | 3534 | 7.27 / 6.24 / 7.25 | +1.03 [+0.91, +1.16] | +0.02 / -1.01 | 0.12 / 0.04 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e collision curve k 16 v 1.9 / 0.6 lead 1 box 0 turn 0 hit static False dpsi 12 x 8.0 y -0.8
- 2021.09.16.14.39.34_veh-42_00297_00935-9a24a904a324581d collision curve k 15 v 3.2 / 5.8 lead 1 box 1 turn 0 hit static False dpsi 4 x 8.5 y -0.2
- 2021.05.25.17.54.41_veh-35_04111_04288-8051c60f20705909 collision straight k 17 v 3.1 / 1.8 lead 1 box 1 turn 0 hit static False dpsi 2 x 9.2 y -1.2
- 2021.06.03.12.02.06_veh-35_00233_00609-abe4fa26de85552d collision straight k 17 v 10.2 / 7.5 lead 1 box 1 turn 0
- 2021.06.28.16.29.11_veh-38_03263_03766-330166def5a35f4d corridor curve k 17 v 1.1 / 0.2 lead 1 box 1 turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-b7c35b12755c5d44 corridor curve k 12 v 3.1 / 0.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00922_01296-d81c4398a40b52b9 corridor curve k 17 v 2.8 / 0.3 lead 0 box nan turn 0
- 2021.09.16.19.49.00_veh-42_00990_01609-9b1bea0cc0d75583 corridor junction k 14 v 4.8 / 3.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-0fa603b5789956f1 corridor straight k 17 v 5.6 / 6.0 lead 0 box nan turn 0
- 2021.09.16.17.40.09_veh-45_02539_02745-661ce644db6d5546 offroad curve k 13 v 5.8 / 4.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc offroad curve k 5 v 1.5 / 1.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-a67f68a6c9e85668 offroad curve k 15 v 7.7 / 4.9 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-320446eca62c59fe offroad junction k 17 v 8.2 / 7.0 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00032_00186-83e895f722575452 offroad straight k 17 v 5.3 / 6.0 lead 0 box nan turn 0

