# RSGboth extra zeros: diagnosis (held 460 @18, rf_rsgbv.py diag)

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

## RSGboth-L18-s1 vs RSG-L18-s1

zeros 38 vs 26; extra in A 19, extra in B 7, shared 19

| set | n | offroad / collision / corridor | straight / curve / junction | turn scene | lead / box (last 3 dec) | v_e median (log) |
|:--|--:|:--|:--|--:|:--|:--|
| extra A | 19 | 7 / 8 / 4 | 5 / 12 / 2 | 0 | 4 / 5 | 5.3 (3.8) |
| extra B | 7 | 2 / 3 / 2 | 3 / 0 / 4 | 0 | 0 / 0 | 4.2 (6.4) |
| shared (A) | 19 | 12 / 3 / 4 | 5 / 8 / 6 | 1 | 1 / 2 | 4.6 (3.9) |

| road class | decisions | v A / B / log (m/s) | A - B [95 % CI] | A - log / B - log | share > log + 1 m/s A / B |
|:--|--:|:--|:--|:--|:--|
| curve | 2132 | 4.99 / 4.06 / 4.80 | +0.93 [+0.79, +1.08] | +0.19 / -0.74 | 0.16 / 0.06 |
| junction | 743 | 5.68 / 5.08 / 5.63 | +0.60 [+0.45, +0.76] | +0.05 / -0.55 | 0.17 / 0.09 |
| straight | 3518 | 7.29 / 6.26 / 7.25 | +1.03 [+0.91, +1.17] | +0.04 / -0.98 | 0.13 / 0.05 |

Extra zeros of A (scene, type, class, v_e / v_log, lead, box gap):

- 2021.06.28.15.02.02_veh-38_02398_02848-7fc1ce68acb7562c collision curve k 17 v 3.2 / 3.8 lead 1 box 0 turn 0
- 2021.06.28.16.29.11_veh-38_03263_03766-b87c15c9f6cf54eb collision curve k 17 v 2.5 / 1.1 lead 0 box 0 turn 0
- 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e collision curve k 16 v 2.0 / 0.6 lead 1 box 0 turn 0 hit static False dpsi 12 x 7.9 y -0.8
- 2021.10.06.08.16.17_veh-52_01430_01579-b227439005885966 collision curve k 15 v 7.1 / 4.1 lead 0 box nan turn 0 hit static False dpsi -179 x 19.2 y -2.4
- 2021.05.25.17.54.41_veh-35_04111_04288-8051c60f20705909 collision straight k 13 v 5.3 / 3.8 lead 0 box nan turn 0 hit static False dpsi -7 x 7.4 y 2.2
- 2021.06.03.12.02.06_veh-35_00233_00609-abe4fa26de85552d collision straight k 17 v 10.3 / 7.5 lead 1 box 2 turn 0
- 2021.06.28.16.57.59_veh-26_00016_00484-beeb0c78917a528a collision straight k 13 v 11.3 / 13.9 lead 0 box nan turn 0 hit static False dpsi 1 x 2.7 y -2.9
- 2021.10.06.08.16.17_veh-52_00181_00574-67077701cdd85c9d collision straight k 15 v 6.3 / 3.8 lead 0 box nan turn 0 hit static False dpsi -177 x 18.6 y -0.0
- 2021.06.28.16.29.11_veh-38_03263_03766-330166def5a35f4d corridor curve k 17 v 1.3 / 0.2 lead 1 box 2 turn 0
- 2021.09.16.14.39.34_veh-42_00297_00935-b7c35b12755c5d44 corridor curve k 13 v 2.9 / 0.1 lead 0 box nan turn 0
- 2021.09.29.15.23.04_veh-28_00601_00802-fc0a60a9d8245aea corridor curve k 17 v 1.8 / 0.1 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00922_01296-d81c4398a40b52b9 corridor curve k 17 v 2.9 / 0.3 lead 0 box nan turn 0
- 2021.09.09.18.04.06_veh-40_00743_01071-152679f2edce5c73 offroad curve k 16 v 8.6 / 8.4 lead 0 box nan turn 0
- 2021.09.16.17.40.09_veh-45_02539_02745-661ce644db6d5546 offroad curve k 13 v 5.4 / 4.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc offroad curve k 5 v 1.5 / 1.7 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01949_02501-a67f68a6c9e85668 offroad curve k 15 v 7.6 / 4.9 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-320446eca62c59fe offroad junction k 16 v 8.1 / 7.0 lead 0 box nan turn 0
- 2021.10.06.08.16.17_veh-52_01590_01725-9e522849163c53b8 offroad junction k 7 v 7.8 / 6.9 lead 0 box nan turn 0
- 2021.09.16.14.39.34_veh-42_00032_00186-83e895f722575452 offroad straight k 17 v 5.0 / 6.0 lead 0 box nan turn 0

