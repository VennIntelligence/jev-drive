# ONPOL-LONG Amendment 12: candidate-scoring feasibility on the zero events of YS-L18-s1 (held 452, 18 decisions)

43 events read (0 dropped: []); 253 decisions, 71599 candidates (283 per decision: p0, 31 local variants, 256 vocabulary trajectories of the speed bin).

**Gate 1 (recoverable, line >= 60 %)**: 42 / 43 = 0.98 -> pass.  **Gate 2 (head ranks a safe progressing candidate above p0, line >= 50 % of the recoverable)**: 10 / 42 = 0.24 -> FAIL.  **Line: STOP**.

| kind | pooled AUC (candidate commits it vs not; collision: agent logit, off-raster: boundary logit, corridor: risk r) | positives / candidates | decisions with both | median per-decision AUC |
|:--|--:|--:|--:|--:|
| collision | 0.680 | 16685 / 71599 | 148 | 0.727 |
| off-raster | 0.897 | 30482 / 71599 | 251 | 0.943 |
| corridor | 0.733 | 37265 / 71599 | 248 | 0.851 |

Per event: window decisions with the count of safe and progressing candidates; k_first = earliest decision with one; k_r = the latest such decision at or before the onset k_on (E1 alarm, else k_ev); at k_r the head's pick (lowest risk among candidates with own arc >= 0.8 x p0's).

| scene | type | k_ev | k_on (E1) | safe+progress counts over the window | k_first | recoverable | k_r | p0 safe+prog at k_r | head pick | pick safe+prog | r(pick) / r(p0) | safe above p0 | AUC r (unsafe vs safe+prog) |
|:--|:--|--:|:--|:--|--:|:--|--:|:--|:--|:--|:--|--:|--:|
| 8051c60f | collision | 9 | 9 | k5:36 k6:20 k7:8 k8:1 k9:0 | 5 | True | 8 | False | voc128 | False | 0.00 / 0.00 | 0 | 0.443 |
| 0722adec | collision | 10 | 10 | k7:14 k8:12 k9:7 k10:0 | 7 | True | 9 | False | voc248 | False | 0.01 / 0.02 | 0 | 0.511 |
| daf38cec | collision | 11 | 8 (E1) | k4:1 k5:0 k6:0 k7:0 k8:0 k9:0 k10:0 | 4 | True | 4 | False | voc158 | False | 0.00 / 0.93 | 1 | 0.287 |
| bd70654a | collision | 14 | 14 | k10:15 k11:19 k12:17 k13:0 k14:0 | 10 | True | 12 | False | voc160 | False | 0.00 / 0.13 | 11 | 0.731 |
| 84b38cda | collision | 14 | 14 | k10:2 k11:0 k12:0 k13:0 k14:0 | 10 | True | 10 | False | lat+1.5 | False | 0.01 / 0.16 | 2 | 0.530 |
| 47e0ff3b | collision | 16 | 16 | k12:5 k13:4 k14:0 k15:0 k16:0 | 12 | True | 13 | False | voc87 | False | 0.22 / 0.88 | 4 | 0.844 |
| 7bbb6cfc | collision | 17 | 17 | k13:51 k14:32 k15:3 k16:0 k17:0 | 13 | True | 15 | False | voc248 | False | 0.01 / 0.17 | 0 | 0.099 |
| b24575ef | collision | 17 | 17 | k13:8 k14:22 k15:13 k16:0 k17:0 | 13 | True | 15 | False | voc149 | False | 0.23 / 0.98 | 13 | 0.449 |
| bf00c76f | collision | 17 | 17 | k13:0 k14:0 k15:0 k16:0 k17:0 | None | False | - | - | - | - | - | - | - |
| aec02ec2 | corridor | 8 | 2 (E1) | k0:38 k1:64 k2:52 k3:27 k4:12 k5:4 k6:1 k7:0 k8:0 | 0 | True | 2 | False | voc198 | False | 0.00 / 0.02 | 7 | 0.350 |
| e14711ef | corridor | 8 | 3 (E1) | k0:38 k1:39 k2:14 k3:5 k4:1 k5:0 k6:0 k7:0 k8:0 | 0 | True | 3 | False | voc111 | False | 0.07 / 0.76 | 4 | 0.274 |
| e6056b57 | corridor | 9 | 2 (E1) | k0:94 k1:115 k2:97 k3:65 k4:75 k5:62 k6:0 k7:0 k8:0 k9:0 | 0 | True | 2 | True | voc191 | False | 0.00 / 0.00 | 27 | 0.406 |
| 8eefb96f | corridor | 10 | 7 (E1) | k3:17 k4:13 k5:9 k6:22 k7:18 k8:12 k9:6 k10:2 | 3 | True | 7 | False | v1.3 | False | 0.00 / 0.00 | 0 | 0.588 |
| a3882e6a | corridor | 10 | 7 (E1) | k3:94 k4:49 k5:7 k6:3 k7:10 k8:8 k9:0 k10:0 | 3 | True | 7 | False | yaw-5 | False | 0.01 / 0.22 | 1 | 0.496 |
| 47da19c4 | corridor | 11 | 6 (E1) | k2:79 k3:73 k4:36 k5:24 k6:13 k7:1 k8:0 k9:0 k10:0 k11:0 | 2 | True | 6 | False | voc146 | False | 0.00 / 0.16 | 1 | 0.038 |
| d7b1349f | corridor | 11 | 9 (E1) | k5:44 k6:28 k7:17 k8:0 k9:0 k10:0 k11:0 | 5 | True | 7 | False | voc198 | False | 0.00 / 0.03 | 0 | 0.019 |
| 02b68b9c | corridor | 11 | 6 (E1) | k2:90 k3:86 k4:77 k5:67 k6:78 k7:109 k8:72 k9:3 k10:0 k11:0 | 2 | True | 6 | False | voc149 | True | 0.00 / 0.99 | 77 | 0.664 |
| a3dd6d53 | corridor | 13 | 9 (E1) | k5:51 k6:36 k7:18 k8:20 k9:8 k10:7 k11:4 k12:1 k13:0 | 5 | True | 9 | False | voc65 | False | 0.01 / 0.03 | 0 | 0.342 |
| b628faa4 | corridor | 13 | 9 (E1) | k5:54 k6:44 k7:37 k8:19 k9:8 k10:2 k11:0 k12:0 k13:0 | 5 | True | 9 | False | voc103 | False | 0.00 / 0.01 | 0 | 0.019 |
| ded4ce0e | corridor | 14 | 10 (E1) | k6:57 k7:70 k8:65 k9:60 k10:43 k11:9 k12:1 k13:1 k14:0 | 6 | True | 10 | False | voc89 | True | 0.00 / 0.39 | 37 | 0.835 |
| 4faa4706 | corridor | 14 | 8 (E1) | k4:53 k5:44 k6:45 k7:29 k8:22 k9:17 k10:18 k11:8 k12:2 k14:8 | 4 | True | 8 | False | v0.85 | False | 0.00 / 0.00 | 0 | 0.068 |
| 94652fa0 | corridor | 15 | 13 (E1) | k9:22 k10:2 k11:3 k12:2 k13:0 k14:0 k15:0 | 9 | True | 12 | False | lat+1 | False | 0.00 / 0.17 | 0 | 0.118 |
| 5ddfd1fe | corridor | 15 | 15 | k11:4 k12:2 k13:14 k14:4 k15:2 | 11 | True | 15 | False | voc95 | False | 0.00 / 0.02 | 0 | 0.317 |
| f42258b6 | corridor | 16 | 16 | k12:60 k13:68 k14:66 k15:62 k16:56 | 12 | True | 16 | True | voc193 | True | 0.00 / 0.45 | 33 | 0.682 |
| b7c35b12 | corridor | 16 | 16 | k12:31 k13:28 k14:29 k15:53 k16:34 | 12 | True | 16 | True | v0.85 | True | 0.00 / 0.00 | 1 | 0.347 |
| 330166de | corridor | 17 | 17 | k13:54 k14:52 k15:48 k16:53 k17:37 | 13 | True | 17 | True | voc149 | False | 0.00 / 0.01 | 7 | 0.635 |
| f3512ae1 | corridor | 17 | 16 (E1) | k12:25 k13:17 k14:13 k15:5 k16:4 k17:0 | 12 | True | 16 | False | voc191 | False | 0.00 / 0.04 | 0 | 0.604 |
| 0fa603b5 | corridor | 17 | 17 (E1) | k13:11 k14:13 k15:16 k16:31 k17:30 | 13 | True | 17 | False | voc56 | False | 0.00 / 0.00 | 0 | 0.602 |
| f1b802f6 | offroad | 0 | 0 | k0:13 | 0 | True | 0 | False | voc102 | False | 0.01 / 0.02 | 0 | 0.341 |
| da6d3b68 | offroad | 0 | 0 | k0:35 | 0 | True | 0 | True | voc181 | True | 0.00 / 0.00 | 12 | 0.896 |
| 64a7186a | offroad | 0 | 0 | k0:101 | 0 | True | 0 | False | voc33 | True | 0.12 / 0.81 | 51 | 0.695 |
| 9a377868 | offroad | 0 | 0 | k0:31 | 0 | True | 0 | True | v0.85 | False | 0.00 / 0.00 | 2 | 0.911 |
| be6f90f0 | offroad | 5 | 5 | k1:32 k2:70 k3:21 k4:13 k5:12 | 1 | True | 5 | True | voc101 | False | 0.00 / 0.07 | 9 | 0.420 |
| 02015675 | offroad | 7 | 7 | k3:13 k4:17 k5:15 k6:11 | 3 | True | 6 | False | voc83 | False | 0.01 / 0.38 | 11 | 0.539 |
| ddc1271e | offroad | 7 | 7 | k3:17 k4:33 k5:30 k6:19 k7:25 | 3 | True | 7 | False | voc119 | True | 0.06 / 0.14 | 6 | 0.697 |
| f9ed38d9 | offroad | 7 | 7 | k3:8 k4:10 k5:28 k6:12 k7:8 | 3 | True | 7 | False | voc133 | False | 0.39 / 0.98 | 0 | 0.071 |
| 078bc102 | offroad | 9 | 7 (E1) | k3:8 k4:9 k5:2 k6:1 k8:0 k9:0 | 3 | True | 6 | False | voc187 | False | 0.22 / 0.99 | 1 | 0.287 |
| 8c9e3fd9 | offroad | 10 | 10 | k6:15 k7:11 k8:3 k9:5 k10:0 | 6 | True | 9 | False | voc45 | True | 0.00 / 0.76 | 4 | 0.591 |
| fde7abb6 | offroad | 11 | 11 | k7:3 k8:0 k9:0 k10:0 k11:0 | 7 | True | 7 | False | voc115 | False | 0.04 / 0.83 | 0 | 0.030 |
| 53f8bc40 | offroad | 12 | 10 (E1) | k6:72 k7:12 k8:17 k9:5 k10:12 k11:11 k12:6 | 6 | True | 10 | False | lat-1.5 | True | 0.00 / 0.01 | 7 | 0.914 |
| c687799d | offroad | 14 | 14 | k10:15 k11:20 k12:30 k13:37 k14:32 | 10 | True | 14 | True | lat-0.5 | True | 0.00 / 0.00 | 2 | 0.924 |
| 152679f2 | offroad | 16 | 16 | k12:63 k13:24 k14:30 k15:21 k16:15 | 12 | True | 16 | False | voc60 | False | 0.00 / 0.00 | 0 | 0.378 |
| 667ea4f7 | offroad | 17 | 17 | k13:23 k14:22 k15:21 k16:15 k17:7 | 13 | True | 17 | False | voc185 | False | 0.00 / 0.00 | 0 | 0.199 |

Open-loop counterfactuals over a 4 s horizon with logged (non-reactive) traffic; labels from the simulator's actor boxes, the road-and-lane raster (t >= 1 s) and the logged path (4 m, t >= 1 s); the head (d223, two seeds, no retraining) was trained on P2H10 student plans.
