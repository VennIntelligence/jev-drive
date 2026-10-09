Scenes common to all 5 drivers: 700 from 27 logs, 7 shards. Scene score = AlpaSim scene score. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10 000 draws, seed 0.

- SH30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes: 0
- AP2-AB-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s0: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s1: 700 scored scenes, 3 run dirs, disagreeing duplicate scenes: 0
- WA-JEPA (reference): 700 scored scenes, 3 run dirs, disagreeing duplicate scenes: 0

## Scores

| driver | scenes | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 700 | 0.9140 [0.8845, 0.9366] | 555 | 50 | 19 | 13 | 18 | 0 | 95 | 32 | 0.948 |
| AP2-AB-s0 | 700 | 0.9223 [0.9003, 0.9404] | 598 | 49 | 21 | 12 | 16 | 0 | 53 | 33 | 0.987 |
| OT30-F-s0 | 700 | 0.9258 [0.9029, 0.9447] | 565 | 42 | 11 | 12 | 19 | 0 | 93 | 23 | 0.950 |
| OT30-F-s1 | 700 | 0.9335 [0.9050, 0.9558] | 580 | 38 | 11 | 10 | 17 | 0 | 82 | 21 | 0.959 |
| WA-JEPA (reference) | 700 | 0.9111 [0.8753, 0.9436] | 594 | 56 | 1 | 34 | 21 | 0 | 50 | 35 | 0.928 |

## Paired differences (per scene, log-clustered CI)

| A - B | mean scene score difference [95% CI] | zeros A / B | at-fault collision zeros A / B |
|:--|:--|--:|--:|
| AP2-AB-s0 - SH30-F-s0 | +0.0083 [-0.0077, +0.0254] | 49 / 50 | 21 / 19 |
| OT30-F-s0 - SH30-F-s0 | +0.0118 [-0.0053, +0.0284] | 42 / 50 | 11 / 19 |
| OT30-F-s1 - SH30-F-s0 | +0.0195 [+0.0009, +0.0390] | 38 / 50 | 11 / 19 |
| OT30 mean of 2 seeds - SH30-F-s0 | +0.0156 [-0.0011, +0.0326] | 40 / 50 | 11 / 19 |
| SH30-F-s0 - WA-JEPA (reference) | +0.0029 [-0.0379, +0.0455] | 50 / 56 | 19 / 1 |
| AP2-AB-s0 - WA-JEPA (reference) | +0.0112 [-0.0242, +0.0474] | 49 / 56 | 21 / 1 |
| OT30-F-s0 - WA-JEPA (reference) | +0.0147 [-0.0214, +0.0526] | 42 / 56 | 11 / 1 |
| OT30-F-s1 - WA-JEPA (reference) | +0.0224 [-0.0191, +0.0649] | 38 / 56 | 11 / 1 |
| OT30 mean of 2 seeds - WA-JEPA (reference) | +0.0185 [-0.0198, +0.0587] | 40 / 56 | 11 / 1 |

## OT30 line (pre-registered, plans/2026-10-09-ot30-closedloop-prereg.md)

- mean of the two OT30 seeds minus SH30-F-s0: +0.0156 [-0.0011, +0.0326]; line: >= +0.010 and CI lower bound > 0 -> **not met**
- at-fault collision zeros, mean of the two seeds 11 vs SH30 19: line: not higher -> **met**
- verdict: **not a candidate**
- each seed alone minus SH30: s0 +0.0118 [-0.0053, +0.0284], s1 +0.0195 [+0.0009, +0.0390]
- offroad + left-corridor zeros: SH30 31, OT30 s0 31, s1 27

## Best-of-k among our own drivers (per-scene maximum = an oracle selector, an upper bound)

| set | oracle mean [95% CI] | best single in the set (mean) | oracle minus best single [95% CI] |
|:--|:--|--:|:--|
| SH30-F-s0 + AP2-AB-s0 | 0.9501 [0.9327, 0.9651] | AP2-AB-s0 0.9223 | +0.0278 [+0.0178, +0.0384] |
| SH30-F-s0 + OT30-F-s0 | 0.9488 [0.9266, 0.9663] | OT30-F-s0 0.9258 | +0.0231 [+0.0159, +0.0306] |
| SH30-F-s0 + OT30-F-s1 | 0.9554 [0.9314, 0.9734] | OT30-F-s1 0.9335 | +0.0220 [+0.0125, +0.0330] |
| AP2-AB-s0 + OT30-F-s0 | 0.9599 [0.9433, 0.9736] | OT30-F-s0 0.9258 | +0.0342 [+0.0232, +0.0469] |
| AP2-AB-s0 + OT30-F-s1 | 0.9642 [0.9459, 0.9782] | OT30-F-s1 0.9335 | +0.0307 [+0.0187, +0.0447] |
| OT30-F-s0 + OT30-F-s1 | 0.9468 [0.9234, 0.9654] | OT30-F-s1 0.9335 | +0.0133 [+0.0056, +0.0221] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 | 0.9695 [0.9556, 0.9807] | OT30-F-s0 0.9258 | +0.0437 [+0.0322, +0.0586] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s1 | 0.9725 [0.9584, 0.9838] | OT30-F-s1 0.9335 | +0.0390 [+0.0250, +0.0564] |
| SH30-F-s0 + OT30-F-s0 + OT30-F-s1 | 0.9609 [0.9418, 0.9762] | OT30-F-s1 0.9335 | +0.0274 [+0.0160, +0.0409] |
| AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9691 [0.9543, 0.9808] | OT30-F-s1 0.9335 | +0.0357 [+0.0206, +0.0549] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9756 [0.9640, 0.9852] | OT30-F-s1 0.9335 | +0.0422 [+0.0254, +0.0650] |

Reference for the selector rows: `OT30-F-s0 + OT30-F-s1` is two seeds of one recipe, i.e. what an oracle gains from training-seed noise alone.

## Per shard (mean scene score / zeros)

| shard | scenes | SH30-F-s0 | AP2-AB-s0 | OT30-F-s0 | OT30-F-s1 | WA-JEPA (reference) |
|:--|--:|--:|--:|--:|--:|--:|
| part001 | 100 | 0.9169 / 7 | 0.9101 / 8 | 0.9334 / 5 | 0.9189 / 7 | 0.9777 / 2 |
| part002 | 100 | 0.9250 / 6 | 0.9338 / 6 | 0.9338 / 5 | 0.9471 / 4 | 0.9758 / 2 |
| part003 | 100 | 0.9471 / 4 | 0.9461 / 5 | 0.9495 / 4 | 0.9703 / 2 | 0.9398 / 5 |
| part005 | 100 | 0.9333 / 6 | 0.9439 / 5 | 0.9428 / 5 | 0.9541 / 4 | 0.9305 / 6 |
| part007 | 100 | 0.8267 / 14 | 0.8641 / 12 | 0.8750 / 9 | 0.8606 / 11 | 0.8941 / 9 |
| part008 | 100 | 0.9046 / 8 | 0.9130 / 8 | 0.9093 / 8 | 0.9259 / 6 | 0.8289 / 16 |
| part009 | 100 | 0.9444 / 5 | 0.9452 / 5 | 0.9366 / 6 | 0.9573 / 4 | 0.8307 / 16 |

Part001 against the rest (mean scene score; difference with a log-clustered CI):

| driver | part001 | other shards | part001 - others [95% CI] |
|:--|--:|--:|:--|
| SH30-F-s0 | 0.9169 | 0.9135 | +0.0034 [-0.0716, +0.0503] |
| AP2-AB-s0 | 0.9101 | 0.9243 | -0.0143 [-0.1082, +0.0342] |
| OT30-F-s0 | 0.9334 | 0.9245 | +0.0089 [-0.0776, +0.0501] |
| OT30-F-s1 | 0.9189 | 0.9359 | -0.0170 [-0.0922, +0.0280] |
| WA-JEPA (reference) | 0.9777 | 0.9000 | +0.0777 [+0.0387, +0.1161] |

## Where the zeros are

WA-JEPA (reference) zeros: 56; of them also zero for SH30 12, AP2 11, OT30 s0 12, s1 9. Zero for none of our drivers but for WA-JEPA: 42; zero for all of our drivers: 14.

| WA-JEPA zero scene | shard | class | SH30-F-s0 | AP2-AB-s0 | OT30-F-s0 | OT30-F-s1 |
|:--|:--|:--|--|--|--|--|
| 2021.05.25.14.16.10_veh-35_00083_00485-5d3e45ad38ef5b9c | part001 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.05.25.14.24.08_veh-25_03764_04034-23ee145aa4de582b | part001 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.05.25.15.59.03_veh-30_03499_03671-00ff8eeb53cc598b | part002 | offroad | offroad | offroad | offroad | 1.00 |
| 2021.05.25.16.37.23_veh-25_00005_00217-5c5f0c06d7035bb7 | part002 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.06.03.13.55.17_veh-35_00789_00999-f74148c131c15381 | part003 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.06.03.13.55.17_veh-35_02866_03582-6be41ab63cf05b6e | part003 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.06.03.13.55.17_veh-35_02866_03582-c72c3c003bc95aab | part003 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.06.03.17.06.58_veh-35_02943_03220-f1b802f6e9a559af | part003 | offroad | offroad | offroad | offroad | offroad |
| 2021.06.28.13.53.26_veh-26_00492_00696-b411a654aa215f1b | part003 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.06.28.20.24.43_veh-38_03385_04952-1347c91c511a5918 | part005 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.06.28.20.24.43_veh-38_03385_04952-c0d4412fa9f15f5b | part005 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.08.16.14.23.37_veh-45_00015_00132-36a229f658875a2e | part005 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.08.16.14.23.37_veh-45_00015_00132-5fadffd02da256e5 | part005 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.08.16.14.23.37_veh-45_00015_00132-a1adc0fae78f5a3f | part005 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.08.16.14.23.37_veh-45_00015_00132-e393bc5cafe95872 | part005 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.14.18.22_veh-48_01298_01492-4c3547b853675e66 | part007 | offroad | 1.00 | 0.90 | 0.93 | 0.96 |
| 2021.09.09.14.18.22_veh-48_01298_01492-6012210b020c53a9 | part007 | left_corridor_laterally | 0.95 | 0.95 | 0.96 | 1.00 |
| 2021.09.09.14.18.22_veh-48_01298_01492-d91edafc567a5fcc | part007 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.14.18.22_veh-48_01503_01761-87339a4d32305504 | part007 | left_corridor_laterally | offroad | 1.00 | offroad | offroad |
| 2021.09.09.14.18.22_veh-48_01503_01761-a7fa8bccce9253cb | part007 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.17.18.51_veh-48_00098_00328-85936ccd1b405f4c | part007 | collision_at_fault | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.17.18.51_veh-48_00889_01147-6782b9c3686f58f3 | part007 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | 1.00 |
| 2021.09.09.17.18.51_veh-48_01248_01450-583e9cc4115258e4 | part007 | left_corridor_laterally | collision_at_fault | collision_at_fault | collision_at_fault | collision_at_fault |
| 2021.09.09.17.18.51_veh-48_01462_01552-1975d15fde2955ff | part007 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.18.04.06_veh-40_00743_01071-1b18327179f15a8f | part008 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.09.18.04.06_veh-40_00743_01071-9d1d720d0e2e511e | part008 | offroad | offroad | 1.00 | offroad | offroad |
| 2021.09.09.18.04.06_veh-40_01340_01425-3756dc24f9e65fd7 | part008 | offroad | 0.96 | 1.00 | 0.98 | 1.00 |
| 2021.09.09.18.29.25_veh-39_01622_01766-e7295df63d0751d2 | part008 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00077_00153-171e0bea742d52d0 | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00077_00153-9f521d00c3ef55b7 | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00180_00342-52d9d533ca4e5980 | part008 | offroad | 1.00 | 1.00 | offroad | offroad |
| 2021.09.16.13.53.10_veh-42_00180_00342-6eac1c0e2bbd57bb | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00180_00342-afe83cb6c0ae5f67 | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00180_00342-bd4dac2ccde55c08 | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00180_00342-c45fc2b353c655af | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00180_00342-fd001bff97c155b3 | part008 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.16.13.53.10_veh-42_00630_00818-92427e4a55475ba6 | part008 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.13.53.10_veh-42_00630_00818-c9150b81da695ae9 | part008 | left_corridor_laterally | 1.00 | left_corridor_laterally | left_corridor_laterally | 1.00 |
| 2021.09.16.13.53.10_veh-42_00860_01069-0802a51b0a1d512c | part008 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00032_00186-4f67484c73e2503a | part008 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00032_00186-6de8e1962fc4559a | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00032_00186-8179a26d74615228 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00032_00186-8451437af5ba59ea | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00032_00186-975c802f6f175888 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00297_00935-0a06f8a3204d5e11 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00297_00935-a3882e6ae8635832 | part009 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00297_00935-c7a44a2e52bc5e22 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00297_00935-dee99345e2015845 | part009 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_00297_00935-f2592c08589e5398 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.14.39.34_veh-42_01111_01448-51552f78760d5a11 | part009 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | 1.00 | 1.00 |
| 2021.09.16.15.12.03_veh-42_01037_01434-5f402207dd7d5977 | part009 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.15.12.03_veh-42_01037_01434-a8fb178e35d25d73 | part009 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.15.47.30_veh-45_01199_01391-037b976caae85af5 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.15.47.30_veh-45_01199_01391-09cd7b3746d65a79 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.15.47.30_veh-45_01199_01391-30e7a7c93a225968 | part009 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.15.47.30_veh-45_01199_01391-bf896d504b4356c7 | part009 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |

## Cold start (decision 0: identical simulator state for every driver)

Plan endpoint (4 s, x forward / y left in the ego frame at decision 0). Difference to SH30 = distance between the two endpoints.

| driver | mean x (m) | mean abs y (m) | scenes with abs y > 2 m | distance to SH30: mean / p90 / max (m) | scenes > 3 m from SH30 | zeros ending within 3 decisions |
|:--|--:|--:|--:|:--|--:|--:|
| SH30-F-s0 | 21.66 | 2.05 | 185 | - | - | 0 |
| AP2-AB-s0 | 23.31 | 2.00 | 174 | 1.97 / 4.95 / 11.55 | 160 | 0 |
| OT30-F-s0 | 21.72 | 2.10 | 188 | 0.43 / 0.80 / 4.21 | 1 | 0 |
| OT30-F-s1 | 21.62 | 2.09 | 186 | 0.43 / 0.83 / 3.45 | 2 | 0 |
| WA-JEPA (reference) | 20.10 | 1.18 | 125 | 2.88 / 5.75 / 11.81 | 268 | 0 |

Scenes whose rollout has no driver-log decisions are not counted in the last column (nd = 0: SH30-F-s0 0, AP2-AB-s0 0, OT30-F-s0 0, OT30-F-s1 0, WA-JEPA (reference) 0).
