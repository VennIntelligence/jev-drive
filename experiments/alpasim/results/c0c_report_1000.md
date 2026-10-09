Scenes common to all 5 drivers: 1000 from 35 logs, 10 shards. Scene score = AlpaSim scene score. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10 000 draws, seed 0.

- SH30-F-s0: 1000 scored scenes, 4 run dirs, disagreeing duplicate scenes: 0
- AP2-AB-s0: 1000 scored scenes, 4 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s0: 1000 scored scenes, 6 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s1: 1000 scored scenes, 6 run dirs, disagreeing duplicate scenes: 0
- WA-JEPA (reference): 1000 scored scenes, 6 run dirs, disagreeing duplicate scenes: 0

## Scores

| driver | scenes | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 1000 | 0.8952 [0.8699, 0.9189] | 768 | 89 | 21 | 27 | 31 | 10 | 143 | 48 | 0.935 |
| AP2-AB-s0 | 1000 | 0.9005 [0.8762, 0.9232] | 833 | 92 | 25 | 23 | 34 | 10 | 75 | 48 | 0.976 |
| OT30-F-s0 | 1000 | 0.9066 [0.8851, 0.9272] | 784 | 78 | 14 | 22 | 32 | 10 | 138 | 36 | 0.936 |
| OT30-F-s1 | 1000 | 0.9136 [0.8887, 0.9370] | 805 | 73 | 15 | 18 | 30 | 10 | 122 | 33 | 0.944 |
| WA-JEPA (reference) | 1000 | 0.8844 [0.8472, 0.9198] | 819 | 108 | 4 | 55 | 39 | 10 | 73 | 59 | 0.920 |

## Paired differences (per scene, log-clustered CI)

| A - B | mean scene score difference [95% CI] | zeros A / B | at-fault collision zeros A / B |
|:--|:--|--:|--:|
| AP2-AB-s0 - SH30-F-s0 | +0.0053 [-0.0082, +0.0185] | 92 / 89 | 25 / 21 |
| OT30-F-s0 - SH30-F-s0 | +0.0113 [-0.0030, +0.0241] | 78 / 89 | 14 / 21 |
| OT30-F-s1 - SH30-F-s0 | +0.0184 [+0.0045, +0.0321] | 73 / 89 | 15 / 21 |
| OT30 mean of 2 seeds - SH30-F-s0 | +0.0149 [+0.0018, +0.0274] | 75.5 / 89 | 14.5 / 21 |
| SH30-F-s0 - WA-JEPA (reference) | +0.0109 [-0.0205, +0.0429] | 89 / 108 | 21 / 4 |
| AP2-AB-s0 - WA-JEPA (reference) | +0.0161 [-0.0149, +0.0457] | 92 / 108 | 25 / 4 |
| OT30-F-s0 - WA-JEPA (reference) | +0.0222 [-0.0090, +0.0541] | 78 / 108 | 14 / 4 |
| OT30-F-s1 - WA-JEPA (reference) | +0.0293 [-0.0043, +0.0628] | 73 / 108 | 15 / 4 |
| OT30 mean of 2 seeds - WA-JEPA (reference) | +0.0257 [-0.0062, +0.0582] | 75.5 / 108 | 14.5 / 4 |

## OT30 line (pre-registered, plans/2026-10-09-ot30-closedloop-prereg.md)

- mean of the two OT30 seeds minus SH30-F-s0: +0.0149 [+0.0018, +0.0274]; line: >= +0.010 and CI lower bound > 0 -> **met**
- at-fault collision zeros, mean of the two seeds 14.5 vs SH30 21: line: not higher -> **met**
- verdict: **candidate**
- each seed alone minus SH30: s0 +0.0113 [-0.0030, +0.0241], s1 +0.0184 [+0.0045, +0.0321]
- offroad + left-corridor zeros: SH30 58, OT30 s0 54, s1 48

## Best-of-k among our own drivers (per-scene maximum = an oracle selector, an upper bound)

| set | oracle mean [95% CI] | best single in the set (mean) | oracle minus best single [95% CI] |
|:--|:--|--:|:--|
| SH30-F-s0 + AP2-AB-s0 | 0.9330 [0.9120, 0.9524] | AP2-AB-s0 0.9005 | +0.0325 [+0.0233, +0.0423] |
| SH30-F-s0 + OT30-F-s0 | 0.9316 [0.9115, 0.9506] | OT30-F-s0 0.9066 | +0.0250 [+0.0186, +0.0318] |
| SH30-F-s0 + OT30-F-s1 | 0.9325 [0.9087, 0.9550] | OT30-F-s1 0.9136 | +0.0188 [+0.0116, +0.0270] |
| AP2-AB-s0 + OT30-F-s0 | 0.9423 [0.9226, 0.9604] | OT30-F-s0 0.9066 | +0.0358 [+0.0263, +0.0465] |
| AP2-AB-s0 + OT30-F-s1 | 0.9442 [0.9226, 0.9636] | OT30-F-s1 0.9136 | +0.0305 [+0.0210, +0.0405] |
| OT30-F-s0 + OT30-F-s1 | 0.9276 [0.9063, 0.9480] | OT30-F-s1 0.9136 | +0.0140 [+0.0076, +0.0203] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 | 0.9507 [0.9320, 0.9675] | OT30-F-s0 0.9066 | +0.0441 [+0.0341, +0.0563] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s1 | 0.9508 [0.9300, 0.9695] | OT30-F-s1 0.9136 | +0.0372 [+0.0262, +0.0497] |
| SH30-F-s0 + OT30-F-s0 + OT30-F-s1 | 0.9406 [0.9209, 0.9594] | OT30-F-s1 0.9136 | +0.0269 [+0.0177, +0.0366] |
| AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9499 [0.9311, 0.9668] | OT30-F-s1 0.9136 | +0.0363 [+0.0249, +0.0488] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9551 [0.9363, 0.9717] | OT30-F-s1 0.9136 | +0.0415 [+0.0288, +0.0573] |

Reference for the selector rows: `OT30-F-s0 + OT30-F-s1` is two seeds of one recipe, i.e. what an oracle gains from training-seed noise alone.

## Per shard (mean scene score / zeros)

| shard | scenes | SH30-F-s0 | AP2-AB-s0 | OT30-F-s0 | OT30-F-s1 | WA-JEPA (reference) |
|:--|--:|--:|--:|--:|--:|--:|
| part001 | 100 | 0.9169 / 7 | 0.9101 / 8 | 0.9334 / 5 | 0.9189 / 7 | 0.9777 / 2 |
| part002 | 100 | 0.9250 / 6 | 0.9338 / 6 | 0.9338 / 5 | 0.9471 / 4 | 0.9758 / 2 |
| part003 | 100 | 0.9471 / 4 | 0.9461 / 5 | 0.9495 / 4 | 0.9703 / 2 | 0.9398 / 5 |
| part005 | 100 | 0.9333 / 6 | 0.9439 / 5 | 0.9428 / 5 | 0.9541 / 4 | 0.9305 / 6 |
| part006 | 100 | 0.8639 / 11 | 0.8899 / 10 | 0.8373 / 14 | 0.8701 / 11 | 0.8489 / 15 |
| part007 | 100 | 0.8267 / 14 | 0.8641 / 12 | 0.8750 / 9 | 0.8606 / 11 | 0.8941 / 9 |
| part008 | 100 | 0.9046 / 8 | 0.9130 / 8 | 0.9093 / 8 | 0.9259 / 6 | 0.8289 / 16 |
| part009 | 100 | 0.9444 / 5 | 0.9452 / 5 | 0.9366 / 6 | 0.9573 / 4 | 0.8307 / 16 |
| part010 | 100 | 0.8475 / 14 | 0.8289 / 17 | 0.8732 / 11 | 0.8841 / 10 | 0.7816 / 21 |
| part011 | 100 | 0.8432 / 14 | 0.8301 / 16 | 0.8748 / 11 | 0.8481 / 14 | 0.8357 / 16 |

Part001 against the rest (mean scene score; difference with a log-clustered CI):

| driver | part001 | other shards | part001 - others [95% CI] |
|:--|--:|--:|:--|
| SH30-F-s0 | 0.9169 | 0.8928 | +0.0241 [-0.0494, +0.0668] |
| AP2-AB-s0 | 0.9101 | 0.8994 | +0.0106 [-0.0837, +0.0617] |
| OT30-F-s0 | 0.9334 | 0.9036 | +0.0298 [-0.0567, +0.0706] |
| OT30-F-s1 | 0.9189 | 0.9131 | +0.0058 [-0.0696, +0.0480] |
| WA-JEPA (reference) | 0.9777 | 0.8740 | +0.1037 [+0.0628, +0.1455] |

## Where the zeros are

WA-JEPA (reference) zeros: 108; of them also zero for SH30 38, AP2 37, OT30 s0 39, s1 35. Zero for none of our drivers but for WA-JEPA: 63; zero for all of our drivers: 39.

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
| 2021.08.30.13.45.25_veh-40_00610_00771-c133861a233a51de | part006 | left_corridor_laterally | 0.87 | 0.95 | 0.99 | 0.97 |
| 2021.08.30.13.45.25_veh-40_00878_01104-257d737fc3865fd1 | part006 | other | other | other | other | other |
| 2021.08.30.13.45.25_veh-40_00878_01104-702320a088cf5d53 | part006 | offroad | offroad | 1.00 | offroad | offroad |
| 2021.08.30.13.45.25_veh-40_00878_01104-70da6b21101d555f | part006 | offroad | offroad | offroad | offroad | offroad |
| 2021.08.30.13.45.25_veh-40_01116_01336-ef81756601bc569f | part006 | collision_at_fault | 0.78 | 1.00 | 0.90 | 0.87 |
| 2021.08.30.13.45.25_veh-40_01116_01336-f6f14df95f6c52af | part006 | left_corridor_laterally | 0.98 | 1.00 | offroad | 1.00 |
| 2021.08.30.14.54.34_veh-40_00439_00835-02f75336f9f55e4f | part006 | left_corridor_laterally | 1.00 | 1.00 | left_corridor_laterally | 1.00 |
| 2021.08.30.14.54.34_veh-40_00439_00835-1ca75f05f31d51cc | part006 | other | other | other | other | other |
| 2021.08.30.14.54.34_veh-40_00439_00835-2bc38f766c8851fa | part006 | offroad | offroad | offroad | offroad | offroad |
| 2021.08.30.14.54.34_veh-40_00439_00835-537935a8e7b653f2 | part006 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.08.30.14.54.34_veh-40_00439_00835-673a88a4037f5b6b | part006 | other | other | other | other | other |
| 2021.08.30.16.16.44_veh-40_00256_00716-eb15f0d956eb5ac4 | part006 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.08.30.16.16.44_veh-40_01099_01351-0d9243e74a1a501a | part006 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.08.30.16.16.44_veh-40_01099_01351-58ad6156b5895541 | part006 | left_corridor_laterally | left_corridor_laterally | 0.98 | left_corridor_laterally | left_corridor_laterally |
| 2021.09.09.14.18.22_veh-48_00221_00299-b1a1b2a18fa4504f | part006 | collision_at_fault | 1.00 | 1.00 | 1.00 | 1.00 |
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
| 2021.09.16.16.20.27_veh-08_02435_02525-c1e76b8992fa5182 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.17.40.09_veh-45_02539_02745-661ce644db6d5546 | part010 | offroad | 1.00 | offroad | 1.00 | 1.00 |
| 2021.09.16.17.40.09_veh-45_02539_02745-a96abad3a09753c5 | part010 | other | other | other | other | other |
| 2021.09.16.19.12.04_veh-42_01088_01192-62eac0a6b7e05fbf | part010 | collision_at_fault | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.12.04_veh-42_01438_01677-3d37ed78124057a1 | part010 | other | other | other | other | other |
| 2021.09.16.19.12.04_veh-42_01438_01677-9104884bcb915c08 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_00472_00711-a2978bba82bd5751 | part010 | offroad | 0.94 | 0.89 | 0.87 | 0.91 |
| 2021.09.16.19.27.01_veh-45_00472_00711-d395fa715d3e58c7 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_00472_00711-e5599a8884235d93 | part010 | other | other | other | other | other |
| 2021.09.16.19.27.01_veh-45_01749_03230-10909749099354e6 | part010 | left_corridor_laterally | collision_at_fault | collision_at_fault | 1.00 | collision_at_fault |
| 2021.09.16.19.27.01_veh-45_01749_03230-3770407dcc67520c | part010 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.16.19.27.01_veh-45_01749_03230-50613cf56a8d5a38 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-572e6c81e32958f2 | part010 | left_corridor_laterally | 1.00 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.09.16.19.27.01_veh-45_01749_03230-5d3573d6da7952d3 | part010 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.16.19.27.01_veh-45_01749_03230-657ea52878935352 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-6b6afd7690245e14 | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-790a5a0973815ab3 | part010 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.16.19.27.01_veh-45_01749_03230-8799520f3ff95bfc | part010 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-89e02236312d5038 | part010 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.09.16.19.27.01_veh-45_01749_03230-9551ef5e14315cc0 | part010 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | 0.77 | 0.81 |
| 2021.09.16.19.27.01_veh-45_01749_03230-9696c1f82bc05ffc | part010 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-b9d14f59883a5496 | part011 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-bdad96248e575296 | part011 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.16.19.27.01_veh-45_01749_03230-d33c6db306f35ef9 | part011 | left_corridor_laterally | collision_at_fault | left_corridor_laterally | collision_at_fault | collision_at_fault |
| 2021.09.16.19.27.01_veh-45_01749_03230-d743862c9c555961 | part011 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-d870256a3b185659 | part011 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.27.01_veh-45_01749_03230-ea1302023ad258ff | part011 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.09.16.19.49.00_veh-42_00990_01609-4302a0a4b9f05b61 | part011 | other | other | other | other | other |
| 2021.09.16.19.49.00_veh-42_00990_01609-458e833803315b4f | part011 | other | other | other | other | other |
| 2021.09.16.19.49.00_veh-42_00990_01609-9b1bea0cc0d75583 | part011 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.49.00_veh-42_00990_01609-a067e1b873c8534d | part011 | other | other | other | other | other |
| 2021.09.16.19.49.00_veh-42_00990_01609-a9609780217c5831 | part011 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.16.19.49.00_veh-42_00990_01609-e31131deed6656ea | part011 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.09.16.19.49.00_veh-42_00990_01609-f9349f5d723b5421 | part011 | other | other | other | other | other |
| 2021.09.16.21.13.37_veh-42_00172_00347-a425dd8a1b5552db | part011 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.29.14.44.26_veh-28_00238_00320-cfb755b8d37458ac | part011 | left_corridor_laterally | 1.00 | left_corridor_laterally | 1.00 | 1.00 |
| 2021.09.29.14.44.26_veh-28_01059_01191-dda361f4db52537a | part011 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |

## Cold start (decision 0: identical simulator state for every driver)

Plan endpoint (4 s, x forward / y left in the ego frame at decision 0). Difference to SH30 = distance between the two endpoints.

| driver | mean x (m) | mean abs y (m) | scenes with abs y > 2 m | distance to SH30: mean / p90 / max (m) | scenes > 3 m from SH30 | zeros ending within 3 decisions |
|:--|--:|--:|--:|:--|--:|--:|
| SH30-F-s0 | 21.29 | 2.10 | 270 | - | - | 0 |
| AP2-AB-s0 | 22.73 | 2.02 | 249 | 1.77 / 4.60 / 11.55 | 182 | 0 |
| OT30-F-s0 | 21.32 | 2.14 | 275 | 0.42 / 0.79 / 4.21 | 1 | 0 |
| OT30-F-s1 | 21.23 | 2.12 | 271 | 0.44 / 0.84 / 3.45 | 2 | 0 |
| WA-JEPA (reference) | 19.79 | 1.23 | 187 | 2.92 / 5.78 / 11.81 | 389 | 0 |

Scenes whose rollout has no driver-log decisions are not counted in the last column (nd = 0: SH30-F-s0 10, AP2-AB-s0 10, OT30-F-s0 10, OT30-F-s1 10, WA-JEPA (reference) 10).
