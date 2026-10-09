Scenes common to all 5 drivers: 1491 from 44 logs, 15 shards. Scene score = AlpaSim scene score. CIs: 95% bootstrap resampling whole logs (nuPlan `date_vehicle`), 10 000 draws, seed 0.

- SH30-F-s0: 1491 scored scenes, 6 run dirs, disagreeing duplicate scenes: 0
- AP2-AB-s0: 1491 scored scenes, 6 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s0: 1491 scored scenes, 11 run dirs, disagreeing duplicate scenes: 0
- OT30-F-s1: 1491 scored scenes, 11 run dirs, disagreeing duplicate scenes: 0
- WA-JEPA (reference): 1491 scored scenes, 11 run dirs, disagreeing duplicate scenes: 0

## Scores

| driver | scenes | mean scene score [95% CI] | score 1 | score 0 | at-fault collision | offroad | left corridor | zero, other | slow (0 < score < 1) | at-fault events | mean progress |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SH30-F-s0 | 1491 | 0.8883 [0.8677, 0.9092] | 1103 | 139 | 32 | 44 | 51 | 12 | 249 | 76 | 0.922 |
| AP2-AB-s0 | 1491 | 0.9038 [0.8835, 0.9232] | 1239 | 132 | 32 | 36 | 52 | 12 | 120 | 68 | 0.967 |
| OT30-F-s0 | 1491 | 0.8978 [0.8793, 0.9183] | 1121 | 125 | 19 | 38 | 56 | 12 | 245 | 57 | 0.926 |
| OT30-F-s1 | 1491 | 0.9127 [0.8947, 0.9312] | 1177 | 106 | 19 | 29 | 46 | 12 | 208 | 48 | 0.932 |
| WA-JEPA (reference) | 1491 | 0.8947 [0.8660, 0.9218] | 1254 | 146 | 5 | 70 | 59 | 12 | 91 | 75 | 0.926 |

## Paired differences (per scene, log-clustered CI)

| A - B | mean scene score difference [95% CI] | zeros A / B | at-fault collision zeros A / B |
|:--|:--|--:|--:|
| AP2-AB-s0 - SH30-F-s0 | +0.0156 [+0.0012, +0.0310] | 132 / 139 | 32 / 32 |
| OT30-F-s0 - SH30-F-s0 | +0.0095 [-0.0057, +0.0229] | 125 / 139 | 19 / 32 |
| OT30-F-s1 - SH30-F-s0 | +0.0244 [+0.0102, +0.0377] | 106 / 139 | 19 / 32 |
| OT30 mean of 2 seeds - SH30-F-s0 | +0.0170 [+0.0032, +0.0294] | 115.5 / 139 | 19 / 32 |
| SH30-F-s0 - WA-JEPA (reference) | -0.0064 [-0.0312, +0.0203] | 139 / 146 | 32 / 5 |
| AP2-AB-s0 - WA-JEPA (reference) | +0.0091 [-0.0125, +0.0312] | 132 / 146 | 32 / 5 |
| OT30-F-s0 - WA-JEPA (reference) | +0.0031 [-0.0214, +0.0313] | 125 / 146 | 19 / 5 |
| OT30-F-s1 - WA-JEPA (reference) | +0.0180 [-0.0064, +0.0443] | 106 / 146 | 19 / 5 |
| OT30 mean of 2 seeds - WA-JEPA (reference) | +0.0105 [-0.0134, +0.0372] | 115.5 / 146 | 19 / 5 |

## OT30 line (pre-registered, plans/2026-10-09-ot30-closedloop-prereg.md)

- mean of the two OT30 seeds minus SH30-F-s0: +0.0170 [+0.0032, +0.0294]; line: >= +0.010 and CI lower bound > 0 -> **met**
- at-fault collision zeros, mean of the two seeds 19 vs SH30 32: line: not higher -> **met**
- verdict: **candidate**
- each seed alone minus SH30: s0 +0.0095 [-0.0057, +0.0229], s1 +0.0244 [+0.0102, +0.0377]
- offroad + left-corridor zeros: SH30 95, OT30 s0 94, s1 75

## Best-of-k among our own drivers (per-scene maximum = an oracle selector, an upper bound)

| set | oracle mean [95% CI] | best single in the set (mean) | oracle minus best single [95% CI] |
|:--|:--|--:|:--|
| SH30-F-s0 + AP2-AB-s0 | 0.9354 [0.9168, 0.9520] | AP2-AB-s0 0.9038 | +0.0316 [+0.0244, +0.0390] |
| SH30-F-s0 + OT30-F-s0 | 0.9281 [0.9130, 0.9442] | OT30-F-s0 0.8978 | +0.0304 [+0.0213, +0.0395] |
| SH30-F-s0 + OT30-F-s1 | 0.9328 [0.9159, 0.9494] | OT30-F-s1 0.9127 | +0.0201 [+0.0133, +0.0268] |
| AP2-AB-s0 + OT30-F-s0 | 0.9445 [0.9292, 0.9590] | AP2-AB-s0 0.9038 | +0.0407 [+0.0321, +0.0494] |
| AP2-AB-s0 + OT30-F-s1 | 0.9491 [0.9332, 0.9630] | OT30-F-s1 0.9127 | +0.0365 [+0.0267, +0.0453] |
| OT30-F-s0 + OT30-F-s1 | 0.9255 [0.9101, 0.9417] | OT30-F-s1 0.9127 | +0.0129 [+0.0082, +0.0176] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 | 0.9530 [0.9383, 0.9663] | AP2-AB-s0 0.9038 | +0.0491 [+0.0393, +0.0584] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s1 | 0.9550 [0.9392, 0.9687] | OT30-F-s1 0.9127 | +0.0424 [+0.0313, +0.0531] |
| SH30-F-s0 + OT30-F-s0 + OT30-F-s1 | 0.9402 [0.9260, 0.9544] | OT30-F-s1 0.9127 | +0.0275 [+0.0194, +0.0351] |
| AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9531 [0.9394, 0.9657] | OT30-F-s1 0.9127 | +0.0405 [+0.0301, +0.0503] |
| SH30-F-s0 + AP2-AB-s0 + OT30-F-s0 + OT30-F-s1 | 0.9580 [0.9440, 0.9705] | OT30-F-s1 0.9127 | +0.0454 [+0.0335, +0.0575] |

Reference for the selector rows: `OT30-F-s0 + OT30-F-s1` is two seeds of one recipe, i.e. what an oracle gains from training-seed noise alone.

## Per shard (mean scene score / zeros)

| shard | scenes | SH30-F-s0 | AP2-AB-s0 | OT30-F-s0 | OT30-F-s1 | WA-JEPA (reference) |
|:--|--:|--:|--:|--:|--:|--:|
| part001 | 100 | 0.9169 / 7 | 0.9101 / 8 | 0.9334 / 5 | 0.9189 / 7 | 0.9777 / 2 |
| part002 | 100 | 0.9250 / 6 | 0.9338 / 6 | 0.9338 / 5 | 0.9471 / 4 | 0.9758 / 2 |
| part003 | 100 | 0.9471 / 4 | 0.9461 / 5 | 0.9495 / 4 | 0.9703 / 2 | 0.9398 / 5 |
| part004 | 100 | 0.8988 / 8 | 0.9203 / 7 | 0.9607 / 2 | 0.9715 / 1 | 0.9504 / 3 |
| part005 | 100 | 0.9333 / 6 | 0.9439 / 5 | 0.9428 / 5 | 0.9541 / 4 | 0.9305 / 6 |
| part006 | 100 | 0.8639 / 11 | 0.8899 / 10 | 0.8373 / 14 | 0.8701 / 11 | 0.8489 / 15 |
| part007 | 100 | 0.8267 / 14 | 0.8641 / 12 | 0.8750 / 9 | 0.8606 / 11 | 0.8941 / 9 |
| part008 | 100 | 0.9046 / 8 | 0.9130 / 8 | 0.9093 / 8 | 0.9259 / 6 | 0.8289 / 16 |
| part009 | 100 | 0.9444 / 5 | 0.9452 / 5 | 0.9366 / 6 | 0.9573 / 4 | 0.8307 / 16 |
| part010 | 100 | 0.8475 / 14 | 0.8289 / 17 | 0.8732 / 11 | 0.8841 / 10 | 0.7816 / 21 |
| part011 | 100 | 0.8432 / 14 | 0.8301 / 16 | 0.8748 / 11 | 0.8481 / 14 | 0.8357 / 16 |
| part012 | 100 | 0.8773 / 9 | 0.9491 / 4 | 0.8874 / 8 | 0.9133 / 6 | 0.9754 / 2 |
| part013 | 100 | 0.8582 / 11 | 0.9417 / 5 | 0.8614 / 10 | 0.8882 / 8 | 0.8910 / 10 |
| part014 | 100 | 0.8868 / 10 | 0.8658 / 13 | 0.8481 / 14 | 0.8997 / 9 | 0.9197 / 8 |
| part015 | 91 | 0.8465 / 12 | 0.8728 / 11 | 0.8381 / 13 | 0.8776 / 9 | 0.8347 / 15 |

Part001 against the rest (mean scene score; difference with a log-clustered CI):

| driver | part001 | other shards | part001 - others [95% CI] |
|:--|--:|--:|:--|
| SH30-F-s0 | 0.9169 | 0.8862 | +0.0307 [-0.0410, +0.0695] |
| AP2-AB-s0 | 0.9101 | 0.9034 | +0.0067 [-0.0869, +0.0543] |
| OT30-F-s0 | 0.9334 | 0.8952 | +0.0382 [-0.0476, +0.0768] |
| OT30-F-s1 | 0.9189 | 0.9122 | +0.0066 [-0.0651, +0.0441] |
| WA-JEPA (reference) | 0.9777 | 0.8887 | +0.0890 [+0.0561, +0.1217] |

## Where the zeros are

WA-JEPA (reference) zeros: 146; of them also zero for SH30 51, AP2 49, OT30 s0 57, s1 50. Zero for none of our drivers but for WA-JEPA: 80; zero for all of our drivers: 54.

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
| 2021.06.28.16.29.11_veh-38_01415_01821-c0ea178930145138 | part004 | collision_at_fault | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.06.28.18.03.27_veh-14_00620_01581-4d38d745131c5de1 | part004 | left_corridor_laterally | 1.00 | 1.00 | 0.98 | 1.00 |
| 2021.06.28.18.03.27_veh-14_00620_01581-f8e2454674f75e0f | part004 | offroad | 0.82 | 0.70 | 0.98 | 0.99 |
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
| 2021.09.29.15.23.04_veh-28_00601_00802-aec02ec2aec85c06 | part012 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.09.29.19.02.14_veh-28_00273_00514-9897a102ee075dee | part012 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.09.29.19.02.14_veh-28_00540_00917-5f09eaa4509f5997 | part013 | left_corridor_laterally | 1.00 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.09.29.19.02.14_veh-28_00540_00917-7e76a2b3918656f9 | part013 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.29.19.02.14_veh-28_01717_01824-71729b03a1e95896 | part013 | offroad | offroad | offroad | offroad | offroad |
| 2021.09.29.19.02.14_veh-28_03198_03360-7da6ba784b8b5ff0 | part013 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.09.29.19.02.14_veh-28_03198_03360-8d06ea883e7853a9 | part013 | left_corridor_laterally | 0.82 | 0.94 | 0.80 | 0.78 |
| 2021.09.29.19.02.14_veh-28_03198_03360-f1ceb70bd72a5048 | part013 | other | other | other | other | other |
| 2021.10.06.07.26.10_veh-52_00006_00398-062a9e3dd60955ce | part013 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.07.26.10_veh-52_00006_00398-38b01bebf6df5fb8 | part013 | left_corridor_laterally | 1.00 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.10.06.07.26.10_veh-52_00006_00398-cbadd750cbd6581b | part013 | offroad | 0.94 | 1.00 | 0.78 | 0.79 |
| 2021.10.06.07.26.10_veh-52_00422_00728-3e42abec9c495419 | part013 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.07.26.10_veh-52_00953_01126-62b0d1b0d5b35c44 | part014 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.07.26.10_veh-52_01245_02064-115d3d7bdadf52f8 | part014 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.07.26.10_veh-52_01245_02064-947b3794a3275a2c | part014 | left_corridor_laterally | 1.00 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.10.06.07.26.10_veh-52_01245_02064-94c0ff5134d45dd1 | part014 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.07.26.10_veh-52_01245_02064-a9956fd52aa15f39 | part014 | left_corridor_laterally | left_corridor_laterally | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.10.06.07.26.10_veh-52_01245_02064-b031e4b0aea8528b | part014 | left_corridor_laterally | 1.00 | 0.98 | 1.00 | 1.00 |
| 2021.10.06.08.16.17_veh-52_00032_00170-4faa4706a50958e2 | part014 | left_corridor_laterally | 1.00 | 1.00 | left_corridor_laterally | left_corridor_laterally |
| 2021.10.06.08.16.17_veh-52_00181_00574-78c26c7e63c3534c | part014 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally |
| 2021.10.06.08.16.17_veh-52_00181_00574-ddc1271ea57154bc | part015 | offroad | 1.00 | 1.00 | 1.00 | offroad |
| 2021.10.06.08.16.17_veh-52_00181_00574-f9ed38d9ddfa531e | part015 | offroad | offroad | 1.00 | offroad | 1.00 |
| 2021.10.06.08.16.17_veh-52_00612_00782-409711b03072566a | part015 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.08.16.17_veh-52_00922_01296-3502b30911d75ed8 | part015 | left_corridor_laterally | 1.00 | offroad | 1.00 | 0.98 |
| 2021.10.06.08.16.17_veh-52_00922_01296-7ae0a03be0c357d2 | part015 | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | left_corridor_laterally | 0.81 |
| 2021.10.06.08.16.17_veh-52_00922_01296-da6d3b6810995466 | part015 | offroad | offroad | 1.00 | 1.00 | 1.00 |
| 2021.10.06.08.16.17_veh-52_00922_01296-da751fd130625cce | part015 | other | other | other | other | other |
| 2021.10.06.08.16.17_veh-52_01430_01579-64a7186ab49b5cdc | part015 | offroad | offroad | offroad | offroad | offroad |
| 2021.10.06.08.16.17_veh-52_01590_01725-078bc1027dde5d1a | part015 | left_corridor_laterally | 1.00 | 1.00 | offroad | 1.00 |
| 2021.10.06.08.16.17_veh-52_01590_01725-9e522849163c53b8 | part015 | offroad | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.08.16.17_veh-52_01949_02501-3b4031def0f45d96 | part015 | left_corridor_laterally | offroad | offroad | offroad | offroad |
| 2021.10.06.08.16.17_veh-52_01949_02501-5e02e80df7fe5f5b | part015 | offroad | 1.00 | 1.00 | left_corridor_laterally | 1.00 |
| 2021.10.06.08.16.17_veh-52_01949_02501-9a3778686fd058d2 | part015 | offroad | offroad | offroad | offroad | offroad |
| 2021.10.06.08.16.17_veh-52_01949_02501-e072351fbbfd5765 | part015 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |
| 2021.10.06.08.16.17_veh-52_01949_02501-ffe92084016a5795 | part015 | left_corridor_laterally | 1.00 | 1.00 | 1.00 | 1.00 |

## Cold start (decision 0: identical simulator state for every driver)

Plan endpoint (4 s, x forward / y left in the ego frame at decision 0). Difference to SH30 = distance between the two endpoints.

| driver | mean x (m) | mean abs y (m) | scenes with abs y > 2 m | distance to SH30: mean / p90 / max (m) | scenes > 3 m from SH30 | zeros ending within 3 decisions |
|:--|--:|--:|--:|:--|--:|--:|
| SH30-F-s0 | 21.44 | 2.11 | 432 | - | - | 0 |
| AP2-AB-s0 | 22.75 | 2.03 | 401 | 1.67 / 4.21 / 11.55 | 245 | 0 |
| OT30-F-s0 | 21.49 | 2.14 | 434 | 0.43 / 0.80 / 4.65 | 4 | 0 |
| OT30-F-s1 | 21.40 | 2.13 | 429 | 0.44 / 0.85 / 3.46 | 4 | 0 |
| WA-JEPA (reference) | 19.76 | 1.19 | 268 | 3.04 / 5.80 / 11.81 | 648 | 0 |

Scenes whose rollout has no driver-log decisions are not counted in the last column (nd = 0: SH30-F-s0 12, AP2-AB-s0 12, OT30-F-s0 12, OT30-F-s1 12, WA-JEPA (reference) 12).
