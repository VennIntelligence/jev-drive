# PAI 281: JTXS-L18-s1+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/JTXS-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 279 of 281 (excluded: J ['9e733560'] ['<AioRpcError of RPC that terminated with:'], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| JTXS-L18-s1+ACC+GAP | 279 | 0.7076 | 65 | 170 | 0.867 | 0.844 |
| JTN-L18-s1+ACC+GAP | 279 | 0.7025 | 68 | 172 | 0.872 | 0.847 |
| RSG+ACC+GAP | 279 | 0.6923 | 75 | 170 | 0.886 | 0.851 |

Paired mean (JTXS-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0154 [-0.0171, +0.0490]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0358 [-0.0681, -0.0036] (-10 zeros).


Paired mean (JTN-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0102 [-0.0232, +0.0447]**; zeros -7.

Paired mean (JTXS-L18-s1+ACC+GAP - JTN-L18-s1+ACC+GAP): **+0.0051 [-0.0134, +0.0248]**; zeros -3.

## Zero types

| type | JTXS-L18-s1+ACC+GAP | JTN-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 5 | 6 | 7 |
| AF-side | 0 | 2 | 2 |
| corridor | 44 | 42 | 52 |
| offroad | 16 | 18 | 14 |

## Partial scores (0 < s < 0.98)

| type | JTXS-L18-s1+ACC+GAP | JTN-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 32 | 28 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 12 | 11 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | JTXS-L18-s1+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | JTXS-L18-s1+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s1+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s1+ACC+GAP) launch | 37 | 0.7269 | 0.7545 | -0.0275 | -0.0036 | 2 | 4 | 0 | 2 |
| launch | 37 | 0.6968 | 0.7545 | -0.0576 | -0.0076 | 2 | 4 | 0 | 2 |
| (JTN-L18-s1+ACC+GAP) moving | 240 | 0.7004 | 0.6843 | +0.0162 | +0.0139 | 65 | 70 | 8 | 13 |
| moving | 240 | 0.7110 | 0.6843 | +0.0267 | +0.0230 | 62 | 70 | 6 | 14 |

## New zeros (zero for JTXS-L18-s1+ACC+GAP, scored for RSG+ACC+GAP): 6

| scene | stratum | JTXS-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s1+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 0de150b0 | moving | 0.000 offroad | 0.947 rear-hit | 0.000 offroad |
| 7aeba6e8 | moving | 0.000 AF-front | 1.000 ok | 0.000 AF-front |
| 69fbc3cb | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 062def2d | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| b8d5d0d3 | moving | 0.000 AF-front | 1.000 ok | 0.000 AF-side |

## Fixed zeros (zero for RSG+ACC+GAP, scored for JTXS-L18-s1+ACC+GAP): 16

| scene | stratum | JTXS-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s1+ACC+GAP |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 76892966 | moving | 1.000 ok | 0.000 corridor | 0.000 AF-front |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 28281e1e | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 272b7bf0 | moving | 0.843 slow | 0.000 corridor | 0.839 slow |
| 89fae1ac | moving | 0.949 slow | 0.000 offroad | 0.996 ok |
| 1f5c9e1f | moving | 0.558 rear-hit | 0.000 corridor | 0.557 rear-hit |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.828 slow | 0.000 corridor | 0.828 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 1.000 ok |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 0.757 rear-hit | 0.000 AF-side | 1.000 ok |
