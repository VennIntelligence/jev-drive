# PAI 281: JTN-L18-s1+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/JTN-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J ['1f7fddbc'] ['<AioRpcError of RPC that terminated with:'], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| JTN-L18-s1+ACC+GAP | 280 | 0.7036 | 68 | 173 | 0.872 | 0.848 |
| JTN-L18-s0+ACC+GAP | 280 | 0.7129 | 65 | 173 | 0.872 | 0.845 |
| RSG+ACC+GAP | 280 | 0.6898 | 76 | 170 | 0.886 | 0.852 |

Paired mean (JTN-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0138 [-0.0220, +0.0498]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0286 [-0.0643, +0.0071] (-8 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0232 [-0.0084, +0.0564]**; zeros -11.

Paired mean (JTN-L18-s1+ACC+GAP - JTN-L18-s0+ACC+GAP): **-0.0094 [-0.0351, +0.0154]**; zeros +3.

## Zero types

| type | JTN-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 6 | 4 | 7 |
| AF-side | 2 | 1 | 2 |
| corridor | 42 | 41 | 53 |
| offroad | 18 | 19 | 14 |

## Partial scores (0 < s < 0.98)

| type | JTN-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 28 | 30 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 11 | 12 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | JTN-L18-s1+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | JTN-L18-s1+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7545 | -0.0319 | -0.0042 | 2 | 4 | 0 | 2 |
| launch | 37 | 0.7269 | 0.7545 | -0.0275 | -0.0036 | 2 | 4 | 0 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 241 | 0.7132 | 0.6814 | +0.0318 | +0.0274 | 62 | 71 | 5 | 14 |
| moving | 241 | 0.7017 | 0.6814 | +0.0202 | +0.0174 | 65 | 71 | 8 | 14 |

## New zeros (zero for JTN-L18-s1+ACC+GAP, scored for RSG+ACC+GAP): 8

| scene | stratum | JTN-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 0de150b0 | moving | 0.000 offroad | 0.947 rear-hit | 0.000 offroad |
| 7aeba6e8 | moving | 0.000 AF-front | 1.000 ok | 0.000 corridor |
| 41f15fb9 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| 69fbc3cb | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for JTN-L18-s1+ACC+GAP): 16

| scene | stratum | JTN-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 9e733560 | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 28281e1e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 272b7bf0 | moving | 0.839 slow | 0.000 corridor | 0.831 slow |
| 89fae1ac | moving | 0.996 ok | 0.000 offroad | 0.952 slow |
| 1f5c9e1f | moving | 0.557 rear-hit | 0.000 corridor | 0.565 rear-hit |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad | 0.000 offroad |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.828 slow | 0.000 corridor | 0.827 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 0.000 AF-side |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |

## Seed 0 vs seed 1 (arena.md / held.txt list s1 and s0 side by side, both against RSG-L18-s0)
pai281 middle column = JTN-L18-s0, reference = RSG+ACC+GAP
