# PAI 281: NTV0-L18-s1+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s1+ACC+GAP | 280 | 0.7384 | 62 | 183 | 0.884 | 0.879 |
| NTV0-L18-s0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |
| RSG+ACC+GAP | 280 | 0.6898 | 76 | 170 | 0.887 | 0.852 |

Paired mean (NTV0-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0486 [+0.0147, +0.0847]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0500 [-0.0857, -0.0143] (-14 zeros).


Paired mean (NTV0-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0545 [+0.0240, +0.0880]**; zeros -17.

Paired mean (NTV0-L18-s1+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0059 [-0.0259, +0.0134]**; zeros +3.

## Zero types

| type | NTV0-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 3 | 3 | 7 |
| AF-side | 1 | 0 | 2 |
| corridor | 44 | 44 | 53 |
| offroad | 14 | 12 | 14 |

## Partial scores (0 < s < 0.98)

| type | NTV0-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 24 | 26 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 11 | 11 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | NTV0-L18-s1+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | NTV0-L18-s1+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (NTV0-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (NTV0-L18-s0+ACC+GAP) launch | 37 | 0.7738 | 0.7545 | +0.0193 | +0.0026 | 3 | 4 | 1 | 2 |
| launch | 37 | 0.7569 | 0.7545 | +0.0024 | +0.0003 | 4 | 4 | 2 | 2 |
| (NTV0-L18-s0+ACC+GAP) moving | 241 | 0.7418 | 0.6814 | +0.0603 | +0.0519 | 55 | 71 | 2 | 18 |
| moving | 241 | 0.7375 | 0.6814 | +0.0561 | +0.0483 | 57 | 71 | 4 | 18 |

## New zeros (zero for NTV0-L18-s1+ACC+GAP, scored for RSG+ACC+GAP): 6

| scene | stratum | NTV0-L18-s1+ACC+GAP | RSG+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 9d585b78 | moving | 0.000 offroad | 1.000 ok | 1.000 ok |
| 7aeba6e8 | moving | 0.000 AF-front+corridor | 1.000 ok | 0.000 corridor |
| c55652d3 | launch | 0.000 corridor | 1.000 ok | 0.000 corridor |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| fd7779a6 | launch | 0.000 offroad | 0.488 rear-hit | 0.462 rear-hit |

## Fixed zeros (zero for RSG+ACC+GAP, scored for NTV0-L18-s1+ACC+GAP): 20

| scene | stratum | NTV0-L18-s1+ACC+GAP | RSG+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 76892966 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 62ec6f99 | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 28281e1e | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7d6cda9e | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 25e9675e | moving | 0.986 ok | 0.000 offroad | 0.615 rear-hit |
| 272b7bf0 | moving | 0.837 slow | 0.000 corridor | 0.825 slow |
| 89fae1ac | moving | 0.968 slow | 0.000 offroad | 0.960 slow |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad | 1.000 ok |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.842 slow | 0.000 corridor | 0.837 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 1.000 ok |
| d5527d43 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
