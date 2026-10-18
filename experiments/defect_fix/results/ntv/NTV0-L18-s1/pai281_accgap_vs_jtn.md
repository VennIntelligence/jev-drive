# PAI 281: NTV0-L18-s1+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 281 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s1+ACC+GAP | 281 | 0.7393 | 62 | 184 | 0.884 | 0.878 |
| JTN-L18-s0+ACC+GAP | 281 | 0.7140 | 65 | 174 | 0.872 | 0.845 |
| RSG+ACC+GAP | 281 | 0.6909 | 76 | 171 | 0.886 | 0.852 |

Paired mean (NTV0-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0484 [+0.0148, +0.0850]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0498 [-0.0854, -0.0142] (-14 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0231 [-0.0082, +0.0563]**; zeros -11.

Paired mean (NTV0-L18-s1+ACC+GAP - JTN-L18-s0+ACC+GAP): **+0.0253 [-0.0018, +0.0531]**; zeros -3.

## Zero types

| type | NTV0-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 3 | 4 | 7 |
| AF-side | 1 | 1 | 2 |
| corridor | 44 | 41 | 53 |
| offroad | 14 | 19 | 14 |

## Partial scores (0 < s < 0.98)

| type | NTV0-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 24 | 30 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 11 | 12 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | NTV0-L18-s1+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | NTV0-L18-s1+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7545 | -0.0319 | -0.0042 | 2 | 4 | 0 | 2 |
| launch | 37 | 0.7569 | 0.7545 | +0.0024 | +0.0003 | 4 | 4 | 2 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 242 | 0.7144 | 0.6827 | +0.0317 | +0.0273 | 62 | 71 | 5 | 14 |
| moving | 242 | 0.7386 | 0.6827 | +0.0558 | +0.0481 | 57 | 71 | 4 | 18 |

## New zeros (zero for NTV0-L18-s1+ACC+GAP, scored for RSG+ACC+GAP): 6

| scene | stratum | NTV0-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 9d585b78 | moving | 0.000 offroad | 1.000 ok | 1.000 ok |
| 7aeba6e8 | moving | 0.000 AF-front+corridor | 1.000 ok | 0.000 corridor |
| c55652d3 | launch | 0.000 corridor | 1.000 ok | 0.011 rear-hit |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| fd7779a6 | launch | 0.000 offroad | 0.488 rear-hit | 0.457 rear-hit |

## Fixed zeros (zero for RSG+ACC+GAP, scored for NTV0-L18-s1+ACC+GAP): 20

| scene | stratum | NTV0-L18-s1+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 76892966 | moving | 1.000 ok | 0.000 corridor | 0.000 AF-front |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 62ec6f99 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 28281e1e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 7d6cda9e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 25e9675e | moving | 0.986 ok | 0.000 offroad | 0.000 offroad |
| 272b7bf0 | moving | 0.837 slow | 0.000 corridor | 0.831 slow |
| 89fae1ac | moving | 0.968 slow | 0.000 offroad | 0.952 slow |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad | 0.000 offroad |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.842 slow | 0.000 corridor | 0.827 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 0.000 AF-side |
| d5527d43 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
