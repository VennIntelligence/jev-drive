# PAI 281: NTVXS-L18-s0+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTVXS-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 279 of 281 (excluded: J ['58eb6cd6'] ['<AioRpcError of RPC that terminated with:'], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| NTVXS-L18-s0+ACC+GAP | 279 | 0.7212 | 66 | 179 | 0.883 | 0.870 |
| NTV0-L18-s0+ACC+GAP | 279 | 0.7434 | 59 | 183 | 0.881 | 0.874 |
| RSG+ACC+GAP | 279 | 0.6923 | 75 | 170 | 0.886 | 0.852 |

Paired mean (NTVXS-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0290 [-0.0018, +0.0610]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0323 [-0.0645, +0.0000] (-9 zeros).


Paired mean (NTV0-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0511 [+0.0208, +0.0837]**; zeros -16.

Paired mean (NTVXS-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0221 [-0.0424, -0.0034]**; zeros +7.

## Zero types

| type | NTVXS-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 4 | 3 | 7 |
| AF-side | 2 | 0 | 2 |
| corridor | 47 | 44 | 52 |
| offroad | 13 | 12 | 14 |

## Partial scores (0 < s < 0.98)

| type | NTVXS-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 22 | 26 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 12 | 11 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | NTVXS-L18-s0+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | NTVXS-L18-s0+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (NTV0-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (NTV0-L18-s0+ACC+GAP) launch | 37 | 0.7738 | 0.7545 | +0.0193 | +0.0026 | 3 | 4 | 1 | 2 |
| launch | 37 | 0.7754 | 0.7545 | +0.0209 | +0.0028 | 3 | 4 | 1 | 2 |
| (NTV0-L18-s0+ACC+GAP) moving | 240 | 0.7407 | 0.6843 | +0.0564 | +0.0485 | 55 | 70 | 2 | 17 |
| moving | 240 | 0.7147 | 0.6843 | +0.0305 | +0.0262 | 62 | 70 | 5 | 13 |

## New zeros (zero for NTVXS-L18-s0+ACC+GAP, scored for RSG+ACC+GAP): 6

| scene | stratum | NTVXS-L18-s0+ACC+GAP | RSG+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 9d585b78 | moving | 0.000 offroad | 1.000 ok | 1.000 ok |
| 0de150b0 | moving | 0.000 offroad | 0.947 rear-hit | 0.757 rear-hit |
| 7aeba6e8 | moving | 0.000 AF-front | 1.000 ok | 0.000 corridor |
| c55652d3 | launch | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 88b2c556 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for NTVXS-L18-s0+ACC+GAP): 15

| scene | stratum | NTVXS-L18-s0+ACC+GAP | RSG+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 7d6cda9e | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 25e9675e | moving | 0.620 rear-hit | 0.000 offroad | 0.615 rear-hit |
| 272b7bf0 | moving | 0.842 slow | 0.000 corridor | 0.825 slow |
| 89fae1ac | moving | 0.966 slow | 0.000 offroad | 0.960 slow |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.834 slow | 0.000 corridor | 0.837 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7f699124 | moving | 1.000 ok | 0.000 offroad | 0.000 offroad |
| d5527d43 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
