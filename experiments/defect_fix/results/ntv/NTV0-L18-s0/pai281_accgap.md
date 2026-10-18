# PAI 281: NTV0-L18-s0+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J ['1059fc66'] ['<AioRpcError of RPC that terminated with:'], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| NTV0-L18-s0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |
| JTN-L18-s0+ACC+GAP | 280 | 0.7129 | 65 | 173 | 0.872 | 0.845 |
| RSG+ACC+GAP | 280 | 0.6898 | 76 | 170 | 0.887 | 0.852 |

Paired mean (NTV0-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0545 [+0.0240, +0.0880]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0607 [-0.0964, -0.0286] (-17 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - RSG+ACC+GAP): **+0.0232 [-0.0086, +0.0562]**; zeros -11.

Paired mean (NTV0-L18-s0+ACC+GAP - JTN-L18-s0+ACC+GAP): **+0.0313 [+0.0086, +0.0552]**; zeros -6.

## Zero types

| type | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 3 | 4 | 7 |
| AF-side | 0 | 1 | 2 |
| corridor | 44 | 41 | 53 |
| offroad | 12 | 19 | 14 |

## Partial scores (0 < s < 0.98)

| type | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 26 | 30 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 11 | 12 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | NTV0-L18-s0+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | NTV0-L18-s0+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7545 | -0.0319 | -0.0042 | 2 | 4 | 0 | 2 |
| launch | 37 | 0.7738 | 0.7545 | +0.0193 | +0.0026 | 3 | 4 | 1 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 241 | 0.7132 | 0.6814 | +0.0318 | +0.0274 | 62 | 71 | 5 | 14 |
| moving | 241 | 0.7418 | 0.6814 | +0.0603 | +0.0519 | 55 | 71 | 2 | 18 |

## New zeros (zero for NTV0-L18-s0+ACC+GAP, scored for RSG+ACC+GAP): 3

| scene | stratum | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 7aeba6e8 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| c55652d3 | launch | 0.000 corridor | 1.000 ok | 0.011 rear-hit |
| 88b2c556 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 20

| scene | stratum | NTV0-L18-s0+ACC+GAP | RSG+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| be9320f3 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 76892966 | moving | 1.000 ok | 0.000 corridor | 0.000 AF-front |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f9cd38 | launch | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 353e8268 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 28281e1e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 7d6cda9e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 25e9675e | moving | 0.615 rear-hit | 0.000 offroad | 0.000 offroad |
| 272b7bf0 | moving | 0.825 slow | 0.000 corridor | 0.831 slow |
| 89fae1ac | moving | 0.960 slow | 0.000 offroad | 0.952 slow |
| 1f5c9e1f | moving | 0.565 rear-hit | 0.000 corridor | 0.565 rear-hit |
| 8832a062 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad | 0.000 offroad |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 4f568cd4 | moving | 0.837 slow | 0.000 corridor | 0.827 slow |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor | 0.000 AF-side |
| d5527d43 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side | 1.000 ok |
