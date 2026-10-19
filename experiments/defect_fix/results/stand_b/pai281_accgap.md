# PAI 281: STANDFMX-L18-s0+ACC+GAP against NTV0-L18-s0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/stand_b/pai281/STANDFMX-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 279 of 281 (excluded: J ['9697fbf4'] ['<AioRpcError of RPC that terminated with:'], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| STANDFMX-L18-s0+ACC+GAP | 279 | 0.6962 | 71 | 179 | 0.874 | 0.851 |
| JTN-L18-s0+ACC+GAP | 279 | 0.7119 | 65 | 172 | 0.872 | 0.845 |
| NTV0-L18-s0+ACC+GAP | 279 | 0.7434 | 59 | 183 | 0.881 | 0.874 |

Paired mean (STANDFMX-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0472 [-0.0780, -0.0182]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0430 [+0.0108, +0.0753] (+12 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0314 [-0.0551, -0.0084]**; zeros +6.

Paired mean (STANDFMX-L18-s0+ACC+GAP - JTN-L18-s0+ACC+GAP): **-0.0157 [-0.0435, +0.0110]**; zeros +6.

## Zero types

| type | STANDFMX-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 9 | 4 | 3 |
| AF-side | 1 | 1 | 0 |
| corridor | 43 | 41 | 44 |
| offroad | 18 | 19 | 12 |

## Partial scores (0 < s < 0.98)

| type | STANDFMX-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 21 | 30 | 26 |
| no-launch | 1 | 0 | 0 |
| creep | 1 | 0 | 0 |
| slow | 6 | 12 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | STANDFMX-L18-s0+ACC+GAP mean | NTV0-L18-s0+ACC+GAP mean | diff | contribution | STANDFMX-L18-s0+ACC+GAP zeros | NTV0-L18-s0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7738 | -0.0512 | -0.0068 | 2 | 3 | 1 | 2 |
| launch | 37 | 0.6542 | 0.7738 | -0.1195 | -0.0159 | 3 | 3 | 2 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 240 | 0.7120 | 0.7407 | -0.0287 | -0.0247 | 62 | 55 | 9 | 2 |
| moving | 240 | 0.7043 | 0.7407 | -0.0364 | -0.0313 | 67 | 55 | 14 | 2 |

## New zeros (zero for STANDFMX-L18-s0+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 16

| scene | stratum | STANDFMX-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 76892966 | moving | 0.000 AF-front | 1.000 ok | 0.000 AF-front |
| 4fcb5fde | launch | 0.000 AF-front | 1.000 ok | 0.768 rear-hit |
| 28281e1e | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 0de150b0 | moving | 0.000 offroad | 0.757 rear-hit | 0.000 offroad |
| 7d6cda9e | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit | 0.000 offroad |
| e69f06a1 | moving | 0.000 AF-side | 0.848 rear-hit | 0.848 rear-hit |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 28bde89a | moving | 0.000 offroad | 0.355 rear-hit | 0.340 rear-hit |
| 8b28b8f9 | launch | 0.000 AF-front | 1.000 ok | 1.000 ok |
| ff7ccca9 | moving | 0.000 corridor | 0.559 rear-hit | 0.537 rear-hit |
| 8832a062 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 7c0fb633 | moving | 0.000 offroad | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| d5527d43 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for NTV0-L18-s0+ACC+GAP, scored for STANDFMX-L18-s0+ACC+GAP): 4

| scene | stratum | STANDFMX-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| c55652d3 | launch | 0.004 rear-hit | 0.000 corridor | 0.011 rear-hit |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| ddd61134 | launch | 0.059 rear-hit | 0.000 corridor | 0.000 corridor |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
