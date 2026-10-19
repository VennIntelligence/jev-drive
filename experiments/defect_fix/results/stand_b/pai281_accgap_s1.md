# PAI 281: STANDFMX-L18-s1+ACC+GAP against NTV0-L18-s0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/stand_b/pai281/STANDFMX-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| STANDFMX-L18-s1+ACC+GAP | 280 | 0.7253 | 63 | 187 | 0.873 | 0.859 |
| JTN-L18-s0+ACC+GAP | 280 | 0.7129 | 65 | 173 | 0.872 | 0.845 |
| NTV0-L18-s0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (STANDFMX-L18-s1+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0190 [-0.0493, +0.0108]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0143 [-0.0143, +0.0429] (+4 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0313 [-0.0552, -0.0086]**; zeros +6.

Paired mean (STANDFMX-L18-s1+ACC+GAP - JTN-L18-s0+ACC+GAP): **+0.0124 [-0.0147, +0.0408]**; zeros -2.

## Zero types

| type | STANDFMX-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 3 | 4 | 3 |
| AF-side | 1 | 1 | 0 |
| corridor | 43 | 41 | 44 |
| offroad | 16 | 19 | 12 |

## Partial scores (0 < s < 0.98)

| type | STANDFMX-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 22 | 30 | 26 |
| no-launch | 1 | 0 | 0 |
| creep | 1 | 0 | 0 |
| slow | 6 | 12 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | STANDFMX-L18-s1+ACC+GAP mean | NTV0-L18-s0+ACC+GAP mean | diff | contribution | STANDFMX-L18-s1+ACC+GAP zeros | NTV0-L18-s0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7738 | -0.0512 | -0.0068 | 2 | 3 | 1 | 2 |
| launch | 37 | 0.7075 | 0.7738 | -0.0663 | -0.0088 | 1 | 3 | 0 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 241 | 0.7132 | 0.7418 | -0.0285 | -0.0246 | 62 | 55 | 9 | 2 |
| moving | 241 | 0.7299 | 0.7418 | -0.0119 | -0.0102 | 61 | 55 | 11 | 5 |

## New zeros (zero for STANDFMX-L18-s1+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 11

| scene | stratum | STANDFMX-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 76892966 | moving | 0.000 AF-side | 1.000 ok | 0.000 AF-front |
| 0de150b0 | moving | 0.000 offroad | 0.757 rear-hit | 0.000 offroad |
| 7d6cda9e | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit | 0.000 offroad |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 28bde89a | moving | 0.000 offroad | 0.355 rear-hit | 0.340 rear-hit |
| ff7ccca9 | moving | 0.000 corridor | 0.559 rear-hit | 0.537 rear-hit |
| 7c0fb633 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| d5527d43 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for NTV0-L18-s0+ACC+GAP, scored for STANDFMX-L18-s1+ACC+GAP): 7

| scene | stratum | STANDFMX-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 787c8ada | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 9e733560 | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 62ec6f99 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| c55652d3 | launch | 1.000 ok | 0.000 corridor | 0.011 rear-hit |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| ddd61134 | launch | 0.059 rear-hit | 0.000 corridor | 0.000 corridor |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
