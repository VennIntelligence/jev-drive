# PAI 281: STANDFMXH-L18-s0+ACC+GAP against NTV0-L18-s0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/stand_b/pai281/STANDFMXH-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| STANDFMXH-L18-s0+ACC+GAP | 280 | 0.7029 | 66 | 174 | 0.870 | 0.850 |
| JTN-L18-s0+ACC+GAP | 280 | 0.7129 | 65 | 173 | 0.872 | 0.845 |
| NTV0-L18-s0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (STANDFMXH-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0414 [-0.0685, -0.0157]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0250 [+0.0000, +0.0536] (+7 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0313 [-0.0552, -0.0086]**; zeros +6.

Paired mean (STANDFMXH-L18-s0+ACC+GAP - JTN-L18-s0+ACC+GAP): **-0.0100 [-0.0359, +0.0153]**; zeros +1.

## Zero types

| type | STANDFMXH-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 8 | 4 | 3 |
| AF-side | 1 | 1 | 0 |
| corridor | 41 | 41 | 44 |
| offroad | 16 | 19 | 12 |

## Partial scores (0 < s < 0.98)

| type | STANDFMXH-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 30 | 30 | 26 |
| no-launch | 0 | 0 | 0 |
| creep | 1 | 0 | 0 |
| slow | 9 | 12 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | STANDFMXH-L18-s0+ACC+GAP mean | NTV0-L18-s0+ACC+GAP mean | diff | contribution | STANDFMXH-L18-s0+ACC+GAP zeros | NTV0-L18-s0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7738 | -0.0512 | -0.0068 | 2 | 3 | 1 | 2 |
| launch | 37 | 0.6728 | 0.7738 | -0.1010 | -0.0133 | 1 | 3 | 0 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 241 | 0.7132 | 0.7418 | -0.0285 | -0.0246 | 62 | 55 | 9 | 2 |
| moving | 241 | 0.7092 | 0.7418 | -0.0326 | -0.0280 | 64 | 55 | 11 | 2 |

## New zeros (zero for STANDFMXH-L18-s0+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 11

| scene | stratum | STANDFMXH-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 76892966 | moving | 0.000 AF-front | 1.000 ok | 0.000 AF-front |
| 7d6cda9e | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit | 0.000 offroad |
| e69f06a1 | moving | 0.000 AF-side | 0.848 rear-hit | 0.848 rear-hit |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 1438905d | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 8832a062 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 7c0fb633 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| d5527d43 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for NTV0-L18-s0+ACC+GAP, scored for STANDFMXH-L18-s0+ACC+GAP): 4

| scene | stratum | STANDFMXH-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| c55652d3 | launch | 0.004 rear-hit | 0.000 corridor | 0.011 rear-hit |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| ddd61134 | launch | 0.060 rear-hit | 0.000 corridor | 0.000 corridor |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
