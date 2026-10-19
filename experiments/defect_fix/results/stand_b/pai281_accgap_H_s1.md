# PAI 281: STANDFMXH-L18-s1+ACC+GAP against NTV0-L18-s0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/stand_b/pai281/STANDFMXH-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 278 of 281 (excluded: J ['9591dbaa', 'cec726ef'] ['<AioRpcError of RPC that terminated with:'], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| STANDFMXH-L18-s1+ACC+GAP | 278 | 0.7145 | 64 | 176 | 0.869 | 0.851 |
| JTN-L18-s0+ACC+GAP | 278 | 0.7109 | 65 | 171 | 0.871 | 0.844 |
| NTV0-L18-s0+ACC+GAP | 278 | 0.7424 | 59 | 182 | 0.881 | 0.874 |

Paired mean (STANDFMXH-L18-s1+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0280 [-0.0563, -0.0013]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0180 [-0.0072, +0.0468] (+5 zeros).


Paired mean (JTN-L18-s0+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0316 [-0.0557, -0.0086]**; zeros +6.

Paired mean (STANDFMXH-L18-s1+ACC+GAP - JTN-L18-s0+ACC+GAP): **+0.0036 [-0.0239, +0.0319]**; zeros -1.

## Zero types

| type | STANDFMXH-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 8 | 4 | 3 |
| AF-side | 0 | 1 | 0 |
| corridor | 40 | 41 | 44 |
| offroad | 16 | 19 | 12 |

## Partial scores (0 < s < 0.98)

| type | STANDFMXH-L18-s1+ACC+GAP | JTN-L18-s0+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 28 | 30 | 26 |
| no-launch | 1 | 0 | 0 |
| creep | 1 | 0 | 0 |
| slow | 8 | 12 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | STANDFMXH-L18-s1+ACC+GAP mean | NTV0-L18-s0+ACC+GAP mean | diff | contribution | STANDFMXH-L18-s1+ACC+GAP zeros | NTV0-L18-s0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (JTN-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (JTN-L18-s0+ACC+GAP) launch | 37 | 0.7226 | 0.7738 | -0.0512 | -0.0068 | 2 | 3 | 1 | 2 |
| launch | 37 | 0.6980 | 0.7738 | -0.0757 | -0.0101 | 2 | 3 | 1 | 2 |
| (JTN-L18-s0+ACC+GAP) moving | 239 | 0.7108 | 0.7396 | -0.0288 | -0.0247 | 62 | 55 | 9 | 2 |
| moving | 239 | 0.7188 | 0.7396 | -0.0208 | -0.0179 | 61 | 55 | 9 | 3 |

## New zeros (zero for STANDFMXH-L18-s1+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 10

| scene | stratum | STANDFMXH-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok | 0.000 offroad |
| 76892966 | moving | 0.000 AF-front | 1.000 ok | 0.000 AF-front |
| 7d6cda9e | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit | 0.000 offroad |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok | 0.000 corridor |
| 062def2d | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| a57e8677 | launch | 0.000 offroad | 1.000 ok | 1.000 ok |
| 7c0fb633 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| d5527d43 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for NTV0-L18-s0+ACC+GAP, scored for STANDFMXH-L18-s1+ACC+GAP): 5

| scene | stratum | STANDFMXH-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 9e733560 | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| c55652d3 | launch | 1.000 ok | 0.000 corridor | 0.011 rear-hit |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| ddd61134 | launch | 0.059 rear-hit | 0.000 corridor | 0.000 corridor |
| 831a442e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
