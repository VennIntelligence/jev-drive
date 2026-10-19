# PAI 281: KW+ACC+GAP against ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/route_follow/pai_input_loop/KA`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| KW+ACC+GAP | 280 | 0.6747 | 62 | 152 | 0.844 | 0.794 |
| ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (KW+ACC+GAP - ACC+GAP): **-0.0696 [-0.1083, -0.0312]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0107 [-0.0286, +0.0464] (+3 zeros).


## Zero types

| type | KW+ACC+GAP | ACC+GAP |
|---|--:|--:|
| AF-front | 6 | 3 |
| AF-side | 4 | 0 |
| corridor | 39 | 44 |
| offroad | 13 | 12 |

## Partial scores (0 < s < 0.98)

| type | KW+ACC+GAP | ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 53 | 26 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 13 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | KW+ACC+GAP mean | ACC+GAP mean | diff | contribution | KW+ACC+GAP zeros | ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.5844 | 0.7738 | -0.1894 | -0.0250 | 5 | 3 | 3 | 1 |
| moving | 241 | 0.6900 | 0.7418 | -0.0517 | -0.0445 | 56 | 55 | 13 | 12 |

## New zeros (zero for KW+ACC+GAP, scored for ACC+GAP): 16

| scene | stratum | KW+ACC+GAP | ACC+GAP |
|---|---|---|---|
| 6ab8fbf5 | moving | 0.000 offroad | 1.000 ok |
| 4fcb5fde | launch | 0.000 corridor | 1.000 ok |
| f4f7641f | moving | 0.000 corridor | 1.000 ok |
| 9d585b78 | moving | 0.000 corridor | 1.000 ok |
| 0de150b0 | moving | 0.000 offroad | 0.757 rear-hit |
| 062def2d | moving | 0.000 AF-side | 1.000 ok |
| 7091c0b9 | moving | 0.000 AF-front | 1.000 ok |
| ff7ccca9 | moving | 0.000 corridor | 0.559 rear-hit |
| a57e8677 | launch | 0.000 AF-front | 1.000 ok |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok |
| 6f6aaecc | moving | 0.000 offroad | 1.000 ok |
| 4d66b2d9 | moving | 0.000 AF-side | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok |
| fd7779a6 | launch | 0.000 offroad | 0.462 rear-hit |
| 4eb422d9 | moving | 0.000 offroad | 1.000 ok |
| bd6a4d25 | moving | 0.000 AF-front | 1.000 ok |

## Fixed zeros (zero for ACC+GAP, scored for KW+ACC+GAP): 13

| scene | stratum | KW+ACC+GAP | ACC+GAP |
|---|---|---|---|
| fde5cd20 | moving | 1.000 ok | 0.000 corridor |
| 6c88b571 | moving | 0.201 rear-hit | 0.000 offroad |
| 942cb4ca | moving | 1.000 ok | 0.000 corridor |
| 975e6436 | moving | 0.408 rear-hit | 0.000 offroad |
| aeb986bc | moving | 0.986 ok | 0.000 corridor |
| 5fb9de99 | moving | 0.389 rear-hit | 0.000 corridor |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor |
| 90f1c66d | moving | 1.000 ok | 0.000 corridor |
| 3c1893bb | moving | 1.000 ok | 0.000 offroad |
| 8ca1f1e0 | moving | 1.000 ok | 0.000 offroad |
| 7f699124 | moving | 1.000 ok | 0.000 offroad |
| 831a442e | moving | 1.000 ok | 0.000 corridor |
| 3793ab49 | launch | 0.552 slow | 0.000 corridor |
