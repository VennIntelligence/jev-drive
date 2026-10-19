# PAI 281: KW (VCONT+LEAD) against VCONT+LEAD, paired

J = `/root/autodl-tmp/ujs/runs/route_follow/pai_input_loop/KN`, reference = `/root/autodl-tmp/ujs/runs/route_follow/pai_input_loop/N0`. Scenes scored by both without a driver RPC error: 281 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| KW (VCONT+LEAD) | 281 | 0.5721 | 59 | 86 | 0.741 | 0.583 |
| VCONT+LEAD | 281 | 0.6114 | 59 | 105 | 0.770 | 0.615 |

Paired mean (KW (VCONT+LEAD) - VCONT+LEAD): **-0.0393 [-0.0762, -0.0018]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0000 [-0.0391, +0.0391] (+0 zeros).


## Zero types

| type | KW (VCONT+LEAD) | VCONT+LEAD |
|---|--:|--:|
| AF-front | 8 | 2 |
| AF-side | 3 | 3 |
| corridor | 40 | 39 |
| offroad | 8 | 15 |

## Partial scores (0 < s < 0.98)

| type | KW (VCONT+LEAD) | VCONT+LEAD |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 92 | 75 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 44 | 42 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | KW (VCONT+LEAD) mean | VCONT+LEAD mean | diff | contribution | KW (VCONT+LEAD) zeros | VCONT+LEAD zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.4236 | 0.5028 | -0.0791 | -0.0104 | 8 | 8 | 3 | 3 |
| moving | 242 | 0.5954 | 0.6290 | -0.0335 | -0.0289 | 50 | 50 | 12 | 12 |

## New zeros (zero for KW (VCONT+LEAD), scored for VCONT+LEAD): 15

| scene | stratum | KW (VCONT+LEAD) | VCONT+LEAD |
|---|---|---|---|
| 787c8ada | moving | 0.000 AF-front | 0.277 rear-hit |
| 6ab8fbf5 | moving | 0.000 offroad | 1.000 ok |
| 4fcb5fde | launch | 0.000 corridor | 0.232 rear-hit |
| 28281e1e | moving | 0.000 corridor | 0.927 slow |
| 9d585b78 | moving | 0.000 corridor | 1.000 ok |
| c425dcf0 | moving | 0.000 corridor | 1.000 ok |
| 11250b18 | moving | 0.000 corridor | 1.000 ok |
| a57e8677 | launch | 0.000 AF-front | 1.000 ok |
| 97ac3ade | moving | 0.000 corridor | 1.000 ok |
| b8d5d0d3 | moving | 0.000 AF-front | 1.000 ok |
| 5793734a | moving | 0.000 AF-front | 0.657 rear-hit |
| bd7214b6 | launch | 0.000 AF-front | 0.615 rear-hit |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok |
| 4eb422d9 | moving | 0.000 offroad | 1.000 ok |
| bd6a4d25 | moving | 0.000 AF-side | 1.000 ok |

## Fixed zeros (zero for VCONT+LEAD, scored for KW (VCONT+LEAD)): 15

| scene | stratum | KW (VCONT+LEAD) | VCONT+LEAD |
|---|---|---|---|
| 361845f6 | moving | 0.846 rear-hit | 0.000 offroad |
| 569fab77 | moving | 1.000 ok | 0.000 corridor |
| aeb986bc | moving | 0.888 rear-hit | 0.000 corridor |
| 8b71fa55 | moving | 0.883 rear-hit | 0.000 offroad |
| 5fb9de99 | moving | 0.350 rear-hit | 0.000 corridor |
| 1f5c9e1f | moving | 0.267 rear-hit | 0.000 offroad |
| 8b28b8f9 | launch | 1.000 ok | 0.000 AF-front |
| 3fd457e2 | moving | 1.000 ok | 0.000 corridor |
| 9f243b4a | moving | 0.668 rear-hit | 0.000 corridor |
| 75c7d11b | moving | 1.000 ok | 0.000 corridor |
| fd7779a6 | launch | 0.220 rear-hit | 0.000 offroad |
| 059bbccf | moving | 1.000 ok | 0.000 offroad |
| 9b3ca23d | launch | 1.000 ok | 0.000 AF-side |
| a3bad8dd | moving | 1.000 ok | 0.000 offroad |
| 831a442e | moving | 1.000 ok | 0.000 corridor |
