# PAI 281: BNC2K-L18-s1+ACC+GAP against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/BNC2K-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 281 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| BNC2K-L18-s1+ACC+GAP | 281 | 0.7152 | 65 | 175 | 0.872 | 0.847 |
| RSG+ACC+GAP | 281 | 0.6909 | 76 | 171 | 0.886 | 0.852 |

Paired mean (BNC2K-L18-s1+ACC+GAP - RSG+ACC+GAP): **+0.0243 [-0.0125, +0.0632]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0391 [-0.0783, -0.0036] (-11 zeros).


## Zero types

| type | BNC2K-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|
| AF-front | 8 | 7 |
| AF-side | 0 | 2 |
| corridor | 41 | 53 |
| offroad | 16 | 14 |

## Partial scores (0 < s < 0.98)

| type | BNC2K-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 30 | 26 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 11 | 8 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | BNC2K-L18-s1+ACC+GAP mean | RSG+ACC+GAP mean | diff | contribution | BNC2K-L18-s1+ACC+GAP zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7035 | 0.7545 | -0.0510 | -0.0067 | 3 | 4 | 1 | 2 |
| moving | 242 | 0.7188 | 0.6827 | +0.0361 | +0.0311 | 61 | 71 | 8 | 18 |

## New zeros (zero for BNC2K-L18-s1+ACC+GAP, scored for RSG+ACC+GAP): 9

| scene | stratum | BNC2K-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok |
| 4fcb5fde | launch | 0.000 AF-front | 0.819 rear-hit |
| 0de150b0 | moving | 0.000 offroad | 0.947 rear-hit |
| 41f15fb9 | moving | 0.000 corridor | 1.000 ok |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok |
| 7091c0b9 | moving | 0.000 AF-front | 1.000 ok |
| 1438905d | moving | 0.000 offroad | 1.000 ok |
| b8d5d0d3 | moving | 0.000 AF-front | 1.000 ok |
| b7a8437e | moving | 0.000 corridor | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for BNC2K-L18-s1+ACC+GAP): 20

| scene | stratum | BNC2K-L18-s1+ACC+GAP | RSG+ACC+GAP |
|---|---|---|---|
| 7e8df1e3 | moving | 1.000 ok | 0.000 corridor |
| 9e733560 | moving | 1.000 ok | 0.000 corridor |
| be9320f3 | moving | 1.000 ok | 0.000 corridor |
| 58eb6cd6 | moving | 1.000 ok | 0.000 corridor |
| 353e8268 | moving | 1.000 ok | 0.000 corridor |
| 28281e1e | moving | 1.000 ok | 0.000 corridor |
| cc46d505 | moving | 1.000 ok | 0.000 corridor |
| 25e9675e | moving | 0.620 rear-hit | 0.000 offroad |
| 272b7bf0 | moving | 0.837 slow | 0.000 corridor |
| 89fae1ac | moving | 0.938 slow | 0.000 offroad |
| 90f1c66d | moving | 1.000 ok | 0.000 corridor |
| 8832a062 | moving | 1.000 ok | 0.000 corridor |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front |
| 6f6aaecc | moving | 1.000 ok | 0.000 offroad |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front |
| 4f568cd4 | moving | 0.824 slow | 0.000 corridor |
| 9b3ca23d | launch | 1.000 ok | 0.000 corridor |
| bd6a4d25 | moving | 1.000 ok | 0.000 AF-side |
| 70d5986f | moving | 1.000 ok | 0.000 AF-side |
| 3793ab49 | launch | 0.488 rear-hit | 0.000 corridor |
