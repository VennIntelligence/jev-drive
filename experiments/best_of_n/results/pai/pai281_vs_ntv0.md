# PAI 281: BNC2K-L18-s1+ACC+GAP against NTV0-L18-s0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/BNC2K-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| BNC2K-L18-s1+ACC+GAP | 280 | 0.7142 | 65 | 174 | 0.873 | 0.847 |
| NTV0-L18-s0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (BNC2K-L18-s1+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0301 [-0.0616, +0.0004]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0214 [-0.0107, +0.0536] (+6 zeros).


## Zero types

| type | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|
| AF-front | 8 | 3 |
| corridor | 41 | 44 |
| offroad | 16 | 12 |

## Partial scores (0 < s < 0.98)

| type | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 30 | 26 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 11 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | BNC2K-L18-s1+ACC+GAP mean | NTV0-L18-s0+ACC+GAP mean | diff | contribution | BNC2K-L18-s1+ACC+GAP zeros | NTV0-L18-s0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7035 | 0.7738 | -0.0703 | -0.0093 | 3 | 3 | 2 | 2 |
| moving | 241 | 0.7176 | 0.7418 | -0.0241 | -0.0208 | 61 | 55 | 12 | 6 |

## New zeros (zero for BNC2K-L18-s1+ACC+GAP, scored for NTV0-L18-s0+ACC+GAP): 14

| scene | stratum | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|
| fdb7652a | moving | 0.000 offroad | 1.000 ok |
| 76892966 | moving | 0.000 AF-front | 1.000 ok |
| 90f9cd38 | launch | 0.000 AF-front | 1.000 ok |
| 4fcb5fde | launch | 0.000 AF-front | 1.000 ok |
| 0de150b0 | moving | 0.000 offroad | 0.757 rear-hit |
| 7d6cda9e | moving | 0.000 corridor | 1.000 ok |
| 41f15fb9 | moving | 0.000 corridor | 1.000 ok |
| 29d87ce7 | moving | 0.000 corridor | 1.000 ok |
| 7091c0b9 | moving | 0.000 AF-front | 1.000 ok |
| 1f5c9e1f | moving | 0.000 corridor | 0.565 rear-hit |
| 1438905d | moving | 0.000 offroad | 1.000 ok |
| b8d5d0d3 | moving | 0.000 AF-front | 1.000 ok |
| b7a8437e | moving | 0.000 corridor | 1.000 ok |
| d5527d43 | moving | 0.000 corridor | 1.000 ok |

## Fixed zeros (zero for NTV0-L18-s0+ACC+GAP, scored for BNC2K-L18-s1+ACC+GAP): 8

| scene | stratum | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|
| 7e8df1e3 | moving | 1.000 ok | 0.000 corridor |
| 9e733560 | moving | 1.000 ok | 0.000 corridor |
| 7aeba6e8 | moving | 1.000 ok | 0.000 corridor |
| cc46d505 | moving | 1.000 ok | 0.000 corridor |
| c55652d3 | launch | 0.006 rear-hit | 0.000 corridor |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor |
| 90f1c66d | moving | 1.000 ok | 0.000 corridor |
| 3793ab49 | launch | 0.488 rear-hit | 0.000 corridor |
