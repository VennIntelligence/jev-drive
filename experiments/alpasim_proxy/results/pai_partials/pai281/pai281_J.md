# PAI 281: J against NTV0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/alpasim_proxy/pai_partials/J`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| J | 280 | 0.7497 | 65 | 200 | 0.907 | 0.911 |
| NTV0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (J - NTV0+ACC+GAP): **+0.0055 [-0.0233, +0.0342]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0214 [-0.0036, +0.0500] (+6 zeros).


## Zero types

| type | J | NTV0+ACC+GAP |
|---|--:|--:|
| AF-front | 4 | 3 |
| AF-side | 1 | 0 |
| corridor | 48 | 44 |
| offroad | 12 | 12 |

## Partial scores (0 < s < 0.98)

| type | J | NTV0+ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 11 | 26 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 4 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | J mean | NTV0+ACC+GAP mean | diff | contribution | J zeros | NTV0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7869 | 0.7738 | +0.0131 | +0.0017 | 5 | 3 | 2 | 0 |
| moving | 241 | 0.7461 | 0.7418 | +0.0043 | +0.0037 | 59 | 55 | 9 | 5 |

## New zeros (zero for J, scored for NTV0+ACC+GAP): 11

| scene | stratum | J | NTV0+ACC+GAP |
|---|---|---|---|
| 300a909c | moving | 0.000 corridor | 1.000 ok |
| 4fcb5fde | launch | 0.000 AF-side | 1.000 ok |
| 353e8268 | moving | 0.000 corridor | 1.000 ok |
| 28281e1e | moving | 0.000 corridor | 1.000 ok |
| 9d585b78 | moving | 0.000 offroad | 1.000 ok |
| dd72f5f4 | launch | 0.000 AF-front | 1.000 ok |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit |
| 947ee626 | moving | 0.000 corridor | 1.000 ok |
| 64b0bb90 | moving | 0.000 corridor | 1.000 ok |
| 89fae1ac | moving | 0.000 offroad | 0.960 slow |
| ff7ccca9 | moving | 0.000 corridor | 0.559 rear-hit |

## Fixed zeros (zero for NTV0+ACC+GAP, scored for J): 5

| scene | stratum | J | NTV0+ACC+GAP |
|---|---|---|---|
| 7e8df1e3 | moving | 1.000 ok | 0.000 corridor |
| 787c8ada | moving | 1.000 ok | 0.000 AF-front |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor |
| 8ca1f1e0 | moving | 1.000 ok | 0.000 offroad |
| 7f699124 | moving | 1.000 ok | 0.000 offroad |
