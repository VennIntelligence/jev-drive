# PAI 281: HL against NTV0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0-hl`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 280 of 281 (excluded: J [] [], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| HL | 280 | 0.7447 | 61 | 190 | 0.884 | 0.878 |
| NTV0+ACC+GAP | 280 | 0.7443 | 59 | 184 | 0.882 | 0.875 |

Paired mean (HL - NTV0+ACC+GAP): **+0.0004 [-0.0136, +0.0134]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0071 [-0.0071, +0.0214] (+2 zeros).


## Zero types

| type | HL | NTV0+ACC+GAP |
|---|--:|--:|
| AF-front | 4 | 3 |
| AF-side | 1 | 0 |
| corridor | 42 | 44 |
| offroad | 14 | 12 |

## Partial scores (0 < s < 0.98)

| type | HL | NTV0+ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 21 | 26 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 8 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | HL mean | NTV0+ACC+GAP mean | diff | contribution | HL zeros | NTV0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7977 | 0.7738 | +0.0239 | +0.0032 | 3 | 3 | 0 | 0 |
| moving | 241 | 0.7386 | 0.7418 | -0.0032 | -0.0027 | 57 | 55 | 3 | 1 |

## New zeros (zero for HL, scored for NTV0+ACC+GAP): 3

| scene | stratum | HL | NTV0+ACC+GAP |
|---|---|---|---|
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit |
| b8d5d0d3 | moving | 0.000 AF-side | 1.000 ok |
| 6f6aaecc | moving | 0.000 offroad | 1.000 ok |

## Fixed zeros (zero for NTV0+ACC+GAP, scored for HL): 1

| scene | stratum | HL | NTV0+ACC+GAP |
|---|---|---|---|
| 8ca1f1e0 | moving | 1.000 ok | 0.000 offroad |
