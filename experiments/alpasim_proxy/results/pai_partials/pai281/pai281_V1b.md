# PAI 281: V1b against NTV0+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/alpasim_proxy/pai_partials/V1b`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/NTV0-L18-s0`. Scenes scored by both without a driver RPC error: 279 of 281 (excluded: J ['5f7ace44'] ['<AioRpcError of RPC that terminated with:'], reference ['1059fc66']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| V1b | 279 | 0.7528 | 63 | 201 | 0.905 | 0.907 |
| NTV0+ACC+GAP | 279 | 0.7454 | 59 | 184 | 0.882 | 0.877 |

Paired mean (V1b - NTV0+ACC+GAP): **+0.0074 [-0.0170, +0.0318]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0143 [-0.0108, +0.0394] (+4 zeros).


## Zero types

| type | V1b | NTV0+ACC+GAP |
|---|--:|--:|
| AF-front | 6 | 3 |
| corridor | 45 | 44 |
| offroad | 12 | 12 |

## Partial scores (0 < s < 0.98)

| type | V1b | NTV0+ACC+GAP |
|---|--:|--:|
| hit | 0 | 0 |
| rear-hit | 11 | 25 |
| no-launch | 0 | 0 |
| creep | 0 | 0 |
| slow | 4 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | V1b mean | NTV0+ACC+GAP mean | diff | contribution | V1b zeros | NTV0+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| launch | 37 | 0.7914 | 0.7738 | +0.0177 | +0.0023 | 4 | 3 | 1 | 0 |
| moving | 240 | 0.7490 | 0.7431 | +0.0059 | +0.0051 | 58 | 55 | 7 | 4 |

## New zeros (zero for V1b, scored for NTV0+ACC+GAP): 8

| scene | stratum | V1b | NTV0+ACC+GAP |
|---|---|---|---|
| b64a6e11 | moving | 0.000 AF-front | 1.000 ok |
| 28281e1e | moving | 0.000 corridor | 1.000 ok |
| 9d585b78 | moving | 0.000 offroad | 1.000 ok |
| dd72f5f4 | launch | 0.000 AF-front | 1.000 ok |
| 25e9675e | moving | 0.000 offroad | 0.615 rear-hit |
| 1f5c9e1f | moving | 0.000 corridor | 0.565 rear-hit |
| ff7ccca9 | moving | 0.000 corridor | 0.559 rear-hit |
| 7c0fb633 | moving | 0.000 corridor | 1.000 ok |

## Fixed zeros (zero for NTV0+ACC+GAP, scored for V1b): 4

| scene | stratum | V1b | NTV0+ACC+GAP |
|---|---|---|---|
| 7aeba6e8 | moving | 1.000 ok | 0.000 corridor |
| 88b2c556 | moving | 1.000 ok | 0.000 corridor |
| 90f1c66d | moving | 1.000 ok | 0.000 corridor |
| 8ca1f1e0 | moving | 1.000 ok | 0.000 offroad |
