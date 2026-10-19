# PAI 281: BNC2K-L18-s1+ACC+GAP against JTN-L18-s1+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/BNC2K-L18-s1`, reference = `/root/autodl-tmp/ujs/runs/defect_fix/pai_accgap/JTN-L18-s1`. Scenes scored by both without a driver RPC error: 279 of 281 (excluded: J [] [], reference ['1f7fddbc']).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| BNC2K-L18-s1+ACC+GAP | 279 | 0.7132 | 65 | 173 | 0.872 | 0.847 |
| NTV0-L18-s0+ACC+GAP | 279 | 0.7434 | 59 | 183 | 0.881 | 0.875 |
| JTN-L18-s1+ACC+GAP | 279 | 0.7025 | 68 | 172 | 0.873 | 0.848 |

Paired mean (BNC2K-L18-s1+ACC+GAP - JTN-L18-s1+ACC+GAP): **+0.0107 [-0.0153, +0.0361]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: -0.0108 [-0.0394, +0.0179] (-3 zeros).


Paired mean (NTV0-L18-s0+ACC+GAP - JTN-L18-s1+ACC+GAP): **+0.0409 [+0.0161, +0.0675]**; zeros -9.

Paired mean (BNC2K-L18-s1+ACC+GAP - NTV0-L18-s0+ACC+GAP): **-0.0302 [-0.0614, -0.0004]**; zeros +6.

## Zero types

| type | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s1+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 8 | 3 | 6 |
| AF-side | 0 | 0 | 2 |
| corridor | 41 | 44 | 42 |
| offroad | 16 | 12 | 18 |

## Partial scores (0 < s < 0.98)

| type | BNC2K-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP | JTN-L18-s1+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 30 | 26 | 28 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 11 | 11 | 11 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | BNC2K-L18-s1+ACC+GAP mean | JTN-L18-s1+ACC+GAP mean | diff | contribution | BNC2K-L18-s1+ACC+GAP zeros | JTN-L18-s1+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (NTV0-L18-s0+ACC+GAP) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (NTV0-L18-s0+ACC+GAP) launch | 37 | 0.7738 | 0.7269 | +0.0468 | +0.0062 | 3 | 2 | 1 | 0 |
| launch | 37 | 0.7035 | 0.7269 | -0.0235 | -0.0031 | 3 | 2 | 2 | 1 |
| (NTV0-L18-s0+ACC+GAP) moving | 240 | 0.7407 | 0.7004 | +0.0403 | +0.0346 | 55 | 65 | 2 | 12 |
| moving | 240 | 0.7165 | 0.7004 | +0.0160 | +0.0138 | 61 | 65 | 4 | 8 |

## New zeros (zero for BNC2K-L18-s1+ACC+GAP, scored for JTN-L18-s1+ACC+GAP): 6

| scene | stratum | BNC2K-L18-s1+ACC+GAP | JTN-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 90f9cd38 | launch | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 4fcb5fde | launch | 0.000 AF-front | 0.798 rear-hit | 1.000 ok |
| 7091c0b9 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 1f5c9e1f | moving | 0.000 corridor | 0.557 rear-hit | 0.565 rear-hit |
| 1438905d | moving | 0.000 offroad | 1.000 ok | 1.000 ok |
| b7a8437e | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for JTN-L18-s1+ACC+GAP, scored for BNC2K-L18-s1+ACC+GAP): 9

| scene | stratum | BNC2K-L18-s1+ACC+GAP | JTN-L18-s1+ACC+GAP | NTV0-L18-s0+ACC+GAP |
|---|---|---|---|---|
| 7e8df1e3 | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 7aeba6e8 | moving | 1.000 ok | 0.000 AF-front | 0.000 corridor |
| cc46d505 | moving | 1.000 ok | 0.000 offroad | 0.000 corridor |
| 25e9675e | moving | 0.620 rear-hit | 0.000 offroad | 0.615 rear-hit |
| 69fbc3cb | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 90f1c66d | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 7c0fb633 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
| 244b1bf1 | moving | 1.000 ok | 0.000 corridor | 1.000 ok |
| 3793ab49 | launch | 0.488 rear-hit | 0.000 offroad | 0.000 corridor |
