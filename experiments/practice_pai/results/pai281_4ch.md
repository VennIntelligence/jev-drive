# PAI 281: pilot against RSG+ACC+GAP, paired

J = `/root/autodl-tmp/ujs/runs/practice_pai/read/PPAIP-L18-s0`, reference = `/root/autodl-tmp/ujs/runs/route_follow/acc_gap/read/s2`. Scenes scored by both without a driver RPC error: 144 of 281 (excluded: J [] [], reference []).

| | scenes | mean | zeros | scored 1 | progress_rel | log distance covered |
|---|--:|--:|--:|--:|--:|--:|
| pilot | 144 | 0.6608 | 45 | 87 | 0.894 | 0.842 |
| control | 144 | 0.6707 | 44 | 89 | 0.894 | 0.845 |
| RSG+ACC+GAP | 144 | 0.6758 | 43 | 89 | 0.894 | 0.850 |

Paired mean (pilot - RSG+ACC+GAP): **-0.0150 [-0.0542, +0.0214]** (log-clustered bootstrap 95 % CI, 10 000 draws, seed 0). Zero-rate difference: +0.0139 [-0.0208, +0.0556] (+2 zeros).


Paired mean (control - RSG+ACC+GAP): **-0.0051 [-0.0336, +0.0235]**; zeros +1.

Paired mean (pilot - control): **-0.0099 [-0.0531, +0.0328]**; zeros +1.

## Zero types

| type | pilot | control | RSG+ACC+GAP |
|---|--:|--:|--:|
| AF-front | 3 | 3 | 4 |
| AF-side | 3 | 1 | 1 |
| corridor | 32 | 31 | 31 |
| offroad | 7 | 9 | 7 |

## Partial scores (0 < s < 0.98)

| type | pilot | control | RSG+ACC+GAP |
|---|--:|--:|--:|
| hit | 0 | 0 | 0 |
| rear-hit | 11 | 10 | 11 |
| no-launch | 0 | 0 | 0 |
| creep | 0 | 0 | 0 |
| slow | 1 | 1 | 1 |

## Strata (contribution = sum of the score differences in the stratum / all scenes; sums to the paired mean)

| stratum | scenes | pilot mean | RSG+ACC+GAP mean | diff | contribution | pilot zeros | RSG+ACC+GAP zeros | new | fixed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (control) static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| static | 2 | 0.5000 | 0.5000 | +0.0000 | +0.0000 | 1 | 1 | 0 | 0 |
| (control) launch | 16 | 0.6793 | 0.6805 | -0.0013 | -0.0001 | 3 | 3 | 0 | 0 |
| launch | 16 | 0.6769 | 0.6805 | -0.0037 | -0.0004 | 3 | 3 | 0 | 0 |
| (control) moving | 126 | 0.6723 | 0.6780 | -0.0057 | -0.0050 | 40 | 39 | 3 | 2 |
| moving | 126 | 0.6613 | 0.6780 | -0.0167 | -0.0146 | 41 | 39 | 5 | 3 |

## New zeros (zero for pilot, scored for RSG+ACC+GAP): 5

| scene | stratum | pilot | RSG+ACC+GAP | control |
|---|---|---|---|---|
| 300a909c | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| e69f06a1 | moving | 0.000 AF-side | 0.848 rear-hit | 0.848 rear-hit |
| 29d87ce7 | moving | 0.000 AF-front | 1.000 ok | 1.000 ok |
| 244b1bf1 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |
| f2eaa260 | moving | 0.000 corridor | 1.000 ok | 1.000 ok |

## Fixed zeros (zero for RSG+ACC+GAP, scored for pilot): 3

| scene | stratum | pilot | RSG+ACC+GAP | control |
|---|---|---|---|---|
| 62ec6f99 | moving | 1.000 ok | 0.000 AF-front | 0.000 AF-front |
| 7d6cda9e | moving | 1.000 ok | 0.000 corridor | 0.000 corridor |
| 7befde90 | moving | 1.000 ok | 0.000 AF-front | 1.000 ok |
