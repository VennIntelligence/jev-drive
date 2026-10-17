# PAI corridor: per-step scorer metrics of the stored PAI 281 reads

Human tracks: 281 scenes; GT length median 163.9 m, < 5 m: 2; end state still / moving / off-heading: 41 / 238 / 2; human speed at the 1.7 s handoff < 0.3 m/s (leash latch): 34

## Per read: zeros, corridor zeros, and corridor zeros on still-end scenes

A corridor zero = score 0 with left_corridor_laterally set (the failure reason can read offroad / collision, which take precedence).

| read | rollouts | mean | zeros | failure = corridor | corridor zeros | tripped (d >= 4 m) | still-end scenes | still-end corridor zeros | of which the trip is past the GT end (progress >= 0.99) | of which latched |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| g10 | 280 | 0.5713 | 75 | 50 | 60 | 116 | 41 | 17 | 14 | 1 |
| g10hl | 281 | 0.6124 | 76 | 52 | 61 | 120 | 41 | 18 | 15 | 1 |
| acc | 280 | 0.6760 | 75 | 50 | 57 | 134 | 40 | 19 | 16 | 1 |
| accgap | 281 | 0.6909 | 76 | 53 | 60 | 143 | 41 | 21 | 16 | 1 |

## The rule on PAI: corridor zeros given a trip, by the human's end state

| read | end | scenes | tripped | corridor zeros | zero / tripped | tripped past the GT end | corridor zeros among them | median lateral / d there |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| g10 | moving | 237 | 98 | 42 | 0.43 | 58 | 3 | 0.11 |
| g10 | off-heading | 2 | 1 | 1 | 1.00 | 1 | 1 | 1.00 |
| g10 | still | 41 | 17 | 17 | 1.00 | 14 | 14 | 1.00 |
| g10hl | moving | 238 | 100 | 42 | 0.42 | 59 | 3 | 0.12 |
| g10hl | off-heading | 2 | 1 | 1 | 1.00 | 1 | 1 | 1.00 |
| g10hl | still | 41 | 19 | 18 | 0.95 | 16 | 15 | 1.00 |
| acc | moving | 238 | 112 | 37 | 0.33 | 74 | 2 | 0.11 |
| acc | off-heading | 2 | 1 | 1 | 1.00 | 1 | 1 | 1.00 |
| acc | still | 40 | 21 | 19 | 0.90 | 18 | 16 | 1.00 |
| accgap | moving | 238 | 118 | 38 | 0.32 | 80 | 2 | 0.10 |
| accgap | off-heading | 2 | 1 | 1 | 1.00 | 1 | 1 | 1.00 |
| accgap | still | 41 | 24 | 21 | 0.88 | 19 | 16 | 1.00 |

## Latched scenes (human < 0.3 m/s at the handoff): what each read did, and the leash3 estimate

Leash3 offline: the ego is kept within 3 m of its handoff point for the whole session. Then d <= 3 m < 4 m (the latch point is on the GT), so no corridor zero; progress score = 1 if the GT is < 5 m, else min(1, s / (0.8 L)) with s <= 3 m the ego's progress along the GT (upper bound: s = 3). Collisions / offroad inside the leash are not modelled (nuPlan leash3 added 11 front collisions on 1 004 latched sessions).

| scene | GT length (m) | end | human end-still (s) | human first > 3 m from the handoff point (s) | g10 score / fail / arc after handoff (m) | g10hl score / fail / arc after handoff (m) | acc score / fail / arc after handoff (m) | accgap score / fail / arc after handoff (m) | leash3 score (upper) |
|---|--:|---|--:|--:|---|---|---|---|--:|
| 5fcb2ba7 | 0.0 | still | 20.0 | never | 1.000 / - / 0.0 | 1.000 / - / 0.0 | 1.000 / - / 0.0 | 1.000 / - / 2.9 | 1.000 |
| 3fe2bac8 | 0.2 | still | 19.4 | never | 0.000 / left_corridor_laterally / 28.1 | 0.000 / left_corridor_laterally / 34.3 | 0.000 / left_corridor_laterally / 22.0 | 0.000 / left_corridor_laterally / 23.4 | 1.000 |
| cbb22323 | 12.0 | still | 5.2 | 7.7 | 0.821 / - / 8.5 | 0.782 / - / 8.2 | 0.655 / - / 7.3 | 1.000 / - / 10.3 | 0.313 |
| 7c6529f6 | 12.6 | moving | 0.0 | 13.1 | 0.222 / - / 2.1 | 0.222 / - / 2.1 | 0.222 / - / 2.1 | 0.504 / - / 4.8 | 0.297 |
| f18c6340 | 19.5 | still | 2.0 | 7.9 | 0.153 / - / 12.1 | 0.153 / - / 12.1 | 0.162 / - / 12.8 | 0.625 / - / 12.6 | 0.192 |
| 81dc4ece | 21.8 | moving | 0.0 | 14.3 | 0.863 / - / 14.8 | 0.863 / - / 14.8 | 0.977 / - / 16.8 | 1.000 / - / 20.1 | 0.172 |
| 5266c7c9 | 26.0 | moving | 0.0 | 4.5 | 0.957 / - / 20.0 | 0.911 / - / 19.1 | 1.000 / - / 21.7 | 1.000 / - / 26.3 | 0.144 |
| 9fa1d6eb | 34.8 | moving | 0.0 | 4.4 | 1.000 / - / 65.0 | 1.000 / - / 64.8 | 1.000 / - / 47.9 | 1.000 / - / 48.7 | 0.108 |
| ca7accbc | 40.2 | moving | 0.0 | 8.0 | 0.708 / - / 23.3 | 0.708 / - / 23.3 | 0.383 / - / 17.2 | 1.000 / - / 67.6 | 0.093 |
| de9db5dc | 41.0 | moving | 0.0 | 5.7 | 0.000 / collision_at_fault / 40.7 | 0.000 / collision_at_fault / 40.7 | 0.000 / collision_at_fault / 50.7 | 0.000 / left_corridor_laterally / 48.3 | 0.091 |
| 22fa5c92 | 43.2 | moving | 0.0 | 8.3 | 0.000 / left_corridor_laterally / 44.5 | 0.000 / left_corridor_laterally / 44.5 | 0.000 / collision_at_fault / 50.9 | 0.000 / collision_at_fault / 44.9 | 0.087 |
| 21829be5 | 44.6 | moving | 0.0 | 8.8 | 0.692 / - / 34.1 | 0.692 / - / 34.0 | 0.606 / - / 28.3 | 0.595 / - / 30.8 | 0.084 |
| c8cab401 | 44.8 | moving | 0.0 | 5.2 | 1.000 / - / 40.4 | 1.000 / - / 40.4 | 1.000 / - / 39.5 | 1.000 / - / 43.5 | 0.084 |
| 68e53160 | 49.9 | moving | 0.0 | 10.2 | 0.000 / left_corridor_laterally / 81.6 | 0.000 / left_corridor_laterally / 81.7 | 1.000 / - / 93.3 | 1.000 / - / 94.8 | 0.075 |
| a1b166cc | 56.0 | moving | 0.0 | 9.0 | 0.000 / left_corridor_laterally / 45.9 | 0.000 / left_corridor_laterally / 45.9 | 1.000 / - / 75.5 | 1.000 / - / 71.2 | 0.067 |
| 2b577fc2 | 62.0 | moving | 0.0 | 5.0 | 1.000 / - / 51.7 | 1.000 / - / 53.1 | 0.825 / - / 43.3 | 0.851 / - / 48.7 | 0.061 |
| 7b5c9fd6 | 69.4 | moving | 0.0 | 7.6 | 0.107 / - / 52.2 | 0.107 / - / 52.2 | 0.247 / - / 53.6 | 1.000 / - / 59.6 | 0.054 |
| 268949d2 | 76.2 | moving | 0.0 | 6.2 | 0.000 / left_corridor_laterally / 104.6 | 0.000 / left_corridor_laterally / 104.6 | 0.000 / left_corridor_laterally / 73.3 | 0.819 / - / 67.2 | 0.049 |
| c7956ed6 | 90.7 | moving | 0.0 | 3.2 | 1.000 / - / 82.6 | 1.000 / - / 82.6 | 1.000 / - / 82.9 | 1.000 / - / 85.4 | 0.041 |
| 2d023524 | 93.0 | moving | 0.0 | 9.0 | 1.000 / - / 86.0 | 1.000 / - / 86.0 | 1.000 / - / 99.4 | 1.000 / - / 105.8 | 0.040 |
| a1f07c94 | 98.2 | moving | 0.0 | 6.7 | 0.738 / - / 57.6 | 0.789 / - / 61.5 | 1.000 / - / 81.1 | 1.000 / - / 81.1 | 0.038 |
| a686c7d0 | 98.6 | moving | 0.0 | 7.2 | 0.051 / - / 52.9 | 0.051 / - / 53.8 | 0.072 / - / 43.4 | 0.189 / - / 45.3 | 0.038 |
| b44a4a44 | 99.7 | moving | 0.0 | 6.5 | 0.000 / collision_at_fault / 27.0 | 0.000 / collision_at_fault / 27.4 | 0.000 / collision_at_fault / 48.8 | 0.816 / - / 64.8 | 0.038 |
| ce161092 | 105.9 | moving | 0.0 | 2.5 | 0.000 / offroad / 125.2 | 0.000 / offroad / 125.9 | 0.482 / - / 65.3 | 0.488 / - / 78.4 | 0.035 |
| 76f7e025 | 108.5 | moving | 0.0 | 8.4 | 1.000 / - / 177.3 | 1.000 / - / 177.3 | 1.000 / - / 133.9 | 1.000 / - / 135.2 | 0.035 |
| 2a47ea01 | 138.2 | moving | 0.0 | 5.5 | 0.257 / - / 70.1 | 0.310 / - / 75.0 | 0.979 / - / 111.3 | 0.990 / - / 112.3 | 0.027 |
| 788c4ad9 | 146.9 | moving | 0.0 | 4.9 | 1.000 / - / 182.9 | 1.000 / - / 182.9 | 1.000 / - / 137.4 | 1.000 / - / 135.2 | 0.026 |
| c8641276 | 147.6 | moving | 0.0 | 4.7 | 0.000 / collision_at_fault / 68.4 | 0.000 / collision_at_fault / 68.4 | 0.934 / - / 109.5 | 0.948 / - / 111.2 | 0.025 |
| ca173de1 | 163.7 | moving | 0.0 | 4.2 | 1.000 / - / 185.7 | 1.000 / - / 185.7 | 1.000 / - / 160.1 | 1.000 / - / 162.5 | 0.023 |
| f1069db5 | 165.6 | moving | 0.0 | 4.8 | 1.000 / - / 138.8 | 1.000 / - / 138.8 | 1.000 / - / 142.7 | 1.000 / - / 153.8 | 0.023 |
| ec1d0b2b | 169.0 | moving | 0.0 | 2.8 | 0.136 / - / 81.7 | 0.136 / - / 82.1 | 0.721 / - / 104.3 | 0.839 / - / 115.3 | 0.022 |
| 903731dd | 173.9 | moving | 0.0 | 2.7 | 0.166 / - / 162.8 | 0.166 / - / 162.8 | 1.000 / - / 173.0 | 1.000 / - / 181.8 | 0.022 |
| 132ae6cb | 220.6 | moving | 0.0 | 2.2 | 0.029 / - / 67.0 | 0.029 / - / 67.0 | 0.041 / - / 68.4 | 0.095 / - / 79.8 | 0.017 |
| fd01e9ab | 235.7 | moving | 0.0 | 4.8 | 0.000 / left_corridor_laterally / 80.5 | 0.000 / left_corridor_laterally / 80.5 | 0.000 / left_corridor_laterally / 58.7 | 0.000 / left_corridor_laterally / 57.3 | 0.016 |

## Net offline estimate of leash3 per read (mean over the read's rollouts; latched scenes only change)

| read | latched | still-to-the-end (GT < 5 m) gain | launch scenes cost | net per scene (281) |
|---|--:|--:|--:|--:|
| g10 | 34 | +1.000 | -12.464 | -0.0409 |
| g10hl | 34 | +1.000 | -12.483 | -0.0409 |
| acc | 34 | +1.000 | -17.868 | -0.0602 |
| accgap | 34 | +1.000 | -22.323 | -0.0759 |

## Still-end corridor zeros by how the human got there (all reads)

| read | still-end corridor zeros | latched at handoff | human stopped after the handoff (moving at handoff) | trip past the GT end | median human end-still (s) |
|---|--:|--:|--:|--:|--:|
| g10 | 17 | 1 | 16 | 14 | 6.0 |
| g10hl | 18 | 1 | 17 | 15 | 5.8 |
| acc | 19 | 1 | 18 | 16 | 5.6 |
| accgap | 21 | 1 | 20 | 16 | 6.0 |

Still-end scenes with a corridor zero in some read (C = corridor flag):

| scene | GT (m) | v at handoff | human end-still (s) | g10 score / fail / prog at trip | g10hl score / fail / prog at trip | acc score / fail / prog at trip | accgap score / fail / prog at trip |
|---|--:|--:|--:|---|---|---|---|
| 054b5901 | 132.3 | 14.76 | 6.0 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | missing | 0.00 / left_corridor_laterally / 1.000 C |
| 127c4f85 | 111.5 | 10.86 | 6.7 | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C |
| 2466266e | 109.7 | 10.83 | 3.7 | 0.00 / offroad / 1.000 C | 0.00 / offroad / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 0.718 C |
| 2bbf1efc | 13.4 | 3.64 | 15.2 | 1.00 / - / 0.998 | 1.00 / - / 0.998 | 1.00 / - / 0.998 | 0.00 / left_corridor_laterally / 0.998 C |
| 3fe2bac8 | 0.2 | 0.00 | 19.4 | 0.00 / left_corridor_laterally / 0.715 C | 0.00 / left_corridor_laterally / 0.715 C | 0.00 / left_corridor_laterally / 0.715 C | 0.00 / left_corridor_laterally / 0.715 C |
| 444fca69 | 104.8 | 8.83 | 3.8 | 1.00 / - / 0.947 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 44df49fe | 82.6 | 10.79 | 5.6 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 47db2eb6 | 117.6 | 8.79 | 6.0 | 1.00 / - / 0.799 | 1.00 / - / 0.974 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 4cb7f7ac | 109.7 | 10.73 | 3.6 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 4ecc114d | 226.2 | 13.60 | 0.6 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 559d4ad6 | 76.6 | 12.09 | 9.0 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 62cfe9d3 | 131.8 | 10.78 | 2.5 | 0.00 / offroad / 0.654 C | 0.00 / offroad / 0.658 C | 0.00 / left_corridor_laterally / 0.840 C | 0.00 / left_corridor_laterally / 0.835 C |
| 68c3be44 | 103.1 | 13.18 | 8.0 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 7381c9fa | 60.6 | 3.38 | 6.2 | 0.00 / offroad / 0.330 C | 0.00 / offroad / 0.330 C | 0.00 / offroad / 0.333 C | 0.00 / offroad / 0.333 C |
| 7750265d | 229.2 | 19.62 | 4.1 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 7d994ed3 | 98.4 | 10.95 | 6.9 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| 9b6c68b4 | 93.0 | 11.53 | 2.4 | 1.00 / - / 0.888 | 1.00 / - / 0.897 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 0.939 C |
| 9f8acd00 | 163.9 | 13.92 | 3.6 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
| ba40be62 | 68.1 | 10.65 | 11.3 | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C |
| bc309fd6 | 46.4 | 10.35 | 13.0 | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C | 0.00 / left_corridor_laterally / 0.999 C |
| fc9fde87 | 138.1 | 9.35 | 1.9 | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C | 0.00 / left_corridor_laterally / 1.000 C |
