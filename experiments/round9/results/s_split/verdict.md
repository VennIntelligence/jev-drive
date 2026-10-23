# S-SPLIT verdict (PAI 281, ACC + GAP, Harmonizer OFF)

278 scenes scored by all six reads (both scorers) with no driver RPC error in any of them.

## Registered lines (new scorer primary; arm >= P0 - 0.02: the removed part is not needed; arm <= NTV0NS + 0.03: it carries the gain)

- new scorer: P0 0.803, NTV0NS 0.696 -> lines 0.783 / 0.726
  - NH SSNH-L18-s0 (removed: the S-row hinges (agent / road / lateral)): 0.716, 19% of the NTV0NS -> P0 gap kept -> **carries the gain**
  - ND SSND-L18-s0 (removed: distill_ho on the S rows): 0.819, 116% of the NTV0NS -> P0 gap kept -> **not needed**
  - together: **the hinges carry the S gain; distill_ho is not needed (body schema = consequence hinge holds)**
- old scorer: P0 0.734, NTV0NS 0.624 -> lines 0.714 / 0.654
  - NH SSNH-L18-s0 (removed: the S-row hinges (agent / road / lateral)): 0.651, 25% of the NTV0NS -> P0 gap kept -> **carries the gain**
  - ND SSND-L18-s0 (removed: distill_ho on the S rows): 0.747, 112% of the NTV0NS -> P0 gap kept -> **not needed**
  - together: **the hinges carry the S gain; distill_ho is not needed (body schema = consequence hinge holds)**

## Reads

| read | new mean | old mean | zeros new / old | offroad | drift (zero-other) + side | route-turn | overshoot | AF-front | rear-hit p | slow p | dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SSNH-L18-s0 | 0.716 | 0.651 | 67 / 85 | 23 | 11 + 2 | 21 | 20 | 8 | 25 | 5 | 0.5326 |
| SSND-L18-s0 | 0.819 | 0.747 | 37 / 57 | 13 | 7 + 1 | 10 | 22 | 4 | 29 | 9 | 0.5352 |
| PCL0-L18-s0 | 0.803 | 0.734 | 42 / 61 | 14 | 7 + 0 | 16 | 22 | 2 | 27 | 9 | 0.5366 |
| NTV0NS-L18-s0 | 0.696 | 0.624 | 73 / 93 | 21 | 13 + 2 | 20 | 23 | 14 | 24 | 6 | 0.5287 |
| NTV0-L18-s0 | 0.814 | 0.743 | 39 / 59 | 12 | 6 + 0 | 15 | 23 | 3 | 25 | 11 | 0.5362 |
| RB0-F-s0 | 0.714 | 0.646 | 69 / 88 | 19 | 14 + 4 | 21 | 22 | 8 | 22 | 8 | 0.5276 |

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | new scorer | old scorer |
|:--|--:|--:|
| SSNH-L18-s0 - PCL0-L18-s0 | -0.087 [-0.129, -0.046] | -0.083 [-0.126, -0.042] |
| SSNH-L18-s0 - NTV0NS-L18-s0 | +0.020 [-0.007, +0.050] | +0.027 [-0.002, +0.058] |
| SSNH-L18-s0 - NTV0-L18-s0 | -0.099 [-0.139, -0.060] | -0.091 [-0.132, -0.053] |
| SSNH-L18-s0 - RB0-F-s0 | +0.002 [-0.030, +0.034] | +0.005 [-0.028, +0.038] |
| SSND-L18-s0 - PCL0-L18-s0 | +0.017 [-0.015, +0.049] | +0.013 [-0.019, +0.047] |
| SSND-L18-s0 - NTV0NS-L18-s0 | +0.123 [+0.086, +0.164] | +0.123 [+0.086, +0.164] |
| SSND-L18-s0 - NTV0-L18-s0 | +0.005 [-0.023, +0.033] | +0.005 [-0.023, +0.033] |
| SSND-L18-s0 - RB0-F-s0 | +0.105 [+0.068, +0.145] | +0.101 [+0.065, +0.141] |
| SSNH-L18-s0 - SSND-L18-s0 | -0.103 [-0.146, -0.064] | -0.096 [-0.140, -0.055] |

## Zero / partial classes vs P0 and NTV0NS (old-scorer classes; count difference, 95 % CI)

| arm - ref | zeros | overshoot | route-turn | offroad | AF-front | side | zero-other | rear-hit | slow | hit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SSNH-L18-s0 - PCL0-L18-s0 | +24 [+12, +36] | -2 [-6, +2] | +5 [-2, +13] | +9 [+1, +17] | +6 [+1, +12] | +2 [+0, +5] | +4 [-1, +9] | -2 [-7, +3] | -4 [-9, +0] | +0 [+0, +0] |
| SSNH-L18-s0 - NTV0NS-L18-s0 | -8 [-17, +1] | -3 [-7, +0] | +1 [-4, +6] | +2 [-3, +8] | -6 [-11, -2] | +0 [-3, +3] | -2 [-7, +3] | +1 [-3, +6] | -1 [-3, +0] | +0 [+0, +0] |
| SSND-L18-s0 - PCL0-L18-s0 | -4 [-14, +5] | +0 [-4, +4] | -6 [-12, -1] | -1 [-7, +5] | +2 [-2, +6] | +1 [+0, +3] | +0 [-3, +3] | +2 [-2, +6] | +0 [-3, +3] | +0 [+0, +0] |
| SSND-L18-s0 - NTV0NS-L18-s0 | -36 [-48, -25] | -1 [-3, +0] | -10 [-17, -4] | -8 [-15, -1] | -10 [-17, -4] | -1 [-5, +2] | -6 [-12, -1] | +5 [-1, +12] | +3 [-1, +8] | +0 [+0, +0] |

## Training logs (means over the last 1 000 steps)

| run | distill_ho | distill | agent_ho | road_ho | on/lat_onc_ho | on/lat_soft_ho | on/lat_side_ho | grad/pool | grad/imit |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| PCL0-L18-s0 | 0.0008793 | 0.001653 | 0.003883 | 0.005627 | 0.006629 | 0.004638 | 0.002665 | 2.237 | 8.523 |
| SSNH-L18-s0 | 0.0006636 | 0.001503 | 0.005376 | 0.009204 | 0.0191 | 0.006236 | 0.003126 | 0 | 8.113 |
| SSND-L18-s0 | 0.001567 | 0.00181 | 0.003656 | 0.005497 | 0.005987 | 0.004602 | 0.002744 | 2.065 | 8.39 |

## Cost

GPU jobs 7.2 job-hours (training 1.08, reads 6.05, gates / smokes 0.09); wall 1.2 h.
