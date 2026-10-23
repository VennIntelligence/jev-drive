# NTV0-NOS verdict (PAI 281, ACC + GAP, Harmonizer OFF)

Old scorer (the read's own, as registered), 278 scenes common to NTV0NS / NTV0-s0 / RB0-F-s0 without RPC error.

- NTV0NS-L18-s0 mean **0.624**; registered line: >= 0.72 / <= 0.67 -> **the S rows carry the gain**
- NTV0NS - NTV0-L18-s0: -0.119 [-0.160, -0.081]
- NTV0NS - RB0-F-s0: -0.022 [-0.051, +0.006]
- NTV0-L18-s0 - RB0-F-s0: +0.097 [+0.060, +0.136]
- NTV0NS - NTV0-L18-s1 (278 scenes): -0.113 [-0.153, -0.075]; s1 - s0: -0.006 [-0.026, +0.014]

| read | mean | zeros | z_offroad | z_zero-other | z_side | z_route-turn | z_overshoot | z_AF-front | p_rear-hit | p_slow | launch | navtrain dev ADE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0NS-L18-s0 | 0.624 | 93 | 21 | 13 | 2 | 20 | 23 | 14 | 24 | 6 | 32/37 | 0.5287 |
| NTV0-L18-s0 | 0.743 | 59 | 12 | 6 | 0 | 15 | 23 | 3 | 25 | 11 | 33/37 | 0.5362 |
| NTV0-L18-s1 | 0.737 | 62 | 14 | 5 | 1 | 17 | 22 | 3 | 24 | 11 | 33/37 | 0.5354 |
| RB0-F-s0 | 0.646 | 88 | 19 | 14 | 4 | 21 | 22 | 8 | 22 | 8 | 32/37 | 0.5276 |

NTV0-L18-s1's row is on its own common set with NTV0NS / NTV0-s0 (pai281_s1.md).

## New scorer (scorer_v2, decision 453)

NTV0NS rescored: mean 0.697 over 279 scenes.

| ref | scenes | NTV0NS | ref | NTV0NS - ref | zeros NTV0NS / ref |
|:--|--:|--:|--:|--:|--:|
| NTV0-L18-s0-ag | 278 | 0.696 | 0.814 | -0.119 [-0.159, -0.081] | 73 / 39 |
| RB0-F-s0-ag | 279 | 0.697 | 0.715 | -0.019 [-0.048, +0.010] | 73 / 69 |
| NTV0-L18-s1-ag (partial: zero ASLs mostly deleted) | 279 | 0.697 | 0.748 | -0.051 [-0.101, -0.003] | 73 / 59 |

Amendment 1: lines as shares of the RB0-F -> NTV0-s0 gap (74% / 21% from the old-scorer values 0.65 / 0.744), on 278 common scenes: RB0-F 0.714, NTV0-s0 0.814 -> lines 0.789 / 0.736; NTV0NS 0.696 = -19% of the gap -> **the S rows carry the gain**.
Old-scorer gap share for comparison: -23%. Both scorers land on the same line.
