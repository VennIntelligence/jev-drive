# DIAG-READS: P1L Harmonizer ON, INTP75-NR re-read (PAI 281 ACC + GAP)

New scorer (scorer_v2 new) primary, old in brackets. Zero classes: old-scorer facts.

## A. P1L with the Harmonizer ON (279 common scenes, 0 RPC scenes excluded)

Renderer logs (harmonizer enabled and weights loaded): 8 / 8 chunks

| read | new mean (old) | zeros new (old) | overshoot | route-turn | offroad | AF-front | side | zero-other |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| P1L OFF | 0.813 (0.735) | 39 (61) | 24 | 10 | 14 | 4 | 3 | 6 |
| P1L ON | 0.804 (0.733) | 41 (61) | 22 | 6 | 18 | 7 | 1 | 7 |
| NTV0-s0 OFF | 0.815 (0.743) | 39 (59) | 23 | 15 | 12 | 3 | 0 | 6 |
| NTV0-s0 ON | 0.809 (0.745) | 40 (58) | 21 | 9 | 17 | 5 | 0 | 6 |
| R8K-s2 OFF | 0.873 (0.805) | 28 (47) | 20 | 2 | 14 | 5 | 2 | 4 |
| R8K-s2 ON | 0.866 (0.794) | 27 (47) | 21 | 2 | 15 | 4 | 1 | 4 |

| pair (95 % bootstrap CI, scenes as units) | new | old | zeros (old) |
|:--|--:|--:|--:|
| P1L ON - OFF | -0.009 [-0.040, +0.023] | -0.002 [-0.035, +0.032] | +0 |
| NTV0-s0 ON - OFF | -0.006 [-0.040, +0.028] | +0.002 [-0.031, +0.035] | -1 |
| R8K-s2 ON - OFF | -0.007 [-0.037, +0.023] | -0.010 [-0.041, +0.020] | +0 |
| P1L ON - NTV0-s0 ON | -0.005 [-0.043, +0.033] | -0.012 [-0.052, +0.027] | +3 |
| P1L ON - R8K-s2 ON | -0.062 [-0.103, -0.021] | -0.062 [-0.105, -0.019] | +14 |

## B. INTP75-NR re-read vs first read (278 common scenes, 0 RPC scenes excluded)

| scorer | identical score | mean abs diff | re-read - first |
|:--|--:|--:|--:|
| new | 278 / 278 | 0.000 | +0.000 [+0.000, +0.000] |
| old | 278 / 278 | 0.000 | +0.000 [+0.000, +0.000] |

Negative flips on NTV0-s0's full-score scenes (new scorer; full = score >= 0.98, 202 scenes):

| read | negative flips | of them to 0 | points lost there |
|:--|--:|--:|--:|
| R8K-s2 OFF | 14 | 9 | 10.53 |
| INTP75 first | 8 | 6 | 6.73 |
| INTP75 re-read | 8 | 6 | 6.73 |

In both INTP75 reads: 8; only the first: 0; only the re-read: 0.

Negative flips on NTV0-s0's full-score scenes (old scorer; full = score >= 0.98, 182 scenes):

| read | negative flips | of them to 0 | points lost there |
|:--|--:|--:|--:|
| R8K-s2 OFF | 15 | 10 | 11.53 |
| INTP75 first | 8 | 6 | 6.73 |
| INTP75 re-read | 8 | 6 | 6.73 |

In both INTP75 reads: 8; only the first: 0; only the re-read: 0.

Cost: 13.10 GPU job-hours (13.10 in the read chunks, 16 GPU jobs).
