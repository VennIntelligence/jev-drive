# INTERP: NTV0-L18-s0 <-> R8K-L18-s2 weight interpolation, PAI 281 ACC + GAP, Harmonizer OFF

theta = (1 - a) NTV0-L18-s0 + a R8K-L18-s2 (INTP50-NR: a = 0.5, INTP75-NR: a = 0.75). 277 scenes scored by all four reads with no driver RPC error in any (0 RPC scenes excluded). New scorer (scorer_v2 new) primary, old in brackets.

## Reads

| read | new mean | old mean | zeros new / old | overshoot | route-turn | offroad | AF-front | side | zero-other |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NTV0-s0 | 0.814 | 0.745 | 39 / 58 | 22 | 15 | 12 | 3 | 0 | 6 |
| R8K-s2 | 0.872 | 0.807 | 28 / 46 | 19 | 2 | 14 | 5 | 2 | 4 |
| INTP50-NR | 0.836 | 0.778 | 36 / 52 | 17 | 5 | 18 | 5 | 1 | 6 |
| INTP75-NR | 0.879 | 0.814 | 26 / 44 | 19 | 3 | 10 | 5 | 2 | 5 |

Zero classes: old-scorer facts (fclass).

## Paired differences (95 % bootstrap CI, scenes as units)

| pair | new scorer | old scorer | zeros (old) |
|:--|--:|--:|--:|
| INTP50-NR - NTV0-s0 | +0.022 [-0.010, +0.053] | +0.033 [+0.004, +0.062] | -6 |
| INTP50-NR - R8K-s2 | -0.036 [-0.074, +0.002] | -0.029 [-0.066, +0.009] | +6 |
| INTP75-NR - NTV0-s0 | +0.065 [+0.028, +0.102] | +0.069 [+0.031, +0.107] | -14 |
| INTP75-NR - R8K-s2 | +0.007 [-0.017, +0.032] | +0.007 [-0.019, +0.034] | -2 |
| R8K-s2 - NTV0-s0 | +0.058 [+0.016, +0.099] | +0.062 [+0.018, +0.104] | -12 |

## Flips against NTV0-s0 (new scorer; full = score >= 0.98)

NTV0-s0 full-score scenes: 201; R8K-s2 fixes (NTV0 < 0.98, R8K >= 0.98): 40.

| read | negative flips | of them to 0 | points lost there | shared with R8K-s2's | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2 fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| R8K-s2 | 14 | 9 | 10.53 | 14 | 40 | 40 |
| INTP50-NR | 9 | 8 | 8.51 | 2 | 20 | 17 |
| INTP75-NR | 8 | 6 | 6.73 | 6 | 36 | 33 |

## Flips against NTV0-s0 (old scorer; full = score >= 0.98)

NTV0-s0 full-score scenes: 182; R8K-s2 fixes (NTV0 < 0.98, R8K >= 0.98): 42.

| read | negative flips | of them to 0 | points lost there | shared with R8K-s2's | fixed (NTV0 < 0.98 -> >= 0.98) | R8K-s2 fixes kept |
|:--|--:|--:|--:|--:|--:|--:|
| R8K-s2 | 15 | 10 | 11.53 | 15 | 42 | 42 |
| INTP50-NR | 6 | 5 | 5.51 | 2 | 20 | 17 |
| INTP75-NR | 8 | 6 | 6.73 | 6 | 37 | 34 |

Cost: 6.09 GPU job-hours (6.08 in the read chunks, 17 GPU jobs).
Load / forward check: passed (18 checks, cuda).
