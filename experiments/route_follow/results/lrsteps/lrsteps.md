# lr x steps pilot: held 452 / 18 decisions (452 paired scenes), base RSG-L18-s0

| cell (lr-scale, steps) | tag | zeros 452 | d zeros vs RSG-L18-s0 [95 % CI] | at-fault | d at-fault | col / off / cor | mean score | d score | GT (m) | dev ADE |
|:--|:--|--:|:--|--:|:--|:--|--:|:--|--:|--:|
| (0.03, 1250) | RSGlr03n1250-L18-s0 | 33 | +7.0 [+0.0, +13.8] | 22 | +1.0 [-3.4, +5.4] | 3 / 19 / 11 | 0.8716 | -0.0093 [-0.0245, +0.0066] | 1.197 | 0.5295 |
| (0.03, 2500) | RSGlr03n2500-L18-s0 | 32 | +6.0 [+1.8, +11.3] | 25 | +4.0 [+0.9, +8.0] | 5 / 20 / 8 | 0.8692 | -0.0117 [-0.0230, -0.0018] | 1.125 | 0.5290 |
| (0.03, 5000) | RSGlr03n5000-L18-s0 | 23 | -3.0 [-7.6, +1.0] | 18 | -3.0 [-6.8, +0.0] | 3 / 15 / 5 | 0.8948 | +0.0139 [+0.0032, +0.0270] | 1.184 | 0.5270 |
| (0.1, 1250) | RSGlr10n1250-L18-s0 | 35 | +9.0 [+3.3, +15.1] | 23 | +2.0 [-2.9, +6.8] | 4 / 19 / 12 | 0.8664 | -0.0145 [-0.0283, -0.0011] | 1.154 | 0.5301 |
| (0.1, 2500) | RSG-L18-s0 | 26 | - | 21 | - | 4 / 17 / 6 | 0.8809 | - | 1.135 | 0.5284 |
| (0.1, 5000) | RSGlr10n5000-L18-s0 | 25 | -1.0 [-6.6, +4.0] | 19 | -2.0 [-7.8, +3.5] | 4 / 15 / 6 | 0.8905 | +0.0096 [-0.0018, +0.0236] | 1.144 | 0.5265 |
| (0.3, 1250) | RSGlr30n1250-L18-s0 | 27 | +1.0 [-4.0, +6.0] | 20 | -1.0 [-6.6, +3.9] | 6 / 14 / 7 | 0.8758 | -0.0050 [-0.0163, +0.0059] | 1.105 | 0.5300 |
| (0.3, 2500) | RSGlr30n2500-L18-s0 | 27 | +1.0 [-4.4, +6.0] | 20 | -1.0 [-5.6, +3.4] | 4 / 16 / 8 | 0.8781 | -0.0028 [-0.0130, +0.0087] | 1.127 | 0.5273 |
| (0.3, 5000) | RSGlr30n5000-L18-s0 | 27 | +1.0 [-3.4, +5.4] | 23 | +2.0 [-3.6, +7.3] | 9 / 14 / 5 | 0.8828 | +0.0019 [-0.0076, +0.0122] | 1.130 | 0.5252 |
| RSG-L18-s1 (seed reference) | RSG-L18-s1 | 26 | +0.0 [-4.7, +4.9] | 19 | -2.0 [-7.0, +2.6] | 6 / 13 / 8 | 0.8830 | +0.0021 [-0.0097, +0.0137] | 1.144 | 0.5270 |

Held-452 zeros, 3 x 3 (rows lr-scale, columns steps):

| lr-scale \ steps | 1250 | 2500 | 5000 |
|:--|--:|--:|--:|
| 0.03 | 33 | 32 | 23 |
| 0.1 | 35 | 26 | 25 |
| 0.3 | 27 | 27 | 27 |

Rule (prereg): fewest zeros = lr03n5000 (RSGlr03n5000-L18-s0, 23), d vs RSG-L18-s0 -3.0 [-7.6, +1.0] -> **FLAT (RSG-L18 stays the base)**. Cells with zeros CI entirely above RSG-L18-s0 (worse): lr03n2500, lr10n1250.

## Reading (prereg [plans/2026-10-15-lrsteps-pilot-prereg.md](../../plans/2026-10-15-lrsteps-pilot-prereg.md))

- **Verdict: FLAT.** The fewest-zeros cell (0.03, 5000) reads 23 zeros vs 26, -3.0 [-7.6, +1.0]. The CI includes 0, so RSG-L18 stays the base.
  No cell beats RSG-L18-s0 on zeros with a CI excluding 0.
- **Sensitivity:** the short or weak schedules are worse. (0.1, 1250) +9.0 [+3.3, +15.1] and (0.03, 2500) +6.0 [+1.8, +11.3] have CIs above 0;
  (0.03, 1250) +7.0 [+0.0, +13.8] sits on the edge. Every lr-scale 0.3 cell reads 27 (+1). (0.1, 5000) reads 25 (-1). On zeros the
  extra corridor zeros (11-12 vs 6) mark the under-trained cells. Longer training at lr x 0.03-0.1 is the only direction that does not lose.
- Secondary: mean score for (0.03, 5000) is +0.0139 [+0.0032, +0.0270] (the CI excludes 0); for (0.1, 5000) it is +0.0096 [-0.0018, +0.0236].
  Turn completion (`follow.md`, 24 held turns) for (0.03, 5000) is 75.0 % vs 66.7 %, +8.3 [+0.0, +20.8] pp. Dev ADE stays within 0.5252-0.5301 in every
  cell, so offline it does not separate the cells.
- Seed reference: RSG-L18-s1 vs RSG-L18-s0 is +0.0 [-4.7, +4.9] zeros, so the cell differences of 1-3 zeros are within seed noise.
- Reproduction check (first cell, RSGlr10n5000): the on-policy row hash at step 100 equals RSG-L18-s0's (f22e42deba570489), step-100 imit
  0.7551 vs 0.7552, dev ADE @ 500 0.5372 vs 0.5372, final 0.5265.
- Cost: 8 trainings 2.65 job-h (two per card at times) plus 48 read chunks 9.0 job-h (up to 4 per card). The box's 3 cards were busy
  1.7 h wall, about 5.1 card-h (estimate 6.8). Addendum (Amendment 1): seed 1 of (0.03, 5000) at priority 4; its row and the 2-seed
  means are appended when they land.
