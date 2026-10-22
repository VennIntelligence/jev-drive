# NU-PAIBASE verdict (plans/2026-10-22-nu-paibase.md, registered lines)

Per arm against B2 of the same seed: (a) arena held native paired mean >= 0; (b) corridor zeros (held, every class containing corridor) < B2; (c) held 460 zeros <= B2 + 2; (d) frozen board-mix PCS >= B2. A share passes when both seeds pass all four.

| arm | ref | arena paired [CI] | corridor zeros arm / B2 | held 460 zeros arm / B2 | held 460 paired [CI] | PCS arm / B2 | a | b | c | d |
|:--|:--|--:|--:|--:|--:|--:|:-:|:-:|:-:|:-:|
| NPB70-L18-s0 | STANDFMXH-L18-s0 | +0.0046 [-0.0054, +0.0154] | 34 / 37 | 49 / 41 | -0.0161 [-0.0381, +0.0046] | 1349 / 1369 | pass | pass | FAIL | FAIL |
| NPB70-L18-s1 | STANDFMXH-L18-s1 | -0.0050 [-0.0178, +0.0066] | 52 / 45 | 48 / 41 | -0.0132 [-0.0353, +0.0087] | 1354 / 1398 | FAIL | FAIL | FAIL | FAIL |
| NPB30-L18-s0 | STANDFMXH-L18-s0 | -0.0021 [-0.0121, +0.0087] | 45 / 37 | 43 / 41 | -0.0039 [-0.0256, +0.0150] | 1350 / 1369 | FAIL | FAIL | pass | FAIL |
| NPB30-L18-s1 | STANDFMXH-L18-s1 | -0.0048 [-0.0161, +0.0054] | 55 / 45 | 44 / 41 | -0.0102 [-0.0326, +0.0101] | 1358 / 1398 | FAIL | FAIL | FAIL | FAIL |

Shares: P70 FAIL, P30 FAIL

## Against JTN-L18-s1 (reported, not judged)

| arm | arena paired [CI] | zeros arm / JTN | held 460 paired [CI] | held 460 zeros arm / JTN |
|:--|--:|--:|--:|--:|
| NPB70-L18-s0 | +0.0509 [+0.0247, +0.0790] | 80 / 210 | -0.0388 [-0.0639, -0.0126] | 49 / 36 |
| NPB70-L18-s1 | +0.0398 [+0.0157, +0.0663] | 110 / 210 | -0.0271 [-0.0534, -0.0009] | 48 / 36 |
| NPB30-L18-s0 | +0.0442 [+0.0207, +0.0690] | 96 / 210 | -0.0266 [-0.0510, -0.0024] | 43 / 36 |
| NPB30-L18-s1 | +0.0400 [+0.0195, +0.0623] | 111 / 210 | -0.0241 [-0.0514, +0.0027] | 44 / 36 |

## Zero layers (arena held, route-fail dropped; stratum from the arena set)

| tag | zeros | corridor, static window | corridor, moving window | offroad | AF-front | all zeros in static windows | PCS |
|:--|--:|--:|--:|--:|--:|--:|--:|
| NPB70-L18-s0 | 80 | 13 | 21 | 47 | 1 | 16 | 1349 |
| NPB70-L18-s1 | 110 | 27 | 25 | 56 | 2 | 30 | 1354 |
| NPB30-L18-s0 | 96 | 24 | 21 | 49 | 2 | 27 | 1350 |
| NPB30-L18-s1 | 111 | 34 | 21 | 51 | 4 | 37 | 1358 |
| STANDFMXH-L18-s0 | 92 | 14 | 23 | 53 | 3 | 17 | 1369 |
| STANDFMXH-L18-s1 | 101 | 21 | 24 | 55 | 2 | 24 | 1398 |
| JTN-L18-s1 | 210 | 130 | 21 | 58 | 7 | 134 | 1247 |

## Easy items (garage 0014 = 1; easy_items.py subject, ref JTN-L18-s1)

| tag | PCS board-mix | zeros | partials | easy zeros | easy partials | easy points lost |
|:--|--:|--:|--:|--:|--:|--:|
| NPB70-L18-s0 | 1349 | 75 | 116 | 17 | 32 | 25.5 |
| NPB70-L18-s1 | 1354 | 105 | 94 | 29 | 20 | 35.7 |
| NPB30-L18-s0 | 1350 | 92 | 105 | 29 | 23 | 36.5 |
| NPB30-L18-s1 | 1358 | 104 | 95 | 34 | 19 | 40.3 |
