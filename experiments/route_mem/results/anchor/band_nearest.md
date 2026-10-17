## A1 First-slot lead (m): the served projected layout, the ordered layout (d302) and the anchored layout

lead = the true route arc of the first valid slot's point minus its label, against the car's tracked arc on AlpaSim's route (the audit's definition); median / p90 / p99; |lead| > 1 m counts (anchored: ahead + / behind -).

| set | decisions | projected | ordered | anchored | projected > 1 m | ordered > 1 m | anchored > 1 m (+ / -) | anchored min / max |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 8129 | 0.00 / 7.83 / 26.24 | -0.00 / 0.00 / 0.01 | -0.00 / 0.00 / 0.00 | 1985 | 58 | 21 (16 / 5) | -1.0 / 4.2 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 8129 | 0.00 / 7.76 / 26.24 | -0.00 / 0.00 / 0.01 | -0.00 / 0.00 / 0.00 | 1970 | 64 | 27 (19 / 8) | -1.0 / 4.8 |
| RSG-L18-s0, public 1491 @10 | 14790 | 0.03 / 10.26 / 28.30 | -0.00 / 0.00 / 0.00 | -0.00 / 0.00 / 0.00 | 4525 | 48 | 21 (16 / 5) | -1.6 / 4.2 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 14790 | 0.03 / 10.18 / 28.30 | -0.00 / 0.00 / 0.00 | -0.00 / 0.00 / 0.00 | 4502 | 47 | 23 (16 / 7) | -1.7 / 4.2 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 13134 | 0.00 / 7.06 / 26.92 | -0.00 / 0.00 / 0.01 | -0.00 / 0.00 / 0.00 | 2999 | 123 | 69 (16 / 53) | -1.7 / 4.2 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 14790 | 0.03 / 10.21 / 28.30 | -0.00 / 0.00 / 0.00 | -0.00 / 0.00 / 0.00 | 4506 | 44 | 21 (16 / 5) | -1.6 / 4.2 |
| NAVSIM navtrain rows (rm_build anchor's rows) | 101313 | 0.01 / 9.89 / 29.52 | -0.00 / 0.00 / 46.68 | -0.00 / 0.00 / 0.01 | 32555 | 1626 | 315 (115 / 200) | -91.1 / 57.5 |

## A2 Turns: decisions with a route turn within 42 m, and U-turn rollouts (first-slot lead median / p90 / p99, m)

| set | turn within 42 m: projected | anchored | U-turn rollouts: projected | anchored | near slot >= 10 m ahead: projected / ordered / anchored |
|:--|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 3.62 / 16.49 / 34.59 | -0.00 / 0.00 / 0.02 | 5.73 / 31.66 / 71.98 | -0.04 / 0.00 / 4.21 | 337 / 30 / 0 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 3.69 / 16.79 / 34.67 | -0.00 / 0.00 / 0.03 | 5.41 / 32.48 / 73.12 | -0.04 / 0.00 / 4.21 | 332 / 28 / 0 |
| RSG-L18-s0, public 1491 @10 | 6.23 / 18.66 / 35.49 | -0.00 / 0.00 / 0.01 | 17.90 / 46.28 / 75.75 | -0.01 / 0.00 / 4.21 | 609 / 13 / 0 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 6.11 / 18.72 / 36.03 | -0.00 / 0.00 / 0.01 | 17.71 / 45.37 / 75.72 | -0.01 / 0.00 / 4.21 | 605 / 12 / 0 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 3.26 / 16.10 / 51.55 | -0.00 / 0.00 / 0.01 | 10.17 / 46.70 / 81.94 | -0.03 / 0.00 / 1.18 | 448 / 21 / 0 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 6.16 / 18.79 / 36.20 | -0.00 / 0.00 / 0.01 | 17.71 / 46.12 / 75.73 | -0.01 / 0.00 / 4.21 | 607 / 11 / 0 |
| NAVSIM navtrain rows (rm_build anchor's rows) | 4.23 / 16.91 / 52.14 | -0.01 / 0.00 / 0.01 | - | - | 4417 / 1200 / 8 |

## A3 The two defects of the ordered layout

fold = rmem.merge's polyline doubles back by > 1 m (merge_a: the anchored merge); offset = AlpaSim's first waypoint one or two spacings further out than L0 (msg lead > 1 m); tracked = the anchored rule took m > 0 extra spacings there; false trims = m > 0 without an offset.

| set | fold decisions (old merge) | merge_a folds | ordered |lead| > 1 at a fold | anchored | offset decisions | ordered |lead| > 1 there | anchored | tracked | false trims |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 32 | 0 | 30 | 0 | 27 | 19 | 12 | 10 | 0 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 31 | 0 | 28 | 0 | 29 | 22 | 13 | 12 | 0 |
| RSG-L18-s0, public 1491 @10 | 16 | 0 | 13 | 0 | 26 | 26 | 12 | 14 | 0 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 16 | 0 | 12 | 0 | 24 | 24 | 12 | 12 | 0 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 29 | 0 | 21 | 0 | 49 | 49 | 16 | 37 | 0 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 16 | 0 | 11 | 0 | 24 | 24 | 12 | 12 | 0 |
| NAVSIM navtrain rows (rm_build anchor's rows) | 1347 | 65 | - | 17 | - | - | - | - | - |

NAVSIM rows built by rm_build anchor (feat/<data>.a.npz, float16) against this rebuild: 8468 rows, 0.9998 within 0.05 m, masks equal on 0.9999.

NAVSIM rows whose route revisits a place (passes within 2.5 m of itself > 15 m of arc apart: the truth's stations are ambiguous there): 92. Of the anchored rows off by > 5 m, 18 of 38 are such rows; of merge_a's folds, 65 of 65 (old merge: 79 of 1347). Without them: anchored |lead| > 1 m 294 (111 / 183), min / max -44.0 / 57.5; ordered 1548 (1391 / 157), min / max -44.0 / 68.8.

## A4 Ablations of the per-message anchor (|lead| > 1 m: ahead + / behind -; min / max)

plain = merge_a + C_k (no trim tracking); cap = min(C_k, s_prev + chord + 1 m per message), the cap first proposed; anchored = the rule (rmem.memory_a).

| set | plain | cap | anchored |
|:--|--:|--:|--:|
| RSG-L18-s0, held 460 @18 | 28 (23 / 5); -1.0 / 8.4 | 58 (22 / 36); -5.0 / 5.3 | 21 (16 / 5); -1.0 / 4.2 |
| RTB1ARO-L18-s0 (V2), held 460 @18 | 36 (28 / 8); -1.0 / 8.5 | 72 (27 / 45); -5.4 / 5.4 | 27 (19 / 8); -1.0 / 4.8 |
| RSG-L18-s0, public 1491 @10 | 35 (30 / 5); -1.6 / 4.2 | 51 (25 / 26); -5.0 / 4.2 | 21 (16 / 5); -1.6 / 4.2 |
| RTB1ARO-L18-s0 (V2), public 1491 @10 | 35 (28 / 7); -1.7 / 4.2 | 51 (23 / 28); -6.4 / 4.2 | 23 (16 / 7); -1.7 / 4.2 |
| RB0-F-s0 (pool on1_L18RB0), train 1020 @18 | 102 (53 / 49); -1.7 / 4.2 | 127 (41 / 86); -5.1 / 4.2 | 69 (16 / 53); -1.7 / 4.2 |
| RTB1O-L18-s0 (arm a, the video), public 1491 @10 | 33 (28 / 5); -1.6 / 4.2 | 50 (23 / 27); -5.0 / 4.2 | 21 (16 / 5); -1.6 / 4.2 |
