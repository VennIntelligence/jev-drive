# FIX rows (round 6 line C): F1 + F2 on MV-LC's draw (offline, before training)

- F1: ph15 rows: label peak lateral acceleration > 3.0 m/s^2 or fed v0 > 30.0 m/s (offroad_mv.label_feats).
- F2: ph / ph05 / ph15 rows: v0 >= 4.0, 0.5 <= v_end < 0.6 v0, max |heading| < 20.0 deg, no r5 LEAD flag.
- Base: R5LC (MV-LC's draw); each PAI part renormalised to its R5LC share (ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows: 32 PAI human rows a batch).

## Removed rows (drawn under R5LC)

| part | drawable | F1 | F1a (a_lat > 3) | F1b (v0 > 30) | F2 | F1 and F2 | removed | of part (draw-weighted) | rows a batch removed | F1 / F2 a batch | scenes removed / drawable | under R3MV: F1 / F2 (offroad_mv: 23 693 / 9 735) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 86412 | 0 | 0 | 0 | 4611 | 0 | 4611 | 5.3% | 0.85 | 0.00 / 0.85 | 181 / 703 | 0 / 4637 |
| ph05 | 73885 | 0 | 0 | 0 | 472 | 0 | 472 | 0.6% | 0.05 | 0.00 / 0.05 | 51 / 680 | 0 / 472 |
| ph15 | 62693 | 19606 | 14636 | 6022 | 4529 | 78 | 24057 | 38.4% | 3.07 | 2.50 / 0.58 | 419 / 692 | 23693 / 4626 |
| all PAI | 222990 | 19606 | 14636 | 6022 | 9612 | 78 | 29140 | | 3.97 | 2.50 / 1.48 | | 23693 / 9735 |

Removed / drawable by label class (r3.label_class):

| part | launch | stand | stop-approach | turn | cruise |
|:--|--:|--:|--:|--:|--:|
| ph | 0 / 0 | 0 / 0 | 63 / 4564 | 142 / 18002 | 4406 / 63846 |
| ph05 | 0 / 0 | 0 / 0 | 0 / 376 | 0 / 9367 | 472 / 64142 |
| ph15 | 0 / 0 | 0 / 0 | 376 / 6735 | 14176 / 18231 | 9505 / 37727 |

## Route-turn and junction rows removed (risk to the turn fix)

`junction (label)` = the 4 s label turns >= 45 deg; `junction ahead` = the human's stitched path turns >= 45 deg within 4 f s or 20 m (r3.pai_events, own clock); `route turn ahead` = the same >= 15 deg (MVE's turn event); `curve` = label 15-45 deg. Removed / drawable (of which by F1 / by F2):

| part | junction (label) | junction ahead | route turn ahead | curve |
|:--|--:|--:|--:|--:|
| ph | 0 / 6684 (0 / 0) | 86 / 7269 (0 / 86) | 347 / 18963 (0 / 347) | 154 / 11425 (0 / 154) |
| ph05 | 0 / 723 (0 / 0) | 0 / 2638 (0 / 0) | 13 / 11123 (0 / 13) | 0 / 8731 (0 / 0) |
| ph15 | 8289 / 8633 (8289 / 0) | 8378 / 8895 (8348 / 30) | 14394 / 18905 (14105 / 312) | 5959 / 9849 (5812 / 171) |

PAI human rows a batch by state (64 pool rows x draw mass; F2 = slow-down, no stop, no lead; F1 = the F1 rows of ph15):

| draw | all | junction_label | junction_ahead | route_turn_ahead | curve | fast30 | slow_nostop | slow_nostop_nolead | f1 | unique PAI rows |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R3MV | 32.00 | 2.26 | 2.60 | 6.77 | 4.13 | 1.84 | 2.34 | 1.41 | 2.74 | 233976 |
| R5LC | 32.00 | 2.42 | 2.77 | 7.13 | 4.32 | 1.24 | 2.47 | 1.48 | 2.50 | 222990 |
| R6FIX | 32.00 | 1.46 | 1.80 | 5.79 | 3.96 | 0.50 | 1.24 | 0.00 | 0.00 | 193850 |

Per part, rows a batch (R5LC -> R6FIX):

| part | junction_label | junction_ahead | route_turn_ahead | curve | fast30 |
|:--|--:|--:|--:|--:|--:|
| ph | 1.24 -> 1.31 | 1.35 -> 1.40 | 3.51 -> 3.64 | 2.12 -> 2.20 | 0.47 -> 0.50 |
| ph05 | 0.08 -> 0.08 | 0.29 -> 0.29 | 1.20 -> 1.21 | 0.95 -> 0.95 | 0.00 -> 0.00 |
| ph15 | 1.10 -> 0.07 | 1.14 -> 0.11 | 2.41 -> 0.93 | 1.26 -> 0.81 | 0.77 -> 0.00 |

## Label peak lateral acceleration (drawn moving rows, draw-weighted; m/s^2)

| part / state / draw | n | v50 | a50 | a90 | a99 | max | > 2 | > 3 | v0 > 30 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph moving R5LC | 86412 | 9.2 | 0.25 | 2.13 | 4.04 | 11.2 | 11.3% | 4.0% | 2.9% |
| ph curve R5LC | 11425 | 8.5 | 1.74 | 3.03 | 4.28 | 6.8 | 38.1% | 10.6% | 0.0% |
| ph junction R5LC | 6684 | 6.1 | 2.51 | 4.07 | 5.93 | 11.2 | 73.3% | 32.5% | 0.0% |
| ph moving R6FIX | 81801 | 9.3 | 0.27 | 2.19 | 4.07 | 11.2 | 12.0% | 4.2% | 3.1% |
| ph curve R6FIX | 11271 | 8.5 | 1.75 | 3.03 | 4.28 | 6.8 | 38.6% | 10.7% | 0.0% |
| ph junction R6FIX | 6684 | 6.1 | 2.51 | 4.07 | 5.93 | 11.2 | 73.3% | 32.5% | 0.0% |
| ph05 moving R5LC | 73885 | 5.2 | 0.08 | 0.52 | 1.11 | 12.7 | 0.1% | 0.0% | 0.0% |
| ph05 curve R5LC | 8731 | 3.8 | 0.59 | 0.98 | 1.72 | 2.7 | 0.3% | 0.0% | 0.0% |
| ph05 junction R5LC | 723 | 3.0 | 0.75 | 1.33 | 5.56 | 6.4 | 4.0% | 2.5% | 0.0% |
| ph05 moving R6FIX | 73413 | 5.2 | 0.08 | 0.53 | 1.11 | 12.7 | 0.1% | 0.0% | 0.0% |
| ph05 curve R6FIX | 8731 | 3.8 | 0.59 | 0.98 | 1.72 | 2.7 | 0.3% | 0.0% | 0.0% |
| ph05 junction R6FIX | 723 | 3.0 | 0.75 | 1.33 | 5.56 | 6.4 | 4.0% | 2.5% | 0.0% |
| ph15 moving R5LC | 62693 | 12.6 | 0.54 | 5.19 | 9.58 | 19.5 | 29.8% | 23.3% | 9.6% |
| ph15 curve R5LC | 9849 | 13.3 | 3.30 | 6.26 | 8.52 | 12.9 | 84.3% | 58.7% | 8.6% |
| ph15 junction R5LC | 8633 | 9.6 | 5.19 | 8.63 | 13.04 | 19.5 | 99.8% | 96.0% | 1.2% |
| ph15 moving R6FIX | 38636 | 11.1 | 0.29 | 1.97 | 2.88 | 3.0 | 9.6% | 0.0% | 0.0% |
| ph15 curve R6FIX | 3890 | 10.6 | 2.20 | 2.85 | 2.98 | 3.0 | 62.4% | 0.0% | 0.0% |
| ph15 junction R6FIX | 344 | 6.7 | 2.69 | 2.94 | 2.99 | 3.0 | 93.9% | 0.0% | 0.0% |

navtrain reference (offroad_mv.md §4, uniform): all moving a50 0.35 / a90 1.58 / > 3 0.3 %; curve 1.07 / 1.95 / 0.3 %; junction 1.69 / 2.51 / 2.0 %; no row >= 25 m/s.

Identity: R6FIXI (the same code, nothing dropped) = R5LC.npz bit for bit; non-PAI weights unchanged (r5.part_drop asserts). Files: fix_rows.json, w_R6FIX.json, w_R6FIXI.json.
