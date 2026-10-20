# ROUND3c: rows a batch by class (offline, before training)

Label-only classes (each row's fed speed v0 and its own 4 s label, every pool alike): **launch** v0 < 0.5 m/s and the 4 s path >= 0.5 m; **stand** v0 < 0.5, path < 0.5 m; **stop-approach** v0 >= 0.5 and some 0.5 s label segment below 0.5 m/s; **turn** else and |heading| >= 15 deg at some label pose; **cruise** else. Batch = navtrain rows (83 CTRL / 51 the PAI arms) + pool rows (CTRL: 32 S rows, JT00S; PAI arms: 64 = 32 S + 32 PAI human rows, the arm's draw file); the 13 yaw-rate rows are not counted. MVL's navtrain draw: launch rows x 3, renormalised inside navtrain.

| arm | launch | stand | stop-approach | turn | cruise | of which PAI: launch / stand / stop / turn / cruise | PAI rows with MVE event (turn / stop / both / none) |
|:--|--:|--:|--:|--:|--:|:--|:--|
| R2C (CTRL) | 8.00 | 0.25 | 11.55 | 31.39 | 63.82 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 | - |
| MV | 5.41 | 0.22 | 9.68 | 28.12 | 71.56 | 0.00 / 0.00 / 1.63 / 6.33 / 24.04 | 6.47 / 3.45 / 0.30 / 21.78 |
| MVL | 11.94 | 0.22 | 8.90 | 25.99 | 67.95 | 0.00 / 0.00 / 1.63 / 6.33 / 24.04 | 6.47 / 3.45 / 0.30 / 21.78 |
| MVE | 5.41 | 0.22 | 12.51 | 41.87 | 54.98 | 0.00 / 0.00 / 4.46 / 20.08 / 7.45 | 20.90 / 10.18 / 0.91 / 0.00 |

Navtrain alone (rows a batch): R2C (CTRL) launch 6.71, stand 0.06, stop-approach 9.07, turn 24.90, cruise 42.27; MV launch 4.12, stand 0.04, stop-approach 5.57, turn 15.30, cruise 25.97; MVL launch 10.65, stand 0.03, stop-approach 4.80, turn 13.17, cruise 22.36

## PAI human rows fed >= 2 m/s (MV's rows): exam events ahead (MVE keeps turn / stop / both)

| part | MV rows | turn only | stop only | both | none (dropped by MVE) | kept share |
|:--|--:|--:|--:|--:|--:|--:|
| ph | 90631 | 18155 | 10234 | 898 | 61344 | 32.3% |
| ph05 | 74250 | 10818 | 2652 | 307 | 60473 | 18.6% |
| ph15 | 69095 | 18107 | 11760 | 941 | 38287 | 44.6% |

## Moving rows: 4 s label distance by fed speed (median m; in brackets median distance / (4 s x v0); n)

Drawable rows (navtrain: full labels; PAI: R2JF weight > 0, i.e. before MV's < 2 m/s cut; last row: MVE's plain rows).

| v0 bin (m/s) | navtrain | PAI ph | PAI ph05 | PAI ph15 | PAI ph, MVE rows |
|:--|:--|:--|:--|:--|:--|
| 0.5-2 | 9.2 (1.95; 16723) | 6.3 (1.24; 7699) | 5.1 (1.03; 13318) | 10.4 (2.32; 4582) | - |
| 2-4 | 11.4 (0.90; 20868) | 13.1 (1.08; 9238) | 12.2 (1.01; 24717) | 16.3 (1.31; 5005) | 7.6 (0.62; 5250) |
| 4-6 | 17.4 (0.91; 23496) | 21.8 (1.05; 12863) | 19.9 (1.00; 20688) | 23.8 (1.22; 4938) | 20.1 (0.98; 6749) |
| 6-8 | 24.4 (0.90; 16073) | 28.1 (1.02; 13801) | 27.6 (1.00; 11251) | 32.6 (1.14; 5860) | 25.9 (0.94; 5978) |
| 8-10 | 32.8 (0.92; 10488) | 35.3 (1.00; 11734) | 35.3 (1.00; 5132) | 36.7 (1.03; 7754) | 31.4 (0.88; 3798) |
| 10-13 | 41.8 (0.95; 6401) | 44.3 (1.00; 14685) | 46.7 (1.00; 5269) | 45.5 (1.01; 9669) | 43.2 (0.98; 4361) |
| 13-16 | 52.0 (0.95; 810) | 57.3 (1.00; 8793) | 57.1 (1.00; 5070) | 57.4 (0.99; 8017) | 55.5 (0.98; 1829) |
| 16- | 64.1 (0.94; 8) | 96.2 (1.00; 19517) | 70.8 (1.00; 2123) | 93.1 (1.00; 27852) | 78.5 (0.99; 1322) |
