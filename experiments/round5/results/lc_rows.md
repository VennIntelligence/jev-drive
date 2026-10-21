# R5-LC (MV-LC): lane-change-off-route PAI human rows (offline, before training)

lane-change-shaped label: fed v0 > 5.0 m/s, the 4 s label point |y| > 2.5 m and |yaw| < 12.0 deg (row frame); off the route lane: the row's route.npz polyline (lateral interpolated at the 4 s point's x, flat past its ends) does not pass within 1.2 m of the 4 s point (d1.py labels' rule; route not ok -> kept; route lateral nan -> off, as D1 counted). Every part on its own rows: own label, fed speed and route.npz (stretched rows on their own clock).

R3MV draws only PAI rows fed >= 2 m/s; MV-LC sets the off-route lane-change rows among them to 0 and renormalises each part to its R3MV share (ph 0.25, ph05 0.125, ph15 0.125 of the 64 pool rows: still 32 PAI human rows a batch).

| part | R3MV drawable | moving (> 5 m/s) | lane-change shaped | off route (removed) | of shaped | draw-weighted of part | route nan | route not ok (kept) | v0 median removed (m/s) | |y4| median removed (m) | scenes removed / drawable |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ph | 90631 | 75862 | 9150 | 4219 | 46.1% | 4.66% | 0 | 7 | 26.4 | 4.17 | 142 / 705 |
| ph05 | 74250 | 38891 | 1036 | 365 | 35.2% | 0.49% | 0 | 0 | 11.2 | 3.32 | 19 / 680 |
| ph15 | 69095 | 61643 | 10607 | 6402 | 60.4% | 9.27% | 0 | 0 | 35.7 | 6.02 | 200 / 707 |

Removed / drawable by label class (r3.label_class; launch and stand are 0 under MV's >= 2 m/s cut):

| part | launch | stand | stop-approach | turn | cruise |
|:--|--:|--:|--:|--:|--:|
| ph | 0 / 0 | 0 / 0 | 0 / 4564 | 90 / 18092 | 4129 / 67975 |
| ph05 | 0 / 0 | 0 / 0 | 0 / 376 | 2 / 9369 | 363 / 64505 |
| ph15 | 0 / 0 | 0 / 0 | 14 / 6749 | 136 / 18367 | 6252 / 43979 |

Check against D1 (all plain rows, any draw): lane-change shaped 10710 (11.9% of moving), with a route 10703, off the route lane 4708 (44%); D1: 10 710 (11.9 %), 10 703, 44 %.

PAI human rows a batch (64 pool rows x draw mass) by class, and the lane-change-shaped share of the drawn moving (> 5 m/s) rows (draw-weighted):

| draw | PAI rows | launch | stand | stop-approach | turn | cruise | lane-change shaped | share of moving | ph / ph05 / ph15 | unique PAI rows |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| R3MV | 32.00 | 0.00 | 0.00 | 1.63 | 6.33 | 24.04 | 2.96 | 12.0% | 12.1% / 2.7% / 17.2% | 233976 |
| R5LC | 32.00 | 0.00 | 0.00 | 1.75 | 6.67 | 23.58 | 1.52 | 6.2% | 6.9% / 1.7% / 7.6% | 222990 |

navtrain (navtrain_full, 12 shards, uniform in both draws): lane-change shaped 3549 of 44512 moving rows (8.0%). Files: lc_rows.json, w_R5LC.json, w_R5LCI.json (R5LCI = R3MV bit for bit).
