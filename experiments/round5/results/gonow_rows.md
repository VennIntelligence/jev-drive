# R5-G: GO-NOW rows (offline, before training)

GO-NOW row: fed < 0.5 m/s and the row's own label path reaches 0.5 m by its 1.0 s pose (the human starts within 1 s; launch_rows.md's 'leaves 0.5 m' on the first second). R5G = R2JF with every PAI human row fed < 2.0 m/s that is not GO-NOW at weight 0, each part renormalised to its share (MV's rule, GO-NOW rows kept) + R3MVL's navtrain weights (launch rows x 3).

PAI rows: drawable = R2JF weight > 0 (outside held and the exam links); stretch rows on their own label and fed speed, LEAD flag of their base plain row (lead_rows.npz, addendum 3). `per batch` = expected rows a batch under R5G (64 pool rows; navtrain 51 rows under R3MVL's nav_w). 4 s dist = the label's 4 s path (median [p25, p75]).

| source | GO-NOW rows | of fed < 0.5 | per batch | 4 s dist (m) | 1 s path med (m) | lead | lead share | per batch behind a lead | 4 s lead / no-lead (m) | scenes |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|:--|--:|
| PAI ph | 871 | 5.4% | 0.15 | 10.75 [7.74, 13.95] | 0.67 | 394 | 45% | 0.07 | 9.67 / 11.65 | 231 |
| PAI ph05 | 225 | 1.3% | 0.02 | 5.23 [1.94, 11.24] | 3.25 | 55 | 24% | 0.01 | 1.37 / 7.47 | 48 |
| PAI ph15 | 1163 | 11.6% | 0.13 | 18.23 [12.62, 24.70] | 0.89 | 500 | 43% | 0.06 | 16.04 / 19.09 | 162 |
| navtrain GO-NOW | 3557 | 42.6% of launch rows | 4.54 | 10.71 [8.27, 13.07] | 0.80 | n/a | n/a | n/a | | |
| navtrain launch rows (x 3) | 8349 | | 10.65 (R3MV 4.12) | 8.28 [6.10, 10.89] | | n/a | | | | |

navtrain: the cache holds no agent boxes, so no LEAD flag. PAI GO-NOW rows a batch in total: 0.31 (R3MV / R3MVL 0).

PAI rows fed < 2 m/s still at weight 0 (as MV): stand (fed < 0.5, 4 s path < 0.5 m) / go later (fed < 0.5, leaves 0.5 m after 1 s) / fed 0.5-2 m/s:

| part | R2JF drawable | MV drawable | R5G drawable | stand | go later | 0.5-2 m/s |
|:--|--:|--:|--:|--:|--:|--:|
| ph | 114453 | 90631 | 91502 | 9557 | 5695 | 7699 |
| ph05 | 104415 | 74250 | 74475 | 11758 | 4864 | 13318 |
| ph15 | 83671 | 69095 | 70258 | 4142 | 4689 | 4582 |

Identity: the same code with no GO-NOW row reproduces R3MV.npz bit for bit; non-PAI weights unchanged; nav_names / nav_w = R3MVL's arrays.
