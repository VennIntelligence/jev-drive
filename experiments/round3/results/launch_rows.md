# ROUND3b: launch rows per pool (offline, before training)

Launch row = fed ego speed v0 < 0.5 m/s and the row's own 4 s label path >= 0.5 m (the human leaves within the label). Rows = drawable under R2JF (J / J-NF; outside held and the exam links). `4 s dist` = the label's path length at 4 s (median [p25, p75]); `/ base` = median ratio of a stretch launch row's 4 s distance to its plain base row's (same human moment; expect about f^2 for a launch from rest); `per batch` = expected launch rows a batch (64 pool rows) under each draw file; `< 2 m/s` = share of the part's drawable rows that MV drops.

| pool | drawable | launch rows | share | 4 s dist (m) | / base | v0 med (m/s) | per batch R2JF | per batch R3NS | per batch R3MV | < 2 m/s |
|:--|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|
| PAI ph | 114453 | 6566 | 5.74% | 4.16 [1.86, 7.70] | 1.00 | 0.04 | 0.92 | 1.84 | 0.00 | 20.8% |
| PAI ph05 | 104415 | 5089 | 4.87% | 1.72 [0.97, 2.88] | 0.27 | 0.19 | 0.39 | 0.00 | 0.00 | 28.9% |
| PAI ph15 | 83671 | 5852 | 6.99% | 8.90 [4.08, 15.75] | 3.03 | 0.00 | 0.56 | 0.00 | 0.00 | 17.4% |
| navtrain_full | 103288 | 8349 | 8.08% | 8.28 [6.10, 10.89] | | 0.06 | 4.12 (of 51) | 4.12 | 4.12 | |

PAI launch rows a batch: R2JF 1.87 (row-weighted median 4 s distance 5.07 m), R3NS 1.84 (row-weighted median 4 s distance 4.16 m), R3MV 0.00 (row-weighted median 4 s distance 0.00 m); navtrain 4.12 at 8.28 m.
