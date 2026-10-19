# Best-of-N pick `BN`

1193 complete windows of 1200 (a window is dropped when any model's rollout failed outside the scorer's zero types: {'JTN-L18-s1': 7, 'NTV0-L18-s0': 7, 'RSG-L18-s0': 7, 'NR1FB10-L18-s0': 7}); informative = best - worst >= 0.2: **248**; ties among the best on informative windows 111 (broken toward JTN-L18-s1, then NTV0, RSG, NR1FB10).

Wins on informative windows (models: jtn1 = JTN-L18-s1, ntv0 = NTV0-L18-s0, rsg = RSG-L18-s0, fb10 = NR1FB10-L18-s0):

| stratum | windows | informative | jtn1 | ntv0 | rsg | fb10 | mean jtn1 | ntv0 | rsg | fb10 | best-of-4 mean | zeros jtn1 | ntv0 | rsg | fb10 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| static | 397 | 164 | 161 | 0 | 0 | 3 | 0.839 | 0.476 | 0.506 | 0.559 | 0.846 | 64 | 208 | 196 | 175 |
| launch | 299 | 57 | 16 | 25 | 9 | 7 | 0.862 | 0.907 | 0.911 | 0.885 | 0.955 | 15 | 21 | 20 | 25 |
| braking | 198 | 11 | 6 | 2 | 2 | 1 | 0.925 | 0.921 | 0.927 | 0.937 | 0.955 | 13 | 14 | 12 | 11 |
| turn | 120 | 12 | 7 | 1 | 2 | 2 | 0.892 | 0.890 | 0.876 | 0.883 | 0.934 | 12 | 12 | 13 | 13 |
| cruise | 60 | 0 | 0 | 0 | 0 | 0 | 0.974 | 0.972 | 0.971 | 0.970 | 0.980 | 0 | 0 | 0 | 0 |
| other | 119 | 4 | 2 | 0 | 2 | 0 | 0.963 | 0.968 | 0.968 | 0.950 | 0.976 | 1 | 1 | 1 | 2 |
| all | 1193 | 248 | 192 | 28 | 15 | 13 | 0.883 | 0.774 | 0.784 | 0.796 | 0.920 | 105 | 256 | 242 | 226 |
