# OTHERS-TURN: where the 24 held turns fail, per driver

where = route station of the last decision before the zero (before = < entry - 20 m, approach = entry - 20 .. entry, turn = entry .. exit + 10, after); heading covered = ego yaw change from the decision 20 m before the entry to the zero (or exit + 10 m) over the route's turn angle (1.0 = turned as much as the route); entry speed = the last decision at or before the entry (m/s).

| label | completed | zero before / approach / turn / after | no zero | heading covered, completed (median) | heading covered, failed (median) | entry speed median |
|:--|--:|:--|--:|--:|--:|--:|
| RSG-L18-s0@18 | 16 / 24 | 0 / 0 / 1 / 1 | 22 | 1.02 | 1.04 | 4.19 |
| YR10m10-F-s0@18 | 9 / 24 | 0 / 0 / 6 / 2 | 16 | 1.00 | 0.94 | 4.44 |
| WAJEPA@18 | 7 / 24 | 0 / 1 / 12 / 2 | 9 | 1.12 | 0.45 | 4.61 |
| LTF@18 | 0 / 24 | 0 / 0 / 23 / 0 | 1 | nan | 0.56 | 4.83 |
| DD@18 | 0 / 24 | 0 / 0 / 24 / 0 | 0 | nan | 0.49 | 4.47 |
| G0014@18 | 1 / 24 | 0 / 5 / 13 / 2 | 4 | 1.09 | 0.62 | 4.19 |
