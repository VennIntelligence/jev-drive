# Change 1 (DAgger rows): teacher b5 on the iteration-1 states (joint.py label)

Rule: arc_j = max(student served, min(human from the matched station, v0 t + 2.0 t^2 / 2)) iff the human's 8 arcs exist, the human keeps >= 1.0 m/s over t_h .. t_h + 6.0 s, |student clock - t_h| <= 3.0 s, that floor's 4 s arc exceeds the student's by > 0.05 m and no object leads on the floored label path (r5 LEAD rule, t0 boxes); floored labels over the T4 line (4 m/s^2) revert to b3. F2 on the final labels: v0 >= 4, 0.5 <= v_end < 0.6 v0, max |heading| < 20 deg, no lead -> weight 0.

Drawable rows (b3, R6DAG): 90300; floored 14677 (16.3%); reasons {"drawable": 90300, "teach_found": 90300, "no_window": 12850, "human_slows": 7817, "lag": 9540, "not_slower": 36991, "candidates": 30704, "lead": 14808, "t4_revert": 1219, "floored": 14677}.
4 s arc increase on floored rows p10 / p50 / p90 [0.79, 5.71, 22.73] m; straight rows (|4 s heading| < 5 deg) floored 8660 / 59827, increase [0.77, 5.3, 22.77] m.
By class (floored / drawable): {"launch": [493, 6020], "other": [8331, 44697], "stop_nolead": [1866, 19275], "turn_sharp": [3987, 20308]}; by speed (m/s): {"2-5": [1652, 14108], "5-10": [4682, 32158], "10-15": [4052, 21314], "15-20": [2178, 8886], "20-25": [626, 5171], ">=25": [1487, 8663]}.
F2 on the b5 labels: 7177 pd rows to weight 0 (on b3 labels the same rule, lead ignored, would take 11443).

G1a b3 reproduced: max |diff| 0.00e+00 m over 100.0% of the drawable rows -> PASS; G1b one-sided: min arc change vs b3 -0.0032 m, change minus (floor - student) max +0.0066 m, floor minus max(student, human) max +0.0000 m (b3 labels starting > 1 m ahead of the student's first arc: 1000 rows) -> PASS; G1c clocks at k = 0: median |lag| 0.000 s (664 rows; all drawable lag p10 / p50 / p90 [-2.34, -0.0, 1.1]) -> PASS; T4 p99 3.76 (b3 3.63) -> PASS.

**G1: PASS.**
