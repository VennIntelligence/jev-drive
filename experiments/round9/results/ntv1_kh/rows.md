# Change 1 (DAgger rows): teacher b5 on the iteration-1 states (joint.py label)

Rule: arc_j = max(student served, min(human from the matched station, v0 t + 2.0 t^2 / 2)) iff the human's 8 arcs exist, the human keeps >= 1.0 m/s over t_h .. t_h + 6.0 s, |student clock - t_h| <= 3.0 s, that floor's 4 s arc exceeds the student's by > 0.05 m and no object leads on the floored label path (r5 LEAD rule, t0 boxes); floored labels over the T4 line (4 m/s^2) revert to b3. F2 on the final labels: v0 >= 4, 0.5 <= v_end < 0.6 v0, max |heading| < 20 deg, no lead -> weight 0.

Drawable rows (b3, R6DAG): 92561; floored 17515 (18.9%); reasons {"drawable": 92561, "teach_found": 92561, "no_window": 12541, "human_slows": 7742, "lag": 9421, "not_slower": 33183, "candidates": 35579, "lead": 16829, "t4_revert": 1235, "floored": 17515}.
4 s arc increase on floored rows p10 / p50 / p90 [0.78, 5.64, 18.42] m; straight rows (|4 s heading| < 5 deg) floored 11372 / 63599, increase [0.77, 5.64, 20.51] m.
By class (floored / drawable): {"launch": [561, 6547], "other": [10213, 46702], "stop_nolead": [1941, 18703], "turn_sharp": [4800, 20609]}; by speed (m/s): {"2-5": [1661, 14347], "5-10": [6024, 33538], "10-15": [4618, 21947], "15-20": [2360, 8965], "20-25": [1008, 5320], ">=25": [1844, 8444]}.
F2 on the b5 labels: 6060 pd rows to weight 0 (on b3 labels the same rule, lead ignored, would take 9862).

G1a b3 reproduced: max |diff| 0.00e+00 m over 100.0% of the drawable rows -> PASS; G1b one-sided: min arc change vs b3 -0.0009 m, change minus (floor - student) max +0.0042 m, floor minus max(student, human) max +0.0000 m (b3 labels starting > 1 m ahead of the student's first arc: 997 rows) -> PASS; G1c clocks at k = 0: median |lag| 0.000 s (665 rows; all drawable lag p10 / p50 / p90 [-1.82, 0.0, 1.38]) -> PASS; T4 p99 3.72 (b3 3.57) -> PASS.

**G1: PASS.**
