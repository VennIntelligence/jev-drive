# DAG-LANE rows: teacher b3L on the iteration-1 DAgger states (dl.py label)

on1_PAID: 110981 rows, 90300 drawable (R6DAG's set, unchanged); teacher files for 100.0% of the drawable rows; 134 of 718 scenes have a qualifying stretch (human in the route lane's parallel neighbour).

Off-human = the label's 4 s (2 s) point lies more than 1.75 m from the human's recorded path. Moved = |b3L - b3| at 4 s. Target = off-human under b3 and moved > 1 m.

| speed (m/s) | drawable | b3 off-human 4 s | 2 s | b3L off-human 4 s | moved > 0.3 m | > 1 m | target | lane-rule weight > 0.5 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 2-10 | 46266 | 4505 | 6414 | 3180 | 2781 | 2059 | 1265 | 2304 |
| 10-15 | 21314 | 2405 | 3132 | 1185 | 1717 | 1460 | 1205 | 1361 |
| 15-20 | 8886 | 1865 | 2052 | 379 | 1819 | 1670 | 1478 | 1703 |
| 20-25 | 5171 | 1206 | 1457 | 126 | 1303 | 1159 | 1078 | 1303 |
| >=25 | 8663 | 1544 | 2255 | 62 | 1742 | 1572 | 1483 | 1802 |

Students already more than 1.75 m off the human's path at t0: 7678 drawable rows.

G-L0: b3 reproduced max |dxy| 0.00e+00 m over 100.0% of the drawable rows; b3L = b3 bit for bit on the scenes without a qualifying pose: True -> **PASS**.
G-L1: target rows at >= 15 m/s 4039 (line >= 300); b3L brings 99.9% of them within 1.75 m of the human (line >= 90 %) -> **PASS**.
G-L4: position-estimator p99 3.68 m/s^2 (b3 3.63), over the line 0 (0.00%); changed rows over the line reverted to b3: 1734 -> **PASS**.
Labels changed against R6DAG's: 14475 drawable rows (6157 at >= 15 m/s).

**PASS.**
