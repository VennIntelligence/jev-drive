# Regression set: s0 = RSG-L18-s0, s1 = RSG-L18-s1; status against s0

34 scenes (the user's 28 + the 12 public no-launch scenes of explore_f.md, 6 overlap), one row per scene and stored read (held18 = held 460 @18, pub10 = public 1491 @10). Failure type and status rules: header of `scripts/nx_regress.py` (right = score >= 0.9; a change needs >= 0.1). progress_rel is the metric as stored.

| scene | set | read | s0 score | s1 score | s0 type | s1 type | s0 prog | s1 prog | s1 vs s0 |
|---|---|---|--:|--:|:-:|:-:|--:|--:|:-:|
| 517e1992 | A | held18 | 0.67 | 0.66 | hit | hit | 0.93 | 0.92 | still wrong |
| 517e1992 | A | pub10 | 1.00 | 1.00 | ok | ok | 0.84 | 0.91 | still right |
| a2839b1b | A | pub10 | 1.00 | 1.00 | ok | ok | 0.80 | 0.88 | still right |
| c240593c | A | pub10 | 1.00 | 1.00 | ok | ok | 1.00 | 1.00 | still right |
| d32f8367 | A | held18 | 0.87 | 0.86 | slow | slow | 0.35 | 0.38 | still wrong |
| d32f8367 | A | pub10 | 0.56 | 0.56 | slow | slow | 0.36 | 0.38 | still wrong |
| da53dac7 | A | held18 | 0.38 | 0.38 | hit | hit | 0.78 | 0.77 | still wrong |
| da53dac7 | A | pub10 | 0.76 | 0.75 | hit | hit | 0.78 | 0.77 | still wrong |
| de4ace24 | A | held18 | 0.48 | 0.82 | hit | slow | 0.64 | 0.65 | still wrong |
| de4ace24 | A | pub10 | 0.92 | 0.94 | slow | slow | 0.74 | 0.75 | still right |
| 0854af02 | B | pub10 | 0.99 | 0.97 | ok | slow | 0.79 | 0.77 | still right |
| 1f544304 | B | held18 | 1.00 | 1.00 | ok | ok | 0.84 | 0.85 | still right |
| 1f544304 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.84 | 0.85 | still right |
| 365c0937 | B | held18 | 1.00 | 1.00 | ok | ok | 0.00 | 0.00 | still right |
| 365c0937 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.00 | 0.00 | still right |
| 4cf608d9 | B | pub10 | 1.00 | 1.00 | ok | ok | 1.00 | 1.00 | still right |
| 53d37873 | B | held18 | 1.00 | 1.00 | ok | ok | 0.88 | 0.85 | still right |
| 53d37873 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.98 | 0.96 | still right |
| 6d2b59f7 | B | held18 | 1.00 | 1.00 | ok | ok | 0.98 | 1.00 | still right |
| 6d2b59f7 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.99 | 1.00 | still right |
| 81a72614 | B | pub10 | 1.00 | 1.00 | ok | ok | 1.00 | 1.00 | still right |
| 9a675656 | B | held18 | 1.00 | 1.00 | ok | ok | 0.97 | 0.97 | still right |
| 9a675656 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.97 | 0.97 | still right |
| 9d1d720d | B | held18 | 1.00 | 1.00 | ok | ok | 0.82 | 0.85 | still right |
| 9d1d720d | B | pub10 | 1.00 | 1.00 | ok | ok | 0.87 | 0.88 | still right |
| a7fa8bcc | B | pub10 | 1.00 | 1.00 | ok | ok | 0.98 | 0.97 | still right |
| a9609780 | B | held18 | 1.00 | 1.00 | ok | ok | 0.33 | 0.33 | still right |
| a9609780 | B | pub10 | 1.00 | 1.00 | ok | ok | 0.33 | 0.33 | still right |
| d33c6db3 | B | pub10 | 0.66 | 0.00 | hit | AF-side | 1.00 | 1.00 | still wrong |
| ddc1271e | B | held18 | 1.00 | 1.00 | ok | ok | 0.86 | 0.88 | still right |
| ddc1271e | B | pub10 | 1.00 | 1.00 | ok | ok | 0.87 | 0.88 | still right |
| 42c7526f | C | held18 | 1.00 | 1.00 | ok | ok | 0.90 | 0.88 | still right |
| 42c7526f | C | pub10 | 1.00 | 1.00 | ok | ok | 0.96 | 0.94 | still right |
| 4211afd0 | D+N | pub10 | 0.57 | 0.67 | slow | slow | 0.45 | 0.54 | still wrong |
| c159f0c3 | D+N | pub10 | 1.00 | 1.00 | ok | ok | 0.84 | 1.00 | still right |
| c2790319 | D | pub10 | 0.65 | 0.68 | slow | slow | 0.52 | 0.54 | still wrong |
| d93e31df | D+N | pub10 | 0.39 | 0.37 | crawl | crawl | 0.26 | 0.24 | still wrong |
| 94652fa0 | E | held18 | 1.00 | 1.00 | ok | ok | 0.92 | 0.89 | still right |
| 94652fa0 | E | pub10 | 1.00 | 1.00 | ok | ok | 0.93 | 0.91 | still right |
| 1b25edea | F+N | held18 | 0.26 | 1.00 | crawl | ok | 0.00 | 0.00 | fixed |
| 1b25edea | F+N | pub10 | 0.96 | 1.00 | slow | ok | 0.00 | 0.00 | still right |
| c665621f | F+N | pub10 | 0.99 | 1.00 | ok | ok | 0.00 | 0.00 | still right |
| 982b57b1 | G+N | pub10 | 0.62 | 0.67 | slow | slow | 0.50 | 0.53 | still wrong |
| 79ecc8e6 | N | pub10 | 0.70 | 0.87 | slow | slow | 0.00 | 0.00 | still wrong |
| e29b049a | N | pub10 | 1.00 | 1.00 | ok | ok | 1.00 | 1.00 | still right |
| ed829447 | N | held18 | 1.00 | 1.00 | ok | ok | 0.70 | 0.70 | still right |
| ed829447 | N | pub10 | 1.00 | 1.00 | ok | ok | 0.70 | 0.70 | still right |
| eeba0804 | N | pub10 | 0.75 | 0.71 | slow | slow | 0.48 | 0.55 | still wrong |
| f3512ae1 | N | held18 | 0.57 | 0.67 | hit | hit | 0.00 | 0.00 | still wrong |
| f3512ae1 | N | pub10 | 1.00 | 1.00 | ok | ok | 0.00 | 0.00 | still right |
| f4eec162 | N | pub10 | 1.00 | 1.00 | ok | ok | 0.61 | 0.63 | still right |

## s1 against s0: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 1 | 5 | 0 | 11 | 17 |
| pub10 | 0 | 9 | 0 | 25 | 34 |
| total | 1 | 14 | 0 | 36 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 0 | 1 | 0 | 20 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 0 | 0 | 0 | 2 | 2 |
| F detour | 1 | 0 | 0 | 2 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 1 | 6 | 0 | 8 | 15 |

## Mean score by class (scene-reads; a row counts in every class it belongs to; descriptive, added after the first read)

| class | rows | s0 | s1 |
|---|--:|--:|--:|
| A longitudinal | 10 | 0.77 | 0.80 |
| B lateral | 21 | 0.98 | 0.95 |
| C route memory | 2 | 1.00 | 1.00 |
| D red light | 4 | 0.65 | 0.68 |
| E rear-ended | 2 | 1.00 | 1.00 |
| F detour | 3 | 0.74 | 1.00 |
| G render | 1 | 0.62 | 0.67 |
| N public no-launch | 15 | 0.79 | 0.86 |
| all rows | 51 | 0.88 | 0.89 |
