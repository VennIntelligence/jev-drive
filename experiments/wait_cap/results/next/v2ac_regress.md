# Regression set: RSG = RSG-L18-s0, V2 = RTB1ARO-L18-s0, V2AC = V2AC-L18-s0; status against V2

34 scenes (the user's 28 + the 12 public no-launch scenes of explore_f.md, 6 overlap), one row per scene and stored read (held18 = held 460 @18, pub10 = public 1491 @10). Failure type and status rules: header of `scripts/nx_regress.py` (right = score >= 0.9; a change needs >= 0.1). progress_rel is the metric as stored.

| scene | set | read | RSG score | V2 score | V2AC score | RSG type | V2 type | V2AC type | RSG prog | V2 prog | V2AC prog | RSG vs V2 | V2AC vs V2 |
|---|---|---|--:|--:|--:|:-:|:-:|:-:|--:|--:|--:|:-:|:-:|
| 517e1992 | A | held18 | 0.67 | 0.67 | 0.66 | hit | hit | hit | 0.93 | 0.94 | 0.93 | still wrong | still wrong |
| 517e1992 | A | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.84 | 0.93 | 0.94 | still right | still right |
| a2839b1b | A | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.80 | 0.61 | 0.63 | still right | still right |
| c240593c | A | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 1.00 | 0.83 | 0.84 | still right | still right |
| d32f8367 | A | held18 | 0.87 | 0.92 | 0.87 | slow | slow | slow | 0.35 | 0.32 | 0.31 | still right | still right |
| d32f8367 | A | pub10 | 0.56 | 0.55 | 0.52 | slow | slow | slow | 0.36 | 0.32 | 0.31 | still wrong | still wrong |
| da53dac7 | A | held18 | 0.38 | 0.34 | 0.40 | hit | hit | hit | 0.78 | 0.79 | 0.81 | still wrong | still wrong |
| da53dac7 | A | pub10 | 0.76 | 0.66 | 0.79 | hit | hit | hit | 0.78 | 0.79 | 0.81 | still wrong | still wrong |
| de4ace24 | A | held18 | 0.48 | 0.47 | 0.88 | hit | hit | slow | 0.64 | 0.62 | 0.69 | still wrong | still wrong |
| de4ace24 | A | pub10 | 0.92 | 0.85 | 0.94 | slow | slow | slow | 0.74 | 0.68 | 0.75 | still wrong | still wrong |
| 0854af02 | B | pub10 | 0.99 | 1.00 | 1.00 | ok | ok | ok | 0.79 | 0.86 | 0.84 | still right | still right |
| 1f544304 | B | held18 | 1.00 | 0.00 | 0.00 | ok | offroad | offroad | 0.84 | 0.94 | 0.95 | fixed | still wrong |
| 1f544304 | B | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.84 | 0.95 | 0.95 | still right | still right |
| 365c0937 | B | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.00 | 0.00 | 0.00 | still right | still right |
| 365c0937 | B | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.00 | 0.00 | 0.00 | still right | still right |
| 4cf608d9 | B | pub10 | 1.00 | 0.00 | 1.00 | ok | offroad | ok | 1.00 | 1.00 | 1.00 | fixed | fixed |
| 53d37873 | B | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.88 | 0.83 | 0.82 | still right | still right |
| 53d37873 | B | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.98 | 0.96 | 0.93 | still right | still right |
| 6d2b59f7 | B | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.98 | 1.00 | 1.00 | still right | still right |
| 6d2b59f7 | B | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.99 | 1.00 | 1.00 | still right | still right |
| 81a72614 | B | pub10 | 1.00 | 0.00 | 0.00 | ok | AF-front+corridor | AF-front | 1.00 | 1.00 | 1.00 | fixed | still wrong |
| 9a675656 | B | held18 | 1.00 | 0.00 | 0.00 | ok | AF-front | AF-front | 0.97 | 0.97 | 0.97 | fixed | still wrong |
| 9a675656 | B | pub10 | 1.00 | 0.00 | 0.00 | ok | corridor | corridor | 0.97 | 0.97 | 0.97 | fixed | still wrong |
| 9d1d720d | B | held18 | 1.00 | 0.00 | 0.00 | ok | offroad | offroad | 0.82 | 1.00 | 1.00 | fixed | still wrong |
| 9d1d720d | B | pub10 | 1.00 | 0.00 | 0.00 | ok | offroad | offroad | 0.87 | 1.00 | 1.00 | fixed | still wrong |
| a7fa8bcc | B | pub10 | 1.00 | 0.00 | 0.00 | ok | AF-side | AF-front | 0.98 | 0.99 | 0.99 | fixed | still wrong |
| a9609780 | B | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.33 | 0.33 | 0.33 | still right | still right |
| a9609780 | B | pub10 | 1.00 | 0.00 | 0.00 | ok | corridor | corridor | 0.33 | 0.33 | 0.33 | fixed | still wrong |
| d33c6db3 | B | pub10 | 0.66 | 0.70 | 0.67 | hit | hit | hit | 1.00 | 1.00 | 1.00 | still wrong | still wrong |
| ddc1271e | B | held18 | 1.00 | 0.00 | 0.00 | ok | offroad | offroad | 0.86 | 0.87 | 0.60 | fixed | still wrong |
| ddc1271e | B | pub10 | 1.00 | 0.00 | 0.00 | ok | offroad | offroad | 0.87 | 0.87 | 0.66 | fixed | still wrong |
| 42c7526f | C | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.90 | 1.00 | 1.00 | still right | still right |
| 42c7526f | C | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.96 | 1.00 | 1.00 | still right | still right |
| 4211afd0 | D+N | pub10 | 0.57 | 0.68 | 0.73 | slow | slow | slow | 0.45 | 0.55 | 0.58 | still wrong | still wrong |
| c159f0c3 | D+N | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.84 | 0.84 | 0.94 | still right | still right |
| c2790319 | D | pub10 | 0.65 | 0.61 | 0.58 | slow | slow | slow | 0.52 | 0.49 | 0.46 | still wrong | still wrong |
| d93e31df | D+N | pub10 | 0.39 | 0.14 | 0.15 | crawl | no-launch | no-launch | 0.26 | 0.11 | 0.12 | still wrong | still wrong |
| 94652fa0 | E | held18 | 1.00 | 0.00 | 1.00 | ok | AF-front | ok | 0.92 | 0.97 | 0.92 | fixed | fixed |
| 94652fa0 | E | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.93 | 0.96 | 0.92 | still right | still right |
| 1b25edea | F+N | held18 | 0.26 | 0.80 | 0.91 | crawl | slow | slow | 0.00 | 0.00 | 0.00 | still wrong | fixed |
| 1b25edea | F+N | pub10 | 0.96 | 0.58 | 0.70 | slow | slow | slow | 0.00 | 0.00 | 0.00 | fixed | still wrong |
| c665621f | F+N | pub10 | 0.99 | 0.31 | 0.25 | ok | no-launch | no-launch | 0.00 | 0.00 | 0.00 | fixed | still wrong |
| 982b57b1 | G+N | pub10 | 0.62 | 0.39 | 0.49 | slow | crawl | slow | 0.50 | 0.30 | 0.35 | still wrong | still wrong |
| 79ecc8e6 | N | pub10 | 0.70 | 0.44 | 0.35 | slow | crawl | crawl | 0.00 | 0.00 | 0.00 | still wrong | still wrong |
| e29b049a | N | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 1.00 | 0.80 | 0.78 | still right | still right |
| ed829447 | N | held18 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.70 | 0.70 | 0.70 | still right | still right |
| ed829447 | N | pub10 | 1.00 | 0.85 | 0.97 | ok | slow | slow | 0.70 | 0.68 | 0.70 | fixed | fixed |
| eeba0804 | N | pub10 | 0.75 | 0.51 | 0.53 | slow | slow | slow | 0.48 | 0.33 | 0.33 | still wrong | still wrong |
| f3512ae1 | N | held18 | 0.57 | 0.66 | 0.91 | hit | hit | hit | 0.00 | 0.00 | 0.00 | still wrong | fixed |
| f3512ae1 | N | pub10 | 1.00 | 1.00 | 1.00 | ok | ok | ok | 0.00 | 0.00 | 0.00 | still right | still right |
| f4eec162 | N | pub10 | 1.00 | 0.45 | 0.38 | ok | slow | crawl | 0.61 | 0.25 | 0.22 | fixed | still wrong |

## RSG against V2: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 5 | 5 | 0 | 7 | 17 |
| pub10 | 11 | 10 | 0 | 13 | 34 |
| total | 16 | 15 | 0 | 20 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 11 | 1 | 0 | 9 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 1 | 0 | 0 | 1 | 2 |
| F detour | 2 | 1 | 0 | 0 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 4 | 7 | 0 | 4 | 15 |

## V2AC against V2: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 3 | 7 | 0 | 7 | 17 |
| pub10 | 2 | 19 | 0 | 13 | 34 |
| total | 5 | 26 | 0 | 20 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 1 | 11 | 0 | 9 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 1 | 0 | 0 | 1 | 2 |
| F detour | 1 | 2 | 0 | 0 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 3 | 8 | 0 | 4 | 15 |

## Mean score by class (scene-reads; a row counts in every class it belongs to; descriptive, added after the first read)

| class | rows | RSG | V2 | V2AC |
|---|--:|--:|--:|--:|
| A longitudinal | 10 | 0.77 | 0.75 | 0.81 |
| B lateral | 21 | 0.98 | 0.46 | 0.51 |
| C route memory | 2 | 1.00 | 1.00 | 1.00 |
| D red light | 4 | 0.65 | 0.61 | 0.61 |
| E rear-ended | 2 | 1.00 | 0.50 | 1.00 |
| F detour | 3 | 0.74 | 0.56 | 0.62 |
| G render | 1 | 0.62 | 0.39 | 0.49 |
| N public no-launch | 15 | 0.79 | 0.65 | 0.69 |
| all rows | 51 | 0.88 | 0.60 | 0.66 |
