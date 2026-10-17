# Regression set: RSG = RSG-L18-s0, V2 = RTB1ARO-L18-s0, NR1H0P = NR1H0P-L18-s0, NR1H15P = NR1H15P-L18-s0, NR1H30P = NR1H30P-L18-s0, NR1H15O = NR1H15O-L18-s0; status against NR1H0P

34 scenes (the user's 28 + the 12 public no-launch scenes of explore_f.md, 6 overlap), one row per scene and stored read (held18 = held 460 @18, pub10 = public 1491 @10). Failure type and status rules: header of `scripts/nx_regress.py` (right = score >= 0.9; a change needs >= 0.1). progress_rel is the metric as stored.

| scene | set | read | RSG score | V2 score | NR1H0P score | NR1H15P score | NR1H30P score | NR1H15O score | RSG type | V2 type | NR1H0P type | NR1H15P type | NR1H30P type | NR1H15O type | RSG prog | V2 prog | NR1H0P prog | NR1H15P prog | NR1H30P prog | NR1H15O prog | RSG vs NR1H0P | V2 vs NR1H0P | NR1H15P vs NR1H0P | NR1H30P vs NR1H0P | NR1H15O vs NR1H0P |
|---|---|---|--:|--:|--:|--:|--:|--:|:-:|:-:|:-:|:-:|:-:|:-:|--:|--:|--:|--:|--:|--:|:-:|:-:|:-:|:-:|:-:|
| 517e1992 | A | held18 | 0.67 | 0.67 | 0.68 | 0.68 | 0.00 | 0.68 | hit | hit | hit | hit | AF-front | hit | 0.93 | 0.94 | 0.95 | 0.94 | 0.95 | 0.94 | still wrong | still wrong | still wrong | still wrong | still wrong |
| 517e1992 | A | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.84 | 0.93 | 0.96 | 0.96 | - | - | still right | still right | still right | - | - |
| a2839b1b | A | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.80 | 0.61 | 0.56 | 0.56 | - | - | still right | still right | still right | - | - |
| c240593c | A | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 1.00 | 0.83 | 0.80 | 0.79 | - | - | still right | still right | still right | - | - |
| d32f8367 | A | held18 | 0.87 | 0.92 | 0.83 | 0.83 | 0.83 | 0.84 | slow | slow | slow | slow | slow | slow | 0.35 | 0.32 | 0.28 | 0.27 | 0.27 | 0.28 | still wrong | still wrong | still wrong | still wrong | still wrong |
| d32f8367 | A | pub10 | 0.56 | 0.55 | 0.47 | 0.47 | - | - | slow | slow | crawl | crawl | - | - | 0.36 | 0.32 | 0.28 | 0.27 | - | - | still wrong | still wrong | still wrong | - | - |
| da53dac7 | A | held18 | 0.38 | 0.34 | 0.39 | 0.39 | 0.39 | 0.39 | hit | hit | hit | hit | hit | hit | 0.78 | 0.79 | 0.79 | 0.79 | 0.79 | 0.79 | still wrong | still wrong | still wrong | still wrong | still wrong |
| da53dac7 | A | pub10 | 0.76 | 0.66 | 0.77 | 0.76 | - | - | hit | hit | hit | hit | - | - | 0.78 | 0.79 | 0.79 | 0.79 | - | - | still wrong | still wrong | still wrong | - | - |
| de4ace24 | A | held18 | 0.48 | 0.47 | 0.83 | 0.83 | 0.83 | 0.84 | hit | hit | slow | slow | slow | slow | 0.64 | 0.62 | 0.65 | 0.65 | 0.65 | 0.65 | still wrong | still wrong | still wrong | still wrong | still wrong |
| de4ace24 | A | pub10 | 0.92 | 0.85 | 0.90 | 0.90 | - | - | slow | slow | slow | slow | - | - | 0.74 | 0.68 | 0.72 | 0.72 | - | - | still right | still right | still right | - | - |
| 0854af02 | B | pub10 | 0.99 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.79 | 0.86 | 0.86 | 0.85 | - | - | still right | still right | still right | - | - |
| 1f544304 | B | held18 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ok | offroad | offroad | offroad | offroad | offroad+corridor | 0.84 | 0.94 | 0.92 | 0.92 | 0.92 | 0.91 | fixed | still wrong | still wrong | still wrong | still wrong |
| 1f544304 | B | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.84 | 0.95 | 0.92 | 0.92 | - | - | still right | still right | still right | - | - |
| 365c0937 | B | held18 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | still right | still right | still right | still right | still right |
| 365c0937 | B | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | still right | still right | still right | - | - |
| 4cf608d9 | B | pub10 | 1.00 | 0.00 | 1.00 | 1.00 | - | - | ok | offroad | ok | ok | - | - | 1.00 | 1.00 | 1.00 | 1.00 | - | - | still right | newly wrong | still right | - | - |
| 53d37873 | B | held18 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.88 | 0.83 | 0.84 | 0.85 | 0.84 | 0.85 | still right | still right | still right | still right | still right |
| 53d37873 | B | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.98 | 0.96 | 0.94 | 0.93 | - | - | still right | still right | still right | - | - |
| 6d2b59f7 | B | held18 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | still right | still right | still right | still right | still right |
| 6d2b59f7 | B | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.99 | 1.00 | 1.00 | 1.00 | - | - | still right | still right | still right | - | - |
| 81a72614 | B | pub10 | 1.00 | 0.00 | 1.00 | 1.00 | - | - | ok | AF-front+corridor | ok | ok | - | - | 1.00 | 1.00 | 1.00 | 1.00 | - | - | still right | newly wrong | still right | - | - |
| 9a675656 | B | held18 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ok | AF-front | AF-front | AF-front | AF-front | AF-front | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | fixed | still wrong | still wrong | still wrong | still wrong |
| 9a675656 | B | pub10 | 1.00 | 0.00 | 1.00 | 0.00 | - | - | ok | corridor | ok | corridor | - | - | 0.97 | 0.97 | 0.97 | 0.97 | - | - | still right | newly wrong | newly wrong | - | - |
| 9d1d720d | B | held18 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ok | offroad | offroad | offroad | offroad | offroad | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | fixed | still wrong | still wrong | still wrong | still wrong |
| 9d1d720d | B | pub10 | 1.00 | 0.00 | 0.00 | 0.00 | - | - | ok | offroad | offroad | offroad | - | - | 0.87 | 1.00 | 1.00 | 1.00 | - | - | fixed | still wrong | still wrong | - | - |
| a7fa8bcc | B | pub10 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | - | ok | AF-side | ok | ok | ok | - | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | - | still right | newly wrong | still right | still right | - |
| a9609780 | B | held18 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | still right | still right | still right | still right | still right |
| a9609780 | B | pub10 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | ok | corridor | corridor | corridor | corridor | - | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | - | fixed | still wrong | still wrong | still wrong | - |
| d33c6db3 | B | pub10 | 0.66 | 0.70 | 0.69 | 0.69 | - | - | hit | hit | hit | hit | - | - | 1.00 | 1.00 | 1.00 | 1.00 | - | - | still wrong | still wrong | still wrong | - | - |
| ddc1271e | B | held18 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | ok | offroad | offroad | offroad | offroad | offroad | 0.86 | 0.87 | 0.61 | 0.60 | 0.66 | 0.66 | fixed | still wrong | still wrong | still wrong | still wrong |
| ddc1271e | B | pub10 | 1.00 | 0.00 | 0.86 | 0.87 | 0.87 | - | ok | offroad | slow | slow | slow | - | 0.87 | 0.87 | 0.69 | 0.70 | 0.70 | - | fixed | still wrong | still wrong | still wrong | - |
| 42c7526f | C | held18 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | still right | still right | still right | still right | still right |
| 42c7526f | C | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.96 | 1.00 | 1.00 | 1.00 | - | - | still right | still right | still right | - | - |
| 4211afd0 | D+N | pub10 | 0.57 | 0.68 | 0.65 | 0.66 | 0.68 | - | slow | slow | slow | slow | slow | - | 0.45 | 0.55 | 0.52 | 0.52 | 0.54 | - | still wrong | still wrong | still wrong | still wrong | - |
| c159f0c3 | D+N | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - | ok | ok | ok | ok | ok | - | 0.84 | 0.84 | 0.85 | 0.85 | 0.85 | - | still right | still right | still right | still right | - |
| c2790319 | D | pub10 | 0.65 | 0.61 | 0.47 | 0.50 | - | - | slow | slow | crawl | slow | - | - | 0.52 | 0.49 | 0.37 | 0.40 | - | - | still wrong | still wrong | still wrong | - | - |
| d93e31df | D+N | pub10 | 0.39 | 0.14 | 0.12 | 0.12 | - | - | crawl | no-launch | no-launch | no-launch | - | - | 0.26 | 0.11 | 0.10 | 0.10 | - | - | still wrong | still wrong | still wrong | - | - |
| 94652fa0 | E | held18 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | ok | AF-front | ok | ok | ok | ok | 0.92 | 0.97 | 0.92 | 0.92 | 0.92 | 0.92 | still right | newly wrong | still right | still right | still right |
| 94652fa0 | E | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.93 | 0.96 | 0.92 | 0.92 | - | - | still right | still right | still right | - | - |
| 1b25edea | F+N | held18 | 0.26 | 0.80 | 0.07 | 0.06 | 0.07 | 0.07 | crawl | slow | no-launch | no-launch | no-launch | no-launch | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | still wrong | still wrong | still wrong | still wrong | still wrong |
| 1b25edea | F+N | pub10 | 0.96 | 0.58 | 0.27 | 0.27 | - | - | slow | slow | no-launch | no-launch | - | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | fixed | still wrong | still wrong | - | - |
| c665621f | F+N | pub10 | 0.99 | 0.31 | 0.16 | 0.17 | - | - | ok | no-launch | no-launch | no-launch | - | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | fixed | still wrong | still wrong | - | - |
| 982b57b1 | G+N | pub10 | 0.62 | 0.39 | 0.38 | 0.38 | - | - | slow | crawl | crawl | crawl | - | - | 0.50 | 0.30 | 0.28 | 0.28 | - | - | still wrong | still wrong | still wrong | - | - |
| 79ecc8e6 | N | pub10 | 0.70 | 0.44 | 0.07 | 0.07 | - | - | slow | crawl | no-launch | no-launch | - | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | still wrong | still wrong | still wrong | - | - |
| e29b049a | N | pub10 | 1.00 | 1.00 | 0.94 | 0.94 | - | - | ok | ok | slow | slow | - | - | 1.00 | 0.80 | 0.72 | 0.72 | - | - | still right | still right | still right | - | - |
| ed829447 | N | held18 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | ok | ok | ok | ok | ok | ok | 0.70 | 0.70 | 0.63 | 0.64 | 0.65 | 0.64 | still right | still right | still right | still right | still right |
| ed829447 | N | pub10 | 1.00 | 0.85 | 0.80 | 0.81 | - | - | ok | slow | slow | slow | - | - | 0.70 | 0.68 | 0.64 | 0.64 | - | - | fixed | still wrong | still wrong | - | - |
| eeba0804 | N | pub10 | 0.75 | 0.51 | 0.49 | 0.48 | - | - | slow | slow | slow | slow | - | - | 0.48 | 0.33 | 0.33 | 0.33 | - | - | still wrong | still wrong | still wrong | - | - |
| f3512ae1 | N | held18 | 0.57 | 0.66 | 0.99 | 0.98 | 0.96 | 1.00 | hit | hit | ok | slow | slow | ok | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | newly wrong | newly wrong | still right | still right | still right |
| f3512ae1 | N | pub10 | 1.00 | 1.00 | 1.00 | 1.00 | - | - | ok | ok | ok | ok | - | - | 0.00 | 0.00 | 0.00 | 0.00 | - | - | still right | still right | still right | - | - |
| f4eec162 | N | pub10 | 1.00 | 0.45 | 0.33 | 0.34 | - | - | ok | slow | crawl | crawl | - | - | 0.61 | 0.25 | 0.20 | 0.20 | - | - | fixed | still wrong | still wrong | - | - |

## RSG against NR1H0P: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 4 | 5 | 1 | 7 | 17 |
| pub10 | 7 | 9 | 0 | 18 | 34 |
| total | 11 | 14 | 1 | 25 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 7 | 1 | 0 | 13 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 0 | 0 | 0 | 2 | 2 |
| F detour | 2 | 1 | 0 | 0 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 4 | 6 | 1 | 4 | 15 |

## V2 against NR1H0P: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 0 | 9 | 2 | 6 | 17 |
| pub10 | 0 | 16 | 4 | 14 | 34 |
| total | 0 | 25 | 6 | 20 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 0 | 8 | 4 | 9 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 0 | 0 | 1 | 1 | 2 |
| F detour | 0 | 3 | 0 | 0 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 0 | 10 | 1 | 4 | 15 |

## NR1H15P against NR1H0P: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 0 | 9 | 0 | 8 | 17 |
| pub10 | 0 | 16 | 1 | 17 | 34 |
| total | 0 | 25 | 1 | 25 | 51 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 6 | 0 | 4 | 10 |
| B lateral | 0 | 8 | 1 | 12 | 21 |
| C route memory | 0 | 0 | 0 | 2 | 2 |
| D red light | 0 | 3 | 0 | 1 | 4 |
| E rear-ended | 0 | 0 | 0 | 2 | 2 |
| F detour | 0 | 3 | 0 | 0 | 3 |
| G render | 0 | 1 | 0 | 0 | 1 |
| N public no-launch | 0 | 10 | 0 | 5 | 15 |

## NR1H30P against NR1H0P: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 0 | 9 | 0 | 8 | 17 |
| pub10 | 0 | 3 | 0 | 2 | 5 |
| total | 0 | 12 | 0 | 10 | 22 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 4 | 0 | 0 | 4 |
| B lateral | 0 | 6 | 0 | 5 | 11 |
| C route memory | 0 | 0 | 0 | 1 | 1 |
| D red light | 0 | 1 | 0 | 1 | 2 |
| E rear-ended | 0 | 0 | 0 | 1 | 1 |
| F detour | 0 | 1 | 0 | 0 | 1 |
| G render | 0 | 0 | 0 | 0 | 0 |
| N public no-launch | 0 | 2 | 0 | 3 | 5 |

## NR1H15O against NR1H0P: counts of scene-reads

| | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| held18 | 0 | 9 | 0 | 8 | 17 |
| pub10 | 0 | 0 | 0 | 0 | 0 |
| total | 0 | 9 | 0 | 8 | 17 |

| class | fixed | still wrong | newly wrong | still right | rows |
|---|--:|--:|--:|--:|--:|
| A longitudinal | 0 | 4 | 0 | 0 | 4 |
| B lateral | 0 | 4 | 0 | 4 | 8 |
| C route memory | 0 | 0 | 0 | 1 | 1 |
| D red light | 0 | 0 | 0 | 0 | 0 |
| E rear-ended | 0 | 0 | 0 | 1 | 1 |
| F detour | 0 | 1 | 0 | 0 | 1 |
| G render | 0 | 0 | 0 | 0 | 0 |
| N public no-launch | 0 | 1 | 0 | 2 | 3 |

## Mean score by class (scene-reads; a row counts in every class it belongs to; descriptive, added after the first read)

| class | rows | RSG | V2 | NR1H0P | NR1H15P | NR1H30P | NR1H15O |
|---|--:|--:|--:|--:|--:|--:|--:|
| A longitudinal | 10 | 0.77 | 0.75 | 0.79 | 0.79 | 0.51 | 0.69 |
| B lateral | 21 | 0.98 | 0.46 | 0.69 | 0.65 | 0.53 | 0.50 |
| C route memory | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| D red light | 4 | 0.65 | 0.61 | 0.56 | 0.57 | 0.84 | - |
| E rear-ended | 2 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| F detour | 3 | 0.74 | 0.56 | 0.17 | 0.17 | 0.07 | 0.07 |
| G render | 1 | 0.62 | 0.39 | 0.38 | 0.38 | - | - |
| N public no-launch | 15 | 0.79 | 0.65 | 0.55 | 0.55 | 0.74 | 0.69 |
| all rows | 51 | 0.88 | 0.60 | 0.69 | 0.67 | 0.62 | 0.64 |
